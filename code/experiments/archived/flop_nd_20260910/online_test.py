"""CPU tests for online provenance, replay semantics, EMA and exact resume."""
from __future__ import annotations

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import random
import tempfile
import unittest

import torch

import common
import learning as L
import online_bank as OB
import train_lm_online as T
from nd_model import ProofModel, ModelConfig, save_checkpoint
from nd_environment import Tokenizer, episode, judge
from structured_generate import make_task


def short_record():
    return {"source_id": "short_source", "family": "and", "prompt": "THM P , Q SEQ ( P & Q ) PRF",
            "body": "N1 P : PR ; N2 Q : PR ; N3 ( P & Q ) : ANDI N1 N2 ; QED", "n_lines": 3}


def long_record(seed=51):
    row = make_task(random.Random(seed), 7, "projection_atomic")
    row["source_id"] = f"long_target_{seed}"
    return row


def simulated_attempt(record, tok, *, step=1):
    ep = episode(record, 0, tok)
    return {**ep, "training_kind": "target", "finish_reason": "qed", "step": step,
            "group": 0, "attempt_index": 0, **judge(ep, ep["response_ids"], tok)}


def small_model():
    torch.manual_seed(199)
    tok = Tokenizer()
    model = ProofModel(ModelConfig(vocab_size=len(tok), d_model=16, n_layers=1,
                                  n_heads=2, d_ff=32, max_seq_len=512)).eval()
    return model, tok


def main_lineage_fixture(directory):
    """Synthetic test fixture, never a research checkpoint or measured result."""
    directory = Path(directory)
    run = directory / "synthetic_main_fixture"
    run.mkdir()
    model, tok = small_model()
    identity = {"version": OB.MAIN_VERSION, "config": {"output": str(run.resolve())}, "test_fixture": True}
    identity_hash = common.digest(identity)
    common.write_json(run / "manifest.json", {"identity": identity, "identity_sha256": identity_hash})
    record, future = long_record(), long_record(83)
    rows = [simulated_attempt(record, tok, step=1), simulated_attempt(future, tok, step=3)]
    common.write_jsonl(run / "attempts.jsonl", rows)
    common.write_jsonl(run / "discoveries.jsonl", [{"step": r["step"], "source_id": r["source_id"], "prompt": r["prompt"],
                     "body": r["body"], "witness_id": common.digest([r["prompt"], r["body"]]), "n_lines": r["n_lines"]} for r in rows])
    checkpoint = run / "step_000002.pt"
    save_checkpoint(checkpoint, model, tok, implementation=OB.MAIN_VERSION, test_fixture=True)
    payload = torch.load(checkpoint, weights_only=True)
    payload["training_state"] = {"step": 2, "identity": identity, "identity_sha256": identity_hash}
    critic = __import__("ppo_core").ValueHead(model.config.d_model)
    with torch.no_grad():
        critic.projection.bias.fill_(.2)
    payload["critic"] = critic.state_dict()
    torch.save(payload, checkpoint)
    manifest = directory / "import.json"
    OB.create_import_manifest(checkpoint, manifest)
    return checkpoint, manifest, [record, future], tok


class OnlineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_long_external_rejected_but_verified_online_admitted(self):
        record, tok = long_record(), Tokenizer()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "external.jsonl"
            common.write_jsonl(source, [record])
            with self.assertRaises(ValueError):
                T.training_sources(source)
        bank = OB.OnlineProofBank([record], tok, max_seq_len=512, max_new_tokens=128)
        attempt = simulated_attempt(record, tok)
        event = bank.admit(attempt, {"kind": "own_online_rollout", "test_fixture": True})
        self.assertEqual(event["n_lines"], 7)
        self.assertFalse(bank.records[record["source_id"]]["external_supervision"])
        frozen = copy.deepcopy(bank.records)
        self.assertIsNone(bank.admit(attempt, {"kind": "own_online_rollout", "test_fixture": True, "second": True}))
        self.assertEqual(bank.records, frozen)
        forged = copy.deepcopy(frozen)
        forged[record["source_id"]]["external_supervision"] = True
        with self.assertRaises(ValueError):
            OB.OnlineProofBank([record], tok, max_seq_len=512, max_new_tokens=128, records=forged)

    def test_online_prefix_reward_mask_and_context_filter(self):
        record, tok = long_record(), Tokenizer()
        bank = OB.OnlineProofBank([record], tok, max_seq_len=180, max_new_tokens=20)
        bank.admit(simulated_attempt(record, tok), {"kind": "own_online_rollout", "test_fixture": True})
        for cut in bank.cuts[record["source_id"]]:
            ep = bank.episodes[record["source_id"]][cut]
            self.assertLessEqual(len(ep["prompt_ids"]) + 20, 180)
            self.assertEqual(judge(ep, ep["response_ids"], tok)["reward"], 1)
            batch = L.collate([ep], tok.pad_id)
            first = len(ep["prompt_ids"]) - 1
            self.assertFalse(bool(batch["mask"][0, :first].any()))
            self.assertTrue(bool(batch["mask"][0, first:].all()))
            self.assertEqual(int(batch["mask"].sum()), len(ep["response_ids"]))
        restored = OB.OnlineProofBank([record], tok, max_seq_len=180, max_new_tokens=20, records=copy.deepcopy(bank.records))
        self.assertEqual(restored.cuts, bank.cuts)

    def test_flop_ema_crosses_on_fifth_and_force_is_strict(self):
        state = {"cut": 2, "n": 0, "success": 0}
        for i in range(4):
            self.assertIsNone(OB.advance_curriculum(state, [0, 1, 2], [1], mode="flop_ema"))
        self.assertAlmostEqual(state["ema"], .8704)
        event = OB.advance_curriculum(state, [0, 1, 2], [1], mode="flop_ema")
        self.assertEqual(event["n"], 5)
        self.assertAlmostEqual(event["ema"], .92224)
        self.assertEqual(state, {"cut": 1, "n": 0, "success": 0, "ema": 0.})
        forced = {"cut": 1, "n": 0, "success": 0}
        self.assertIsNone(OB.advance_curriculum(forced, [0, 1], [0, 0], mode="flop_ema", force_episodes=2))
        event = OB.advance_curriculum(forced, [0, 1], [0], mode="flop_ema", force_episodes=2)
        self.assertEqual(event["trigger"], "forced_episode_count")
        batched = {"cut": 1, "n": 0, "success": 0}
        event = OB.advance_curriculum(batched, [0, 1], [1] * 8, mode="flop_ema")
        self.assertEqual(event["old_cut_group_outcomes_unused_after_advance"], 3)

    def test_import_hash_lineage_and_later_step_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, manifest, records, tok = main_lineage_fixture(directory)
            bank = OB.OnlineProofBank(records, tok, max_seq_len=512, max_new_tokens=128)
            result = OB.import_main_discoveries(bank, manifest, checkpoint)
            self.assertEqual(result["imported"], 1)
            self.assertEqual(result["later_than_checkpoint_not_admitted"], 1)
            self.assertNotIn(records[1]["source_id"], bank.records)
            bad = Path(directory) / "wrong.pt"
            payload = torch.load(checkpoint, weights_only=True)
            payload["metadata"]["different"] = True
            torch.save(payload, bad)
            with self.assertRaises(ValueError):
                OB.import_main_discoveries(bank, manifest, bad)
            metadata = json.loads(manifest.read_text())
            attempts = Path(metadata["files"]["attempts"]["path"])
            with attempts.open("a") as f:
                f.write("{}\n")
            with self.assertRaises(ValueError):
                OB.import_main_discoveries(bank, manifest, checkpoint)

    def test_online_on_off_and_exact_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            checkpoint, imported, records, tok = main_lineage_fixture(directory)
            sources, targets = directory / "sources.jsonl", directory / "targets.jsonl"
            common.write_jsonl(sources, [short_record()])
            common.write_jsonl(targets, [{"source_id": r["source_id"], "prompt": r["prompt"]} for r in records[:1]])
            config = {"checkpoint": str(checkpoint), "data": str(sources), "retention": str(sources),
                      "targets": str(targets), "initial_bank": str(imported), "online_bank": True,
                      "target_fraction": .5, "objective": "ppo", "start": "reset", "empty_fraction": 0.,
                      "curriculum_mode": "flop_ema", "max_updates": 2, "prompts_per_step": 2, "group_size": 2,
                      "max_new_tokens": 5, "retention_batch": 1, "microbatch": 2, "batch_size": 4,
                      "checkpoint_every": 1, "epochs": 2, "cpu_threads": 1, "seed": 17,
                      "max_seconds": 60, "max_logprob_error": 1e-5}
            full, resumed, off = directory / "full", directory / "resumed", directory / "off"
            cfg = directory / "config.json"
            for output, steps, enabled in ((full, 2, True), (resumed, 1, True), (off, 1, False)):
                common.write_json(cfg, {**config, "output": str(output), "max_updates": steps, "online_bank": enabled})
                with redirect_stdout(io.StringIO()):
                    T.run(T.arguments(["--config", str(cfg)]))
            T.append_jsonl(resumed / "admissions.jsonl", [{"step": 2, "uncommitted": True}])
            common.write_json(cfg, {**config, "output": str(resumed), "resume": str(resumed / "final.pt")})
            with redirect_stdout(io.StringIO()):
                T.run(T.arguments(["--config", str(cfg)]))
            a, b = (torch.load(path / "final.pt", weights_only=True) for path in (full, resumed))
            for key in ("model", "critic"):
                for name, value in a[key].items():
                    torch.testing.assert_close(value, b[key][name], rtol=0, atol=0)
            self.assertEqual(a["training_state"]["online_bank"], b["training_state"]["online_bank"])
            self.assertEqual(a["training_state"]["online_curriculum"], b["training_state"]["online_curriculum"])
            self.assertEqual(len(common.read_jsonl(resumed / "admissions.jsonl")), 1)
            on_groups = common.read_jsonl(full / "groups.jsonl")[:2]
            off_groups = common.read_jsonl(off / "groups.jsonl")
            for on, disabled in zip(on_groups, off_groups):
                self.assertEqual(on["source_id"], disabled["source_id"])
                self.assertEqual(on["reset_draw"], disabled["reset_draw"])
                if on["training_kind"] == "source":
                    self.assertEqual(on["reset_index"], disabled["reset_index"])
                else:
                    self.assertGreater(on["reset_index"], 0)
                    self.assertEqual(disabled["reset_index"], 0)
                    self.assertTrue(on["online_assisted"])
            self.assertEqual(json.loads((full / "manifest.json").read_text())["critic_initialization"], "starting_checkpoint")


if __name__ == "__main__":
    unittest.main(verbosity=2)
