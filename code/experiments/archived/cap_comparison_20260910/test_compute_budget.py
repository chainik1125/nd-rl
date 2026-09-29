"""CPU-only invented-fixture compute, sampling and checkpoint regressions."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent / "vendor"))
import torch
import run

torch.set_num_threads(1)
PROMPT = "THM P , Q SEQ ( P & Q ) PRF"
BODY = "N1 P : PR ; N2 Q : PR ; N3 ( P & Q ) : ANDI N1 N2 ; QED"
SHORT = {"prompt": "THM P SEQ P PRF", "body": "N1 P : PR ; QED"}


def model(tok):
    torch.manual_seed(731)
    return run.ProofModel(run.ModelConfig(vocab_size=len(tok), d_model=24,
        n_layers=1, n_heads=2, d_ff=48, max_seq_len=512)).eval()


def args(output, **changes):
    values = dict(command="sft", mode="direct", max_proof_lines=6,
        scratch_style="dependency", seed=71, steps=2, batch_size=2, microbatch=1,
        lr=1e-3, output=str(output), log_every=100, save_every=0,
        checkpoint=None, data=None, guard_json=None, retention_data=None,
        max_flop_proxy=None, flop_milestones=[], temperature=.8, group_size=2,
        prompts_per_step=2, policy_epochs=2, policy_weight=1., max_new_tokens=8,
        decode_batch=4, retention_weight=.1, retention_batch=2,
        max_logprob_error=1e-4, clip=.2, loss_normalization="fixed",
        prompt_schedule="uniform", frontier_decay=.8, frontier_floor=.25,
        allow_unguarded_smoke=True, advantage="loo")
    values.update(changes)
    return SimpleNamespace(**values)


class ComputeTests(unittest.TestCase):
    def test_exact_budget_parser_and_invalid_weights(self):
        self.assertEqual(run.compute_integer("9.007199254740993e15"), 9007199254740993)
        for value in ("nan", "inf", "-1", "1.2"):
            with self.assertRaises(Exception): run.compute_integer(value)
        for values in ([{}, {"sampling_weight": 1}], [{"sampling_weight": True}],
                       [{"sampling_weight": 0}], [{"sampling_weight": float("nan")}],
                       [{"sampling_weight": "0.5"}]):
            with self.assertRaises(ValueError): run.sampling_weights(values)

    def test_weighted_sampling_and_historical_rng_path(self):
        rows = list(range(5))
        a, b = random.Random(29), random.Random(29)
        self.assertEqual(run.sample_training_rows(a, rows, 50), [b.choice(rows) for _ in range(50)])
        a, b = random.Random(29), random.Random(29)
        weights = [1, 2, 3, 4, 5]
        self.assertEqual(run.sample_training_rows(a, rows, 50, weights), b.choices(rows, weights=weights, k=50))
        schedule = run.PromptSchedule(["short", "long", "book"], weights=[.5, .45, .05])
        self.assertEqual(schedule.select(random.Random(2), 50),
                         random.Random(2).choices(schedule.prompts, weights=[.5, .45, .05], k=50))
        with self.assertRaises(ValueError): run.PromptSchedule(["a"], weights=[1, 2])

    def test_dense_proxy_formula_and_backward_multiplier(self):
        ledger = run.ComputeLedger(model(run.ScratchTokenizer()))
        forward = 2 * ledger.parameters * 3 * 7 + 4 * ledger.layers * ledger.width * 3 * 7 * 11
        ledger.account("cached", 3, 7, 11)
        ledger.account("trained", 3, 7, 11, backward=True)
        self.assertEqual(ledger.total, 4 * forward)
        self.assertEqual(ledger.phases["trained"]["backward_padded_token_slots"], 21)
        self.assertEqual(ledger.phases["trained"]["forward_attention_pair_slots"], 231)

    def test_cached_generation_charges_finished_rows_and_padding(self):
        tok = run.ScratchTokenizer(); m = model(tok); ledger = run.ComputeLedger(m)
        observed = []
        hook = m.register_forward_pre_hook(lambda module, pos, kw:
            observed.append((pos[0].shape[0], pos[0].shape[1], kw.get("cache_start", 0) + pos[0].shape[1])), with_kwargs=True)
        choices = [torch.tensor([[tok.eos_id], [tok.stoi["P"]]]),
                   torch.tensor([[tok.eos_id], [tok.stoi["P"]]]),
                   torch.tensor([[tok.eos_id], [tok.eos_id]])]
        with patch.object(torch, "multinomial", side_effect=choices):
            rows = run.sample(m, tok, [PROMPT, SHORT["prompt"]], mode="direct",
                max_new_tokens=8, batch_size=2, compute=ledger)
        hook.remove()
        self.assertEqual(sorted(len(r["response_ids"]) for r in rows), [1, 3])
        self.assertEqual(len(observed), 3)
        self.assertEqual(ledger.total, sum(ledger.cost(*shape) for shape in observed))
        self.assertEqual(ledger.phases["generation_decode"]["forward_padded_token_slots"], 4)
        self.assertEqual(ledger.phases["generation_prefill"]["forward_padded_token_slots"], 2 * len(tok.encode_prompt(PROMPT)))

    def test_policy_audit_epochs_retention_and_no_gradient_ablation(self):
        tok = run.ScratchTokenizer()
        for weight, retain in ((1., .1), (0., .1), (0., 0.)):
            m = model(tok); before = copy.deepcopy(m.state_dict())
            rows = run.sample(m, tok, [PROMPT, SHORT["prompt"], PROMPT], mode="cot", max_new_tokens=8)
            for row, advantage in zip(rows, [1., -1., 0.]): row["advantage"] = advantage
            retained = [run.encode_training(SHORT, tok, "direct", 512)] * 3
            options = args("/tmp", microbatch=2, policy_weight=weight, retention_weight=retain)
            ledger = run.ComputeLedger(m)
            run.policy_update(m, tok, rows, torch.optim.AdamW(m.parameters(), lr=1e-3), options, retained, compute=ledger)
            self.assertEqual(ledger.phases["behavior_teacher_audit"]["forward_calls"], 2)
            self.assertEqual(ledger.phases["policy"]["forward_calls"], 4)
            self.assertEqual(ledger.phases["policy"]["backward_calls"], 4 if weight else 0)
            self.assertEqual(ledger.phases.get("retention", {}).get("backward_calls", 0), 4 if retain else 0)
            self.assertEqual(ledger.total, sum(v["flop_proxy"] for v in ledger.phases.values()))
            changed = any(not torch.equal(v, before[k]) for k, v in m.state_dict().items())
            self.assertEqual(changed, bool(weight or retain))

    def test_sft_budget_stops_first_crossing_and_saves_baseline(self):
        tok = run.ScratchTokenizer(); m = model(tok)
        encoded = run.encode_training(SHORT, tok, "direct", 512)
        width = len(encoded["prompt_ids"]) + len(encoded["response_ids"]) - 1
        step = run.ComputeLedger(m).cost(2, width, backward=True)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            options = args(tmp, steps=10, max_flop_proxy=step + 1, flop_milestones=[0, step + 1])
            result = run.train_sft(m, tok, [SHORT], options)
            self.assertEqual(result["actual_steps"], 2)
            self.assertEqual(result["stop_reason"], "flop_budget")
            self.assertEqual(result["actual_flop_proxy"], 2 * step)
            self.assertEqual(result["flop_proxy_overshoot"], step - 1)
            self.assertLess(result["actual_flop_proxy"] - result["last_step_flop_proxy"], step + 1)
            baseline = torch.load(Path(tmp) / "flops_0.pt", weights_only=True)
            final = torch.load(Path(tmp) / f"flops_{step + 1}.pt", weights_only=True)
            self.assertEqual(baseline["metadata"]["step"], 0)
            self.assertEqual(final["metadata"]["step"], 2)
            self.assertEqual(final["metadata"]["training_compute"]["actual_flop_proxy"], 2 * step)

    def test_no_budget_preserves_original_sft_parameters(self):
        source = Path(__file__).resolve().parents[2] / "r1-sprint-20260909-1625/code/run.py"
        spec = importlib.util.spec_from_file_location("cap_original_runner", source)
        old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
        tok = run.ScratchTokenizer(); initial = model(tok)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            a, b = copy.deepcopy(initial), copy.deepcopy(initial)
            run.train_sft(a, tok, [SHORT, {"prompt": PROMPT, "body": BODY}], args(tmp))
            old.train_sft(b, tok, [SHORT, {"prompt": PROMPT, "body": BODY}], args(tmp))
        self.assertTrue(all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items()))

    def test_rl_budget_counts_complete_iteration_and_bound(self):
        tok = run.ScratchTokenizer(); m = model(tok)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            options = args(tmp, command="rl", steps=5, policy_epochs=1, max_flop_proxy=1,
                           retention_weight=0, flop_milestones=[0, 1])
            result = run.train_rl(m, tok, [{"prompt": PROMPT, "sampling_weight": .9},
                                          {"prompt": SHORT["prompt"], "sampling_weight": .1}], options)
            self.assertEqual(result["actual_steps"], 1)
            self.assertTrue(result["budget_reached"])
            self.assertLessEqual(result["actual_flop_proxy"], result["worst_case_one_step_flop_proxy"])
            self.assertEqual(set(result["compute"]["phases"]),
                             {"generation_prefill", "generation_decode", "behavior_teacher_audit", "policy"})
            self.assertEqual(len((Path(tmp) / "rollouts.jsonl").read_text().splitlines()), 4)

    def test_cli_completion_uses_actual_steps_and_saved_metadata(self):
        tok = run.ScratchTokenizer(); m = model(tok)
        row = run.encode_training(SHORT, tok, "direct", 512)
        cost = run.ComputeLedger(m).cost(2, len(row["prompt_ids"]) + len(row["response_ids"]) - 1, backward=True)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp); source = root / "parent.pt"; data = root / "training.jsonl"; out = root / "sft"
            run.save_checkpoint(source, m, tok, stage="prepared")
            data.write_text(json.dumps(SHORT) + "\n")
            argv = ["run.py", "sft", "--checkpoint", str(source), "--data", str(data), "--output", str(out),
                "--mode", "direct", "--device", "cpu", "--precision", "fp32", "--steps", "5",
                "--batch-size", "2", "--microbatch", "1", "--save-every", "0", "--allow-unguarded-smoke",
                "--max-flop-proxy", str(cost + 1), "--flop-milestones", "0", str(cost + 1)]
            with patch.object(sys, "argv", argv): run.main()
            done = json.loads((out / "completed.json").read_text())
            manifest = json.loads((out / "manifest.json").read_text())
            saved = torch.load(out / "final.pt", weights_only=True)["metadata"]
            self.assertEqual((done["stage"], done["steps"], done["requested_steps"]), ("sft", 2, 5))
            self.assertEqual(saved["step"], done["actual_steps"])
            self.assertEqual(saved["args"], manifest["args"])
            self.assertEqual(saved["training_compute"]["compute"], done["compute"])
            self.assertEqual(done["final_sha256"], run.sha256(out / "final.pt"))


if __name__ == "__main__":
    unittest.main()
