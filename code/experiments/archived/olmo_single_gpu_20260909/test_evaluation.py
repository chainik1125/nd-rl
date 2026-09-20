"""CPU fixtures for leakage gates, literal proof judging, and paired metrics."""

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from analyze import paired_bootstrap, replay_run
from evaluate import FINAL_NOT_BEFORE, attempt_key, check_textbook_release, planned_batches, read_resume_rows, run as evaluate_run, stable_hash, token_accounting, validate_resume
from protocol import SPRINT, judge, protocol_identity, render_completion, render_prompt


PROMPT = "THM P SEQ ( P v Q ) PRF"
PROOF = "N1 P : PR ;\nN2 ( P v Q ) : ORI1 N1 ;\nQED"
REDUNDANT = "N1 P : PR ;\nN2 P : R N1 ;\nN3 ( P v Q ) : ORI1 N2 ;\nQED"


class FixtureTokenizer:
    eos_token_id = 9999
    model_max_length = 999999
    def __len__(self):
        return 10000
    def encode(self, value, add_special_tokens=False):
        return [ord(character) for character in value]
    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join("<eos>" if token == self.eos_token_id else chr(token) for token in ids)


class LiteralProtocolTests(unittest.TestCase):
    def test_gold_and_optional_scratch_accept_literal_proof(self):
        text = "Use left disjunction introduction.\n" + render_completion(PROOF)
        result = judge(PROMPT, text)
        self.assertTrue(result["accepted"], result)
        self.assertEqual(result["n_lines"], 2)
        self.assertEqual(result["proof"], "\n" + PROOF + "\n")
        self.assertGreater(result["scratch_characters"], 0)

    def test_wrong_prompt_not_replaced_by_output(self):
        result = judge("THM Q SEQ ( P v Q ) PRF", PROMPT + "\n" + render_completion(PROOF))
        self.assertFalse(result["accepted"])
        self.assertIn("premise block", result["reason"])

    def test_rejects_missing_duplicate_and_reversed_markers(self):
        for text in (PROOF, "<FINAL>" + PROOF, render_completion(PROOF) * 2, "</FINAL><FINAL>" + PROOF, "<FINAL><FINAL>" + PROOF + "</FINAL>"):
            self.assertFalse(judge(PROMPT, text)["accepted"], text)

    def test_rejects_nonterminal_qed_trailing_material_and_bad_citations(self):
        for text in (render_completion(PROOF + " junk"), render_completion(PROOF + " QED"), render_completion(PROOF) + " next theorem", render_completion(PROOF.replace("ORI1 N1", "ORI1 N9"))):
            self.assertFalse(judge(PROMPT, text)["accepted"], text)

    def test_closed_box_discharge_and_scope_rules(self):
        prompt = "THM SEQ ( P > P ) PRF"
        proof = "N1 | P : AS ; N2 | P : R N1 ; N3 ( P > P ) : IMPI N1 N2 ; QED"
        self.assertTrue(judge(prompt, render_completion(proof))["accepted"])
        self.assertFalse(judge(prompt, render_completion(proof.replace("IMPI N1 N2", "IMPI N1 N3")))["accepted"])

    def test_all_demonstrations_are_verbatim_and_primary_prompt_is_shared(self):
        examples = [{"prompt": PROMPT, "proof": PROOF} for _ in range(32)]
        rendered = render_prompt(PROMPT, examples)
        self.assertEqual(rendered.count(render_completion(PROOF)), 32)
        self.assertTrue(rendered.endswith("Theorem:\n" + PROMPT + "\nAnswer:\n"))
        self.assertIn("Example 32", rendered)
        self.assertNotIn("Example 1", render_prompt(PROMPT))


class GateAndBudgetTests(unittest.TestCase):
    def test_final_gate_rejects_early_before_resolving_checkpoint_or_data(self):
        release = {
            "kind": "textbook_final_release", "allowed_arms": ["base_zero"],
            "settings": {"max_new_tokens": 2048, "sampled_k": 4, "temperature": 1.0, "top_p": 1.0, "top_k": 0},
            "not_before_utc": FINAL_NOT_BEFORE.isoformat(),
        }
        with self.assertRaisesRegex(ValueError, "window has not opened"):
            check_textbook_release(release, "base_zero", "textbook_final", None, datetime(2026, 9, 9, 23, tzinfo=timezone.utc))

    def test_development_gate_requires_completed_training_before_payload(self):
        release = {"kind": "textbook_development_release", "allowed_arms": ["base_zero"], "settings": {"max_new_tokens": 2048, "sampled_k": 4, "temperature": 1.0, "top_p": 1.0, "top_k": 0}}
        with self.assertRaisesRegex(ValueError, "pin completed training"):
            check_textbook_release(release, "base_zero", "textbook_dev", None)

    def test_exact_native_budget_includes_failure_and_eos(self):
        class Tokenizer:
            eos_token_id = 99
            def encode(self, value, add_special_tokens=False):
                return list(value)
        generation = {"text": "failed scratch", "response_ids": [1, 2, 3, 99], "prompt_tokens": 7000}
        accounting = token_accounting(Tokenizer(), generation, judge(PROMPT, generation["text"]))
        self.assertEqual(accounting["generated_native_tokens"], 4)
        self.assertEqual(accounting["eos_native_tokens"], 1)
        self.assertEqual(accounting["failed_attempt_native_tokens"], 4)
        self.assertEqual(accounting["prefill_native_tokens"], 7000)

    def test_resume_rejects_duplicate_and_changed_plan(self):
        target = {"prompt": PROMPT, "canonical_hash": "a", "cohort": "short_panel"}
        key = attempt_key(target, "sampled", 0)
        planned = {key: (target, "sampled", 0)}
        row = {"attempt_key": key, "plan_sha256": "x", "prompt": PROMPT, "mode": "sampled", "draw": 0}
        self.assertEqual(validate_resume([row], planned, "x"), {key})
        with self.assertRaises(ValueError):
            validate_resume([row, row], planned, "x")
        with self.assertRaises(ValueError):
            validate_resume([row], planned, "y")

    def test_batches_never_mix_modes_and_reproduce_whole_rng_units(self):
        targets = [{"prompt": PROMPT, "canonical_hash": str(i), "cohort": "short_panel"} for i in range(5)]
        planned, batches = planned_batches(targets, ["greedy", "sampled"], 4, 42)
        self.assertEqual(len(planned), 25)
        self.assertEqual([len(batch["attempt_keys"]) for batch in batches], [4, 1, 4, 4, 4, 4, 4])
        self.assertEqual((planned, batches), planned_batches(targets, ["greedy", "sampled"], 4, 42))
        for batch in batches:
            self.assertEqual({planned[key][1] for key in batch["attempt_keys"]}, {batch["mode"]})

    def test_interrupted_append_preserves_tail_and_committed_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attempts.jsonl"
            committed = b'{"one": 1}\n'
            tail = b'{"two":'
            path.write_bytes(committed + tail)
            self.assertEqual(read_resume_rows(path), [{"one": 1}])
            self.assertEqual(path.read_bytes(), committed)
            backups = list(Path(directory).glob("*.bin"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), tail)
            path.write_bytes(committed + b'bad committed record\n')
            with self.assertRaisesRegex(ValueError, "committed"):
                read_resume_rows(path)


class AnalysisTests(unittest.TestCase):
    def fixture(self, path: Path, omit_last=False):
        targets = [
            {"name": "one", "canonical_hash": "a", "cohort": "short_panel", "family": "ori", "prompt": PROMPT, "reference_lines": 2},
            {"name": "two", "canonical_hash": "b", "cohort": "short_panel", "family": "ori", "prompt": "THM Q SEQ ( P v Q ) PRF", "reference_lines": 2},
        ]
        _, batches = planned_batches(targets, ["greedy", "sampled"], 4, 42)
        batch_for_key = {key: (batch, index) for batch in batches for index, key in enumerate(batch["attempt_keys"])}
        plan = {"version": 2, "arm": "fixture", "protocol": protocol_identity(), "settings": {"modes": ["greedy", "sampled"], "max_new_tokens": 2048, "batch_size": 4, "seed": 42}, "batches": batches, "target_identity": stable_hash(targets), "target_count": 2, "cohorts": ["short_panel"], "model_id": "fixture", "model_revision": "fixture", "demonstrations_per_query": 0}
        (path / "evaluation_plan.json").write_text(json.dumps(plan))
        (path / "targets.jsonl").write_text("".join(json.dumps(target) + "\n" for target in targets))
        rows = []
        for target in targets:
            for mode in plan["settings"]["modes"]:
                for draw in range(1 if mode == "greedy" else 4):
                    text = render_completion(REDUNDANT) if draw == 0 else "no proof"
                    verdict = judge(target["prompt"], text)
                    tokenizer = FixtureTokenizer()
                    prompts = tokenizer.encode(render_prompt(target["prompt"]))
                    responses = tokenizer.encode(text) + ([] if draw == 0 else [tokenizer.eos_token_id])
                    generation = {"text": text, "response_ids": responses, "prompt_ids": prompts, "prompt_tokens": len(prompts), "completion_tokens": len(responses), "finish_reason": "final_marker" if draw == 0 else "eos"}
                    key = attempt_key(target, mode, draw)
                    batch, index = batch_for_key[key]
                    generation.update(seed=batch["seed"], sampling={"temperature": 0.0 if mode == "greedy" else 1.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0})
                    rows.append({"attempt_key": key, "plan_sha256": stable_hash(plan), "prompt": target["prompt"], "mode": mode, "draw": draw, "batch_id": batch["batch_id"], "batch_index": index, "sampling_batch_seed": batch["seed"], "compute_call_id": "call-" + batch["batch_id"], "generation": generation, "judge": verdict})
        by_key = {row["attempt_key"]: row for row in rows}
        ledger = []
        for batch in batches:
            call = "call-" + batch["batch_id"]
            width = max(len(by_key[key]["generation"]["response_ids"]) for key in batch["attempt_keys"])
            native_outputs = []
            for key in batch["attempt_keys"]:
                row = by_key[key]
                generation = row["generation"]
                generation.update(batch_generation_steps=width, post_stop_padding_positions=width-len(generation["response_ids"]), prompt_padding_positions=0)
                row["token_accounting"] = token_accounting(FixtureTokenizer(), generation, row["judge"])
                native_outputs.append({"attempt_key": key, **{field: generation[field] for field in ("response_ids", "prompt_tokens", "finish_reason", "batch_generation_steps", "post_stop_padding_positions", "prompt_padding_positions")}})
            ledger.extend([{"phase": "started", "call_id": call, "plan_sha256": stable_hash(plan), "batch": batch, "already_committed_attempts": []}, {"phase": "completed", "call_id": call, "plan_sha256": stable_hash(plan), "batch": batch, "native_outputs": native_outputs, "wall_seconds": .01}])
        (path / "compute_batches.jsonl").write_text("".join(json.dumps(row) + "\n" for row in ledger))
        if omit_last:
            rows.pop()
        (path / "attempts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_replay_counts_all_attempts_and_strict_compression(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.fixture(path)
            result, targets, certificates = replay_run(path, tokenizer=FixtureTokenizer())
            self.assertEqual(result["observed_attempts"], 10)
            self.assertEqual(result["groups"]["short_panel"]["sampled"]["overall"]["solved"], 1)
            self.assertEqual(result["groups"]["short_panel"]["sampled"]["overall"]["n"], 2)
            self.assertEqual(result["actual_compute_totals_including_regeneration"]["completed_batches"], 3)
            self.assertEqual(result["actual_compute_totals_including_regeneration"]["extra_native_tokens_beyond_unique_scored_attempts"], 0)
            self.assertEqual(len(certificates), 2)
            solved = [row for row in targets if row["solved"]]
            self.assertTrue(all(row["shortest_raw_lines"] == 3 and row["shortest_compressed_lines"] == 2 for row in solved))
            self.assertTrue(all(row["analysis_only_never_training"] for row in certificates))

    def test_partial_sample_group_is_pending_not_counted_as_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.fixture(path, omit_last=True)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                replay_run(path, tokenizer=FixtureTokenizer())
            result, _, _ = replay_run(path, allow_partial=True, tokenizer=FixtureTokenizer())
            overall = result["groups"]["short_panel"]["sampled"]["overall"]
            self.assertEqual((overall["n"], overall["intended_n"]), (1, 2))
            self.assertEqual(result["pending_theorem_mode_groups"], 1)

    def test_paired_bootstrap_preserves_theorem_covariance(self):
        identical = paired_bootstrap([0, 1, 0, 1], [0, 1, 0, 1], draws=200)
        self.assertEqual(identical["ci95_percentile"], [0, 0])
        improvement = paired_bootstrap([0, 0], [1, 1], draws=200)
        self.assertEqual(improvement["difference_right_minus_left"], 1)
        self.assertEqual(improvement["ci95_percentile"], [1, 1])

    def test_replay_rejects_native_text_count_prompt_and_accounting_tampering(self):
        mutations = [
            (lambda row: row["generation"]["response_ids"].insert(0, ord("X")), "counts"),
            (lambda row: row["generation"].update(text="sanitized replacement"), "native-token reconstruction"),
            (lambda row: row["generation"]["prompt_ids"].pop(), "prompt IDs"),
            (lambda row: row["token_accounting"].update(generated_native_tokens=999), "token accounting"),
        ]
        for mutate, message in mutations:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                self.fixture(path)
                records = [json.loads(line) for line in (path / "attempts.jsonl").read_text().splitlines()]
                mutate(records[0])
                (path / "attempts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
                with self.assertRaisesRegex(ValueError, message):
                    replay_run(path, tokenizer=FixtureTokenizer())

    def test_partial_batch_resume_replays_seed_and_charges_repeated_compute(self):
        targets = [{"name": name, "canonical_hash": name, "cohort": "short_panel", "family": "fixture", "prompt": PROMPT, "reference_lines": 2} for name in ("one", "two")]
        calls = []
        def sampler(model, tokenizer, prompts, max_new_tokens, temperature, batch_size, seed):
            calls.append(seed)
            responses = tokenizer.encode(render_completion(PROOF))
            return [{"text": render_completion(PROOF), "prompt_ids": tokenizer.encode(prompt), "response_ids": responses,
                "prompt_tokens": len(tokenizer.encode(prompt)), "completion_tokens": len(responses), "finish_reason": "final_marker",
                "seed": seed, "sampling": {"temperature": temperature, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0},
                "batch_generation_steps": len(responses), "prompt_padding_positions": 0, "post_stop_padding_positions": 0,
                "sampling_batch_size": len(prompts)} for prompt in prompts]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            model = base / "model"
            model.mkdir()
            (model / "config.json").write_text(json.dumps({"max_position_embeddings": 999999}))
            (model / "download_completed.json").write_text("{}")
            args = SimpleNamespace(arm="base_zero", data_dir=SPRINT / "data", out=base / "eval", model_path=model,
                adapter_path=None, cohorts=["short_panel"], modes=["sampled"], batch_size=4, max_new_tokens=2048,
                seed=42, max_attempts=4, release=None)
            with patch("evaluate.load_targets", return_value=(targets, {})), patch("modeling.load_tokenizer", return_value=FixtureTokenizer()), patch("modeling.load_model", return_value=SimpleNamespace()), patch("modeling.memory_snapshot", return_value={}), patch("modeling.sample_completions", side_effect=sampler):
                first = evaluate_run(args)
                self.assertEqual(first["attempts"], 4)
                attempts = args.out / "attempts.jsonl"
                committed = attempts.read_text().splitlines()
                attempts.write_text("\n".join(committed[:2]) + "\n")
                second = evaluate_run(args)
                self.assertEqual(second["attempts"], 4)
                self.assertEqual(calls[0], calls[1])
                args.max_attempts = None
                third = evaluate_run(args)
                self.assertEqual(third["status"], "complete")
            result, _, _ = replay_run(args.out, tokenizer=FixtureTokenizer())
            costs = result["actual_compute_totals_including_regeneration"]
            self.assertEqual(costs["generated_attempt_positions"], 12)
            self.assertEqual(result["observed_attempts"], 8)
            self.assertEqual(costs["extra_native_tokens_beyond_unique_scored_attempts"], 4 * len(FixtureTokenizer().encode(render_completion(PROOF))))


if __name__ == "__main__":
    unittest.main(verbosity=2)
