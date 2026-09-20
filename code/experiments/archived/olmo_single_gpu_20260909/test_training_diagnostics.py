"""CPU fixtures for training accounting; no production predictions are read."""

import json
from pathlib import Path
import tempfile
import unittest

from protocol import judge, render_completion
from training_diagnostics import Snapshot, build_report, epoch_exposure_summary, optimizer_summary, pair_summary, rollout_summary


PROMPT = "THM P SEQ ( P v Q ) PRF"
VALID = render_completion("N1 P : PR ; N2 ( P v Q ) : ORI1 N1 ; QED")
GOLD = {"a": {"canonical_hash": "a", "prompt": PROMPT, "family": "ori"}}


def exposure(size, steps, epoch, runs=1):
    events = [{"kind": "stage_start", "rows": size}, {"kind": "train_begin", "step": 0},
        {"kind": "trainer_log", "step": steps, "metrics": {"epoch": epoch}}]
    summary = [{"optimizer_steps": steps, "metrics": {"epoch": epoch}}]
    state = {"global_step": steps, "epoch": epoch, "train_batch_size": 1}
    return epoch_exposure_summary("dpo", events, {"per_device_train_batch_size": 1, "gradient_accumulation_steps": 4, "dataloader_drop_last": False}, summary, state, runs, True, [{"family": "f"}] * size)


class ExposureTests(unittest.TestCase):
    def test_five_pairs_flush_one_row_at_each_epoch_end(self):
        result = exposure(5, 40, 20.0)
        self.assertEqual(result["actual_row_presentations"], 100)
        self.assertEqual(result["response_presentations"], 200)
        self.assertEqual(result["nominal_steps_times_batch_times_accumulation"], 160)
        self.assertEqual(result["optimizer_cardinality_pattern_per_epoch"], [4, 1])
        self.assertEqual(result["actual_optimizer_update_cardinality_histogram"], {4: 20, 1: 20})
        self.assertEqual(result["family_row_presentations_for_complete_epochs"], {"f": 100})

    def test_partial_epoch_presentations(self):
        result = exposure(5, 15, 7.8)
        self.assertEqual(result["actual_row_presentations"], 39)
        self.assertEqual(result["actual_optimizer_update_cardinality_histogram"], {4: 8, 1: 7})
        self.assertIsNone(result["family_row_presentations_for_complete_epochs"])

    def test_divisible_sft_schedule_matches_nominal(self):
        result = exposure(32, 80, 10.0)
        self.assertEqual(result["actual_row_presentations"], 320)
        self.assertEqual(result["actual_optimizer_update_cardinality_histogram"], {4: 80})

    def test_resumes_and_contradictory_epochs_return_unknown(self):
        self.assertTrue(exposure(5, 40, 20, runs=2)["status"].startswith("unknown"))
        self.assertTrue(exposure(5, 40, 19.5)["status"].startswith("unknown"))

    def test_first_change_probe_is_not_nonzero_gradient_count(self):
        events = [{"kind": "train_begin", "step": 0}, *[{"kind": "gradient", "step": index, "norm": norm} for index, norm in enumerate([1.0, .2, 0.0])], {"kind": "parameter_update", "step": 0, "max_abs_change": .01}]
        result = optimizer_summary(events, [], None)
        self.assertEqual(result["nonzero_gradient_observations"], 2)
        self.assertEqual(result["positive_parameter_change_probe_events"], 1)
        self.assertEqual(result["zero_gradient_observations"], 1)


class RolloutTests(unittest.TestCase):
    def rows(self, batch, rewards, mechanical=False):
        return [{"batch": batch, "index": index, "canonical_hash": "a", "theorem": PROMPT,
            "completion": VALID if reward else "bad proof", "reward": reward,
            "verdict": judge(PROMPT, VALID if reward else "bad proof"), "completion_ids": [1, 2],
            "behavior_logprobs": [-1.0, -1.0], "mechanical_reward_fixture": mechanical}
            for index, reward in enumerate(rewards)]

    def test_group_classes_resume_counter_reset_and_incomplete_tail(self):
        rows = self.rows(0, [1, 0, 0, 0]) + self.rows(1, [0, 0, 0, 0]) + self.rows(0, [1, 1, 1, 1]) + self.rows(1, [0, 0, 0]) + self.rows(2, [1, 0, 0, 0], mechanical=True)
        result = rollout_summary(rows, [], 4, GOLD)
        self.assertEqual(result["actual_verifier_rollout_records"], 15)
        self.assertEqual(result["mechanical_rollout_records_excluded"], 4)
        self.assertEqual(result["complete_group_counts"], {"mixed": 1, "all_zero": 1, "all_one": 1})
        self.assertEqual(result["trailing_incomplete_group_rows"], 3)
        self.assertEqual(result["positive_reward_groups"], 2)
        self.assertAlmostEqual(result["mixed_group_fraction"], 1 / 3)

    def test_malformed_group_indices_are_not_guessed(self):
        rows = self.rows(0, [0, 0, 0, 0])
        rows[2]["index"] = 8
        result = rollout_summary(rows, [], 4, GOLD)
        self.assertEqual(result["malformed_indexed_reward_batch_chunks"], 1)
        self.assertEqual(result["binary_classified_groups"], 0)

    def test_adaptive_pair_samples_and_coverage_are_separate(self):
        samples = [{"canonical_hash": "a", "attempt": attempt, "sample": {"text": text, "response_ids": [1, 2], "finish_reason": "final_marker", "completion_tokens": 2}, "verdict": judge(PROMPT, text)} for attempt, text in enumerate([VALID, VALID, "bad proof"])]
        pairs = [{"canonical_hash": "a", "rejected": "bad proof", "generation_policy": "sft_adapter", "source_adapter": "fixture", "source_adapter_sha256": "x"}]
        result = pair_summary(samples, pairs, GOLD)
        self.assertEqual(result["sample_records"], 3)
        self.assertEqual(result["strictly_valid_samples"], 2)
        self.assertEqual(result["pair_records"], 1)
        self.assertEqual(result["covered_shared_theorems"], 1)


class SnapshotTests(unittest.TestCase):
    def test_partial_tail_is_reported_and_input_stays_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            content = b'{"kind":"one"}\n{"kind":'
            path.write_bytes(content)
            snapshot = Snapshot()
            self.assertEqual(snapshot.read(path, True), [{"kind": "one"}])
            self.assertEqual(path.read_bytes(), content)
            self.assertEqual(snapshot.warnings[0]["kind"], "incomplete_trailing_append_pending")

    def test_empty_and_mechanical_stages_stay_out_of_research_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = build_report(root, native_replay=False)
            self.assertEqual(report["status"], "pending_no_research_training_records")
            run = root / "runs/sft"
            run.mkdir(parents=True)
            (run / "run.json").write_text(json.dumps({"mechanical_fixture": True}))
            (run / "events.jsonl").write_text(json.dumps({"kind": "gradient", "step": 0, "norm": 1}) + "\n")
            report = build_report(root, native_replay=False)
            self.assertEqual(report["stages"]["sft"]["status"], "excluded_mechanical_stage")
            self.assertEqual(report["status"], "pending_no_research_training_records")

    def test_explicit_override_preserves_preupdate_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs/grpo_isolated"
            run.mkdir(parents=True)
            (run / "run.json").write_text(json.dumps({"mechanical_fixture": False, "args": {"group_size": 4}}))
            (run / "events.jsonl").write_text(json.dumps({"kind": "actual_grpo_likelihood_audit", "step": 0, "max_abs_error": .5, "tolerance": .1}) + "\n")
            report = build_report(root, native_replay=False, stage_directories={"grpo": run})
            stage = report["stages"]["grpo"]
            self.assertEqual(stage["status"], "failed_numerical_audit_before_recorded_optimizer_step")
            self.assertEqual(stage["optimizer"]["pre_optimizer_callback_observations"], 0)
            self.assertFalse(stage["completion_record_present"])
            with self.assertRaises(ValueError):
                build_report(root, native_replay=False, stage_directories={"grpo": root.parent / "outside"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
