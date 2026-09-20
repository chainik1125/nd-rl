"""Isolated CPU length-audit fixtures; never inspect production rollouts."""

from contextlib import ExitStack
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import rollout_length_audit as audit_module
from protocol import judge, render_completion


SHORT = "N1 P : PR ; QED"
REDUNDANT = "N1 P : PR ; " + " ".join(
    f"N{i} P : R N{i - 1} ;" for i in range(2, 9)) + " QED"
DEPENDENT = "N1 P : PR ; " + " ".join(
    f"N{i} ( P & P ) : ANDI N{i - 1} N{i - 1} ; "
    f"N{i + 1} P : ANDE1 N{i} ;" for i in range(2, 9, 2)) + " QED"


class LiteralTokenizer:
    eos_token_id = 2_000_000

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        if skip_special_tokens or clean_up_tokenization_spaces:
            raise AssertionError("Native replay must not clean or skip tokens")
        return "".join(chr(token) for token in ids)


class LengthAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "runs/grpo_fixture"
        self.run.mkdir(parents=True)
        (self.root / "data").mkdir()
        self.bank_path = self.root / "data/shared_cap6.jsonl"
        self.rollouts_path = self.run / "rollouts.jsonl"
        self.bank = []
        formula = "P"
        for index in range(32):
            prompt = f"THM {formula} SEQ {formula} PRF"
            self.bank.append({"canonical_hash": hashlib.sha256(prompt.encode()).hexdigest(),
                "prompt": prompt, "proof": f"N1 {formula} : PR ; QED",
                "family": "fixture_identity", "n_lines": 1})
            formula = f"( P > {formula} )"
        self.rows = [self.row(0, SHORT), self.row(1, REDUNDANT),
                     self.row(2, DEPENDENT), self.row(3, raw="not a proof")]
        self.write_bank()
        self.write_rows()
        (self.root / "plans").mkdir()
        self.freeze_path = self.root / "plans/primary_source_freeze_v4.json"
        self.vendor = self.root / "vendor"
        self.vendor.mkdir()
        (self.vendor / "fixture_verify.py").write_text("# synthetic source receipt\n")
        model_dir = self.root / "models/olmo32b"
        model_dir.mkdir(parents=True)
        for name in audit_module.MODEL_METADATA:
            (model_dir / name).write_text("{}\n")
        source_dir = self.root / "code"
        source_dir.mkdir()
        self.sources = []
        for name in ("modeling.py", "protocol.py", "training_diagnostics.py", "proof_diagnostics.py"):
            path = source_dir / name
            path.write_text("# synthetic source receipt\n")
            self.sources.append(path)
        self.stable_paths = [self.bank_path, *self.sources, self.vendor / "fixture_verify.py",
                            *(model_dir / name for name in audit_module.MODEL_METADATA)]
        freeze_sha = self.write_freeze()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(audit_module, "ROOT", self.root))
        self.stack.enter_context(patch.object(audit_module, "PRIMARY_FREEZE_SHA", freeze_sha))
        self.stack.enter_context(patch.object(audit_module, "REPLAY_SOURCES", tuple(self.sources)))
        self.stack.enter_context(patch.object(audit_module, "VENDOR", self.vendor))
        self.stack.enter_context(patch.object(audit_module, "load_tokenizer", return_value=LiteralTokenizer()))

    def row(self, index, proof=None, raw=None):
        source = self.bank[0]
        text = raw if raw is not None else render_completion(proof)
        verdict = judge(source["prompt"], text)
        ids = list(map(ord, text))
        return {"batch": 0, "index": index, "canonical_hash": source["canonical_hash"],
            "theorem": source["prompt"], "completion": text, "completion_ids": ids,
            "verdict": verdict, "reward": verdict["reward"],
            "behavior_logprobs": [-1.0] * len(ids), "mechanical_reward_fixture": False}

    def write_bank(self):
        self.bank_path.write_text("".join(json.dumps(row) + "\n" for row in self.bank))

    def write_rows(self):
        self.rollouts_path.write_text("".join(json.dumps(row) + "\n" for row in self.rows))

    def write_freeze(self):
        values = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in self.stable_paths}
        # Deliberately absent protected inputs must not be opened by the helper.
        values[str(self.root / "textbook/DO_NOT_READ.jsonl")] = "0" * 64
        values[str(self.root / "data/heldout_DO_NOT_READ.jsonl")] = "0" * 64
        self.freeze_path.write_text(json.dumps({"sha256": values}, sort_keys=True))
        return hashlib.sha256(self.freeze_path.read_bytes()).hexdigest()

    def test_raw_and_compressed_counts_preserve_reward_and_input_bytes(self):
        before = {p: p.read_bytes() for p in (self.bank_path, self.rollouts_path)}
        report = audit_module.audit(self.run)
        self.assertEqual(report["counts"], {"responses": 4, "valid_responses": 3,
            "valid_raw_above_six": 2, "valid_compressed_above_six": 1,
            "invalid_responses": 1})
        self.assertEqual(report["valid_raw_line_histogram"], {1: 1, 8: 1, 9: 1})
        self.assertEqual(report["valid_compressed_line_histogram"], {1: 2, 9: 1})
        self.assertEqual(report["by_batch"]["0"]["responses"], 4)
        self.assertEqual([row["analysis_compressed_lines"]
                          for row in report["longer_positive_responses"]], [1, 9])
        self.assertEqual([row["provided_gold_lines"]
                          for row in report["longer_positive_responses"]], [1, 1])
        self.assertEqual([self.rows[i]["reward"] for i in (1, 2)], [1.0, 1.0])
        self.assertEqual(report["provided_gold_max_lines"], 1)
        self.assertEqual(report["provided_gold_cap_lines"], 6)
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)
        inputs = {entry["path"]: entry for entry in report["source_prefixes"]}
        for path, content in before.items():
            self.assertEqual(inputs[str(path)]["sha256_of_observed_bytes"],
                             hashlib.sha256(content).hexdigest())
        self.assertFalse(report["held_out_answers_read"])
        self.assertFalse(report["protected_payloads_read"])
        self.assertIn(str(self.root / "models/olmo32b/tokenizer.json"), report["stable_input_sha256"])
        self.assertIn(str(self.sources[-1]), report["stable_input_sha256"])
        self.assertNotIn(str(self.rollouts_path), report["stable_input_sha256"])

    def test_native_text_and_stored_reward_or_verdict_mismatch_rejected(self):
        original = copy.deepcopy(self.rows)
        for field, bad in (("completion_ids", [ord("x")]), ("reward", 0.0),
                           ("verdict", dict(self.rows[0]["verdict"], n_lines=8))):
            with self.subTest(field=field):
                self.rows = copy.deepcopy(original)
                self.rows[0][field] = bad
                self.write_rows()
                with self.assertRaises(ValueError):
                    audit_module.audit(self.run)

    def test_literal_invalid_suffix_is_not_repaired_or_compressed(self):
        self.rows = [self.row(0, raw=render_completion(REDUNDANT) + "extra")]
        self.write_rows()
        with patch.object(audit_module, "compress_proof", side_effect=AssertionError("Invalid raw proof reached compressor")):
            report = audit_module.audit(self.run)
        self.assertEqual(report["counts"]["invalid_responses"], 1)
        self.assertFalse(report["valid_raw_line_histogram"])

    def test_sampled_terminal_eos_is_ignored_only_for_literal_text(self):
        self.rows = [self.row(0, SHORT)]
        self.rows[0]["completion_ids"].append(LiteralTokenizer.eos_token_id)
        self.write_rows()
        self.assertEqual(audit_module.audit(self.run)["counts"]["valid_responses"], 1)

    def test_mechanical_or_unclassified_samples_and_out_of_bank_theorems_rejected(self):
        original = copy.deepcopy(self.rows)
        for field, value in (("mechanical_reward_fixture", True),
                             ("mechanical_reward_fixture", None),
                             ("canonical_hash", "unknown"),
                             ("theorem", "THM Q SEQ Q PRF")):
            with self.subTest(field=field, value=value):
                self.rows = copy.deepcopy(original)
                self.rows[0][field] = value
                self.write_rows()
                with self.assertRaises(ValueError):
                    audit_module.audit(self.run)

    def test_bad_compression_is_rejected_not_counted_as_success(self):
        for value in ({"compressed_proof": "N1 Q : PR ; QED", "compressed_length": 1},
                      {"compressed_proof": SHORT, "compressed_length": 2},
                      {"compressed_proof": REDUNDANT, "compressed_length": 8}):
            with self.subTest(value=value):
                with patch.object(audit_module, "compress_proof", return_value=value):
                    with self.assertRaises(ValueError):
                        audit_module.audit(self.run)

    def test_provided_gold_longer_than_cap_rejected_even_if_valid(self):
        self.bank[0].update(proof=REDUNDANT, n_lines=8)
        self.write_bank()
        audit_module.PRIMARY_FREEZE_SHA = self.write_freeze()
        with self.assertRaises(ValueError):
            audit_module.audit(self.run)

    def test_duplicate_bank_identity_rejected_even_in_a_consistent_fixture_freeze(self):
        self.bank[-1]["canonical_hash"] = self.bank[0]["canonical_hash"]
        self.write_bank()
        audit_module.PRIMARY_FREEZE_SHA = self.write_freeze()
        with self.assertRaisesRegex(ValueError, "distinct canonical identities"):
            audit_module.audit(self.run)

    def test_static_source_tokenizer_and_freeze_drift_rejected_before_tokenizer_loading(self):
        for path in (self.sources[0], self.sources[-1],
                     self.root / "models/olmo32b/tokenizer.json", self.freeze_path):
            with self.subTest(path=path):
                old = path.read_bytes()
                path.write_bytes(old + b" \n")
                with patch.object(audit_module, "load_tokenizer", side_effect=AssertionError("Loaded before provenance check")):
                    with self.assertRaises(ValueError):
                        audit_module.audit(self.run)
                path.write_bytes(old)

    def test_unpinned_imported_source_rejected(self):
        frozen = json.loads(self.freeze_path.read_text())
        del frozen["sha256"][str(self.sources[-1])]
        self.freeze_path.write_text(json.dumps(frozen))
        audit_module.PRIMARY_FREEZE_SHA = hashlib.sha256(self.freeze_path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "not pinned"):
            audit_module.audit(self.run)

    def test_later_live_append_does_not_invalidate_observed_snapshot(self):
        compressor = audit_module.compress_proof
        appended = False

        def append_while_compressing(*args):
            nonlocal appended
            if not appended:
                with self.rollouts_path.open("a") as stream:
                    stream.write(json.dumps(self.row(4, SHORT)) + "\n")
                appended = True
            return compressor(*args)

        with patch.object(audit_module, "compress_proof", side_effect=append_while_compressing):
            report = audit_module.audit(self.run)
        self.assertEqual(report["counts"]["responses"], 4)
        self.assertEqual(len(self.rollouts_path.read_text().splitlines()), 5)
        observed = next(value for value in report["source_prefixes"]
                        if value["path"] == str(self.rollouts_path))
        self.assertEqual(observed["parsed_records"], 4)

    def test_static_input_changed_during_replay_is_rejected(self):
        compressor = audit_module.compress_proof

        def mutate_while_compressing(*args):
            self.sources[-1].write_text("# changed during audit\n")
            return compressor(*args)

        with patch.object(audit_module, "compress_proof", side_effect=mutate_while_compressing):
            with self.assertRaisesRegex(ValueError, "changed during replay"):
                audit_module.audit(self.run)

    def test_malformed_committed_record_rejected_but_pending_append_reported(self):
        complete = self.rollouts_path.read_bytes()
        self.rollouts_path.write_bytes(complete + b'{"batch":\n')
        with self.assertRaises(ValueError):
            audit_module.audit(self.run)
        self.rollouts_path.write_bytes(complete + b'{"batch":')
        report = audit_module.audit(self.run)
        self.assertEqual(report["counts"]["responses"], 4)
        self.assertEqual(report["warnings"][0]["kind"], "incomplete_trailing_append_pending")
        self.assertTrue(report["non_atomic_training_snapshot"])

    def test_audit_outside_runs_rejected_before_any_snapshot_read(self):
        with patch.object(audit_module.Snapshot, "read", side_effect=AssertionError("Unexpected read")):
            with self.assertRaises(ValueError):
                audit_module.audit(self.root / "textbook")

    def test_only_expected_source_and_training_fixture_files_are_read(self):
        original_open = Path.open
        seen = []

        def guarded_open(path, *args, **kwargs):
            resolved = path.resolve()
            seen.append(resolved)
            allowed = {*self.stable_paths, self.freeze_path, self.rollouts_path}
            if resolved.is_relative_to(self.root) and resolved not in allowed:
                raise AssertionError(f"Unexpected fixture payload read: {resolved}")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", guarded_open):
            audit_module.audit(self.run)
        self.assertIn(self.bank_path, seen)
        self.assertIn(self.rollouts_path, seen)
        self.assertFalse(any("textbook" in str(path) or "prediction" in str(path) for path in seen))


if __name__ == "__main__":
    unittest.main(verbosity=2)
