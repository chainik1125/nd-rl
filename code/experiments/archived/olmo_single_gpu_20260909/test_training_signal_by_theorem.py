"""Small native/verifier/event-chain fixtures, with no real-run access."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from protocol import judge, render_completion, render_prompt
from modeling import trim_completion
from training_signal_by_theorem import Snapshot, digest, event_links, replay

PROMPT = "THM P SEQ ( P v Q ) PRF"
PROOF = "N1 P : PR ;\nN2 ( P v Q ) : ORI1 N1 ;\nQED"


class Tokenizer:
    eos_token_id = 9999
    def __len__(self): return 10000
    def encode(self, value, **kwargs): return [ord(char) for char in value]
    def decode(self, ids, **kwargs): return "".join("<eos>" if value == self.eos_token_id else chr(value) for value in ids)


def fixture():
    tokenizer = Tokenizer()
    bank = [{"canonical_hash": str(index), "prompt": PROMPT, "proof": PROOF, "n_lines": 2,
             "name": f"theorem{index}", "family": "fixture"} for index in range(4)]
    rows, events = [], [{"kind": "train_begin", "step": 0}]
    for batch in range(2):
        group_rows = []
        for theorem in range(4):
            for draw in range(4):
                valid = theorem == 1 or (theorem == 0 and (batch == 1 or draw == 0)) or (theorem == 3 and draw > 0)
                capped = theorem == 3 and draw == 0
                text = render_completion(PROOF) if valid else "x" * 128 if capped else "no proof"
                ids = tokenizer.encode(text) + ([tokenizer.eos_token_id] if not valid and not capped else [])
                group_rows.append({"batch": batch, "index": len(group_rows), "canonical_hash": str(theorem),
                    "theorem": PROMPT, "completion": text, "completion_ids": ids,
                    "behavior_logprobs": [-.2] * len(ids), "reward": float(valid), "verdict": judge(PROMPT, text),
                    "mechanical_reward_fixture": False})
        rows.extend(group_rows)
        counts = Counter(trim_completion(tokenizer, row["completion_ids"])[1] for row in group_rows)
        tokens = sum(len(row["completion_ids"]) for row in group_rows)
        success = sum(row["reward"] for row in group_rows)
        mixed = sum(0 < sum(row["reward"] for row in group_rows[start:start+4]) < 4 for start in range(0, 16, 4))
        # Deliberately do not equate reward batch ID with optimizer step index.
        events.extend([{"kind": "generation_batch", "step": batch + 5, "completion_tokens": tokens,
            "prompt_tokens": 16 * len(tokenizer.encode(render_prompt(PROMPT))), "finish_counts": dict(counts)},
            {"kind": "reward_batch", "batch": batch, "examples": 16, "groups": 4, "completion_tokens": tokens,
             "success_fraction": success/16, "mixed_groups": mixed},
            {"kind": "gradient", "step": batch + 5, "norm": 1.0}])
        if batch == 0:
            events.append({"kind": "trainer_log", "step": 6})
    return rows, events, bank, tokenizer


class ReplayTests(unittest.TestCase):
    def test_duplicate_or_regressing_generation_steps_rejected(self):
        _, events, _, _ = fixture()
        for second_step in (5, 4, -1, True):
            changed = deepcopy(events)
            generations = [row for row in changed if row["kind"] == "generation_batch"]
            generations[1]["step"] = second_step
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                event_links(changed)

    def test_vocabulary_size_is_cached_not_recomputed_per_action(self):
        rows, events, bank, _ = fixture()
        class CountedTokenizer(Tokenizer):
            calls = 0
            def __len__(self):
                self.calls += 1
                return super().__len__()
        tokenizer = CountedTokenizer()
        replay(rows, events, bank, tokenizer, cap=128, max_length=100000)
        self.assertEqual(tokenizer.calls, 1)

    def test_repeated_theorem_signal_and_generated_ahead_are_separate(self):
        rows, events, bank, tokenizer = fixture()
        result = replay(rows, events, bank, tokenizer, cap=128, max_length=100000)
        self.assertEqual(result["progress"]["confirmed_completed_optimizer_steps"], 6)
        self.assertEqual(result["aggregate"]["optimizer_completed"]["groups"], 4)
        self.assertEqual(result["aggregate"]["generated_not_confirmed_optimized"]["groups"], 4)
        self.assertEqual(result["theorems_presented_more_than_once"], 4)
        visits = [row for row in result["visits"] if row["canonical_hash"] == "0"]
        self.assertEqual([row["group_class"] for row in visits], ["mixed", "all_one"])
        self.assertEqual([row["generation_optimizer_step"] for row in visits], [5, 6])
        self.assertEqual(result["aggregate"]["all_generated_groups"]["capped_responses"], 2)
        self.assertTrue(all(row["covers_full_bank_exactly_once"] for row in result["observed_sweep_blocks"]))

    def test_partial_group_is_pending_not_a_failed_group(self):
        rows, events, bank, tokenizer = fixture()
        result = replay(rows[:-2], events, bank, tokenizer, cap=128, max_length=100000)
        self.assertEqual(result["native_replayed_responses"], 30)
        self.assertEqual(len(result["visits"]), 7)
        self.assertEqual(result["incomplete_groups_pending"][0]["observed_responses"], 2)

    def test_unjoined_rows_are_not_called_optimized(self):
        rows, _, bank, tokenizer = fixture()
        result = replay(rows[:16], [], bank, tokenizer, cap=128, max_length=100000)
        self.assertEqual(result["aggregate"]["rollouts_without_event_join"]["groups"], 4)
        self.assertEqual(result["progress"]["confirmed_completed_optimizer_steps"], 0)

    def test_native_text_reward_and_behavior_tampering_rejected(self):
        mutations = [lambda row: row.update(completion="repaired text"),
                     lambda row: row.update(reward=0.0),
                     lambda row: row["behavior_logprobs"].pop(),
                     lambda row: row["verdict"].update(accepted=False)]
        for mutation in mutations:
            rows, events, bank, tokenizer = fixture()
            mutation(rows[0])
            with self.assertRaises(ValueError):
                replay(rows, events, bank, tokenizer, cap=128, max_length=100000)

    def test_changed_theorem_identity_and_event_native_counts_rejected(self):
        rows, events, bank, tokenizer = fixture()
        rows[0]["theorem"] = "THM Q SEQ Q PRF"
        with self.assertRaisesRegex(ValueError, "Theorem text"):
            replay(rows, events, bank, tokenizer, cap=128, max_length=100000)
        rows, events, bank, tokenizer = fixture()
        events[1]["completion_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "receipts"):
            replay(rows, events, bank, tokenizer, cap=128, max_length=100000)


class SnapshotTests(unittest.TestCase):
    def test_pinned_source_digest_rejects_replacement_map(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "freeze.json"
            original = b'{"sha256":{"input":"original"}}\n'
            path.write_bytes(original)
            expected = digest(original)
            self.assertEqual(Snapshot().read(path, expected_sha256=expected),
                             {"sha256": {"input": "original"}})
            path.write_bytes(b'{"sha256":{"input":"replacement"}}\n')
            with self.assertRaisesRegex(ValueError, "Pinned source digest"):
                Snapshot().read(path, expected_sha256=expected)

    def test_uncommitted_tail_ignored_but_committed_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prefix.jsonl"
            raw = b'{"one":1}\n{"unfinished":'
            path.write_bytes(raw)
            snapshot = Snapshot()
            self.assertEqual(snapshot.read(path, jsonl=True), [{"one": 1}])
            self.assertEqual(path.read_bytes(), raw)
            self.assertGreater(snapshot.files[0]["pending_trailing_bytes"], 0)
            path.write_bytes(b'{"one":1}\ncorrupt\n')
            with self.assertRaises(ValueError):
                Snapshot().read(path, jsonl=True)


if __name__ == "__main__":
    unittest.main()
