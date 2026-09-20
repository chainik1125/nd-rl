import copy
import json
from pathlib import Path
import tempfile
import unittest
import common
from nd_environment import strict_complete
from symbolic_general_env import Environment
import analyze_symbolic as A


class ReplayAuditTests(unittest.TestCase):
    def fixture(self, root):
        run = root / "eval"
        run.mkdir()
        population, audit, attempts = [], [], []
        for i, goal in enumerate(("P", "Q")):
            prompt = f"THM ( P & Q ) SEQ {goal} PRF"
            env = Environment(prompt, max_lines=2, max_scope=4)
            actions = env.actions()
            index = next(j for j, a in enumerate(actions) if a.rule == "ANDE1")
            action = actions[index]
            env.step(action)
            success = env.success()
            sid = str(i)
            population.append({"source_id": sid, "prompt": prompt})
            audit.append({"source_id": sid, "prompt": prompt, "n_lines": 2, "length_bin": "2", "family": "ANDE"})
            pruned = env.pruned_body() if success else None
            attempts.append({"source_id": sid, "prompt": prompt, "attempt": 0, "family": "unlabeled",
                "success": success, "supplied_prefix_lines": 0, "generated_actions": 1,
                "registered_premise_lines": 1, "candidates_scored": len(actions),
                "body": env.body(), "verdict": strict_complete(prompt, env.body()),
                "pruned_body": pruned, "pruned_verdict": strict_complete(prompt, pruned) if success else None,
                "action_trace": [{"rule": action.rule, "formula": action.formula, "refs": action.refs,
                    "depth": action.depth, "candidate_count": len(actions), "selected_index": index}]})
        data, audit_path = root / "population.jsonl", root / "audit.jsonl"
        common.write_jsonl(data, population)
        common.write_jsonl(audit_path, audit)
        program = Path(A.__file__).parent
        config = {"population": str(data), "population_sha256": common.sha(data), "checkpoint": None,
            "checkpoint_sha256": None, "code_hashes": {"symbolic_general_env.py": common.sha(program / "symbolic_general_env.py")},
            "checker_sha256": common.sha(program / "nd_environment.py"), "policy": "random", "attempts": 1,
            "sample": False, "seed": 1, "max_lines": 2, "max_scope": 4}
        common.write_json(run / "symbolic_general_eval_config.json", config)
        common.write_jsonl(run / "symbolic_general_attempts.jsonl", attempts)
        result = {"status": "complete", "config": config, "attempts": 2, "theorems_with_any_success": 1,
                  "groups": {"all": {"successes": 1}}, "attempts_sha256": common.sha(run / "symbolic_general_attempts.jsonl")}
        common.write_json(run / "symbolic_general_eval_summary.json", result)
        return run, audit_path, program, attempts, result

    def test_valid_and_failed_certificates(self):
        with tempfile.TemporaryDirectory() as d:
            run, audit, program, _, _ = self.fixture(Path(d))
            result = A.analyze(run, audit, program)
            self.assertEqual(result["summaries"]["all"]["any"], 1)
            self.assertEqual(result["summaries"]["all"]["maximum_pruned"], 2)
            self.assertEqual(result["summaries"]["all"]["raw_eligible_512_first"], 1)
            self.assertEqual(result["root_replay_counts"]["inferences_replayed"], 2)

    def test_false_archive_claims_rejected_even_with_updated_archive_hash(self):
        mutations = [lambda rows: rows[0].update(supplied_prefix_lines=1),
                     lambda rows: rows[0]["action_trace"][0].update(selected_index=9999),
                     lambda rows: rows[0].update(pruned_body="QED"),
                     lambda rows: rows.pop(),
                     lambda rows: rows.append(copy.deepcopy(rows[0]))]
        for mutate in mutations:
            with self.subTest(mutation=mutate), tempfile.TemporaryDirectory() as d:
                run, audit, program, rows, result = self.fixture(Path(d))
                mutate(rows)
                common.write_jsonl(run / "symbolic_general_attempts.jsonl", rows)
                result["attempts_sha256"] = common.sha(run / "symbolic_general_attempts.jsonl")
                common.write_json(run / "symbolic_general_eval_summary.json", result)
                with self.assertRaises(ValueError):
                    A.analyze(run, audit, program)


if __name__ == "__main__":
    unittest.main()
