"""Reject changed final populations, endpoints, profiles and incomplete releases."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import common
import final_comparators as F


class FinalReleaseTests(unittest.TestCase):
    def test_release_and_mutation_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, cp, summary = [root / x for x in ("data", "checkpoint", "summary")]
            data.write_text("frozen theorem prompts")
            cp.write_text("frozen weights")
            common.write_json(summary, {"status": "complete"})
            lm = root / "lm.json"
            common.write_json(lm, {"status": "frozen", "checkpoint_paths": {"one": str(cp)},
                                   "checkpoint_sha256": {"one": common.sha(cp)}})
            task = {"name": "one", "command": ["python", "eval", "--max-lines", "32"],
                    "endpoint_paths": [str(cp)]}
            spec = {"input_sha256": {str(data): common.sha(data)}, "tasks": [task],
                    "required_training_summaries": [str(summary)], "selection": "all endpoints"}
            path = root / "spec.json"
            common.write_json(path, spec)
            release = F.freeze_release(spec, path, lm)
            F.verify_task(task, release, path)
            changed = copy.deepcopy(task)
            changed["command"][-1] = "64"
            with self.assertRaises(ValueError):
                F.verify_task(changed, release, path)
            data.write_text("changed theorem")
            with self.assertRaises(ValueError):
                F.verify_task(task, release, path)
            data.write_text("frozen theorem prompts")
            cp.write_text("changed weights")
            with self.assertRaises(ValueError):
                F.verify_task(task, release, path)
            cp.write_text("frozen weights")
            common.write_json(summary, {"status": "time_limit"})
            with self.assertRaises(ValueError):
                F.freeze_release(spec, path, lm)
            common.write_json(summary, {"status": "complete"})
            old = json.loads(lm.read_text())
            old["status"] = "active"
            common.write_json(lm, old)
            with self.assertRaises(ValueError):
                F.freeze_release(spec, path, lm)
            changed_release = copy.deepcopy(release)
            changed_release["all_comparator_endpoints_frozen"] = False
            with self.assertRaises(ValueError):
                F.verify_task(task, changed_release, path)


if __name__ == "__main__":
    unittest.main()
