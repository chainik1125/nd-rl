"""A remotely copied marker must not release analysis before transfer success."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import finalize


class StopWaiting(Exception):
    pass


class Readiness(unittest.TestCase):
    def test_copied_marker_waits_for_coordinator(self):
        for success in (False, True):
            with self.subTest(success=success), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                for relative in ("plans", "reports", "runs/invented"):
                    (root/relative).mkdir(parents=True)
                (root/"plans/protocol.json").write_text(json.dumps({"arms":["invented"]}))
                (root/"runs/invented/pipeline_completed.json").write_text('{}')
                (root/"reports/recovery_state.json").write_text(json.dumps({"finished":{"invented":{"success":success}}}))
                with patch.object(sys,"argv",["finalize.py","--root",str(root)]), \
                        patch.object(finalize.subprocess,"run") as run, \
                        patch.object(finalize.time,"sleep",side_effect=StopWaiting):
                    if success:
                        finalize.main()
                        self.assertTrue(run.called)
                        self.assertTrue(json.loads((root/"reports/finalization_state.json").read_text())["complete"])
                    else:
                        with self.assertRaises(StopWaiting): finalize.main()
                        run.assert_not_called()
                        self.assertFalse((root/"reports/finalization_state.json").exists())


if __name__ == "__main__":
    unittest.main()
