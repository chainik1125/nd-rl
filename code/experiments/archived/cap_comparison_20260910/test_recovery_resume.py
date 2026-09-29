"""Supervisor recovery must preserve completed and running trials."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import recover


class StopWaiting(Exception):
    pass


class Recovery(unittest.TestCase):
    def test_resume_adopts_without_relaunch_or_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for relative in ("plans/jobs","reports","jobs/done","runs/done"):
                (root/relative).mkdir(parents=True)
            protocol={"arms":["done","live","pending"],"input_sha256":{}}
            recover.write(root/"plans/protocol.json",protocol)
            digest=recover.sha(root/"plans/protocol.json")
            recover.write(root/"reports/recovery_protocol.json",{"training_protocol_sha256":digest,"remote_quiet_seconds":600})
            result={"returncode":0}
            recover.write(root/"jobs/done/result.json",result)
            recover.write(root/"runs/done/pipeline_completed.json",{})
            before={"protocol_sha256":digest,"pending":["pending"],"assignments":{},
                    "running":{"live":{"pid":123,"host":"simplex1","gpu":2,"job":"live_retry1","adopted":False}},
                    "finished":{"done":{"success":True,"job":"done","result":result}}}
            recover.write(root/"reports/recovery_state.json",before)
            with patch.object(sys,"argv",["recover.py","--root",str(root),"--resume"]), \
                    patch.object(recover,"owned_alive",return_value=True), \
                    patch.object(recover,"state",return_value={2:{"pids":[456],"memory":1000,"util":50}}), \
                    patch.object(recover,"remote_state",return_value={}), \
                    patch.object(recover.subprocess,"Popen") as launch, \
                    patch.object(recover.time,"sleep",side_effect=StopWaiting):
                with self.assertRaises(StopWaiting):recover.main()
                launch.assert_not_called()
            after=json.loads((root/"reports/recovery_state.json").read_text())
            self.assertEqual(after["finished"],before["finished"])
            self.assertEqual(after["pending"],before["pending"])
            self.assertEqual(after["running"]["live"]["pid"],123)
            self.assertEqual(after["running"]["live"]["job"],"live_retry1")
            self.assertTrue(after["running"]["live"]["adopted"])


if __name__ == "__main__":
    unittest.main()
