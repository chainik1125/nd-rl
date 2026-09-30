#!/usr/bin/env python3
"""Run a job list on one rented A40: test_time_scaling/launch.py, with the TTRL files staged too.

  source .envrc.local && code/.venv/bin/python \
      code/experiments/current/ttrl/launch.py <name> <jobs.txt>

Job lines are `<out-name> ttrl.py <args...>`; outputs land in artifacts/ttrl/<name>/.
"""

import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CUR = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(CUR, "test_time_scaling"))
import launch as tts  # noqa: E402

tts.LOCAL = os.path.join(CUR, "..", "..", "..", "artifacts", "ttrl")
_stage = tts.stage


def stage(name, jobs):
    d = _stage(name, jobs)
    shutil.copy2(
        os.path.join(CUR, "autoresearch", "harness.py"), os.path.join(d, "harness_frozen.py")
    )
    shutil.copy2(os.path.join(CUR, "combined_pipeline_20260928", "leon_rl.py"), d)
    for f in os.listdir(HERE):
        if f.endswith((".py", ".txt")) and f != "launch.py":
            shutil.copy2(os.path.join(HERE, f), d)
    return d


tts.stage = stage
if __name__ == "__main__":
    tts.main()
