#!/usr/bin/env python3
"""One recipe x arm x seed on its own pod: stage, rent, run, pull, terminate.

  source .envrc.local && code/.venv/bin/python \
      code/experiments/current/state_recipes_20260930/launch.py <recipe> <arm> <seed> \
      [--smoke-dir D]

  recipe  198 | 201 | 204 | 207    autoresearch pretrain.py, with the state-arm data path (README)
  arm     C0 | SN | SH             whole proof | state, env names | state + history
                                   (harness_state.py)

Runs are st-<recipe>-<arm>_s<seed> under artifacts/autoresearch/. --smoke-dir only stages into D
(for a local `harness.py --smoke`) and rents nothing. Reuses the combined pipeline's run_staged.
"""

import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "combined_pipeline_20260928"))
import launch as combined  # noqa: E402  (pod helpers: run_staged)

# RL is fixed in samples, so the alarm only has to be long enough; the state arms' step-level
# sampling is slower than whole-proof sampling (set from the benchmark pod, README).
TOTAL_SECS = int(os.environ.get("ST_TOTAL_SECS", 4 * 3600))


def stage(recipe, arm, d):
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    with open(os.path.join(HERE, "remote.sh")) as f:
        remote = f.read()
    with open(os.path.join(d, "remote.sh"), "w") as f:
        f.write(f"export ND_TOTAL_SECS={TOTAL_SECS} STATE_ARM={arm}\n" + remote)
    shutil.copy2(os.path.join(HERE, "harness_state.py"), os.path.join(d, "harness.py"))
    shutil.copy2(os.path.join(HERE, f"pretrain_{recipe}.py"), os.path.join(d, "pretrain.py"))
    for f in ("harness_frozen.py", "state_pairs.py"):
        shutil.copy2(os.path.join(HERE, f), d)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recipe", choices=("198", "201", "204", "207"))
    ap.add_argument("arm", choices=("C0", "SN", "SH"))
    ap.add_argument("seed", type=int)
    ap.add_argument("--smoke-dir")
    a = ap.parse_args()
    name = f"st-{a.recipe}-{a.arm}"
    if a.smoke_dir:
        print(stage(a.recipe, a.arm, a.smoke_dir))
        return
    # pod self-destruct: the alarm plus boot, setup and slack
    os.environ["ND_POD_MAX_SECS"] = str(TOTAL_SECS + 1800)
    d = os.path.join(combined.pod.LOCAL_ART, "state_stage", f"{name}_s{a.seed}")
    combined.run_staged(stage(a.recipe, a.arm, d), name, a.seed)


if __name__ == "__main__":
    main()
