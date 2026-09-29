#!/usr/bin/env python3
"""One factorial cell x seed on its own pod: stage, rent, run, pull, terminate.

  source .envrc.local && code/.venv/bin/python \
      code/experiments/current/factorial_20260929/launch.py <tok> <pre> <rl> <seed> [--smoke-dir D]

  tok  abs | lean     tokenizer (abs = the token era's; lean = Dan's lean_seq)
  pre  naive | best   pretrain_naive.py (012 baseline) | pretrain_best.py (186 incumbent)
  rl   ei | leon      frozen harness expert iteration | c005's recipe (Leon's EI + guided
                      sampling + 1,800 s deadline)

Runs are fact-<tok>-<pre>-<rl>_s<seed> under artifacts/autoresearch/. --smoke-dir only stages
into D (for a local `harness.py --smoke`) and rents nothing.
"""

import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMBINED = os.path.join(HERE, "..", "combined_pipeline_20260928")
AR = os.path.join(HERE, "..", "autoresearch")
sys.path.insert(0, COMBINED)
import launch as combined  # noqa: E402  (pod helpers: run_staged)

TOTAL_SECS = 4200  # the 2,400 s pipeline + ~300-600 s of pass@k sampling + slack
TOK = {"abs": "abs", "lean": "lean_seq"}
LEON_CFG = {"guided": True, "deadline": 1800}  # c005


def stage(tok, pre, rl, seed, d):
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    with open(os.path.join(AR, "remote.sh")) as f:
        remote = f.read()
    with open(os.path.join(d, "remote.sh"), "w") as f:
        f.write(f"export ND_TOTAL_SECS={TOTAL_SECS} COMBO_TOK={TOK[tok]} COMBO_RL={rl}\n" + remote)
    shutil.copy2(os.path.join(AR, "harness.py"), os.path.join(d, "harness_frozen.py"))
    shutil.copy2(os.path.join(HERE, "harness_combo.py"), os.path.join(d, "harness.py"))
    shutil.copy2(os.path.join(HERE, f"pretrain_{pre}.py"), os.path.join(d, "pretrain.py"))
    shutil.copy2(os.path.join(HERE, "passk.py"), d)
    shutil.copytree(os.path.join(HERE, "textbook72"), os.path.join(d, "textbook72"))
    for f in ("leon_rl.py", "fast_sample.py", "guided.py", "line_checker.py"):
        shutil.copy2(os.path.join(COMBINED, f), d)
    with open(os.path.join(d, "leon_cfg.json"), "w") as f:
        json.dump(LEON_CFG, f)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tok", choices=TOK)
    ap.add_argument("pre", choices=("naive", "best"))
    ap.add_argument("rl", choices=("ei", "leon"))
    ap.add_argument("seed", type=int)
    ap.add_argument("--smoke-dir")
    a = ap.parse_args()
    name = f"fact-{a.tok}-{a.pre}-{a.rl}"
    if a.smoke_dir:
        print(stage(a.tok, a.pre, a.rl, a.seed, a.smoke_dir))
        return
    d = os.path.join(combined.pod.LOCAL_ART, "factorial_stage", f"{name}_s{a.seed}")
    combined.run_staged(stage(a.tok, a.pre, a.rl, a.seed, d), name, a.seed)


if __name__ == "__main__":
    main()
