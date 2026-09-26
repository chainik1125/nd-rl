#!/usr/bin/env python3
"""lengthexp/evaluate.py, unchanged, pointed at this experiment's test sets and output directory.

    python mt_evaluate.py --ckpt PATH --label NAME [--sets std,matched,long] [evaluate.py options]

testsets/std.jsonl and testsets/matched.jsonl are byte copies of lengthexp/testsets (the final draw, so every arm here
sees the same theorems as lengthexp's arms A-D); testsets/long.jsonl is built by build_long.py from the data agent's
test_long.jsonl. Output: eval/<label>/ (same layout as lengthexp/eval). k = 32, T = 0.8, max_new 1024 (evaluate.py
defaults), through the GPU lock (--gpu auto).
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
LE = '/scratch/cowifornia/nd-rl/ab3/lengthexp'
sys.path.insert(0, LE)
import evaluate as E  # noqa: E402

E.TESTSETS = os.path.join(HERE, 'testsets')
E.EVAL = os.path.join(HERE, 'eval')

if __name__ == '__main__':
    os.environ.setdefault('AB_WORKER', 'midtrain')
    sys.exit(E.main())
