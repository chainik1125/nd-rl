"""Put the pinned fork code and Dmitry's strict verifier on sys.path.

Import this first in any ws3 script or test:

    import ndpath  # noqa: F401

It adds `code/ws3/fork_pin` (model.py, tokenizer.py, nd_verify) and
`code/experiments/current/r1_cot/vendor` (strict_verify.py, nd_metrics.py). Nothing else.
"""

import os
import sys

WS3 = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(WS3)
FORK_PIN = os.path.join(WS3, "fork_pin")
VENDOR = os.path.join(CODE, "experiments", "current", "r1_cot", "vendor")

for p in (FORK_PIN, VENDOR, WS3):
    if p not in sys.path:
        sys.path.insert(0, p)

FORK_DATA = os.path.join(FORK_PIN, "data")
FORK_CKPT = os.path.join(FORK_PIN, "ckpt", "pretrain.pt")
TEXTBOOK72 = os.path.join(WS3, "textbook72")
