"""Make the ws3 modules importable in tests.

ws3/ndpath.py adds fork_pin and the vendored verifier to sys.path.
"""

import os
import sys

WS3 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ws3")
if WS3 not in sys.path:
    sys.path.insert(0, WS3)
import ndpath  # noqa: E402,F401
# isort: split
