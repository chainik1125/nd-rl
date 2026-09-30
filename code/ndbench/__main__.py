"""`python -m code.ndbench ...` entry point (run from the repository root, per plan §A1)."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
