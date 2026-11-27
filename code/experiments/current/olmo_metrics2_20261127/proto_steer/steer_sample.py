"""DRAFT, NOT RUN. Wrapper around Dan's unchanged sampler: install the patch, then run olmo_ifladder/sample.py's main()."""

import os
import sys

sys.path.insert(0, os.environ.get("ND_SAMPLER_DIR", "/workspace/nd-takehome/olmo_ifladder"))
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
if os.environ.get("STEER_LAYER"):
    import steer_vllm_patch

    steer_vllm_patch.install()
import sample  # Dan's file, unchanged; imports vllm inside main()

sys.exit(sample.main())
