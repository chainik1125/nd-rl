import os, sys, json
sys.path.insert(0, "/private/tmp/claude-501/-Users-robbie-repos-nd-rl/a1d02fb4-ba43-48c4-838b-e227293b6c2a/scratchpad/probe"); sys.path.insert(0, ".")
import ndmodel as M, torch
torch.set_num_threads(3)
from em_lp import em_proof, em, parse_prompt  # noqa
