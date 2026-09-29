#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Uploaded to the pod as harness.py for every factorial cell: the frozen autoresearch harness
(uploaded as harness_frozen.py, byte-identical) with two switches from the staged remote.sh

  COMBO_TOK  abs | lean_seq   the harness's tokenizer check follows the cell (abs samples are
                              judged by nd_verify alone, as in the token era; lean_seq by Lean AND
                              nd_verify)
  COMBO_RL   ei | leon        ei: the frozen 4-round expert iteration; leon: leon_rl.stage_rl with
                              leon_cfg.json = c005's recipe (Leon's EI + guided sampling, 1,800 s
                              deadline)

and, after the frozen dev eval, the pass@k scoring of passk.py on the same in-memory model.
eval.pipeline_secs is the harness clock at the end of the dev eval (the 2,400 s budget's
reading); total_secs also includes the pass@k sampling.
"""

import os
import time

import harness_frozen as H
import passk

H.TOK_MODE = os.environ["COMBO_TOK"]
if os.environ["COMBO_RL"] == "leon":
    import leon_rl

    H.stage_rl = leon_rl.stage_rl

_stage_eval = H.stage_eval


def stage_eval(a, tok, model, dev_pool, heldout_recs):
    out = _stage_eval(a, tok, model, dev_pool, heldout_recs)
    out["pipeline_secs"] = time.time() - a.T0
    out["passk"] = passk.score(a, tok, model)
    return out


H.stage_eval = stage_eval

if __name__ == "__main__":
    H.main()
