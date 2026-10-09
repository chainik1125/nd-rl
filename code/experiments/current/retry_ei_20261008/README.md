---
written_on: 2026-10-08
written_by: agent:claude
---

# Retry EI: multi-attempt episodes with Lean error tokens

Summary and results:
[`experiment-summaries/2026-10-08-retry-ei`](../../../../experiment-summaries/2026-10-08-retry-ei/README.md).

| file | what it is |
|---|---|
| `harness.py` | the loops harness run 590 was scored with, plus `oracle.sample_episodes` and the episode eval |
| `pretrain.py` | run 590's file plus the arms, chosen with `ND_ARM` (see its docstring) |
| `episodes.py` | the multi-attempt sampler: chunked cached prefill for long contexts, the gate, error tokens |
| `errors.py` | error vocabulary, the classifier, and `keep_prefix` for the resume arms |
| `lean_prefilter.py` | vendored from `dan-pandori/nd-takehome` `903bade4`; local changes marked `retry_ei` |
| `launch.py`, `remote.sh` | one RunPod A40 per (arm, seed): rent, run, pull, terminate |
| `diversity.py` | repeat rates and distinct-proof counts, in episode against independent samples |
| `analyze.py` | reads every `summary.json` under `artifacts/retry_ei_20261008/` |

    python launch.py go retry 0        # needs RUNPOD_API_KEY
    python analyze.py

A CPU plumbing check needs a checkout of the fork at `51604b3` and Lean 4.34:

    ND_FORK=<fork> ND_ARM=retry python harness.py --out /tmp/smoke --smoke

Tests: `code/tests/test_retry_ei_errors.py`.
