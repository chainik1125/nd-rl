---
written_on: 2026-10-06
written_by: agent:codex
---

# Historical CPU scheduling wrapper review

Accepted source SHA256 `d59b1427996e7315ec5c406f8faa99a07f86ccdfba6b75daed26c2cfb2d393a0` for `analysis/run_historical_cpu.py`. Read-only inspection covered the wrapper and immutable six-command plan; no historical outputs or MAIN live quality outcomes were opened, and no scientific commands were executed.

The initial phase preserves the prepared first two commands; the remaining phase preserves the final four. Each launch requires all nine MAIN records complete and bound to the original MAIN registry, with nonempty phase records and every phase returncode0. Registry/plan hashes, current scientific entrypoint bytes, and absent output targets are checked both in phase preflight and immediately before each child. Dangling output symlinks are rejected. Prediction/config inputs for the remaining phase must exist before execution. Child environments disable GPU access; logs, elapsed child timings, exact argv, completion codes, launcher source and barrier hashes are retained. Scientific argv and scientific sources are unchanged.

This successor closes the previously reported phase-long source/output validation window. Child scientific guards remain responsible for complete input/policy signatures, while this scheduling wrapper binds the exact prepared plan. Root release is an external authorization, recorded by the required tool-time argument; the string alone is not independent proof of authorization. No mocked execution was necessary for this bounded source check, and no numerical evidence is claimed.
