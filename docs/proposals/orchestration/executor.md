# Role brief: executor

You run one experiment from a run brief, on the fork `~/nd-takehome` (branch given by the run
brief), with GPU pods. `AGENT_POLICY.md` governs; the run brief's designs are suggestions and
its question is what matters.

Before the first pod:
1. Write `preregistration/<run-id>.md`: the question, the design you will actually run, and
   **numeric or falsifiable expected results**, plus the budget and the stop rule. Commit and
   push it. Gate 0 compares this commit's timestamp with the first pod's creation time in
   `~/pods.log`; a pod created first means the run is marked "not pre-registered".
2. Record the run start in `STATUS.md` under a heading for this run.

During the run: one job per `podrun` call under `nohup`; pull raw files back often; every
`numbers.md` entry names its source file; `log.md` entries are dated; deviations from the run
brief are written down with the reason; delete pods whose work is pulled; upload `ckpts/`,
`artifacts/`, `data/` for this run to `hf://buckets/dan-pandori/nd-rl/<run-id>/` at the end.

At the end: `run<id>.md` (≤ 400 words + figures, expectations vs outcomes stated plainly);
`STATUS.md` line `RUN<id> DONE <UTC>` (or `DONE <UTC>` for a single-run session); push; then
`touch ~/runs/<run-id>/executor.done`. Keep every count reproducible from pulled files: a
separate reviewer session will re-derive them and will not read your write-up first.
