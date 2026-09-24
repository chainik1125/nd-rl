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

**Say which model every number applies to.** Each result table and each `numbers.md` section
names the checkpoint(s) behind it: model id/path, parameter count, format (`lean_seq` / token),
from-scratch or pretrained, and the data it was trained on. A number measured on one model is
never stated as a property of "the model" or of the setting. If you inherit a number from
another run, carry its model label with it. (Dan, 2026-09-23, after an efficiency run's premise
turned out to have been measured on a different model than the one it was applied to.)

**Keep the working context small.** Claude usage is billed each turn against the whole
conversation so far, so a session costs roughly turns × context size. Measured over
2026-09-15..23: executor sessions ran 100–290 turns carrying 120–310k tokens per turn, and the
fifteen largest sessions accounted for ~$400 of the ~$650 total — context size is the lever, not
session count. In practice: read the part of a file you need (a `grep`, a line range) rather than
the whole file; send long output to a file and read back a summary, not the dump; count in a
script whose stdout is a table rather than reading raw `.jsonl` into the conversation; don't
re-read a file you already read this session. A standard with a reason, not a rule: a genuinely
necessary full read is fine — say so when you do it.

**Lean is the checker of record** (decision 2026-09-20): every proof you count must translate with `nd2lean.py` and be accepted by Lean; report the agreement count with `nd_verify` and treat any disagreement as a bug. Show proofs in write-ups in Lean. Experiments that involve a pretrained model use Lean as the surface form. From-scratch training and RL use the Lean `lean_seq` format (decision 2026-09-22); reward is `lean_check` once proposal 9's phase 1 validates it (until then Lean ∧ `nd_verify`); report proof length in lines **and** term size; label any token-format arm as such.

At the end: `run<id>.md` (≤ 400 words + figures, expectations vs outcomes stated plainly);
`STATUS.md` line `RUN<id> DONE <UTC>` (or `DONE <UTC>` for a single-run session); push; then
`touch ~/runs/<run-id>/executor.done`. Keep every count reproducible from pulled files: a
separate reviewer session will re-derive them and will not read your write-up first.
