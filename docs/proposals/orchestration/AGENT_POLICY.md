# Standing policy for agent runs (from Dan; canonical copy, updated 2026-09-27)

This file lives in `nd-rl` at `docs/proposals/orchestration/AGENT_POLICY.md` and `runrole` reads it
from there for every role, so there is one copy. Copies of `AGENT_POLICY.md` inside fork branches
are historical; where they disagree with this one, this one wins. Every brief is read together with
this file, and where a brief and this file disagree, this file wins.

## A firm default: no outside contact

**Do not contact or involve other humans.** No PR reviewers, no messages, nothing sent to
Dmitry or anyone else. Draft PRs with no reviewers are fine. Dan reads and shares things himself, and agents
notify only Dan. This is a firm default, not an absolute (Dan, 2026-10-13): in a genuinely extreme situation, use
your judgment, and tell Dan at once what you did and why.

## Standards with reasons (keep them; say when and why you deviate)

- **Lean alone decides (Dan, 2026-09-27).** A proof counts iff Lean 4 core accepts it: for
  `lean_seq` models, the literal sampled text parses in the strict grammar and Lean accepts it;
  for free-form Lean, `lean_check` accepts it (elaborated-term allowlist, including `Not.elim`).
  **Do not use `nd_verify` to judge anything** — not reward, not evaluation, not counting. It stays
  in the repository unmodified only so old numbers can be reproduced. Proofs in write-ups are shown
  in Lean. *Why:* Lean is sound, readable, and what a pretrained model already knows; the proofs
  `nd_verify` rejects and Lean accepts are valid proofs that skip a step ND's rule format demands
  (e.g. `n.elim` on `n : ¬A`). Numbers from before 2026-09-27 were measured under Lean ∧ `nd_verify`
  — label which checker a number is under whenever you compare across that date. Decision note:
  `docs/project_strategy/2026-09-20-lean-default.md`.
- **Report proof length in term size as well as lines.** Under Lean a proof may be shorter than the
  shortest ND proof and may omit premise re-statements, so line counts and the ladder's ND-derived
  `L_true` labels (upper bounds under Lean) are no longer the whole story. Term size is invariant
  to both.
- **Training format: Lean `lean_seq`** (Dan, 2026-09-22). Token-format arms are labelled as such.
- **Sampling settings (Dan, 2026-09-28): use the fast settings by default.** The fast decode path is
  already `sample.py`'s default (`ND_SAMPLE_PATH=fast`, finished rows dropped, per-row seeding) — keep
  it. Then:
  - **Batch:** the largest that fits. On the 3.2 M `lean_seq` model, batch 4,096 at `max_new` 288
    peaks at ≈ 11 GB (run `efficiency`), so at most two such jobs per 24 GB card. Record peak memory
    (`torch.cuda.max_memory_allocated`) in your first job and report it with the batch.
  - **`max_new`:** set it from the pool — the longest proof you intend to count, in tokens, with a
    margin — and report the fraction of samples that hit it. If more than ≈ 0.1 % of any stratum you
    report are cut off, raise it: a length cap that truncates silently biases exactly the length
    results this project is about.
  - **Comparisons:** hold batch, `max_new` and decode flags fixed across every arm of one experiment.
    Comparing with an on-file number measured at other settings is fine: a settings change is a
    sampling re-draw (bf16 reduction order changes; ≈ 1 row in 128 decodes differently), never a
    correctness change — Lean still checks every sample. Say so, and quote `NOISE_FLOOR.md`. The one
    exception is a test that needs a byte-identical sample stream against an old one: then run
    `ND_SAMPLE_COMPACT=0` at the old batch, as `ds-generator` did.
  - *Why:* ≈ 2× on RL sampling (2.16× end to end, 2.35× on the sampler, measured on one RTX 3090),
    nearly all of it from the batch. Until 2026-09-28 runs held batch 512 to match earlier results
    and got only ≈ 1.1×. Budget estimates made before then assume the old throughput; re-measure.
- **Guided read-out by default (Dan, 2026-10-04).** Every evaluation of a proof-state model reports **two** numbers:
  - **plain:** independent attempts, comparable with every earlier result;
  - **guided:** `guided_sample.py` / `guided_eval.py` on the fork's `dan`. Each step is checked by `step_check.py`,
    which agrees with Lean (0 false rejects on 314,541 steps against Lean). A failing step is redrawn from the same
    state without replacement, up to 10 rejections per attempt.

  Label which number is which, and compare across runs only plain with plain or guided with guided. Guided is the
  best-system number. RL rollouts stay plain unless a brief says otherwise. *Why:* in `guided-tts`, guided beat
  plain by 5–9 pp at matched sampled tokens on textbook problems in 6 / 6 models, and the gain came entirely from
  the logical check.
- **Every number names the model it was measured on** (Dan, 2026-09-23): checkpoint, parameter
  count, format, from-scratch or pretrained, training set. An inherited number carries its label;
  a reviewer treats an unlabelled number as a finding. *Why:* an efficiency run applied a premise
  measured on a different model than the one it was about.
- **Record the compute behind every arm** (Dan, 2026-09-29). Per arm, and per round for RL, record:
  - GPU-seconds and the GPU type;
  - generated tokens (and attempts or actions);
  - training steps and training tokens;
  - Lean / environment checks.

  Write them as results-registry rows (`record.py`; metric names `gpu_seconds`, `gen_tokens`,
  `train_steps`, `train_tokens`, `lean_checks`, with `arm`, `seed` and `labels.round`), and put a
  per-arm table in the write-up. Until the scripts record these themselves, derive them from job logs
  and say how. The pre-registration states what the compared arms are matched on (attempts, actions,
  tokens, steps). The write-up flags any arm that used more than 1.25× its comparator's compute on any
  of these. Matching is a design choice for each experiment, not a rule. *Why:* a more elaborate method
  can win by spending more (Dan's question about search as the EI expert). Recorded compute lets
  anyone check that afterwards, or rerun at matched compute, instead of guessing.
- **Plan seeds against the measured noise floor** (Dan, 2026-09-28; `NOISE_FLOOR.md` on the fork's `dan`
  branch). Every pre-registration states the minimum detectable difference for its headline quantity at
  its seed count, and a comparison whose predicted effect is inside it is redesigned (more seeds, a
  within-run or paired design) before it is launched. Use the same seeds in every arm. Report per-seed
  values plus the interquartile mean with a stratified-bootstrap 95 % interval (Agarwal et al., NeurIPS
  2021). Once fast Stage-1 training is merged (`FAST_STAGE1.md`), default to 5–10 Stage-1 seeds. *Why:*
  at n = 2, 18 of 20 standing findings sat inside the floor. A difference inside it is not a finding.
- Learned from real bugs, each of which produced a wrong number once: start-index-normalised
  distinct-proof counts; a frozen control at equal attempts; a base-model reachability number
  with every "RL solved X"; two seeds before calling a difference; expected results written
  down (and committed) before a run; splits disjoint by renaming class; every count
  reproducible from files that are pulled back.
- The take-home's cap 6 is the take-home's rule, not the SPAR project's: arms above it are allowed
  and are labelled "cap N" in every row. No hand-written or LLM-written training proofs. At most
  one test-file run per take-home submission.

## Everything else is a suggestion

Experiment designs, pod counts, GPU classes, deadlines, deliverable formats. Use judgment. If
a different or bigger experiment answers the project's question better, propose it or do it.
The question is the SPAR project's: to what extent does RL against a verifier produce
genuinely new capability rather than eliciting rare-but-known behaviour, and what limits it.

## Dan is reachable

He is not watching live but reads `STATUS.md` and answers questions. Put questions in
`QUESTIONS.md` at the repo root — dated, with the default you will follow if unanswered —
commit, and proceed on the default. Direction, scope, spend, and "what would you find most
useful" are all fair questions.

## Spend and compute

- **Pod budget:** the brief's declared budget, registered with `podbudget <run> --set <hours> <usd>`
  **before the first pod** (`pod_budget_watch` warns at 80 % and deletes the run's pods at 100 %).
  $50 if a brief declares none. Extend within the declared budget with
  `podbudget <run> --extend <hours> "reason"`; beyond it, ask in `QUESTIONS.md`.
- **Balance floor: keep the RunPod balance above $35** (Dan, 2026-10-06; was $100 from 2026-09-27, $130 from
  2026-09-24). Check it with `rpbalance`. `balance_floor` deletes every pod on the host if the balance falls below $25.
  If planned work would take the balance below $35, ask in `QUESTIONS.md`.
- No limit on concurrent pods; no fixed GPU class. Record the GPU class **and** its real billed
  rate — `podbudget`'s dollar column falls back to $0.50/h for classes it does not know. Delete
  pods when their work is pulled. Upload to the bucket as you go, not only at the end.
- **Claude usage:** roles share Dan's Max 20x subscription; there will be no API keys (Dan,
  2026-09-23, cost). Keep the working context small (see the executor brief). If a usage limit is
  slowing progress, write `LIMIT_SLOWING <UTC> <what it cost you>` in `STATUS.md`.
- **Disk.** Run worktrees are made by `newrun` as sparse checkouts: earlier runs' `artifacts/<dir>/`
  trees are not checked out (≈ 4 GB saved per run). Read an old artifact with
  `git show HEAD:artifacts/<dir>/<file>`, or check a directory out with
  `git sparse-checkout add /artifacts/<dir>/`. Write only under your run's own `artifacts/<dir>/`.
  Never edit another run's files or shared `data/` in place — some are hard links shared across
  worktrees and read-only; write a new file instead, and never `chmod` them writable. Local disk is
  scratch: upload as you go, and check `df -h` and `~/ALERTS.md` between stages (at 85 % use, `disk_watch`
  removes finished, fully uploaded worktrees and writes there if that is not enough).
- VPS: 4 GB / 2 vCPU, shared by up to two runs. **Never run more than 4 processes at once on the VPS**
  (use `xargs -P 4`, not one `&` per file). On 2026-09-29 one run started 160 parallel `hf` downloads,
  exhausted memory, and killed both runs' sessions on the host for 8 hours. Heavy or parallel work
  belongs on a pod. New VPSs can be created with `newvps <name>` (needs Dan's Claude and gh
  logins afterwards).

## Target branches

- `nd-rl`: Dan's integration branch is **`dan`**. Librarians branch from `origin/dan` and merge back
  into `dan` after review; one standing draft PR `dan` → `main` exists. Never merge into `main`.
- Fork (`dan-pandori/nd-takehome`): the integration branch is **`dan`** (from 2026-09-27; it carries
  the Lean-only judge). New run worktrees branch from `origin/dan` unless the brief names another
  base.

## Artifacts

At `DONE`, upload the run's `ckpts/`, `artifacts/` and `data/` to the public HF bucket:
`hf buckets sync <dir> hf://buckets/dan-pandori/nd-rl/<run-name>/<dir>` (the `hf` CLI is
logged in on the VPS; `podrun` passes `HF_TOKEN` to pods). Record the bucket paths in
`numbers.md`. The bucket is **public**: never upload `~/.config`, shell logs, or anything that
could contain a token, and never print tokens.
