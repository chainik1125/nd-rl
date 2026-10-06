# Agent brief: overnight autonomous attempt at the ND take-home

You are running unattended overnight. Nobody will answer questions. Make the judgment calls
yourself, write them down, and keep going. Read `README.md` and `spec.md` in this repo first;
they are the assignment and they win over anything here if the two conflict.

## Goal

Maximise the robust proof-length frontier L − P: how many lines beyond the cap-6 training
data an RL-trained model can prove theorems at, with L defined as the README defines it (the
longest length at which the model wrote **≥ 5 distinct verifier-accepted proofs**), and with
the controls the README demands. A well-controlled honest number beats an inflated one.

## Where things run

- **This machine (VPS, no GPU, 2 vCPU / 2 GB).** You are here. The git working copy is
  `~/nd-takehome`, a fork at `github.com/dan-pandori/nd-takehome`. Write and edit code here.
  Do not train here.
- **GPU pod (1× A40 48 GB, 96 CPU cores, 500 GB RAM, PyTorch 2.8 + CUDA already installed in
  the system `python3`).** Use it through three helpers on `PATH`:
  - `podsync` — push `~/nd-takehome` to the pod at `/workspace/nd-takehome` (excludes `.git`,
    `data/`, `artifacts/`, `ckpts/`). Run it after every code change, before `podrun`.
  - `podrun "<command>"` — run a shell command on the pod inside `/workspace/nd-takehome`.
    Long jobs: run them under `nohup ... > artifacts/<name>.log 2>&1 &` and poll the log,
    so a dropped SSH connection does not kill training.
  - `podpull <relpath>` — copy `data/`, `artifacts/`, `ckpts/...` etc. back here.
- Install any extra Python packages on the pod with `podrun "pip install <pkg>"`; keep the
  list in `requirements.txt`. `numpy`, `matplotlib`, `tqdm` are fine to add.
- **Pull results back often.** The pod is deleted by a kill switch at **16:00 UTC** no matter
  what. Anything only on the pod at that moment is lost. `podpull` `data/`, `ckpts/` and
  `artifacts/` after every stage, and commit + push small files (jsonl, md, png) to the fork.
- **Commit and push** to the fork (`git push origin main`) at every milestone. Push
  checkpoints only if under 50 MB each; otherwise leave them in `ckpts/` on this machine.

## Time budget

- Now: about 03:30 UTC. Hard stop for you: **15:15 UTC** — by then everything must be pulled
  back, `writeup.md`/`numbers.md`/`log.md` written, and pushed. Plan to have a complete,
  submittable result by **10:00 UTC** and use the rest to improve it.
- Suggested split: Stage 1 done and evaluated by 06:00 UTC; first expert-iteration rounds
  with controls by 09:00 UTC; then either more rounds, a second seed of the main comparison,
  or GRPO — whichever the evidence says is the best use of time.
- Check the clock (`date -u`) at every milestone. If you are behind, cut scope, not controls.
- Compute is cheap here (a 3.3M-param model trains in minutes on the A40); your own reasoning
  time is the expensive resource. Prefer running an experiment over deliberating about it.

## Hard rules (the assignment's, restated; violating them makes the result worthless)

1. Every supervised training proof has verifier length ≤ 6. Store `n_lines` from
   `verify_text` on every record and assert it before training.
2. The model is a decoder-only transformer trained **from scratch**. Reference size: 4 layers,
   d=256, about 3.3M params. Report the exact parameter count.
3. Training data comes only from a **procedural random generator** that you write, plus (in
   Stage 2) proofs the model itself wrote that the verifier accepted. **Never write proofs by
   hand or from your own reasoning as training data**, and never use the example file,
   reference proofs, or anything from the internet as training data.
4. `nd_verify` is the only judge; never modify it.
5. **Never train on `targets/validation_36*.jsonl` or renamings of it.** Inspecting it,
   evaluating on it, and studying failures on it are all encouraged.
6. **The test set (`targets/test_short_prompts.jsonl`, `targets/test_long_prompts.jsonl`) may
   be run exactly once**, with your final model, near the end. Do not open the files, do not
   look at per-theorem results, do not run it twice. Before running it, create
   `artifacts/TEST_RUN_DONE` and record the time in `log.md`; if that file already exists, the
   test set has been used and you must not run it again. Report the two `score_test.py` lines
   verbatim, for the Stage-1 model too if you run both in that single session (that is
   allowed: one run of `prove.py` per checkpoint per file, all in the same sitting, no tuning
   after).
7. `prove.py` must match `submission_template/prove.py` exactly in interface, and must not call
   the verifier or retry.
8. Train/held-out/RL-target/transfer splits are disjoint **by theorem** (canonical sequent
   string), and you report the atom-renaming overlap estimate.

## Methodology (defaults; you own the details and may deviate with a written reason)

**Stage 1.** Generator that samples valid proofs of length 2–6 by forward application of the
rules with random subproof structure (so `IMPI`, `NEGI`, `ORE` boxes appear), verifies every
emitted proof, deduplicates by theorem, and records rule/length/premise histograms. Aim for
≥100k proofs with a reasonably flat length distribution and few trivial theorems; measure
and report the trivial fraction. Tokenise one symbol per token; consider **relative line
references** (cite "k lines back" rather than absolute `N<i>`) — the project's earlier work
found absolute indices generalise badly to longer proofs. Train, then report held-out greedy
solve rate by length 2..6 with Wilson CIs, and failure reasons.

**Stage 2 (main).** Expert iteration with a length curriculum:
- Build an RL-target pool and a disjoint transfer pool of theorems whose *generating* proofs
  are 7–16 lines (use the same generator with a higher cap; store the generating length as an
  upper bound only).
- Each round: sample k proofs per RL target at temperature ~0.8–1.0 (k of 16–64; large k
  matters because success is rare at the frontier), keep verifier-accepted proofs of the
  prompted sequent, dedupe, fine-tune on them mixed with a retained slice of Stage-1 data (so
  in-distribution performance does not collapse), and evaluate.
- **Every round report:** RL-target solve rate, transfer solve rate by generating length,
  found-proof-length histogram, the robust frontier (≥5 distinct verified proofs), and
  Stage-1 held-out greedy rate.
- **Frozen-model control:** the Stage-1 model with the same total number of attempts, no
  retraining. Report it on the same plots. This is what makes the RL number meaningful.
- Watch for reward hacking that the verifier allows but the spirit forbids: proofs padded
  with reiterations, trivially long proofs of easy theorems. Report written length *and* a
  dependency-pruned length if you can compute it cheaply.

**Stage 2b (if time).** One of: a second seed of the main comparison (preferred if the main
effect is within ~3pp of the control), or GRPO/leave-one-out policy gradient with binary
reward, fixed loss divisor (no length normalisation), no KL. Keep it a clearly separate arm.

**Stage 3.** The README's table (rows = models, columns = Stage-1 held-out, transfer by
length, validation by bin, test short/long), the per-length transfer curve with the frozen
control, and the found-length histogram across rounds. Look closely at validation failures:
which theorems, what failure reasons, whether `explosion` and the `>6` bin move.

Known hard part from the project's earlier work: RL gets almost no learning signal when the
model never succeeds on a target (all-fail groups). Curriculum and large k exist to fix that.
If a round yields zero new proofs, change the curriculum or k; do not keep running it.

## Deliverables (in this repo, committed and pushed)

- `README.md` (append a "Reproduction" section: commands, seeds, hardware, wall-clock).
- `writeup.md` — leave the executive summary as a clearly marked draft the human will
  rewrite; fill in every section with method details, tables, figures (PNG in `figures/`),
  limitations, and "what next". Positive voice, numbers with n and CIs.
- `numbers.md` — every number in the writeup with the file it came from.
- `log.md` — dated log of what you tried, in order, including dead ends and judgment calls.
- `data/` — train, held-out, RL-target, transfer pools as jsonl with `thm`, `prompt`,
  `text`, `n_lines`.
- `prove.py`, `requirements.txt`, and the Stage-1 and final checkpoints (in `ckpts/`).
- `STATUS.md` — **update at every milestone** (what is done, what is running on the pod,
  next step, current time). If your session is interrupted and resumed, read this first.

## Working style

- Keep code small and in a few files (`gen.py`, `tokenizer.py`, `model.py`, `train.py`,
  `sample.py`, `expert_iter.py`, `prove.py`, `eval_*.py`). No frameworks beyond PyTorch.
- Unit-test the generator against the verifier and the tokenizer round-trip before training.
- When a run finishes, look at actual outputs (10–20 proofs), not only the aggregate.
- Never delete the pod, never touch `~/nd-rl`, never run anything on the test set except as
  in rule 6. Do not spend on anything beyond this pod.
- When done, write the final `STATUS.md` line `DONE <UTC time>` and stop.
