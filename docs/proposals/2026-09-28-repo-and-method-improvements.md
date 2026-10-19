---
written_on: 2026-09-28
written_by: agent:claude
status: proposed — analysis for Dan; nothing started
---

# Proposal 15: where the repository's time, money and statistical power go, and what to change

Dan's question (2026-09-28), after a single experiment was estimated at 12–18 hours: is there
low-hanging fruit in the setup overall, judged against practice in the literature and industry?
Yes — mostly in three places: the GPU is almost idle during pretraining, RL rounds spend a third of
their time having Lean reject samples, and most comparisons are run at a seed count the project's
own noise floor says cannot resolve them. Every number below was measured or read from the logs on
2026-09-28; the model is the 3.2 M from-scratch `lean_seq` GPT unless stated.

## Ranked

| # | change | why (measured) | expected gain | effort |
|---|---|---|---|---|
| 1 | Make Stage-1 training use the GPU | **0.78 % of an A40's bf16 peak**; ≈ 61k useful tokens/s; ≈ 1.9× padding waste | ≈ 10× faster pretraining (and many seeds per GPU) | 1–2 days |
| 2 | Default to many seeds, planned against the noise floor | most two-seed differences are inside `NOISE_FLOOR.md` (18 of 20 findings) | findings that survive review | policy + (1) |
| 3 | Take Lean checking off the critical path | ≈ 30 % of an EI round; 3 workers; ≈ 85 % of checked samples are rejects; 36 % of Lean time is process start-up | RL rounds ≈ 3–5× faster with (4) | 2–3 days |
| 4 | Run the sampler at the batch the policy now asks for | rounds sampled at batch 512 ran ≈ 1.1× of an available ≈ 2.2× | ≈ 2× on sampling | done in policy; scripts' defaults pending |
| 5 | One results registry; checkpoints straight to the bucket | EI checkpoints lost or never uploaded in three runs; numbers live in prose; every review recounts from scratch | fewer lost artefacts, cheaper reviews | 1 day |
| 6 | Repository hygiene: artefacts out of git, a package, configs, CI | 4.1 of 4.4 GB tracked is artefacts; 106 top-level scripts; 1 unit-test file; no CI | 250 MB checkouts; smaller agent contexts; bugs caught before a run | 2–3 days, incremental |
| 7 | Per-job compute that releases itself | idle pods cost ≈ $180 before the guards (≈ $107 to 09-21, ≈ $76 on 09-22) | removes a class of incident | 1–2 days |
| 8 | RL-method hygiene from the literature | EI / GRPO details that published work fixed | better baselines | after `state-env` |

With (1), (3) and (4) together, the `state-env`-sized experiment that was estimated at 12–18 hours of
wall-clock would be dominated by the agent's engineering and the review, not by compute.

## 1. Stage-1 training runs at under 1 % of the GPU

**Measured.** A 6,000-step Stage-1 run takes 1,288 s on an A40 (≈ 215 ms per step at batch 128).
Sequences average 102 tokens (p99 198), so the run processes ≈ 61k useful tokens/s — ≈ 1.2 TFLOP/s of
model compute against an A40's ≈ 150 TFLOP/s bf16 dense peak, **0.78 %**. `train.py` builds every
batch in Python one sequence at a time (`torch.tensor` per row, then `.to(dev)`), pads each batch to
its longest row (≈ 1.9× compute per useful token at p99 198 vs mean 102), uses batch 128 and no
`torch.compile`. The `efficiency` run found the same model's sampler kernel-launch-bound; training is
the same shape of problem. The whole run is ≈ 78 M tokens.

**Practice.** nanoGPT (already cloned beside this repo) and its speed-run descendants: pre-tokenised
data resident on the GPU, packed sequences (with document masking where it matters), a larger batch
with the learning rate scaled, bf16, fused AdamW, `torch.compile`. For a model this small the other
standard move is to train **several seeds in one process or on one GPU** — the GPU is idle either way.

**Proposal.** Pre-tokenise once to a tensor file; sample batches on the GPU; bucket by length or pack;
compile the model; sweep batch size with the learning rate against the current recipe's held-out
curve (per-length validation, now instrumented) so the result is the same model, faster; then run N
seeds concurrently per GPU. Target: a Stage-1 model in 1–3 minutes. The acceptance test is equivalence,
not speed: per-length validation loss and held-out accuracy within the noise floor of the current
recipe at equal tokens.

## 2. Seeds: plan every comparison against the noise floor

**Measured.** `NOISE_FLOOR.md`: at n = 2 the smallest resolvable difference is ±235 % on frozen-ladder
solves and ±82 pp on the 6-line held-out bin; the depth-3 slice is not resolvable at all.
`stage1-dynamics` showed the depth-3 "mode" is an oscillation within a run. Of twenty standing findings,
two survive the floor. Proposal 10 spent ≈ $134 on ten two-seed arms, most of which cannot be told
apart.

**Practice.** Agarwal et al., "Deep reinforcement learning at the edge of the statistical precipice"
(NeurIPS 2021): report interquartile means with stratified-bootstrap confidence intervals and
performance profiles, not means of a few runs; Henderson et al., "Deep reinforcement learning that
matters" (AAAI 2018) on seed sensitivity. Paired designs (the same seeds in every arm) and power
analysis before spending.

**Proposal.** Each pre-registration states the minimum detectable difference for its headline quantity
at its seed count, from `NOISE_FLOOR.md`, and the run is not launched if the predicted effect is inside
it. Default to 5–10 Stage-1 seeds once (1) makes them cheap; pair seeds across arms; report IQM and
bootstrap intervals. If `ckpt-avg` finds that checkpoint averaging collapses the oscillation, adopt it
first — it is the cheapest variance reduction available.

## 3. Lean checking: most of it is spent rejecting

**Measured** (`la_T1_dsc_a1_s1.log`, `noise-floor`, round 1 of a T1 ladder, 1,932 s): gating the
143,840 target samples took 353 s wall (1,055 CPU-s on **3 workers**) and the 73,120 transfer samples
206 s; round 2, 512 s + 233 s. Of 92,074 distinct texts checked in round 1, **78,151 were rejected**.
The `efficiency` run measured ≈ 2.0 s of fixed start-up per `lean` process — 36 % of gate process
time — and found a persistent server worth ≈ 5 s of a 49 s round at that workload.

**Practice.** LeanDojo (Yang et al., 2023), Pantograph and the community `repl`: a long-lived Lean
process serving many checks. AlphaProof validates each tactic in its environment and discards invalid
ones before they cost anything downstream.

**Proposal.** (a) A fast, sound **reject-only** pre-filter in Python for the `lean_seq` fragment —
parse, scope, declared-type bookkeeping (the `state-env` renderer does most of this already) — that
never rejects a text Lean would accept (measured on a large sample; any disagreement is a bug), so
Lean only sees plausible candidates. (b) Lean on more cores: a CPU pod or the VPS for the gate while
the GPU samples the next batch, instead of 3 workers on the GPU pod. (c) A persistent Lean server
once the step-level environment needs per-step checks. Expected: the gate from ≈ 30 % of a round to a
few percent.

## 4. Sampler batch

Rounds in every post-`efficiency` ladder ran at batch 512 for comparability; the policy now asks for the
largest batch that fits (2026-09-28). Remaining: raise the scripts' defaults once `lean-judge`'s edits
have merged, so a brief that forgets `--batch` gets the fast path.

## 5. One results registry, checkpoints to the bucket on write

**Measured.** Checkpoints lost or never uploaded: the 2026-09-22 cleanup (`ds-composition`,
`ds-generator` session 1); no Lean-format EI checkpoint was ever uploaded before `support-curves`, which
had to retrain two; `support-curves`' seed-1 EI checkpoint. Numbers live in `numbers.md` prose across
27 branches; reviewers re-derive everything, and the librarian re-reads it all.

**Practice.** Experiment tracking (Weights & Biases, MLflow, Aim) or at least a structured registry:
every run writes config, git SHA, data hash, checkpoint md5 and bucket path, and metrics, to one table.
Sculley et al., "Hidden technical debt in machine learning systems" (NeurIPS 2015) on configuration and
lineage debt.

**Proposal.** A `record.py` every script calls: appends a row (run id, arm, seed, git SHA, config,
dataset hash, checkpoint md5 + bucket URI, metric name, value, source file) to a per-run JSONL that is
synced to the bucket and merged into one table. `save_ckpt` uploads before returning. Reviewers then
re-derive a sample and check the registry; librarians query it instead of reading prose.

## 6. Repository hygiene

**Measured.** 4.1 of the 4.4 GB tracked on the fork's `dan` is committed artefacts (why each worktree
was 4 GB until `newrun`); 106 Python files and 50 Markdown files at the top level; one unit-test file
(`tests/test_lean_only_judge.py`); no CI; no configuration files — every setting is a CLI flag spread
over run logs; 27 run branches, each a divergent copy.

**Practice.** A package (`nd_rl/`: data, model, train, sample, judge, env, rl, eval) with thin run
scripts; configs recorded with each run (YAML, or Hydra); large artefacts in the bucket (or Git LFS /
DVC), not git; CI on every push with fast tests — checker agreement on a fixed sample, tokenizer and
state round-trips, environment replay, a 50-step CPU training smoke test. Runs differ by config, not by
branch.

**Proposal.** Incremental, not a rewrite: stop committing `artifacts/` (bucket only, registry points at
it); move library code into a package as each file is next touched; add CI with the four tests above.
Side benefit: agents read less to orient, which is the main driver of Claude usage (context size per
turn).

## 7. Compute that releases itself

**Measured.** Idle pods after an agent died: ≈ $107 to 2026-09-21, then ≈ $76 on 2026-09-22 — the
guards (`pod_budget_watch`, the fixed `pod_watchdog`) now cap it, but the failure mode remains: a pod
outlives the process that wanted it.

**Practice.** Per-job compute with autostop (SkyPilot's autostop, spot instances; RunPod serverless or
jobs that terminate their own pod on exit); a queue that packs small jobs onto a GPU.

**Proposal.** A `podjob` wrapper: create pod → run one job → upload → delete, with the pod's lifetime
tied to the job; several small jobs packed per GPU (the `efficiency` run measured aggregate throughput
peaking at 3 co-tenants).

## 8. RL-method hygiene (after `state-env`)

- **EI**: ReST-EM (Singh et al., 2023) retrains from the base each iteration to avoid drift; Polu et al.,
  "Formal mathematics statement curriculum learning" (2022), keep the shortest proof per theorem and
  schedule by difficulty — our T2 / T4 rungs are close; adopt the shortest-proof rule as default.
- **GRPO**: DAPO (Yu et al., 2025) and Dr. GRPO (Liu et al., 2025) — drop all-correct / all-wrong groups
  rather than wasting them, token-level loss normalisation, no std normalisation of advantages; and an
  anchor (a KL term or AlphaProof's 10 % supervised mix) — our round-2 GRPO lost 0.2–0.6 of held-out
  accuracy without one.
- **Evaluation**: per-theorem success probabilities and pass@k curves with the unbiased estimator (Chen et
  al., 2021) as the default readout, as `support-curves` did, rather than `L*` at 256 attempts alone.
- **Search**: if `state-env` helps, AlphaProof / HTPS (Lample et al., 2022) / DeepSeek-Prover-V1.5 style
  search over the step-level environment (proposal 14, item 1).

## Suggested order

(1) and (3a) first — they cut both wall-clock and cost, and (1) is what makes (2) affordable. (5) and
the "stop committing artefacts" half of (6) next — cheap and they stop recurring losses. (7) and the rest
of (6) incrementally. (8) after `state-env` reports.
