---
written_on: 2026-09-20
written_by: "agent:claude"
reviewed_by_human: false
---

# Pretrain workstream: what is frozen, what is constrained, what is open

> **NOT REVIEWED BY ROBBIE.** This is an agent draft (Claude), written at Robbie's request from
> a chat on 2026-09-20. Robbie has not read, checked, or endorsed any of it. Nothing here is a
> decision. Every number quoted from Dan's runs was read from the reviewed summaries on
> `origin/dan`, not re-derived; none of the fork's code (`minlen.py`, `expert_iter.py`, the pool
> files) was run or inspected by the author. Treat it as a proposal to argue with.

## Summary

Each of the four group members sets up an autoresearch pipeline that optimises one piece of the
setup. Robbie has pretraining. This note proposes the rules of that loop: what is frozen so that
runs are comparable, what bounds model size and compute, and what the agent may change. The
success criterion for the workstream is downstream RL outcome (true proof length past the cap-6
training data), not pretrain loss.

Decisions Robbie has stated (the only human input in this note):

- Parameter cap of about 10M, **and** a wall-clock cap on a named GPU, because a parameter cap
  alone can be gamed (for example by heavily looped transformers).
- The systems layer (kernels, compile, precision, dataloader) is **not** frozen.

Everything else below is the agent's proposal.

## Evidence this plan rests on

Observed results from `origin/dan` (Dan's orchestrated runs; code in the public fork
`dan-pandori/nd-takehome`, artifacts in the public HF bucket `dan-pandori/nd-rl`), and from
Robbie's take-home (`~/repos/nd-takehome`):

1. **A true-length metric and labelled pools exist**
   ([ladder A summary on `origin/dan`](../../experiment-summaries/), proposal
   `2026-09-17-rl-technique-ladder.md` on that branch). `minlen.py` is a bounded prover that
   labels each theorem with its shortest proof length `L_true`. `L*` is the largest L with ≥ 5
   distinct theorems solved at `L_true ≥ L`. Committed pools: transfer 2,285 theorems (never
   trained on), RL-target 4,495. Frozen cap-6 model `L*` = 7; all six expert-iteration variants
   `L*` = 10 on both seeds.
2. **Scale did not help in Dan's runs** (round 3, runs 4a / 4b). Held-out greedy 0.885–0.891 at
   25M / 85M against 0.871–0.893 at 3.2M; 85M missed its quality gate on both schedules; larger
   models ignited less often on the first schedule (confounded with learning rate, per the
   reviewer). Interpretation: a 10M cap gives up nothing on current evidence.
3. **Dan's baseline pretrain is not saturated.** `abs` codec (absolute line indices): held-out
   greedy ≈ 0.88, 6-line bin 0.49–0.59, validation loss 0.082. Robbie's take-home with
   content-addressed citations: held-out 99.5%, loss 0.002. The two used different data and
   evals, so this is suggestive, not a controlled comparison. Interpretation: tokenization is
   plausibly the largest single pretrain lever.
4. **Noise.** On narrow pattern pools, outcomes are bimodal across pretrain seeds, and one
   checkpoint under two sampling seeds gave 54 vs 1 eight-line proofs (round 3, run 2). On the
   broad ladder pool seeds agree much better (612 vs 623 transfer theorems solved), but one
   near-identical config still spanned 328–390 solved at `L_true ≥ 9`, roughly ±10%.
5. **Cost.** One full ladder arm (1.15M samples, 8 rounds) took 55–112 minutes on an RTX 3090.
   Transfer `L*` by round ran 7, 8, 9, 9, 10, 10, 10, 10 for most arms, so a 4-round protocol
   already separates models.
6. **GPU utilisation at this scale** (agent's rough estimate from the take-home log: 45 ms per
   step, batch 1,024, 3.16M params): about 4–5% MFU on an H100. Wall-clock grows much more slowly
   than FLOPs with model size here, so under a time budget alone an optimiser will go straight to
   the parameter cap. The parameter cap is the constraint that binds; the time cap exists to stop
   compute being hidden behind shared weights.

## Proposal

### Frozen (hash-checked at run start)

| Item | Value |
| --- | --- |
| Verifier | `nd_verify` unmodified, plus the strict backward-reference check |
| Pretrain data | Dan's cap-6 Stage-1 file; no proof longer than 6 lines; no newly generated theorems |
| RL targets | Ladder RL-target pool v2 (4,495) |
| Eval | Ladder transfer pool (2,285) with `L_true` labels, split once into dev and holdout |
| RL protocol | Plain expert iteration (ladder rung T1), shortened |
| Seeds | Pretrain seeds {0, 1, 2}, sampling seed fixed; all reported, none dropped | 
| Test files, validation-36 | Untouched |

1 pretrain seed only!! no more!!

- **Eval split.** Split the transfer pool by renaming-class hash into dev (≈ 1,143) and holdout
  (≈ 1,142). The loop sees dev only; holdout is scored only when a config is promoted. Without
  this, a loop running tens of experiments a day overfits the transfer pool.
- **RL protocol.** k = 32 samples per target, T = 0.8, `max_new` 512, 600 fine-tune steps per
  round, 20k retained Stage-1 records; 4 rounds on a fixed subsample of 1,500 targets (192k
  samples, about one sixth of a full ladder arm). Fixed in samples, not in time.
- **Pretrain data is "fixed initial dataset"** until the data workstream ships a successor.

### Constraints

- **Parameters.** At most 10M unique trainable parameters, embeddings included. Looped or shared
  weights count once.
- **Runtime.** One wall-clock budget T from process start to final eval, `torch.compile` time
  included; only pod boot and data download are excluded. Because the RL and eval stages are
  fixed in samples, a model that is slow to sample pays out of its own pretrain time. Set T to
  2× the measured end-to-end time of the baseline (expected around 30 minutes).
- **Hardware.** One pinned GPU SKU for every leaderboard entry; if it is out of stock, wait
  rather than substitute. With the systems layer open, speedups are SKU-specific and cross-SKU
  numbers are not comparable. Agent's suggestion: A40 ($0.49/hr, Dan's validated template,
  Ampere like the A100). Robbie mentioned A100; Dan's notes list A100 stock as low. Undecided.
- **From scratch.** Every weight update happens inside the timed run from random initialisation:
  no offline teacher, no distillation, no external weights, no hand-written proofs.
- **Decoder rule** (needed because tokenization is open). The map from model tokens to the spec
  proof format must be deterministic and single-pass; it may not call the verifier or try
  alternatives. Content-addressed citation lookup passes; "try each candidate line until it
  verifies" does not. This changes what "the model proved it" means and needs group agreement.
- **Inference.** One sample stream per attempt; no search or best-of-N beyond the protocol's k.
- **Inherited from Dan's reviewer rules.** Frozen control at equal attempts; at least 2 seeds
  before a difference is called a difference; expected results committed before the run.
  (This is fine, but if first seed fails quit, only validate with second seed.)

### Open to change

Architecture, positional scheme, looping, optimizer, schedule, batch size, precision, kernels and
compile settings, tokenization (within the decoder rule), loss masking, auxiliary objectives,
data order, curriculum, epochs, atom-renaming augmentation (stays within existing theorem
classes, so it adds no information).

Bookkeeping suggestion, since the systems layer is open: tag each change `systems` or `modeling`
in the experiment log, so pure speedups can be separated from pretrain improvements afterwards.

### Metric

- **Primary:** number of dev-transfer theorems solved with `L_true ≥ 9` after the short
  protocol, mean over the 3 seeds. `L*` is an integer that ties at 10 for every trained arm so
  far, so it cannot be hill-climbed.
- **Secondary:** `L*`, solve rate by `L_true` bin, the frozen (no-RL) control for the same
  checkpoint, and solves by rule family.
- **Candidate cheap proxy:** the frozen control's pass@32 on the dev pool. If it tracks the
  primary metric, it becomes the inner loop and only winners get the RL stage.
- **Accept rule:** a change is accepted only if its 3-seed mean beats the incumbent by more than
  2 standard errors of the baseline.

## Risks and limitations

- **Coupling to the RL recipe.** This optimises pretraining for plain expert iteration. Dan's
  round 2 run 4 found GRPO elicits faster but drops held-out greedy from about 0.9 to 0.34–0.67,
  so the RL workstream's eventual recipe may prefer a different base. Re-score the top 3 configs
  under that workstream's current best recipe every few days.
- **Narrow wins.** The ladder's +3 in `L*` was one proof family (implication towers) written
  longer; 17 of 19 textbook schemata stayed at about 0 under every technique. A pretrain "win" on
  the primary metric could be more of the same, hence solves by rule family.
- **Unverified tooling.** The author has not run or read the fork's code; `minlen.py`, the
  pools and `expert_iter.py` are known only through the summaries.
- **Dan's pause.** `docs/STATE.md` on `origin/dan` records "no new experiment launches until Dan
  lifts the pause" (2026-09-18). The author assumed this binds Dan's agents only. Unconfirmed.
- The `L_true` labels carry the prover's caveat: non-existence of a shorter proof is established
  only within the subformula closure plus negations.

## Next steps

1. Robbie reviews this note; until then nothing in it is agreed.
2. Port the baseline into the timed harness and run 5 seeds, giving σ for the accept rule and T.
3. Run the 4-round and the full 8-round protocol on 3 checkpoints; check that the short protocol
   ranks them the same way.
4. Measure how well the pass@32 proxy (and pretrain loss) correlates with the primary metric.
5. Content-addressed tokenization against `abs` under the decoder rule.
6. Raise with the group: the decoder rule, the GPU SKU, and whether pretrain data is Dan's
   Stage-1 file.
