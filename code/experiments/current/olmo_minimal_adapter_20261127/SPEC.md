---
written_on: 2026-11-27
written_by: agent:claude (opus, orchestrator)
---

# Olmo minimal adapter: spec

Written 2026-11-27 2:05 am PT, before any pod exists. This is the contract between the orchestrator and the build
agents. The readings fixed in advance are in [PREREG.md](PREREG.md); what ran is in `log.md`. The design came out of a
written discourse between a proposer (Sonnet, who read the EDL papers in full) and an auditor (Opus, who read them
independently): [discourse/edl_design.md](discourse/edl_design.md). Reading notes: [literature/](literature/).
The first draft ([DESIGN_DRAFT.md](DESIGN_DRAFT.md)) is superseded where this file differs.

## Questions (Leon, 2026-11-27)

- **A.** How small an adapter, and how few examples, bring a checkpoint to the final model's plain pass rate on the
  48 reach prompts?
- **B.** Does that still work when no training token comes from an RL-trained model later than the checkpoint?

Background: the 2026-11-25 run (`../olmo_elicitation_20261125/`) trained one rank-16 LoRA on all projections
(134,217,728 trained weights) for one pass on 64 of the final model's (`s2300`) responses. It lifted the pre-RL model
`dpo` from 0 % to 63.5 % [53.7, 73.2] checker passes on the reach prompts, against the final model's 69.4 %
[60.7, 77.7]. No size or example sweep was run, and the examples came from the RL-trained model.

## Stage 1: `dpo` only

### Example sources (all checker-only or filter-only: no hand labels are used to choose any example)

Prompts: Dan's arm-A IFBench prompts minus the 48 reach prompts (176). Matched set: prompts where both `dpo` and `s2300`
have an eligible stored draw (strict pass, `finish == stop`, contains `</think>`, at most 6,144 response tokens).
A seeded shuffle (seed 20261127) gives 64 train prompts, whose order is the nested order (n = 1, 4, 16, 64 are its
prefixes), and the next 32 as the dev set. The same prompts serve every source. One draw per prompt, a uniform seeded
pick (seed 20261125) among the source's eligible draws. Builder: `build_pools.py`; pools in
`/scratch/cowifornia/nd-rl/olmo_elicit/v2/pools/dpo/<source>/`.

| source | draws | filter | role |
|---|---|---|---|
| (i) `own` | `dpo`'s stored plain-prompt draws | strict pass + stop + `</think>` + cap | main condition for question B: verifier-selected own samples (rejection sampling) |
| (ii) `final` | `s2300`'s stored plain-prompt draws | same | distillation reference; the old condition |
| (iv) `unsel` | `dpo`'s stored draws | stop + `</think>` + cap, checker ignored | control: training on any own response in the template |
| (iii) independent answers | not run | | IFBench has no reference answers; a Think model's target needs a thinking segment, and rule-written or empty thinking is far off-distribution; rule-written answers that satisfy a format checker are gaming by construction; Claude-written reasoning traces for training are blocked in this project |

Hand labels (Dan's genuine / gamed) are used only to report the labelled-gamed share of each training set.

### Adapter ladder (P = trained weights; scale alpha/r = 2 on every rung; all exported as PEFT LoRA files vLLM serves)

| rung | what is trained | P |
|---|---|---|
| A16 | LoRA r16 on q, k, v, o, gate, up, down, all 64 layers (the old adapter) | 134,217,728 |
| A1 | LoRA r1, same modules | 8,388,608 |
| O1 | LoRA r1 on o_proj only, all 64 layers | 655,360 |
| M65k, M16k, M4k, M256 | rank-1 LoRA on all modules; A frozen at its seeded PEFT init; k uniformly random entries of B trained (masked gradient, weight decay 0), all other B entries fixed at 0 | 65,536 / 16,384 / 4,096 / 256 |
| M4k-attn | as M4k, entries drawn from the q, k, v, o modules only | 4,096 |

The sparse rungs are our construction: Bits That Count (BTC) does not specify its sparse subsets, and Donoway et al.
2025 draw from A and B (selected A entries get no gradient while B = 0). P counts trainable entries only; the fixed A is
a seeded random matrix.

### Training rule (`mini_train.py`, one rule per cell)

- AdamW (0.9, 0.999, eps 1e-8, weight decay 0), constant lr, batch 1, grad clip 1.0, loss = mean token cross-entropy
  over the whole response (thinking + answer, as BTC's reasoning runs), bf16 base, fp32 adapter.
- Steps cycle through the n examples (epoch 0 in the nested order, later epochs seeded shuffles).
- lr grids (scale 2): A16 {1e-4, 3e-4, 1e-3}; A1 {3e-4, 1e-3, 3e-3}; O1 {1e-3, 3e-3, 1e-2}; M {3e-2, 1e-1, 3e-1}.
  Edge rule: a selected lr on a grid edge adds the next value of the 1-3-10 sequence beyond it, once.
- Step budgets: A16 at n = 64: 128 steps, dev evals {16, 32, 64, 128}. O1, A1, M rungs of source (i) at n = 64:
  all lrs to 128 steps (evals 16, 32, 64, 128), then the lr with the lowest dev loss at 128 continues to 512 (evals
  256, 512). n = 4 and n = 1: 64 steps, evals {1, 2, 4, 8, 16, 32, 64}. Capacity rungs of source (ii): 64 steps
  (evals 16, 32, 64); M16k and M4k then extend at their best lr to 512.
- Early stop of one lr: its dev loss is above its own running best at two consecutive evals.
- **"Trained"** = the (lr, step) with the lowest dev loss per token (total) on the 32 dev responses of the same source.
  Flags: `did_not_train` (no eval below step 0), `unconverged` (for the selected lr, the dev loss still fell by more
  than 1 % of (L0 − L_best) between the last two evals).
- EDL (bits = nats / ln 2): with step i scoring example i before its first update, at a step t with m = min(t, n),
  EDL(t) = Σ_{i≤m} nats_i − (Σ_{i≤m} tokens_i) · L_dev(θ_t). Primary capacity number: EDL at t = 64 (n = 64), the lr with
  the lowest dev loss at 64 ("one-pass C" = EDL_64(P) / EDL_64(A16)); for the extended (ii) rungs also EDL at the
  dev-selected step. Own-source cells report the checker bound Σ log2(1/a_i) instead (a_i = the filter's acceptance
  rate on prompt i, (k + 0.5)/(N + 1) from the stored draws).

### Scoring

- The 48 reach prompts (never in any training or dev set), 8 draws each, Dan's sampler settings unchanged (vLLM 0.11.0,
  olmo_thinker template, T 1.0, top-p 1.0, 32,768 max tokens, bf16, prefix caching), on vLLM's LoRA path with several
  adapters in one engine (`multi_sample.py`, which imports his sampler's functions; only the multi-adapter driver
  differs), graded by his `olmo_ifomega/grade.py`, strict.
- Gate before sampling (`lora_gate.py`): teacher-forced per-token NLL of 4 examples under vLLM, without and with each
  adapter, against PEFT's: PASS needs correlation ≥ 0.9 of the per-token changes and a summed-change ratio in
  [0.75, 1.33]; UNINFORMATIVE (effect below the bf16 noise) is flagged; FAIL is not sampled.
- Prompt control (no adapter): `dpo` with Dan's answer-only suffix (his olmo-answeronly A1 wording) on the 48 reach
  prompts, 8 draws, his sampler unchanged.
- Reference: the final model's plain rate on the 48 prompts from Dan's stored `B_s2300` draws (32 per prompt; new
  draws taken after the reach set was fixed, `olmo_ifcurve/reuse_inventory.csv`): 69.4 %. `dpo` plain: 0 strict passes
  in Dan's stored reach draws.

### Readouts (`analyze_cells.py`, `fig_cells.py`)

Per sampled cell: G = (cell − `dpo` plain) / (reference − `dpo` plain), the share of the gap closed, with a 95 %
two-level paired bootstrap (prompts, then draws within prompt for cell and reference; 10,000 reps); the pass rate and
the paired difference to the reference in points beside it; prompts with a pass in 8; median response and thinking
tokens and answer words; the rate split by whether the reach prompt's instruction types occur in the cell's training
prompts (`types_table.py`). Paired contrasts: own − unsel, final − own, every cell − prompt control.

## Order on the pods (stage 1)

- H100 80 GB (training): own A16 n64, own O1, own n4, own n1, the (ii) capacity rungs, then by the search rule A1 or
  the M4k pair, and the confirmation cell.
- H200 (training first, then sampling): final A16 n64, unsel A16 n64; then the prompt control; then batches of adapters
  as the search rule calls them (gate, sample, grade). Weights of `dpo` are kept on both pods until stage 1 ends.
- Every result is fetched to serrano with md5 on both sides before a pod is released.

## Where things go

- Code and notes: this folder (branch `leon/olmo-elicitation`). It imports from `../olmo_elicitation_20261125/`
  (unchanged).
- Data and results: `/scratch/cowifornia/nd-rl/olmo_elicit/v2/` (`pools/`, `runs/<ckpt>/`, `stage/`, `figures/`,
  `footprint/`, `papers*/`).
- Summary: `experiment-summaries/2026-11-27-olmo-minimal-adapter/README.md`.
