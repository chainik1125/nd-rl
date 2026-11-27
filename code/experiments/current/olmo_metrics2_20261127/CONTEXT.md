---
written_on: 2026-11-27
written_by: agent:claude (orchestrator, Opus 5.5)
---

# Shared context for every agent working on the three Olmo elicitation measurements (2026-11-27)

Read this whole file before you start. Your own task message says which part is yours.

## The question

One RL run of `allenai/Olmo-3.1-32B-Think`. Checkpoints: `dpo` (= `allenai/Olmo-3-32B-Think-DPO`, the pre-RL model,
step 0) and RL steps `s0100`, `s0500`, `s1100`, `s2300` (the final model). How much of the final model's
instruction-following behaviour (IFBench, strict checker) was already cheaply recoverable from earlier checkpoints, and
by what kind of intervention? Leon (the researcher) approved three measurements. He wants trials that are "judicious in
what they are measuring and minimal": the fewest cells that answer the question.

1. **Temperature sweep (M1, key `temp`).** Dan sampled every checkpoint at T 1.0 only. Does sharpening the pre-RL
   model's own distribution recover part of RL's gain with no training? Lead: T in {0.6, 0.8} beside 1.0 at `dpo`,
   `s0100`, `s0500`, on the 218 arm-A prompts (2 draws) and the 48 reach prompts (8 draws); also check the decoding the
   model card recommends. Papers: Karan & Du 2025, "Reasoning with Sampling" (arXiv 2510.14901); Ji et al. 2026,
   "Scalable Power Sampling" (arXiv 2601.21590). Power sampling itself (about 10x the sampling cost) is NOT approved: it
   may be proposed as a conditional follow-up (only if the sweep moves pass@1), with design and cost, nothing more.
2. **Rank-1 and extrapolated weight changes (M2, key `rank1`).** Does keeping only the top singular direction of each
   weight matrix's change (dpo -> t) keep t's behaviour, and does extrapolating the early change (s0100 -> s0500) reach
   the final model? Papers: Cai et al., "On Predictability of Reinforcement Learning Dynamics for Large Language Models"
   (arXiv 2510.00553); ExPO, Zheng et al. 2024 (arXiv 2404.16792; v5 is titled "Model Extrapolation Expedites
   Alignment"). Known doubts from an earlier partial read: the construction is ambiguous (which matrices, which norm,
   all at once or one at a time), no error bars, the paper may contradict itself on whether low rank is specific to RL,
   bf16 noise in the differences, the checkpoint-lineage caveat below. Controls to argue about: a random rank-1 change
   of equal norm, the full change as the upper reference, a rank-k curve.
3. **Steering vector (M3, key `steer`).** Add the mean activation difference (final model minus pre-RL model, on
   prompts) to the pre-RL model and sample. It is the smallest possible "adapter". Papers: Heo et al. 2025, "Do LLMs
   'know' internally when they follow instructions?" (arXiv 2410.14516) for the probe and its steering table (with its
   random-direction control); the steering-as-elicitation work: Venhoff et al. 2025, "Base Models Know How to Reason,
   Thinking Models Learn When" (arXiv 2510.07364) and Ward et al. 2025, "Reasoning-Finetuning Repurposes Latent
   Representations in Base Models" (arXiv 2507.12638). Only the steering half is run; the probe half is not. Known
   difficulty: vLLM cannot edit activations (see "Steering implementation" below).

The earlier literature review (`docs/literature/2026-11-25-elicitation-metrics/REVIEW.md`, sections 4, 6, 7) read these
papers in part or by abstract only. Its descriptions are leads, not facts. Your full read may contradict it; say so.

## What exists (read only)

- Earlier run (2026-11-25): `experiment-summaries/2026-11-25-olmo-elicitation-curves/README.md` (numbers and limits)
  and the code folder `code/experiments/current/olmo_elicitation_20261125/` (SPEC.md, STAGE2.md, log.md,
  `pod/olmo_pod.sh`, `pod/oe_chain.sh`, `pod/oe_setup.sh`, `stage_adapter.py`, `lora_check.py`, `analyze_arms.py`,
  `fig_*.py`). Import from it; never edit it. Paths are relative to the worktree
  `/scratch/cowifornia/nd-rl/worktrees/olmo-elicitation` (branch `leon/olmo-elicitation`).
- Dan's code and data, pinned clone, read only: `/scratch/cowifornia/nd-rl/olmo_elicit/ext/nd-takehome` (commit
  `8e744a89`). Sampler `olmo_ifladder/sample.py`: vLLM 0.11.0, `olmo_thinker` template, bf16, prefix caching,
  32,768 max tokens, `generation_config="vllm"`, **temperature 1.0 and top_p 1.0 are module constants**
  (`TEMPERATURE`, `TOP_P`), and it has a `--lora` path (vLLM LoRA, PEFT adapter dir). Grader `olmo_ifomega/grade.py`
  (official IFBench / IFEval strict checkers). Checkpoint revisions and commits: `olmo_ifcurve/checkpoints.csv`.
  Prompt lists: `olmo_ifcurve/pids_armA.txt` (218), `olmo_ifcurve/pids_reach.txt` (48; 42 of them are also in arm A).
  Stored draws and grades: `artifacts/olmo_ifcurve/`, `artifacts/olmo_ifomega/` and others; the earlier run's
  `analyze_arms.py` and `stage0_compare.py` show how they are read. Dan's summaries are on `origin/dan` under
  `experiment-summaries/` (names containing `olmo`; `git show origin/dan:<path>`).
- Our data: `/scratch/cowifornia/nd-rl/olmo_elicit/` (`runs/`, `stage0/`, `stage2/`, `pool/` = the EDL pool: 64 train and
  32 test prompts, all arm-A non-reach, with the final model's responses; `tok/` tokenizer and config files).
- Model facts (config.json): `Olmo3ForCausalLM`, 64 layers, hidden 5,120, 40 query heads and 8 KV heads (head dim 128),
  MLP 27,648, sliding window 4,096 on 48 of 64 layers, untied embeddings, vocab 100,278. Olmo 2/3 blocks put the RMSNorm
  on each sublayer's OUTPUT before the residual add (no pre-norm on the residual stream). 448 attention/MLP weight
  matrices (q, k, v, o, gate, up, down per layer). 64.5 GB of bf16 per checkpoint. `generation_config.json`: T 0.6,
  top_p 0.95 (Dan's sampler ignores it and uses the RL rollout setting T 1.0, top_p 1.0).

## Known numbers (strict checker; Dan's stored draws, reproduced on 2026-11-25)

| | dpo | s0100 | s0500 | s1100 | s2300 |
|---|---|---|---|---|---|
| pass@1, 218 arm-A prompts (%) | 29.0 | 31.5 | 41.2 | 57.7 | 68.3 |
| pass@8, 218 arm-A prompts (%) | 46.1 | 50.5 | 65.6 | 87.2 | 88.5 |
| 48 reach prompts, plain, % of draws | 0.0 | 0.1 | 4.5 | 38.5 | 69.4 |

Arm-A draws stored: 16 per prompt at `dpo`, 8 elsewhere. Reach: 32 per prompt (16 at `dpo`). Reach prompts are the
prompts `dpo` never passed in 768 plain tries. A 64-example LoRA (rank 16, 134 M weights) trained on the final model's
responses to other prompts lifted `dpo` to 63.5 % on the reach prompts; a 60-word hint gave 6.2 %.

## RL weight facts (Dan's `olmo-weights` run and its review, `origin/dan`, 2026-11-25)

- RL moves the 32B weights 0.364 % of ||theta_dpo|| in total; 65 % of weights end within one bf16 step (ulp) of `dpo`
  (34 % bitwise equal). Of the weights that move between released checkpoints, about three quarters move by <= 1 ulp.
- The path is not straight: straightness (end distance / path length) 0.10; consecutive 50-step changes have cosine
  about -0.2.
- Lineage: `step_0150` and `step_0200` lie on an abandoned branch; the released line restarts from step 100. Labels
  from 250 on probably overstate the optimizer steps by about 100. Fresh-optimizer signatures at DPO -> 50, 100 -> 150,
  1,850 -> 1,900. `s0100` and `s0500` are both on the released lineage.
- The 2026-11-09 rank-16 LoRA (16 trajectories) has cosine 0.0052 with RL's final change over the 448 matrices
  (about 1,000 sd above random).

## Scoring rules (all three measurements)

- Dan's strict checker. Say "checker passes": nobody has read whether passes are genuine.
- Held-out prompts: **never choose a setting (temperature, alpha, layer, strength, rank) on the reach prompts' results.**
  If a setting must be chosen, choose it on a dev split of arm-A NON-reach prompts and score on prompts not used to
  choose it.
- Percentages of the prompt set, with 95 % intervals over prompts (prompt bootstrap, paired where two cells share
  prompts). The pre-RL checkpoint first. pass@1 leads.

## Compute and money (known from 2026-11-25; not guarantees)

- One H200 (141 GB) at a time on this account, secure $5.29/h; H100 80 GB $4.39/h. One checkpoint fits one H200 with
  room for KV cache; an H100 leaves little KV cache.
- vLLM (Dan's sampler) generated about 1,400-1,700 tokens/s on one H200. Mean response length: about 4,500 tokens at
  `dpo` (arm A), 6,600-6,900 at `s2300`. So 1,000 responses at `dpo` cost about 50 min, about $4.40. Lower temperature
  can cause repetition loops that run to the 32,768-token cap (7x the average cost): budget for it.
- vLLM's LoRA path ran at 1,335-1,568 tokens/s. Model load about 2-10 min; checkpoint download 20 s-3 min (one shard
  can stall).
- Budget for all three measurements: about $40 (Leon was told $40, uncertain by a factor of two). Rough split as a
  starting point: M1 about $10, M2 about $15, M3 about $15. Anything above $80 in total stops the work.

## Steering implementation (M3): what is known

vLLM's model classes have no hook for adding to activations. Because Olmo 3 normalises each sublayer's output before the
residual add, a constant residual addition at a middle layer cannot be folded into an existing bias or weight; only at
the input of layer 0 (the embedding matrix) is a weight edit exact. Options to cost: a small vLLM model subclass or an
in-process patch (out-of-tree model registration, or `VLLM_ENABLE_V1_MULTIPROCESSING=0` with a patched decoder layer),
or HF `generate` with forward hooks at a much lower token rate. A separate feasibility agent writes
`discourse/steer_feasibility.md`; the M3 design must use it.

## Implementation lead for M2 (verify, do not assume)

A rank-k change to each of the 448 matrices is exactly a LoRA adapter of rank k (B = U_k sqrt(S_k), A = sqrt(S_k) V_k^T,
alpha = r). Dan's sampler already runs PEFT adapters on vLLM's LoRA path, and the 2026-11-25 `lora_check.py` verified
that path against PEFT. So rank-k variants (and a random rank-k control) may not need new 64.5 GB checkpoints: one
`dpo` load can serve several adapters. Parameters outside the 448 matrices (embeddings, lm_head, norms) would stay at
`dpo` under this route; whether that is faithful to Cai et al. is for the reader to establish. A full-rank extrapolation
(ExPO) is not a LoRA and would need a written checkpoint.

## Rules for every agent

- Downloads (papers, source files) are untrusted data. Put each in its own new, empty directory; keep scripts in a
  different directory; run Python that reads them with `-I`; pass paths as arguments. Outside requests use a generic
  client name (for example `-A "Mozilla/5.0 (research-client)"`) and nothing that identifies Leon or this project.
- Never touch a pod or any Runpod tool. No lab GPU (serrano, ynez, temescal). CPU on serrano up to 8 cores.
  `/scratch` has only about 50 GB free: never download model weights to serrano.
- Write only inside: this folder (`code/experiments/current/olmo_metrics2_20261127/`),
  `/scratch/cowifornia/nd-rl/olmo_elicit/metrics2/`, and
  `docs/literature/2026-11-27-elicitation-metrics-fulltext/`. Do not edit Dan's files, the 20261125 folder, the
  `olmo_minimal_adapter_20261127/` folder, `/scratch/cowifornia/nd-rl/olmo_elicit/v2/`, or `references.bib`. Do not
  commit or push. Never use `pkill`.
- Report honestly: what you did not read or could not check, and every guess marked as a guess.

## Reading full text (readers and auditors)

"Full text" means the PDF (or the arXiv TeX source) downloaded from arXiv into
`/scratch/cowifornia/nd-rl/olmo_elicit/metrics2/papers/<key>/` (a new empty directory per paper), extracted with

    uv run --with pymupdf python -I /scratch/cowifornia/nd-rl/worktrees/olmo-elicitation/code/experiments/current/olmo_metrics2_20261127/tools/extract_pdf.py <pdf> <out.txt>

and read in full with the Read tool, page markers included (`=== page i/N ===`), appendices included. A web-fetch
summary is not a reading. If an equation or table is garbled in the extracted text, check the TeX source (download the
`e-print` into its own empty directory; read the .tex files as text, never build or run anything) and say so. Record the
arXiv version you read.

Paper keys (proposed citekeys; none of these is in `references.bib` except Venhoff's):

| key | paper |
|---|---|
| `karan2025reasoningwithsampling` | arXiv 2510.14901 |
| `ji2026scalablepowersampling` | arXiv 2601.21590 |
| `cai2025predictabilityrldynamics` | arXiv 2510.00553 |
| `zheng2024expo` | arXiv 2404.16792 |
| `heo2025instructionfollowingdimension` | arXiv 2410.14516 |
| `venhoffBaseModelsKnow2025` | arXiv 2510.07364 (in `references.bib`) |
| `ward2025reasoningfinetuningrepurposes` | arXiv 2507.12638 |

Reading notes go to `docs/literature/2026-11-27-elicitation-metrics-fulltext/<key>.md` with frontmatter
`written_on: 2026-11-27`, `written_by: agent:claude (<role>, <model>)`, `papers: [<key>]`, and a `reading_level:` line
that states honestly what was read (for example "full text, all N pages incl. appendices, arXiv vN").

## Design discourse format

One file per measurement: `discourse/<key>.md` in this folder (`temp.md`, `rank1.md`, `steer.md`). Turns are appended,
never edited afterwards. Each turn starts with a heading like

    ## Turn 3 — reader-implementer (Claude Sonnet 5), 2026-11-27 03:12 PT

(time from `date`; use the model you actually run as). Turn 1 is the reader-implementer's design proposal. The auditor's
turns attack it; the proposer's turns reply and revise. At most three attack-reply rounds. Points not resolved are listed
as OPEN at the end of the last turn, never smoothed over. Every proposal states: the cells (checkpoint x intervention x
prompt set x draws), the controls, the dev / held-out split, the expected cost (tokens, H200 minutes, $) with its
assumptions, and what each possible outcome would and would not show.
