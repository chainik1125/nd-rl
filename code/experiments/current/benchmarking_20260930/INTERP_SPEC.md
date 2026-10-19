---
written_on: 2026-09-30
written_by: agent:claude
---

# ndbench interp: spec (planning by the main agent, Opus; coding by Sonnet)

Agreed with Leon on 2026-09-30 after discussion. Four measures, results in the same Store and on the same
dashboard as the benchmarks. Models: base `ab2/frozen/pretrain.pt`, RL `ab2/monitor/final/champion2_n060_fullpool/final.pt`
(same architecture: 4 layers, d=256, RoPE; the RL model is fine-tuned from the base, so residual coordinates are shared).
Dropped after discussion: lens comparisons (except a free logit-lens byproduct, optional), generic success-vs-failure
A/B vectors, Venhoff-style optimised vectors. PLAN.md §A (environment, CPU/GPU, git rules) applies unchanged.

## Data (fit on reserve, evaluate on dev/sealed; split by theorem)
- Neutral proofs: `ab2/data/reserve.jsonl` (426k theorems, 406k with `minlen_proof`, L 2–10; use L 6–10, drop
  anything the `ExclusionIndex` marks `seen_train`), `ab2/sealed/sealed_L12-14.jsonl` `minlen_proof` (1,634, L 12–14),
  family witnesses from the `families` suite when present (long proofs).
- Model proofs: verified samples saved by benchmark runs (`Store` samples, `keep_samples=True`), RL and base.
- Evaluation sets for steering/interpolation: the `gen` items of `ab2/data/dev_L8-11.jsonl` (720) and of
  `ab2/sealed/sealed_L12-14.jsonl` (1,240), plus `textbook_dev` — built as ndbench suites (reuse `suites.existing`
  builders where they exist, else add `interp_eval` suites in your own module).

## 1. Ordering-marginalised likelihood (`interp/orderings.py`, `interp/likelihood.py`)
- Two proofs are "the same up to order" if one is a reordering of the other that keeps every line after the lines it
  cites, keeps each box contiguous with its AS line first and its last (discharged) line last, keeps PR lines first in
  their declared order and the conclusion last, and moves no line into or out of a box. Model it hierarchically: at
  each level the units are single lines and closed boxes; a unit depends on the units containing the lines it cites
  (box citations `N<start> N<end>` cite the box unit). Renumber lines and citations for each ordering.
- `count_orderings(body)` (exact: product over levels of the linear extensions of that level's DAG; bitmask DP when
  a level has ≤ 22 units, else report `None` and use sampling), `enumerate_orderings(body, cap)`,
  `sample_orderings(body, m, rng)` (sequential: pick uniformly among ready units; return log q(σ) for importance
  weights). **Every produced ordering must verify with nd_verify** (test it on the 60 fixture proofs and on
  constructed proofs with independent branches).
- `marginal_logprob(adapter, seq, body, cap=10_000, m=512)`: if count ≤ cap, logsumexp of `adapter.logprob` over all
  orderings (logprob already marginalises the start shift); else the importance-sampling estimate
  log(mean exp(logp(σ) − log q(σ))) with its ESS. Return single-order logp, marginal logp, n_orderings, method, ESS.
- Disagreement statistics (teacher-forced, both models on the same RL-verified proof): fraction of proof positions
  where the two models' argmax tokens differ; at those positions the base's rank of the RL model's argmax token and
  the base's entropy (Akgül et al. 2026 style).
- Run (`interp likelihood`): proofs = neutral minimal proofs (≤ 2,000) + RL verified samples + base verified samples
  (both from the Store when present; skip with a note otherwise). Rows: pid, origin, L, n_orderings, logp_single and
  logp_marginal under base and RL, disagreement stats, plus the per-theorem base success estimate p̂_base from the
  largest base run in the Store if any (to mark the "crux" set: RL solves, base ≈ never).

## 2. RL − base mean-difference steering (`interp/steering.py`)
- `fit_diff_vectors(base, rl, pairs, layers, position_types)`: `activations` of both models on the same teacher-forced
  (prompt + proof) sequences (2–5k reserve minimal proofs, L 6–10); δ_ℓ = mean over selected positions of
  (h_rl − h_base) after block ℓ; position types: all proof tokens (default), prompt tokens, line-start index tokens.
  Save as `.npy` under `out/interp/vectors/` with a JSON sidecar (layers, positions, n, source sha). Also record ‖δ_ℓ‖
  relative to the mean residual norm.
- `steering_sweep(cfg)`: for each (layer ∈ {0,1,2,3}, α ∈ {0.25, 0.5, 1, 1.5, 2}) build
  `base.with_steering((SteeringTerm(layer, path, α, "proof"),))` and run the standard `runner.run` on the eval suites
  (k=8, T=0.8). α=0 is the unsteered base run. Steered runs are ordinary Store runs (identity includes the spec hash).
  Also steer the RL model with −δ as a control (does removing δ hurt RL?), on the best layer only.
- Optional (flag, off by default): the early-stopping contrast vector — from base samples in the Store, attempts that
  stopped short (written < reference lines and failed) vs attempts that reached the reference length, mean activation
  difference at the last written line; same sweep on its best layer.

## 3. Weight interpolation (`interp/interpolate.py`)
- `interpolate(base_ckpt, rl_ckpt, alpha, out_dir) -> Path`: θ = θ_base + α(θ_rl − θ_base) for every tensor (check the
  two state dicts have identical keys/shapes; strip `_orig_mod.`), saved under `out/interp/ckpts/interp_a{α}.pt` with
  a sidecar (parents' sha256, α). The `absgpt` adapter loads it unchanged; its identity is its own sha256 plus the
  sidecar.
- α ∈ {0, 0.25, 0.5, 0.75, 1, 1.25, 1.5}. Evaluate with the standard runner: dev + sealed gen items at k=8 (pass@1),
  and a fixed 150-theorem subset (stratified by L, both sets) at k=128 (pass@k curve). The question: does pass@1
  rise at small α while pass@128 stays flat (sharpening), or does coverage appear only near α ≈ 1?

## 4. Linear probes (`interp/probes.py`)
- Datasets from teacher-forced neutral proofs through both models (identical inputs), features = residual stream at
  the embedding output and after each block (5 feature sets per model). Targets, at the line-start index token of
  each line (before the line's content is written):
  - `next_rule`: the rule of this line (15 classes);
  - `lines_to_go`: total written lines − current line index (report as regression R² and as 5-bin accuracy);
  - `cited_back` (optional): distance back to the first line this line cites (classes 1..8, 9+).
  - `success` (optional, needs Store samples): at the `PRF` token, whether an attempt of this model on this theorem
    succeeded (per-theorem p̂ ≥ 0.5), probing the model's own prediction of solvability.
- Probes: multinomial logistic regression / ridge in numpy or torch (no sklearn in the venv), L2 tuned on a
  validation split; train/val/test split by theorem. **Length generalisation for `lines_to_go`:** train on L ≤ 10,
  test on L 12–14 and family witnesses (L up to ~40).
- MDL probing (Voita & Titov 2020, online code over blocks 1%,2%,4%,…,100% of the training set) for `next_rule` and
  `lines_to_go` (binned): report code length and compression vs the uniform code, per layer, per model.
- Optional free byproduct: logit lens (final norm + unembedding applied to each layer's residual), top-1 agreement
  with the final prediction per layer per model.
- Readout: base ≈ RL decodability ⇒ information present in the base, RL changed its use; RL ≫ base ⇒ RL built it.

## Storage, views, dashboard, CLI
- Rows go under `out/store/interp/<kind>/<run_id>.jsonl` with an index `out/store/interp/runs.jsonl`
  (kind ∈ likelihood, vectors, probes, lens); steering and interpolation evaluations are ordinary Store runs whose
  model labels follow `steer:<layer>:<alpha>` and `interp:<alpha>` (keep `base` and `rl` for the two originals).
- Views in `analysis/interp.py` (`view_likelihood`, `view_steering`, `view_interpolation`, `view_probes`), registered
  in `analysis/__init__.py`'s registry (additive edit allowed). Pages `interp_likelihood`, `interp_steering`,
  `interp_weights`, `interp_probes` in a new `dash/interp_pages.py`, wired into `dash/build.py` and the nav
  (additive edits to WP6's `dash/build.py`/`dash/pages.py` allowed; you own those two files until you finish).
  Charts: likelihood scatter (marginal vs single; base vs RL) and histograms by origin, crux set highlighted;
  pass@1 vs α per layer with CIs; pass@1 and pass@128 vs α; probe accuracy/R² and MDL per layer, base vs RL, and the
  lines-to-go length-generalisation curve. Scores as percentages; "no data yet" panels when empty.
- CLI: a separate entry point `python -m code.ndbench.interp <likelihood|fit-vectors|steer-sweep|interpolate|probes>
  --out /scratch/cowifornia/nd-rl/bench/out --config <yaml>` (do **not** edit `code/ndbench/cli.py`; another agent owns
  it). Configs for the real runs go in `code/experiments/current/benchmarking_20260930/configs/interp_*.yaml`.
- GPU runs are scheduled by the main agent later (GPU 4 only, via `runner.gpu.lease`, with `CUDA_VISIBLE_DEVICES=4`
  set in the environment before the process starts). Budget estimate for all four: ≈ 1–1.5 GPU-h; put per-run
  estimates in the configs.
