---
written_on: 2026-10-06
written_by: agent:codex
---

# Frozen test evaluation and count-only stopping controls

`evaluate_live_path.py` permits development evaluation immediately. Test evaluation
requires `--frozen-protocol PATH`, after model and checkpoint selection on development.
The JSON file has this schema:

```json
{
  "status": "frozen",
  "selection_split": "dev",
  "test_data_sha256": "SHA256 of the exact test prompt JSONL",
  "test_seed": 62006,
  "test_repeats": 2,
  "test_theorems": 400,
  "registered_models": [
    {
      "arm": "no_cost",
      "rl_seed": 101,
      "base_sha256": "copied from the selected DEV config.json",
      "checkpoint_sha256": "copied from the selected DEV config.json, or null for format-only",
      "inference_sha256": "copied from the selected DEV config.json",
      "counter_hazards_sha256": "SHA256 of this model's frozen DEV hazard JSON",
      "budget_baselines_sha256": "SHA256 of this model's frozen DEV budget-baseline JSON"
    }
  ],
  "postprocessing_hashes": {
    "fit_counter_hazards.py": "SHA256",
    "reanalyze_counter.py": "SHA256",
    "fit_budget_baselines.py": "SHA256",
    "reanalyze_budget_baselines.py": "SHA256"
  },
  "statistical_analysis": {
    "script_sha256": "SHA256 of aggregate_live.py",
    "bootstrap_samples": 10000,
    "bootstrap_seed": 72006,
    "confidence": 0.95,
    "family_data_sha256": "SHA256 of the exact test family JSONL"
  }
}
```

The inference fingerprint binds the base and policy checkpoint, context and sampling
parameters, precision, live protocol version, hashes of `live.py`, `controller.py`,
`train_live.py`, `train_controller.py`, `evaluate_live_path.py`, and
`evaluation_protocol.py`, and source dependency hashes. It excludes the dataset, evaluation seed,
output directory, and descriptive utility coefficient. The test data hash is bound
separately, together with the prespecified seed and two draws per theorem. Changing model weights or the generation interface requires a new
development evaluation and registration before test outcomes are inspected.

Primary development configs obtain their fingerprint with
`evaluation_protocol.fingerprint(evaluation_protocol.signature(config))`. Older
pilot configs lacking any required behavior hash remain development-only and must
be evaluated again under the registered implementation. Root owns the final freeze
file and the registered model list. Hashing a test prompt file does not evaluate proofs.

`fit_counter_hazards.py --paths DEV/public_paths.jsonl --data DEV_PANEL.jsonl
--out HAZARDS.json` fits count-only hazards using public model stopping probabilities.
It reads no private correctness labels. At each count it averages the learned hazard,
weighted by the learned probability of reaching that count, over nonforced states.
The resulting control matches the development stopping-count distribution when
forcing occurs only at the terminal budget. Context-dependent forcing can break
that equality, so realized attempts and compute must still be reported.

Counter files bind the fitting dataset and public paths, model checkpoint, base,
and inference fingerprint. Applying a file to a different model or interface is an
error. Baseline controller compute is charged only for terminal selection among two
or more candidates; learned stopping requires a model forward at every reached
nonforced decision, including the first. Actual full-path measurement expense is
reported separately from expected deployable policy work.

Categorical stopping and index expectations are primary. Prespecified secondary
readouts use greedy index selection with uniform exact ties, including matched
fixed-count controls, and a fully greedy stop threshold of probability at least 0.5.
These variants require no additional generation and must all be reported rather
than selected using test outcomes.

`fit_budget_baselines.py` reads development public paths and aligned private labels
to select a context-length threshold from the prespecified grid 512, 768, 1024,
1536, 2048, 3072, 4096, 8192, and 16384. It maximizes the checkpoint's native
expected development reward, with ties preferring fewer attempts and then the
smaller threshold. The threshold uses the public controller prefix token length,
with ordinary hard-limit fallback and the same categorical index readout.

The same fitter freezes mixtures of fixed counts 1/2/4/8 that match development
expected attempts or combined generator and controller forward tokens. Among exact
matches it maximizes native development reward. A mixture matching both is also
reported when the learned budget lies inside the fixed-mixture convex hull; an
infeasible joint match is explicitly recorded. These are measured forward-token
proxies, not exact floating-point operation counts. The final registry must retain
the fitting-file hashes before test labels open. `reanalyze_budget_baselines.py`
applies the frozen weights and threshold to immutable evaluated paths; test budget
differences and any descriptive test envelope must not refit those weights.

Both fitters reject paths collected under a different current behavior
implementation. Both reanalyzers additionally check the fitter's recorded source
hash. Test reanalysis requires `--frozen-protocol` matching the original path run's
registry hash, the matching per-model fitting-file hash, and registered fitter and
reanalysis source hashes before private labels are read. These checks prevent a
new cost calculation or baseline fit from silently being credited to an older run.

`aggregate_live.py --runs RUNS.json --families TEST_FAMILIES.jsonl
--frozen-protocol FREEZE.json --out RESULTS.json` requires all three cost and
no-cost training seeds. The run manifest lists `arm`, `seed`, the evaluated folder
as `paths`, and derived stopping readouts as `counter` and `budgets`. Optional
controls include `format_only` with `seed: null`, and the single-seed fixed-budget
training controls. The registry uses matching `arm` and `rl_seed` fields.

Aggregation validates statistical settings and each primary model, arm, dataset,
and draw registration before reading that model's integrated outcomes. It then
validates supplied derived-readout provenance and registered fit hashes before
including those readouts. An invalid fit prevents any aggregate output; this is
not an all-or-nothing manifest gate before every primary outcome read.

The primary estimate averages the two draws within each theorem, then the three
trained models within each arm. Paired bootstrap samples resample whole theorems
within each family, preserving family composition, with 10,000 samples and seed
72006. Intervals are conditional on the trained models; each seed and the standard
deviation of seed means are also reported. Single-seed control intervals do not
measure control training variability. Cross-arm utilities use common cost
coefficients 0, 0.01, 0.03, and 0.1 rather than subtracting unlike native rewards.

The format-only initialization is a named secondary control. Its fresh stop and
selection heads are untrained, so native head readouts are diagnostic. Meaningful
generation comparisons use its fixed-first correctness and retained correct-proof
availability on the finite forced path. Oracle availability is not a deployable
selection rule. No test result selects a greedy readout or a utility coefficient.

## Scheduled test launches

`launch_live_test.py` executes one root-registered model at a time. Required
arguments are `--registry`, `--benchmark main|historical`, `--data`, `--arm`,
`--gpu`, and a new `--out`; trained models also specify `--training-seed`.
The historical mode additionally requires `--main-registry`. `--family-data`
defaults to the evaluation dataset; specify it when the registered metadata file
differs. Both registries must bind the reviewed `launch_helper_sha256`.

The helper checks selected DEV config/model/source/runtime bytes and fixed MAIN
inference settings, registered dataset/draw/statistical metadata and fitting
artifacts, and fresh GPU occupancy before launching. Existing output or launch
artifact paths are refused. Root controls GPU scheduling; the helper does not
wait for or displace an occupied device.

Helper/evaluator snapshots and the exact command are stored beside the output.
The launch ledger records preflight, whole helper-main wall time, and complete
subprocess times. The generation/checking subprocess includes model setup,
diagnostic replay, and Lean CPU checks; these are separated further by the path
summary. Registered counter/budget reanalysis runs on CPU for primary arms only,
producing `counter_integrated.json` and `budget_integrated.json` inside the new
path folder. CPU reanalysis is not additional GPU inference work. Interrupted or
failed artifacts remain available and are never silently overwritten.
