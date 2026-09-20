Implementation for the proof-fragment sprint
==========================================

Root owns frozen configurations, population release, GPU leases and launches.
The implementation agent ran only tiny-policy CPU checks. Use
`/data/users/dmitry/simplex-research/.venv/bin/python` with
`PYTHONDONTWRITEBYTECODE=1`. Runtime imports the prior sprint's unchanged small
decoder, tokenizer vocabulary, forward sampler, parser and full verifier;
all imported source identities are recorded. Root discovery works from a
versioned snapshot beneath this sprint.

`populations.py` writes complete source certificates and provenance in six
JSONL files: candidate, adaptation, retention, selection, confirmation and
final_confirmation. The three probe roles have fixed 2–6, 7–9 and 10–12 source
length quotas. All source caps and length bins count total ND body lines,
including PR premise lines and AS assumption lines; prompt text and the
PRF, FINAL and QED markers do not count. Canonical context hashes determine
role assignment; probes additionally exclude canonical theorem/context
identities from every supplied `--exclude-training` corpus. Only blind textbook
guard hashes are read. An
incomplete generation writes recoverable artifacts with status
`incomplete_quotas_do_not_train`; root must document any quota amendment.

Each record preserves `prompt`, `body`, `text`, `source_id`, canonical
`theorem_id`/`context_id`, actual `n_lines`, `provenance`, `family`,
`default_cut`, rule sequence and fragment descriptors. Primary candidate
families are terminal rule types. IMPI is flat when the complete source's
maximum assumption depth is at most one, and nested when it is greater than
one. ORE is the thirteenth family when collection supports it. The default
intervention cut initially leaves the final three total body lines, or the
whole body for shorter sources: `cut = max(0, n_lines - 3)`. If the goal is
already available at that prefix, the cut moves earlier until it is admissible;
the resulting continuation can therefore contain more than three lines. This
first selector does not mine every possible internal semantic subgoal.

`nd_environment.py` independently replays rules and assumption scopes using
the original parser. Complete success requires this checker AND the unchanged
full verifier with strict preceding-reference enforcement. Prefixes contain
complete lines and never QED or a goal already available at top level. The
episode input is `original theorem PRF + supplied relative prefix + PRF`;
the model samples FINAL and its continuation. Prefix and continuation IDs are
decoded together so relative citations preserve absolute line positions.
Original premises are never replaced by prefix conclusions.

`fragment_experiment.py` commands:

* `adapt --checkpoint BASE --data adaptation.jsonl --steps 1000 --batch-size 32
  --fixed-pad 512 --lr 5e-5 --output NEW`: shared adaptation, with equal-size
  empty-start and randomly cut source pools.
* `calibrate --checkpoint ADAPTED --data candidate.jsonl --retention
  retention.jsonl --selection selection.jsonl --steps 100 --batch-size 32
  --fixed-pad 512 --lr 5e-5 --score-first-step --actual-displacement --top-k 3
  --output NEW`: each family and a source-unique matched ordinary control
  start from identical weights and empty AdamW state. Half of each batch is
  retention, with an independent shared RNG and asserted schedule hash.
  Optional `--families ...` restricts prespecified primary families. The
  primary score equally weights mean whole-proof bits in each reference
  length stratum. Both absolute and control-relative changes are saved.
* `intervene --checkpoint ADAPTED --data candidate.jsonl --retention
  retention.jsonl --families FAMILY1 FAMILY2 FAMILY3 --steps 100 --batch-size
  32 --fixed-pad 512 --output NEW`: frozen family mixture trial. Omit
  `--families` for the full family-balanced bank. Equal distinct-source quotas
  are required; the code never duplicates examples to fill a rare family.
* `evaluate --checkpoint MODEL --data POPULATION --start full --samples 8
  --temperature .8 --max-new-tokens 512 --audit-likelihood --output NEW`:
  saves per-theorem reference NLL, every raw sampled attempt, strict verdict,
  legal-prefix diagnostics and reverified compressed witnesses. For permitted
  training-bank diagnostics, `--start reset` uses the latest admissible cut;
  `--start fragment` uses the verified default boundary described above and
  additionally saves conditional NLL. Full evaluation raises if its fixed
  response-token budget does not fit the context window.
* `rl --checkpoint ADAPTED --data candidate.jsonl --retention retention.jsonl
  --start full|reset --families ... --steps 1000 --max-flop-proxy BUDGET
  --flop-milestones M1 M2 --max-generated-tokens CEILING --max-new-tokens 512
  --output NEW`: on-policy leave-one-out update with exact sampled behavior
  probabilities, teacher-likelihood audit before every update, a fixed shared
  response-token denominator, and separately recorded retention. Defaults:
  groups of eight, four source groups/update, 20% empty starts, 64 local
  rollouts before moving the cut, backward success threshold >.8 and forward
  threshold <.1. Prompt, reset and retention RNG streams are separate.

All learning commands accept `--device cuda --precision fp32 --seed N`.
FP32 is the default for tight behavior/teacher checks; `--max-logprob-error`
must be fixed before a launch. Every sampled token, including invalid tokens,
scratch, FINAL, QED and sampled PAD, receives action credit. Supplied context,
prefix separators and padding have zero direct loss. Context activations can
still receive gradients through later action losses.

The compute ledger reports padded/actual token counts and a declared dense
FLOP approximation: projections/MLP use 2×parameters×tokens for forward and
6× for backward; attention adds 4×layers×dimension×token-pairs or 12× for
backward. Generation includes padded prefill and all executed cache decode,
including work on rows that have already stopped. Teacher audit and retention
are included. These are not hardware-measured FLOPs. RL completes its current
on-policy update when crossing a ceiling and records overshoot plus a
conservative maximum one-update bound. Common FLOP milestones save the first
checkpoint at or above each specified budget and record actual compute.
Checkpoints contain model weights and metadata, not resumable optimizer/RNG
state; restarting training is a new declared run, not an exact continuation.

Confirmation evaluation requires root's `--release` JSON, checked using
identity metadata before proof payload parsing:

```
{"frozen": true, "no_further_selector_changes": true,
 "data_sha256": ["..."], "checkpoints": ["..."],
 "no_further_training": false}
```

For `final_confirmation`, `no_further_training` must be true. Neither probe
role can be optimized. No textbook development/final payload is needed.

The optional actual-displacement prediction is
`g_Q dot (delta_control - delta_family)`, using the true parameter changes
from common AdamW initialization and a whole-proof bits gradient at the base
policy. It is a first-order approximation, not an exact optimizer model.
Conditional NLL is an operational coding descriptor. This code does not claim
to measure intrinsic Krajicek complexity or EDL from replayed SFT examples.

Run `python code/test_readiness.py` for twelve semantic CPU checks. Separate
CLI smoke receipts are in `code/tests/cli_smoke_receipt.json`; all smoke
outputs use invented proofs/fresh synthetic records and a tiny random policy,
so they are readiness evidence rather than scientific performance results.

Population amendment and main-bank preparation
---------------------------------------------

After the immutable collector exits, root can run:

```
python code/release_populations.py --input data/populations_v1 \
  --output data/released_v1 --amendment configs/collection_amendment_v1.json
```

This validates the original final collector manifest, input/source hashes,
every actual certificate and provenance count, all twelve 64-source primary
quotas, adaptation/retention counts, and each probe stratum. It repeats the
blind textbook guard and baseline exposure/context audit. It preserves the
original complete/incomplete collection status and creates a new primary
release. Available ORE records receive role `diagnostic_ore` in a separate
file; primary training rejects that role. Their contexts still belong to the
original candidate allocation, so they are not a separate held-out set.

When root has frozen three selected family identities:

```
python code/make_bank.py --data data/released_v1/candidate.jsonl \
  --mode selected --families FAMILY1 FAMILY2 FAMILY3 --seed 20260910 \
  --output data/banks/selected_v1
python code/make_bank.py --data data/released_v1/candidate.jsonl \
  --mode uniform --reference-bank data/banks/selected_v1/bank.jsonl \
  --seed 20260910 --output data/banks/uniform_v1
```

Each bank has 192 distinct sources. The selected bank includes all 64 sources
per declared family. The uniform bank samples without replacement from the
entire released primary pool, conditional on exactly matching the selected
bank's full source-length histogram. It may overlap the selected bank; the
manifest records that overlap, exact source identities, source/context
diversity, scope statistics and token-length distributions. Token lengths are
reported, not silently matched. Both builders refuse nonempty outputs, bind
their inputs and seed, and read no model-outcome files. Family choice remains
part of root's outcome-informed selection procedure.

`python code/test_population_release_and_banks.py` checks these integrity
conditions using the tiny smoke corpus; eight tests pass. The implementation
agent has not run a production population release or built a production bank.

Prepared oracle-information control
-----------------------------------

`oracle_sft.py` is a separate, explicit **cap12 direct-supervision** arm. It
does not change the cap6 loaders. It consumes the union of primary candidate
certificates (≤6 lines) and selection-probe certificates (≤12), with their
original roles retained in `oracle_source_index.jsonl`. It binds those files,
retention data and the adapted starting checkpoint to the actual selector's
calibration input manifest. Confirmation, final-confirmation, diagnostic and
textbook training inputs are rejected. Its former selection probes are now
training data; evaluation belongs only on further frozen confirmation data.

```
python code/oracle_sft.py --checkpoint ADAPTED_CHECKPOINT \
  --candidates data/released_v1/candidate.jsonl \
  --selection data/released_v1/selection.jsonl \
  --retention data/released_v1/retention.jsonl \
  --selector-manifest CALIBRATION_DIRECTORY/manifest.json \
  --flop-budget TOTAL_PROXY_BUDGET --budget-rounding floor \
  --fixed-pad 1024 --batch-size 32 --seed 20260910 \
  --output NEW_PLAN_DIRECTORY --plan-only
```

The default budget policy is `exact`: the requested integer budget must be a
multiple of one full padded update's dense proxy. An explicitly chosen
`floor` policy derives the largest whole-update count below the requested
budget and records the unused remainder. `--plan-only` loads weights on CPU,
validates the already released training inputs and writes the exact schedule;
it performs no optimizer update. A root-authorized training launch uses a new
empty output, removes `--plan-only`, and adds `--device cuda` as appropriate.

The initial control samples uniformly over distinct candidate+selection
certificates with replacement, while reserving the same fixed retention
half-batch as calibration. The larger candidate population is not silently
reweighted. Source-role/reference-length exposure counts, response tokens,
unique exposed sources and exact sample identities are saved, together with
the acquired certificate/token inventory and full fixed-shape compute ledger.
Selection-probe losses are never reported by this trainer as held-out results.
The five CPU tests in `test_oracle_sft.py` cover source gates, preserved roles,
exact/floor budget arithmetic and a tiny-policy update with matching proxy and
exposure accounting. No production oracle-control job has been launched by
the implementation agent.

Confirmation scoring and fixed-bank sampling
-------------------------------------------

`score_only.py --checkpoints CHECKPOINT... --data CONFIRMATION_JSONL
--release ROOT_RELEASE_JSON --output NEW_DIRECTORY` scores complete reference
proofs from empty starts without generation. The release must bind every
checkpoint and the data hash, freeze selector choices, and certify
`no_further_selector_changes`; final-confirmation additionally requires
`no_further_training`. Scores retain per-proof identities and report equal
weighting across the three reference-length strata. The release is checked
before proof payloads are read.

Add `--score-temperature 0.8` for a prospectively labeled supplementary
comparison at the rollout sampling temperature. The default is exactly 1 and
reproduces the original scorer. Temperature is saved in each score, manifest,
summary and scoring ledger. The helper `reference_scoring.py` leaves the
frozen sampler implementation and its reuse hash unchanged. Its CPU tests
compare temperature 1 exactly to the original scorer and temperature 0.8 to
a manual sum of action-token log-softmax probabilities.

The initial/current candidate coding-cost diagnostic is separate:

```
python code/characterize_candidates.py \
  --base-checkpoint results/base_sft_v1/final.pt \
  --adapted-checkpoint results/adapt_v1/final.pt \
  --data data/released_v1/candidate.jsonl --output NEW_CHARACTERIZATION_DIR
```

It accepts only a frozen candidate population or bank, checks every actual
cap6 certificate, and performs no generation or optimization. Both checkpoints
use the same common double-PRF format, including pre-adaptation SFT weights.
It records full-start and default-fragment conditional bits, bits per token,
source/context/provenance identities, per-family summaries and paired changes.
The initial label refers to the pre-adaptation SFT model; the format shift is
explicitly recorded. These are candidate descriptors, not held-out results.

`fragment_experiment.py intervene --sampling uniform-sources` accepts an
immutable `make_bank.py` bank with unequal family counts and samples its
distinct source inventory uniformly. Omit `--families` for this mode. Default
`balanced-families` behavior still requires equal family counts. The actual
family, source-length and context distributions are saved in `mixture.json`.

Separately labeled initial-cut diagnostic
---------------------------------------

The original latest-cut controller remains the default. The optional pilot
uses only frozen candidate certificates, with eight attempts at every
admissible cut, including empty starts. It chooses maximum `k*(8-k)`, breaking
ties toward the earliest cut. If every score is zero, it chooses the earliest
all-success cut, or the latest admissible cut when every draw failed. Prefixes
that already establish the goal remain inadmissible.

```
python code/all_cut_pilot.py fit --checkpoint results/adapt_v1/final.pt \
  --data data/released_v1/candidate.jsonl --seed 20260912 \
  --reuse results/adapt_candidate_full_v1 results/adapt_candidate_reset_v1 \
  --output NEW_FIT_DIRECTORY --plan-only
python code/all_cut_pilot.py audit --checkpoint results/adapt_v1/final.pt \
  --data data/released_v1/candidate.jsonl \
  --initial-cuts FIT_DIRECTORY/initial_cuts.json --seed 20260913 \
  --output NEW_AUDIT_DIRECTORY --plan-only
```

Defaults match the existing candidate diagnostics: fp32, temperature 0.8,
512 sampled tokens and eight attempts per state. Root's actual launch removes
`--plan-only`, specifies the authorized device and uses a new empty output.
Reuse requires exact policy, sampler, state-serialization, model dependency,
temperature and response-budget identities. Every reused state is rebuilt,
its reward rechecked, and its saved behavior probabilities audited again.
Incompatible producer metadata triggers fresh sampling. Reused producer
compute is conservatively charged in full, with separate reuse receipts.

The fitted map is written and hashed before the separate audit. Audit seeds
must differ from all fit and reused-producer seeds; chosen, empty and latest
conditions use separate fresh draws. Their mixed groups, advantage energy,
active action tokens, verifier attempts and compute are reported separately.
Pilot-optimized mixed scores are not validation results.

An explicitly labeled reset follow-up passes
`fragment_experiment.py rl --initial-cuts FIT_DIRECTORY/initial_cuts.json`.
The map must match the starting checkpoint and sampling contract and cover
every training-bank source. All curriculum counters start at zero; the
original empty-start fraction and move thresholds are unchanged. The map's
pilot cost is reported separately from the direct-training compute ceiling.
`test_initial_cuts.py` covers the exact decision rule and contract checks;
the tiny CPU fit, fresh audit and initialized-RL smoke outputs reside in
`code/tests/` and provide implementation evidence only.

Retention-only control
----------------------

`retention_only.py` removes the RL rollout term while preserving the common
adapted initialization, ordered retention population, independent retention
RNG stream, mean per-proof token loss times 0.1, AdamW and gradient clipping.
It requires exactly 512 strictly verified cap6 retention-role sources and
binds checkpoint/data identities and optimizer/schedule settings to a main
RL input manifest. It performs no policy sampling or candidate supervision.

```
python code/retention_only.py --checkpoint results/adapt_v1/final.pt \
  --retention data/released_v1/retention.jsonl \
  --main-manifest results/rl_selected_full_seed1_v1/manifest.json \
  --save-at-steps MAIN_ARM_UPDATES_1 MAIN_ARM_UPDATES_2 \
  --max-flop-proxy 4e15 --output NEW_RETENTION_DIRECTORY --plan-only
```

Root supplies actual completed main-arm update counts. The same trajectory
saves every requested `step_NNNNNN.pt`, with exact retained-example/token
counts, schedule-prefix hashes and actual compute. Those checkpoints are
explicitly labeled **exposure matched only**. The separately labeled total
compute endpoint continues on the same retention stream until the declared
proxy ceiling is crossed, finishes that update and reports the overshoot and
its one-update bound. A step safety ceiling that stops first is labeled as
such. A requested exposure checkpoint beyond either stopping ceiling fails
before training. The CPU plan computes the exact deterministic schedule and
proxy endpoint, then training checks actual counts and compute against it.

For a root-authorized run, remove `--plan-only`, supply the authorized device
and use a new empty output. This is a removal-of-RL-term procedure control;
Adam state, clipping and parameter drift prevent interpreting it as an
additive gradient decomposition. `test_retention_only.py` checks role/cap and
main-schedule gates, checkpoint/compute labels and a five-update equivalence:
zero-advantage main updates and pure retention match every parameter and Adam
state exactly on CPU at the same retained batches.

Failed-rollout local-certificate diagnostic
------------------------------------------

`mine_failed_prefixes.py` reads at most the first 1,000 completed updates of a
full-start candidate RL run. It only considers failed attempts with an
unambiguous sampled FINAL boundary. It stops checking a prefix at the first
invalid inference and retains only complete prefixes of at most six lines
whose last conclusion is at top level with all scopes closed. The new goal
must differ from the original target and every original premise. All original
premises, line indices, scopes and dependencies remain intact; the complete
local certificate plus QED must pass both independent and strict verifiers.

```
python code/mine_failed_prefixes.py --run results/rl_selected_full_seed1_v1 \
  --populations data/released_v1 \
  --baseline-data ../r1-sprint-20260909-1625/data/sft_cap6.jsonl \
  --guard-json ../r1-cot-20260909/data/all_textbook_guards.json \
  --through-step 1000 --output NEW_DIAGNOSTIC_DIRECTORY
```

The frozen release binds training-population, baseline and blind-guard hashes.
Novel canonical theorems must belong to the same public candidate context
partition and pass blind textbook guards; existing baseline, adaptation,
retention and primary-candidate theorems are excluded. Confirmation and
textbook payloads are never parsed. Canonical duplicates within the mined
stream are counted and discarded. Each retained item stores the parent source,
step, action hash, raw failed attempt, full failed-attempt length/reason and
complete retained certificate. Output role `diagnostic_failed_rollout` is
rejected by the training loader. This CPU extraction adds no model rollout or
optimizer update; the reused parent rollout costs remain in the main ledger.

The corresponding reference-only acquisition control is:

```
python code/enumerate_reference_prefixes.py \
  --candidates data/released_v1/candidate.jsonl \
  --mining-directory code/diagnostics/failed_rollout_first1000_v1 \
  --output NEW_REFERENCE_PREFIX_DIRECTORY
```

This enumerates all 768 already supplied cap6 candidate references with the
same closed-prefix rule, public candidate-context partition, blind guards and
known complete-endpoint exclusion. Closed-source proofs are scanned even when
their new local conclusions change the original closed-formula context hash.
It saves diagnostic-only canonical representatives plus every admissible
prefix occurrence, with complete parent reference/provenance. The overlap
report compares mined endpoints under the standard atom-renaming and premise-
permutation equivalence. Exact-prefix matching allows atom renaming while
preserving ordered premises, inference order, scopes, rule labels and
citations. It uses all occurrences before endpoint deduplication; choosing one
representative must not hide alternative already supplied proofs. Neither
selection nor confirmation sources are enumerated, and no model generation or
training is performed.

Frozen RL-calibration selection scoring
--------------------------------------

`score_rl_selection.py` requires the exact 48 short/long endpoint paths in the
frozen RL-calibration pipeline, together with its frozen design. It accepts
only the selection hash in the declared released population manifest. Before
parsing proof payloads it binds every checkpoint, completed producer result,
source input, frozen program and dependency. Each producer's positive step
costs, running cumulative costs, one-step bound, completed ledger, first
budget crossing and reported overshoot must reconcile. Family/control labels
and short/long milestones must agree with the declared source banks and
training commands.

```
python code/score_rl_selection.py \
  --pipeline configs/pipeline_rl_calibration_v1.json \
  --design configs/rl_calibration_design_v1.json \
  --selection data/released_v1/selection.jsonl \
  --base-checkpoint results/adapt_v1/final.pt \
  --output NEW_SELECTION_SCORE_DIRECTORY \
  --actual-displacement --device cpu --precision fp32
```

The manager may replace `--device cpu` with the authorized launch device after
freezing the code. All 48 producers must have completed. Scores use the common
empty-prefix input format and the complete reference response at fixed T=1;
there is no sampling or parameter update. The base policy is scored once in
addition to the declared endpoints. Outputs include per-source bits, family,
reference-length stratum, context and theorem identities, plus per-checkpoint
summaries and compute ledgers. `result.json` contains a `checkpoints` array
with `family`, `arm`, `horizon` and `scores_directory` fields. Ranking is left
to the manager's frozen analysis.

With `--actual-displacement`, one gradient of the equal-length-stratum mean
whole-reference bits is computed at the frozen adapted policy, saved with its
identities and charged once. Its dot product with each actual endpoint-minus-
base parameter displacement and that displacement's norm are then reported.
This is a retrospective projection through the realized training trajectory;
it is not a cheap predictor available before that trajectory. Gradient
extraction never calls an optimizer and clears model gradients afterward.
The CPU tests verify release and hash gates, endpoint/budget tampering, and
masked bits, length weighting and the displacement statistic against an
analytic constant-logit example.

Projected raw-LOO diagnostic
---------------------------

`projected_loo.py` reuses the original adapted candidate full-start groups and
the independently seeded `full` component of the difficulty audit. Each draw
set must contain exactly eight attempts for every one of the 768 released
candidate states, at the common adapted checkpoint, temperature 0.8 and a
complete 512-action budget. The full component is included regardless of the
cut chosen by the fitting pilot. Both producers must be complete before use.

```
python code/projected_loo.py \
  --pipeline configs/pipeline_rl_calibration_v1.json \
  --design configs/rl_calibration_design_v1.json \
  --checkpoint results/adapt_v1/final.pt \
  --candidates data/released_v1/candidate.jsonl \
  --selection data/released_v1/selection.jsonl \
  --banks-manifest data/rl_calibration_banks_v1/manifest.json \
  --original results/adapt_candidate_full_v1 \
  --audit results/difficulty_audit_v1 \
  --output NEW_PROJECTED_LOO_DIRECTORY --device cpu --precision fp32
```

The manager selects an authorized launch device after freezing the code.
The command binds source, code, checkpoint, sampler, state, action and behavior
identities; independently replays strict rewards; and checks saved behavior
likelihoods against the unchanged model. It retains sampled termination and
every other sampled action, including a sampled PAD ID, while assigning zero
direct weight to supplied input and batch padding. A stopped sample cannot be
extended or silently shortened.

One frozen gradient of the equal-length-stratum mean whole-reference selection
bits is computed at T=1. With identity preconditioning, each source estimate
is `D = g_Q · sum_i A_i grad(log P_i) / 8`, where
`A_i = (8 R_i - sum_j R_j) / 7`. This is the sample reward/score covariance,
using the complete response log probability at T=0.8. The command saves D and
D/512 separately. The raw-ascent family benefit is
`(mean_control_D - mean_family_D) / 512`; neither Adam nor a learning rate is
implied. All-equal groups have algebraically zero estimates; they still pass
reward and behavior checks, and their zero estimate does not establish zero
true gradient or variance. No per-response full gradients or Fisher estimates
are stored. The policy is checked for exact parameter equality afterward.

Original and fresh source estimates are separate JSONL files. Each source is
computed once per draw set and reused across all 24 original ordered 64-source
banks, whose matching receipts are checked. The result retains both sets of
family/control contrasts and their two-draw average. Optional within-source
resampling synchronizes each resampled value across every bank membership.
Its ranges are **descriptive conditional resampling sensitivity**, not
calibrated Monte Carlo confidence intervals: two groups can miss rare
successes entirely. No LOO group is pooled or recomputed during resampling.
Paired-difference variance estimates state their independent-draw assumptions
and the two-replicate limitation. These statistics do not change any selector.

The ledger charges the query backward pass once, behavior scoring for every
reused action, each mixed group's teacher backward pass and parameter dot
arithmetic. Previously charged rollout costs are receipts, separated from new
derivative costs to prevent double counting. No new rollout, parameter update
or final-population access occurs. CPU tests enumerate the exact Bernoulli
group distribution to check unbiased covariance, sign, the 1/8 factor and the
fixed 512 denominator; they also check variable response lengths, sampled PAD
actions, all-equal groups, source/state/reward gates and shared-source
resampling cancellation.
