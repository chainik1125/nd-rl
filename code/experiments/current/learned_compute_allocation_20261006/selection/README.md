---
written_on: 2026-10-06
written_by: agent:codex
status: completed
---

# Selection from eight frozen proof-state attempts

Experiment 1 of the [proposal](../../../../../proposals/learned-compute-allocation/README.md) trains the actual `state_env_october_r16` model to recognize useful attempts. The generator is fixed at checkpoint seed 0; chooser training seeds are independently initialized copies. [Method freeze](method_freeze.md) records the primary pooled chooser and factorized actual-model control. [Human-readable summary](../../../../../experiment-summaries/2026-10-06-learned-compute-allocation/selection/summary.md) separates evidence from open questions.

## Source and model provenance

The workspace branch is `dmitry/main`, sprint base `315f36266bb4dd7feeff87228cb2662f8a36badc`. The clean upstream checkout lives under `artifacts/learned_compute_allocation_20261006/source/nd-takehome-dan`, pinned to `4dcd0e4f336b3bf0ee24230dc5335ba3dba8969c`. Existing untracked `proof-state/` files were inspected read-only and never modified.

The base checkpoint came from `hf://buckets/dan-pandori/nd-rl/rl-continue/ckpts/rc/ladder/la_T1_best12_s0_r16.pt`, locally stored in the sprint `checkpoints/` folder, SHA256 `ba5d03de5dba890be643da82f9edca826d254c862cdcd33c221954b70918da24`. It has 9,560,832 parameters: six layers, width 384, eight heads, MLP width 1280, four ALiBi/four NoPE heads, Peri-LN and the 111-symbol `lean_staten` tokenizer. Its configuration says 1024 tokens. The primary 8192-token chooser explicitly extrapolates beyond training support.

Upstream `model.save_ckpt` uploads by default. All experiment jobs set `ND_OFFLINE=1`; selector saves use a local-only `torch.save` wrapper. Metadata is JSON-normalized to remain compatible with safe weights-only reload.

## Frozen theorem distribution

`data/main_v1/manifest.json` freezes a 50/50 mixture of broad forward-generated formulas and two-leaf compound-formula disjunction elimination, with 1600 train, 320 development and 400 test theorems. All 2320 atom-erased theorem skeletons are distinct across the 1600/320/400 split. Shared skeleton hashes assign splits; canonical atom-bijection/premise-permutation matches to known training and public benchmark theorem metadata are excluded. Constructed references remain private target-validity evidence, never model supervision.

Exposure includes all 155000 cap12 pretraining rows, 4495 expert-iteration targets and the actual 90436-record r16 mixture. `source/actual_r16_exposure_audit.json` finds no final-corpus theorem match and no actual-mixture theorem outside the previously audited exposure union. Protected short/long prompts were never inspected. Earlier fresh-v1 data had two validation36 training matches; its safe successor excluded them before learned training. The main corpus also excludes all previous development pilot cases.

`data.py` records the original easier ladder pilot. `challenge_data.py` records bounded goal-only dead ends and the broader forward panel. `ore_data.py` builds verifier-checked development target references. `main_data.py` prospectively freezes the final distribution before model outcomes. These generator settings and reference lengths do not establish minimum proof complexity. A train/development classical truth-table audit in `data/main_v1/train_dev_classical_shape_audit.json` additionally finds many contradictory forward-family premises and tautological goals; it is descriptive and does not change the frozen corpus.

## Public and private candidate views

`collect.py` samples eight fresh-context attempts at temperature 0.8, at most 256 tokens per action and 48 actions. It retains exact sampled tokens, including genuine sampled PAD tokens and the failing action, before deterministic environment assignment. Behavioral logprob estimates replay exact raw actions under their actual state prefixes at sampling temperature; diagnostic base logprob and entropy use temperature 1. BF16 full-forward replay is not bit-exact cached sampling logits.

Schema3 explicit `effective_tokens` are deterministic normalized action traces. They can omit implicit branch binders and are therefore not always standalone certificates. `render_transitions.py` constructs schema4 `transition_tokens` from every public environment text delta, retaining failed/truncated attempted text. This is the primary chooser view. It reads no private verdict or reference and rejects extra actions after termination. Raw action fields remain immutable for probability auditing.

Separate private files contain strict Lean labels, failure reasons, completion and assembled proof text. Inference cannot read those fields. Public syntax baselines must replay the visible attempted tokens, rather than consume the private completion flag. `audit_views.py` verifies completed literal identity and independently checks every rendered attempted proof with Lean.

## Evidence and execution

Calibration pools live in `pools/dev_pilot_v2`, `dev_temperature1p1_pilot`, `forward_deep_dev_pilot`, `ore2_dev_pilot` and `ore_dev_pilot`. The final two usable panels yielded 301/400 and 281/400 valid attempts; a four-leaf stress stratum yielded only 1/344. Root analysis supplies strong baselines and paired uncertainty independently.

Combined main development files are `pools/main_dev/public_transitions.jsonl` and `private.jsonl`. All 2086 completed rendered proofs exactly match the checked private source. Independent Lean checking of all 2560 attempted renderings accepts the same 1813 original positives, with zero disagreement. Two of 320 complete chooser inputs exceed 8192 and retain uniform fallback.

Training collection is complete in `pools/main_train_forward` and `main_train_ore2`, with combined views in `pools/main_train/`. Of 1600 training pools, 866 have mixed validity, 610 have all eight valid, and 124 have none valid. All 10389 completed transcripts match literally; direct Lean checking agrees on all 12800 attempts. Final MAIN attempts were sampled only after the immutable decisions parent, and evaluated only after public sealing and the final registry. Every run saves source snapshots, hashes, exact command, seed, precision and compute accounting. `train.py` implements both actual-model methods; `evaluate.py` evaluates frozen checkpoints without training inputs. Tiny interface/view smoke results in `selectors/` are descriptive and support no final recognition claim.

All six primary runs finished; `selectors/frozen_primary_dev_choices.json` records chosen epochs and checkpoint hashes. [Secondary critic freeze](bce_method_freeze.md) declares three independent original-checkpoint binary-loss runs, selecting minimum development log loss. `train_bce.py` and its frozen outputs expose canonical per-candidate sigmoid scores for staged stopping; sequential controllers do not inherit these weights.

`evaluate.py` requires a registered frozen test protocol before opening private labels. Its data-independent `evaluation_signature(args)` binds checkpoint, trainer, evaluator, objective, view, precision, context and permutation. `register_signatures.py` prepares `selectors/final_signature_inventory.json` without reading test files; the inventory is not itself an authorized evaluation protocol. `evaluate_live_critic.py` implements the separately declared external critic readout of saved live paths, joining labels by theorem and repeat and consuming only theorem plus public transition tokens.

`analyze_dev_errors.py` preserves all family, validity-count, completion, input-length and selected-position strata for every primary choice. `resource_ledger.py` inventories measured selection-owned wall-time scopes while preserving unknown/failed early runs and CPU-overhead limitations. The independent root owns paired statistical tests and public-feature baselines.

Compute comparisons charge the full-forward likelihood/entropy replay required by the public-feature critic. Its classifier is small, but the complete feature pipeline is not inference-free. Text-only neural readouts omit those diagnostics and pay their own model forwards. Root threshold-policy v3 accounts for required replay plus generator prefill; summed per-candidate counts remain lower bounds rather than exact batched work.

## Registered MAIN results

All 37 MAIN inference jobs completed under `final_evaluation/main_protocol.json`: 27 trained-model/order evaluations, nine original-model controls, and one frozen scalar head. Primary three-seed theorem averages at order 77106 are pooled 0.816667, factorized 0.86125 and BCE 0.847917, versus the frozen public-feature baseline 0.875, random 0.7159375 and oracle 0.895. Every neural method improves over its declared original readout, but fails the strongest-baseline criterion. The factorized control also changes architecture and context burden; it does not isolate a long-context mechanism. The [reader summary](../../../../../experiment-summaries/2026-10-06-learned-compute-allocation/selection/summary.md) gives exact paired uncertainty and all-complete mixed-validity results.

Artifacts, relative to `artifacts/learned_compute_allocation_20261006/`:

- `selectors/final_main/`: all 37 raw inference predictions/configs; `selectors/main_test_queue_v1/`: immutable launch sources, PID, registry hashes and measured per-process durations.
- `analysis/final_main/selection/`: machine-readable three-seed primary metrics, per-theorem records and PNG/SVG. Statistical source: `aggregate_selection.py`.
- `analysis/final_main/frozen_head_paired/` and `frozen_head_thresholds/`: separate secondary head comparison and frozen development threshold evaluation. Sources: `compare_frozen_head.py` and `frozen_head_thresholds.py`.
- `pools/main_test/transition_lean_audit.json`: all 3200 rendered Lean checks agree with private labels; all 2559 completed strings agree literally, zero exclusions.
- `freeze_inputs/selection_v2/`: locked inference/statistical entries and inventories; `final_evaluation/main_protocol.json`: sealed registry.

Historical 72 inference completed after the independently verified all-nine MAIN barrier. All 37 jobs succeeded; all 576 transcript verdicts match direct Lean, and all 290 complete strings match collected certificates. Pooled/factorized/BCE three-seed accuracies are 52.78%/48.15%/54.63%, versus 55.56% features and 62.50% oracle availability; the frozen head scores 56.94%. Historical prompts have known reuse/exposure and are not fresh generalization evidence. Detailed role-weighted intervals and workload-shift limitations are in the reader summary. Original trainers, evaluators, entries and final registries remain unchanged; the historical aggregation required a separately sealed role-schema successor after an authorized-label assertion failure, retained in `analysis/final_historical/selection_launch_v1`. Successor outputs are `analysis/final_historical/selection/`; approved addendum and original/successor hashes are `analysis_successors/historical_selection_roles_v2/`.
