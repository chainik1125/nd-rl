---
written_on: 2026-10-06
written_by: agent:codex
---

# Allocation research log

## 2026-10-06 01:47 UTC — Interface and likelihood audit

- Read repository instructions, kickoff, proposal, sprint skill and all four references.
- Audited old local proof-state read-only and corrected checkpoint/source lineage using selection's clean checkout. Actual base is six-layer ALiBiGPT, not old GPT/RoPE.
- Inference leakage risks: LEANREJ/LEANPARSE wrappers, ended_reason, correctness metadata. Raw emitted introduction names can conflict with environment-renamed references. Coherent effective history is primary input; raw emissions are scored separately.
- Implemented pure protocol environment and factorized base-copy controller. Reviewer identified missing inferred forced termination and alternating state enforcement; fixed with regression tests.
- Nine protocol/likelihood tests pass. Targeted Ruff and shared Pyright pass. Repository-wide Ruff fails on thousands of preexisting old-source issues; preserved that source.
- GPU2 synthetic actual checkpoint forward/backward:512/1024/2048 token histories supported, warm2048 ~0.013s, peak0.584GB. Uniform heads output .5 ACCEPT and uniform available selection. Actual mixed padded vs separate forwards differ by at most5.96e-8.
- Wrote complete-episode outcome-only REINFORCE runner, fixed-k chooser training control, paired forced-k evaluation, and exact frozen-order selection-relative continuation-value diagnostic. No training run launched before pool audit.
- Future-candidate length must not determine legal actions. Changed to fixed declared prospective per-attempt bound. Oversize pools fail explicitly; no theorem dropping.
- Spend: $0. GPU2 released after short smoke. Selection generates100theorem dev pilot; root owns separate contextual-generation existence audit on GPU3.
- Decision: continue with development-only occupancy audit and controller smoke, then live same-model generation/update using raw generation likelihoods plus decision likelihoods.

## 2026-10-06 01:52 UTC — Debug run quarantined; live likelihood smoke passed

- Two-update controller debug run finished on32train theorem pools. Runtime was seconds, not a substantive learning test.16episodes contained one forced8-attempt selection; mean attempted2.81proofs. No learning inference is warranted.
- Selection subsequently identified validation36-equivalent IDs in the fresh_v1 source. The debug pool includes `da42d4814cf3a388b9fc`; all artifacts in `artifacts/learned_compute_allocation_20261006/allocation/smoke_cost_s11/` are quarantined. Checkpoints will never initialize reported runs or enter evidence claims. Training paused pending filtered successor.
- Implemented live generation/replay module. A no-optimizer-update first-attempt plumbing smoke generated17actions/395tokens, completed in1.35s, with zero raw replay-likelihood discrepancy and nonzero trunk generation gradient. This checks mechanics, not research performance.
- Every future run will snapshot evolving scripts and save hashes, PID, full command, start time, base/pool hashes and immutable checkpoints. Source/helper hashes will be included in live runner.
- Spend remains$0; no GPU2 process left running.

## 2026-10-06 01:59 UTC — Safe same-model joint updates and held-out baseline

- Safe successor `fresh_v1_safe` excludes protected source theorem equivalences and contains1578train/324dev/293test. Test outcomes remain unopened.
- Joint safe cost.03smoke: two updates with four complete episodes/update took5.10/5.22s. Both generation-head and stopping-head gradients were nonzero; all replay-token likelihood discrepancies were exactly zero. Eight episodes contained no hard-forced decisions. Raw records and immutable snapshots live in `artifacts/learned_compute_allocation_20261006/allocation/live_safe_smoke_cost_s11/`. This is a mechanics check, not performance evidence.
- Independent fixed2 no-cost update smoke verified the acceptance-head None-gradient guard. Two correct outcomes gave zero centered advantage and zero gradient, as the objective requires. All fixed decisions were interventions with no stop loss.
- Eleven regression tests now pass, including all sampled PAD/EOS generation tokens and entropy at a fixed diagnostic temperature. Targeted Ruff and shared Pyright pass.
- Untrained raw-history chooser on100devT1.1pools confirms all8attempt histories fit declared16384context. Uniform heads provide only a protocol baseline; a single draw per theorem is not a reliable learned-effect comparison.
- Actual live unadapted forced1/2/4 evaluation on48safe development theorems is active onGPU2, session55204, output `allocation/unadapted_live_dev48_s101/`. No updates; original r16 with uniform control heads. This separates interface/context shift from future learning.
- Selection is constructing a deeper ORE challenge because base broad-theorem success and a syntax-only chooser leave little semantic headroom. Substantive controllers wait for the challenge audit and frozen protocol.
- Elapsed20min / remaining9h40min. Additional spend$0. Other sprint devices are coordinated centrally; unrelated devices preserved.

## 2026-10-06 02:27 UTC — Context shift and main protocol

- Root's matched broad-development generation probe exposed severe new-interface failure: original base180/192 correct first proofs, one prior effective proof5/192, four prior proofs0/192. EOS alone retained177/192; EOS plus full proof history still failed. These are initialization-format effects, not learned stopping evidence.
- Outcome-blind format adaptation uses safe train raw actions and public effective histories without checker verdicts, stop labels, selector weights, or test rows. Initial200-update adaptation restored history generation but reduced clean accuracy to156/192. Low token KL does not establish proof retention.
- Main distribution froze at02:06:47UTC:1600train/320dev/400test, balanced forward-deep and ORE2, distinct structural groups and protected-exposure exclusions. Test outcomes remain unopened.
- Collected96main-train theorems with five attempts each for format adaptation. Revised500-update replay used clean-state weight4 and first-action oversampling25%; actual proof gates remain required before substantive joint RL.
- Controller observations now include full public Env.text transitions, including implicit ORE binders and closures; generator histories remain effective actions. Live MAIN transcript exactly matched independent selection renderer.
- Cost hooks count actual padded model forwards, forward tokens, and attention cells separately from diagnostic replay. Twelve tests pass. Separate head learning rates and optional frozen-initialization logprob diagnostics added; first-attempt checker judgments are post-episode diagnostics, reward remains selected-proof outcome.
- Matched FP32 live fixed1 evaluation running on64balanced MAIN development theorems, two draws each. Main training waits for interface gates. Spend remains$0.

## 2026-10-06 02:36 UTC — FP32 format gate and exact path evaluation

- Matched MAIN64 balanced-development live FP32 fixed1 draws (two/theorem, seed52006): base81/128 correct, revised500-step format adapter74/128. Paired difference−5.47pp; theorem-cluster bootstrap95% interval−16.4pp to+6.25pp. This is an exploratory retention gate, not formal noninferiority evidence.
- A further independent outcome-blind adapter completed1000steps from r16: LR1e-5, clean-token weight8, first-action mixture50%, same96train×5 pool. Actual development generation checks are pending; KL is not a substitute.
- New two-update safe-broad protocol smoke validates separate trunk/head AdamW rates and frozen-initialization sampled-logprob diagnostics. Both generation and stopping gradients were nonzero; behavior replay error0. Default safe torch.load and strict Controller.load_state_dict reload passed after stringifying TorchVersion metadata. These artifacts remain protocol-only.
- New shared forced-eight-path evaluator integrates stopping and selection probabilities exactly, checks all proofs only after the path, and saves public histories separately from private labels. Expected policy work weights each measured prefix by reach probability; benchmark collection expense is reported separately. Synthetic enumerated expectations and actual2theorem×8attempt smoke passed. Fourteen tests pass; targeted Ruff passes.
- Failure/truncation regression distinguishes partial canonicalization on failed EOS actions from raw truncated output. Public transition view agrees with the definitive renderer; raw and effective rendered texts are retained separately. Spend remains$0.

## 2026-10-06 03:07 UTC — Optimizer pilot and primary method freeze

- Outcome-blind initialization selected the lower-rate 1,000-step adapter after exploratory first-proof retention and full-path generation audits; mean eight-attempt validity was0.3789 versus0.21875 for the midpoint. Full-history quality still decays substantially. No further initial format grid is planned.
- Matched no-cost optimizer pilots trained50 updates of eight episodes with trunkLR1e-5 and headLR1e-4/1e-3. Theorem scheduling and episode seeds were separated from variable rollout consumption; actual source/base/data hashes and schedules matched.
- DEV32 primary categorical correctness was0.66468 at1.921 attempts for headLR1e-4 and0.68666 at1.219 attempts for headLR1e-3. The latter rate was selected by the declared development objective. Both trail or match their own trained-generator fixed1 baseline, so this establishes no allocation advantage.
- A latent truncated-action PAD retention mismatch was corrected only after both pilots finished. Across800 completed episodes, no observed EOS-missing raw action differed from its old effective representation; zero continuation histories changed. New main protocol explicitly retains raw truncated tokens.
- Main direct joint tests are now frozen: three independent RL seeds101/102/103, cost0/.03,200 updates×8 episodes, shared outcome-blind format initialization, fresh uniform heads, and checkpoints at100/200 completed updates. Native expected utility on DEV32 selects checkpoints; ties use fewer attempts then earlier step.
- Final TEST evaluation prespecifies two full paths per theorem at seed62006. The exact checkpoint/base/inference and test-data hashes must be registered before evaluation; guards and counter-hazard provenance checks have regression coverage. Legacy prototype evaluators are DEV-only.
- Main seed101 no-cost and cost conditions are active on GPUs2/3. Source snapshots are immutable. New checkpoint filenames report completed updates, avoiding an off-by-one ambiguity. Fifteen tests and targeted Ruff pass; spend remains$0.

## 2026-10-06 03:23 UTC — Replicates, supplementary control, and evaluation binding

- No-cost seed 102 started on GPU 4 after the selector owner explicitly released it and a fresh occupancy check found it idle. Seed 101 remains on GPU 2; cost arms are coordinated by root on GPU 3. All active main runs retain the same immutable training implementation and independent rollout randomness.
- Prespecified a supplementary independently trained fixed-four-attempt control: seed 101, 200 updates of eight episodes, zero cost, identical format initialization and optimizer, and terminal selection/generation gradients without stopping gradients. Its completed-step 100/200 checkpoint uses the same development correctness selection rule. The launch script is ready for an available coordinated GPU.
- Test registration now binds six behavior files, the nine source dependencies, model/base hashes, exact test dataset hash, and the prespecified seed 62006 with two paths per theorem. Older pilot evaluations remain development-only when their snapshots lack required hashes.
- Added counter-policy reanalysis of saved paths with exact theorem/repeat alignment and model/interface provenance checks. This needs no new generation. Fifteen regression tests and targeted lint pass; spend remains $0.

## 2026-10-06 03:53 UTC — First primary development selection

- No-cost seed 101 completed 200 updates: 2,170.56 summed monotonic update seconds, mean 1.456 attempts, and zero replay likelihood discrepancy. Seed 102 also completed 200 updates: 1,709.20 update seconds and mean 1.263 attempts. These are training diagnostics, not held-out performance claims.
- The frozen DEV32 rule selected seed 101 step 200: categorical expected correctness 0.77534 at 1.717 attempts versus step 100 at 0.74925 and 1.217 attempts. Its own fixed-first baseline is 0.78125, so this development result does not establish an allocation benefit. Greedy index always selects the first proof on this panel.
- Development-fitted count-only stopping scores 0.77485 at exactly the same mean attempt count. The prescribed context grid selects 512 tokens, scoring 0.77772 at 2.125 attempts. Fixed-budget mixtures are matched on expected attempts and measured forward-token work; their weights remain frozen for test.
- Fitters and applications validate actual current implementation hashes against immutable path snapshots. Test applications additionally require registered fitting-file and postprocessing-code hashes before private labels open. Exact fitter/application source copies are stored with final baseline artifacts.
- No-cost seed 103 is training on GPU 2. The supplementary fixed-four control is training on GPU 4; its initial 17 updates average 27.79 seconds, with zero stopping gradients and exact replay likelihoods. Test outcomes remain unopened. Spend remains $0.

## 2026-10-06 04:24 UTC — Statistical freeze and remaining development evaluations

- Root completed development checkpoint selection for all three cost-aware seeds; all selected step 200. Native expected correctness is approximately 0.7813, versus 0.78125 for each own-generator first-attempt baseline, with expected attempts 1.017–1.031. These development results do not establish a useful allocation gain.
- The guarded final aggregator now requires all primary seeds and prespecified native/greedy readouts, verifies primary model/data/code registration before reading primary outcomes and validates registered fitting artifacts before including derived readouts, and reports paired family-stratified theorem bootstrap intervals conditional on the fitted models. Reviewer independently checked the statistical source and 21 tests; a subsequent queue isolation regression brings the targeted suite to 22. No test outcomes have been opened.
- No-cost seed 103 is still training and currently uses more attempts than the cost-aware runs. The fixed-four control remains active. Root will evaluate no-cost seed 102 on GPU 1 after the independent frozen-backbone feature extraction; our GPU 2 queue will evaluate seed 103 after training finishes.
- A copied development launcher initially retained a cost artifact path and exited before GPU inference. Its temporary overwrite of the root launcher snapshot was flagged immediately; root restored the exact originally launched source. Primary results/checkpoints and evaluator snapshots were unaffected. The corrected helper constructs a complete no-cost-only output plan before any write, rejects resolved path escapes, and uses only that validated plan. Failed/waiting helper logs are preserved; they consumed no GPU inference work.

## 2026-10-06 05:11 UTC — Primary checkpoint choices and final run commitments

- All six primary cost/no-cost arms completed 200 updates and selected step 200 by the frozen DEV32 rule. No-cost seeds 102 and 103 score 0.74949 and 0.74717, respectively, versus their own first-attempt baseline of 0.75; expected attempts are 1.197 and 3.771. Together with seed 101, these development outcomes show substantial stopping variation without an allocation advantage.
- The supplementary fixed-one control selected step 100 by an earliest-checkpoint tie: both checkpoints score 0.71875 first-attempt correctness. The fixed-four control completed 200 updates and exactly 6,400 training attempts, with zero stopping-head gradients and replay discrepancy. Its checkpoint evaluations are running; the current-protocol format-only audit follows them.
- Nine-model MAIN and historical run manifests now commit exact output paths and registered counter/budget readout destinations before any test outcomes. The freeze-entry builder binds both manifest hashes. Root alone will compose and seal the complete registries after the two remaining development inputs exist.
- The reviewed launch helper refuses altered model/source/data/statistical/fitting registrations, occupied GPUs, and existing output paths. Prepared queues allocate GPUs 2 and 4 separately and enforce completion of all nine MAIN jobs before any historical generation. No final test job has launched. The targeted protocol/controller suite passes all 26 tests; spend remains $0.

## 2026-10-06 05:17 UTC — Complete pre-test allocation registration

- The fixed-four development rule selected step 200: four-attempt categorical correctness 0.71751, versus 0.71547 at step 100 and own first-attempt correctness 0.71875. The format-only current-protocol DEV32 audit completed with first-attempt correctness 0.625; its untrained categorical chooser yields 0.375 at four attempts and 0.24219 at eight. These controls remain separate from the six primary RL arms.
- The complete nine-model allocation entry file is now ready for root's seal and binds both planned run manifests. All current choices were made on development data; no fresh or historical final-test outcomes have been opened. Root retains final composition and launch coordination.


## 2026-10-06 05:34 UTC — Registered final live evaluation begins

- Root sealed MAIN and historical registries at 05:33:52 UTC after the independent public pools completed. The exact registry hashes matched before launch; the allocation entry and planned manifest commitments remain unchanged.
- GPU 2 now evaluates cost seed 101, then no-cost seed 102, then format-only. GPU 4 evaluates no-cost seed 101, then no-cost seed 103, then the fixed-four control. Each per-model launch captures immutable snapshots and monotonic phase timings; both queues require all nine MAIN evaluations to complete before their historical jobs.
- First launches passed preflight and are running. No frozen method or registry is modified in response to final outcomes. Results are not yet available.

## 2026-10-06 06:20 UTC — Explicit numerical and operational addenda

- A DEV-only reproducer found 79 of 2,112 integrated rows rejected by the original probability-domain guard: FP32 selector sums yield correctness up to 1 + 8.62e-8, within single-precision roundoff. The reviewed analysis-only successor changes only the correctness-domain admission tolerance from 1e-8 to 1e-6. It does not clip, renormalize, or change estimates; all 2,033 originally admitted metric arrays remain bitwise identical. Other guards and bootstrap settings remain unchanged.
- Root approved a separate analysis addendum binding original registries/entries/manifests and exact successor/helper hashes. Raw generation configurations retain their original registry hashes; original statistical source files are preserved. The first aggregation watcher was stopped before any phases. The approved versioned CPU watcher now waits for all nine MAIN and all nine historical runs, producing explicitly named fp32_v2 summaries.
- The numerical correction used no completed JOINT-LIVE TEST integrated outcomes. Earlier frozen-pool selection/staged TEST results were already known in the session, and two causal TEST conditions had automatically completed analysis; this disclosure is recorded in the approved addendum.
- Foreign-user jobs began sharing claimed GPUs 1–3 while current live jobs continued. Root approved an independent operational addendum: successor launchers require at least 64 GiB free memory, reject same-UID model collisions, preserve exact scientific commands and registry bindings, and record foreign occupancy and actual source hashes. No foreign process is modified. The reviewed live successor is available only for remaining jobs if original admission fails; current children and completed outputs remain intact. Co-located timings are not isolated latency measurements.

## 2026-10-06 07:05 UTC — Training coverage and measured evaluation throughput

- The training-only schedule audit finds 1,002 / 987 / 1,013 unique training theorems in the 1,600 replacement-sampled episodes for seeds 101 / 102 / 103. All cost/no-cost theorem IDs and episode RNG seeds match at every update/slot; both fixed-count controls share the complete seed-101 schedule. The selected fixed-one checkpoint covers only its first 800 episodes and 622 unique theorems; all other selected RL models use 1,600 episodes. Across the three primary schedules, 1,520 of 1,600 training theorems appear. Input and audit source hashes are saved in `artifacts/learned_compute_allocation_20261006/allocation/training_theorem_coverage.json`.
- Three MAIN live jobs are complete without reading their integrated quality outcomes: no-cost seed 101 takes 3,648.92 seconds for generation/checking, cost seed 101 takes 4,053.21 seconds, and cost seed 102 takes 4,097.01 seconds. Remaining original queues continue successfully after idle-GPU admission; no rerun or generation change was needed.
- Root explicitly released GPU 1 at 06:55 after all registered causal conditions completed. Cost seed 103 starts there through the reviewed operational successor, recording its source/addendum identities and a fresh empty-GPU snapshot. GPU 3 is evaluating the fixed-one control. The all-nine MAIN barrier and unchanged planned raw paths remain in force; the approved numerical aggregation successor still waits at that barrier.

## 2026-10-06 09:17 UTC — Fresh joint evaluation complete

- All nine MAIN launch ledgers completed successfully before the verified 09:03:25 UTC barrier. Registry, entry, and both planned manifest byte hashes matched. Only then did the approved FP32 analysis successor read integrated joint-live outcomes; its MAIN aggregate completed in 10.33 CPU-process seconds.
- Cost-aware native correctness is 85.319% at 1.021 expected attempts versus its own first-proof 85.333%. Correctness-only is 83.974% at 2.325 attempts versus first-proof 84.333%. Tiny positive count-only contrasts and the positive cost DEV-work-mixture comparison are retained; no-cost loses accuracy to its DEV-work mixture despite more TEST work. Greedy-index correctness matches first-proof accuracy within numerical precision, without establishing discrete always-first behavior.
- Presentation-only v2/v3 figures display saved paired contrasts, without refitting statistics or changing frozen generation/analysis. The v3 caption defines rewards, categorical readout, seed points, theorem-bootstrap intervals and charged expected input-token work. Original figures remain preserved.
- Historical live queues started after the same all-nine MAIN barrier. GPU3 was explicitly released by selection at 09:09:29 and checked idle before the registered cost102 → fixed-one historical pair launched. Observed historical model process durations are 337–435 seconds; the historical outcome barrier remains in force. Historical known-exposure/source-role limitations are unchanged.

## 2026-10-06 09:31 UTC — Historical completion and reporting

- All nine historical live launch ledgers completed with every phase exit0. The frozen historical run-manifest bytes matched before integrated outcomes were read; the approved aggregation watcher completed both registered summaries and exited0. Final GPU3 was checked idle and released; no further model work is planned.
- Historical native correctness is 41.403% with 1.039 attempts for cost-aware and 40.951% with 2.372 attempts for correctness-only, compared with 41.435% for each own first-proof baseline. Format-only first-proof correctness is 41.667%; fresh-task generation improvements do not establish historical transfer. Original 58/14 source roles and at least one known pretraining exposure remain explicit.
- DEV-calibrated work mixtures are applied unchanged, rather than refitted historically. Cost-aware exceeds its work-mixture correctness by 0.1104 percentage points but processes 144.66 more input tokens; correctness-only is below its work mixture by 0.1350 points despite 447.87 more tokens. Historical calibration mismatch is therefore reported directly.
- All three reader-facing allocation reports have current MAIN/historical results and separately labeled figures/CSV/provenance. MAIN independent arithmetic/binding/Lean audit and the v3 presentation audit are accepted; final historical audit remains pending. v3 summary PNG/PDF replicas now exactly copy artifact bytes, preserving the slightly different initial summary renders and documenting all hashes in a copy manifest.
- Allocation-owned measured device-phase wall lower bound is 51,248.754 seconds: development 14,732.701; MAIN 33,305.386; historical 3,210.667. Three early debug jobs have missing timing. Root-owned jobs and CPU-only aggregation/reanalysis/waits are excluded. This ledger is not CUDA kernel time or an isolated-latency benchmark; foreign workload colocation and diagnostic timing scopes remain caveats.
