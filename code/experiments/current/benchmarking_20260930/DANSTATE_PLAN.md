---
written_on: 2026-10-01
written_by: agent:claude
---

# Plan: Dan's best-state models through ndbench

Planning by an Opus agent (read-only), amended by the main agent (section A, which overrides the
rest). Coding by Sonnet agents in work packages WP-D0 to WP-D6 (section 7).

## A. Main-agent amendments (override everything below)

- A1. All 12 checkpoint md5s were verified against their `.upload.json` (not only one).
  Checkpoints: `/scratch/cowifornia/nd-rl/bench/dan_ckpts/best-state/{stage1,ladder}/`.
- A2. Dan's Stage-1 training sets and held-out set were found in his public bucket and downloaded to
  `/scratch/cowifornia/nd-rl/bench/dan_ckpts/data/`, md5s matching his summary:
  - `train_depth3_f0_a1.jsonl` (cap 6; 155,000; md5 29276f24...; bucket `state-env/data/p2/`)
  - `train_k12.jsonl` (cap 12; 155,000; md5 800b5486...; bucket `cap-horizon/data/kh/`)
  - `heldout.jsonl` (5,000; md5 ae6daf49...; bucket `state-env/data/p2/`)
  So `dan_tiers()` can be built now (section 3: Stage-1 set + `data/ladder/rl_targets.jsonl` +
  `transfer.jsonl` from the fork clone).
- A3. No GPU job of any kind (including gates V5-V8 and throughput) runs until Leon approves the
  GPU budget. Coders use CPU only; tests use tiny random models.
- A4. Dan's fork clone (`/scratch/cowifornia/nd-rl/bench/nd-takehome-dan`, HEAD 0c492902) is
  read-only for us in practice: never commit, push or write files there. Vendor via
  `git -C <clone> show 0c492902:<path>`.
- A5. Lean: `~/.elan/bin/lean` is 4.34.0 (Dan's version). Use at most 8 Lean worker processes
  (shared 48-CPU host).
- A6. New panel jobs for these models go into `code/ndbench/panel.yaml`'s tier scheme (see
  `code/ndbench/README.md`) or a `schedule_dan.yaml` using the same `tier`/`est_min` fields.

## 0. Verified facts (planner, CPU-only, read-only)

- **Checkpoints**: all 12 are `tok_mode='lean_staten'` (arm SN, environment-assigned names), cfg `{vocab 111, 6x384, 8 heads, d_ff 1280, arch 'best'}`, 9,560,832 params, load with `weights_only=True`. Stage1 and T1 state dicts have identical keys/shapes; the T1 ladder `init`s from the Stage-1 checkpoint (`artifacts/bs/la_T1_best12_s0/args.json`); rel. L2 distance T1 vs stage1 (best12 s0) = 0.26, so interpolation and steering are meaningful.
- **Our `textbook_dev` v1 (58) is prompt-identical to Dan's `data/eval_only/textbook72/textbook_dev.jsonl`** (58/58). Dan's dev58 at k256 Lean (`artifacts/bs/analysis_stdout.txt:13-28`): Fz best12 27/22/21, T1 best12 42/44/42, Fz best6 13/13/15, T1 best6 34/33/26. T1 best12 s0 train14 = 10, so 42+10 = 52/72 (his headline).
- **100% of our suites are expressible**: Dan's `prompt_tokens`, `decompose(canon=True)` and `nd2lean.translate` over all 19 built suite versions: 100% prompts (at most 7 premises vs `MAXH=8`, `lean_tok.py:28`; atoms P,Q,R,S only), 100% references decompose and translate; at most 36 actions (`max_steps` 96), action length at most 210 tokens (`max_action` 512), state length at most 940 tokens (families). Only risk: 80 families/v1 items need more than 32 canonical names (max 35); with the env's random name base U[0,32] (`state_sample.py:26,60-62`) names can run out on at most 3/33 attempts of those items.
- Dan's `nd_verify` is byte-identical to our vendored copy (sha 5dc502c2...).
- **Dan's throughput** (A40, batch 2048, `env_wall_s` in `artifacts/bs/eval/*.json`): textbook72 k256 Fz 196-228 attempts/s, T1 79/s; dev1108 Fz 102-162/s, T1 69/s; rr600 T1 55/s. Peak memory 17-36 GB, so **batch 2048 does not fit GPU 4 (24 GB RTX 3090)**.
- **Dan's code is unusable as-is**: `model.load_ckpt` does lazy bare imports (`model.py:129-134`); `lean_gate` writes `artifacts/lean_gate.jsonl` relative to cwd (`lean_gate.py:207`) and defaults to Lean workers = all CPUs (`:53`); `canonicalise` hard-codes name base 0 (`state_env.py:469`).
- **ndbench gaps**: the runner judges per request (`runner.py:490`), so one Lean process per item; `analysis/_common.latest_run_id` has no judge filter (`:85-116`); `cmd_run` cannot set `RunSpec.options` (`cli.py:256-270`); interp hard-codes `NDJudge` (`runner_support.py:41`), labels base/rl/interp:alpha and one checkpoint directory. No `lean*` runs exist in the store yet, so bumping the Lean judge id is safe.

## 1. Adapter: `code/ndbench/adapters/danstate.py` -> `DanStateAdapter`

**Vendor, don't import from the clone.** Copy byte-identically from nd-takehome@0c492902 into `code/ndbench/vendor/danstate/`: `best_model, model, tokenizer, lean_tok, nd2lean, state_env, state_sample, sample, record, lean_gate` (+ `SHA256SUMS`, and `code/ndbench/tools/vendor_danstate.py` that copies via `git show <sha>:<path>` and never writes into the fork). Reasons: the clone is writable and on a moving branch; plan §0 vendoring convention; tests run without the clone; the bare-name collisions (`model`, `tokenizer`, `sample`, `record`) need a namespaced loader anyway.

`vendor.load_danstate()` modelled on `load_absgpt_module`: import under `ndbench_danstate_*`, strip the `sys.path.insert` entries Dan's modules add, restore any bare `sys.modules` entries, never call functions that do lazy bare imports (`load_ckpt`, `make_tokenizer`, `generate`, `env_generate` with `gate=True`). Reuse our `nd_verify`.

**Loading**: `torch.load(weights_only=True)`, `LeanTokenizer('lean_staten')`, `ALiBiGPT(**cfg)`, `load_state_dict(strict=True)`; assert `tok.canon`, vocab 111, `arch == 'best'`. Coerce string `--model-arg` values to numbers.

**`sample(reqs)`**: group requests by temperature (T=0 means greedy). Flatten all attempts; each attempt's name base `b = Random(f"{req.seed}:{i}").randint(0, 32)`. `_rollout()` is a line-for-line port of `state_sample.env_generate` (`:35-98`) with three changes: (1) no gate, no record calls; (2) returns per attempt `(nd, lean_text, steps, end, gen_tokens)`; (3) a KV-memory guard that splits each wave into sub-calls of `sample.generate_ids_fast(..., seed=seed*100003+wave, early='eos', compact=True)` so rows x (state length + `max_action`) x 9,216 B stays within `kv_budget_gb` (default 14). Otherwise exactly Dan: bf16 autocast, worklist refill, `Env(prompt, canon=True, base=b, assign=True)`. The call seed is a hash of all request seeds (as `absgpt._cross_request_seed`). Each `Attempt`: `body` = `inverse(text)` (an N1-based ND proof) or `UNPARSEABLE env: <reason>`; `native` = JSON `{"lean": text, "base": b, "steps": n, "end": "done|syntax:<why>|truncated|step_cap"}`; `finished` false only for truncated/step_cap; `n_tokens` = total decoded action tokens. Defaults `batch=1024, max_action=512, max_steps=96, name_base_max=32`, all in `identity()` together with the checkpoint sha256, cfg, vocab sha, vendor commit and `rollout="danstate_v1"`. Capabilities: SAMPLE | PREFIX | LOGPROB | ACTIVATIONS | STEER | STEP_ENV.

**Prefix**: accept only `PrefixSpec.body = "STATEPREFIX m=<m> | <full reference ND body>"` (built by `prefix_state`, section 3); anything else raises `NotSupported`. Canonical tokens from vendored `canonicalise`, shift every name by `+b`, `split_actions` to get the action list, replay the first m actions in the env, assert `env.renamed == 0`, then sample as usual; the ND body then includes the forced prefix automatically. (An ND-line prefix can't be used directly: opening a box in Lean commits to the box's goal, which the first j ND lines don't determine.)

**`logprob(pairs)`**: steps from `decompose(prompt, body, canon=True)`. For each base b in 0..32, shift names in both state and action by b (canonical names at base b are exactly base-0 names + b); skip bases where a name would exceed 64. Score = sum of log p over action tokens including `<eos>` (the training mask, `state_train.py:39-40,53`), fp32, length-bucketed batches (copy `absgpt._length_bucketed_chunks`). `per_encoding[b] = sum log p - log 33`; `nats = logsumexp(per_encoding)`, i.e. the average over env-chosen bases, so the existing `LogProb` invariant holds. Sensitivity variant: exclude the defining-name tokens, which the env overrides anyway. **Proofs outside the grammar** return `LogProb(nan, 0, ())` with the reason in `adapter.last_logprob_errors`: PR lines not a prefix in declared order; more than 64 names; a box cited other than by its last line, or an ORE branch ending in False; an uncited box (`proof_tokens` silently drops it, so the re-encoding is a different proof; reject with `allow_reencode=False`). Interp callers must filter NaN. Add a public `canonical_actions(seq, body)` for ordering dedupe.

**`activations(pairs, layers, positions="act"|"action")`**: teacher-forced (state, action) rows at base 0; returns `(n_layers, n_positions, d)` at each step's `<act>` token (the decision point) or at all action tokens.

**`with_steering(spec)`**: deep-copy the model; forward hooks on `model.blocks[l]` (a `PeriLNBlock` called as `b(x, None, None, bias, cache)`). During sampling a block call with T > 1 is the prefill = state ("prompt"); T == 1 is a decode step = action ("proof"). For cacheless forwards use a per-row prompt-length tensor, as absgpt does. `identity()` adds the spec hash.

## 2. Judge

Upgrade `judge/lean.py` to `LeanJudge`, `judge_id="lean_v2"` (update `test_judge_id`): if `native` has `"lean"`, check `statement(prompt) + " " + literal text` (exactly what Dan's gate checks); otherwise check vendored `nd2lean.translate(prompt, body, require_all_pr=False)` (as `lean_judge.py:112`). Lean with Dan's prelude, `-j 1`, chunks of 400, at most 8 workers, no prefilter. Verdict cache `out/cache/lean_v2/verdicts.sqlite` keyed by sha256 of the source. Assert Lean 4.34.0, else raise `NotAvailable`. Add `judge_many(items)`, which the runner calls once per item group when present (without it, online Lean judging costs about an hour of per-item process starts). `pruned` via `prune.pruned_length` (a citation closure, validity-independent).

Add `rejudge --run-id X --judge {nd,lean}` (CPU only, no lease): reads `samples/X.jsonl.gz`, recomputes per-chunk `ItemResult`s with the runner's `_ChunkAccum`, writes a derived run (`run_id=X~<judge>`, `derived_from`, `gpu_s=0`). Add `judge_id=` filters to `run_records`, `latest_run_id`, `all_run_ids`, `models_present`.

**Headline verdict for Dan's models: Lean (`lean_v2`)**: his protocol and RL reward, and a superset of nd_v2 on denotable proofs. The judges disagree where his models omit restating premises (41.7% of Lean-only accepts), use `.elim` on not-A (25.9%), other (6.2%); 24.6% have no ND denotation (Dan's `LEAN_JUDGE.md:140`). Dan runs sample online with `--judge lean` (adaptive stopping on the headline judge), then `rejudge --judge nd`. The existing base/rl/armC runs get `rejudge --judge lean` (137 MB of samples; translation pre-filters most failures; about 800 texts/s). The cross-family headline table uses Lean for every model, nd_v2 alongside.

## 3. Benchmark feasibility

| benchmark | status | notes |
|---|---|---|
| B8 Yue textbook_dev / textbook_law | as-is | dev58 parity with Dan; add a train14 suite for the 72 total |
| B2 families | as-is | `seen_train` must use Dan's tiers (A2); 40 per cell |
| B4 laws | as-is | states up to 622 tokens |
| B1 glue (+ glue_parts_renamed) | as-is | R needs each model's own pass rate on the renamed parts |
| B5 padding v2 | as-is | the bases were chosen by our rl's mastery (caveat) |
| B3 skills | needs a subsample suite | legacy 11,524 items have no Dan runs; about 1,200 stratified |
| B6 prefix | needs `prefix_state` + replay | action prefix m(j) = smallest m whose actions materialise at least j ND lines; tags j, m, A, actions_left, open_boxes; no per-line hazard under Lean |
| B7 EDL | **skip for now** | needs a state-format Learner (AdamW on (state, action) pairs, NLL averaged over bases) |
| I1 likelihood | needs `logprob` + dedupe | deduplicate orderings by action sequence |
| I2 steering | needs ACTIVATIONS/STEER, configurable labels/judge | stage1 and T1 share an architecture |
| I3 interpolation | `interpolate()` works unchanged | needs per-pair checkpoint dir, label prefix, judge parameter |
| I4 probes | needs a step dataset | features at `<act>`; labels next action kind / ND rule (about 16 classes), actions_to_go; baseline step index + #premises + frame depth |
| gen_ctrl | skip | our rl's hill-climb selection set |

`seen_train` for Dan: a per-family seen sidecar from a new `dan_tiers()` (A2).

## 4. Panel and GPU estimate (GPU 4) -- awaiting Leon's approval (A3)

Planning rates on the 3090 at batch about 1024, `max_action` 512: T1 45 attempts/s, Fz 90/s, x1.3 slower for long-state suites; G0/V8 replaces these. Model ids `dan{6,12}-{fz,t1}-s{0,1,2}`.

| id | job | ckpts | attempts | est. min |
|---|---|---|---|---|
| D0 | gates V5-V8 + throughput | dan12 s0 pair | ~80k | 25 |
| D1 | textbook_dev adaptive 256/2048/32 | dan12 s0 pair | ~157k | 41 |
| D2 | textbook_dev adaptive 256/1024/32 | dan12 s1, s2 pairs | ~168k | 45 |
| D3 | textbook_dev + train14 k256 | dan6 s0-s2 pairs | 110k | 31 |
| D4 | families k16 | dan12 s0 pair | 79k | 29 |
| D5 | laws k16 | dan12 s0 pair | 56k | 20 |
| D6 | glue k32 + parts_renamed k16 | dan12 s0 pair | 93k | 26 |
| D7 | padding v2 k32 | dan12 s0 pair | 47k | 13 |
| D8 | prefix_state k16 | dan12 s0 pair | 33k | 12 |
| D9 | textbook_law k64 | dan12 s0 pair | 26k | 7 |
| D10 | skill_pool sub_v1 k16 | dan12 s0 pair | 38k | 11 |
| D11 | interp likelihood + probes | dan12 s0 pair | forward only | 15 |
| D12 | steering 2 layers x alpha {0.5,1,2}, sealed k8, -delta on T1 | dan12 s0 pair | ~85k | 24 |
| D13 | interpolation alpha {.25,.5,.75,1.25} + ends, k128 subset | dan12 s0 | ~115k | 32 |
| total | | | | about 5.5 h |
| R | reserve: reruns, families on dan6 s0 / dan12 s1, higher-cap re-reads | | | at most 1.5 h |

Speed-up to test in G0: most truncated actions never terminate and set every wave's time (Dan's 2x cap diagnostic barely changed outcomes); if `max_action` 256 changes solved counts by at most 1 on textbook_dev k64, use 256 for D4-D13 (about 30% faster); keep 512/96 for textbook runs (Dan parity). Fallback about 2.6 h: D0; dan12 s0 pair textbook_dev 256/1024/32; other four pairs at k256; families 20/cell; glue, prefix_state, D11; interpolation alpha {0.5, 1.25} k64. Use large `items_per_call` (each rollout call holding at least 16k attempts) so the worklist tail stays short.

## 5. Validation gates (in order; stop on failure)

- V0 (CPU): `SHA256SUMS` match `git show 0c492902:<file>`; 12 md5 match `.upload.json` (done, A1); `sha256_prefix` matches `models.yaml`.
- V1 (CPU): strict load, `n_params == 9,560,832`.
- V2 (CPU) replay: 500 references (parts_pool, families, prefix) at bases {0, 17, 32}: full action replay reproduces `inverse(canonicalise(ref))`, `renamed == 0`; Lean accepts all 500; Dan's reassembly check holds.
- V3 (CPU) rollout parity: tiny random ALiBiGPT, our `_rollout` equals vendored `env_generate(gate=False)` exactly (same prompts, seed, batch; Dan's base rule in parity mode).
- V4 (CPU): greedy rollouts of the real T1 checkpoint on 20 theorems; per-step rollout log-probs match `logprob` at that base within 1e-3 (fp32).
- V5 (GPU, ~3 min): held-out greedy on Dan's `heldout.jsonl` (suite `dan_heldout`; k1, T0, 256/48). Expect T1 s0 at least 4,940/5,000 (Dan .992) and Fz s0 4,632 +- 25 (.926).
- V6 (GPU, ~10 min): textbook_dev + train14 k256, T 0.8, 512/96, Lean. Expect T1 s0 42/58 + 10/14 = 52/72 (+-3), Fz s0 27/58 (+-3; seeds 21-27). Truncation about 0.1%.
- V7: on V6 samples: Lean on literal text agrees with Lean on `translate(inverse(text))` 100%; nothing nd_v2 accepts is Lean-rejected; Lean-only accepts broken down missing-PR / `.elim` on not-A / other; negative controls (swapped citation, wrong theorem) rejected.
- V8: throughput and peak memory at batch {768, 1024, 1536} x `max_action` {256, 512}; rescale the schedule.

## 6. Models and dashboard

12 `configs/models.yaml` entries `{adapter: danstate, ckpt, sha256_prefix, md5, family, role: base|rl, cap, seed}` (`cli._resolve_model` ignores the extra fields):

| label | file | sha256 prefix |
|---|---|---|
| dan12-fz-s0/s1/s2 | stage1/stage1_best12_s{0,1,2}_b1200.pt | 2a2a9407eb8ad321 / 9e814c12ce53a879 / bfd9633c5d1ffc7b |
| dan12-t1-s0/s1/s2 | ladder/la_T1_best12_s{0,1,2}_r8.pt | a992485e436a85bb / fa9d2528e0659891 / 5cdf8cf78b29ee8f |
| dan6-fz-s0/s1/s2 | stage1/stage1_best6_s{0,1,2}_b1200.pt | 552d238173868d19 / 7f18a7f6b933492b / 1386e376410abd69 |
| dan6-t1-s0/s1/s2 | ladder/la_T1_best6_s{0,1,2}_r8.pt | 2e12e48207a6175c / 27ffb6acd53a7853 / d470e638510fe0ce |

Dashboard: existing pages keep base/rl/armC only. A new "State-env models" page set (`dash/dan_pages.py`) with small multiples, one panel per family (ours | dan12 | dan6) on a shared y-axis; within a Dan panel Fz uses the base colour and T1 the rl colour, s0 solid with CI band, s1-s2 thin same-hue lines or a min-max band, our rl a grey dashed reference. A judge banner on each page; build twice (`--judge lean_v2`, `--judge nd_v2`). A summary row with a cross-family table (Lean value, nd_v2 in parentheses, seed ranges). Pairs from `models.yaml` (`family`/`role`) instead of the hard-coded base/rl (`yue._pick_pair`, `summary.MODEL_COLOR`).

## 7. Work packages (Sonnet; tests under `code/tests/ndbench/`, each file under 60 s on CPU)

1. **WP-D0 vendor + loader**: `vendor/danstate/`, `vendor/__init__.py:load_danstate`, `tools/vendor_danstate.py`; `test_dan_vendor.py` (checksums; no leftover bare modules or sys.path entries; absgpt and danstate both work in one process).
2. **WP-D1 adapter**: `adapters/danstate.py` + `ADAPTERS` entry; `test_dan_adapter.py` with a tiny ALiBiGPT (2x128, d_ff 256) saved in Dan's checkpoint format: V3 parity; the `Attempt` contract; STATEPREFIX replay identity; logprob vs a manual sum incl. the base convention and NaN cases; activation shapes; steering with zero/nonzero vectors and the T==1 rule; identity fields; kwarg coercion.
3. **WP-D2 judge/runner/CLI**: `judge/lean.py` v2, runner `judge_many` hook, `store.new_derived_run`, `cli.py` `rejudge` and `--option`; `test_dan_lean_judge.py` (fake Lean for literal-vs-translate, one batched call per group, cache; real Lean if present: 10 golden accepted, 10 corruptions rejected, a PR-omitting proof Lean accepts and nd_v2 rejects); `test_dan_runner_rejudge.py`.
4. **WP-D3 suites**: `suites/danstate.py` (`dan_jsonl` for heldout and train14, `prefix_state`, `subsample`), configs, `logic/exclusion.dan_tiers` + seen sidecars; `test_dan_suites.py` (m(0)=0, m(L)=A, m monotone; deterministic strata).
5. **WP-D4 integration (main agent)**: `models.yaml`, `schedule_dan.yaml`, `dan_gates.py` (V0-V8), runbook section; gates, then D1-D13 once approved.
6. **WP-D5 analysis/dash**: judge filters, pairs from `models.yaml`, prefix view for actions_left, `analysis/judges.py` agreement view, Dan pages; `test_dan_views.py` on a synthetic store.
7. **WP-D6 interp**: `interp/danstate_support.py` (step dataset, action positions, ordering dedupe); `__main__`/configs take model labels, judge and checkpoint dir; `test_dan_interp.py`.

Order: D0 -> D1 -> (D2 || D3 || D6) -> D4 gates -> runs; D5 can start in parallel on a synthetic store.

## Not verified by the planner

Throughput on the 3090 (estimates); Lean throughput on this host (Dan measured 745-872 texts/s with 7-8 workers); whether Dan's prefilter ever falsely rejects (he reports 0; we run without it, so our counts can only match or exceed his).
