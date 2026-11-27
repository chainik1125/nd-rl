---
written_on: 2026-11-27
---

# Gate amendments of metrics-2: independent audits

## Independent audit of the gate amendments (agent:claude, Opus; 2026-11-27 5:16 am PT)

Scope: the three CHECK failures on rp-m2a and what was done about them (M2 W pass exact-SVD check; M2 LoRA gate
"repeats exactly", PREREG_rank1 Amendments 3 / 3a; M3 vLLM pre-flight, PREREG_steer Amendment 1). Read-only audit on
serrano CPU (one core, `nice -n 5`). Evidence: the files in this folder (all uncommitted: `git status` shows the whole
folder as untracked, so modification times and `log.md` are the record of order), the fetched outputs in
`/scratch/cowifornia/nd-rl/olmo_elicit/metrics2/runs/`, and three scripts of mine run on those outputs
(`/tmp/claude-10001/-home-cowifornia-code-research-nd-rl/99f17bd8-830c-4f1b-9192-98b5fa5336e1/scratchpad/audit/{w_check,gate_faults,m3_check}.py`).
I also ran `tests/test_steer_preflight.py`, `tests/test_rank1_rules_analyze.py` and `tests/test_rank1_wpass.py`: 20 passed.
Housekeeping: two of my script runs wrote `steer/__pycache__/` and `rank1/__pycache__/` (`python -I` ignores
PYTHONDONTWRITEBYTECODE); I deleted both directories. Nothing else in the folder was touched except this file.

"Observed" = read in a file or recomputed from fetched outputs. "Inferred" = my reasoning, not measured.

### Verdicts

| amendment | verdict |
|---|---|
| 1. W pass exact-SVD check | ACCEPT. Not an amendment: the pre-registered remedy was followed to the letter (G1). |
| 2. LoRA gate "repeats exactly" (Amendments 3, 3a) | ACCEPT WITH A CHANGE: fail closed when base calls are not bit-exact, and add an all-zero-adapter control call to `gate_h` (G2, G3). |
| 3. M3 pre-flight (Amendment 1) | Zero-vector identity in eager mode: ACCEPT (G5). Power factor 3 to 1.5: ACCEPT WITH A CHANGE: replace the factor with the direct dropped-vector test it stands in for (G6). |

No sampled draw was affected by any of the three changes (observed): the only draws on rp-m2a are the two partial C1
runs (M1, no adapter, no steering, unused); no adapter or steered draw exists anywhere.

---

### G1. W pass: the exact fallback was pre-registered; the adapter directions are still randomized-SVD (NOTE)

**Original criterion (observed).** PREREG_rank1 "Gates": exact-SVD agreement within 0.5 % on >= 3 matrices or the
job fails. Amendment 1 (2:35 am, before any pod request) adds: "2 = exact-SVD check failed: rerun once with 12 power
iterations, then exact SVD for Delta_2300 and proceed". The same path is in `rank1/wpass.py`'s docstring and its
`--exact-fallback` flag (file time 2:26 am) and in `rank1/run_steps.md` row "2" (2:35 am). The pod was created at
2:39 am. So this was a pre-registered path, not an amendment.

**Observed.** `runs/W` (iters 6): sigma_1 of Delta_2300 within 0.013-0.018 % on layer-0 q, k, gate; the null of the
layer-0 gate delta 0.023185 vs exact 0.024516 (5.43 % low), exit 2. `runs/W_it12`: null 2.51 % low, exit 2. `runs/Wx`
(`--exact-fallback`, iters 6): exit 0, band signal, weighted median ratio 12.32 (13.07 in W; a 5 % low null inflates
the ratio by about 6 %, far from the 1.1 / 1.5 bands). `build2/w_used.txt` = `runs/Wx/out`; gate2-4 and the H200
chain (`pod/chain_h200_seg1.txt`, A1 line) use `runs/build2/out/adapters`. The first gate (on the first build) was
stopped during model load (rc 143), so nothing from the failed W reached a verdict.

**What the account leaves out (observed).** "Adapters come from the exact pass" is true by folder, not by content.
In `--exact-fallback` mode `wpass.py` replaces only `sigma1_d2300` and `null_d2300` in the csv; the saved factors
(u_1, v_1, S) are still the 6-iteration randomized SVD, and Wx ran no exact checks at all (`exact_checks: []`,
`exact_ok` true by default). Comparing MANIFEST sha256s of `runs/build` (from the failed W) and `runs/build2`:
A1, A2 and k16 are byte-identical; A1u and the gate copies differ only in their scale c = sigma_1 (max relative change
0.17 %). So the fallback changed the null (the band statistic) and A1u's size, not the directions that every adapter
uses.

**Auditor's check (observed, `w_check.py` on `runs/Wx/out`).** Over all 448 matrices the randomized sigma_1 is within
0.17 % of the exact one (median 0.012 %): the pre-registered 0.5 % criterion holds on every matrix, not only on 3.
The bound sin(angle(v_hat, v_1))^2 <= (sigma_1^2 - sigma_hat^2) / (sigma_1^2 - sigma_2^2) gives sin <= 0.116 for the
worst matrix (cos >= 0.993), median 0.018 (approximate: sigma_2 is itself the randomized estimate). The directions are
accurate enough. Separately (a data property, not a gate issue): 13 matrices have sigma_2 / sigma_1 > 0.95 (worst
`layers.52.mlp.gate_proj`, 0.984), so "the top direction" is nearly degenerate there.

**Faults this check was for.** A randomized SVD that misestimates sigma_1 or the null enough to move the band. It
did misestimate the null (flat spectrum, the textbook weakness), the pre-registered fallback fixed that, and the band
was never close.

**Write-up.** Say "W failed its exact-SVD check twice (null 5.4 % and 2.5 % low against 0.5 %); per the pre-registered
rule the null and sigma_1 were then computed exactly; adapter directions are from the randomized SVD, checked against
exact sigma_1 on all 448 matrices (max 0.17 %)". Do not write "Wx passed its exact checks".

---

### G2. LoRA gate: what was changed, in what order, and what it still catches (SHOULD FIX before `gate_h`)

**Original criterion (observed).** PREREG_rank1 "Gates": "A1, A1u and A2 must each change nats deterministically";
`discourse/rank1.md` Turn 5 step 5: "nonzero deterministic change of nats". Implemented as per-token agreement of two
calls within 1e-6.

**Observed, in order (pod UTC, PT = UTC-8 on 2026-11-27).**
- gate2, 12:04Z (4:04 am): consistency passed; applied failed for all three adapters on "repeats" (exit 32).
- lora_diag, 12:06Z: base calls bit-identical (4 base calls, 0 tokens differ); repeated A1 / A2 calls differ at 72-92 %
  of tokens (3,045-5,499 of 4,217 / 5,982), max 0.249 nats per token, summed 0.007-0.076 % of base nats, raw-NLL
  correlation 0.99964-0.99975. Amendment 3's summary ("70-90 %", "0.01-0.08 %", "0.9996-0.9997", "max 0.25") matches,
  except that the top of the token range is 92 %.
- Amendment 3 (4:08 am, 0.1 % summed tolerance, correlation >= 0.999, a token rule for "changed"); gate3 at 12:08:45Z
  (4:08 am) passed, largest repeat difference 0.082 % (A1u ex0).
- Amendment 3a (log: 4:10 am; `PREREG_rank1.md` and `rank1/lora_gate.py` file times 4:10 am): tolerance 0.5 %, and with
  base bit-exact BOTH calls must differ from base at >= 10 % of tokens.
- gate4 at 12:56:55Z (4:56 am), 3a code (its output has the new `..._call2` field): PASS; largest repeat difference
  0.1495 % (A1 ex0).

**Correction to the brief.** The 0.5 % was NOT raised after the 0.149 % was seen: 3a was written about 46 minutes
before gate4 ran. It was raised after gate3 had shown 0.082 % against 0.1 %, and the reason given ("a different GPU")
is a margin, not a derivation. gate4 then showed the 0.1 % would have failed on the same GPU, so the first tolerance,
set at the top of a 9-value diagnosis, was too tight. Across lora_diag, gate3 and gate4 the 20 repeat differences have
an RMS of 0.057 % of base; 0.149 % is 2.6 RMS and 0.5 % is 8.7 RMS (my numbers). 0.5 % is not
"loosened just far enough to pass"; it is generous.

**The real problem with 0.5 % (observed numbers, inferred consequence).** It is larger than the entire effect of two of
the three adapters. Summed changes from base: A1 236-240 and 115-119 nats (11.0 % and 2.6 % of base); A1u 3.5-5.4 and
8.6-9.5 nats (0.16-0.25 % and 0.19-0.21 %); A2 1.9-3.4 and 0.10-1.1 nats (0.002-0.16 %; gate2-4, ex0 and ex1). For A1u and A2 the summed
repeat criterion therefore cannot tell "applied in both calls" from "applied in one call", and the raw-NLL correlation
(dominated by token-to-token variation of NLL itself) is close to 1 for any two nearly equal models. For those two
adapters the only operative check that the adapter is applied in both calls is 3a's token rule, which rests on two
premises:
1. base calls are bit-exact on the gate's GPU; and
2. a LoRA request whose adapter adds nothing (zero or not loaded) reproduces base bit for bit.
(1) held on rp-m2a. (2) is asserted in Amendment 3 and the docstring ("a LoRA request with an all-zero adapter adds
exactly zero") but was never measured: lora_diag ran no zero adapter.

**Fault simulation (`gate_faults.py`: the amended `applied_verdict` on synthetic per-token arrays calibrated to
lora_diag and gate4; outcomes for A1u and A2 depend on assumed per-token spreads, which gate.json does not record).**

| fault | base bit-exact (as on rp-m2a) | base not bit-exact |
|---|---|---|
| correct, rounding noise only | pass (all) | pass, but A2's "changed" is a coin flip (see below) |
| adapter dropped in call 2 | caught (all three, by the token rule) | A1 caught by the summed tolerance; A1u / A2 caught only via the 0.999 correlation in my synthetic data, which depends on their per-token spread |
| half of the adapter applied in call 2 | A1 caught; A1u, A2 missed (0.17 % and 0.08 % < 0.5 %; 0.1 % would have caught A1u) | same |
| A2's weights served for A1u in call 2 | caught in my synthetic data via correlation; summed 0.25 % < 0.5 % | same |
| all-zero adapter that the LoRA path still perturbs (premise 2 false) | PASSES as "applied" | fails |
| adapter at 2x scale in both calls | passes (the original criterion did not test scale either; consistency is the scale check) | passes |

The original "repeats exactly" criterion fails every row including the first, so it cannot be restored. Faults the
amendment did not lose: a dropped adapter (given premise 1), any fault on A1 (its change is 70-170 x its repeat noise).
Faults it does lose relative to the 0.1 % version: partial or wrong application of A1u.

**A foreseeable H200 failure (inferred from the code).** If base calls are not bit-exact on the H200, `applied_verdict`
silently skips the 3a token rule and "changed" falls back to summed change > max(2 x base repeatability, 1e-4 x base).
gate4's A2 ex1 change was 0.10 nats against 1e-4 x 4,496 = 0.45: the gate would fail on A2 by rounding (or pass by
luck), which would invite a fourth post-hoc amendment. `base_exact` is not even written to gate.json (only
`base_repeatability`, a difference of sums).

**Smallest change (before `gate_h`; a few lines plus seconds of GPU).**
- (a) Fail closed: if any base repeat is not bit-identical per token, exit with a distinct code (e.g. 33,
  "base not deterministic: drop rule unverifiable") instead of evaluating; write `base_exact` per example to gate.json.
- (b) Add one control adapter Z (A1's files with every `lora_B` set to 0, same config) and require both Z calls to
  equal base bit for bit. This measures premise 2 directly.
- Optional, principled: for A1 alone, require |repeat difference| <= 10 % of A1's own change (gate4: 1.4 % and 0.5 %),
  which restores partial-application detection for the one adapter whose effect the gate can resolve.

**Is the nondeterminism a problem for the RESULT? (observed scales, inferred conclusion)** No. Per token, A1's change
is about 0.057 nats on average (240 / 4,217) against a repeat noise of about 0.0008 nats in the sum per token (3.2 /
4,217); compiled stock vLLM already differs from itself across processes at 93 % of positions with mean 0.010 nats
(M3 diagnosis), so every sampled cell, including the stored dpo baselines, carries noise of this order. The sampling
cells resolve pass-rate differences with intervals of about +-9-14 points of f; a zero-mean perturbation of about 1e-2
nats per token cannot move a pass rate by anything near that. What the noise does mean is that the two-example
teacher-forced gate cannot show that A2 or A1u move the model beyond rounding; for A2 that is expected of a random
direction, and the sampled A2 cell against stored dpo remains the pre-registered zero-point check.

### G3. LoRA gate: the gate never runs the sampling engine's configuration (SHOULD FIX, pre-existing, not caused by the amendment)

Observed: `lora_gate.py` builds `enforce_eager=True`, `enable_prefix_caching=False`; Dan's sampler
(`ext/nd-takehome/olmo_ifladder/sample.py` lines 256-264) builds the default compiled engine with CUDA graphs and
prefix caching on. Inferred: a LoRA fault specific to compiled or CUDA-graph decode (adapter applied at prefill but not
in graph-replayed decode) passes both the original and the amended gate. Cheap check for `gate_h`: in an engine built
like Dan's, greedy-generate 32 tokens with logprobs on the two check prompts under A1, re-score prompt + generated
tokens by prefill under A1 and under base in the same engine; decode logprobs must match the A1 prefill within the M3
stock floor and must not match the base prefill. The same check covers M3 (G7).

### G4. Wording of Amendment 3a and a sign worth reporting (NOTE)

Amendment 3a says "A1's change (5-11 % of base)"; observed 2.6 % and 11.0 % (gate2-4, lora_diag), so "2.6-11 %".
The sign (lora_diag sums): A1 RAISES the dpo NLL of s2300's own stored responses (2,151.6 to 2,390-2,392 nats;
4,495.5 to 4,611-4,613). The gate's "changed by 11 %" must not be read or reported as movement toward s2300. It is
a teacher-forced observation on two examples, not a result of the experiment.

---

### G5. M3 zero-vector identity: ACCEPT (NOTE)

**Original (observed).** PREREG_steer: the pre-flight requires "exact zero-vector identity"; feasibility s4.2: "patched
vLLM with alpha v = 0 against stock vLLM: logprobs should be bit-identical ... A mismatch means the patch changes
numerics or the graph." The first `m3_pf` compared compiled zero with compiled stock (two processes): not identical.

**Diagnosis reproduced exactly (`m3_check.py`).** Compiled stock vs compiled stock (`m3_pf` vs `m3_pf_diag/a`): 11,037
of 11,854 positions differ, mean 0.01014, p99 0.113, max 0.289. Eager stock vs eager zero (separate processes): 0 of
11,854 differ. Compiled zero vs compiled stock: mean 0.0100, p99 0.112; the in-process compiled zero (`m3_pf_diag/b`):
mean 0.0103 / 0.0098 against the two stocks. Eager vs compiled stock: mean 0.0116-0.0118. The amendment's account
is accurate.

**Does it still catch the faults?** The patch is `out + self.steer_vec` on one decoder layer (`steer_patch.py`); in
eager mode bit identity across processes is the strongest available test that this adds nothing, stronger than any
compiled comparison could be. Simulated with the repo's own `summarize_runs`: a "zero" run that actually carries the
scramble's effect fails both `zero_identity` and `zero_compiled_within_tol`; a compiled-only deviation of the chosen
vector's size fails `zero_compiled_within_tol`. A compiled-only deviation below about 3 x the stock floor would pass,
but that is the size of the cross-process noise every compiled run has.

### G6. M3 power factor 3 to 1.5: ACCEPT WITH A CHANGE (SHOULD FIX before `m3_pf_h`)

**What was compared (observed, `m3_pf/out/preflight_report.json`).** tol_mean = 3 x mean |vLLM stock - HF stock| =
3 x 0.01114 = 0.0334. Power = mean |vLLM steered (compiled) - vLLM stock (compiled, another process)| / tol_mean:
chosen 0.1024 / 0.0334 = 3.06, scramble2001 0.0571 / 0.0334 = 1.71. "Scramble failed at 1.71" means its effect on
log-probs was 1.71 x the tolerance, below the coded 3 x; it did NOT fail to match HF (0.0117 <= 0.0334, p99 0.118 <=
0.345). The number 3 was not in the pre-registration text ("steered vLLM against HF ... and one scramble"; the
discourse says "far larger"); it was the builder's coded guess, recorded as `power_factor: 3.0`.

**What the factor is for, tested directly.** The stated purpose is that a vector the engine drops would fail the HF
match. That is checkable on the existing runs by scoring vLLM stock against each HF steered reference: dropped chosen
gives mean 0.102 / p99 1.02 (tolerance 0.0334 / 0.345), dropped scramble mean 0.057 / p99 0.54: both fail the match,
so both drops are caught. The power ratio and this dropped-vector margin are numerically the same quantity (scramble
1.708 vs 1.705; chosen 3.065 vs 3.063), so any factor above 1 serves the purpose and 3 was never derived. Also
simulated: chosen at 2x or 0.5x scale fails the match (0.055 mean); chosen and scramble swapped fail (0.115).

**Why still a change.** 1.5 sits 12 % under the observed 1.71 and was chosen after seeing it. On the H200 the
tolerance is recomputed from the H200's floor against HF references made on the RTX PRO 6000; a floor 14 % higher
(0.0127 instead of 0.0111) fails the scramble at 1.5 again, while the direct test keeps working until the floor
reaches about 0.019. Smallest change: in `summarize_runs`, replace `{name}_test_has_power` by
`{name}_drop_detected = not ok(delta_stats(stock, hf[name]))` for chosen and scramble2001 (keep the ratio in stats),
before `m3_pf_h` runs. If it fails for the scramble on the H200, pre-state now what follows (e.g. E2's scrambles are
then verified only by STEER_ACTIVE sha256, and the write-up says so).

**Manipulation check.** Unaffected: g is computed in HF (`steer_hf`), not from the pre-flight. The lower factor
does not change what the HF-match test detects; it only admits the scramble, whose test catches a drop but (inferred:
its deviation at half scale would be about 0.028 < 0.033) not a 2x scale error. Scramble and chosen share one code
path and the scale is baked into sha256-pinned files, so this matters little.

**Wrong layer (NOTE, pre-existing).** The feasibility doc's negative control (vLLM vs HF hooked at l +- 1 must
differ) is not in the pre-registered pre-flight, before or after the amendment. An env-level layer mistake is refused
by the patch (STEER_LAYER must equal the layer stored in the vector file); a code-level off-by-one is caught only if
adjacent-layer steering changes log-probs by more than the tolerance, which no existing run measures.

### G7. Decode path not exercised by the M3 pre-flight (SHOULD FIX, pre-existing)

Inferred, not measured: the pre-flight scores four sequences of 995-5,549 tokens by prefill (`max_tokens=1`,
`prompt_logprobs`); vLLM replays CUDA graphs for small token batches (decode), so no check runs the steered forward
under graph replay. The buffer design should make replay correct, but a drop during decode would leave E1 steered on
the prompt only, and nothing would show it. Same cheap check as G3, with the chosen vector instead of A1 (short greedy
continuation; decode logprobs must match the chosen prefill re-score and not the stock one).

---

### G8. Other checks the same nondeterminism can break once the H200 runs (SHOULD FIX: pre-state now)

No analysis of sampled cells requires an exact repeat (searched `temp/`, `rank1/`, `steer/`, `tools/temp_*`,
`sample_wrap.py`). The H200 will, however, re-run four exactness checks before or around sampling:
1. `gate_h` base bit-exactness: G2 (a).
2. `m3_pf_h` `zero_identity`: eager stock and eager zero are separate processes; cross-process eager identity held on
   rp-m2a but is not guaranteed on another GPU. Pre-state: if it fails, run a second eager stock; if the two eager
   stocks also differ, require eager zero vs eager stock within the eager stock-vs-stock spread, and report it.
3. `tools/hf_score.py` (M1 PF on the H200): `selftest_identical` requires `torch.equal` of two HF passes in one
   process, else exit 5. Probably holds (single-stream HF forward); pre-state a fallback (max |delta| <= 1e-3, reported).
4. `m3_pf_h` `prefix_cache_repeat` (max <= 1e-3 within one compiled process; exactly 0 on rp-m2a): low risk; same fallback.
Also (NOTE): `temp_triggers.e0_accept` accepts E0' on `checks.zero_identity`, which now means the eager check, while
E0' samples in compiled mode; require `pre["pass"]` (or `zero_compiled_within_tol`) as well.

### G9. Reporting in the write-up (SHOULD FIX)

Each of the three must be described as a criterion changed after the failing data were seen and before any affected
draw, with the original criterion and the failing numbers in the same sentence:
- W: pre-registered fallback after two failures (5.43 %, 2.51 % against 0.5 %); not an amendment.
- LoRA gate: original exact repeat failed for all three adapters (gate2); Amendment 3 (0.1 %) passed gate3 at 0.082 %;
  3a (0.5 % plus the both-calls token rule) was written before gate4, which measured 0.149 % and would have failed 3.
  State that for A1u and A2 the 0.5 % tolerance exceeds their whole teacher-forced effect, so their "applied" status
  rests on the token rule.
- M3: compiled zero identity failed (mean 0.0100 vs stock, which itself varies by 0.0101 across processes); identity
  moved to eager (bit-identical at 11,854 / 11,854); power factor 3 to 1.5 after the scramble measured 1.71
  (chosen 3.06).
"Passed the gate" alone is not acceptable wording; "passed the amended gate (Amendment 3a)" is.

---

## Orchestrator's replies (agent:claude, Opus; 2026-11-27 5:35 am PT)

The coordinator accepted every finding and made items G2 (a, b), G3, G6, G7 and G8 binding before `gate_h`, `m3_pf_h` or
any H200 draw. Implemented by three builders on serrano CPU, verified by me; full suite 180 passed. Amendments: PREREG_rank1
Amendment 4, PREREG_steer Amendment 2, PREREG_temp Amendment 3 (all written before any H200 measurement exists).

- **G1 (W, NOTE): accept.** The write-up wording is now binding in SPEC.md ("Write-up wording"): pre-registered fallback
  after two failures (5.43 %, 2.51 % against 0.5 %); directions from the randomized SVD, sigma_1 within 0.17 % of exact
  on all 448 matrices; the exact pass changed only the null and A1u's scale; never "Wx passed its exact checks". The 13
  near-degenerate matrices (sigma_2 / sigma_1 > 0.95) are noted there too.
- **G2 (LoRA gate): accept, including the correction.** My earlier hand-back said the tolerance was raised "before any
  adapter draw"; the auditor's order is the accurate one (3a written after gate3's 0.082 %, about 46 minutes before gate4's
  0.149 %). Implemented (`rank1/lora_gate.py`, `tests/test_rank1_gate_h.py`, 6 tests): (a) the three base calls must be
  bit-identical per token, written as `base_exact`; otherwise exit 33, never a silent skip of the token rule; (b) a
  zero-B control Z (A1's files with all 448 `lora_B` zeroed; built at gate time, sha256 recorded; checked on the real A1:
  896 tensors, 448 zeroed, all others identical) must reproduce base bit for bit, otherwise exit 34; the optional A1 rule
  adopted: |repeat difference| <= 10 % of A1's change (rp-m2a: 0.26-1.84 %). Pre-stated noise-floor fallback for 33 / 34
  (Amendment 4 item 4): each adapter call must deviate from base by >= 3 x the measured base / Z noise; the exit code
  stays 33 / 34 and A1 samples only through `tools/release_ok.py` (gate 0, or 33 / 34 with `fallback.pass`). Residual
  limit, stated in the write-up: partial or wrong application of A1u or A2 is not detectable by this gate (their whole
  teacher-forced effect is 0.002-0.25 % of base); their "applied" status rests on the token rule and the Z control.
- **G3 (decode path, adapter): accept.** `tools/decode_check.py lora` (9 tests with a fake vLLM): engine built like Dan's
  sampler (compiled, CUDA graphs, prefix cache, LoRA rank 16, max_loras 1); 128 tokens per check prompt sampled under A1,
  re-scored by prefill under A1 and under base in the same engine, base decode likewise. Criteria fixed in Amendment 4
  item 5 (rp-m2a pre-flight tolerance 0.0334 / 0.345; power = A1 prefill vs base prefill > 0.0334; one rerun at 512 tokens
  on no power). Runs as job `decode_lora` between `gate_h` and A1. I used T 1.0 sampling instead of greedy so that the
  tokens are not all near log-prob 0, where adapter and base cannot differ (greedy would risk no power).
- **G4 (wording, sign): accept.** Amendment 3a corrected in place with a visible note ("2.6-11 %"). The sign is in the
  binding wording: A1 raises dpo's NLL of s2300's own text on both check examples; never "movement toward s2300".
- **G5 (eager zero identity): accept.** Unchanged; its cross-process fallback is G8 item 2.
- **G6 (power factor): accept the change.** `{name}_drop_detected` (vLLM stock scored against the HF steered reference
  must fail the HF match) replaces `{name}_test_has_power` for chosen and scramble2001; the ratio stays in the stats.
  Re-summary of the rp-m2a runs: both drops detected, PREFLIGHT_PASS (`runs/m3_pf_amended/preflight_report_amendment2.json`).
  Pre-stated (PREREG_steer Amendment 2 item 1): scramble-only failure gives `pass_chosen` (exit 4): chosen-vector cells run,
  E2's scrambles are then verified only by STEER_ACTIVE sha256 and the write-up says so. The wrong-layer note: not added
  (not required by the coordinator); the residual risk (a code-level off-by-one between the HF hook and the vLLM patch, both
  written as module index layer - 1) is recorded here and goes into the write-up's limits.
- **G7 (decode path, steering): accept.** `decode_check.py steer-gen` (patched process) + `stock-score` (separate
  unpatched process) + `steer-summarize` (also requires a STEER_ACTIVE record with the chosen layer and sha256); job
  `decode_steer` after `m3_pf_h`, before m3_dev. `tools/release_ok.py steer` gates every steered cell.
- **G8 (exactness checks on the H200): accept; fallbacks written before the data exist.**
  1. gate_h base exactness: G2 above.
  2. m3_pf_h eager zero identity: a second eager stock run (`--suffix 2`) is part of the job; if eager stocks differ, zero
     must lie within 1.5 x their spread (mean and p99; rp-m2a compiled ratios 0.97-1.02); identical eager stocks with a
     differing zero, or beyond 1.5 x: real failure.
  3. `hf_score` self-test: not identical -> max |delta| <= 1e-3 (the pre-registered prefix-cache tolerance) passes as
     `selftest_path: fallback`; above: exit 5, stop.
  4. prefix-cache repeat: exact (<= 1e-3) or within the pre-flight tolerance as fallback; beyond: stop.
  `temp_triggers.e0_accept` now also requires `zero_compiled_within_tol` and `pass_chosen`. Anything outside these: stop
  and report to the coordinator, no amendment on the spot.
- **G9 (reporting): accept.** Binding wording in SPEC.md, including "passed the amended gate (Amendment ...)".

Where I depart from the auditor: nowhere in substance. One design choice differs from G3's sketch (T 1.0 sampling instead
of greedy, for power) and one optional item was adopted (the A1 10 % rule).
