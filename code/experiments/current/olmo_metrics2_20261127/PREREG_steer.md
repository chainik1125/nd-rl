---
written_on: 2026-11-27
written_by: agent:claude (orchestrator, Opus 5.5)
---

# Pre-registration, M3 (`steer`): does adding one mean-difference vector to dpo recover part of the final model's behaviour?

Written 2026-11-27 2:27 am PT (`date`), before any pod was requested and before any activation or sample existed.
The design of record is `discourse/steer.md` Turn 5 section F with the auditor's Turn 6 fixes d1-d7 (file md5
1ff1a983149224275bde6fc2ed6c2590) and `SPEC.md`. This file freezes the readings; where it is silent, section F holds.

## Construction (fixed)

- Vector prompts V32: lines 0-31 of `pool/train.jsonl` (arm-A non-reach). The other 144 arm-A non-reach prompts are
  permuted with seed 0: dev 64, held-out 80. S16 / H16: the first 16 dev / held-out prompts (permutation order) with a
  stored strict-pass `s2300` draw.
- V_resp (primary): mean over response positions of `dpo`'s own stored response (one draw per V32 prompt, 12,000-token
  cap), of h_s2300[l] - h_dpo[l], both models reading the same text; mean within prompt, then across prompts. V_prompt:
  the same over user tokens plus the trailing template tokens. h[l] = HF `hidden_states[l]` = output of `layers[l-1]`.
- Steering: alpha * v added to the output of `layers[l-1]` at every position (prompt and generated). alpha = 1 adds
  exactly the mean difference.
- Selection by teacher-forced NLL in HF (never by pass rates, never on reach): grid l in {16, 20, 24, 32, 40} x alpha
  in {0.5, 1, 2, 4, 8} for V_resp, then V_prompt at the best V_resp layer; admissible if the NLL rise on `dpo`'s own
  S16 text is at most 2 x D_rev (D_rev = NLL_s2300 - NLL_dpo on that text, total tokens); choose the admissible
  setting with the lowest total NLL of `s2300`'s S16 strict-pass text; ties within 0.005 nats -> smaller alpha, then
  smaller layer, then V_resp; fallbacks as section F with d2. Grid shrink order if the NLL block projects above $3:
  drop alpha 0.5, then l 40, then the V_prompt row.
- Split-half stop: c = cosine of the 16-prompt half means at the chosen layer and type; c < 0.5 stops sampling (exit
  code, reported with sqrt(2c/(1+c)) and top-5-removed versions).

## Cells

- Dev guard: chosen setting, 64 dev prompts x 1 draw (cut (2) taken in advance), 12,000-token cap, seed 3000; fails
  if cap hits > 10 %, mean length outside 0.7-1.3 x dpo's, or dev pass@1 more than 10 points below dpo's stored rate
  at the same truncation; then alpha is halved once.
- E0': `dpo` + zero vector, 48 reach x 8 (384), seed 1000; also M1's fresh reach baseline.
- E1: chosen setting, 48 reach x 8 + 80 held-out non-reach x 2 (544), seed 1001.
- E2 (released only by the trigger): three sign scrambles of alpha * v (seeds 2001-2003), same draws as E1 split
  across them (reach 3 / 3 / 2 per prompt; held-out by prompt i -> scramble (i mod 3) + 1); under the budget cap,
  reach only (cut 6).

## Primary readouts

1. Reach pass@1, E1 against E0' (8 draws each), paired prompt bootstrap (10,000, seed 0), plus raw counts and
   pass@8.
2. Held-out non-reach pass@1, E1 against Dan's stored `dpo` draws (16 per prompt).
3. Manipulation check on H16 `s2300` text: g = (NLL_dpo - NLL_steered) / (NLL_dpo - NLL_s2300), total, thinking and
   answer tokens, with a joint 95 % prompt-bootstrap interval; bands < 5 % "barely moved", 5-20 % "slightly", >= 20 %
   "substantially", assigned only if the whole interval lies inside a band. Also g at alpha = 1, the scrambles' g,
   the necessity ratio R(alpha = 1) = (NLL_{s2300 - v} - NLL_s2300) / (NLL_dpo - NLL_s2300), and the damage rise.

## Rules fixed now

- E2 trigger: E1 reach passes minus E0' reach passes >= 2, or |held-out pass@1 of E1 minus stored dpo| >= 6
  points. If E2 does not run, the write-up says "specificity was tested in NLL only, not in sampled behaviour".
- A scramble with g >= 0.5 x the real g means the real direction is not special in NLL.
- Reach is framed in checkpoint equivalents: `s0100` 0.1 %, `s0500` 4.5 %, `s1100` 38.5 %, 60-word hint 6.2 %,
  64-example LoRA 63.5 %. Resolution: reach rates of about 2-3 % over >= 3 prompts are separable from E0'; held-out
  about +-6-8 points.
- Steering correctness: the vLLM pre-flight (stock noise floor, exact zero-vector identity, steered vLLM against HF
  with the hook at the chosen setting and one scramble, compile against eager, prefix-cache repeat, `STEER_ACTIVE`
  with the vector hash in every sidecar) must PASS before any steered draw; a failure stops M3's sampling.

## Priors (written before data; not results)

- Auditor (independent read): 0 to +5 points on held-out pass@1 and 0-3 % on reach; a gain of 15 points on held-out
  or 20 % on reach with flat scrambles would contradict Venhoff's "when, not how" and Heo's finding that no single
  direction works across instruction types.
- Orchestrator: g in the "barely moved" or "slightly" band at the selected setting, reach E1 below `s0500`'s 4.5 %,
  and held-out within +-6 points of dpo.

## What would and would not follow

- E1 above E0' on reach with scrambles flat: one static mean-difference vector moves `dpo` toward the final model on
  prompts it never passes. Not that RL "is" this vector, not optimality, not other layers, not genuine passes.
- A null with g "substantially": the vector moves `dpo`'s token distribution toward `s2300` without moving checker
  passes. A null with g "barely moved": the construction does not move `dpo` in layers 16-40 at alpha <= 8; it says
  nothing about learned or gated vectors.

## Amendment 1 (4:56 am PT by `date`, before any steered draw): what the pre-flight's identity and power checks can require

The pre-flight on rp-m2a (RTX PRO 6000, `runs/m3_pf`) failed two checks and passed all others (chosen and scramble2001
match the HF hook within tolerance, also beyond 4,096 tokens; compile vs eager; prefix-cache repeat; `STEER_ACTIVE` with
the right layer and sha256 in every steered run). A diagnosis (`runs/m3_pf_diag`) found:
- **Zero-vector identity.** Compiled vLLM does not reproduce itself across processes. Two stock processes differ at 11,037 of
  11,854 positions (mean |delta log-prob| 0.0101, max 0.289), so "zero (compiled) bit-identical to stock (compiled)" cannot
  hold for any patch. In eager mode, stock and the zero-vector patch are bit-identical at all 11,854 positions: the patch
  adds exactly nothing. Compiled zero vs compiled stock (mean 0.0100, p99 0.112) is within the stock repeat spread.
- **Power.** The code required steered-vs-stock mean >= 3 x tol_mean for chosen and scramble2001 (tol_mean = 3 x the stock
  floor = 0.0334). chosen 0.1024 (3.06 x) passed; scramble2001 0.0571 (1.71 x) failed. The purpose was that a vector the
  engine silently drops would fail the HF match; that holds whenever the effect exceeds the tolerance.

Amended (`steer/steer_preflight.py`, tests updated): zero identity = bit identity of `stock --eager` and `zero --eager`
(both now required runs), plus compiled zero vs compiled stock within the tolerance; power factor 1.5 (an effect at least 4.5 x
the stock floor) for chosen and scramble2001. Re-summarized over the same runs plus the two eager runs: PREFLIGHT_PASS
(`runs/m3_pf_amended/preflight_report_amended.json` on serrano). The pre-flight runs again on the H200, with the two eager
runs added, before any steered draw there; PREREG_temp Amendment 1's `checks.zero_identity` refers to this eager check.

## Amendment 2 (5:25 am PT by `date`, before `m3_pf_h`, the steering decode check and any steered draw): a direct drop test instead of a power factor, fallbacks written now, and a decode check

What I had seen when writing this: only rp-m2a data (`runs/m3_pf`, `runs/m3_pf_diag`, Amendment 1) and the independent
audit `discourse/gate_amendments_audit.md` (G5-G8). No H200 measurement exists. Binding by the coordinator's decision.

1. **Drop test replaces the power factor (audit G6).** Amendment 1's factor 1.5 was set after the scramble measured 1.71.
   Replaced by the test it stood for: vLLM stock scored against the HF steered reference must FAIL the HF match
   (`chosen_drop_detected`, `scramble2001_drop_detected`), while the steered vLLM run must pass it (`*_matches_hf`, as
   before). The effect / tolerance ratio stays in the stats only. On rp-m2a both drops are detected (stock vs HF chosen:
   mean 0.102, p99 1.02; vs HF scramble: 0.057, 0.54; tolerance 0.0334 / 0.345).
   Pre-stated: if only scramble checks fail on the H200 (`pass_chosen` true, `pass` false), the chosen-vector cells (dev,
   E0', E1) run; E2's scrambles are then verified only by `STEER_ACTIVE` sha256, and the write-up says so.
2. **Eager zero identity across processes (G8 item 2).** `m3_pf_h` runs eager stock twice. If eager zero equals eager stock
   bit for bit: pass. If not, and the two eager stocks also differ: measured instead, eager zero vs eager stock within
   1.5 x the eager stock-vs-stock spread (mean and p99). Source of 1.5: on rp-m2a the compiled zero-vs-stock / stock-vs-stock
   ratios were 0.97-1.02 (mean) and 0.99 (p99). Real failure: eager stocks identical but zero differs, or zero beyond 1.5 x
   the spread (the patch changes numerics): M3 sampling stops, I report.
3. **Prefix-cache repeat (G8 item 4).** Exact criterion unchanged (max <= 1e-3 within one compiled process; 0 on rp-m2a).
   Fallback: mean and p99 of the repeat difference within the pre-flight's own tolerance (3 x the H200 stock floor, built
   by the pre-registered rule; a different quantity from the one judged). Real failure: beyond it (the cache returns
   different values than a fresh pass beyond rounding): stop and report.
4. **Decode path (audit G7).** Before m3_dev, `tools/decode_check.py steer-gen` (patched engine built like Dan's sampler)
   samples 128 tokens per check prompt with the chosen vector (T 1.0, seed 4242) and re-scores them by prefill in the same
   process; `stock-score` (separate unpatched process) re-scores the same tokens and its own. Pass = steered decode matches
   steered prefill (mean <= 0.0334, p99 <= 0.345, the rp-m2a pre-flight tolerance), steered prefill differs from stock
   prefill by mean > 0.0334 (power), stock decode matches stock prefill, and a `STEER_ACTIVE` record with the chosen
   layer and sha256. No power: rerun once with 512 tokens. Otherwise a failure stops M3 sampling and I report.
5. **E0' as M1's baseline (G8 note; also PREREG_temp Amendment 3).** `temp_triggers.e0_accept` now also requires
   `zero_compiled_within_tol` and `pass_chosen` (E0' samples with the compiled engine; `zero_identity` is the eager check).
6. **Anything not covered here: stop and report to the coordinator; no amendment on the spot.**

## Amendment 3 (10:43 am PT by `date`, after the first dev guard failed and before m3_dev_half exists): what the halved alpha carries with it

Seen when writing: the H200 pre-flight and decode check (both PASS, with the chosen vector), the dev guard at alpha 4.0 (FAIL on
cap hits only: 7 / 64 = 10.9 % > 10 %; length ratio 1.29; pass@1 +8.5 points). Not seen: anything at alpha 2.0.
1. The pre-registered remedy runs: alpha halved once (`vecs/chosen_half.pt`, alpha 2.0, written by p3 before any sampling),
   pre-flight mode chosen_half on the H200 (HF references exist), guard rerun as m3_dev_half; a second failure stops E1 and E2.
2. Release of m3_dev_half and E1 at the halved alpha: `pass_chosen`, `chosen_half_matches_hf`, the chosen_half STEER_ACTIVE
   record, and the decode check PASS. The decode check ran with the alpha 4.0 vector; the halved vector uses the same patch
   and buffer, so for it the decode path is covered only by that run (stated as a limit in the write-up).
3. E2 at the halved alpha, if the trigger fires: the pre-registered E2 cells are sign scrambles of alpha * v; at alpha 2.0 they
   are exactly 0.5 x the existing scramble files (`tools/half_vecs.py`, fp32 scaling by 0.5 is exact; the tool reproduces the
   stored chosen_half bit for bit from chosen). No HF references exist for them, so they are verified by their STEER_ACTIVE
   sha256 only (PREREG_steer Amendment 2 item 1's pre-stated scramble fallback), and the write-up says so.
4. Everything else unchanged (E1 draws, seeds, trigger rule, analysis).
