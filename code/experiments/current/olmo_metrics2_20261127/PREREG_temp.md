---
written_on: 2026-11-27
written_by: agent:claude (orchestrator, Opus 5.5)
---

# Pre-registration, M1 (`temp`): does sharpening dpo's own distribution recover part of RL's gain?

Written 2026-11-27 2:24 am PT (`date`), before any pod was requested and before any new sample existed. The design of
record is `discourse/temp.md` Turn 5 Part 3 with the auditor's Turn 6 fixes 1-8 (file md5
9288b782f78db54fdc74939ad8955e4c) and `SPEC.md`. This file freezes the readings; where it is silent, Part 3 holds.

## Cells

- C1: `dpo`, T 0.6, top_p 0.95 (the model card's and the Olmo 3 report's evaluation setting; "card decoding"), one
  invocation: 176 arm-A non-reach prompts x 2 draws + 48 reach prompts x 8 draws (the 42 reach prompts in arm A use
  their first 2 draws for arm-A scoring). 736 draws.
- S5R: `s0500`, card decoding, 48 reach x 8 (384).
- PF: HF scoring under `dpo` (bf16, raw logits, T 1) of, per reach prompt, the first 4 stored strict-pass `s2300`
  draws and the first 8 stored `dpo` T 1.0 draws (the card-decoded group is descriptive and the first thing cut).
- Released only by triggers (orchestrator decides at release): C3 `s2300` card decoding, arm A x 2; C4 `dpo` pure T
  0.6 / top_p 1.0, arm A x 2 (trigger A); C2 `dpo` T 0.8 / top_p 1.0, arm A x 2 (trigger B).
- Baselines (not re-sampled): Dan's stored T 1.0 draws (dpo arm A 16 per prompt, s0500 arm A 8 and reach 32,
  s2300 arm A 8 and reach 32); M3's E0' as the fresh `dpo` T 1.0 reach baseline (proxy: Stage 0c's 0 of 84 if E0'
  is not graded and identity-checked when C1 is graded).

## Primary readouts

1. Delta(dpo, C1) = mean over the 218 arm-A prompts of (pass rate under C1, first 2 draws) minus (stored T 1.0 rate,
   all 16 draws), paired prompt bootstrap (10,000, seed 0), 95 % interval. pass@1 leads.
2. share_RL = Delta(dpo, C1) / 39.3 points (the stored dpo -> s2300 arm-A gap), joint bootstrap of numerator and
   denominator.
3. Reach: per-prompt rate difference C1 minus E0' (8 draws each), prompt bootstrap, plus raw pass counts; S5R against
   its stored 4.5 % (32 draws) and "any pass in 8" against the first 8 stored draws.

Every cell also reports: draws cut at 32,768, draws without `</think>`, the frozen repetition flag
(`tools/temp_repeat_flag.py`, sha256 in every sidecar), response tokens (mean, p50, p90, p99). Loops, cuts and
missing `</think>` count as failures in the headline.

## Rules fixed now

- Trigger A (releases C3 + C4): the 95 % interval of Delta(dpo, C1) excludes 0 (either sign, no floor), or the reach
  rule: C1 reach has >= 3 passes from >= 2 prompts and C1 passes minus E0' passes >= 3 (proxy baseline: >= 3 passes
  from >= 2 prompts). Until C4 exists, every headline says "card decoding", never "temperature" or "sharpening".
- Trigger B (releases C2): trigger A did not fire and, on C1's arm-A draws, the repetition-flag share is >= 1.5
  points above the same detector's rate on stored dpo T 1.0 arm-A draws, or the cut-at-cap share is >= 1.5 points
  above the stored 0.06 %.
- Labels: an arm-A effect inside its interval but below +4 points is "detected, small" (under 10 % of the gap).
  Every null is reported with its minimum detectable effect (about 3.3 points at dpo arm A; about 3.2 points for
  S5R reach at 8 draws).
- PF verdict (feeds only a possible power-sampling proposal): TAIL if, for the full response, either S_pt or S_adj
  has an upper 95 % bound below 0 or either P_low is at least 50 %; PASS only if both S have upper bounds at or
  above 0 and both P_low are at most 25 %; else UNRESOLVED. Printed by `tools/temp_pf_stats.py` from frozen
  thresholds.
- Secondary split (dpo draws 1-8 define "ever passed" 100 / "never passed" 118 prompts; baseline from draws 9-16):
  reported as secondary only.
- Nothing is chosen on the reach prompts. No setting is tuned.

## Priors (orchestrator's and auditors', written before data; not results)

- The Olmo 3 card reports `dpo` IFBench 34.4 at card decoding against Dan's 28.0 strict / 33.8 loose at T 1.0 on all
  300 prompts (different harness), so the auditor expects at most about 0-6 points at `dpo` arm A, i.e. at most about
  15 % of RL's gain. The orchestrator expects the same order: Delta(dpo, C1) between -3 and +6 points, reach C1 at
  or below 1 % of draws, and S5R between 3 % and 10 %.
- The report's footnote 71 (thinking models "degenerate quickly" at low temperature) makes a negative Delta at T 0.6
  possible; that would also be a result.

## What would and would not follow

- A null at card decoding: sharpening `dpo` to the card's own setting does not recover RL's gain on this checker. It
  does not rule out power sampling (Ji et al. Table 1, DeepSeek-Math-7B: low T closes 3 % of the gap, power sampling
  78 %).
- A positive Delta: part of the arm-A gain is reachable by decoding alone, measured against s2300 at T 1.0; whether
  the gap at matched decoding shrinks needs C3, and whether it is temperature rather than top_p needs C4.
- Checker passes only; nobody reads whether passes are genuine.

## Amendment 1 (2026-11-27 2:29 am PT by `date`, before any pod request; the heading first said 2:52 am, a typing slip, corrected at 2:35 am)

The build (`tools/temp_triggers.py`) accepts M3's E0' as the reach baseline only if every recorded check passes; three
fields (`engine_config_sha256`, `dtype`, `enforce_eager`) and the template sha are recorded by neither the sampler
sidecar nor the pre-flight report, so they cannot be verified. Fixed now: the trigger script is run with
`--allow-unverified` for exactly these fields (E0' and the pre-flight both run the same patched engine on the same
`dpo` weights in bf16 with Dan's defaults); every other check must pass (sidecar status done, T 1.0 / top_p 1.0, no
LoRA, 32,768 cap, pre-flight `checks.zero_identity` true, at least 384 graded rows). Otherwise the pre-set proxy
applies. The template is checked separately through Dan's own meta (`prompt_sha256_by_pid` against his stored jobs).

## Amendment 2 (2:36 am PT by `date`, before any pod request): when the triggers are evaluated

C1 runs first on the pod, before M3 has produced E0'. Fixed now: the arm-A part of trigger A and trigger B are computed
as soon as C1 is graded; the reach part of trigger A, and every reach comparison of C1 with a T 1.0 baseline, are
computed once, after M3's E0' job has ended (or M3's sampling has been abandoned), with E0' if it is accepted under
Amendment 1 and the proxy otherwise. Trigger-released cells run last on the pod in any case, so this costs no pod time.

## Amendment 3 (5:25 am PT by `date`, before the PF job and before E0' exists): HF self-test fallback and E0' acceptance

Seen when writing: rp-m2a data only; no PF run and no E0' exist. Binding by the coordinator's decision after the audit
(`discourse/gate_amendments_audit.md` G8).
- `tools/hf_score.py` self-test (two HF passes over the first sequences in one process, `torch.equal`): if not identical,
  measured instead the max |delta| of per-token log-probs; pass if <= 1e-3 (the pre-flight's pre-registered prefix-cache
  repeat tolerance), recorded as `selftest_path: fallback`. Real failure: above 1e-3, or non-finite: exit 5, PF not
  used, I stop and report.
- E0' is accepted as the reach baseline only if, in addition to Amendment 1's list, the H200 pre-flight has
  `zero_compiled_within_tol` true and `pass_chosen` true (`zero_identity` is now the eager check; PREREG_steer Amendment 2).
