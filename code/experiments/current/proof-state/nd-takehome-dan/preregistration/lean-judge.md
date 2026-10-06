# Pre-registration — run `lean-judge`

- Run id: `lean-judge`. Role: executor. Repository `~/work/lean-judge`, branch `dan` of the fork
  (`dan-pandori/nd-takehome`), base `origin/dan` @ `3c0c361`.
- Written 2026-09-27, before any pod of this run.
- Pod budget: **8 pod-hours / $4** (registered with `podbudget lean-judge --set 8 4`). Hard stop 24 h.
- This is an engineering run with a measured acceptance test, not an experiment.

## Question

Can `nd_verify` be removed from every judging path on the fork's `dan` branch — reward,
evaluation, coverage, counting — and replaced by Lean alone (Dan, 2026-09-27), **without losing a
single proof that the old Lean ∧ `nd_verify` gate counted**, without leaking a verdict marker into
expert-iteration training text, and without making the in-loop judge slower?

## Design I will actually run

1. **`lean_judge.py`** — `verify_text(text) -> (ok, reason, n_lines)`, a drop-in for
   `nd_verify.verify_text`, plus `judge_many(pairs)` for batching. Decision order:
   (a) `LEANPARSE`/`LEANREJ` marker on the ND string ⇒ reject;
   (b) verdict registered by `lean_gate` for this `(prompt, ND string)` ⇒ that verdict;
   (c) otherwise translate with `nd2lean.translate` and check in Lean core, **batched** through
   `lean_gate`'s chunked checker.
   `n_lines` = `;`-count of the ND body, the same expression `nd_verify.verify_text` uses.
2. **`lean_gate.gate()`** stops calling `nd_verify`; it registers per-sample Lean verdicts in an
   in-process registry keyed on `(prompt, ND string)` and returns **clean ND strings for accepted
   samples** (marker only on rejects).
3. **Six import swaps**: `expert_iter.py`, `ladder_ei.py`, `coverage.py`, `eval_set.py`,
   `eval_targets.py`, `grpo.py` import `verify_text` from `lean_judge` instead of `nd_verify`.
   Three of them judge strings the gate never saw and are additionally **batched**:
   `expert_iter`'s hindsight relabelling (a rewritten theorem `newp`), `coverage.py` (uses
   `generate_ids`, never the gate, and judges `norm(nd)`), `grpo.py` (uses `generate_ids`).
4. **`lean_check.py`** ported from `origin/dan_lean_only` with `Not.elim`, `Not.intro`, `And.elim`,
   `Iff.elim` added to the elaborated-term allowlist, so the free-form judge accepts the same
   inferences the `lean_seq` grammar can reach through `.elim`.
5. `nd2lean.py` keeps rendering `BOTE` as `.elim`; the `False.elim` "fix" is **not** ported.
   `minlen.py`'s `L_true` labels stay ND-derived and are labelled upper bounds under Lean.
   Term size is not computed in the loop.

## Expected results (numeric / falsifiable) — one line per acceptance test

| # | Test | Pre-registered expectation |
|---|---|---|
| 1 | No judging path imports `nd_verify` | grep test passes: **0** of {6 loop files, `lean_gate.py`, `lean_judge.py`} import or call `nd_verify`; the test fails loudly if one does |
| 2 | Zero regressions on stored samples the old gate counted (`found_*.jsonl` accepted proofs + coverage `proofs[]`, ≥ 50,000 records) | **0 losses** (new judge accepts 100.0 %). Falsified by ≥ 1 loss |
| 3 | Stored `nd_rej & lean_ok` disagreements now counted | **100 %** accepted by the new judge; the Lean-only excess over the old gate is **0.01–0.07 %** of distinct checked texts in the source runs; classification dominated by omitted premise re-statement (expect **> 80 %**) with `Not.elim`-type BOTE the rest |
| 4 | Line counts agree with `nd_verify`'s on proofs both accept | **100 %** identical `n_lines` |
| 5 | One expert-iteration round end to end on a GPU pod, existing `lean_seq` checkpoint | **0** occurrences of `LEAN` in round `found_*.jsonl` proof text; accepted-distinct count ≥ the old gate's on the same samples, excess entirely in the Lean-only class; ≥ 1 hindsight-relabelled acceptance checked by hand |
| 6 | Throughput on a fixed stored batch | new judge **≥** as fast as the old path (it drops the `nd_verify` pass): registry-hit path **< 0.05 s / 1,000 strings**; fallback (nd2lean + batched Lean) reported separately, expected **10–120 s / 1,000 strings** on 2 vCPU |
| 7 | `lean_check.py --selftest` | **all** cases pass, including a `Not.elim` case now **accepted** and `sorry` / `simp` / `Classical.em` / library lemmas still **rejected**; the canonical example `~P ⊢ P > Q` via `N2 ( P > Q ) : BOTE N1` accepted by **both** the `lean_seq` path and `lean_check` |

Additional pre-registered quantity, because the design depends on it (pitfall 3): the number of
stored coverage proofs where Lean's verdict on the **literal** sampled text differs from its verdict
on `nd2lean.translate(prompt, norm(nd))` is expected to be **0** (the two differ only by hypothesis
renaming, which Lean's verdict is invariant to). If it is not 0, `coverage.py` judges before
normalising instead, and I report the count.

Registry-conflict count (two literal texts denoting the same ND proof with different Lean verdicts)
expected **0**.

## Stop rule

- Stop and report when tests 1–7 have run, or at 8 pod-hours / $4 / 24 h, whichever comes first.
- Test 5 is the only test needing a GPU pod; if no `lean_seq` checkpoint can be fetched from the
  bucket within 1 pod-hour, test 5 is run on the smallest available checkpoint and the deviation is
  recorded in `log.md` rather than the test dropped.
- A failure of test 2 (any loss) blocks the change: I would report it and not swap the imports.

## Model labels

No model is trained in this run. Test 5 uses an existing **`lean_seq`**, **from-scratch** checkpoint
(`ds-composition`/`ds-rendering` C0 class, ~19 M parameters, trained on this project's Stage-1
Lean-format pool), fetched from `hf://buckets/dan-pandori/nd-rl/`. Every number in tests 2–4 and 6 is
a property of **stored samples from runs `cap-horizon`, `noise-floor`, `ds-rendering`** (all
`lean_seq`, from-scratch), not of any model, and is labelled with its source file in `numbers.md`.
Stored verdicts were produced under **Lean ∧ `nd_verify`** (pre-2026-09-27); new verdicts are under
**Lean alone**.
