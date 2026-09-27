# Run brief: lean-judge — make Lean the only judge on the fork's `dan` branch

Run id `lean-judge`. Repository `~/work/lean-judge` (branch **`dan`** of the fork — the new
integration branch, currently `origin/dan_noise-floor`'s tip; you commit and push to `dan`
directly). Role: executor. Pod budget **$4**, ceiling **8 pod-hours**, hard stop 24 h. This is an
engineering run with a measured acceptance test, not an experiment; most of it needs no GPU.

Read first: the update of 2026-09-27 in `~/nd-rl/docs/project_strategy/2026-09-20-lean-default.md`
(the decision and its reasons), `~/nd-rl/docs/proposals/orchestration/AGENT_POLICY.md`,
`lean_gate.py`, `sample.py` around its gate call, and `lean_check.py` on `origin/dan_lean_only`.

## The decision you are implementing (Dan, 2026-09-27)

**Lean alone decides whether a proof counts.** `nd_verify` must not judge anything in new runs —
reward, evaluation, coverage, counting. It stays in the repository, unmodified, only to reproduce
old numbers. **`Not.elim` is allowed**: `n.elim` on `n : ¬A` is a valid proof that ND's `BOTE`
format happens to forbid. `nd2lean.py` keeps rendering `BOTE` as `.elim` — do **not** port the
`False.elim` "fix" from `dan_lean_seed2`/`dan_lean_only`.

## What exists and where it decides

`lean_gate.gate()` runs inside `sample.generate()` for every `lean_seq` generation. It checks each
distinct (prompt, literal Lean text) with bare Lean, *also* runs `nd_verify` on the denoted ND
proof, and returns the ND strings, prefixing `LEANREJ ` where Lean rejected. **The actual decision
is then made downstream by `nd_verify.verify_text(prompt + ' ' + s)`** in `expert_iter.py`,
`ladder_ei.py`, `coverage.py`, `eval_set.py`, `eval_targets.py`, `grpo.py` — so today a proof
counts iff both accept.

The strict `lean_seq` grammar is the allowlist for model text: its 107-token vocabulary has no
`sorry`, no tactic automation and no library lemma, so bare Lean on grammar-parsed text reaches
only the ND rules plus the eliminators `.elim` resolves to. Verify this claim yourself against
`lean_tok.py` and state it in `LEAN_JUDGE.md`; the judge's soundness rests on it.

## Four pitfalls found while tracing this (design around each, and test each)

1. **Accepted proofs become expert-iteration training data.** Whatever string the judge accepts is
   what gets written to the round files and trained on. A verdict *prefix* on accepted samples
   (`LEANOK …`) would leak into training text. Accepted samples must leave the gate as clean ND
   strings; only rejects may carry a marker.
2. **Hindsight relabelling** (`expert_iter.py` ≈ line 59) re-checks a proof against a *rewritten*
   theorem (`newp`: the premises plus the proof's own last formula). The gate never saw that
   statement, so no cached verdict applies; it needs its own Lean check against `newp`.
3. **`coverage.py` normalises proofs before judging** (`norm(nd)`, start-index renumbering), so a
   verdict cache keyed on the raw ND string misses. Judge before normalising or key on the
   normalised form — and prove whichever you choose gives identical counts.
4. **Lean costs ~1 s per process start.** Any fallback path that checks one string at a time will be
   catastrophically slow inside a loop. Fallback checks (ND strings the gate never saw: hindsight
   relabels, dataset records, token-format samples) must be **batched** through `lean_gate`'s
   chunked checker.

## Suggested design (a suggestion — a better one is welcome; say why)

- `lean_judge.py` exposing `verify_text(text) -> (ok, reason, n_lines)`, a drop-in for
  `nd_verify.verify_text` so each loop file changes only its import, plus `judge_many(texts)` for
  batching. Decision order: `LEANPARSE`/`LEANREJ` marker ⇒ reject; verdict registered by the gate
  for this sample ⇒ that verdict; otherwise translate with `nd2lean.py` and check in Lean (batched).
- The gate stops calling `nd_verify` and registers per-sample Lean verdicts in an in-process
  registry. `n_lines` comes from the ND denotation's line count.
- Term size is **not** computed in the loop (keep the loop fast); counting and analysis compute it
  post hoc with `lean_check` on counted proofs.
- `lean_check.py` from `origin/dan_lean_only`, with `Not.elim` and `Not.intro` added to the
  allowlist, and the other eliminators `.elim` can reach from the grammar (`And.elim`, `Iff.elim`)
  so the free-form judge and the `lean_seq` judge accept the same inferences. Self-tests updated:
  a `Not.elim` case now **accepted**; `sorry`, `simp`, `Classical.em`, library lemmas still
  rejected.
- Data tooling (`gen.py`, `make_coverage_sets.py`, `patterns*.py`, `dsg_assemble.py`, …) validates
  generated ND proofs with `nd_verify`. Switch the ones a new run would use to Lean where it is
  cheap; list the rest in `LEAN_JUDGE.md` with why. `minlen.py`'s `L_true` labels stay ND-derived
  (upper bounds under Lean) — say so; do not relabel the pools in this run.

## Acceptance test (pre-register the expected outcome of each in `preregistration/lean-judge.md`)

1. **No judging path calls `nd_verify`.** A grep-based test in the repository that fails if any of
   the six loop files, the gate, or `lean_judge.py` imports it.
2. **Zero regressions.** On stored samples from recent runs (the gate logs and disagreement files
   under `~/work/cap-horizon/artifacts/kh/`, `~/work/noise-floor/artifacts/nf/`,
   `~/work/ds-rendering/artifacts/dsr/`, or pulled from the bucket): every sample the old gate
   counted (Lean ∧ `nd_verify`) is counted by the new judge. Expected: 0 losses.
3. **The Lean-only class is now counted.** Every stored `nd_rej & lean_ok` disagreement is accepted
   by the new judge; report the count and a classification (`Not.elim`, omitted premise
   re-statement, other). Expected excess over the old gate: ≈ 0.01–0.07 % of distinct checked texts.
4. **Line counts agree** with `nd_verify`'s on every proof both accept (100 %).
5. **No marker in training data.** One expert-iteration round (k small, an existing `lean_seq`
   checkpoint from the bucket, e.g. `ds-composition` C0) end to end on the cheapest pod: round
   files contain no `LEAN*` prefix; acceptance on the same samples ≥ the old gate's, with the excess
   entirely in the Lean-only class. Also one hindsight-relabelled acceptance checked by hand.
6. **Throughput.** Judge a fixed stored batch with the old and new paths: the new must be no slower
   (it drops `nd_verify`); report the fallback path's cost per 1,000 strings separately.
7. `lean_check.py --selftest` passes with the new cases; the canonical example from the decision note
   (`~P ⊢ P > Q` via `N2 ( P > Q ) : BOTE N1`) is accepted by both the `lean_seq` path and
   `lean_check`.

## Budget and pods

Register first: `podbudget lean-judge --set 8 4`. Most of this runs on the VPS (it has Lean at
`~/.elan/bin/lean`; 2 vCPU, so keep regression batches to what finishes in minutes, or use one
cheap pod's cores). Use a GPU pod only for test 5. Record GPU class and the real billed rate.

## Deliverables

`preregistration/lean-judge.md` (before any pod); `lean_judge.py`, the gate change, the six import
swaps, `lean_check.py`, tests, all pushed to `dan`; **`LEAN_JUDGE.md`** at the repo root — one page:
what decides now, the grammar-as-allowlist argument, the pitfalls and how each is handled, what
still uses `nd_verify` and why, and the acceptance-test results table; `numbers.md` § lean-judge
(every count names its source file); `log.md` dated; `STATUS.md` line `LEAN-JUDGE DONE <UTC>`;
bucket `hf://buckets/dan-pandori/nd-rl/lean-judge/artifacts`; pods deleted;
`touch ~/runs/lean-judge/executor.done`. Questions for Dan in `QUESTIONS.md` with your default.
