---
written_on: 2026-09-27
written_by: agent:claude
run: lean-judge
---

# Lean alone decides

Dan, 2026-09-27: **`nd_verify` judges nothing in a new run** — not the in-loop reward, not evaluation, not
coverage, not counting. It stays in this repository, unmodified, only so that pre-2026-09-27 numbers can be
reproduced. Decision and reasons: `nd-rl/docs/project_strategy/2026-09-20-lean-default.md`, update of
2026-09-27. Every number measured before that date was measured under **Lean ∧ `nd_verify`** and must be
labelled with that when it is compared with a number measured after it.

## What decides now

| model surface | judge | file |
|---|---|---|
| `lean_seq` (from-scratch models) | the literal sampled text parses in the strict `lean_seq` grammar **and** Lean 4 core accepts it | `lean_tok.decode` → `lean_gate.gate` → `lean_judge.verify_text` |
| ND strings the gate never saw (hindsight relabels, dataset records, `coverage.py`, `grpo.py`) | `nd2lean.translate` + Lean 4 core, batched | `lean_judge.judge_many` |
| free-form Lean (pretrained models) | `lean_check`: elaborated-term allowlist, axioms ⊆ {`propext`, `Classical.choice`, `Quot.sound`} | `lean_check.py` |

`lean_judge.verify_text(text) -> (ok, reason, n_lines)` is a drop-in for `nd_verify.verify_text`, so each loop
file changed only its import. `judge_many(pairs)` is the batched form and is what every loop uses.

Decision order for one `(prompt, ND proof)`:

1. the ND string carries a marker — `LEANPARSE <reason>` (the sampled tokens are outside the grammar) or
   `LEANREJ <nd>` (the text parsed but Lean rejected it) — ⇒ **reject**;
2. `lean_gate.gate` registered a Lean verdict for this `(prompt, ND string)` ⇒ **that verdict** (a dictionary
   lookup: the judging pass after `sample.generate()` runs no second Lean process);
3. otherwise translate with `nd2lean.translate(..., require_all_pr=False)` and check in Lean core, **batched**.

`n_lines` is the `;`-count of the ND body, the same expression `nd_verify.verify_text` returns. Term size is
**not** computed in the loop (it would cost a second Lean pass per sample); counting and analysis compute it
post hoc with `lean_check.py` on the proofs they count. `minlen.py`'s `L_true` labels stay **ND-derived**, so
under Lean they are **upper bounds** on the true minimal length; the pools were not relabelled in this run.

## The grammar is the allowlist (the judge's soundness rests on this)

Verified against `lean_tok.py` in this run: the `lean_seq` vocabulary is **107 tokens** —
`2` specials + `11` formula symbols + `6` prompt symbols + `16` term/tactic symbols + `h1…h8` + `n1…n64`. The
only tokens that name a Lean constant or a projection are

```
P  Q  R  S  False  Prop        (type formers and atoms — not inferences)
.1  .2  .elim  Or.inl  Or.inr  Or.elim  Classical.byContradiction
```

There is no `sorry`, no `simp`/`decide`/`omega`/`tauto`/`exact?`, no `Classical.em`, no `Decidable.*`, no
library lemma, and no identifier token at all beyond `h1…h8` and `n1…n64`. The only tactics are `have`,
`exact` and `by`. So beyond the ND rules, the inferences reachable are exactly: function application (modus
ponens / ¬-elimination), `fun` (→/¬ introduction), the anonymous constructor `⟨_,_⟩` (`And.intro`, or
`Iff.intro` when the target is an `↔`), the two `Or` constructors, `Or.elim`, `Classical.byContradiction`, and
whatever `.1`/`.2`/`.elim` resolve to by the head type of their receiver — `And.left`/`And.right`,
`Iff.mp`/`Iff.mpr`, and `False.elim` / `Not.elim` / `Or.elim` / `And.elim` / `Iff.elim`. Every one of those is
a sound constructor or eliminator of a core logical connective. **`Not.elim` is allowed** (Dan, 2026-09-27):
`n.elim` on `n : ¬A` is a valid proof that ND's `BOTE` rule happens to forbid, because `BOTE` must cite a line
that is literally `F`. `nd2lean.py` keeps rendering `BOTE` as `.elim`; the `False.elim` "fix" from
`dan_lean_seed2` / `dan_lean_only` is **not** ported.

One more thing the argument needs: a *truncated* text can be accepted by Lean's parse-error recovery. It
cannot reach Lean here, because the strict-grammar parse runs **first** (`lean_tok.inverse` consumes the whole
token sequence, with balanced boxes and an `<eos>`), and a text that fails it becomes `LEANPARSE` and is never
Lean-checked. `lean_check.py`, which does see free-form text, rejects on **any** error attributed to a
theorem's lines for the same reason.

## The four pitfalls and how each is handled

1. **Accepted proofs become training data.** Whatever the judge accepts is written to the round files and
   trained on, so a verdict prefix on an accepted sample would be trained on. `lean_gate.gate` returns
   **clean ND strings** for accepted samples; only rejects carry `LEANREJ ` / keep `LEANPARSE `. Tested by a
   unit test (`tests/test_lean_only_judge.py`) and end to end by acceptance test 5.
2. **Hindsight relabelling** re-checks a proof against a *rewritten* theorem (`newp`), which the gate never
   saw, so no cached verdict applies. `expert_iter.relabel` was split into `relabel_candidate` (structural,
   no judging) and `relabel_batch` (one batched Lean run for every candidate of the round).
3. **`coverage.py` normalises before judging.** The verdict is keyed on the **normalised** string: the file
   judges `norm(nd)` with `judge_many`, once per distinct normalised string. `nd2lean`'s rendering of
   `norm(nd)` differs from the literal sampled text only by hypothesis renaming, which Lean's verdict is
   invariant to — measured, not assumed (§Results, test 3b).
4. **Lean costs ~1 s per process start.** Nothing judges one string at a time inside a loop. `eval_set.judge`,
   `eval_targets`, `grpo` (both sites), `coverage` and `expert_iter`'s relabelling all call `judge_many` once
   per batch; `lean_gate.check_sources` chunks (400 theorems per Lean process) and runs the chunks in a thread
   pool. A test asserts no deeply-indented one-string `verify_text` call remains in the six loop files.

## Files changed

- **`lean_judge.py`** (new): the judge. Registry, marker handling, batched nd2lean + Lean fallback, `stats()`.
- **`lean_gate.py`**: no longer imports `nd_verify`; registers Lean verdicts; returns clean ND strings for
  accepted samples; `check_sources()` generalises the chunked checker to multi-line sources (needed by the
  fallback, since `nd2lean.translate` emits indented boxes); prelude gained `set_option maxRecDepth 4000`;
  `$LEAN_GATE_DUMP` appends every distinct checked `(prompt, literal text, ND string, verdict)` as an audit
  trail; rejects go to `<log>.leanrej.jsonl`.
- **Six loop files** now import the Lean judge: `expert_iter.py`, `ladder_ei.py`, `coverage.py`,
  `eval_set.py`, `eval_targets.py`, `grpo.py`. `eval_set.judge`, `eval_targets`, `grpo` (×2), `coverage` and
  `expert_iter`'s relabelling were rewritten to batch.
- **`coverage.py`**: `_verify_one` (judge + classify, in a fork pool) became `_class_one` (classify only);
  judging is one `judge_many` call per theorem in the main process, before the pool.
- **`nd2lean.py`**: `translate(..., require_all_pr=False)` added — nothing else. Lean does not require a
  premise to be re-stated; `lean_tok.inverse` only accepts `h`-citations that are a prefix `h1…hk` in order,
  so the k-th `PR` line is still the k-th declared premise and the rendering is unchanged. The remaining
  premises stay declared and unused, which cannot make a false theorem provable. Without this, the judge would
  reject the largest class of proofs Lean accepts and `nd_verify` rejects.
- **`lean_check.py`** ported from `origin/dan_lean_only` with `Not.elim`, `Not.intro`, `And.elim`, `Iff.elim`
  added to the allowlist, and `nd_verify` moved behind an opt-in `--compare_nd` flag.
- **`tests/test_lean_only_judge.py`** (new), **`lj_regress.py`** (new, the acceptance harness),
  **`pod/lj/{setup,t5}.sh`** (new).
- **`sample.py`**: comments only.

One behaviour change a reader of old stats files should know: the `reasons` field of
`eval_set.judge`'s rows (and therefore of every `round_<r>.json`) now carries the **judge's**
reasons — `lean rejected`, `lean_seq parse: <reason>`, `nd2lean: <reason>` — not `nd_verify`'s ND rule
names. Reason *histograms* are therefore not comparable across 2026-09-27; the accept/reject counts
are (Lean alone accepts a superset).

## What still uses `nd_verify`, and why

| file | use | why it stays |
|---|---|---|
| `nd_verify/` | the ND checker itself | unmodified, so pre-2026-09-27 numbers can be reproduced |
| `lean_tok.py`, `prune.py`, `nd2lean.py`, `minlen.py`, `patterns.py` | `from nd_verify.verify import parse_proof_tokens / parse_formula` | **parsers, not judging**: the ND grammar has one implementation and there is no reason to fork it |
| `patterns.py` (`verify_text`) | its own hand-written self-test fixtures | asserts the classifier's test proofs are well-formed ND; judges nothing in a run |
| `minlen.py` | checks the proofs its ND search finds | it *is* an ND prover; its `L_true` labels are ND-derived and are **upper bounds under Lean**. Relabelling the pools in both units is a separate job and was not done here |
| data tooling: `gen.py`, `make_coverage_sets.py`, `patterns2.py`, `precursors.py`, `dsg_assemble.py`, `chain_pool.py`, `ore_pool.py`, `reductio_pool.py`, `required_pool.py`, `textbook_pool.py`, `novelty.py`, `necessity.py`, `run3_inject.py`, `intuit.py`, `try_proof.py`, `score_test.py`, `verify_cli.py`, `train.py`, `tokenizer.py` | validate *generated* ND proofs | These generate ND proofs by construction and check their own output. Switching them costs a Lean pass over millions of records and changes no count: `nd2lean` + Lean agreed with `nd_verify` on 181,464/181,464 and 253,397/253,397 pool proofs (round-2 run-1, `lean-only`), and Lean accepts a **superset**, so a generator that passes `nd_verify` passes Lean. They are listed here rather than switched, as the brief allows. **`gen.py` is imported by `expert_iter.py` for `canon_key`**, so `nd_verify` is still a transitive import of `expert_iter`; the acceptance test is about the judging path in the file that decides, and `canon_key` judges nothing |
| analysis and review scripts (`*_analysis.py`, `review_*.py`, `nf_*.py`, `dsg_*.py`, `lean_format_analysis.py`, `peek_samples.py`) | re-derive or compare old numbers | they reproduce results measured under Lean ∧ `nd_verify`; that is their purpose |

## Acceptance-test results

Model labels: no model was trained for these numbers except in test 5. Tests 2–4 and 6 are properties of
**stored samples** from runs `cap-horizon`, `noise-floor`, `ds-rendering`, `efficiency` and `lean-format` —
all **`lean_seq`**, **from-scratch**, ~19 M parameters, trained on this project's Stage-1 Lean-format pools —
whose stored verdicts were produced under **Lean ∧ `nd_verify`**. Test 5 is on
`ckpts/dsc/stage1_a3_s0.pt` (run `ds-composition`, arm a3, seed 0: `lean_seq`, from-scratch, ~19 M
parameters, Stage-1 pool `train_a3`). Sources for every count: `numbers.md` § lean-judge.

| # | test | pre-registered expectation | measured | verdict |
|---|---|---|---|---|
| 1 | no judging path calls `nd_verify` | 0 of the 8 files | **0** of 8 (`ast`-parsed, not grepped); 26/26 checks pass on the VPS and the pod | **pass** |
| 2 | zero regressions on stored samples the old gate counted | 0 losses, ≥ 50,000 records | **0 losses** on **281,817** distinct `(prompt, ND proof)` the old gate counted, from 1,533,741 records of 1,000 Lean-gated `found*.jsonl` and 34 Lean-gated coverage files: **281,817 accepted, 100.000 %**. A further 1,195 collected records are **not ND proofs** (run `efficiency` stores literal Lean text in `proof`) and are excluded — `nd_verify` rejects all 1,195, so the `nd=True, lean=False` cell is **0** | **pass** |
| 3 | the Lean-only class is now counted | 100 % accepted; excess 0.01–0.07 % of distinct checked texts; > 80 % omitted premise re-statement | **6,419 / 6,419 = 100 %**. Classes of all 8,514: omitted premise re-statement **41.7 %**, `BOTE` on a non-`F` line (`Not.elim`) **25.9 %**, no ND denotation (excluded) 24.6 %, other 6.2 %, `NEGE` on `A`/`A > F` 1.6 % | **pass**, one expectation **wrong**: the > 80 % figure came from `ds-generator`'s disagreements; over all five runs the class is 41.7 % and `Not.elim` is 25.9 % |
| 3b | pitfall 3: normalising before judging | 0 disagreements | Lean on the literal text vs on `nd2lean(nd)` vs on `nd2lean(norm(nd))`: **6,419 / 6,419 agree, 0 disagreements** | **pass** |
| 4 | line counts agree with `nd_verify`'s | 100 % | **281,817 / 281,817 = 100 %** on proofs both accept; 0 disagreements | **pass** |
| 5 | one expert-iteration round, no marker in training data | 0 `LEAN*` in the round files; acceptance ≥ the old gate's, excess in the Lean-only class; ≥ 1 hindsight relabel by hand | **0 / 0 / 0** markers in `found_1` (228), `found_transfer_1` (56), `mix_1` (3,912). On the same 3,506 distinct samples: old gate 2,348, new judge 2,348 — **0 losses, 0 excess** (at the stored 0.01–0.07 % rate, 3,506 samples predict 0.4–2.5, so 0 is the expected order). **166** hindsight relabels against theorems the gate never saw accepted, 80 rejected; one shown in Lean in `numbers.md` § L5. 0 registry conflicts | **pass** |
| 6 | throughput: the new judge no slower | registry hit < 0.05 s / 1,000; fallback 10–120 s / 1,000 | per 1,000 strings: old `nd_verify` pass **0.0790 s** (dropped), new registry hit **0.0123 s**, marker 0.0021 s, fallback **3.556 s**. The Lean run on the literal texts is unchanged, so the judging step is **6.4× faster**; the fallback beat its expectation by 3–34× | **pass** |
| 7 | `lean_check --selftest` with the new cases | all pass; the canonical example accepted by both paths | **39 / 39** on Lean 4.34.1 (VPS) and 4.34.0 (pod). `~P ⊢ P > Q` via `BOTE N1`: `lean_judge` returns `(True, '', 2)`, `lean_check` accepts it at term size 1, `nd_verify` returns `rule check failed: BOTE (line 2)` | **pass** |

Sources for every number: `numbers.md` § lean-judge (L0–L8). Raw files: `artifacts/lj/`, and the bucket
`hf://buckets/dan-pandori/nd-rl/lean-judge/artifacts`.

## What a run on this branch must now say

- A number measured here is under **Lean alone**; a number from before 2026-09-27 is under
  **Lean ∧ `nd_verify`**. Say which when you compare across that date. Lean alone accepts a
  **superset**, so a Lean-only rerun can only raise acceptance counts.
- **Report term size beside lines** (`lean_check.py` computes it). Lines are not invariant: a Lean proof
  may take fewer steps than the shortest ND proof, and 41.7 % of the Lean-only class omits premise
  re-statement lines, which term size does not count (`R`/`PR` = 0).
- The ladder's `L_true` labels come from `minlen.py`, an ND prover, so under Lean they are **upper
  bounds** on the true minimal length. The pools were not relabelled in this run.

