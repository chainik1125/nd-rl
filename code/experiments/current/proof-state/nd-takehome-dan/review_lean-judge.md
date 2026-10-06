---
written_on: 2026-09-27
written_by: agent:claude
role: reviewer
run: lean-judge
base: 3c0c361 (origin/dan before the run)  head: c3253b2
---

# Review of run `lean-judge`

Phase 1 (recount) was done in `~/review/lean-judge`, a copy of the run's repository with the
executor's write-ups removed, from the pre-registration, the brief, the code and the raw artefacts.
Every number below the heading **§Recount** comes from code I wrote (`~/review/lean-judge/rv/`,
listed at the end) and from my own Lean runs; I did not read `run_lean_judge.md`, `numbers.md`,
`log.md` or `STATUS.md` before writing it.

`LEAN_JUDGE.md` is not in the excluded-write-up list, so it was present in the phase-1 workspace and
I read it. Everything below was nevertheless re-derived independently; where I quote one of its
figures it is labelled as the executor's.

Lean on this VPS is **4.34.1** (the pod used 4.34.0). All my Lean checks use `lean_check.check`,
which gives a **positive per-theorem verdict** (a `nd_check k` command after each theorem), an
**axiom check** (⊆ `propext`, `Classical.choice`, `Quot.sound`) and an **allowlist check** on the
elaborated term — strictly stronger than the gate's absence-of-error test, which is what the run
itself used for tests 2, 3, 3b and 5. `nd_verify` appears below only where I reconstruct the old
gate's count or check `n_lines`; it judged nothing.

---

# §Recount

## Hard constraints

| constraint | check | result |
|---|---|---|
| `nd_verify` unmodified | `git rev-parse origin/main:nd_verify` vs `HEAD:nd_verify` | both `9437bb72b8009d6660dc5d85496ec7e580c815fe` — **identical** |
| `nd_verify` not used as a judge | my own `ast` walk over the 6 loop files, `lean_gate.py`, `lean_judge.py` (`rv/rv_t1.py`): direct imports, `from … import`, and `nd_verify.<attr>` calls | **0 of 8**. The only `nd_verify` call sites reachable from the judge are `nd2lean.main()` (the CLI) and `lean_check.main()` behind `--compare_nd`; neither is on a judging path |
| `artifacts/TEST_RUN_DONE` unchanged | `git log 3c0c361..HEAD -- artifacts/TEST_RUN_DONE` | **0 commits**; md5 `e2eea349c5320ccb597c2279a0391991` |
| no evaluation file read in training code | grep for `targets/test_short`, `targets/test_long`, `targets/test` over every `.py`, and over every file this run touched | **no hit** outside `score_test.py` (the scorer) and `audit_test_overlap.py` (the overlap auditor) |
| at most one test-file run | the run trains nothing but test 5 and reads no test file | **no test-file run in this run** |

No hard-constraint violation. Not quarantined.

**One caveat on the first two rows.** `nd2lean.py` has `from nd_verify import verify_text` at module
level, and `lean_judge` imports `nd2lean`, so **every** judging path transitively imports
`nd_verify` — not only `expert_iter.py` through `gen.canon_key`, which is the only transitive import
`LEAN_JUDGE.md` mentions. Nothing calls it (the name is used only inside `nd2lean.main()`), so this
is a documentation point, not a judging point.

## Collection: what population the acceptance tests are measured on

My own collector (`rv/collect_rv.py`, same two definitions as the pre-registration, written from the
pre-registration rather than from `lj_regress.py`):

| quantity | reviewer | executor (`artifacts/lj/collect.json`) |
|---|---|---|
| Lean-gated `found*.jsonl` files read | 1,284 | "1,000" |
| Lean-gated coverage files read | 102 | "34" |
| records read | 1,863,151 | 1,533,741 |
| **distinct (prompt, ND proof) the old gate counted** | **281,817** (281,198 `found` + 619 `coverage`) | 283,012 in `collect.json`, **281,817** in the test-2 row |
| records in `found*.jsonl` that are not ND proofs | 2,912 records / **1,195 distinct** | 1,195 |
| distinct (prompt, literal text, ND) in the `*.disagree.jsonl` files | **8,514** | 8,514 |
| … of which have an ND denotation | **6,419** | 6,419 |

The distinct counts reproduce exactly. The file and record counts differ because several earlier
runs' artefacts are checked out in more than one worktree, so I read the same file several times;
the de-duplication removes it. **`LEAN_JUDGE.md`'s "1,000 Lean-gated `found*.jsonl` and 34 Lean-gated
coverage files" is not reproducible as stated** — the glob in `lj_regress.py collect` matches 1,284
and 102 today.

`collect.json`'s `accepted_distinct: 283,012` and the test-2 denominator 281,817 differ by exactly
the 1,195 non-ND records, because the exclusion was added to `lj_regress.py` **after** the corpus was
built and `collect` was not re-run. The committed script therefore no longer reproduces the committed
corpus file, and `corpus_accepted.jsonl` itself is gitignored (165 MB, in the bucket). Re-running
`collect` today gives 281,817, which is the number the run reports, so nothing is wrong with the
headline — but the two numbers in the artefacts have no explanation in the files themselves.

Source of the accepted corpus, by the artefact group that produced it (a record is counted under
every group that contains it, so the columns overlap):

| group | run | distinct (prompt, ND) | unique to it |
|---|---|---|---|
| `kh` | cap-horizon | 262,653 | 238,765 |
| `dsc` | ds-composition | 35,441 | 10,561 |
| `lo` | efficiency (lean-only arms) | 15,456 | 1,851 |
| `dsg` | ds-generator | 14,897 | 842 |
| `lf` | lean-format | 11,338 | 770 |
| `nf` | noise-floor | 8,631 | 413 |
| `dsg`, `nf` coverage | | 1,927 + 495 | 209 + 86 |

All the runs `LEAN_JUDGE.md` names do contribute; the corpus is dominated by `cap-horizon`.

## The seven acceptance tests, re-derived

### Test 1 — no judging path calls `nd_verify`

**Reproduces.** 0 of 8 files, by my own AST walk (`rv/rv_t1.json`). The executor's own suite
`tests/test_lean_only_judge.py` also passes on this VPS: **26/26**, re-run by me.

### Test 2 — zero regressions on the proofs the old gate counted

`nd_verify` accepts **281,817 / 281,817** of the corpus, so the corpus really is the old gate's
counted population (Lean had already accepted every one of them at gate time).

My own re-judge of **all 281,817** — `nd2lean.translate(…, require_all_pr=False)` → my own
`lean_check.check` run (positive per-theorem verdict + axiom check + allowlist), 5,372 s of Lean on
this VPS:

```
n_input 281,817   translatable 281,817   accepted 281,817   rejected 0   reject_reasons {}
term size  min 2 / median 6 / mean 6.22 / max 24      n_lines  min 7 / median 10 / mean 10.06 / max 22
```

**0 losses reproduces exactly, under a strictly stronger check than the run used.** Not one of the
281,817 proofs uses a constant off the allowlist or an axiom outside
{`propext`, `Classical.choice`, `Quot.sound`}; the judge's soundness argument holds on the whole
counted population, not only on the grammar.

On a random 1,000 (`rv/rv_t2_pilot.json`): 1,000 translatable, **1,000 accepted, 0 rejected**, no
allowlist or axiom rejection; term size min 2 / median 6 / mean 6.16 / max 22 against `n_lines`
min 7 / median 10 / max 20.

**From the stored literal text.** `found*.jsonl` records keep only the ND proof, but Lean-gated
coverage records keep `lean_text` as well. For all **2,021** coverage proofs that have one (the 619
new to the corpus plus 1,402 already present from `found` files) I ran Lean on the literal text and
on `nd2lean(nd)`: **(accepted, accepted) 2,021 / 2,021**, no allowlist or axiom rejection; term size
min 2 / median 4 / max 9. Together with test 5's 3,506 literal texts and the Lean-only class's 6,419,
that is **11,946 counted proofs re-checked in Lean from their stored literal text**, far above the
100-per-arm the reviewer brief asks for.

**The 1,195 excluded records — the executor's figure is wrong, in the run's favour.**
`artifacts/lj/t2_freeform.json` reports `lean_check` accepting **567** of them and rejecting **628**
for `axiom:[sorryAx]`. I reproduce 567/628 exactly with the executor's wrapper — and the split is
precisely the arm split: every rejected record comes from a `*_seq_*` arm and every accepted one from
a `*_free_*` arm. `lj_t2_freeform.py` wraps every record as `theorem t … := <text>`, but the `_seq_`
arms store a **tactic block** (`have … ; … ; exact n`), which is a parse error after `:=`; Lean's
error recovery then yields `sorryAx`. Wrapping each record for its own arm's surface
(`:= <term>` for `_free_`, `:= by <block>` for `_seq_`) gives **1,195 / 1,195 accepted**
(`rv/rv_freeform.json`; with the wrong wrapper the errors are explicit:
`unexpected token 'fun'; expected '{' or tactic`). `nd_verify` accepts **0** of the 1,195, as claimed.

So the correct statement is stronger than the one in `LEAN_JUDGE.md`: of everything any Lean-gated
run counted, **nothing at all is lost** — 281,817 under the `lean_seq`/ND judge and 1,195 more under
the free-form judge.

### Test 3 — the Lean-only class is counted, and what it is made of

**Accepted: reproduces, and more strongly.** All **6,419** records with an ND denotation are accepted
— and so are all **909** distinct (prompt, ND) proofs and all **6,419** distinct literal texts, with
**no allowlist and no axiom rejection** (`rv/rv_t3b.json`). Term size of the 909: min 0 / median 4 /
mean 4.10 / max 13.

**Denominator.** The 6,419 are **records** (distinct literal texts); they collapse to **909 distinct
(prompt, ND proof)**. `LEAN_JUDGE.md` presents 6,419 as the size of the class; the number of distinct
proofs recovered is 909. Both are in the run's own artefacts (`t3_judge.json`'s
`judge_stats.fallback: 909`), but the write-up reports only the record count.

**Classification.** My classifier (`rv/rv_t3_classify.py`) reports every property that holds rather
than the first one in a precedence order; no record has more than one, so the classes are the same
partition:

| class | reviewer, records | reviewer % | executor % |
|---|---|---|---|
| omitted premise re-statement | 3,524 | **41.4 %** | 41.7 % |
| `BOTE` on a non-`F` line (`Not.elim`) | 2,205 | **25.9 %** | 25.9 % |
| no ND denotation (excluded) | 2,095 | **24.6 %** | 24.6 % |
| other | 557 | **6.5 %** | 6.2 % |
| `NEGE` on `A` / `A > F` | 133 | **1.6 %** | 1.6 % |

The 0.3 pp difference is a bug in `lj_regress.classify_leanonly`: for a premise-free theorem
(`THM SEQ … PRF`) the expression `prompt[4:].split(' SEQ ')[0].split(' , ')` returns the whole
prompt as one "premise", so the proof is scored as having omitted a premise re-statement. 874 of the
8,514 records are premise-free and **26** of them are misclassified this way. Small, but the same
expression is used by `lj_t5_compare.py`.

**Two things the write-up's classification hides.**

1. **At the distinct-proof level the ordering inverts.** Over the 909 distinct proofs:
   `Not.elim` **445 (49 %)**, other **245 (27 %)**, omitted premise re-statement **215 (24 %)**,
   `NEGE` 4. "41.7 % of the class omits premise re-statement" is a property of how often such
   samples were re-drawn, not of how many distinct proofs the change recovers.
2. **"other" is not a residual.** Running `nd_verify` on its 232 distinct proofs gives
   `IMPI` 116, `DN` 78, `NEGI` 10, `ORI1` 8, `R` 8, `ORI2` 5, `ANDI` 3,
   "final formula is not the conclusion" 3, `ANDE1` 1 — and every example I read is the same
   single cause: **`¬A` and `A > F` are the same proposition in Lean and different formulas in ND**
   (e.g. `N6 (~(~Q)) : IMPI N4 N5` from an assumption `(~Q)` and `F`). Together with the `NEGE` row
   that is **664 records / 7.8 %** of one further nameable, sound class, not noise.

**Pre-registered excess rate.** Over every Lean-gated gate log on this box (225 distinct logs):
`nd_rej & lean_ok` **14,995** of **70,882,951** distinct checked texts = **0.021 %**, inside the
pre-registered 0.01–0.07 %. In the same 70.9 M, `nd_ok & lean_rej` is **0** — the empirical basis for
"Lean accepts a superset" holds at that scale.

The pre-registered "> 80 % omitted premise re-statement" is a **miss** (41.4 % by records, 24 % by
distinct proofs). The executor reports it as a miss, with the reason.

### Test 3b — normalising before judging

**Reproduces.** For each of the 6,419 records I ran Lean on (i) the literal sampled text,
(ii) `nd2lean(nd)`, (iii) `nd2lean(norm(nd))`: **(True, True, True) 6,419 / 6,419, 0 disagreements**
(`rv/rv_t3b.json`). Pitfall 3 is closed.

### Test 4 — line counts

**Reproduces.** On all 281,817 proofs both checkers accept, `nd_verify`'s `n_lines` equals the
`;`-count the judge returns: **281,817 / 281,817**, 0 disagreements (my own count, 42 s).

### Test 5 — one expert-iteration round

Model label, from `artifacts/lj/t5_ei/args.json`: `ckpts/stage1_a3_s0.pt`, `lean_seq`,
from-scratch, Stage-1 pool `train_a3`; k = 16, T = 0.8, seed 0, one round.

| claim | reviewer |
|---|---|
| no `LEAN*` marker in the round's files | **0** in `found_1.jsonl` (228 lines), `found_transfer_1.jsonl` (56), `mix_1.jsonl` (3,912) and `round_1.json` — my own grep for `LEAN`, `LEANPARSE`, `LEANREJ` over the whole line, not just the proof field |
| the dump is the round's whole judged population | 3,506 records = 2,573 + 685 + 48 + 200, the four `distinct_checked` values in `t5_gate.jsonl`. 3,496 distinct (prompt, text); 848 distinct (prompt, ND) |
| stored Lean verdicts | I re-ran Lean myself on all 3,506 stored literal texts: **3,506 / 3,506 agree** with the dump, 2,348 accepted, no allowlist or axiom rejection |
| old gate vs new judge | `nd_verify` × my Lean run: `(True,True) 2,348`, `(False,False) 1,158`, `(True,False) 0`, `(False,True) 0` ⇒ **old 2,348, new 2,348, 0 losses, 0 excess** — exactly the executor's numbers. At the distinct level 484 = 484 |
| line counts | 2,348 / 2,348 |
| registry conflicts | 0 in all four gate calls |
| term size | accepted samples: min 1 / median 4 / mean 4.26 / max 7 |

0 excess is unremarkable: at the 0.021 % stored rate, 3,506 distinct texts predict 0.7.

**Hindsight relabelling — the pre-registered clause is not met by the round, and cannot be.**

The executor's figure (246 candidates, **166 accepted**, 80 rejected) reproduces exactly under my own
rewrite and my own Lean run: 246 candidate pairs → 78 distinct (rewritten theorem, proof), of which
**38 distinct / 166 pairs accepted**, 39 distinct / 80 pairs rejected (`rv/rv_relabel.json`).

But that number is measured by `pod/lj/t5_relabel.py` on the **gate dump**, which stores the *clean*
ND string of every checked sample. The expert-iteration loop does not see those strings. The round's
own summary says:

```
round_1.json:  "relabelled_new": 0,  "relabelled_total": 0
```

and it is 0 **by construction**, not by chance:

* the new `lean_gate.gate` returns `'LEANREJ ' + nd` for **every** Lean-rejected sample (the old gate
  marked only the `nd_ok & lean_rej` cell, which is empty);
* `expert_iter.relabel_batch` passes that same string to `judge_many`, whose **first** rule is
  "the ND string carries a marker ⇒ reject", before the rewritten theorem is ever looked at;
* hindsight relabelling is, by definition, about samples that **failed** their prompted theorem.

Measured on the round's own 187 relabel candidates (`rv/`, inline experiment): `relabel_batch` on the
clean strings accepts **125**; on the strings the loop actually receives it accepts **0**. All 125
come from Lean-rejected samples; none from an accepted one. Three of them clear every other filter
the loop applies (`n_lines ≥ 7`, canonical key not in `eval_keys` incl. `validation_36`) and were
still not recorded.

So pitfall 2's fix works on the batching (the loop does make one batched Lean run for the round's
candidates) but the relabelling itself is dead on this branch. The fix is one line: strip a leading
`LEANREJ ` before relabelling — the marker means "Lean rejected this text *for the prompted
theorem*", which says nothing about the rewritten theorem. `LEANPARSE` must stay a reject (there is
no ND proof at all).

### Test 6 — throughput

Re-timed on this VPS (2 vCPU, 1 gate worker, 2,000 stored strings, with the full test-2 re-judge
running alongside, so absolute numbers are pessimistic — `rv/rv_t6.json`):

| path | reviewer, s / 1,000 | executor (pod) | pre-registered |
|---|---|---|---|
| old `nd_verify` pass (dropped) | 0.104 | 0.079 | — |
| new registry hit | **0.022** | 0.012 | < 0.05 ✓ |
| new marker | 0.002 | 0.002 | — |
| new fallback (nd2lean + batched Lean) | **11.6** | 3.556 | 10–120 ✓ |

Both pre-registered bands hold and the direction reproduces: on a **gated** path the judging step is
4.7× faster here (6.4× on the pod) because the `nd_verify` pass is gone. On an **ungated** path —
`coverage.py` and `grpo.py`, which use `generate_ids` and never see the gate — the new judge costs
**112× more per string** than `nd_verify` did here (45× on the pod). That is disclosed in the
fallback row but is not what "the new judge no slower" says.

### Test 7 — `lean_check --selftest`

**Reproduces:** 39/39 on Lean 4.34.1 (`rv/rv_selftest_rerun.json`), including the canonical
`~P ⊢ P > Q` via `BOTE N1` accepted at term size 1 and rejected by `nd_verify`.

I added 17 adversarial cases of my own (`rv/rv_t7_mine.json`); **17/17** behave as the decision
requires: `sorry`, `admit`, `native_decide` (`t2._native.native_decide.ax_1`), `exact?`,
`Classical.em`, `Classical.choice`, `by_cases` (`Classical.propDecidable`), `id`, `trivial` and a
`theorem t : False` are all rejected; `¬`-elimination with the wrong argument, `h.elim` used as
double-negation elimination without `Classical`, and a half-`sorry`'d conjunction are rejected;
plain `fun`/`Or.elim`/binder-shadowing proofs are accepted with the right term size.

## Splits

Disjointness by renaming class, my own canonicaliser (atoms renamed by first occurrence, premises as
a sorted multiset), all six pairs of `data/lj/{train_retain, heldout200, targets200, transfer50}`:

```
train_retain 3000 records / 3000 classes    heldout200 200/200
targets200    200 records /  200 classes    transfer50  50/50
train_retain × heldout200 0    train_retain × targets200 0    train_retain × transfer50 0
heldout200   × targets200 0    heldout200   × transfer50 0    targets200   × transfer50 0
```

**0 shared renaming classes in every pair.** My canonical form agrees with the `key` field stored on
all 3,450 records.

## Term size beside lines (policy, Dan 2026-09-27)

Re-derived with `lean_check`, which returns the term size of the elaborated value:

| population | n | term size min / median / mean / max | `n_lines` median |
|---|---|---|---|
| accepted corpus (random 1,000) | 1,000 | 2 / 6 / 6.16 / 22 | 10 |
| accepted corpus (all) | 281,817 | **2 / 6 / 6.22 / 24** | 10 (mean 10.06) |
| Lean-only class, distinct | 909 | 0 / 4 / 4.10 / 13 | — |
| test-5 accepted samples | 2,348 | 1 / 4 / 4.26 / 7 | — |
| test-5 accepted hindsight relabels | 38 | 2 / 3 / — / 6 | — |
| the 1,195 free-form records | 1,195 | 2 / 4 / — / 9 (executor's, reproduced) | n/a |

Term size is roughly 60 % of the line count on the accepted corpus, as expected: `R` and `PR` lines
count 0.

## The "no ND denotation" class (2,095 records, 24.6 %)

All 2,095 come from run `efficiency`'s `lo` arms: **1,423 from `*_seq_*` arms** (genuine `lean_seq`
samples whose literal text Lean accepts but `lean_tok.inverse` cannot decode — the example I read
re-uses the name `n64` for two different `have`s, which Lean allows and the grammar does not) and
**671 from `*_free_*` arms**, where the strict `lean_seq` grammar does not apply at all. Excluding
them from a `lean_seq` acceptance test is right under "the grammar is the allowlist", and the
executor raised it in `QUESTIONS.md` with a default; but 671 of them are not `lean_seq` samples and
arguably do not belong in the denominator of that percentage either.

## Two extra checks the pre-registration does not ask for

**The vocabulary really is the allowlist.** `lean_tok.LeanTokenizer('lean_seq')` has exactly
**107** tokens. Stripping `h1…h8` and `n1…n64` leaves, in full:

```
<pad> <eos> ( ) , : := ; => .1 .2 .elim ⟨ ⟩ ¬ → ∧ ∨
theorem t Prop P Q R S False
have exact by fun hh Or.inl Or.inr Or.elim Classical.byContradiction
```

No `sorry`, no `simp`/`decide`/`omega`/`tauto`/`exact?`/`admit`/`native_decide`, no `Classical.em`,
no `Decidable`, no library identifier. This matches `LEAN_JUDGE.md` exactly, with one harmless
over-statement: there is **no `↔` token**, so `Iff.intro`/`Iff.mp`/`Iff.mpr` are not reachable from
`lean_seq` at all (the write-up lists them as reachable "when the target is an `↔`").

**The gate's new multi-line checker agrees with a positive-verdict checker.** `check_sources` was
generalised in this run from one-line theorems to multi-line sources, with a new `bisect` line →
theorem attribution, and it still decides by *absence of an error*. On the 909 multi-line
`nd2lean` sources of the Lean-only class it agrees with `lean_check` (per-theorem `nd_check` verdict
+ axioms + allowlist) **909 / 909**; on 200 of those deliberately perturbed (`Or.inr` → `Or.inl`) it
agrees **200 / 200** (186 accepted, 14 rejected), so the agreement is not the vacuous "everything
passes". Together with the 3,506 / 3,506 agreement on test 5's literal texts, the attribution is
sound in both directions on this run's data.

## Reviewer's code

`~/review/lean-judge/rv/`: `collect_rv.py` (my collector), `rv_judge.py` (nd2lean → my
`lean_check` run, per-record rows), `rv_t1.py` (AST), `rv_t3_classify.py` (my classifier + a
re-run of the executor's for the delta), `rv_t3b.py`, `rv_t5.py`, `rv_relabel.py`, `rv_freeform.py`,
`rv_t6.py`, `rv_t7.py`, with their `.json` outputs.

## §Recount summary

| test | pre-registered | executor | reviewer | verdict |
|---|---|---|---|---|
| 1 | 0 of 8 files call `nd_verify` | 0 of 8 | **0 of 8** (my own AST walk); 26/26 unit tests pass here | reproduces |
| 2 | 0 losses, ≥ 50,000 records | 0 losses on 281,817 | **281,817 / 281,817 accepted**, 0 rejected, allowlist and axioms clean | reproduces |
| 2′ | — | 1,195 non-ND records: 567 accepted, 628 `sorryAx` | **1,195 / 1,195 accepted** with the right wrapper per arm | **executor's number wrong** (in the run's favour) |
| 3 | 100 % accepted; excess 0.01–0.07 %; > 80 % omitted premise | 6,419/6,419; 41.7 % / 25.9 % | **6,419 / 6,419 records = 909 / 909 distinct proofs**; excess **0.021 %** over 70.9 M gate-checked texts; 41.4 % / 25.9 % by records, **24 % / 49 % by distinct proof** | reproduces; denominator and the "other" class need restating; the > 80 % expectation is a miss (reported as one) |
| 3b | 0 disagreements | 0 | **0 / 6,419** | reproduces |
| 4 | 100 % | 281,817/281,817 | **281,817 / 281,817** | reproduces |
| 5 | 0 markers; new ≥ old; ≥ 1 hindsight relabel | 0/0/0; 2,348 = 2,348; 166 relabels | 0/0/0; **2,348 = 2,348**, 0 losses, 0 excess, my own Lean run agrees with the dump 3,506/3,506; **166 relabels reproduce — but the round itself recorded `relabelled_new: 0`, and it is 0 by construction** | markers and counts reproduce; **the relabelling clause is not met** |
| 6 | hit < 0.05 s/1,000; fallback 10–120 | 0.0123 / 3.556 | **0.022 / 11.6** on 2 vCPU | reproduces (both bands) |
| 7 | all pass | 39/39 | **39/39**, plus 17/17 adversarial cases of my own | reproduces |
| splits | disjoint | — | **0 shared renaming classes in all six pairs** | clean |

---

# §Compare

Read after §Recount was committed (`0b14a38`): `run_lean_judge.md`, `numbers.md` § lean-judge
(L0–L8), `log.md` § lean-judge, `STATUS.md`, `QUESTIONS.md`, `figures/lj_*.png`.

## Process

| item | check | verdict |
|---|---|---|
| expectations written before the run | `preregistration/lean-judge.md` = `579ee16`, committed **17:20:43Z**; first pod `lj1` in `~/pods.log` at **17:38:36Z**; `podbudget lean-judge --set 8 4` at 17:15:07Z | **gate 0 holds** (18 min) |
| budget | 0.6631 pod-h / **$0.32** of 8 h / $4 (`~/podhours.log`), A40 with its real billed rate ($0.49/h) recorded | within budget, rate recorded |
| misses reported as misses | the `> 80 %` expectation of test 3 is labelled **wrong** in `LEAN_JUDGE.md`, `run_lean_judge.md` and `numbers.md` | yes |
| deviations recorded | `require_all_pr=False`; test 5's retain slice (`data/p2/heldout.jsonl` 200–3,200, because `data/dsc/train_a3.jsonl` is in neither git nor the bucket); the `ckpts/lj` mkdir failure — all three in `log.md` | yes |
| checker named on both sides of 2026-09-27 | `numbers.md` § lean-judge opens with an explicit **Checker labels** paragraph; `LEAN_JUDGE.md`, `run_lean_judge.md` and `STATUS.md` each say stored = Lean ∧ `nd_verify`, new = Lean alone | yes, on every write-up |
| model labels | every table names `lean_seq` / from-scratch / ~19 M / Stage-1 pool, and test 5 names `ckpts/dsc/stage1_a3_s0.pt` | yes |
| `nd_verify` used as a judge anywhere in the run | `lj_regress.py judge --compare_nd`, `lj_t5_compare.py` and `throughput` import it, each to reconstruct the **old** count or to time the dropped pass | no — correct use |
| questions for Dan | three in `QUESTIONS.md`, dated, each with the default followed | yes |
| one test-file run | none in this run | fine |

## Claims

| claim (executor) | my independent value | verdict |
|---|---|---|
| test 1: **0 of 8** judging files import or call `nd_verify`; 26/26 checks | 0 of 8 by my own AST walk; 26/26 re-run here | **reproduces** |
| test 2: **281,817 / 281,817 = 100.000 %**, **0 losses** | 281,817 / 281,817, 0 rejected, and additionally 0 allowlist / 0 axiom rejections | **reproduces**, under a stronger check |
| test 4: `n_lines` identical on **281,817 / 281,817** | 281,817 / 281,817 | **reproduces** |
| L0: 283,012 collected, of which 1,195 not ND ⇒ 281,817 | 281,817 distinct ND; the 1,195 are exactly the non-ND records | **reproduces** |
| L0: "1,533,741 records in **1,000** `found*.jsonl` … **34** coverage files … **398** `*.disagree.jsonl` (29,070 records)" | 1,863,151 records in **1,284** found + **102** coverage files; **492** disagree files, 39,587 records | **differs** — file and record counts depend on how many run worktrees are checked out; the *distinct* counts (281,817 / 8,514) are identical, which is what the tests use |
| L2: "a spot check … accepts 567 and fails 628 on elaboration … **the statement form differs**, not that the proofs are bad. Out of scope" | the failures are entirely `*_seq_*` arms and entirely a **wrapper** problem: `:= <tactic block>` does not parse. With `:= by <block>` for `_seq_` and `:= <term>` for `_free_`, **1,195 / 1,195 accepted** | **direction right, number wrong**: the reading ("not that the proofs are bad") is correct, but 567/628 is an artefact and the real answer is 1,195/1,195. `LEAN_JUDGE.md`'s test-2 row states 567/628 without that caveat |
| test 3: **6,419 / 6,419 = 100 %** | 6,419 / 6,419 records — and **909 / 909 distinct (prompt, ND proof)**, 6,419 / 6,419 distinct literal texts, allowlist and axioms clean | **reproduces**; the class is 909 distinct proofs, not 6,419, and no write-up says so |
| test 3 classes: 41.7 / 25.9 / 24.6 / 6.2 / 1.6 % | 41.4 / 25.9 / 24.6 / 6.5 / 1.6 % of records; **24 / 49 / – / 27 / 0.4 %** of the 909 distinct proofs | **differs by 0.3 pp** (a premise-parsing bug in `classify_leanonly` on premise-free theorems: 26 of 874 records); **the record-level ordering inverts at the distinct-proof level** |
| "other 6.2 %" (unexplained) | 204 of its 232 distinct proofs are `IMPI` 116 / `DN` 78 / `NEGI` 10 rule-check failures, all one cause: **`¬A` and `A > F` are the same proposition in Lean** — with the `NEGE` row that is 7.8 % of the class, a fourth nameable class | **not supported as a residual**; should be named |
| "the 0.01–0.07 % rate" | **0.021 %** = 14,995 `nd_rej & lean_ok` of 70,882,951 distinct checked texts, over 225 gate logs | **reproduces** (inside the band) |
| "all 29,070 stored disagreements are `nd_rej & lean_ok`; the old gate's `LEANREJ` path never fired in five runs" | **0** `nd_ok & lean_rej` in 39,587 disagreement records **and** in the 70.9 M distinct texts the gate logs summarise | **reproduces, and is stronger than stated** — the 70.9 M denominator is the right one |
| test 3b: **0** disagreements on 6,419 | 0 / 6,419 (literal / `nd2lean(nd)` / `nd2lean(norm(nd))`) | **reproduces** |
| test 5: **0 / 0 / 0** markers in `found_1` (228), `found_transfer_1` (56), `mix_1` (3,912) | 0, and 0 in `round_1.json` too | **reproduces** |
| test 5: 4,250 samples / 476 parse failures / 3,506 distinct checked; 2,348 Lean-accepted; 0 registry conflicts | 4,250 / 476 / 3,506; my own Lean run on the 3,506 stored texts agrees with the dump **3,506 / 3,506**, 2,348 accepted; 0 conflicts | **reproduces** |
| test 5: old 2,348 = new 2,348, **excess 0, losses 0** | 2,348 = 2,348, excess 0, losses 0 (records); 484 = 484 (distinct) | **reproduces** |
| test 5: "**166** hindsight relabels accepted, 80 rejected, one shown in Lean" | 246 candidate pairs → 78 distinct; **166 pairs / 38 distinct accepted**, 80 / 39 rejected — exactly the executor's numbers | **the number reproduces; the claim it supports does not.** The round's own `round_1.json` says `relabelled_new: 0, relabelled_total: 0`. 166 is measured by `pod/lj/t5_relabel.py` on the **gate dump**, which stores clean ND strings; the loop is handed `'LEANREJ ' + nd` for every Lean-rejected sample and `judge_many` rejects on the marker first. Measured: 125 accepted on the clean strings, **0** on the strings the loop sees; all 125 come from Lean-rejected samples. Three clear every other loop filter and were still not recorded. **Pitfall 2 is not closed end to end** |
| test 6: hit **0.0123**, marker 0.0021, fallback **3.556** s/1,000; "**6.4× faster**" | 0.022 / 0.002 / 11.6 s/1,000 on 2 vCPU; 4.7× faster | **reproduces** (pod vs VPS); both pre-registered bands hold |
| "the new judge is faster" (STATUS, run write-up) | true on a **gated** path. `coverage.py` and `grpo.py` use `generate_ids` and never see the gate, so every string takes the fallback: **45× slower per string than `nd_verify` on the pod, 112× here** | **needs rewording** — the fallback row is in `numbers.md`, but no write-up says which files pay it |
| test 7: **39/39**, Lean 4.34.1 and 4.34.0 | 39/39 on 4.34.1, plus 17/17 adversarial cases of my own (`admit`, `native_decide`, `exact?`, `Classical.choice`, `by_cases`, `id`, `trivial`, a `theorem t : False`, a mis-typed `Not.elim`, `h.elim` as DN without `Classical`) | **reproduces**, and the allowlist holds against cases the self-test does not cover |
| L8: vocabulary **107** tokens, no `sorry`/`simp`/`Classical.em` | 107; full non-index token list re-derived and matches | **reproduces**. One over-statement: there is **no `↔` token**, so the `Iff.intro` / `Iff.mp` / `Iff.mpr` cases `LEAN_JUDGE.md` lists as reachable from `lean_seq` cannot arise |
| "model labels: stored samples from `cap-horizon`, `noise-floor`, `ds-rendering`, `efficiency`, `lean-format`" | all contribute: `kh` 262,653, `dsc` 35,441, `lo` 15,456, `dsg` 14,897, `lf` 11,338, `nf` 8,631 distinct (overlapping) | **reproduces**; `ds-composition` (`dsc`) and `ds-generator` (`dsg`) also contribute and are not named |
| splits disjoint | 0 shared renaming classes in all six pairs of `data/lj/*` | **reproduces** |
| gitignore: "the 165 MB corpus is reproducible with `python3 lj_regress.py collect`" | re-running `collect` today gives **281,817** records, not the committed file's 283,012 (the non-ND exclusion was added after the file was written) | **differs** — the file is in the bucket, so nothing is lost, but the committed script does not regenerate the committed artefact |

## Wording against n

Every number here is a property of stored files or of code, not of a model comparison, so the n = 2
rule does not bite. Nothing in the write-ups says "bistable", "wall" or "never" about a measurement.
"The old gate's `LEANREJ` path never fired in five runs" is the one absolute, and it is supported
(0 of 70.9 M). "0 losses" is exact, not an estimate, and I reproduce it exactly.

The one place the wording outruns the evidence is **"all seven acceptance tests pass"**
(`STATUS.md`, `run_lean_judge.md`): test 5 has three clauses and its third — the hindsight-relabel
clause — is met by a side script and not by the round.

# §Verdict

## What stands

- **The judge is sound and the change is safe.** Every one of the 281,817 proofs the old
  Lean ∧ `nd_verify` gate counted is still counted, with no constant off the allowlist and no axiom
  outside {`propext`, `Classical.choice`, `Quot.sound`} — I re-derived this over the whole population,
  not a sample, with a positive-per-theorem checker rather than the gate's absence-of-error test.
  Line counts are identical on all of them. Tests 1, 2, 3, 3b, 4, 6 and 7 reproduce.
- **Nothing is lost anywhere.** Beyond the 281,817, the 1,195 records the corpus could not translate
  are also all accepted (by the free-form judge, correctly wrapped): **1,195 / 1,195**. The run's
  "0 losses" is if anything understated.
- **Pitfall 1 is closed.** 0 markers in the round's training files, checked over whole lines, and the
  gate's contract is unit-tested.
- **Pitfall 3 is closed.** 6,419 / 6,419 agreement across the literal text and both translations.
- **Pitfall 4 is closed on the gated path.** The registry hit is 0.022 s / 1,000 here.
- **The grammar really is the allowlist.** 107 tokens, no automation, no library identifier; and my
  own adversarial cases (17) all fall the right way.
- Gate 0, the budget, the deviations, the checker labels and the model labels are all in order, and
  the one missed expectation is reported as missed.

## What must be reworded

1. **"All seven acceptance tests pass" → tests 1–4, 6, 7 pass; test 5 passes on markers and counts
   and fails on hindsight relabelling.** The round recorded `relabelled_new: 0`.
2. **The 628 `sorryAx` rejects** (`LEAN_JUDGE.md` test-2 row, `numbers.md` L2, `t2_freeform.json`)
   are a wrapper artefact. Replace 567 / 628 with **1,195 / 1,195 accepted**, `_free_` arms as terms
   and `_seq_` arms as `by` blocks.
3. **"6,419 / 6,419"** is a count of stored samples; say **909 distinct proofs** beside it, and give
   the class fractions at the distinct level too (`Not.elim` 49 %, other 27 %, omitted premise 24 %) —
   the record-level ordering is a property of re-sampling, not of the class.
4. **"other 6.2 %"** should be named: `¬A` versus `A > F`, the same definitional identity as the
   `NEGE` row; together 7.8 %.
5. **"the judging step is 6.4× faster"** should say *on a gated path*; on `coverage.py` and `grpo.py`,
   which never see the gate, judging costs 45× more per string than `nd_verify` did.
6. **The classification's 41.7 %** is 41.4 %; fix the premise parse in `classify_leanonly`
   (premise-free prompts) — `lj_t5_compare.py` shares the expression.
7. Minor: `LEAN_JUDGE.md`'s `Iff.*` sentence (no `↔` in the vocabulary); L0's file counts
   (1,000 / 34 / 398) are environment-dependent; the gitignore note that `collect` reproduces
   `corpus_accepted.jsonl` (it now produces 281,817 rows, not 283,012); the model-label list should
   add `ds-composition` and `ds-generator`; `ladder_ei.py` and `eval_set.py` import `verify_text`
   from `lean_judge` and never call it.

## What is not supported

**Pitfall 2 — "hindsight relabelling is judged, not silently accepted" — is closed in the unit test
and in a side script, and open in the loop.** `lean_gate.gate` now marks *every* Lean-rejected
sample with `LEANREJ `, where the old gate marked only the `nd_ok & lean_rej` cell (which is empty in
five runs of data). `expert_iter.relabel_batch` passes that marked string to `judge_many`, whose
first rule is "marker ⇒ reject". Hindsight relabelling operates by definition on samples that failed
their prompted theorem, so it now rescues nothing: **125 → 0** on this round's own candidates. This
is a regression introduced by the run, not a pre-existing one.

The fix is small and local: strip a leading `LEANREJ ` in `relabel_candidate`/`relabel_batch` before
judging — the marker records Lean's verdict *for the prompted theorem*, which says nothing about the
rewritten one. `LEANPARSE` must keep rejecting (there is no ND proof at all). Then re-run test 5's
round; on this round's data I expect ~3 relabels to be recorded (n_lines ≥ 7 and not in `eval_keys`).

## The next measurement

1. **Re-run test 5's round after the `LEANREJ` fix**, and report `relabelled_new` from
   `round_1.json` rather than from a script over the dump. That is the only acceptance test whose
   evidence does not come from the artefact the loop writes; it is also how the gap was hidden.
2. **One `lean_check` pass over the ladder and coverage pools** to attach `term_size` and a
   Lean-minimal length, so `minlen.py`'s ND-derived `L_true` stops being an unlabelled upper bound.
   The executor proposes this in `QUESTIONS.md`; my re-derivation gives the scale of the gap on the
   counted population — term size **median 6** against `n_lines` **median 10** (mean 6.22 vs 10.06),
   so line counts overstate proof size by about 40 % under Lean.
3. **Decide the `no-denotation` class** (`QUESTIONS.md` question 1). 2,095 stored samples, of which
   **1,423 are `lean_seq`** samples whose literal text Lean accepts and the grammar rejects (the one
   I read re-uses a `have` name, which Lean allows) and **671 are free-form arms** where the
   `lean_seq` grammar does not apply. Whatever is decided, those 671 should leave the denominator of
   the "24.6 %".

**Recommendation: accept, with the six rewordings above, and treat the `LEANREJ`-blocks-relabelling
regression as a fix to land before the next expert-iteration run on this branch.** No hard constraint
is violated; nothing is quarantined.
