---
written_on: 2026-09-27
written_by: agent:claude
---

# lean-judge: Lean alone decides on the fork's `dan` branch

Status: completed 2026-09-27 17:15–18:35 UTC (executor) and independently reviewed
18:20–20:37 UTC. Review verdict: **accept, with six rewordings and one regression to fix**. No
hard-constraint violation, nothing quarantined. Numbers and wording below are the reviewer's,
including the reviewer's own words where they differ from the executor's. Run id `lean-judge`. This
run happened directly on **`dan`, the fork's own integration branch** (`dan-pandori/nd-takehome`,
not `nd-rl`'s `dan`) — the executor's brief describes it as "the new integration branch," so unlike
every other run this week there is no `dan_<slug>` branch to merge; everything below is already on
that branch.

Review:
[`review_lean-judge.md`](https://github.com/dan-pandori/nd-takehome/blob/be03514/review_lean-judge.md)
(commits [`0b14a38`](https://github.com/dan-pandori/nd-takehome/commit/0b14a38) phase-1 blind
recount, [`be03514`](https://github.com/dan-pandori/nd-takehome/commit/be03514) phase-2 compare and
verdict; reviewer code under `~/review/lean-judge/rv/` at that commit). Executor write-up:
[`run_lean_judge.md`](https://github.com/dan-pandori/nd-takehome/blob/be03514/run_lean_judge.md),
standing reference
[`LEAN_JUDGE.md`](https://github.com/dan-pandori/nd-takehome/blob/be03514/LEAN_JUDGE.md) (quote this,
with the corrections below applied — it was not edited after review), numbers `numbers.md` §
lean-judge (L0–L8), log `log.md`, open questions `QUESTIONS.md`, all on branch `dan` of
`dan-pandori/nd-takehome`. Pre-registration:
[`preregistration/lean-judge.md`](https://github.com/dan-pandori/nd-takehome/blob/be03514/preregistration/lean-judge.md)
(`579ee16`, committed 17:20:43Z, 18 min before the first pod). Brief:
[`docs/proposals/followups/BRIEF_lean-judge.md`](../../docs/proposals/followups/BRIEF_lean-judge.md),
implementing the 2026-09-27 update of
[the Lean-default decision note](../../docs/project_strategy/2026-09-20-lean-default.md) (proposal
9's follow-on: "Lean alone is the reward and the checker for every new run," `Not.elim` admitted).

Question: with `nd_verify` removed from every judging path (in-loop gate, expert-iteration
relabelling, coverage, GRPO, evaluation) in favour of Lean 4 core alone, are there zero regressions
against everything the old `Lean ∧ nd_verify` gate ever counted, is the newly-counted Lean-only class
sound and small, and are the four pitfalls found while tracing the decision (marker leakage into
training data, hindsight relabelling against a stale verdict, `coverage.py`'s pre-judging
normalisation, per-string Lean startup cost) actually closed end to end rather than only in a unit
test?

## Configuration

- **What changed.** New `lean_judge.py` (marker ⇒ reject; a gate-registered verdict ⇒ that verdict;
  otherwise batched `nd2lean.translate(..., require_all_pr=False)` + Lean, `judge_many` for
  batching). `lean_gate.py` no longer imports `nd_verify`, registers per-sample Lean verdicts, and
  returns **clean** ND strings for accepted samples (only rejects carry `LEANREJ `/`LEANPARSE `, so a
  verdict marker never becomes training text). Six loop files re-pointed at the judge:
  `expert_iter.py`, `ladder_ei.py`, `coverage.py`, `eval_set.py`, `eval_targets.py`, `grpo.py`.
  `lean_check.py` ported from `origin/dan_lean_only` (proposal 9) with `Not.elim`, `Not.intro`,
  `And.elim`, `Iff.elim` added to the allowlist; `nd_verify` stays in the repository, unmodified,
  behind an opt-in `--compare_nd` flag, only so pre-2026-09-27 numbers reproduce.
  `nd2lean.translate(..., require_all_pr=False)` is new and used only from the judge — Lean does not
  require every premise re-stated, and this is the single largest class of proofs Lean accepts and
  `nd_verify` rejects.
- **Model labels.** No model was trained except for test 5. Tests 1–4 and 6 are properties of
  **stored samples** from runs `cap-horizon`, `noise-floor`, `ds-rendering`, `efficiency`,
  `lean-format`, `ds-composition` and `ds-generator` (the last two contribute to the corpus but are
  not named in `LEAN_JUDGE.md` — a reviewer correction) — all **`lean_seq`**, **from-scratch**,
  ~19 M parameters, trained on this project's Stage-1 Lean-format pools, whose stored verdicts were
  produced under **Lean ∧ `nd_verify`** (pre-2026-09-27). Test 5 is a fresh one-round
  expert-iteration run on `ckpts/dsc/stage1_a3_s0.pt` (run `ds-composition`, arm a3, seed 0:
  `lean_seq`, from-scratch, ~19 M parameters, Stage-1 pool `train_a3`), k = 16, T = 0.8, seed 0, 200
  depth-3 targets, one A40 pod `lj1`.
- **Reviewer's checker is strictly stronger than the run's own.** The run's acceptance tests (2, 3,
  3b, 5) used the gate's absence-of-error test; the reviewer used `lean_check.check` throughout — a
  **positive per-theorem verdict** plus an axiom check (⊆ `propext`, `Classical.choice`,
  `Quot.sound`) and a constant-allowlist check on the elaborated term — and re-ran it over **entire
  populations**, not samples, everywhere it matters (all 281,817 test-2 proofs, all 6,419 Lean-only
  records, all 3,506 test-5 texts): **11,946 counted proofs re-checked in Lean from their stored
  literal text**, well above the reviewer brief's own 100-per-arm floor.
- **Hard constraints, all pass** (reviewer's own re-derivation): `nd_verify` blob-identical to
  `origin/main`'s (`9437bb7…`); `artifacts/TEST_RUN_DONE` unchanged (0 commits touch it, md5 stable);
  no evaluation file read outside the scorer/auditor; one caveat — `nd2lean.py` imports
  `nd_verify.verify_text` at module level and `lean_judge` imports `nd2lean`, so every judging path
  transitively imports `nd_verify` (nothing calls it; a documentation point, not a judging one).
- **Budget and cost.** `podbudget lean-judge --set 8 4` registered before the first pod; one
  **NVIDIA A40, $0.49/h**, pod `lj1`: **0.6631 pod-hours, $0.32** of $4 / 8 h. RunPod balance $150.91
  after (floor $130). Bucket: `hf://buckets/dan-pandori/nd-rl/lean-judge/{artifacts/lj,data/lj}`
  (179 MB, including the gitignored 165 MB `corpus_accepted.jsonl`). No checkpoint uploaded (test 5's
  one-round fine-tune is a smoke test, not a result).

## Observations

**The judge is sound and the change is safe — nothing is lost anywhere.** Every one of the
**281,817** distinct `(prompt, ND proof)` pairs the old `Lean ∧ nd_verify` gate ever counted (from
1.86 M records across every Lean-gated run this project has produced) is accepted by the new judge,
re-derived over the **whole population** with the reviewer's stronger checker: **281,817 / 281,817,
0 losses**, no constant off the allowlist, no axiom outside the three permitted, line counts
identical to `nd_verify`'s on all of them. The **1,195** collected records the run flagged as "not
ND proofs, out of scope" (literal Lean text stored in run `efficiency`'s `proof` field) are also
**all** accepted once wrapped for their own arm's surface (`:= <term>` for free-form arms, `:= by
<block>` for `_seq_` arms): **1,195 / 1,195**, not the run's reported 567/628 — that split was
entirely a wrapper artefact (the `_seq_` arms store a tactic block, which is a parse error after
`:=` with the wrong wrapper). So the run's "0 losses" understates: **nothing counted by any prior
Lean-gated run is lost**, and the excluded records are recoverable too.

**The Lean-only class is counted, and it is smaller and differently shaped than the write-up says.**
All 6,419 stored records with an ND denotation are accepted (100 %), but they collapse to **909
distinct (prompt, proof) pairs** — `LEAN_JUDGE.md` reports only the record count. At the
distinct-proof level the class composition **inverts**: `Not.elim` is the largest class at **49 %**
(not 25.9 % of records), "other" is **27 %** and is not a residual — 204 of its 232 distinct proofs
are one nameable cause, **`¬A` and `A > F` being the same proposition in Lean and different formulas
in ND** (`IMPI`/`DN`/`NEGI` rule-check failures under `nd_verify`), which together with the `NEGE`
row is **7.8 %** of the class — and omitted premise re-statement drops to **24 %** (not 41.7 %; a
0.3 pp record-level discrepancy traces to a premise-parsing bug in `classify_leanonly` on
premise-free prompts). The pre-registered excess rate holds: **0.021 %** of 70.9 M distinct
gate-checked texts across every Lean-gated log on the box, inside the 0.01–0.07 % band; the
pre-registered "> 80 % omitted premise re-statement" expectation is a genuine miss, reported as one.
Pitfall 3 (normalising before judging) is fully closed: literal text, `nd2lean(nd)` and
`nd2lean(norm(nd))` agree on all 6,419 records, 0 disagreements.

**Pitfall 2 (hindsight relabelling) is not closed — it is a regression this run introduces.** The
executor's headline "166 hindsight relabels accepted" reproduces exactly, but it is measured by a
side script over the gate's audit dump, which stores the *clean* ND string of every checked sample.
The expert-iteration loop itself never sees those clean strings: the new `lean_gate.gate` prefixes
`LEANREJ ` onto **every** Lean-rejected sample (the old gate marked only the empty `nd_ok & lean_rej`
cell), and `expert_iter.relabel_batch` passes that marked string straight to `judge_many`, whose
first rule is "a marker ⇒ reject" — before the rewritten theorem is ever looked at. Hindsight
relabelling is by definition about samples that *failed* their prompted theorem, so on this branch it
now rescues nothing: measured on the round's own 187 relabel candidates, `relabel_batch` accepts
**125** of them on the clean strings and **0** on the strings the loop actually receives; three of the
125 clear every other loop filter and were still not recorded. The round's own `round_1.json` already
says `"relabelled_new": 0, "relabelled_total": 0` — the gap was visible in the artefact the loop
writes, and the write-up's "all seven acceptance tests pass" is the one place the wording outruns the
evidence (test 5 has three clauses; the third fails). The fix is one line: strip a leading `LEANREJ `
before relabelling (`LEANPARSE` must keep rejecting — there is no ND proof at all in that case).

**Everything else the run measured reproduces.** Test 1 (0 of 8 judging files call `nd_verify`, by
AST walk): reproduces, 26/26 unit tests re-run. Test 4 (line counts): 281,817/281,817 agree with
`nd_verify`'s. Test 5's markers-and-counts clauses: 0 `LEAN*` markers in `found_1`/`found_transfer_1`/
`mix_1`, and the old gate vs the new judge agree exactly on the round's 3,506 stored samples
(2,348 = 2,348, 0 losses, 0 excess — expected, since 3,506 samples predict 0.4–2.5 Lean-only proofs
at the measured rate). Test 6 (throughput): both pre-registered bands hold in both directions
(registry hit 0.012–0.022 s/1,000 < 0.05; fallback 3.6–11.6 s/1,000, inside 10–120) and the judging
step really is faster on a **gated** path (4.7–6.4×, since the dropped `nd_verify` pass is the whole
saving) — but on an **ungated** path, `coverage.py` and `grpo.py`, which call `generate_ids` and
never see the gate, every string pays the fallback, and the new judge costs **45–112× more per
string** than `nd_verify` did. "The new judge is faster" is true only on the gated path, and no
write-up says which files pay the fallback. Test 7 (`lean_check --selftest`): 39/39 on both Lean
versions used, plus 17/17 adversarial cases of the reviewer's own (`admit`, `native_decide`,
`exact?`, `Classical.choice`, `by_cases`, a mis-typed `Not.elim`, `h.elim` as double-negation
elimination without `Classical`, …), all falling the right way. The grammar-is-the-allowlist claim
(107-token vocabulary, no `sorry`/tactic automation/library lemma) reproduces exactly, with one
harmless over-statement: there is no `↔` token, so `Iff.intro`/`Iff.mp`/`Iff.mpr` are not reachable
from `lean_seq` at all. Term size beside lines (policy): on the accepted corpus, median term size **6**
against median `n_lines` **10** — under Lean, line counts overstate proof size by about 40 %, and
`minlen.py`'s ND-derived `L_true` labels are therefore **upper bounds**, not relabelled in this run.
Splits: 0 shared renaming classes in all six pairs of `data/lj/{train_retain, heldout200, targets200,
transfer50}`.

**Questions for Dan, each with a default followed** (`QUESTIONS.md`, 2026-09-27, unanswered as of
this write-up): (1) the 2,095 "no ND denotation" Lean-only records (1,423 genuine `lean_seq` samples
the grammar can't decode, e.g. a reused `have` name; 671 free-form arms the grammar doesn't apply to)
— default: excluded, the grammar stays the allowlist; (2) `require_all_pr=False` — default: keep it
judge-only, don't change what generators write; (3) relabelling `minlen.py`'s pools to Lean-derived
`L_true` — default: not done here, a separate CPU-only job.

## Limitations and next step

- **Land the one-line `LEANREJ`-strip fix in `relabel_candidate`/`relabel_batch` before the next
  expert-iteration run on this branch**, then re-run test 5's round and report `relabelled_new` from
  `round_1.json` itself rather than from a script over the dump — the only acceptance test whose
  evidence didn't come from the artefact the loop actually writes, which is how the gap stayed
  hidden. On this round's data the reviewer expects ≈ 3 relabels recorded (`n_lines ≥ 7` and not in
  `eval_keys`).
- One `lean_check` pass over the ladder and coverage pools to attach `term_size` and a Lean-minimal
  length, so `minlen.py`'s `L_true` stops being an unlabelled upper bound (raised by the executor in
  `QUESTIONS.md`; scale of the gap now measured: term size ≈ 60 % of line count on the counted
  corpus).
- Decide the no-denotation class (`QUESTIONS.md` question 1): whatever is decided, the 671 free-form
  records in it (the `lean_seq` grammar never applied to them) should leave the denominator of any
  "Lean-only class" percentage either way.
- `LEAN_JUDGE.md` needs the six rewordings above applied (it was not edited after review): the
  1,195/628 figure, the 909-distinct-proof denominator and its inverted class ordering, "other 6.2 %"
  named, "6.4× faster" scoped to gated paths, the 41.7 % → 41.4 % parsing-bug fix, and the missing
  `↔`/model-label corrections.
