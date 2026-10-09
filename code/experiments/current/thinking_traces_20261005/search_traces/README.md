---
written_on: 2026-10-08
written_by: agent:claude
---

# Honest-search thinking traces (`searchtrace.py`): design, pilot and audit

**Status (2026-10-08):** generator, validator extension, tests, a 5,000-proof K12 pilot at every level and the audit
are done. Everything ran on CPU, with at most 12 worker processes under `nice`. The full K12 set was not generated and
nothing was trained. Branch `leon/thinking`, worktree `/scratch/cowifornia/nd-rl/think/nd-rl`, base commit `864ee17`;
the changes are uncommitted. Leon approved the direction on 2026-10-08; the coordinator's later additions (the ladder,
labels, the control arm and the gates) are folded in below.

## Hard rule

Every think line is written by code, by [`code/thinking/searchtrace.py`](../../../../thinking/searchtrace.py). No
language model wrote, paraphrased, compressed or rewrote any trace content. The examples below are generator output,
pasted by a script (`traces/search/audit/examples_md.py`). The search never reads the reference proof:
- `prove(prompt, ...)` and `Search(env, ...)` take the theorem or the environment state only;
- `trace_for` reads a reference only after the proof is built, for statistics;
- a test checks that traces are identical with the reference given, removed or replaced by another proof.

## Verdict: correct and honest, but the search is shallow on K12

The corpus is correct. Every trace passes `trace_dsl.validate(strict=True)` and the environment replays it. The
independent checker reports 0 disagreements, the countermodel lines included. Lean accepts 600 of 600 sampled proofs.

The search is shallow on K12, so it does not deliver the main thing it was built for. **Genuine (non-ex-falso) dead
ends occur in 12.9% of e256 decision blocks**, about the same as v1 (13.4%) and v2 (8.8%). In about 87% of blocks
the strategy's first or closing candidate is already viable. Gate G2 of PLAN_CRITIQUE §3.1 needs ≥ 25%, so on K12
this generator **fails G2**. That is the key go/no-go number.

What the search does change:
- the ex-falso ritual is gone: 1.6% of dead ends, against 71% in v1;
- a verdict can no longer be predicted from the line's form: 0.0 points over the majority baseline, against 16.1 in v1;
- every dead end carries a checkable countermodel;
- proofs are the search's own: identical to the reference in 35%, mean length 0.85 × the reference's.

On the 13–20 set, analysed for statistics only, dead ends are more frequent (17.4%) but the generator fails on 12% of
theorems (§ open issues). The opt-in complete mode (§ Complete mode) brings that to 99.93%.

## What was built

| piece | where | what |
|---|---|---|
| generator | `code/thinking/searchtrace.py` (new) | per-step backward search, 4 strategies, effort budgets, `min_level` labels, batch CLI, stats, diversity, `--emit-plain` |
| DSL extension | `trace_dsl.py` (`kleene`, `parse_countermodel`, `refutes`, `fmt_countermodel`, and 3 hooks in `check_step`) | a `<no>` may be followed by a countermodel, which is checked; lines without one are checked as before |
| graded hook | `code/thinking/data.py` (`build_graded(..., any_proof=False)`, `--graded-any-proof`) | lets a level's file prove a theorem differently, with `0:PATH:PROB`; the default contract is unchanged |
| tests | `code/tests/thinking/test_searchtrace.py` (new, 26 tests) | validity, determinism, reference independence, certified dead ends, budgets, labels, one proof per theorem, control arm, graded hook |
| pilot and audit | `/scratch/cowifornia/nd-rl/think/traces/search/` (`run_pilot.sh`, `pilot/`, `audit/`, `pickles/`, `sealed_analysis_13_20/`) | everything below; `run_pilot.sh` reruns it (about 5 min) |

## Design and judgment calls

**One step = one search.** The setting is Markov (PROPOSAL §1): at each decision state the search runs afresh on the
state alone.
- It is a depth-first backward search over the DSL's reductions `ctx ⊢ G := SKEL`.
- Candidates are the following, with a subgoal that is already a hypothesis closed by its name:
  - the assumption (G in scope);
  - introduction by G's connective: pair, `Or.inl`, `Or.inr`, a box `fun ( x : A ) => ...`;
  - elimination spines: up to 3 applications or projections of a hypothesis, ending in G, in False then `.elim`, or in
    a disjunction to split on;
  - case splits (`Or.elim`);
  - the classical detour `Classical.byContradiction ⊢ ¬¬G`;
  - ex falso.
- Each candidate's holes are checked by truth tables, which are complete for this fragment:
  - a dead candidate is written with `<no>` and its countermodel;
  - the first viable candidate is followed into its first open subgoal, down to a reduction that is one environment
    action. That action is committed, so the action is always the search's own.
- A subgoal whose candidates all fail sends the search back to its parent; the record shows this by restating the
  parent goal.
- Premise restatements and an `exact` of a last line that already is the goal are forced: their think is empty.
- Judgment calls:
  - **No candidate (action) lines.** The search works on reductions, and the last goal line determines the action. A
    candidate line would only repeat it, which is v1's finding 1 ritual. TRACE_AUDIT's candidate string-match metric
    is therefore not defined for these traces. I report a reduction-level analogue instead.
  - **Truth tables are the value function.** The search checks validity by truth table. It writes dead ends it has
    certified. It does not write viable alternatives that it skipped without a reason (finding 3).
  - **Progress guards** are rules for generating candidates, not claims. They keep a Markov search from looping:
    - no subgoal that is already on the path under binders that add nothing;
    - no split on a disjunction with a side in scope;
    - a split is *relevant* only if the disjunction lies in some minimal support of the goal (computed exactly by
      subsets); irrelevant splits are a last resort;
    - in an inconsistent context, a box whose binder is already a hypothesis is a last resort, and candidates are
      ordered by one-step closability. Truth tables refute nothing there: 41% of decision states are inconsistent.
  - **Closers first.** A candidate that closes the goal outright, with no open subgoal, is examined first at every
    level.
    - It is part of the search order, not a hidden check: closed candidates are never dead.
    - At every level the written action is then the same, so every level writes **one canonical proof** per theorem.
      That is what the T0-S control needs (coordinator, 2026-10-08).
    - `--closers-high-only` restores the variant in which only levels ≥ 128 look for closers first. There e256 wrote a
      different proof for 6.7% of theorems; `--graded-any-proof` handles that variant.
  - **Premises are restated in order**, as in every K12 proof, from the theorem alone.
  - **Reference-free length cap.** `--max-steps 40` env steps, then the next strategy is tried; `--ref-slack K` is an
    opt-in cap of reference + K. The brief allowed a reference + 4 cap, but a cap that depends on the reference would
    make the trace depend on it. The test of reference independence would then fail, and new theorems have no
    reference. With the default, 3.1% of pilot proofs are longer than reference + 4.

**Strategies** (≥ 3 required; 4 built). One per proof, drawn from sha256 of the id. The strategy is recorded in
`gen.strategy` and never appears in the text. If a strategy fails, the next one in a hash-determined order is tried.

| strategy | order |
|---|---|
| `dfs` | introductions before eliminations, hypotheses oldest first, then splits, the classical detour, ex falso |
| `hyp` | eliminations (hypotheses newest first) before introductions |
| `best` | best-first: subgoals closable in one step, then small, first; detours last |
| `rand` | a per-theorem random order of introductions and eliminations, and of candidates within each kind (stable across steps) |

Dead-end form: `dfs` and `hyp` write the explicit two-line form (`⊢ G := R` / `⊢ H <no> ρ`); `best` writes the
short form (`⊢ G := R <no> ρ`); `rand` picks one per theorem. Below 128 tokens every strategy uses the short form.

**Effort is the search budget.** A level b sets the token cap (240 at b = 256, as in `progtrace`; b otherwise). It
also sets how many candidates per goal may be recorded, whether dead ends carry countermodels, the dead-end form and
the lookahead depth into a committed box (`budget_for`):

| level | cap | candidates recorded per goal | countermodels | lookahead into the box |
|---|---|---|---|---|
| e0 | 0 | 0 | – | – |
| e16 | 16 | 2 | no | 0 |
| e64 | 64 | 4 | yes | 0 |
| e256 | 240 | 8 | yes | 1 level |
| e512 / e1024 / … | b | unbounded | yes | 2 levels |
| `inf` (analysis only) | 4,096 safety cap | unbounded | yes | 2 levels |

The record is written while the search runs. When a unit (one candidate's lines) does not fit, recording stops and the
search finishes unrecorded; the natural length is still counted. The levels are configurable (`--levels`, default
`0,64,256`, coordinator 2026-10-08). The cap follows the level, and the default behaviour at 256 is unchanged. Effort
tokens above `<e256>` are not in the tokenizer: the generator accepts any level, but `data.py --graded` does not.

**The DSL extension: countermodel evidence** (brief item 4: built, not deferred).
- Syntax: a goal line's `<no>` may be followed by literals, `⊢ G := SKEL <no> P ( ¬ Q )`. Atoms come in P Q R S order,
  each at most once.
- What the validator checks: the partial assignment, under Kleene's strong three-valued evaluation, makes every
  hypothesis of the scope (and the line's binders) true and the goal false. On a reduction it must do this for one of
  its holes. Any completion is then a countermodel.
- What the generator writes: the cheapest such assignment by tokens.
- Strictness and compatibility: the validator stays strict. A wrong, unordered or duplicated literal is a hard error;
  lines without literals are checked exactly as before.
- Tests: the experiment's own `test_trace_dsl.py` (27 tests) still passes, and new tests cover the rejection cases.
- Independent check: the audit's checker has its own Kleene evaluator and verified all 2,880 e256 countermodels with 0
  disagreements.
- Cost: 4.4 tokens per dead-end line on average (p50 5, p90 8, max 16). 91% of lines are within PLAN_CRITIQUE's
  8-token limit. e16 carries none.

**`min_level` labels** (coordinator, 2026-10-08).
- The commit-on-record action of level b is what a search that trusts only its own record commits to:
  - the first candidate its record shows viable;
  - otherwise, from the deepest goal it reached, the strategy's top unverified guess, followed by top guesses down to
    an action;
  - at e0, the top guess from the root.
- The oracle (`judge_action`: the proof can still be completed) labels each of these actions.
- `min_level` is the smallest level of the ladder whose commit-on-record action is viable, or `null` if none is.
- Format, in every step of every record:
  - `"min_level"`: 0 on forced steps;
  - `"viable_levels"`: the list, on decision steps.
- Neither field is in the think text. `data.py` ignores both today.
- An "auto-effort" record (deferred by the coordinator) would read each step's `min_level`, write `<e{min_level}>`,
  then that level's trace. `--level auto` already writes such proofs; their action at each step is the
  commit-on-record action of `min_level`.
- Caveat: an atomic `have` never makes the proof impossible, so the label only bites at box openers, `Or.elim` and
  `exact`. It says nothing about wasted steps.

## Pilot: 5,000 K12 theorems

These are the first 5,000 K12 rows, `--limit 5000`. The eval-class exclusion is the same as progtrace's
(`data.DEFAULT_EXCLUDE` + textbook72 train) and drops 0 rows here.

**Success, fallbacks, time.**
- 4,988 of 5,000 proofs succeed (99.76%) at every level, auto 99.74%.
- 4,985 needed only the sampled strategy, 1 needed a second and 2 needed a third.
- The 12 failures (counting the first strategy's reason) are: 6 over 40 steps, 4 stuck, 2 out of names past n64.
  They are runaway searches in inconsistent (reductio) contexts.
- Time per proof, including the labels: mean about 0.01 s, p99 0.07 s, max 0.8 s, on one core. A whole level takes
  about 5 s wall at 12 workers.

| | e0 | e16 | e64 | e256 | uncapped (`inf`) |
|---|---|---|---|---|---|
| think tokens per decision step, mean / p50 / p90 / max | 0 | 4.2 / 0 / 13 / 16 | 25.5 / 23 / 53 / 64 | 44.8 / 25 / 116 / 240 | 49.7 / 25 / 127 / 653 |
| think tokens per proof, mean / p50 / p90 | 0 | 11 / 5 / 29 | 67 / 50 / 136 | 118 / 65 / 281 | 131 / 65 / 315 |
| decision blocks where the record hit the budget | – | 64.9% | 16.0% | 1.2% | 0.0% (safety cap: 0 hits) |
| blocks with a dead end (all non-ex-falso) | – | 1.2% | 9.8% | 12.9% | 13.4% |
| largest pair (state + effort token + think + action + `<eos>`) | 309 | 310 | 368 | 489 | 786 |

All levels write the same proofs: mean 6.3 env steps, p99 22, max 40.

**Natural length of the search record** (uncapped, top-tier search; decision steps; tokens; share of blocks over each
length):

| proofs | blocks | p50 | p90 | p99 | max | > 16 | > 64 | > 256 | > 512 | > 1,024 |
|---|---|---|---|---|---|---|---|---|---|---|
| K12, all | 13,342 | 25 | 127 | 291 | 653 | 65.5% | 25.2% | 1.6% | 0.09% | 0 |
| K12, ≤ 6 lines | 4,953 | 26 | 135 | 324 | 653 | 71.7% | 25.2% | 2.1% | 0.22% | 0 |
| K12, 7–9 lines | 3,803 | 25 | 127 | 289 | 501 | 63.3% | 25.5% | 1.6% | 0 | 0 |
| K12, 10–12 lines | 4,401 | 24 | 117 | 260 | 570 | 60.4% | 25.0% | 1.1% | 0.02% | 0 |
| 13–20 set (stats only) | 35,214 | 28 | 155 | 360 | 8,961 | 56.4% | 34.7% | 2.9% | 0.24% | 0.03% |

- Records do not grow with K12 proof length: a step's search depends on its state, not on the theorem.
- With the e256 parameters and no cap, 0.9% of blocks exceed 256 tokens, so the 240 cap rarely binds.
- **A 1,024 level would need max_len 1,024** for these proofs. Per pair: p50 100, p99 356, p99.9 566, max 786; 0 of
  31,213 pairs exceed 1,024.
- An e1024 level writes 49.7 tokens per decision step against 44.8 at e256: there is no evidence of need on K12.
- On 13–20, 0.03% of blocks exceed 1,024 tokens.

**What an e16 block can hold under real search.** Of e16 decision blocks:
- 56.4% are empty;
- 40.1% hold one candidate (a viable reduction, usually the action's own line when the goal is short);
- 2.3% hold two viable lines (a one-level descent);
- 1.2% hold a refuted candidate (no countermodel at e16).

Its written action never differs from e256's (one proof). The action its record supports (commit-on-record) differs
from e256's in 10.4% of decision steps. With e16 in the ladder, e16 is the smallest sufficient level in only **0.65%**
of decision steps. That supports dropping it.

**`min_level`** (ladder 0,64,256; decision steps of the e256 proofs):

| proofs | e0 | e64 | e256 | none |
|---|---|---|---|---|
| all | 90.2% | 6.9% | 2.7% | 0.27% |
| ≤ 6 lines | 92.1% | 5.3% | 2.1% | 0.50% |
| 7–9 lines | 88.8% | 7.9% | 3.2% | 0.13% |
| 10–12 lines | 89.2% | 7.7% | 3.0% | 0.11% |
| 13–20 set | 86.6% | 8.5% | 4.3% | 0.60% |

- **e0's top guess is already viable in 90.2% of decision steps.**
- The viable levels are not upward-closed in 0.1% of steps: a level can be viable while a higher one is not, because
  their records end at different points.

**Diversity across strategies** (all four on 499 theorems, e256):
- the mean number of distinct proofs per theorem is 1.16, and 3 at most;
- pairs of strategies write the same proof in 87–97% of theorems;
- they write the same proof and the same thinks in 57–75%;
- the mean think-line Jaccard is 0.76–0.89.

Strategies mostly differ in which dead ends the record shows and which closer comes first, rarely in the proof. When
a K12 state has a viable move it is usually forced.

## Audit: v1, v2 and search traces, side by side

Same 4,988 K12 ids (the pilot's), TRACE_AUDIT's own scripts adapted. Scripts and outputs are in
`/scratch/cowifornia/nd-rl/think/traces/search/audit/`:
- `b_indep.py`: the v2 audit's independent checker, plus its own Kleene check of countermodels;
- `c_stats.py`;
- `m_compare.py` → `compare.md`;
- `gates.py`, `ref_agree.py`, `pairs.py`.

Hard checks:
- the independent checker finds **0 disagreements** in every corpus;
- `trace_dsl.validate(strict=True)` passes on every written record;
- **Lean accepts 600 of 600** random assembled proofs, 120 per level file (`lean_check.py`);
- negative control: 0 of 191 proofs whose final `exact` was corrupted are accepted, while 200 of 200 intact ones are.

| metric | v1 | v2 | search e256 | search e64 |
|---|---|---|---|---|
| candidate lines (actions + verdict) | 21,560 | 20,993 | 0 | 0 |
| verdict predictable by plan string match (TRACE_AUDIT finding 1) | 88.7% | 51.5% | n/a: no candidates | n/a |
| reduction verdict predictable from the line's head alone, points over the majority baseline | +16.1 | +2.4 | **+0.0** | +0.0 |
| action = term of the last plan line (blocks with a plan) | 86.9% | 44.5% | 63.2% | 76.4% |
| action = term of the first reduction line (G1) | 37.2% | 27.1% | 58.6% | 57.1% |
| ex-falso share of dead reductions | 71.2% | 35.2% | **1.6%** | 1.3% |
| decision blocks with a genuine (non-ex-falso) dead end | 13.4% | 8.8% | **12.9%** | 9.8% |
| decision blocks with any `<no>` | 70.9% | 52.1% | 12.9% | 10.2% |
| think tokens per proof, mean (p50 / p90) | 313 (273 / 614) | 237 (202 / 481) | 118 (65 / 281) | 67 (50 / 136) |
| Spearman ρ, think tokens vs K12 proof lines | 0.76 | 0.75 | 0.23 | 0.30 |
| Spearman ρ, think tokens vs the proof's own steps | 0.76 | 0.75 | 0.71 | 0.75 |
| Spearman ρ, block tokens vs search expansions / vs goal-formula tokens (G4) | – | – | 0.79 / 0.62 | – |
| distinct line templates (formulas → F, names → n) | 127 | 121 | 222 | 128 |
| top-10 template share / template entropy (bits) | 60.7% / 4.99 | 68.5% / 4.60 | 57.4% / 5.24 | 69.0% / 4.65 |
| token trigram entropy / conditional entropy given 2 tokens (bits) | 8.52 / 1.78 | 8.53 / 1.77 | 8.35 / 1.79 | 8.33 / 1.71 |
| proof length vs reference, env steps: mean / p50 / p90 / p99 | 0 (reference) | 0 | −1.75 / −1 / 0 / +13 | same proof |
| proofs longer than reference + 4 | 0 | 0 | 3.1% | 3.1% |
| proof identical to the reference | 100% | 100% | 35.5% | 35.5% |
| action agreement with the reference at the reference's own decision states | 100% | 100% | 55.9% (atomic 62.9%, box 41.2%); e0 top guess 51.0% | – |

Reading:
- The verdict on a reduction now depends on the formulas: the head predicts nothing beyond "viable". Countermodels
  make each `<no>` checkable.
- The e256 search is honest and finds its own proofs.
- Each block is short and mostly decides at the first or second candidate, so genuine dead ends stay near 12%.
- Think length tracks search size (ρ 0.79 with expansions). It still tracks goal size too (ρ 0.62), because every goal
  line restates its goal (TRACE_AUDIT finding 6).
- Proofs are shorter than the K12 references on average, which are often redundant. A tail of 3.1% is more than
  4 steps longer: the inconsistent-context runaways.

**CPU gates** (PLAN_CRITIQUE §3.1; the coordinator's four headline numbers come first):

| gate | criterion | search e256 | |
|---|---|---|---|
| G2, real search | ≥ 25% of decision blocks with a certified non-ex-falso dead end | **12.9%** | **fail** |
| G1, shortcut | string-match verdict rule ≤ 60% | not defined (no candidate lines); the head-only rule is at the majority baseline | pass in spirit |
| G1, shortcut | action = term of the first reduction line ≤ 50% | 58.6% | fail |
| G3, length | found ≤ minimal + 2 in ≥ 90%; mean found / reference ≤ 1.10 | ≤ reference + 2 in 95.7%; mean ratio 0.85 (reference, not minimal) | pass vs the reference |
| G4, length tracks search | ρ(think, expansions) ≥ 0.6; ρ(think, goal length) ≤ 0.3 | 0.79; 0.62 | half |
| G5, fit | ≥ 99% of e256 pairs ≤ 1,024 tokens | 100% (max 489) | pass |

## Examples (generator output, verbatim; `|-` ⊢, `~` ¬, `&` ∧, `v` ∨, `>` →, `F` False, `<| |>` ⟨ ⟩)

These are 3 example decision steps at each level for one theorem. Then one theorem where two strategies differ.

How to read decision step 1:
- e16 shows one refuted candidate, with no room for its countermodel.
- e64 shows the refutation with its countermodel `Q ( ~ S )` and then the viable split.
- e256 shows the refutation in the explicit form, the split, and a lookahead into the split's second branch.
- e0's top guess was the refuted candidate, so `min_level` is 64.

**k12_3380** (`THM ( Q v S ) , ( Q > Q ) , ( S > ( Q & ( Q > Q ) ) ) SEQ Q PRF`), strategy `rand`; the three decision steps of the proof, which is the same at every level. Forced steps (premise restatements, `exact`) are omitted.

Decision step 1:
```
STATE <st> h1 : ( Q v S )
      h2 : ( Q > Q )
      h3 : ( S > ( Q & ( Q > Q ) ) )
      n1 : ( Q v S )
      n2 : ( Q > Q )
      n3 : ( S > ( Q & ( Q > Q ) ) )
      |- Q <act>
--- e0 (0 tokens)
(empty)
--- e16 (10 tokens)
|- Q := ( n3 |- S ) .1 <no>
--- e64 (42 tokens)
|- Q := ( n3 |- S ) .1 <no> Q ( ~ S )
|- Q := Or.elim n1 ( fun ( n5 : Q ) => n5 ) ( fun ( n6 : S ) => |- Q )
--- e256 (57 tokens)
|- Q := ( n3 |- S ) .1
|- S <no> Q ( ~ S )
|- Q := Or.elim n1 ( fun ( n5 : Q ) => n5 ) ( fun ( n6 : S ) => |- Q )
n6 : S |- Q := ( n3 n6 ) .1
>>> have n4 : Q := Or.elim n1 ( fun ( n5 : Q ) => by
min_level 64 (ladder 0,64,256; viable at [64, 256]); with e16 in the ladder: viable at [16, 64, 256]
```

Decision step 2:
```
STATE <st> h1 : ( Q v S )
      h2 : ( Q > Q )
      h3 : ( S > ( Q & ( Q > Q ) ) )
      n1 : ( Q v S )
      n2 : ( Q > Q )
      n3 : ( S > ( Q & ( Q > Q ) ) )
      n5 : S
      |- Q <act>
--- e0 (0 tokens)
(empty)
--- e16 (8 tokens)
|- Q := ( n3 n5 ) .1
--- e64 (8 tokens)
|- Q := ( n3 n5 ) .1
--- e256 (8 tokens)
|- Q := ( n3 n5 ) .1
>>> have n6 : ( Q & ( Q > Q ) ) := n3 n5 ;
min_level 0 (ladder 0,64,256; viable at [0, 64, 256]); with e16 in the ladder: viable at [0, 16, 64, 256]
```

Decision step 3:
```
STATE <st> h1 : ( Q v S )
      h2 : ( Q > Q )
      h3 : ( S > ( Q & ( Q > Q ) ) )
      n1 : ( Q v S )
      n2 : ( Q > Q )
      n3 : ( S > ( Q & ( Q > Q ) ) )
      n5 : S
      n6 : ( Q & ( Q > Q ) )
      |- Q <act>
--- e0 (0 tokens)
(empty)
--- e16 (5 tokens)
|- Q := n6 .1
--- e64 (5 tokens)
|- Q := n6 .1
--- e256 (5 tokens)
|- Q := n6 .1
>>> have n7 : Q := n6 .1 ;
min_level 0 (ladder 0,64,256; viable at [0, 64, 256]); with e16 in the ladder: viable at [0, 16, 64, 256]
```

**Two strategies, k12_188** (`THM ( P & P ) , ( ( P & P ) > ( ~ S ) ) , ( ~ ( ~ P ) ) SEQ P PRF`), e256, decision step only:
```
--- strategy dfs
|- P := n1 .1
>>> have n4 : P := n1 .1 ;
--- strategy hyp
|- P := Classical.byContradiction n3
>>> have n4 : P := Classical.byContradiction ( fun hh => n3 hh ) ;
```

## Commands: the full K12 set and the Stage-1 pickles (not run)

The traces go to `/scratch/cowifornia/nd-rl/think/traces/search/full/`. Every level writes the same proofs, so the
think arm and the no-think control T0-S train on identical proofs and actions (coordinator, 2026-10-08).

```
cd /scratch/cowifornia/nd-rl/think/nd-rl
export ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
F=/scratch/cowifornia/nd-rl/think/traces/search/full
for L in 0 64 256; do
  nice code/.venv/bin/python -m code.thinking.searchtrace --level $L --workers 16 --out $F/e$L.jsonl
done
# e16 only if the original 20/20/30/30 mix is wanted (the coordinator's ladder drops it):
nice code/.venv/bin/python -m code.thinking.searchtrace --level 16 --workers 16 --out $F/e16.jsonl
# the canonical (e256) proofs as K12-style records, for the no-think control:
nice code/.venv/bin/python -m code.thinking.searchtrace --level 256 --workers 16 --no-labels \
  --out $F/e256_again.jsonl --emit-plain $F/plain_e256.jsonl
```

The last command re-generates e256 deterministically, identical to `$F/e256.jsonl`. Alternatively call
`searchtrace.emit_plain(e256.jsonl, plain.jsonl)` directly.

Stage-1 pickles (`data.py`, as in [graded_traces/README.md](../graded_traces/README.md)):

```
TB=/scratch/cowifornia/nd-rl/think/nd-takehome/data/eval_only/textbook72/textbook_train.jsonl
D=/scratch/cowifornia/nd-rl/think/runs/data
# think arm S, graded 20/20/30/30 as in graded_traces/README.md (levels 0/16/64/256):
nice code/.venv/bin/python -m code.thinking.data --graded 0:0.2 16:$F/e16.jsonl:0.2 64:$F/e64.jsonl:0.3 \
  256:$F/e256.jsonl:0.3 --extra-exclude $TB --no-lean --out $D/S1g4.pkl
# the same on the coordinator's ladder 0/64/256 (mix 20/40/40 here; Leon's choice):
nice code/.venv/bin/python -m code.thinking.data --graded 0:0.2 64:$F/e64.jsonl:0.4 256:$F/e256.jsonl:0.4 \
  --extra-exclude $TB --no-lean --out $D/S1g3.pkl
# no-think control T0-S on the same proofs, bare actions (the T0 layout; `--no-trace-mode empty` writes
# `<think> </think>` instead):
nice code/.venv/bin/python -m code.thinking.data --plain $F/plain_e256.jsonl --no-trace-mode none \
  --extra-exclude $TB --out $D/T0S.pkl
```

Notes on the build:
- With the default closers-first at every level, no `--graded-any-proof` is needed.
- With `--closers-high-only`, levels can prove a theorem differently. Then add `--graded-any-proof` and give level 0
  its own file (`0:$F/e0.jsonl:0.2`).
- The pilot versions of the S1g3 and T0-S pickles were built: `traces/search/pickles/pilot_{S1g3,T0S}.pkl`.
  - Both hold 4,988 records and 31,213 pairs, and the states are identical step by step.
  - S1g3: pairs mean 107 tokens, max 428, none over 1,024; 3.33M tokens, 0.92M of them loss tokens.
  - T0-S: pairs mean 92 tokens, max 306; 2.86M tokens, 0.48M of them loss tokens.
- The training commands are as in `graded_traces/README.md`, with `--data` pointing at these pickles. For a
  `lean_thinke` pickle, `train.py`'s `auto_max_len` keeps max_len 1,024. Nothing was launched.

**CPU estimate for the full 154,977-proof run.**
- The pilot measured about 11 ms of one core per proof per level, labels included (5,000 proofs in 4.5 s wall at
  12 workers).
- Per level: 154,977 × 11 ms ≈ **28 CPU-minutes**, about 2 minutes wall at 16 workers.
- Levels 0, 64 and 256 plus the plain emission come to about **1.5 CPU-hours**, about 10 minutes wall. Adding e16 is
  about 2 CPU-hours.
- `data.py` re-validates every trace in a single process: the pilot's graded build took 20 s for 4,988 records
  (about 4 ms each), so about 10–12 minutes per pickle for the full set.
- The heavy tail (about 1 s for the slowest single proof; node cap 3,000) is included in the mean. I'd allow 2× margin.

## Complete mode (`--complete`): a planner fallback for harder theorems

**Status (2026-10-09):** built and measured on CPU (at most 12 workers, `nice`), not committed. Code:
[`code/thinking/searchcomplete.py`](../../../../thinking/searchcomplete.py) (new), hooks in `searchtrace.py`, tests in
`code/tests/thinking/test_searchcomplete.py` (new). Base commit `79749ce`.

**What it does.** It is opt-in (`trace_for(..., complete=True)`, CLI `--complete`) and runs only after all four
default strategies have failed, so every proof the default generator writes is unchanged.
- It replays the honest search step by step, with the same strategy, budgets and records.
- It counts as trouble when the search is stuck, the box depth reaches 5, names pass n56, or the step count passes a
  length budget. The budget is a whole-theorem plan + 6 when one is found in 20 s; otherwise 48 steps, re-armed after
  each repair.
- On trouble it backs off to the states where the open boxes were opened: boxes opened in an inconsistent context
  first (outermost first), then the others (innermost first), then the current state.
- At each point a planner solves that point's sub-theorem: the hypotheses in scope ⊢ the focused goal, reduced to a
  minimal support by truth tables. The planner is the fork's shortest-proof search (`minlen` through
  `test_21_30/fastlen.py`, iterative deepening on proof length). It reads the state only. A test checks that no
  planner function takes a reference.
- If no point is solved within 6 s (30 s at the current state), a second round tries the same points with 90 s. The
  last resort is a plan for the whole theorem from the root, which closes the proof.
- The walk rewinds to the point and follows the plan, translated into the environment's names, until the plan closes
  its frame. Then the honest search resumes.
- A plan step keeps the search's record when the search, run at that state, commits to the same action. Otherwise
  its think is empty and the step carries `"planner": true` in the step metadata. No record is written for a move
  the search did not make. Labels (`min_level`) are computed at every step as usual. Every step passes `viable`,
  `check_step` and strict validation, as in default mode.

**Backward compatibility.** Default output is byte-identical to `79749ce`: sha256 over the first 120 K12 records at
levels 0, 64, 256 and auto matches the digests computed at that commit (test). With `complete=True`, default
successes are unchanged (test, 40 rows).

**Results at e256** (ladder 0/64/256). The sealed 13–20 set was run with `--analysis-only`: statistics only, no
traces written. The fresh sets are composites built with the 13–20 pipeline (`build_composites.py` on the same
pieces, other seeds) and screened with `excl.py`'s index (`seal.hits`): 2 hits and 1 duplicate removed from 16–20,
none from 13–15. `hard712/pool.jsonl` did not exist when this ran.

| set | theorems | default success | `--complete` success | repaired by the planner | decision steps from a plan (without a record) |
|---|---|---|---|---|---|
| sealed 13–20 (statistics only) | 3,003 | 87.98% | **99.93%** (2 fail) | 359 (12.0%) | 7.0% (2.6%) |
| fresh composites, 13–15 lines | 1,490 | 92.75% | **100%** | 108 (7.2%) | 4.1% (1.6%) |
| fresh composites, 16–20 lines, held-out half | 749 | 85.31% | **99.73%** (2 fail) | 108 (14.4%) | 8.0% (3.2%) |
| fresh composites, 16–20 lines, dev half (used to develop the fixes) | 748 | 83.16% | **99.87%** (1 fail) | 125 (16.7%) | 9.3% (3.7%) |
| K12 pilot theorems the default cannot prove | 9 | 0% | **100%** | 9 | — |

In repaired proofs, about half the steps come from a plan: 53% on fresh 13–15, 42% on 16–20. About a third of steps
carry no record: 33% and 26%.

- **Genuine dead ends** (non-ex-falso dead-end lines in a decision block): 16.5% of decision blocks on sealed
  13–20, 15.6% on fresh 13–15 and 16.0% on fresh 16–20. Ex-falso dead ends are under 1% of dead-end lines. The
  default generator reached 17.4% on the 13–20 theorems it could prove, and 12.9% on K12. Planner steps without a
  record have no dead ends, so they dilute the share slightly.
- **`min_level`** (share of decision steps), e0 / e64 / e256 / none viable: 87.0 / 8.3 / 4.1 / 0.6% on sealed
  13–20, 87.4 / 7.6 / 4.3 / 0.7% on fresh 13–15, and 87.6 / 7.5 / 4.3 / 0.7% on fresh 16–20 (K12 default: e0 90.2%).
- **Length overhead against the shortest proof** (ND lines). The honest search's proofs are long, and most of the
  overhead does not come from repairs.
  - Sealed 13–20, the 1,672 proofs (55.7%) whose shortest length is certified exact (`minimality.cert`): mean +5.7,
    median +4, p90 +16, mean ratio 1.38; 28.8% are shortest and 40.8% within +2. For the other 1,329, the shortest
    length is only bounded, and the mean overhead lies between +8.3 and +11.0.
  - Fresh 13–15, the 666 proofs certified exact by `fresh/certify.py` (fork `minlen`, staged to 14 lines; 726 of 1,490
    run, 60 timeouts left out): mean +4.9, median +3, p90 +14, ratio 1.36, 30.8% shortest. Repaired proofs: mean +4.6,
    median +2 (46 certified).
  - Fresh 16–20: not certified, since minlen does not finish at 16–20 lines. The overhead against the construction's
    length is a lower bound on the overhead against the shortest proof: mean ≥ +7.7, median ≥ +6. For repaired proofs
    it is ≥ +13.0 (median ≥ +11), because a local repair keeps the drift that came before it.
- **Cost.** The sealed 13–20 run took 55 min of wall time on 6 workers, about 5.5 CPU-hours (default mode: 33 s on 12).
  Per theorem: median 0.07 s, p99 49 s. Repairs cost up to minutes. A failure costs up to about 20 min, because the
  last-resort whole-theorem plan has a 600 s limit.

**Residual failures.** All 5 remaining failures (2 sealed, 2 held-out, 1 dev) are "planner found no plan" on 18–20-line
theorems. The search re-drifts after a repair. Every back-off point's sub-proof then needs more than 90 s, and the
whole theorem more than 600 s. A larger budget, or a planner that is not limited to shortest proofs, would close some
of them.

**Bugs found on the way.**
- Under `python -m`, `searchcomplete`'s import created a second `searchtrace` module, so its `ProofFail` was not
  caught. The run counted such failures as crashes. `main()` now aliases the module.
- When the goal was already in scope, the planner returned no proof, because minlen treats goal ∈ premises as
  nothing to prove. That ended in a failed whole-theorem plan. The planner now plans `exact`.
- The first version of that `exact` cited the first name with the goal's formula. The environment accepts only the
  frame's last name, so it now restates the goal first, as the search does. This was found through one sealed-set
  failure ("non-viable action"), reproduced on fresh composites (f1315_0098, f1315_0124) and fixed before the final
  runs.

**Development data.** The fixes were developed on K12, the fresh composites and the 16–20 dev half. The sealed
13–20 set was only re-measured, after each round of fixes; its failure messages were read as statistics.

**Commands.**

```bash
# statistics only on the sealed 13-20 set (no traces written)
python -m code.thinking.searchtrace --source /scratch/cowifornia/nd-rl/think/runs/pools/test_13_20/test_13_20.jsonl \
  --level 256 --workers 10 --no-exclude --analysis-only --complete --out .../t1320_e256_complete5.jsonl
# fresh composites (traces written)
python -m code.thinking.searchtrace --source .../complete/fresh1620/held_src.jsonl --level 256 --workers 6 \
  --no-exclude --complete --out .../complete/fresh1620/held_e256_complete.jsonl
python .../complete/hard_report.py STATS.json ROWS.jsonl t1320|fresh [CERTIFIED.jsonl]
```

## Open issues and decisions for Leon

1. **Shallow search on K12 (G2 fail).**
   - Genuine dead ends occur in 12.9% of e256 blocks.
   - In 87% of decision blocks the strategy's first or closing candidate is viable. In 41% of states the context is
     inconsistent, and truth tables cannot refute anything there.
   - Options, each a design decision:
     - (a) build the dead ends from the *policy's* own wrong actions (PLAN_CRITIQUE alternative 1; see item 6);
     - (b) deliberately weaker strategies that try more doomed candidates first. That is honest but artificial;
     - (c) run the generator on harder theorems. On 13–20, 17.4% of blocks have a dead end.
   - On K12 alone, G2 will not be met by an honest, sensible search.
2. **Harder theorems: the generator fails on 12% of the 13–20 set** (stats-only analysis, no traces written).
   - Causes: 40-step cap overruns, stuck searches and names past n64.
   - With `--max-steps 64` the failure rate drops to 6.6%, but proofs get longer: median +6 steps over the
     (near-minimal) reference.
   - The cause is Markov drift in reductio contexts. The search re-derives the box's own goal through the box's binder,
     so the proof grows.
   - **Addressed by complete mode** (§ Complete mode, opt-in `--complete`): 99.93% on this set (statistics only),
     with planner steps marked and left without a record.
3. **Length still tracks goal size** (ρ 0.62, G4 half-fail): each goal line restates its formula (TRACE_AUDIT finding
   6). Hole ids or a "same goal" token would cut this. That is a vocabulary change.
4. **The action is the first reduction's term in 58.6% of blocks** (G1's second clause). This is inherent to a
   search that is right first time. Making it fall would mean either showing more doomed candidates first, or
   checking the policy's candidates (item 1).
5. **Strategy diversity is low in proofs, higher in thinks.** There are 1.16 distinct proofs per theorem across 4
   strategies, because valid moves in K12 states are mostly forced.
6. **Future hook: model-proposed wrong actions** (coordinator item 5, not built).
   - The search would take, per state, a list of the policy's rejected actions from a file keyed by the state's token
     string (not by K12 step: the search's states differ from the reference's in 64% of proofs).
   - It would examine them as candidate lines before its own reductions, each with the environment's or `step_check`'s
     verdict and `<no>`, as v2 did with distractors.
   - Existing distractors (`gen_runs/wave*/distractors.jsonl`) are keyed by reference step, so they would need
     re-sampling on these states.
7. **Labels.**
   - `min_level` uses viability (the proof can still be completed). An atomic `have` is always viable even when it is
     wasted, so labels mostly flag box, `Or.elim` and `exact` decisions.
   - e0's top guess is viable in 90.2% of decision steps.
   - The deferred auto-effort mode would make most steps e0.
8. **Inconsistent-context runaways** make 3.1% of proofs more than 4 steps longer than the reference. 0.24% of
   theorems fail all four strategies; they are counted and not written.
9. **The DSL extension changes `trace_dsl.py`.**
   - It is backward-compatible: every existing trace validates unchanged, and the 27 experiment tests pass.
   - During this work another process in the shared worktree briefly reset `trace_dsl.py` and restored it with a
     stash/commit/pop at 08:16 UTC. The extension is intact now; `searchtrace` relies on it, so commit it with
     `searchtrace.py`.

## Checks that ran

- `pytest code/tests/thinking`: **335 passed, 1 skipped**, including the 26 new search-trace tests.
- `pytest .../test_trace_dsl.py`: **27 passed**.
- pyright on the whole project: **0 errors**.
- ruff check and format: clean for every file I touched.
- `make -C code check` exits 2, but only on ruff findings in `collapse_audit/` (37) and on unformatted files in
  `collapse_audit/` and `hard712/`. Both are other agents' untracked folders, not part of this work.
- The pilot, audit and Lean checks above, via `traces/search/run_pilot.sh`.
- Complete mode (2026-10-09): `pytest code/tests/thinking` **374 passed, 1 skipped**, including 11 complete-mode tests
  (byte-identical default output at e0/e64/e256/auto against `79749ce`, default successes unchanged, repairs validate
  strictly, reference independence). ruff and pyright are clean on the changed files. A stress test followed a
  planner plan from all 2,532 states of 130 fresh-composite proofs (2,008 plans): 0 non-viable actions, every plan
  closed its frame.

## Files

| what | where |
|---|---|
| generator, CLI | `code/thinking/searchtrace.py` |
| validator extension | `code/experiments/current/thinking_traces_20261005/trace_dsl.py` |
| graded hook | `code/thinking/data.py` (`any_proof`, `--graded-any-proof`) |
| tests | `code/tests/thinking/test_searchtrace.py`, `code/tests/thinking/test_searchcomplete.py` |
| complete mode (planner fallback) | `code/thinking/searchcomplete.py`; runs in `/scratch/cowifornia/nd-rl/think/traces/search/complete/` (`fresh/`: 13–15 composites, `certified.jsonl`, `e256_complete3.*`; `fresh1620/`: 16–20 dev/held-out halves; `hard_report.py`) and `sealed_analysis_13_20/t1320_e256_complete5.stats.json` (statistics only) |
| pilot traces and stats (5,000 K12) | `/scratch/cowifornia/nd-rl/think/traces/search/pilot/` (`e{0,16,64,256,1024,inf,auto}.jsonl`, `e256_nocap.jsonl`, `plain_e256.jsonl`, `*.stats.json`, `e256.diversity.json`) |
| 13–20 analysis (stats only, no traces) | `/scratch/cowifornia/nd-rl/think/traces/search/sealed_analysis_13_20/` |
| audit | `/scratch/cowifornia/nd-rl/think/traces/search/audit/` (`compare.md`, `gates.json`, `ref_agree.json`, `pairs.json`, `lean_check.json`, per-corpus `b_indep.json` / `c_stats.json`) |
| pilot pickles | `/scratch/cowifornia/nd-rl/think/traces/search/pickles/` |
| driver | `/scratch/cowifornia/nd-rl/think/traces/search/run_pilot.sh` |
