# Follow-up brief: replicate the depth-3 result, test reductio properly, run the cap-8 dial

Unattended, up to ~30 hours. Read `campaign.md`, `phase2.md`, and **`review_campaign.md`** first —
the review is the reason this run exists. `CAMPAIGN_BRIEF.md` still applies (hard rules, pod
helpers, budget discipline, `LIMIT_HIT` flagging, normalised counts, base reachability with
every claim, numbers with sources as you go). Branch: **`dan_novelty`**; push at milestones;
never push `main`; never touch `~/nd-rl`; no test-file runs.

Pods: none are running; create what you need with `podnew p1` etc. (A40, $0.49/hr), up to
**3 at once**; delete each when its work is pulled. Kill switch: **2026-09-17 16:00 UTC**; be
finished, pulled and pushed by **2026-09-17 12:00 UTC**. `STATUS.md` is fresh (the first
campaign's is `STATUS_campaign1.md`); update it at every milestone; end with `DONE <UTC>`.

## A. Replicate depth-3 at f = 0 (highest priority; start first)

The first campaign had two training seeds on one f = 0 set (acquisition 0.341 / 0.271) and two
on one f = 0.1 set (0.355 / 0.316). Separate the two sources of variance:
- Assemble **three new f = 0 depth-3 pretraining sets** with different assembler seeds
  (`make_coverage_sets.py`; verify zero written-form depth-3 proofs in each by reclassifying
  the file, and record the achieved reductio/derived-ORE frequencies). Train **two training
  seeds on each** → 6 new Stage-1 models (plus the existing 2 = 8 arms at f = 0).
- Assemble **two new f = 0.1 sets**, two training seeds each → 4 new arms (6 total at f = 0.1).
- Run the identical expert iteration (k = 32, 8 rounds, retain 20k of the arm's own set) on
  the existing depth-3 target pool, and a **frozen control for every new Stage-1 model** (256
  attempts per target).
- Report: acquisition per arm, then mean ± SD and the full list for f = 0 and f = 0.1; a
  variance decomposition (between-set vs between-training-seed); first round with a depth-3
  proof per arm; and the frozen controls' depth-3 counts (expected 0). The replication
  question is: does the f = 0 distribution overlap the f = 0.1 distribution, and is any f = 0
  arm near zero? Plot both distributions as dot strips on one axis.
- Base reachability for a sample of ≥ 200 f = 0 depth-3 proofs per new arm (`novelty.py` under
  the arm's own Stage-1 model). Fix the coverage bookkeeping caveat from the review: when you
  run any base pass@k, store per-distinct-proof hit counts, not just one proof per theorem.

## B. A reductio test that can fail (second priority)

The first campaign's reductio targets never required the reductio shape (56 were classical-only
but all had a `( ~ ( ~ X ) )` premise or hypothesis; `DN` on it sufficed). Build a pool where the
double negation must be **derived**:
- From the strict long pool (or a fresh generator run), take theorems whose generating proof
  contains the reductio pattern **and** whose sequent contains **no `( ~ ( ~ … ) )` subformula
  anywhere** (premises or conclusion) **and** that are not intuitionistically provable
  (`intuit.py`) **and** have `min_lines_ub ≥ 7` or none. Target ≥ 500; transfer ≥ 250; disjoint
  by class from every training set and validation-36. Print ten of them and check by hand-eye
  that a proof genuinely needs to assume the negated goal.
- Report two acquisition predicates: the strict reductio shape (`patterns.reductio`) and a
  loose one — `DN` whose cited line is not `PR`/`AS` (a derived double negation). Both
  classified on the model's proof.
- Arms: reductio f = 0 (3 training seeds on the existing set) and f = 0.1 (2 seeds), frozen
  controls, and base pass@10⁴ on 300 targets for one f = 0 model. Same EI settings.
- Expected (write your own expectation before running): if the first campaign's "RL does not
  invent unseen rule sequences" is right, f = 0 acquisition stays ≈ 0 under both predicates
  *and* the plain solve rate on these targets is near the frozen control's; if RL composes
  `NEGI` + `DN` from their separate uses, the loose predicate rises first.

## C. Cap-8 dial for non-degenerate derived-ORE (third priority; skip if A+B are not done by 2026-09-17 04:00 UTC)

Within cap 6, 97% of expressible derived-OREs are the degenerate `( X v X )` form, so the
first campaign's P1 dial measured a template. Raise the cap for this arm only:
- Generate a **cap-8** raw pool (every proof `verify_text` length ≤ 8; assert at load with
  `--cap 8`), label with `patterns.derived_ore_strict` (disjunction obtained by a rule, disjuncts
  differ). Assemble 155k-proof sets with strict-derived-ORE frequency f ∈ {0, 10⁻³, 10⁻²}
  (degenerate derived-OREs held at their natural cap-8 rate, reported), two training seeds at
  f = 0 and one at each other f.
- Targets: strict-derived-ORE theorems with `min_lines_ub ≥ 9` or none (bound the prover at 8),
  ≥ 500; transfer ≥ 250; disjoint by class.
- Same EI, frozen controls, strict-predicate acquisition, base reachability. This arm is
  labelled **cap 8** everywhere; never pool it with cap-6 numbers.

## Deliverables

`followup.md` (≤ 400 words: did depth-3 replicate, what reductio showed, what cap-8 showed;
figures `figures/followup_*.png`), sections appended to `phase2.md`, `numbers.md` and
`log.md` (dated), `STATUS.md` ending in `DONE`. Pull all results and checkpoints before
deleting pods. Every count normalised; every "RL solved X" with base reachability; expected
results written down before each block runs.
