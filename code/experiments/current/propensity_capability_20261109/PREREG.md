---
written_on: 2026-11-09
written_by: agent:claude
---

# Pre-registration: propensity and capability by proof direction, base to RL end

Written before any result exists. At freeze, the coordinator records in `log.md` the sha256 of this file, of
`analyze.py` and of the runner's `pc_lib.py`. `analyze.py` implements every rule below; the constants at its top
must equal the numbers here. Anything changed after a result has been seen is an amendment: it is dated, says
who made it and why, and the frozen tables stay beside the new ones. Build spec: [SPEC.md](SPEC.md).

## Question

For a trained prover and a proof direction d (reductio, negation, conditional, cases), separate how often the
model opens with d unprompted (**propensity**) from how often it finishes when its first proof action is forced
to be d's opening (**capability**). From a pretrained base to the end of its RL line, which of the two moved?

| propensity | capability | reading |
|---|---|---|
| up | flat | **elicitation**: RL selected something the base could already do |
| flat, rare or down | up | **candidate capability gain** |
| up | up | **mix**, sized by the two factors |
| flat, rare or down | flat, rare or down | **neither** |

"Candidate" because a forced first move can put the model in a state it rarely visits (see traps).

## What went wrong in the factor read, and what stops it here

The factor read ([PREREG](../factor_read_20261106/PREREG.md), [summary](../../../../experiment-summaries/2026-11-06-factor-read-strict-106/README.md))
had three failures this design is built against.

| factor read | here |
|---|---|
| The primary measure was the exact log-probability of one reference proof, averaged in log over theorems. RL's attempts left the reference route, so it fell where sampled rates rose, and the mean of logs was dominated by a few theorems near zero. | Nothing follows a reference proof past one action: E scores only d's opening. Both factors are **means of probabilities over theorems** (the share of attempts), and logs are taken of those means. E is checked against U's sampled first moves on every checkpoint. |
| What counted as "picking the route" was redefined after the results. | The openings, the first-action rule and "uses anywhere" are fixed below and in `pc_lib.py` (sha256 at freeze). A change after any result is an amendment, with the frozen tables kept. |
| A dose comparison at 8 to 13 times unequal RL compute produced a claim that had to be withdrawn. | Stated up front, below. No reading compares doses, and the tables print x beside every RL end. |

Also new: "flat" must be shown by an interval, not by the absence of a significant change.

## Definitions

**Sets.** `strict106` (the 106 strict theorems, where the reductio is required) and `transfer_other` (240 of the
other 726 in `transfer_track832.jsonl`, fixed before the read). Read-only evaluation sets. **Sets are never pooled.**

**First action.** Dan's environment accepts premise lines (`have n : F := h<k> ;`) only first, in order, at depth 0,
and a premise not restated cannot be cited. So the first action is **the first non-premise action**. E and F work at
the state after every premise line, in order; U classifies each attempt's own first non-premise action.

**Openings and applicability** (`pc_lib.opening`, `pc_lib.applicable`, read off the parsed formulas; a property of
the theorem, identical in every arm and at every checkpoint):

| d | opening (one action) | applicable when |
|---|---|---|
| reductio | `have n : ¬¬G := ( fun ( x : ¬G ) => by` on the goal G | always |
| negation | `have n : ¬A := ( fun ( x : A ) => by` | the goal is ¬A |
| conditional | `have n : A → B := ( fun ( x : A ) => by` | the goal is A → B |
| cases | `have n : G := Or.elim p ( fun ( x : A ) => by` on the first premise A ∨ B | a premise is a disjunction |
| direct | first action opens no box | always; classification only, never forced |

**Propensity, exact first move (primary).** p_d(t) = exp(`logp_open`): the probability that the first action is d's
opening at the state after all premise lines, under softmax(logits / 0.8), marginalised over the env's 33 name bases.
Set level: **P̄ = the mean of p_d(t) over applicable theorems**, the expected share of attempts that open with d.

**Propensity, sampled first move.** Σ`first` / Σk from U (k = 32). Used only to check E.

**Propensity, uses anywhere (secondary).** Σ`used` / Σk from U: the attempt opens d somewhere, at any depth, with
every line up to the box's assumption valid (`pc_lib.moves`): a reductio on **any** formula X (the box ¬¬X with
binder ¬X), a negation or conditional introduction on the goal in focus, a case split on a premise disjunction.

**Capability (primary).** **C̄ = Σ`solved` / Σk over applicable theorems** from F: k_f = 16 attempts per theorem whose
prefix is every premise line then d's opening; Lean decides. Endings by the step cap and by refusal are recorded apart.

**Success through d's opening.** Per theorem, P(solve and open with d) = p_d(t) × c_d(t), exactly, because forcing the
opening and sampling the rest is conditioning the model on that first action. Set level S = mean of p_d(t) c_d(t).

## Primary measure and why

For each line (dose, seed), set and direction: **Δlog10 P̄ and Δlog10 C̄, base to RL end.**

1. The forced arm conditions on exactly the event the first-move propensity measures, so the two factors split one
   quantity, success through d's opening. "Uses anywhere" has no such identity with F, so it is secondary.
2. E has no sampling floor, so propensities far below 1/k are measured, not bounded.
3. Means of probabilities cannot be dominated by a handful of theorems at 10⁻¹⁰, as the factor read's mean of logs was.
4. Log ratios put both factors on the scale on which they multiply, and they answer the reading's question
   (could the base already do it?) at any level: 1 % → 10 % is a large change, 90 % → 95 % is not.

## The reading rule (frozen)

- **Change:** Δ = log10(end / base) of the set-level values. For F and U rates, (x + 0.5) / (k + 1), so 0 of k stays
  finite. P̄ is clamped at 10⁻¹² before the log.
- **Interval:** 95 % percentile bootstrap over theorems, 2,000 resamples, paired (the same resampled theorems at base and
  end), RNG seeded by the cell's name, so a rerun gives the same interval.
- **State of each factor:**
  - **rare** if it is below **1 %** at both checkpoints (its change is not read);
  - **up** if Δ ≥ **+0.15** (×1.41) and the interval's lower bound is above 0;
  - **down** if Δ ≤ −0.15 and the upper bound is below 0;
  - **flat** if the whole interval lies inside **±0.30** (×2);
  - **unresolved** otherwise.
- **Reading:** the table above. Propensity up with capability rare or down is **other (outside the table)**. Any
  unresolved factor makes the reading **unresolved**.
- **No reading** (said as such, with the reason) when the cell has fewer than **20** applicable theorems, when more than
  5 % of its applicable theorems lack complete rows, when E fails its check against U at either checkpoint (below), or
  when more than **half** of the forced attempts end by refusal or the step cap at either checkpoint.
- **Sizing a mix:** the change in P̄ × C̄ is split exactly between the factors by the Shapley rule,
  ΔP̄ (C̄_b + C̄_e) / 2 and ΔC̄ (P̄_b + P̄_e) / 2, in points of attempts; the share from propensity is reported for a mix.
  It is withheld, with a flag, when the change in S and the change in P̄ × C̄ differ by more than a factor of 2 (or in
  sign) while either is at least 1 point: then the set-level factors do not account for the success they are meant to
  split.
- **E against U:** per checkpoint, set and direction, the sampled count Σfirst against the count E predicts, Σ k p(t).
  It fails when they are at least 3 attempts apart, |z| > 4 under binomial variance, and the ratio is beyond ×2.

## Why these thresholds: the factor read's noise

From `python analyze.py --calibrate /scratch/cowifornia/nd-rl/factor_read/summaries` (counts only). The closest
analogues there: capability = Lean-solved from the forced pick on the reference T; propensity = the exact, name-free
probability of that pick (the first action on the 37 theorems with no premises, after the reference setup elsewhere).
Three seeds per dose, so each SD below rests on three values.

| quantity | 0 % | 0.1 % | 1 % |
|---|---|---|---|
| seed-to-seed SD of Δlog10 C̄ | 0.15 | 0.11 | 0.05 |
| seed-to-seed SD of ΔC̄ in points | 4.5 | 5.6 | 4.6 |
| seed SD of log10 P̄, bases (all P̄ < 3.3 %) | 0.66 | 1.46 | 0.92 |
| seed SD of log10 P̄, RL ends | 0.59 (P̄ < 10⁻⁵) | 0.97 (P̄ < 10⁻⁵) | 0.08 (P̄ 11–15 %) |
| 95 % half-width of Δlog10 C̄ at k_f 16 (thinned from 64) | 0.14–0.16 | 0.15–0.18 | 0.12–0.14 |
| 95 % half-width of Δlog10 P̄ | 0.51–0.53 | 0.52–0.64 | 0.22–0.42 |

- **0.15 for "counts":** the largest within-dose seed SD of the capability change (0.15), and about twice the seed SD of
  log propensity where propensity is above 1 % (0.08). A smaller change is within what reruns of the same recipe differ by.
- **±0.30 for "flat":** twice that. At the 106's interval half-width (0.12–0.18) a flat capability can be shown when the
  point change is within about ±0.15, so elicitation is a reachable reading and not ruled out by design. A "flat" cannot
  hide a doubling.
- **1 % floor:** below 1 %, seeds of one recipe differ by 0.6–1.5 in log10 propensity, 4 to 10 times the threshold, so a
  change there cannot be told from which seed trained the model. Such changes are reported, never read.
- The same thresholds apply to both factors, so neither is favoured in the split. On the 240 set intervals should be
  narrower (by about √(106/240) if theorems vary alike); this is not measured.

## Seeds, theorems and doses

- **Theorems** are the unit of the bootstrap. **Seeds** are the unit of replication: never pooled into one bootstrap,
  always shown separately. Dose summaries give the mean and range of Δ over seeds, with no interval.
- **Dose-level reading:** the reading all read seeds share (at least two), or the reading of two seeds when the third
  is unresolved or has no reading ("mix (2 of 3)"). Every seed giving the same non-reading reports that. Otherwise
  **seeds disagree**. This rule is the guard against reading noise across 3 doses × 4 directions × 2 sets.
- **Checkpoints:** a line's RL end is the `rl_end` with the largest x; any other is a labelled secondary, never in a
  dose reading. dko_s1, dd01_s1 and dd01_s2 end on alarm checkpoints (the factor read found their padding); they are
  read and flagged. A missing checkpoint gives that seed no reading and the dose rule uses the others. Dan's full-corpus
  base has no RL end: levels only.

## Unequal RL compute, stated up front

The 0 % and 0.1 % lines ran 8 to 13 times the RL compute of the seed-matched 1 % lines: x at the RL ends is 5.07,
5.93, 5.94 (0 %), 4.42, 5.43, 5.63 (0.1 %) and 0.41, 0.45, 0.69 (1 %). The guard also halved the lr on most long lines.
So:

- Every reading is within one line, base to its own end, and describes RL at that line's compute.
- **No claim compares doses** ("capability rose more at 0 % than at 1 %"). Such contrasts are exploratory and carry
  the x ratio. The tables print x beside every RL end, and the figure's legend gives each dose's x range.
- The factor read's matched-compute snapshots still exist (`/scratch/cowifornia/nd-rl/factor_read/matched/ckpts/`:
  dko_s0 it 60 x 0.417, dko_s1 it 70 x 0.479, dko_s2 it 100 x 0.712, dd01_s0 it 60 x 0.440). They are not in the
  SPEC's list. If the coordinator adds them with `stage: "matched"`, they are read by the same rule as a secondary, the
  only place a dose contrast is made. Matched x is not matched lr. If they are not read, no dose contrast is made.

## Primary, secondary, exploratory

- **Primary:** the reading per line for each direction and set with at least 20 applicable theorems, and the
  dose-level reading. The 9 dose lines only.
- **Secondary:** (1) the same rule with propensity = uses anywhere; (2) finishing once chosen (U's solved / used) beside
  C weighted by each theorem's use and beside C̄: their gaps are selection within and across theorems; (3) unforced
  success, the share of it using each direction, first-move shares, and S; (4) forced against unforced endings;
  (5) capability on the familiar subset (below); (6) router or default shift (below); (7) matched compute, if read.
- **Exploratory:** reductio on the 106 by schema; Dan's full base; secondary ends; any dose contrast; if the runner adds
  `solved_first` to U rows, U's solved / first beside C weighted by first moves (they should agree, by the identity);
  anything else, labelled post-hoc.

## Router or default shift (secondary, added 2026-11-09 before any result)

When RL raises a direction's propensity, does it rise only where the direction is needed (a router, the disposition
tied to context) or also where it is not (a shift of the default)? Each theorem and direction has a class fixed before
the read (`dir_class`): **required**, **helpful**, **idle**, or null.

- **Known asymmetry:** Dan's RL targets are only 2.7 % covered by theorems that require the reductio (figure from the
  coordinator), so RL had little direct pressure to use it where it is optional. A router reading is what weak pressure
  predicts anyway and is weak evidence for a context-tied disposition; a default shift would be strong evidence, since
  it happened without direct pressure.
- **Measure:** per class, P̄ (exact first move) at base and end, and Δ in **log10 odds** of P̄, with the same bootstrap
  (theorems resampled within class). Log odds because the simplest default shift, a constant bias on the opening's logit,
  moves every theorem's log odds by the same amount and does not saturate as the required class nears 100 %; below 10 %
  it equals the Δlog10 used elsewhere. Uses anywhere is reported per class beside it, with no verdict.
- **Classes compared:** required against idle. When no optional theorem is labelled helpful or idle (the runner sets
  them null), the **fallback** compares required against the optional theorems as a whole. Helpful is reported between.
  Each compared class needs 20 theorems.
- **Ratio:** f = Δ(not needed) / Δ(required). A pure default shift gives f ≈ 1, a pure router f = 0.
- **Verdict:** **no rise to locate** unless the required class is up (same state rule as above). Then **router** if the
  not-needed class is flat, rare or down, or the interval of Δ(not needed) − ½ Δ(required) lies below 0; **default
  shift** if the not-needed class is up and that interval lies above 0; otherwise **unresolved**. A not-needed class
  below 1 % at both checkpoints reads as router, flagged: a shift that never reaches 1 % does not change behaviour, and
  below 1 % the seed noise is 0.6–1.5 in log10. Dose level: the same agreement rule.
- **Confound:** for the reductio, "required" is the 106 and "not needed" is drawn from the other set, so the contrast
  is also a contrast of sets (schemas, lengths, a held-out σ set). In the fallback this is all it is.

## Known traps and what the analysis does

| trap | what the analysis does |
|---|---|
| A forced reductio on an optional theorem can be finished by proving the goal inside the box. | The sets are never pooled. The reductio question proper is read on the 106; on the other set the reductio's capability is labelled as including direct proof. |
| A forced first move puts the model in a state it rarely visits. | "Candidate" in the label; no reading when over half of forced attempts are refused or capped; forced endings beside unforced ones; capability on the **familiar subset**, theorems where the base itself opens with d at least 1 % of the time (if 20 or more); the selection gaps of secondary (2). |
| "Uses anywhere" and "opens with" are different. | Both reported; the primary is named here, before the read; their readings are tabled side by side and disagreements reported, not resolved. |
| Zero of k is not zero. | Levels come from E; U's 0 of k is printed beside E's value; the 1 % floor decides what is read. |
| Many cells (3 doses × 4 directions × 2 sets). | The dose-level agreement rule; per-seed readings are shown but claims are made at dose level. |

## When we will say the split is not informative here

Decided first in the tables, before any reading is reported. The split is **not informative** if:

- (a) more than half of the primary per-seed cells (20 or more applicable theorems) end unresolved, outside the table
  or with no reading; or
- (b) no dose-level reading (one of the four, including 2 of 3) exists for any direction in either set; or
- (c) E fails its check against U in more than a quarter of checkpoint cells with 20 or more applicable theorems.

Then the write-up says so first, reports levels descriptively and does not present the readings that did come out as
findings. Separately, a single reading is uninformative about skill when forced attempts are mostly refused or capped
(no reading), and its size is withheld when S and P̄ × C̄ disagree (above).

## Checks the analysis makes on the result files

Schema and ranges of every row (hard errors); one set of metadata per checkpoint; no duplicate rows; applicability and
`dir_class` the same across arms and checkpoints (hard errors); solved or first above used (warnings, counted in the
tables); E against U per cell (above); theorems dropped for missing rows, per cell. Tests: `test_analyze.py`.

## Post-hoc amendment 1 (after results)

Written 2026-11-10, 7:27 pm PT (2026-11-11 03:27Z), by agent:claude (a subagent of the orchestrator), at Leon's
question "Router or default shift: does propensity rise only where the direction is needed, or everywhere?". It
was written **after every result above had been seen**, so everything it produces is **exploratory**. It changes
no pre-registered text, threshold or table: the frozen tables in `results/tables.md` stand. Script:
`posthoc_anywhere.py` (imports `analyze.py`, does not change it). Output: `results/posthoc_anywhere.{md,json}` and
the figure `results/posthoc_anywhere.{pdf,svg,png}`.

**What was seen that motivated it.** The primary propensity (exact first-move probability, E) is blind to the
reductio here: at the 1 % RL ends only 0 to 0.7 % of attempts open with it, while accepted proofs on the strict
106 use a reductio opened later (`log.md`, last entry). The router secondary above, built on that measure,
returned "no rise to locate" for every dose because the required class's first-move P̄ stayed below 1 %. The same
table's descriptive column "uses anywhere" moved at 1 %: 1.5 → 30.4, 9.8 → 26.9 and 3.5 → 43.0 % of attempts on
the required class, against 3.4 → 3.7, 4.3 → 2.8 and 3.6 → 8.0 % on the idle class. Those numbers were seen
before this amendment was written.

**Measure.** Propensity of d in a class = Σ `used` / Σ `k` over the class's theorems, from U rows (k = 32): the
share of attempts that use d anywhere, as defined in "Definitions" above (`pc_lib.moves`; for the reductio, a box
¬¬X with binder ¬X on any formula X).

**Rules carried over unchanged.** The theorems (applicable with complete rows at base and end, both sets, as in
the frozen router cell), the classes (`dir_class` as in the rows: required 136, helpful 8, idle 194 for the
reductio), the comparison (required against idle; helpful reported between, with no reading), the state rule
(Δ ≥ 0.15 with the interval above 0 is up, the mirror is down, an interval inside ±0.30 is flat, below 1 % at both
checkpoints is rare, n ≥ 20 per class), the bootstrap (2,000 resamples over theorems within class, paired base and
end, the frozen router cell's RNG keys), the verdict (no rise to locate unless required is up; router if idle is
flat, rare or down or the interval of Δ(idle) − ½ Δ(required) lies below 0; default shift if idle is up and that
interval lies above 0; otherwise unresolved) and the dose agreement rule over seeds.

**The one threshold that cannot carry over as written.** The frozen Δ is the change in log10 odds of P̄, a mean
of exact probabilities clamped at 10⁻¹²; a sampled rate, unlike an exact probability, can be 0 of n, so its log
odds can be infinite. Closest equivalent: log10 odds of the pooled rate (x + 0.5) / (n + 1), x and n the class's
total uses and attempts, which is the frozen rule's own treatment of sampled rates ("For F and U rates"). The
numbers 0.15, 0.30 and 1 % are kept: they are in log10 units and were calibrated on a sampled rate (the forced
solve rate) as well as on exact probabilities, and with about 4,350 (required) and 6,200 (idle) attempts per class
and checkpoint the correction moves no rate by more than about 0.01 points.

**Additions, all exploratory.**

1. *Uses carried by accepted proofs, against the rest.* `solved` on a U/<d> row is, as `pc_run.py` writes it,
   the accepted attempts whose proof goes **through** d (`pc_lib.classify`, "through": for the reductio, its DN
   line is on the conclusion's dependency path); `solved` on U/any counts accepted attempts. So uses split exactly
   into (accepted, through d) and (opens d, but d carries no accepted proof: a failed attempt, or a detour inside an
   accepted one). Reported per class; the same rule is also applied to "uses d, excluding the attempts whose
   accepted proof goes through d" (the second part over k minus the first). Where every accepted attempt goes
   through d (checked per theorem and checkpoint, and reported), this is exactly "uses d among attempts not
   accepted": not tied to success mechanically, but conditioned on not succeeding.

   *Correction, 7:30 pm PT (03:30Z), before any reading was written:* the first version of this item said `solved`
   on U/<d> counts "accepted attempts that use d", following SPEC.md's wording. The first run of the script
   warned (a logarithm of a negative number) on rows where that cannot hold (all 32 attempts accepted and all 32 opening a reductio, U/reductio
   `solved` 0), which led to `pc_run.py` line 401 and the definition above. The measure was redefined to what the
   rows hold, and the item rewritten. That first run had already printed the dose verdicts of the carried-over
   rule (frozen classes, all attempts; this correction does not touch that measure): 0 % seeds disagree, 0.1 %
   seeds disagree, 1 % router.
2. *The required class by set.* While preparing this amendment (after seeing the rows), the 30 classical-only
   theorems of the other set turned out to be proved at the RL ends mostly by proofs that do not go through a
   reductio: "required" there means "needs a classical step (DN)", not "needs a reductio". The rule is therefore also run with the
   strict 106 alone against idle, and with the 30 alone against idle (flagged: the label does not hold for them).
   The carried-over reading stays the one on the frozen classes.
3. *Seed-pooled per dose.* Per theorem, the three seeds' counts summed at base and at end, theorems resampled.
   That interval covers theorem sampling only, so the pooled rows are descriptive and carry no verdict; the dose
   verdict is the agreement rule over seeds, as frozen.
4. *Elicitation against capability with "uses anywhere".* The per-seed readings are the frozen secondary (1)
   above, recomputed and checked against `results.json`; the frozen dose agreement rule is applied to them, which
   `tables.md` does not do.
5. *Negation, conditional, cases.* Their `dir_class` is null, so there is no required class and no router reading;
   the optional class's uses are reported only.
6. *Matched compute.* The four matched checkpoints are reported separately, base → matched, and never pooled or
   used in a dose verdict.
