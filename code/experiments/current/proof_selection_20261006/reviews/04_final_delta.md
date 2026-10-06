---
written_on: 2026-10-06
written_by: agent:codex
---

The trained judge's improvements survive review; the title still overstates the scope of “half the stopping gap.”

Reviewed [summary](/data/users/dmitry/ndrl-selection-sprint/experiment-summaries/2026-10-06-proof-selection-sprint/summary.md) (Q), commit `e4421c4`. Source paths below are relative to [S](/data/users/dmitry/ndrl-selection-sprint/code/experiments/current/proof_selection_20261006); A4/B6/B7 mean the corresponding worker reports in `results/`.

1. **High — “Half” combines selection and stopping, and depends on operating point** (Q:10,39–44,249–255). The actual fraction is `(trained-threshold accuracy − matched count/sibling accuracy) / (first-valid oracle accuracy − matched count/sibling accuracy)`. `code/phase7.py` interpolates the count comparator at each generator's realised attempt count, then averages generators; the oracle interpolates or saturates at its 16-attempt-cap ceiling. Thus Q's adjacent 0.915@2.8 and 0.935@3.1 cells do not themselves define the calculation. Recomputed from `B7_results.json`:

   | Set | Dev target → realised attempts | Fraction closed |
   |---|---|---|
   | holdout250 | 3→2.32; 4→3.12; 8→6.45 | 50.1%; 53.4%; 44.3% |
   | textbook72 | 3→7.94; 4→9.14; 8→12.05 | 55.5%; 60.4%; 54.0% |

   At target 2, improvement is effectively zero. Holding the trained selector fixed instead gives **46/48/30%** and **36/40/23%**: the larger headline includes better picking. These are accuracy-gap fractions, not compute savings; ratio uncertainty is unreported. **Fix:** say “trained selection plus threshold stopping closes roughly half the count/sibling-to-oracle accuracy gap at selected budgets,” and show both comparator definitions.

2. **Medium — Selection proximity and transfer need explicit scope** (Q:10,33–37,200–205,313). The s0, k=8 figures are correct: **93.57% / 93.42%** of `(oracle − random expectation)` recovered. But “within 0.01” is false: the textbook residual is **0.01243**, reaching **0.01722** on s1-generated textbook proofs. Against sibling selection, s0 gains are **+0.005 [−0.001,0.012] / +0.015 [−0.001,0.033]**, both unresolved. Transfer is clearest in AUROC; textbook s1 selection improves only **0.00028** over siblings, unresolved (`selection_*_p4.json`). **Fix:** scope “almost as well as Lean” to finite-pool selection accuracy, report residuals, and specify that the reliable cross-generator gain is ranking AUROC.

3. **Medium — Two passages silently change comparison** (Q:200–201,252–253). Second-draw **+0.075 / +0.094** and their intervals compare against **plain s1**, although the sentence starts with the sibling mean; sibling-relative values are **+0.059 [0.035,0.085] / +0.121 [0.058,0.189]** (`phase4_within.json`). The stopping paragraph's **+0.008 [0.001,0.015]** uses holdout target **3**, absent from the table; at its target-4 row the difference is **+0.006 [−0.001,0.013]**, unresolved (`B7_results.json`). **Fix:** name each comparator and target beside its estimate, including the unresolved target-4 comparison.

4. **Medium — The stated training effect reverses an observed direction** (Q:34–35,188–190). Pairwise training optimizes relative separation; it does not ensure valid-proof likelihood rises. A's 07:01 log says valid log-probability **falls from −0.050 to −0.086/−0.095**, while invalid falls further to **−0.307/−0.332**. **Fix:** replace “raises valid and lowers invalid” with “increases valid-versus-invalid relative likelihood.”

5. **Medium — Some categorical claims remain unsupported.** “Scalar heads never beat” ignores the likelihood-input head's textbook **+0.011 [−0.102,0.125]** (`phase2_within_ci.json`); “does not transfer” for valid-only training means unresolved evidence, not established absence. Q:291's draw-independent differences are not established: textbook pairwise-minus-s1 AUROC changes **0.159→0.094**. Q:286 still denies test-label influence on *any comparison*, despite the disclosed adaptive research sequence. **Fix:** use “no resolved advantage,” “direction replicated in two draws,” and distinguish frozen policy parameters from test-informed research questions.

6. **Medium — Neither new executive figure establishes its entire bullet alone.** Cold-reading **A4_trained_judge.png** shows higher conditional AUROC across generators/draws, with theorem counts and uncertainty; it contains no random/oracle selection comparison or scalar-head results, so cannot establish “93%” or “heads never beat.” **Fix:** add a k=8 random/selected/oracle panel and the head comparator, or narrow the bullet. **B7_headline.png** now correctly labels development stars and min–max bands, but does not identify stars' target budgets or display the gap denominator; its oracle curve ends before most threshold points. It also mixes earlier B4/B5 curves with B7 results (`phase7_figs.py`). **Fix:** use consistent B7 comparisons, label targets and exact points, and annotate the matched gap calculation.

7. **Low — Double rounding and remaining provenance.** Q's phase-7 table should round directly from JSON: sibling holdout target-4 attempts **3.84912→3.8**, trained holdout target-8 **6.44648→6.4**, textbook count target-8 **10.7475→10.7**, trained textbook target-8 **12.04685→12.0**. The single K12 overlap and “files read only during the first minutes” still lack direct evidence in the supplied record. **Fix:** generate displayed numbers from full precision and attach provenance or mark those claims unverified.

Previous issues **1–11**: **2,6,8,9,10 addressed**; **7 substantially addressed**. **1 partial** (categorical exposure denial survives); **3 partial** (uncertainty improved, categorical negatives survive); **4 partial** (cost disclosure improved, new oracle-proximity overstatement); **5 partial** (5/6 and intervals supplied, but stage-1 remains called weaker on textbook despite CI crossing zero—say unresolved); **11 unresolved**. Details and fixes are above.

**Executed all five requested checks.** Checked every number and qualifier in the title, executive bullets 3–4, finding 3's phase-4 text/table, and finding 4's phase-7 text/table. Used `HEADLINE_SOURCES.md`, independently parsed all eight phase-4 selection files, `phase4_within.json`, and `B7_results.json`, checked B7 text/table, and spot-verified the map's B6 validity/pass@64 entries. Remaining requested numerical entries agree within stated precision, apart from issues above.

Read both workers' phase 4–7 logs and `research_log.md`. Same-dev early stopping/threshold choice, s0-only training, two judge scoring passes, and uncalibrated-ranker limitations match the record. Material omissions are selection uncertainty, stopping-only gains, and the s1-pool self-initialization caveat; total inference cost remains unmeasured here. Inspected Git history/diffs and both PNGs; used read-only shell/Python arithmetic, not training, neural rescoring, or Lean reruns. Only this `/tmp` review was written.
