---
written_on: 2026-10-06
written_by: agent:codex
---

The selection result survives review; the executive summary overstates validation, mechanisms, and proximity to the oracle (M:14–27; issues below).

Sources: Q = [summary](/data/users/dmitry/ndrl-selection-sprint/experiment-summaries/2026-10-06-proof-selection-sprint/summary.md); S = `/data/users/dmitry/ndrl-selection-sprint/code/experiments/current/proof_selection_20261006`. Paths below are relative to S. A1/A2/A3 denote `results/worker_A_report.md`, `results/worker_A_phase2.md`, `results/worker_A_phase3.md`; B1–B5 denote `results/worker_B_report.md`, then `results/worker_B_phase2.md` through `results/worker_B_phase5.md`. M = `reviews/02_midpoint_audit.md`; D = `reviews/01_design_audit.md`. These aliases plus line numbers identify sources.

Ranked issues and one-line fixes:

1. **High: “Every comparison…fixed” and “No test label influenced any choice” are false broadly** (Q:47–51,230–233). Historical label-derived headroom explicitly informed design (`research_log.md:27–30`). New textbook labels were summarized at **02:07:36**, before the **02:08** freeze; B scored textbook at **02:08** (`logs/worker_B.md:62–69`) and read test results at **02:38**, before A’s **03:28** freeze (M:31). B5’s cost-matched analysis was added *after* its test read (B5:8–10,85); phase-1 comparators also used test costs (M:44). This establishes exposure, not individual-label leakage (M:31). **Fix:** give worker-specific chronology and distinguish frozen policies from exploratory questions and descriptive comparisons.

2. **High: mixed subset draws understate the sibling–oracle gap** (Q:27–28,118,245). On A’s registered subsets, sibling accuracy is **0.92766 / 0.5670833333333333**, with oracle **0.93958 / 0.5942361111111111** (`results/selection_{holdout250,textbook72}_p3.json:1`). The quoted textbook **0.573** comes from B’s different draw, whose oracle is **0.598**, not **0.594** (B3:148–151). Across generators, textbook oracle gaps include **0.645−0.607**, exceeding “0.01–0.02.” Q:47–48 also obscures that the initial headline uses generator s0. **Fix:** use one evaluation draw throughout and label generator versus head seeds.

3. **High: scarcity/undertraining and causal explanations are overclaimed** (Q:33–37,149–167). A2 says undertraining is “unlikely” (34–37); its ladder changes theorem mix, uses one seed below the three-seed top rung, and has uncertain slope (131–139). Textbook head-minus-likelihood AUROC is **−0.049 [−0.196,+0.089]**, unresolved (A2:20–21). “Learns which theorems are hard” is an interpretation (M:25–27). “Level,” “no self penalty,” and “weakness is not inherited” overread **+0.021 [−0.009,+0.051] / −0.029 [−0.094,+0.045]** and inconsistent diagonals (A3:21–40). **Fix:** report point estimates, uncertainty, and untested explanations; retain scarcity and optimization as possibilities.

4. **High: “matches,” “nearly solved,” and “free” exaggerate practical readiness** (Q:39–44,175–176,245). Count-stop achieves **0.924 at 3.92** versus oracle **0.927 at 4**, and **0.587 at 7.60** versus **0.607 at 8** (B4:59–67). These compare variable allocation with fixed-k sampling, without an equivalence test. Each sibling requires another forward pass (B2:125–126). **Fix:** say “near these oracle point estimates at similar mean attempt counts”; report total inference cost and “no additional training.”

5. **Medium: self-judging and ancestry certainty exceed the intervals** (Q:28–31,140–143). The diagonal is lowest in **5/6**, not every row; the exception is **0.788 self versus 0.786 s2** (B3:16–19). R8-minus-r16 is **+0.005 [−0.022,0.030] / −0.008 [−0.095,0.073]**; “exactly as badly (within ±0.01)” mistakes point estimates for equivalence. Stage-1’s textbook deficit is **−0.082 [−0.181,0.008]**, not an established −0.10 effect (B4:135–143). **Fix:** state unresolved differences for the single tested lineage; leave causal training chronology open.

6. **Medium: the mechanism language changes descriptive evidence into localization** (Q:30–31,134–138). The “two to three times” concerns the *self-minus-sibling log-likelihood excess*, **+0.033–0.040 valid / +0.071–0.107 invalid nats/token**, not proof probabilities (B4:25–27). Step types pool whole-proof labels and have no intervals (B4:157–158), so they do not identify the erroneous step. “No dependence on proof length” exceeds “no consistent concentration in long proofs” (B4:113–125). **Fix:** name the measured excess, describe the pattern, and remove universal “Any sampler” and causal “exactly where.”

7. **Medium: additional numerical/superlative corrections.** Q:122’s SD bounds **≤0.01 / ≤0.03** should allow **0.014 / 0.033** (B3:150–151). Q:149–150’s “highest pooled/lowest within” fails: history-head pooled AUROC is **0.934/0.969** (A1:90–91), and shortest-proof within AUROC is **0.642/0.537** (B1:257,265). Q:247’s sibling pooled range rounds to **0.77–0.79**, from **0.7701169948942378 / 0.790317392041085** (`results/B2_auroc_{textbook72,holdout250}__s0_r16__k64.json:2023`). Q:130’s strict ±0.01 bound includes **+0.0103** (`results/B3_results.txt:64`). “Best or tied-best” also lacks an equivalence test: sibling versus all-three AUROC reaches **0.856 versus 0.866** (B3:60). **Fix:** correct these values and remove ranking superlatives.

8. **Medium: “closes a fifth” needs its denominator and selection scope** (Q:43,194–199). The quoted low-budget sibling points close approximately **18–23%**, not a directly sourced 15% endpoint: gap fractions include **0.1758165096660944** and **0.23425314720891693** (`results/B5_matched.json:65,186`). Higher-budget gains weaken: T-sib holdout m=8 is **+0.0027 [−0.0001,+0.0054]** (`results/B5_matched.txt:4`). The textbook “two of three intervals include 0” is accurate (B5:48–50). **Fix:** specify selected operating points, descriptive interpolation, and the count-to-first-valid accuracy denominator.

9. **Medium: Limitations needs material additions.** Include the ladder confounds above; s0-only head training (A3:113–114) and stopping-development data (B5:84); dev-fitted allocation reused with a mismatched sibling selector (B2:271–273); pre-freeze holdout proof-multiplicity inspection (B2:278–280); and pointwise intervals across many comparisons (A1:168–170; B2:277); and unchecked equivalence of offline Lean judging to the original gate (A1:173–174). “Unclipped” history still clips **0.3–2.0%** (A1:125–127); the initial “up to 64” excludes the later **96**-attempt pool (A2:67). **Fix:** disclose these boundaries and scope initial-pool counts explicitly.

10. **Medium: next steps outrun the evidence** (Q:246–253). AUROC is discrimination, not a calibration test; scalar-head signal loss is explicitly an untested hypothesis (A3:38–40). Lower error correlation of sibling rewards and RL benefit were not demonstrated (B3:110–113; Q:239–240). **Fix:** propose fresh-test, compute-matched experiments comparing calibration, per-token judges, and RL rewards as competing hypotheses.

11. **Low: remaining unsupported provenance.** The claimed single K12 overlap and “files were not read after the first minute” remain **unverified** in supplied evidence (Q:86,217–218; A3:124; `research_log.md:18–23`). **Fix:** attach direct provenance or mark unverified.

Cold-reader assessment, covering every executive component (Q:14–51):

| Component | Understandable alone? Missing definitions |
|---|---|
| Opening | Mostly; holdout250’s provenance and “held out” from which training process are missing. |
| Claim 1 | Mostly; define finished versus Lean-valid and likelihood’s masked per-token score; “knows” overstates ranking. |
| Claim 2 | Partly; sibling is explained, but self penalty, blind spot, round 8, and stage-1 need glosses. |
| Claim 3 | No; scalar head, AUROC/chance level, pooled versus within-theorem populations, hard negatives, and ranker are undefined. |
| Claim 4 | Partly; explain c, pass@4, absolute confidence, development tuning, difficulty-aware allocation, and which gap is “a fifth.” |
| Validation paragraph | No; explain sha-frozen, paired theorem bootstrap, phases, and generator versus sampling seeds. |

Fix: define these terms at first use and state the theorem counts and conditional AUROC population.

The four executive figures, in order:

| Figure | Does it establish its bullet? Fix |
|---|---|
| A_selection_vs_m | Partly. Random curve uses **0.841760/0.398958333** at k=8, not text’s expectation **0.835737500/0.405451389** (M:17,23; `code/make_figures.py:12,34`). Plot expectation; restrict “~75%” to k=8. |
| B3_judge_matrix | Supports 5/6, contradicts “every”; self-selection **0.917/0.543** differs from text **0.918/0.540**. Shows neither oracle proximity nor mechanism/ancestors (B3:58–74). Align estimates and shorten bullet. |
| A2_auroc_vs_data | Shows the tested ladder, not pooled AUROC or convergence; lacks uncertainty (A2:47–61,133–139). Add intervals and narrow its title. |
| B5_headline | Shows separation from first-valid; purple is a **test threshold sweep**, not frozen operating points (B5:79–85). Mark dev-selected points and the gap calculation; keep min–max bands distinct from confidence intervals. |

Editorial fixes: replace “already knows,” “cash it in,” “useless,” and “nearly solved” with measured behavior (Q:16,21,39,56). Replace negative-form headlines with “Likelihood outperforms this head recipe” and “Stopping retains substantial headroom” (Q:33,169,178). Split the mechanism-heavy second executive bullet; move withdrawn-overlap history out of the long setup paragraph (Q:26–31,80–88).

Commands executed: `pwd`; `rg --files` and `rg -n`; `cat`, `nl -ba`, `sed -n`, `wc -l`; read-only `git --no-optional-locks log --follow -p`, `diff`, `diff --cached`, `diff --numstat`; `python3 -B` for JSON parsing, exact-value extraction, SD/gap arithmetic and SHA256; `view_image` on all four figures. `python` was unavailable; some probed paths were missing. No training, raw rescoring, or Lean reruns. Only this review file was written.

