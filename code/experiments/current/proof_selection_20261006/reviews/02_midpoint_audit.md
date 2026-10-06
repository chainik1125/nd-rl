---
written_on: 2026-10-06
written_by: agent:codex
---

The selection numbers reproduce; the causal explanation and some freeze/stopping language exceed the evidence. Audited phase 1 on `dmitry/selection-sprint`, HEAD `6bd08f4`, including current uncommitted changes. No sprint or run files were modified.

**Executed:** independent raw-file selection, AUROC, and stopping calculations; hash verification; permutation-artifact comparisons; label-denied baseline/shallow inference; verifier input loading; all-test environment replay; masking/EOS probes; scorer-path `rg`; Git history/diff inspection. Read all requested plans, reports, tables, logs, and code. **Not executed:** generation, training, Lean rejudging, full neural rescoring/permutation, cached-decoder numerical validation, or fresh-seed replication. Scripts and full outputs accompany this review: [selection](recompute.py), [stopping](stopping_audit.py), [replay](replay_audit.py), [isolation](isolation_audit.py).

The independent k=8 calculation uses the registered 200 shared subsets, lowest-index ties, equal theorem weights, and includes unsolvable theorems:

| Metric | holdout250 | textbook72 |
|---|---:|---:|
| completion_loglik | 0.917520000 | 0.539791667 |
| verifier_ens | 0.905960000 | 0.514375000 |
| completion | 0.895660000 | 0.519444444 |
| Random expectation over sampled subsets | 0.835737500 | 0.405451389 |
| Oracle on those subsets | 0.939580000 | 0.594236111 |
| Gap recovered by completion_loglik | 78.756290% | 71.160566% |

Recomputed completion_loglik − verifier_ens: **0.011560 [0.0011395, 0.0219625]** and **0.025416667 [0.010128472, 0.043473958]**, respectively, using the registered paired theorem bootstrap.

The saved `random` noise scorer is different: 0.841760 / 0.398958333. True marginal random accuracy is 0.835937500 / 0.406901042. Analytically averaging *all* k=8 subsets gives likelihood 0.917305718 / 0.542083035, verifier 0.906687906 / 0.517388605, oracle 0.939467498 / 0.596806836. Thus the reported figures are reproducible Monte Carlo estimates; their small sampling error does not explain the ranking.

1. **HIGH — “The verifier learned difficulty, not correctness” is an interpretation, not an identified mechanism.** Recomputed pooled AUROC is 0.921309 / 0.951879 for verifier_ens; finished-only pooled AUROC is 0.724336 / 0.714580. Its within-theorem finished AUROC is **0.646038 / 0.596169**, versus likelihood **0.787945 / 0.687197**. Both implementations correctly give ties half credit and average AUROC equally over only the **116/250 and 34/72** theorems containing both valid and invalid finished attempts. They do not pool within-theorem pairs; homogeneous theorems are excluded. These are conditional discrimination metrics, not population-wide recognition accuracy.

   The reversal is consistent with between-theorem cues, but also with inadequate hard-negative learning, input loss, or optimization/selection mismatch. I counted **86,217/97,776 valid training attempts (88.18%, not the research log’s 94%)**, just **3,195 finished-invalid examples**, and **1,227 mixed finished-training theorems**. BCE balances overall classes, not finished-invalid examples against easy unfinished negatives. History inputs reduce clipping but change representation; hist_s0’s best checkpoint is its final capped step. One ranking run cannot eliminate undertraining. Clipping-stratified *pooled* AUROC does not establish its causal irrelevance to selection. The saved paired AUROC interval on textbook72 is −0.194677 to +0.006520: the within-theorem advantage is not resolved there. No causal discrimination among these explanations was executed.

2. **HIGH — Hash integrity passes; the broad chronology claim does not.** All **63/63** frozen score/checkpoint hashes match disk; all **12/12** manifest pool/label/split/generator hashes match. The intrinsic grid and dev-choice hashes also match their logged prefixes. Selection JSON timestamps follow the 03:28:44 freeze: 03:28:51 and 03:29:03.

   However, `collect_textbook72.log` records pool writing at **02:07:32** and aggregate label results at **02:07:36**, contradicting the 02:08 statement “before any new test attempt is generated.” Worker A’s 02:09 history-format decision explicitly uses **textbook72 clipping rates**, so calling it dev-motivated is inaccurate. Worker B logs test evaluation at **02:38**, before A’s 03:28 global-sounding freeze; B’s report says its second freeze was 03:15 while its log says 03:23. Git first commits the 02:08 freeze text at 03:35. These observations require narrower, worker-specific exposure claims; current hashes cannot prove files were unread earlier. They do not themselves establish label leakage into the primary selector.

3. **MEDIUM — Stopping arithmetic passes, but the comparator protocol changed.** Independently reconstructed both primary dev quantile grids and checked dev cap/threshold choices. `stopping_headline.py` chooses nearest thresholds within each cap, then can prefer higher dev accuracy within a 25% budget tolerance. Shared 50 permutations, earliest-consumed ties, first-attempt fallback, and incumbent selection are implemented consistently. Raw-file simulation reproduces saved per-theorem arrays within 9×10⁻¹⁵.

   At the dev m=4 likelihood choice, N=16 and τ=−0.02083669795679931:

   | Test | Attempts | Accuracy | Fixed-k at realised cost | First-finished ceiling: accuracy / attempts |
   |---|---:|---:|---:|---:|
   | holdout250 | 3.925280 | 0.918880 | 0.907269837 | 0.910800 / 2.620560 |
   | textbook72 | 12.076666667 | 0.563333333 | 0.558225926 | 0.548611111 / 8.809444444 |

   Recomputed gains over fixed-k: **0.011610163 [0.003660652, 0.020854632]** and **0.005107407 [−0.003097604, 0.014217739]**. Over FF: **0.008080 [−0.001840, 0.017520]** and **0.014722222 [−0.026701389, 0.050569444]**.

   Contrary to the 02:08 freeze, headline fixed-k mixture weights use *realised test cost*; FF’s nearest cap also uses test cost. These are descriptive comparisons, not dev-frozen comparator policies. Adjacent-k mixtures can undersell fixed-k frontiers, although my stronger all-pairs mixture check leaves holdout250’s m=4 comparator unchanged and improves textbook72’s by only 0.000278. The adaptive test-optimized frontier is additionally optimistic and cannot confirm H2. FF cannot spend the policy’s budget beyond its ceiling. Non-significance establishes no resolved benefit, not equivalence. Order averaging and bootstraps remain conditional on one pool draw.

4. **MEDIUM — Narrow the stopping conclusion.** “Fails to transfer” is supported for **budget calibration**: dev 3.986211 attempts becomes 12.076667 on textbook72. It does not establish zero accuracy benefit or failure of all stopping rules. Worker B’s “no dev-chosen intrinsic policy separates from either comparator” contradicts its table: textbook72 completion_minstep beats fixed-k by **+0.023 [0.011, 0.038]**, and completion_loglik_sum by **+0.016 [0.005, 0.031]**; neither beats FF conclusively. These secondary numbers were read, not independently resimulated. The reported post-hoc count-stop c=3 also beats FF on holdout250, **+0.011 [0.002, 0.020]** at 4.03 attempts. Qualify the negative headline by policy, budget, and exploratory status.

5. **MEDIUM — Isolation evidence is narrower than “all scorers passed permutation.”** Scorer-path inspection found no inference-label dependency: likelihood reads pools; verifier input construction reads trajectories; neural inference calls `load_items(..., with_labels=False)`; supervised labels enter training/dev evaluation. I reran four baselines and both shallow scorers on permuted dev labels with label-file opens forbidden: **6/6 byte-identical**, and loaded **17,728** verifier inputs without label access. The saved verifier permutation artifact used a **smoke checkpoint**, per A’s log; it differs from final verifier_s0 scores and does not validate the final ensemble. Copied likelihood inputs in the shallow permutation directory are not a likelihood-rescoring test. Full neural permutation remains unexecuted.

6. **LOW — Masking/completion checks largely pass, with bounded caveats.** Independently replayed **20,608 test attempts / 336,160 steps**: zero state-hash, done, step-count, or finished-ND mismatches. Mask positions match actual `have`, box-binder, and first-Or.elim overrides on every finished attempt. Raw-prefix conditioning and citation scores are retained; inserted closures/second binders are not scored as emissions. EOS probes confirm inclusion of real EOS and omission on truncation; all **13** truncated test actions lack EOS. On **120 failed/truncated steps**, heuristic masking includes positions the environment never reaches; this cannot affect completion_loglik’s finished ranking. T=1 masked mean likelihood is a diagnostic, not a marginal proof probability.

   Across **129,456 attempts**, ND presence equals done exactly, including **5,195 finished-invalid** attempts; no extra validity bit leaks through presence. Completion is pool-derived before judging, but includes symbolic scope/goal/binder checks. Its contribution must remain explicit. The likelihood increment beyond completion is strong: **+0.021860 [0.012260, 0.0312415] / +0.020347222 [0.004578125, 0.040138889]**, independently recomputed. That is firmer evidence than the proposed learning mechanism. Finally, token-cost claims need correction: `scorer_cost()` charges a three-head ensemble only one forward pass and omits several derived scorers’ costs; attempt-axis results are unaffected.
