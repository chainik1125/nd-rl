# olmo-ifcurve in plain language

**What we asked.** Reinforcement learning (RL) trained Olmo 3.1 Think 32B for 2,300 steps, and AI2 published
snapshots along the way. Earlier runs looked at only a few of them (the start, steps 100 and 750, and the end). This run
follows instruction following (answering while obeying checkable rules) across the whole run.

**What we did.** Nine snapshots (steps 0 to 2,300) answered 218 IFBench prompts several times each; the official
checker scored every answer. We also followed 48 "reach" prompts that the
starting model never solved genuinely, 32 tries per snapshot. Checkers can be fooled, so blinded readers (Claude
sub-agents) labelled 1,782 checker passes "genuine" or "gamed" without knowing the snapshot.

**What we found.**
- **The checker score rises at every snapshot** (within noise early on). The share passed on one try (pass@1) goes from 0.29 to 0.68 (an
  estimated 0.25 to 0.50 genuinely). After step 1,100 the share passed within eight tries
  shows no detectable gain while pass@1 rises ("sharpening"), and the estimated genuine share stops rising too.
- **Every hard prompt gets a genuine solution at some snapshot**, first between steps 100 and 2,300 in its 32 regular
  tries. That is partly built in: the prompts were chosen because the final model solved them. At the end, the final
  model shows a genuine success on 40 of 48; on 4 others all its checker passes were gamed or absent. Most became frequent within one or two snapshots of the first rare
  success; none jumped straight from nothing to frequent. On 18 prompts a rare success already showed one snapshot earlier.
- **Easy-to-coax prompts tend to come first.** Prompts that a hint unlocks in the starting model were first solved
  around step 500. Those needing fine-tuning came around step 750, and the seven that resisted both around 1,100. That
  is the order "habit before skill" predicts. But the advance test put the gap's lower bound exactly at zero, so we
  report "no clear order"; the same test on the second genuine success passes.
- **Gaming does not fade.** On a random sample of checker passes the gamed share stays between 0.05 and 0.28 per
  snapshot (flat by our advance test). On the hard prompts it grows: late snapshots often stuff keywords or use
  formatting tricks.
- **Answers get much shorter.** Most of the shortening (261 to 93 words) happened by step 750, alongside the hard
  prompts' progress. Answers settled near 50 words only at step 1,600, so by our advance test shortening did not come
  first.

**How sure.** One training run. Two labelling rules (keyword stuffing) were added during the run. Several advance predictions failed and are reported as misses: a clear-cut class order, the share of
gradual prompts, early gaming on hard prompts, how many hard prompts the final model solves (40 of 48 established), and
how often answers hit the length cap.

**Cost.** \$154.67 of the \$160 GPU budget (\$34.84 lost to idle machines a broken tool failed to stop).
