---
written_on: 2026-11-22
written_by: agent:claude (executor, run olmo-ifcurve; reviewed 2026-11-22, corrections marked by the librarian)
---

> **Reading copy (librarian, 2026-11-22).** This is the plain-language summary of run `olmo-ifcurve`, written by the
> executor. The source is fork `dan-pandori/nd-takehome`, branch `dan_olmo-ifcurve`, file
> `olmo_ifcurve/plain_summary.md` at
> [`0498d0cf`](https://github.com/dan-pandori/nd-takehome/blob/0498d0cf/olmo_ifcurve/plain_summary.md). The
> corrections from its [review](https://github.com/dan-pandori/nd-takehome/blob/8e744a89/review_olmo-ifcurve.md)
> are applied, and each change is marked *[review]*. The letters are the review's §Verdict "Reword" items; two
> notes come from its §Recount and §Compare tables:
> - (a) the prediction for how many hard prompts the final model solves genuinely is "not established", not failed;
> - (c) three of the 48 count as solved only under a labelling rule;
> - (d) when the estimated genuine share stops rising is not established;
> - (g) the starting model is named;
> - (§Compare row 9) one count depends on whose labels are used;
> - (§Recount 8, §Compare rows 10–11, §Verdict) the review's own gaming shares, and the 16 early checker passes that
>   the growth on the hard prompts rests on.
>
> The review's verdict is "Stands". See the [summary](README.md) for the full list.

# olmo-ifcurve in plain language

**What we asked.** Reinforcement learning (RL) trained Olmo 3.1 Think 32B for 2,300 steps, and AI2 published
snapshots along the way. Earlier runs looked at only a few of them (the start, steps 100 and 750, and the end). This run
follows instruction following (answering while obeying checkable rules) across the whole run.

**What we did.** Nine snapshots (steps 0 to 2,300; *[review]* step 0 is the RL run's starting model,
`allenai/Olmo-3-32B-Think-DPO`, and the others are checkpoints of `allenai/Olmo-3.1-32B-Think`) answered 218 IFBench prompts several times each; the official
checker scored every answer. We also followed 48 "reach" prompts that the
starting model never solved genuinely, 32 tries per snapshot. Checkers can be fooled, so blinded readers (Claude
sub-agents) labelled 1,782 checker passes "genuine" or "gamed" without knowing the snapshot.

**What we found.**
- **The checker score rises at every snapshot** (within noise early on). The share passed on one try (pass@1) goes from 0.29 to 0.68 (an
  estimated 0.25 to 0.50 genuinely). After step 1,100 the share passed within eight tries
  shows no detectable gain while pass@1 rises ("sharpening"), and the estimated genuine share stops rising too.
  *[review]* That last part holds only loosely. From step 750 on, the estimated genuine share is 0.41–0.51,
  depending on whose labels are used (it rises from 750 to 1,100 on the executor's labels and dips on the review's),
  so when it stops rising is not established.
- **Every hard prompt gets a genuine solution at some snapshot**, first between steps 100 and 2,300 in its 32 regular
  tries. That is partly built in: the prompts were chosen because the final model solved them. *[review]* For three
  of them only under a labelling rule: two prompts accept a single fixed answer, which a rule counts as genuine, and
  one is solved only by answers its checker reads as a single sentence, genuine only under one reading of another
  rule. At the end, the final
  model shows a genuine success on 40 of 48; on 4 others all its checker passes were gamed or absent. *[review]* The
  40 counts only the checker passes that were read. The review's own full read (its labels, rules 1–12) gives 44 of
  48 (41 with at least two), exactly at the advance prediction, and the count depends on the labelling rules. Most became frequent within one or two snapshots of the first rare
  success; none jumped straight from nothing to frequent. On 18 prompts a rare success already showed one snapshot earlier (*[review]* 17 with the review's labels).
- **Easy-to-coax prompts tend to come first.** Prompts that a hint unlocks in the starting model were first solved
  around step 500. Those needing fine-tuning came around step 750, and the seven that resisted both around 1,100. That
  is the order "habit before skill" predicts. But the advance test put the gap's lower bound exactly at zero, so we
  report "no clear order"; the same test on the second genuine success passes.
- **Gaming does not fade.** On a random sample of checker passes the gamed share stays between 0.05 and 0.28 per
  snapshot (flat by our advance test). On the hard prompts it grows: late snapshots often stuff keywords or use
  formatting tricks. *[review]* With the review's own readers, on half of the same sample, 0.05–0.30 per snapshot,
  also flat. The growth on the hard prompts rests on only 16 early checker passes.
- **Answers get much shorter.** Most of the shortening (261 to 93 words) happened by step 750, alongside the hard
  prompts' progress. Answers settled near 50 words only at step 1,600, so by our advance test shortening did not come
  first.

**How sure.** One training run. Two labelling rules (keyword stuffing) were added during the run. Several advance predictions failed and are reported as misses: a clear-cut class order, the share of
gradual prompts, early gaming on hard prompts, how many hard prompts the final model solves (40 of 48 established), and
how often answers hit the length cap. *[review]* The final model's count is not a miss but "not established":
the review's full read of its checker passes gives 44 of 48, exactly at the predicted threshold.

**Cost.** \$154.67 of the \$160 GPU budget (\$34.84 lost to idle machines a broken tool failed to stop).
