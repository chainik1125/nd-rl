---
written_on: 2026-10-23
written_by: agent:claude
---

> **Reading copy (librarian, 2026-10-23).** This is the plain-language summary of run `style-edl` (fork
> `dan-pandori/nd-takehome`, branch `dan_style-edl`, DONE
> [`ae20620a`](https://github.com/dan-pandori/nd-takehome/commit/ae20620a), file `run_style_edl_plain.md`), written by
> the executor, with the corrections from its
> [review](https://github.com/dan-pandori/nd-takehome/blob/e378b944/review_style-edl.md) applied. Each change is marked
> *[review]*. The numbers in parentheses are the review's claim numbers; "Must be reworded" and "Not supported, or
> missing" are the lists in its verdict.
> - The shortest proofs' apparent cost comes *mostly*, not wholly, from the split. Swapping the halves leaves 0.06–0.33
>   bits, not 0.1–0.3, and other corrections leave up to about 1 bit per proof (23–24, "Must be reworded" 1).
> - The alternative styles' split-corrected EDL is "none above S_own2 + F", in the review's words, not "essentially
>   zero"; one held-out theorem is missed in one cell (21–22 and 28, "Must be reworded" 2 and 6).
> - "EDL charges only for style the learner can pick up" is a hypothesis from exploratory numbers. It rests on 14–39
>   differing proofs per cell and is confounded by the learning rate; the "1–4 bits" per shortest proof is 0.6–4 in a
>   count over both halves (25, "Must be reworded" 3).
> - The Kalmár number depends on the limit on training passes, which every Kalmár fit at the chosen learning rate
>   reached (26, "Not supported, or missing" 1).
> - The style-blind score counts only the correct proofs the run collected, so its dip may be valid proofs outside the
>   collection (27, "Must be reworded" 4).
> - "No EDL gap" becomes "no detectable gap", for small style differences only. The general rule about systematic
>   styles is dropped; what remains says that Kalmár's cost mixes style with capability (29, "Must be reworded" 6,
>   "Not supported, or missing" 2).
>
> The corrections take it from 594 to about 850 words, against the brief's 600. The executor's text, byte for byte, is
> this file at commit `edd07b7`. See the [summary](README.md) for the reviewed numbers.

# style-edl in plain language (2026-10-23; with the review's corrections) *[review]*

**What was asked.** *Excess description length* (EDL) asks how much a learner must absorb from examples: it is the
learner's surprise at each new batch of proofs, beyond the surprise left at the end. EDL scores exact proof text, so a
learner that can already prove a theorem might pay EDL just because the proofs it sees are in another *style*. Then an
EDL gap between a base and a reinforcement-learning (RL) model could be style, not skill.

**What was done.**
- Three small proof models (9.6 million parameters each) as learners; 256 short logic theorems each proves in at
  least 29 of 32 tries.
- Six "streams" of correct proofs of these theorems, all checked by the Lean proof checker:
  1. the learner's own most likely proofs;
  2. a random sample of its own proofs;
  3. another learner's proofs;
  4. the shortest proofs a search finds;
  5. the proofs of an RL-trained version of the learner;
  6. proofs built by a fixed textbook procedure (Kalmár's construction), unlike anything the learner writes.
- Each learner was fine-tuned on 128 proofs of a stream in growing batches; the other 128 theorems tested whether it
  could still prove them.

**What came out.**
1. **On these theorems, styles that keep the learner's ability barely differ from its own.** The other learner, the RL
   model and the shortest-proof search write exactly the learner's proof for 85–94 % of the theorems.
2. **Their EDL, once corrected for which theorems landed in the training half, is not detectably above that of the
   learner's own resampled proofs.** *[review]* None exceeds it by more than the 0.1-bit floor, and the nine
   stream-and-seed cells span −0.40 to +0.06 bits per proof around it *[review]*.
   After training on them, the learner still proves every held-out theorem but one, missed in one of the nine cells
   *[review]*.
3. **One apparent effect was mostly an artefact.** *[review]* By the brief's formula, the shortest proofs cost about 1
   bit per proof. Most of that *[review]* came from which theorems landed in the training half: it held more of the
   theorems where the shortest proof differs. Swapping the halves (an exploratory check) shrinks the cost to 0.06–0.33
   bits *[review]*. Three reasonable corrections for the split leave anywhere from below zero to about 1 bit per proof,
   so a cost of a few tenths of a bit cannot be told apart from zero *[review]*.
4. **A hypothesis from exploratory numbers: EDL may charge only for style the learner can pick up.** *[review]* The
   shortest proofs follow a pattern (avoid detours) that the learner picks up quickly (6–13 bits of surprise drop to
   about 1), paying roughly 0.6–4 bits of EDL each, counted over both halves *[review]*. The RL model's and the other
   learner's differences look random: never learned, they stay in the final surprise. This rests on only 14–39
   differing proofs per stream and seed. The shortest proofs were also trained at a learning rate ten times that of
   the other short streams, and part of their cost (0.04–0.09 bits per proof) falls on proofs identical to the
   learner's own, so that part is not style *[review]*.
5. **A really different style costs a lot, but it also breaks the learner.** Kalmár proofs are about 25 times longer
   and at first surprise it by about 1,300 bits each. It learns most of the style from 128 examples (down to about 90
   bits), an EDL of about 180 bits per proof. Every Kalmár fit at the chosen learning rate stopped at the limit of 8
   passes over its data, so this number depends on that limit *[review]*. But then it proves only 25–38 % of the
   held-out theorems (100 % before): it writes Kalmár-like proofs and loses track of the bookkeeping.
6. **Counting any correct proof the run collected as a hit** *[review]*, instead of the exact text, the Kalmár stream's
   EDL turns negative: training on it made correct proofs less likely overall. The shortest-proof stream shows a small
   cost (0.1–0.2 bits), from a dip in the collected correct proofs midway through learning *[review]*. The models
   midway may have moved to correct proofs outside the collection; that was not checked *[review]*.

**What it means.** For this base model and its RL version, on theorems both prove, style alone creates no *detectable*
EDL gap *[review]*: the RL proofs are mostly identical to the base's, and the rest do not look like a learnable style (an
exploratory reading) *[review]*. The styles that keep the learner's ability differ from its own proofs on only 6–15 %
of the theorems, so a gap of a few tenths of a bit per proof would not have been detected; this is not evidence that
EDL ignores style *[review]*. The one systematic, very different style tested, Kalmár's, could not be learned without
losing ability, so its cost mixes style with capability *[review]*. So "EDL is inflated by style at fixed capability"
is not supported here; harder theorems, where correct proofs differ more, could change that.

**What it cost.** $1.86 of a $10 budget: about 7 hours of rented GPU time.
