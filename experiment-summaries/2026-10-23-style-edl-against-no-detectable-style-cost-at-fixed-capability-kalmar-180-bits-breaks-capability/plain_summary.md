# style-edl in plain language (2026-10-23)

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
2. **Their EDL is essentially zero**, as for the learner's own proofs, and the learner still proves every held-out
   theorem.
3. **One apparent effect was an artefact.** By the brief's formula, the shortest proofs cost about 1 bit per proof.
   That came from which theorems landed in the training half: it held more of the theorems where the shortest proof
   differs. Swapping the halves shrinks the cost to 0.1–0.3 bits.
4. **EDL charges only for style the learner can pick up.** The shortest proofs follow a pattern (avoid detours) that
   the learner picks up quickly (6–13 bits of surprise drop to about 1), paying roughly 1–4 bits of EDL each. The RL
   model's and the other learner's differences look random: never learned, they stay in the final surprise.
5. **A really different style costs a lot, but it also breaks the learner.** Kalmár proofs are about 25 times longer
   and at first surprise it by about 1,300 bits each. It learns most of the style from 128 examples (down to about 90
   bits), an EDL of about 180 bits per proof. But then it proves only 25–38 % of the held-out theorems (100 % before):
   it writes Kalmár-like proofs and loses track of the bookkeeping.
6. **Counting any correct proof as a hit**, instead of the exact text, the Kalmár stream's EDL turns negative: training
   on it made correct proofs less likely overall. The shortest-proof stream shows a small cost (0.1–0.2 bits), from a
   dip in correct proofs midway through learning.

**What it means.** For this base model and its RL version, on theorems both prove, style alone creates no EDL gap: the
RL proofs are mostly identical to the base's, and the rest are not a learnable style. EDL charges for style only when
it is systematic and quite different, and the one such style tested could not be learned without losing ability. So
"EDL is inflated by style at fixed capability" is not supported here; harder theorems, where correct proofs differ
more, could change that.

**What it cost.** $1.86 of a $10 budget: about 7 hours of rented GPU time.
