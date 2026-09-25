**It is a composition failure, not a prompt artefact. From L10 upward it shows first as a refusal to write a proof long enough.**

1. **The parts are solvable; the joined theorem is not.**
   - For the RL arms (one seed each), the two parts alone have a mean pass@1 of 54-56% when the joined length is 6-9, and 41-48% when it is 10-14.
   - If the joined proof were just the two part proofs written one after the other, with independent errors, joined pass@1 would be about the product of the two part rates. That product is 28-31% at joined L6-9 and 16-23% at L10-14.
   - The joined theorem is solved at 4.3% (A), 5.9% (B) and 6.0% (C) at L6-9. That is 15-21% of the product (bootstrap intervals 11-27%). At L10-18 it is 0.00% for every arm, even though the product at L10-14 is 16.1% (A), 23.3% (B) and 22.7% (C).
   - It collapses already at L8. That is inside B's and C's training range. C scores about 19% on the standard L8 set (main report, mean of 3 seeds), but here it gets 0.8% against a product of 26.9%.
   - Pretrain is worse still: 0.3% against a product of 14.2%.
2. **It is not the prompt format.** Changing the prompt, on the same theorems, leaves pass@1 unchanged within about 1 percentage point. Every 95% interval includes zero:
   - (a) premises grouped by part;
   - (b) conjuncts in the order the parts' premises appear;
   - (c) the original atom letters.

   The renamed parts and the original, un-renamed parts are solved at the same rate, within 3 points of the product.
3. **The model will not write a joined proof long enough.**
   - Joined attempts stop at a median of 7 written lines for A at every joined length from 7 to 18, and at 7-9 for B and C (8-9 from L10 on), whatever the required length.
   - As a result, 35-55% of joined attempts are as long as L_true at L6-9, 0.4-5.2% at L10-14 and 0.0% at L15-18.
   - The same checkpoints write the parts alone to their full length: 98-99% of part attempts are long enough for parts of joined L6-9 theorems.
4. **Even when the attempt is long enough, joining fails.**
   - At joined L6-9, attempts that reach L_true are correct 10-13% of the time, against the 28-31% product.
   - The failing line is usually a shortcut: writing a conjunct or the goal with an invalid step, or citing the other part's premises. Across all seeds at L6-9, these account for 29-33%, 17-22% and 22-25% of failures respectively.
   - About three quarters of failures happen before either conjunct has been validly derived. The other part's premises and the conjunction goal disrupt even the first sub-proof.
5. **Premise count makes everything harder but does not explain the gap.**
   - Correction to the brief: the cap-6 pretraining data has 0-3 premises, not 0-1 (41% have 2, 4% have 3). It never has 4 or more, whereas 57% of these joined theorems do.
   - Joined and part pass rates both fall as premises grow.
   - Joined theorems with 3 premises, which are common in RL training, still run at about a quarter of their product: 8-10% against 35-40% at L6-9.
6. **A likely reason.** "Prove A & B from two independent sets of premises" is almost absent from training. Only 0.3% of the RL targets are separable conjunctions (`ab3/data_build/FEASIBILITY.md`). RL on lengths 6-11 hardly changes the joined result: at L6-9, C reaches 6.0% against A's 4.3%, while its parts improve.
7. **Update, arm D (section 6).** Training on constructed composition targets, arm D, largely fixes joining at short
   lengths. At joined L6-9, D reaches 0.80 of the part product, where C reaches 0.14. It does not fix joining at L10 and
   above: the ratio is 0.04 at L10-14. D only rarely solved long constructed training targets, so RL had little to learn
   from there.

**Caveats.**
- Sections 1-4 use one seed per arm (s101). The main evaluation's seed spread was small.
- The product assumes independent errors. It is not a ceiling; the weaker part's rate ("min") is.
- Control (c) is possible for only 18% of theorems (234).
- Failure places are assigned by which part's letters the rejected line uses. This is a heuristic.
- Most L17-18 labels are certified upper bounds (`ub_cert14`), which does not matter at 0%.
- GPU cost: about 23 GPU-minutes in total. The four jobs ran 5-7 minutes each through the lock.
