# Open questions for Charles (pretrain-data)

Each entry: the question, the default being followed until answered, and the date asked.

1. **Seed / test split.** Default: seeds = candidate_v0 (forall x ×3, Carnap) + the Lean-sourced
   47; test = candidate_v1 (Roy, Carr), batch 3, hand-proved 14, the 72 in aggregate. Roy's
   exercises are heavier than forall x's, so E1 could fail on the test split for reasons of
   difficulty rather than transfer; the seed-split sets are also reported as an in-distribution
   check. Alternative: split every set in half by renaming class, which measures within-book
   transfer instead. (2026-10-03)

2. **Budget and who runs the ladders.** Phase A (4 arms × 1 seed) plus phase B (2 arms + baseline
   × 3 seeds) is about 10 Stage-1 runs and 10 T1 ladders on A40s. Default: no pod is started until
   the budget is confirmed; the ladders run from our own worktree of Dan's branch with his
   scripts unchanged, on pods we start and stop ourselves. Alternative: hand the pool files to
   Dan and ask him to run them in his harness. (2026-10-03)

3. **Permutations of seed problems.** P1 multiplies each sub-problem class by its 24 letter
   permutations to raise its weight. This is a renaming of candidate problems, which the rules
   allow for seeds (only the 72 are protected). Default: allowed, recorded per record with the
   seed name. (2026-10-03)

4. **Which pools go to the first Stage-1 pilot, and is the baseline still K12?** Eight pools are built
   (`pools.md`): K12, minimal K12, K12 + 1 % / 10 % joins, K12 + 1 % / 10 % sub-problems, the 1 % spice
   rack, minimal K12 + 10 % joins. Finding that changes the question: 89 % of K12's long-mode rows are
   padded, so Dan's cap-12 gain may come from proof *length* rather than content, and minimal K12 removes
   that length. Default for phase A if nothing else is said: K12 (baseline, exists), `k12_minimal`,
   `k12_joins10`, `k12_sub1`, `k12_minimal_joins10`, one seed each; `k12_spice1` and the 1 % join slice
   only if a 10 % slice moves a cell. (2026-10-06)
   Update (2026-10-06, late): thirteen more pools from the literature scan are built (`pools.md` "Scan pools":
   the E8 fraction dial × 4, depth tail × 2, distractors × 3, substitution × 2, long easy auxiliary, premise
   pairs). The size ladder (E7) adds four. That is 25 candidate pools against a budget of about 11 ladders.
   Default if nothing else is said: phase A runs the two CPU screens (held-out loss on a textbook-shaped
   validation set and base pass@k on the type cells) on *every* pool's Stage-1 checkpoint, which needs only
   Stage-1 time, and spends ladders on the E8 dial (four arms, the one experiment that settles the slice
   fraction for everything else) plus the best two screened pools and K12.

5. **"Spurious rewards (0 if right, 1 if wrong)": a pretraining pool or an RL reward?** Charles asked for a pool
   (2026-10-07), so it was built as pretraining data: `k12_wrong10/100`, K12 proofs with one local error each,
   every one rejected by the pinned verifier (plan E14). The other reading, an *inverted verifier reward in the
   ladder* (reward 1 for a rejected proof), is a one-flag change to the ladder's scorer, not a dataset; it would
   test whether the ladder's gains survive a meaningless reward (Shao et al. 2025 found they did on Qwen because of
   a pretraining prior, and did not on other families). Default: the pool is built now; the inverted-reward ladder
   arm is written up as E15 only if Charles wants it, since it spends a ladder on a control.

6. **Seven more pools (2026-10-07, `new_pools.py`): which, if any, join the pilot list?** Built from ideas that had no
   builder: closure sub-theorems (E16), lemma-then-use compositions (E17), recovery data with one dead line (E18), second
   proofs per theorem (E19), ⊥ / ¬¬ duals (E20), a seed-split-shape-matched resample of the generator population (E21)
   and a rule-balanced resample (E22). The slice sizes (10 %, 20 %, 10 %, 10 %, 1 %, whole, whole) are defaults, not
   measured. Default: they go through the same data screens as the other 27 and the same checkpoint screens when those
   run; none is pre-placed on the ladder list. `k12_shape` and `k12_balance` differ from K12 in *everything* at once, so
   a gain there needs the slice pools to say which cell carried it.

7. **Does `k12_tail10` keep its ladder slot?** On Dan's Stage-1 base the depth-3 tail slice is already redundant (pass@32
   0.87, loss 0.76× K12) and saturated after 8 RL rounds (1.00), so E9's gate is met before any new data
   (`screens_analysis.md` § 7). Default: it leaves the ladder list and `k12_distract10` (base 0.63, +0.05 after RL, the one
   slice RL does not lift) takes the slot; the tail pool stays as a Stage-1 screen check only. (2026-10-08)

8. **Does `k12_prem4_10` take a ladder slot, and is `k12_jdist10` too far?** The four-premise pool is 0.00 at base on
   Robbie's model at every fraction (coverage 0.69), the joins-plus-distractors pool 0.05 (`pools.md` "K pools:
   reading"). The band the screens were calibrated on puts 0.05 near "far". Default: `k12_prem4_10` goes on the ladder
   list beside `k12_distract10` and `k12_joins10`; `k12_jdist10` is a stretch arm, run only if a ladder is spare or if
   the Dan-base re-screen puts it above 0.1. (2026-10-09)
