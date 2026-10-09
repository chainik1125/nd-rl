# Post-analysis of the pool screens (2026-10-07)

What the data screens (`screen_data.py`) and the frozen-checkpoint screens (`screen_ckpt.py`) found across the 27
pools built this week, what the numbers signal, what the screening did well, what it still lacks, and where the next
steps go. Sources: `pools.md` ("Data screens", "Checkpoint screens" and their readings), `data/pretrain_data/screens/
{data_screens,ckpt_screens,followups}.json`. Every number here is one checkpoint seed (`scale-d384-p1200_s0`), one
validation set (224 seed-split problems), 160 prompts × 32 samples per slice unless stated; the noise floor on pass@32 is
about ±0.04 (see § 3).

## 1. The two screens measure different things, and that is the result

The data screens ask whether the pool *contains the shapes the textbook problems need*. The checkpoint screens ask
how *far the base model is from the pool's new material*. Across the 19 pools whose slices are new problems, the two
barely correlate (Spearman, pass@32 vs next-rule accuracy −0.38, vs nearest-sequence share −0.01, vs need-weighted
coverage −0.02). They are two axes, relevance and novelty, and a pool is placed on both:

| | **far / near-far for the base** (pass@32 < 0.35) | **mid-edge** (0.35–0.8) | **redundant** (≥ 0.8) |
| --- | --- | --- | --- |
| **relevant** (data screens above K12) | `k12_joins1/10`, `k12_minimal_joins10` (0.14–0.20); `k12_pairs1`, `k12_spice1` (0.33) | `k12_sub1/10` (0.51–0.54, partly circular); `k12_tail10/30` (0.61–0.66, coverage 0.97) | — |
| **not distinguishable from K12 on data** | `k12_dial_*` (0.25–0.29); `k12_aux10` (0.00) | `k12_distract10/30`, `k12_minimal_distract10` (0.54–0.58) | `k12_subst30/100` (0.81–0.85); `k12_500k` rows (0.975); cap6 (0.99) |

The target quadrant is the top-left: material the textbook needs and the base cannot yet produce. The join pools sit
there alone, with the pairs and spice pools at its edge. The tails are the most *relevant* pools (coverage 0.97, the
depth-3 cells nothing else supplies) but the base already reaches two thirds of depth-3 problems within 32 samples, so
their job is to raise a rate, not to create an ability. The distractors are invisible to the data screens by
construction (they add no shapes) and visible to the checkpoint screen as a 34-point deficit. The dial is the reverse
case: invisible to presence screens because its cells already exist in K12, and placed near "far" by the checkpoint
screen because the base finds the *composition* rare. The bottom-right quadrant is material the model already has:
nothing to learn from it except epochs.

## 2. What the numbers signal, pool by pool

**Padding is noise, with a matched-row proof.** On the same 3,000 K12 rows (the ones `minlen` could shorten), the
base model's per-token loss is 0.254 on K12's padded proof and 0.109 on the minimal proof; the padded proof costs
40.4 nats in total against 9.3, and the minimal proof is the lower-loss one on 77 % of rows. The padded rows are the
long half of K12, and 0.254 is well above K12's average 0.171: the lazy ∨E case split is the single hardest-to-predict
thing in the pool the model was trained on for 20 epochs. This is model-side evidence for the data-side finding that
89 % of the long half is padding, and it raises the prior on E3 (minimal) and E6 (minimal + joins). It also means the
cap-12 result Dan reported may rest on a *length* signal the model finds hard to learn, which is exactly what E3 tests.

**The join pattern is composed rarely, not never.** The dial's ↔-elimination problems are four short lines (∧E on
(a→b)∧(b→a), then →E). The base gives them *lower* per-token loss than K12's own proofs (0.135, 0.79×) yet solves
them at 6 % per sample and 26–29 % within 32, with zero exposure. So the pattern is not absent at base, as derived-∨E
was in Dan's dial (0 % at f = 0): it is present at a low rate, produced by composing two rules the model knows. Two
consequences. The dial will measure *amplification* of a rate from 0.06, not surfacing from zero, and should be read
against Shen's distinction. And the 42 hard textbook problems that contain ↔ are not failing for want of the four-line
pattern; they fail because the pattern has to sit inside 14+-line proofs at depth 3. The dial's readout must therefore
be on the textbook ↔ cells, not on the slice's own problems, and a second dial pattern with a near-zero base rate is
worth having (the per-family follow-up in § 5 picks it).

**Premise selection is a measured weakness.** With one or two unused premises inserted into K12 problems the base
otherwise solves at 0.89 within 32, pass@32 falls to 0.55 and pass@1 from 0.76 to 0.40. The generator guarantees that
every premise is cited, so the model has learned "use everything", and a premise it cannot use derails it. This is
the 4-premise cell's failure in miniature (0 of 7 four-premise textbook problems solved by any model) and it gives E10
a concrete pre-registrable expectation: a distractor-trained Stage-1 should close most of the 34-point gap on
distracted K12 problems *before* RL, and the ladder's gain on 4-premise textbook problems is the transfer test.
`k12_minimal` already carries a 29 % dose of accidental distractors (the dropped ∨E premise), and the base solves
those at 0.875, so the deficit is specific to premises that are *irrelevant*, not merely uncited.

**Depth 3 is reachable; length 13+ is not addressed by anything.** The tails' depth-3 problems are solved at
0.61–0.66 within 32 and their loss is 1.03–1.07× K12's: the base has the ability and the pool raises the rate. But
the generator's long mode stops at 12 lines, every pool's nearest-sequence distance to the 13+-line references is
4.4–6.0 of a cap of 6, and the only pool with long proofs (`k12_aux10`) has the wrong shapes (∧/→ chains, pass@32
0.00, loss 4.6×). The length gap, 52 of the 77 hard problems at 14+ lines, is the one axis no pool in the set covers
with relevant material. This is the biggest hole in the pool set.

**Substitution has no leverage at this width.** The base solves substituted problems at 0.81–0.85 and finds their
proofs *easier* than K12's (0.9×). The model generalises over formula size already; what it lacks is structure, not
width. `k12_subst*` also costs 45 % more tokens per record, so at fixed wall-clock it sees fewer proofs. Drop unless a
ladder is spare.

**More of the same is redundant.** The 345,000 fresh generator rows in `k12_500k` are solved at 0.975 and have loss
0.97× K12's. At 9.56M parameters and 20 epochs, K12 already saturates its generator's distribution; the size ladder's
upper arm tests epochs, not content, and its lower arms (212 and 64 epochs) are where Kopiczko's repetition result
would show. The held-out head's pass@32 (0.86) and loss (0.170) match the training rows (0.89, 0.171), so 20 epochs
have not produced memorisation; there is room for more.

**The wrong pools need a filter, and the base already knows validity.** 11.6 % of one-error proofs cannot be encoded
by the lean_seq tokenizer (rule swaps with no Lean rendering), so a lean_seq Stage-1 on `k12_wrong*` must drop them
and the "100 % wrong" arm is really 88 %. On the rows that encode, loss is 2.1× K12's: the base assigns one-error
proofs clearly less probability than correct ones, which means the pretrain has learned validity constraints and not
only format. That sharpens E14: if the all-wrong Stage-1 plus the ladder still reaches K12's cells, RL is supplying
something the data demonstrably withheld.

**Retention is not a concern at the 10 % dose.** No slice removes a type cell with more than six K12 records; at
30 % the distractor and tail slices take half of 10–12 small cells. Huang's rare-task interference is a risk for the
30 % doses only, and only for those cells.

## 3. What went well

- **Cost.** Data screens 10 s per pool; checkpoint screens 4 min per pool on a laptop CPU, 1 h 40 min for all 24.
  Nothing was trained and no money was spent. The whole screening tier cost one working day.
- **Negative controls caught a bad metric before it misled.** The count-model perplexity ranked the all-wrong pool
  *better* than K12 (14.8 vs 21.6) and was withdrawn; only its accuracy survived. Without `k12_wrong100` in the set,
  perplexity would have gone into the ranking as a loss proxy. The control is now a standing requirement for every
  screen.
- **The two tiers are independent.** Weak rank correlation between data and checkpoint screens means the second
  tier added information rather than repeating the first. The relevance × novelty placement in § 1 would not exist
  with either tier alone.
- **Internal consistency gives a noise floor.** The four dial arms share one problem distribution and score
  pass@32 0.25–0.29; `k12_distract10/30` 0.55/0.54; `k12_sub1/10` 0.54/0.51; `k12_wrong10/100` 0.875/0.894. The
  binomial standard error at n = 160 is ±0.04, and the repeats agree with it. Differences under 0.05 are noise.
- **Concrete, pre-registrable expectations fell out.** Distractor deficit 0.89 → 0.55; minimal-proof loss 0.109 vs
  0.254 matched; join pass@32 0.14–0.20; depth-3 0.61–0.66; substitution 0.81–0.85. Each is a number a Stage-1 on
  the pool should move, and each goes into the proposal before any run.
- **Infrastructure held.** The val head is K12's first 1,024 rows in every pool, so held-out loss is comparable; the
  pools.md sections survive regeneration; the screens re-run from their JSON (`--doc_only`); the lost K12 file and
  fork clone were recovered and verified against existing artefacts.

## 4. What is still lacking

- **The checkpoint is not the ladder's base.** The screen used Robbie's `lean_seq` pretrain (whole-proof tokens,
  loss on proof tokens). The plan fixes Stage-1 as Robbie's recipe *as ported by Dan* in the `lean_staten`
  proof-state format with Dan's T1 ladder (plan § 4). Same data, same size, different tokenisation and objective
  (action tokens per state). The regimes are those of a sibling model. They are probably similar, since both are
  9.56M models of the same data, but "probably" is not a measurement: the screen should be re-run on Dan's cap-12
  best-state checkpoint with `state_eval.py` before any GPU decision rests on it.
- **The screens evaluate the base on the slice's own problems, not on the textbook targets.** Pass@32 on join
  problems says the base cannot do joins in isolation; it does not say a Stage-1 on joins would raise the ladder's
  yield on textbook ↔ problems. That mapping needs training (continued pretraining, J9; round-1 yield, J12). The
  screens rank and place pools; they do not predict ladder deltas, and should not be read as if they did.
- **Loss is confounded with proof length and shape.** Per-token loss is lower for short formulaic proofs whatever
  the model's ability (the dial case), and the first regime labels built on it were wrong. The right dial metric is
  *pattern-local*: the loss on the join's own tokens (the →E line after the ∧E) rather than the whole proof. Not yet
  computed.
- **Sample sizes.** 160 prompts per slice gives ±0.04; the join slice's nine families were sampled at ~18 prompts each,
  too few to read per family (the follow-up in § 5 fixes this for the joins only). The validation set is 224 problems
  with 58 unsolved and few 13+-line references, so the "need" weights rest on a small set.
- **One checkpoint seed.** `s1` and `s2` exist on the bucket and would cost 1 h 40 min each; the noise floor above is
  within-seed only.
- **Thresholds are stipulated.** far < 0.1, redundant ≥ 0.8, loss flags at 1.5× and 4×. They are reasonable and
  stated, but the regime a pool lands in near a threshold (distractors at 0.55, tails at 0.61) is a label, not a fact.
- **Sub-problem circularity has no clean fix.** The sub-problem slices were cut from the seed problems' reference
  proofs, so their data-screen distances are inflated by construction. The checkpoint screen does not share the
  problem (the base had never seen them) and puts them mid-edge; that is the number to trust.
- **Seven pools from the parallel build are unscreened** (`k12_shape`, `k12_balance`, `k12_multi10` and the others
  that bring the status doc's count to 34). Four minutes each; should be done before the ranking is used.
- **Temperature mismatch.** Screens sampled at T 1.0; the ladder samples at T 0.8 (plan § 4). Pass@32 at 0.8 would be
  a little higher for the base; the ranking is unlikely to change, the absolute numbers would.
- **No pool addresses length 13+ with relevant shapes** (§ 2). This is a gap in the pool set, not in the screens, but
  the screens are what exposed it.

## 5. Follow-ups run today

**Matched-row loss (done).** 3,000 shortened K12 rows, same problems: padded 0.254 / token (40.4 nats total, 158
tokens), minimal 0.109 / token (9.3 nats, 85 tokens), minimal lower on 76.9 % of rows. Source: `screens/followups.json`.

**Base pass@32 per join family (done).** 100 prompts × 32 samples per family on the join slice (frozen K12 model,
T 1.0, pinned verifier). Source: `screens/followups.json`.

| family | what it is | n in slice | pass@1 | pass@32 |
| --- | --- | --- | --- | --- |
| four_prem | four premises, all needed | 2,237 | 0.000 | **0.00** |
| nested_ore | ∨E inside ∨E | 1,129 | 0.000 | **0.00** |
| bicond_intro | prove (a→b)∧(b→a) | 869 | 0.013 | 0.04 |
| depth3 | three nested boxes | 1,729 | 0.014 | 0.05 |
| negation | ¬E / ¬I joins | 2,555 | 0.068 | 0.18 |
| bicond_elim | ∧E on ↔ then →E (the E8 dial pattern) | 1,390 | 0.054 | 0.23 |
| nested_ande | ∧E of ∧E | 2,353 | 0.093 | 0.30 |
| impe_chain | →E of →E | 1,884 | 0.205 | 0.31 |
| ande_impe | ∧E then →E on plain formulas | 1,354 | 0.173 | 0.45 |

Two families are at exactly zero over 3,200 samples each: **four premises** and **nested ∨E**. The first matches
the textbook result (0 of 7 four-premise problems solved by any model) and says the failure is not length: these are
4–9-line problems the base cannot start. The second is the shape of the lazy case split's *real* use, nested, which
the base has never seen because K12's ∨E is almost always a single padded split over a premise disjunction.
↔-introduction (0.04) and depth 3 (0.05) are near zero. The E8 dial pattern, ↔-elimination, is in the middle at
0.23: the base composes it at a low rate, so E8 measures amplification. A surfacing dial needs a zero-rate family:
**four_prem is the natural choice** (largest family, textbook-relevant, zero at base), with nested_ore as the second.

## 6. Where the next steps go

1. **Re-run the checkpoint screen on the ladder's actual base** (Dan's `lean_staten` cap-12 best-state checkpoint,
   `state_eval.py`, CPU, ~2 h). Keep the lean_seq run as the sibling comparison. Until this is done, treat the regimes
   as provisional. *Done 2026-10-08 for the ten pools on the list (§ 7): the rankings hold, two regimes move.*
2. **Screen the seven unbuilt-by-me pools** (`screen_data.py --pools …`, `screen_ckpt.py --pools …`, ~5 min each).
3. **Build the missing pool: long and relevant.** 13–16-line, depth ≤ 3 proofs with textbook skeletons: raise the
   generator's long-mode cap to 16 with the depth filter at 3, or prove the reductio-shaped join templates at bound 16
   as `hard_problems.md` proposed. Screen it the same way. Without it, nothing in the set speaks to 52 of the 77 hard
   problems.
4. **Fix the dial.** Keep `k12_dial_*` (↔-elimination, base 0.23) but read it on textbook ↔ cells, and add a second
   dial on **four-premise problems** (base 0.00 over 3,200 samples; 2,237 instances in the join pool, so 10⁻⁴ … 10⁻¹
   slices need permutations only at 10⁻¹), so one dial measures amplification and the other surfacing. A nested-∨E
   dial (also 0.00, 1,129 instances) is the alternative if the four-premise one is judged structural rather than a
   pattern. Compute pattern-local loss for both.
5. **Filter `k12_wrong*`** to rows both tokenizers can encode, and record the real wrong share, before E14.
6. **Pre-register the Stage-1 expectations from the screens** in each proposal: distractor pass@32 on distracted K12
   problems 0.55 → ≥ 0.80 for E10; join-slice pass@32 0.19 → ≥ 0.5 for E5; depth-3 0.61 → ≥ 0.8 for E9; substitution
   unchanged for E11; minimal-proof loss unchanged at 0.109 while padded-proof loss rises for E3. Each is a G1-style
   gate on the frozen Stage-1, before any ladder.
7. **Then the first GPU spend, J9** (continued pretraining from the base, 2,000 steps per pool against a K12-continued
   control, ≈ 2.5 A40-hours), with the screens' predictions recorded first so J16's calibration has something to
   calibrate against. Needs the budget confirmed and a `proposal.md`.
8. **Ladders:** the dial arms, K12, `k12_minimal_joins10`, `k12_joins10`, `k12_tail10`, `k12_distract10`, and the long
   pool from step 3 if it screens well. `k12_subst*` and `k12_aux10` only if a ladder is spare.

The one-sentence version: the screens show the pool set is strong on joins and depth, measurably weak on premise
selection (four-premise problems: zero at base), and silent on length; the next cheap step is to confirm the regimes on the ladder's own base model, and the
next build is the long pool nobody has made yet.

## 7. Confirmation on the ladder's own base (Dan's checkpoints, 2026-10-08)

Step 1 of § 6, run on the ten pools of the pilot list plus four finished before the list was cut (`k12_spice1`,
`k12_sub1`, `cap6`, `k12_joins1`). Source: `data/pretrain_data/screens/ckpt_dan_{stage1_best12_s0_b1200,la_T1_best12_s0_r8}.json`,
table in `pools.md` "Checkpoint screens on Dan's lean_staten checkpoints". Loss is Dan's action-token loss in the state format,
pass@k is his environment sampler at T 1.0 judged by the pinned verifier, 128 prompts × 32 per slice (resolution 1/128 on
pass@32), one checkpoint per row, no seeds. Robbie's lean_seq run (§ 1–2) is the comparison.

**The ranking survives the change of base.** Over the 14 slices screened on both models, Spearman ρ between Robbie's and
Dan's Stage-1 pass@32 is 0.93, between the loss ratios 0.86, between raw losses 0.87. Dan's base is higher on 12 of 14
slices (cap-12 state format against cap-6-era lean_seq), by +0.05 to +0.31 on pass@32. The regime label agrees on 12 of 14:

| moved | Robbie (lean_seq) | Dan Stage-1 | why |
| --- | --- | --- | --- |
| `k12_minimal` | redundant (pass@32 0.88, loss 0.64×) | edge (0.98, 1.95×) | the state-format model solves K12's problems but assigns minimal proofs twice its K12 per-token loss: it has learned the padded shape, as the matched-row read in § 5 found for the lean_seq base (0.254 vs 0.109) in the other direction. Problems redundant, proofs not. |
| `k12_tail10` | edge (0.61, 1.03×) | redundant (0.87, 0.76×) | depth-3 tails are already solved by the cap-12 state base, so E9's gate ("depth-3 0.61 → ≥ 0.8") is met at base on the ladder's own model, before any new data. |

Loss ratios are uniformly larger on Dan's base (joins 3.3–3.5× against 1.4×) because his K12 training loss is a third of
Robbie's (0.067 vs 0.171): the denominator moved, not the slices. The 4× "far" line was set on the lean_seq run and should
not be read literally on the state model; by pass@32 the placement is the same as before.

**What eight RL rounds did to each slice (r8 − Stage-1, pass@32).** K12 itself: pass@1 0.67 → 0.94, pass@32 0.98 → 1.00,
loss 0.067 → 0.057. The slices:

| slice | Stage-1 → r8 pass@32 | Δ | loss ×(r8 / Stage-1) |
| --- | --- | --- | --- |
| `k12_dial_1e1` (↔-elimination) | 0.35 → 0.77 | +0.41 | 0.72 |
| `k12_joins10` | 0.36 → 0.69 | +0.33 | 1.14 |
| `k12_sub10` | 0.54 → 0.84 | +0.30 | 0.93 |
| `k12_minimal_joins10` | 0.30 → 0.55 | +0.26 | 1.10 |
| `k12_joins1` | 0.45 → 0.68 | +0.23 | 1.15 |
| `k12_pairs1` | 0.63 → 0.85 | +0.22 | 0.89 |
| `k12_tail10` | 0.87 → 1.00 | +0.13 | 0.92 |
| `k12_distract10` | 0.63 → 0.69 | +0.05 | 1.27 |
| `k12_wrong100` (K12's own problems) | 0.99 → 1.00 | +0.01 | 1.02 |
| `k12_minimal` (K12's own problems) | 0.98 → 0.99 | +0.01 | 0.88 |
| `cap6` | 0.98 → 1.00 | +0.02 | 0.85 |
| `k12_aux10` | 0.00 → 0.00 | 0 | 1.33 |

Three readings:

1. **RL lifts what has room, except distractors.** Excluding the degenerate `aux10`, Δ pass@32 is anti-correlated with
   Stage-1 pass@32 (ρ −0.89, n = 11). The one slice off that line is `k12_distract10`: base 0.63, only +0.05 after RL,
   and the loss on its proofs rose 27 %. Premise selection among distractors is the thing the ladder does not learn from
   K12 targets, so it is the clearest case for supplying it in pretraining. (The RL eval's 7 four-premise problems at 0 of
   64 are the textbook face of the same gap.)
2. **Loss after RL is not solvability.** On the join slices pass@32 rose 0.23–0.33 while the loss on the pool's reference
   proofs rose 10–15 %: the RL model solves join problems its own way and moves away from the join pool's proofs. The
   regime rule's loss leg therefore flips joins and distractors to "far" at r8 (ratios 4.5–5.5×, partly because the K12
   denominator fell 16 %). Regime labels on an RL checkpoint are not comparable with Stage-1 labels; use them on Stage-1 only.
3. **The pilot list changes by one.** Joins stay the only relevant-near-far quadrant on the ladder's own base (0.30–0.45 at
   base, 0.55–0.69 after RL, so RL closes about half the gap unaided). `k12_tail10` loses its ladder slot: redundant at
   base (0.87), saturated by r8 (1.00). `k12_distract10` moves up for the reason in reading 1. `k12_aux10` is confirmed far
   on both models and stays off the list.

**Re-based gates for § 6 item 6** (same form, Dan's Stage-1 numbers as the baseline, G1-style on the frozen Stage-1):
distractor pass@32 0.63 → ≥ 0.85 for E10; join-slice pass@32 0.36 → ≥ 0.60 for E5 and 0.30 → ≥ 0.55 for E6; sub-problem slice 0.54 → ≥ 0.75 for
E1; E9's depth-3 gate is already met at base and is dropped as a gate (kept as a check that it does not fall); minimal-proof
loss ratio 1.95× → ≤ 1.2× for E3 (the state model should stop preferring the padded shape). The ↔ dial reads 0.35 at
10⁻¹; the lower fractions were not screened on Dan's checkpoint.

**Caveats.** One checkpoint per row and one seed; Dan's noise-floor study puts most of the between-run variance in the
training run, so a second Stage-1 seed would be needed before any Δ under ±0.1 is read. 128 prompts per slice.
`k12_wrong100` decomposes for only 42 % of rows (the state format rejects the wrong lines), so its loss is on the encodable
subset. The four pools finished before the list was cut (`spice1`, `sub1`) have no r8 row.

