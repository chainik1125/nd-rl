---
written_on: 2026-11-01
written_by: agent:claude
---

# Proof strategies of the GRPO + thinking-trace models, and Figure 1 ("capability emergence")

This reruns Dan's proof-strategy classification (taxonomy v1.2) on our own pipeline:
- the Stage-1 corpus of base `S1g3`;
- the Lean-accepted proofs in the existing pass@k reads of base S1g3 and of the GRPO
  thinking-trace checkpoints.

From that it builds Figure 1 of the high-RL regime paper. Leon asked for this on 2026-11-01. The
work was CPU only: no model was sampled or trained, and no GPU was used.
- **Branch.** `leon/high-rl-paper` at `1ab6ab7`, uncommitted (the coordinating agent integrates).
- **Human-readable summary.**
  [`experiment-summaries/2026-11-01-proof-strategies-grpo-thinking/`](../../../../experiment-summaries/2026-11-01-proof-strategies-grpo-thinking/README.md).
- **Run log.** [`log.md`](log.md).
- **Pre-registration.** [`PREREG.md`](PREREG.md), sha256 `0e9d361d…b3e6`, frozen at 11:09 am PT
  before the first model-proof count.
- **Every number.** [`numbers.md`](numbers.md) / `numbers.json`.

## Rerun

From the repository root, with `ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 nice
code/.venv/bin/python -m code.experiments.current.proof_strategies_grpo_20261101.<script>`:

| script | what | time |
|---|---|---|
| `classify k12` / `corpus` / `reads` / `theorems` / `pools` | Dan's labels for K12, the S1g3 corpus (replayed from its trace actions), the 87 reads, theorem-level labels (G4ip), the GRPO pools; 8 worker processes | ≈ 4 min in all |
| `analyze [--draws 4000]` | `numbers.json`, `numbers.md` | ≈ 1 min |
| `agree` | the reviewer's blind classifier vs Dan's on every distinct proof → `agree.json` | ≈ 5 min |
| `spotcheck sample` / `compare` | blind sample for an independent labeller; agreement → `spotcheck.json` | seconds |
| `fig1 [--draws 4000]` | `docs/papers/high-rl-regime/figures/fig1_emergence.{pdf,png}`, `fig1_numbers.json`, Lean re-check of the shown proofs | ≈ 1 min |
| `python -m pytest code/experiments/current/proof_strategies_grpo_20261101/test_ps.py` | 18 tests (below) | 6 s |

- **Large outputs** go to `/scratch/cowifornia/nd-rl/proof_strategies_grpo_20261101/`: labels,
  per-read proof tables, the spot-check files.
- **Dan's classifier** is used unchanged, from a read-only shallow clone of `dan-pandori/nd-takehome`
  branch `dan_proof-strategies` at `e6bfe0ea`, in `/scratch/cowifornia/nd-rl/ext/nd-takehome-ps`
  (env `ND_PS_FORK`). File hashes are in the log.

## Inputs and units

- **Stage-1 corpus.**
  - `S1g3.pkl` is a graded build of 154,630 programmatic search traces of K12 theorems
    (`think/traces/search/full/e256.jsonl`). Its trained actions equal the `e256` actions on every
    record.
  - The corpus is classified by replaying each record's actions in the fork's state environment
    and converting the Lean text to ND (`lean_tok.inverse`). This reproduces `plain_e256.jsonl` on
    154,630 / 154,630 records.
  - Only 35.5 % of these search proofs equal K12's generator proof. **So the base's corpus is not
    Dan's K12.**
- **Model proofs.**
  - Each read row's `accepted` is {Lean text: count}. It is per attempt: the counts sum to `n_ok`,
    the number of accepted attempts out of k.
  - Texts that differ only in hypothesis names are merged after the conversion to ND.
  - Shares count **attempts** (units A, B, C) unless a table says "distinct proofs" (unit D).
- **Units (PREREG.md).**
  - **A**: % of attempts that are accepted *and* use the strategy, averaged over problems. It mixes
    competence with strategy choice.
  - **B**: % of a solved problem's accepted attempts that use the strategy, averaged over solved
    problems. This is strategy choice.
  - **B-matched**: B on the problems both arms solve.
  - **C**: % of problems with at least one such proof.
- **Strategy names.**
  - "Working #10" is a proof by contradiction whose reductio does logical work.
  - "Template-only" means every reductio in the proof is vacuous, pointless or an idle scaffold.
  - "Opens with ¬goal" means the first written step assumes the negated goal, and the root is DN
    of that ¬I.
- **Sets.**
  - Charles149 at k 1,024: 149 textbook problems; base and L3 end for seeds 0–2, and 16 seed-0
    checkpoints from 7.6e-5 to 0.45 R.
  - t1320s250 at k 512: 250 problems with 13–20-line reference proofs, the same seed-0
    checkpoints.
  - Smaller-k reads: charles173, l2test, l3test, t1320, t2130.
  - textbook72 is left out.

## Results (observed)

**Q1. Pretraining mix.** Shares are % of proofs.
- **S1g3's corpus.**
  - working #10 1.47; MT 0.13; DS 1.19; working ¬I 5.56; genuine cases 6.04.
  - Against Dan's K12: 0.04 / 0.05 / 0.03 / 2.15 / 8.79.
  - Against Charles's reference proofs of the 149: 17.45 / 7.38 / 14.09 / 28.86 / 34.23.
- **Proof by contradiction in the corpus.** The corpus has about 37× K12's working reductios but
  almost none of K12's template ones (template-only 0.92 vs 10.75 %). It uses a reductio when the
  theorem needs one: 22.7 % of its proofs of the 3.9 % classical-only theorems, against 0.61 % on
  the intuitionistic ones.
- **The other textbook strategies.** MT stays 57× and DS 12× below the reference-proof rates.
  K12's reproduction matches Dan's table on all 117 cells.
- **The GRPO training pools are different again.**
  - L2 / L3 are 39.5 / 44.5 % classical-only theorems.
  - Their generator proofs use working #10 in 40.2 / 46.6 % and MT in 8.1 / 13.4 %.

**Q2. Base vs RL on the same problems** (Charles149, seed 0, L3 end unless noted).
- **The base already uses the textbook strategies at textbook rates when it succeeds.**
  - B: working #10 17.5, MT 9.9, DS 19.1, working ¬I 32.8, genuine cases 33.4.
  - Means over 3 base seeds: 16.7 / 7.4 / 19.6 / 30.6 / 35.3.
  - This is despite the corpus's 0.1–6 %: the problems, not the corpus mix, set the strategy.
- **Per attempt (A), every strategy rises** with pass@1 (39.2 → 70.3 %):
  - working #10 3.0 → 16.6 (+13.7 [+9.3, +18.3]);
  - DS 7.2 → 14.0;
  - working ¬I 8.8 → 19.4;
  - genuine cases 11.4 → 25.0;
  - MT 2.0 → 4.0.
- **Like for like (B-matched, 117 problems).**
  - working #10 +5.3 [+1.7, +9.5]; template-only #10 +12.9 [+8.3, +18.0] (1.5 → 14.4 %).
  - MT −2.1 [−4.7, −0.1]; DS −1.1 [−3.9, +1.5]; working ¬I −1.1 [−4.2, +1.6]; genuine cases
    +0.5 [−0.0, +1.6].
  - Proofs opening with ¬goal rise from 6.2 to 13.0 %; forward openings fall from 23.2 to 18.9 %.
  - Dead lines per proof rise from 0.08 to 0.38, and pruned length from 9.0 to 12.3 lines.
- **By theorem class.**
  - On the 119 intuitionistic problems, RL uses working #10 in 12.4 % of proofs (base 3.8) and
    template-only #10 in 17.2 % (base 1.9).
  - On the 30 classical-only problems it is forced: 89.7 vs 83.8 %.
- **13–20 lines (t1320s250, 192 matched).** The shift is much larger: working #10 7.1 → 34.6
  (+27.5 [+23.3, +32.0]); template-only +29.8; DS +8.4; working ¬I +9.1; MT +0.4. On the 225
  intuitionistic problems, working #10 goes from 3.2 to 33.0 %. Opening with ¬goal goes from 1.1
  to 13.3 %.
- **RL-only problems.**
  - At seed 0, 16 problems are RL-only at k 1,024 and none is base-only; 7 of the 16 are
    classical-only (44 %, against 20.1 % of the set). Pooled over 3 seeds, 9 problems are RL-only,
    3 of them classical-only.
  - RL's proofs of the seed-0 RL-only problems use working #10 in 66.6 % and DS in 38.7 %, against
    22.8 / 18.1 % on base-solved problems. They are longer (19.3 vs 11.7 pruned lines).

**Q3. Emergence against R** (seed-0 chain; logistic vs line in log10 R).
- **Base-unsolved solve rate** (32 problems the base never solves at k 1,024).
  - It stays at 0.0–0.3 % of attempts for R ≤ 6.6e-3, except one checkpoint (kp16 st32, R
    1.8e-3, 3.3 %).
  - It then reaches 7.4 % at 0.03, 23.3 % at 0.15 and 27.5 % at the L3 end.
  - Logistic: ΔAIC +54.7, midpoint R 0.054, width 0.22 decades. **S-shaped by the pre-registered
    test.**
- **Working #10 per attempt.**
  - Charles149: 3.0 → 16.6 %, ΔAIC +56.7, midpoint 0.046.
  - 13–20 lines: 0.43 → 13.1 % (30×), ΔAIC +35.9, midpoint 0.083. As a share of proofs (B) it
    goes from 7.0 to 37.8 % (ΔAIC +42.6).
- **No strategy's share of proofs (B) is S-shaped on Charles149, apart from #10.** For MT, DS,
  working ¬I and genuine cases, ΔAIC < 2.
- **Yue et al.'s median (1.2e-4).** Every curve is at base level there.
- **Tracking the pass@k gap.** Across the 16 checkpoints the curves track the pass@1024 gap more
  closely than the pass@1 gap. Spearman ρ: base-unsolved rate 0.97 vs 0.84; working #10 0.96 vs
  0.83.

**Q4. Against Dan's EI ladders.**
- **Dan's ladders.** Working #10 among new distinct proofs goes from 0.17–0.53 % (round 1) to
  1.4–4.5 % (round 16) on his full ladders: 4.6–8.9× as shares, 13–30× in counts.
- **Ours, base → L3 end.**
  - Distinct proofs: 22.4 → 51.6 % (2.3×; count 371 → 4,938, 13×) on Charles149; 6.2 → 45.5 %
    (7.4×; count 229 → 10,087, 44×) on 13–20 lines.
  - Per attempt: 5.6× and 30×.
- **The comparison.** The fold changes are of the same order as Dan's, but our levels are about
  10–130× higher. Our base already writes working reductios, because its search-proof corpus contains them;
  Dan's K12-trained base barely does.
- **Not like for like.** Dan counts proofs of RL targets during EI; we count eval reads of
  checkpoints.

## Interpretation (not tested here)

- **RL does not introduce a strategy the base lacks.** It raises how often each strategy ends in
  an accepted proof. It shifts strategy choice toward proof by contradiction: working and,
  more so, template reductios, and proofs that open by assuming the negated goal. The shift is
  largest on the long 13–20-line problems.
- **The shift matches the GRPO curriculum.** About 40–45 % of the pool theorems need a reductio,
  so the most likely reading is transfer of a habit the RL distribution rewards, not discovery.
- **The capability signal is real.** It is the base-unsolved solve rate, S-shaped in R with a
  midpoint near 0.05. Its strategy content on those problems is long compositions of reductio,
  DS and case analysis.

## Expectations ledger (PREREG.md)

| id | result |
|---|---|
| E1 | 3 / 5 in range: working #10 17.5, working ¬I 32.8, genuine cases 33.4 hit. MT 9.9 (expected 1–8) and DS 19.1 (2–12) are higher: the base is at textbook rates |
| E2 | Miss for template-only #10 (+12.9 > 10 points); every other tracked strategy hits (\|Δ\| ≤ 5.3; LEM, DNE and ex falso within 1.0). Composition explains about half of working #10's pooled rise (B +10.6 overall vs +5.3 matched). "Dead lines fall": **miss** (0.08 → 0.38) |
| E3 | Hit: every tracked strategy rises in A; working #10 5.6× |
| E4 | Hit: RL-only problems are 44 % (seed 0) and 33 % (pooled) classical-only; working #10 66.6 vs 22.8 %, DS 38.7 vs 18.1 % |
| E5 | Shape hit: flat to about 1e-3, rising 3e-3–0.15, flat to ki78, S-shaped by the test. "Tracks pass@1 more than pass@1024": **miss** (ρ 0.83–0.84 vs 0.96–0.97). The rule picked MT on Charles149 (see Deviations) |
| E6 | Hit: larger per-attempt folds on 13–20 lines for all five strategies (working #10 30× vs 5.6×) |
| E7 | **Miss**: per-attempt fold 5.6× (Charles149) and 30× (13–20 lines), not ≤ 3×. Dan's count folds are 13–30× |

## Deviations

- **Panel (c).** The pre-registration says a selection rule picks panel (c)'s strategy, but not on
  which set.
  - On Charles149 the rule picks MT (base B 9.85 %, just under the 10 % bar), whose share of proofs
    falls under RL. On 13–20 lines it picks working #10.
  - The figure shows the pre-registered fallback quantity on Charles149, the base-unsolved solve
    rate, next to the rule's 13–20-line pick. Both are unit A, as pre-registered.
- **Added after the pre-registration** (logged at 11:12–11:20 am, after the first counts):
  - the opening-move feature;
  - the corpus split by theorem class;
  - the pool mixes.
- **Panel (b) examples** (revised at the coordinator's request, 11:47 am). In each, the shown
  proof is the shortest one without dead lines across RL seeds 0–2.
  - The first example is now `textbook_49b7d961bb8d378750b8`, the only one of the 9 pooled
    base-never problems (0 of 3,072) with a short proof using a rare strategy: 13 lines, a
    reductio from the negated goal with a modus tollens (RL 34 % of 3,072).
  - The second stays `textbook_5b38b6ab0a04f4c4bb28` (two DS, 17 lines; base 4 of 3,072, RL
    52 %).
  - The other 8 base-never problems fail:
    - 7 are too long: shortest RL proof 15–55 lines, shortest proof with a rare strategy 16–55;
    - d337's 11-line proof uses only case analysis, which is not rare in the corpus;
    - none failed the textbook72 guard.
  - The first version used `textbook_99f1…` (base 5 of 3,072); it is replaced.

## Validation

- **Dan's classifier.**
  - Reproduces his K12 summary exactly: 117 / 117 cells and box-free.
  - His 39 hand-built tests pass, both directly and after a round trip through the models' Lean
    surface.
- **Converter.**
  - The K12 round trip ND → Lean → ND keeps every pruned label on 2,000 / 2,000 sampled proofs.
  - Corpus replays equal `plain_e256` on 154,630 / 154,630 records.
  - No accepted read text failed to convert.
- **Hand-built proofs.** 13 proofs in the models' Lean surface get the expected labels. Lean
  accepts all 13 and rejects a broken-citation control.
- **Reviewer's blind reimplementation** (`review_ps/cls.py`). Identical uses, working flags,
  primary and curried primary on 168,566 / 168,566 distinct model proofs and 154,630 / 154,630
  corpus proofs.
- **Blind spot-check** (an independent subagent given TAXONOMY.md only; 100 proofs, 9 positives
  per tracked stratum + 28 random).
  - Agreement is 100 / 100 on all 13 uses labels, the 4 working flags and the primary.
  - Limits: no LEM positive; pruned length ≤ 16. The labeller wrote its own pruning and
    dependency scripts, so its strategy calls were by hand but its bookkeeping was by script.
- **Theorem labels.** Dan's G4ip agrees with Charles's `classical_only` field on 149 / 149.
- **Lean.** Both proofs shown in Figure 1 are re-checked from the model's own texts: 6 / 6
  accepted.
  Both are identical to Charles's reference proofs.
- **Sealed set.** 0 of the 149 Charles problems are in a textbook72 renaming class. textbook72 was
  never read.
- **Lint.** ruff check and format pass (line length 100).

## Limitations and next step

- **Seeds and trajectory.** The R chain is seed 0 and mixes runs (kp16, kdense, mini L2, L2, L3),
  so it is not one trajectory. The base-unsolved set is defined by base seed 0 at k 1,024.
- **Selection of RL-only problems.** RL-only counts are seed-noisy. Example: base s0 never solves
  `textbook_26c5…`, s2 solves it 398 / 1,024.
- **Definitions.** The taxonomy is syntactic, and "working" does not mean "necessary".
  Strategy-level necessity was not run.
- **Next.** A per-iteration count would need rollout dumps (`--dump_batches`; none exist). A
  matched-curriculum control would separate learned style from transfer: GRPO on a pool with
  K12's 4 % classical-only share.

## Files

`common.py` (paths, loaders, Lean ↔ ND), `classify.py`, `analyze.py`, `agree.py`, `spotcheck.py`,
`fig1.py`, `test_ps.py`; `PREREG.md`, `log.md`, `caption.md` (draft caption), `numbers.md` /
`.json`, `fig1_numbers.json`, `agree.json`, `spotcheck.json`.
