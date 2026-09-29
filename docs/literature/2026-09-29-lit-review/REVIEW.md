---
written_on: 2026-09-29
written_by: agent:claude
papers:
  - Hubert_2025
  - poesia2024learningformalmathematicsintrinsic
---

# Literature review: techniques we have not tried yet (run `lit-review`, 2026-09-29)

Run brief: nd-rl `docs/proposals/literature/BRIEF_lit-review.md`. Pre-registration:
`preregistration/lit-review.md` (commit `dae83e61`, 03:22 UTC, before any screening). 58 papers screened,
23 read in depth (notes in `lit_review/notes/`). Sources are arXiv HTML or PDF text fetched with
`lit_review/fetch.py` into `~/lr_sources/` (kept out of git). Every quote below gives the id, version and
location, and `python3 lit_review/quote.py <id> '<regex>'` prints it back in context.

> **Librarian's note (2026-09-29).** This is the reviewed copy of the fork's `lit_review/REVIEW.md`
> (branch `dan_lit-review`, `67aa57fb`). The review
> ([`review_lit-review.md`](https://github.com/dan-pandori/nd-takehome/blob/2da89271/review_lit-review.md))
> verified 43 / 43 checked claims, 55 / 55 identifiers and every code mapping, and asked for six
> rewordings. They are applied here, each marked *[reviewed: F…]*: F1 (#1 and #5 are already proposed in
> [proposal 14](../../proposals/2026-09-28-after-state-env.md) item 1 and
> [proposal 13](../../proposals/2026-09-28-state-conditioned-environment.md)), F2 (ladder-A T5 was a static
> supply arm), F3 (the "≈ 99 %" pairing claim is dropped), F5 (four screened-table rows repaired from the
> executor's `~/lr_out/screened_rows.md`), F6 (He et al. §3.4 is analytic), F8 (AlphaProof's T_steps is the
> longest branch). Paths such as `lit_review/…` and `preregistration/…` refer to the fork. The reviewer
> would rank #2 level with or above #1, and asks for a same-checkpoint EI re-run pair before either (see the
> [experiment summary](../../../experiment-summaries/2026-09-29-lit-review-untried-techniques-search-expert-and-frontier-supply/README.md)).

**Which models "here" means.** The bottleneck numbers are inherited, not re-measured.
- **SN / S**: 3,216,384-parameter `lean_staten` / `lean_state` GPTs, from scratch, trained on
  `data/p2/train_depth3_f0_a1.jsonl` (155k, cap 6), Stage-1 seeds s0–s5 (`state-frontier`).
- **C0**: 3,214,336-parameter `lean_seq`, trained on the same set (`lean-format` s0/s1).
- The `L_true` labels are ND-derived upper bounds under Lean.

## Bottom line

1. **Nobody has run the ExIt experiment we can run.** Across the search papers, none shows that *search as
   the expert* trains a better policy than *sampling as the expert* at matched compute, with the policy
   then evaluated without search.
   - ExIt's own comparison is against REINFORCE in Hex.
   - TS-LLM's comparison is one unmatched run: 47.9 vs 47.5 (2309.17179v2 Table 4).
   - The state env built in proposal 13 makes that experiment cheap. It is also the most direct literature
     test of "search creates capability", so it ranks first.
   - *[reviewed: F1]* Search with a steps-to-go value head is already proposed as
     [proposal 14](../../proposals/2026-09-28-after-state-env.md) item 1 (and named as proposal 13's next
     step). #1 here is a sharpened design of that proposal: best-first before a value, minimal-proof
     targets, and the apprentice read out *without* search. The "nobody has run it" claim holds among the 58
     papers screened; the reviewer did not search beyond them.
2. **Our curriculum has tried *allocation* but no *model-conditioned supply* past `L*`.** *[reviewed: F2]* Ladder-A's difficulty weighting (T2) and
   moving window (T4) are learnability sampling in the LILO / online-filtering sense (`ladder_ei.allocate`,
   `ladder_ei.py:60`), and neither moved the frontier. That result is from ladder-A (2026-09-18): the token-format `stage1_abs.pt`, judged by `nd_verify`, before the Lean-only rule. Cross-seed sharing (T6) is Lee et al.'s cross-seed
   pooling. What the literature adds is *new targets just past the frontier* (STP, Lee et al.,
   Goedel-Prover-V2). We have none that are conditioned on the model: ladder-A T5 injected a static cap-6
   generator pool (2,505 records at round 4), with no gain.
3. **The cheapest high-value item is a measurement, not a technique:** score every EI-found proof of a
   base-unreached theorem step by step under the base model. It answers the open "new moves vs compounding
   reliability" question from files already on disk (§c).

## (a) Ranked shortlist

| # | technique | source (verified location) | bottleneck | plugs in | cost |
|---|---|---|---|---|---|
| 1 | **Search as the EI expert** in the state env: best-first on length-normalised log-prob, training only on the minimal proof of every solved node; truncate-and-resume as the no-search baseline | BFS-Prover 2502.03438v3 Eq. 1 `score = Σ log p / L^α`, Table 1 70.83 %; HTPS 2205.11491v1 Table 4 (minimal proofs of all solved nodes 78.1 vs all proofs 40.6, Equations) and §7.2.1 "always leads to improved performance"; InternLM2.5-StepProver 2410.15700v2 §3.2 (BF finds proofs of mean length 1.66 vs 4.44 with a critic); ExIt 1705.08439v4 §6.1 | B1, B2 | *[reviewed: F1: already proposal 14 item 1]* new `search_generate` next to `state_sample.env_generate` (`state_sample.py:34`); `Env` (`state_env.py:172`) needs a clone (or replay via `decompose`, :512); the sampler must return per-action log-probs (`sample.generate_ids_fast`, `sample.py:131`, discards them); called from `state_ladder_ei.generate` (:51) | ≈ 1 executor session of code; 12 state ladders ≈ 25–35 pod-h ≈ $12–18 (A40 billed $0.49/h in `state-frontier`) |
| 2 | **Frontier target supply**: extend or mutate solved proofs, and take the open subgoals of failed attempts; keep targets whose current-model p̂ ∈ (0, 1/4] | STP 2502.00212v4 §3.2 (keep conjectures with P̂ ∈ (0,1/4]); abstract (28.5 % vs "13.2 % achieved through expert iteration", an earlier paper's number); Lee et al. 2502.01612v2 §7.1 ("too difficult … is detrimental"); Goedel-Prover-V2 2508.03613v1 §2.2 (`extract_goal` on unsolved states); Firoiu 2103.03798v2 §5 (fixed generator saturates) | B1 | new mutation script beside `gen.Gen` (`gen.py:66`); p̂ from the fast sampler; targets fed to `state_ladder_ei` / `ladder_ei` target lists; open goals from `Env` at failure (`state_sample.py:73–79`) | small code; ≈ $10–15 |
| 3 | **Support-expanding GRPO advantages**: unlikeliness rank penalty; pass@k advantage; distinct-proof bonus as a third arm | He et al. 2506.02355v2 §4.1 (β_rank 0.25), Table 2 (8,065 / 9,600 training problems vs 7,860 GRPO, 7,707 static); Chen et al. 2508.10751v1 Table 1 (Enigmata pass@1/pass@k 12.9/21.3 → 17.9/29.8); PKPO 2505.15201v5 Tables 5–6; Darling 2509.02534v1 §5 | B2 | one line: `adv = R - R.mean(1)` at `grpo.py:117` | ≈ $10–20 (18 GRPO runs + support curves) |
| 4 | **Transductive EI on textbook theorems plus programmatic variants** | HTPS §7.1.3 (Identities 91.3 %, "while a supervised model never exceeds 36 %"); AlphaProof (Nature, DOI 10.1038/s41586-025-09833-y) Methods TTRL, "programmatically created variants", ED Fig. 3b "More synthetic variants consistently improves the final prove rate" | B4, B2 | a variant generator (drop / strengthen a hypothesis, split a conjunctive goal, sub-lemmas); targets passed to `state_ladder_ei` | ≈ $8–15 |
| 5 | **A steps-to-go value token**, trained from free labels, used as the search score | AlphaProof Methods: value "−T_steps" (*[reviewed: F8]* T_steps is the number of tactics in the longest branch needed to resolve all subgoals; equal to steps-to-go for our linear proofs, not with `Or.elim` subgoals), reward −1 per tactic "to incentivize the discovery of the shortest proof"; Polu 2022 2202.01344v1 §4.3.2 (proofsize buckets); HTPS Table 5 (soft critic 78.1 vs hard 63.1 vs none 65.6: a bad value is worse than none); Minimo 2407.00695v2 App. A (8.45 M model with a value token) | B1 | extra head or `VALUE` token in `model.GPT.forward` (`model.py:103`; `load_ckpt` :128 is strict) and in `fast_train.packed_logits` (`fast_train.py:225`, a second copy of the forward); labels from `state_env.decompose` | ≈ 1 session; ≈ $5–10 after #1 |
| 6 | **EI data selection**: least-likely correct proof per theorem (He et al.'s mechanism ported to EI), set against the already-proposed shortest-proof rule | He et al. §3.4 (*[reviewed: F6]* an analytic argument, assuming a uniform (1 + ε) uplift, shows the pass@N gain needs solutions with p₀ ≈ 1/N; not a measurement); HTPS Table 4 | B2 | the per-theorem choice under `--max_per_thm` in `ladder_ei.py` / `state_ladder_ei.py` | one flag; ≈ $5 |

**1. Search as the EI expert.**
- *Mechanism.* A priority queue over (theorem, state) nodes. Expand the best node with w sampled steps and
  push the children. With α > 0 the queue stops penalising depth. The tree yields many proofs per solved
  node, and HTPS trains only on the minimal ones.
- *Why here.* Independent sampling almost never finds two proofs of a hard theorem, so neither our proposed
  shortest-proof rule nor HTPS's minimal-proof rule has anything to act on without search. The open question
  is B2: does search-found data extend what the apprentice reaches without search?
- *Smallest test.* Six SN Stage-1 seeds (s0–s5 exist), each run through two 8-round T1 ladders from the same
  checkpoint: sampling EI vs best-first EI (α = 1, width 4) at an equal number of generated actions.
  Evaluate both apprentices *without search* at k = 256 on `transfer_long` rr600, generator-only bins. Run a
  third arm, truncate-and-resume (restart from the state before the failing step, DeepSeek-Prover-V1.5
  2408.08152v1 §3.1), on 2 seeds as the cheap baseline.
- *Expectation.* Best-first apprentice `L*` ≥ sampling apprentice + 1 in ≥ 5 of 6 pairs, and the mean
  `L_true` of found proofs is higher at α = 1 than at α = 0.
- *Falsifier.* The difference is ≤ 0 in ≥ 3 of 6 pairs.
- *Seeds.* `NOISE_FLOOR.md`'s nearest quantity is the frozen-ladder transfer `L*`: sd 0.354, MDD 2 points at
  n = 2 and 1.794 × 0.354 = 0.63 at n = 6, so a 1-point difference is resolvable only at n ≈ 6.
  *[reviewed: F3]* Pairing removes the Stage-1 draw, but the variance of two EI arms from one shared
  checkpoint is unmeasured (`NOISE_FLOOR.md`'s ≈ 99 % is run-to-run training variance, and each arm is a
  further training run), so pairing's gain is unknown until a same-checkpoint EI re-run pair is measured.
  `transfer_long` has no measured floor, so a re-sample of one arm at a second sampling seed is part of
  the run. The reviewer also notes that "≥ 5 of 6 pairs" is a sign test with one-sided p ≈ 0.11 (only 6 of 6
  reaches 0.05), that 4 of 6 is neither expectation nor falsifier, and that `AGENT_POLICY.md` asks for
  per-seed values plus IQM with a stratified-bootstrap 95 % CI; this applies to #2 and #3 too.

**2. Frontier target supply.**
- *Mechanism.* EI gets signal only where p̂ > 0. The fixed pool has almost none past `L*`, and reallocating
  attempts inside it (T2 / T4) did not help. New targets built by extending proofs the model already solves
  sit just past the frontier by construction.
- *Test.* Six SN seeds, T1 vs T1 + supply. Supply = 32 attempts per candidate at `L_true` `L*`+1 … +4; keep
  targets with p̂ ∈ (0, 1/4].
- *Expectation.* ≥ 5 % of candidates pass the filter, and `L*` rises by ≥ 1 in ≥ 5 of 6 pairs.
- *Falsifier.* Under 5 % pass the filter (supply fails, which is the memory note's "generator cannot make
  run-2 shapes" risk), or `L*` does not move. Same floor as #1.

**3. Support-expanding advantages.**
- *Mechanism.* GRPO shifts mass toward already-likely correct samples ("rank bias", He et al. Fig. 4). The
  unlikeliness penalty and the pass@k advantage move credit to rare correct proofs.
- *Caveats.*
  - Every variant except negative-only reinforcement (2506.01347v2) gives zero gradient on all-fail groups,
    so any gain on base-unreached theorems has to come from transfer.
  - No paper evaluates beyond k = 512.
  - He et al.'s "outperforms expert iteration" has no experiment behind it in either version.
  - `grpo.py` is whole-proof only.
- *Test.* Six new C0-style Stage-1 seeds (≈ 6.5 pod-min each). Arms: GRPO default, unlikeliness, pass@4
  advantage, and distinct-proof bonus. Read out with the `support-curves` protocol: base-unreached theorems
  reached at equal attempts, paired per theorem.
- *Expectation.* Unlikeliness and pass@k reach ≥ 1.2× the default's base-unreached theorems in ≥ 5 of 6
  seeds, at a held-out greedy cost ≤ 2 pp.
- *Falsifier.* Otherwise. Raw solve-count floors are 167–257 % at n = 2, so use paired per-theorem readouts.

**4. Transductive EI on textbook theorems.**
- *Why.* In the `long-pool` re-read, textbook theorems were solved 10 times in 73 × 14 theorem-model reads.
  HTPS's Identities result is this exact shift (generated statements → book identities).
- *Test.* Split the 282 textbook theorems in `transfer_long` in half. Arms: frozen at equal attempts; EI on
  half A; EI on half A + variants. Report half A (transductive) and half B (transfer) separately. 6 SN seeds.
- *Expectation.* The variants arm solves ≥ 2× EI-only on half A, and more than the frozen control on half B.
- *Falsifier.* Variants ≤ EI-only.

**5. Value token.** *[reviewed: F1]* The steps-to-go value head is already in proposal 14 item 1 and proposal 13; the reviewer would fold it into #1's write-up rather than list it separately.
- *Test.* Only after #1. Compare best-first scored by log-prob against log-prob + steps-to-go value, at equal
  expansions, on the 29 survivors and on bins 13–16.
- *Expectation.* ≥ 1.5× solves at `L_true` ≥ 14.
- *Falsifier.* Value ≤ no value, as with HTPS's hard targets.

**6. EI data selection.** Three arms on 6 SN seeds: shortest / least-likely / random proof per theorem.
- *Expectation.* The least-likely arm raises distinct base-unreached theorems reached.
- *Falsifier.* No pairwise difference beyond the re-sample spread.

## (b) Considered and rejected

| technique | source | reason |
|---|---|---|
| Learnability / p(1−p) target sampling, SEC bandit, PLR | 2502.12272v6 Alg. 2; 2505.14970v4 Table 1; 2010.03934v4 | Already done in substance: ladder-A T2 difficulty weighting and T4 window, with no frontier gain. At frontier p ≈ 1e-5, p(1−p) ≈ p. A uniform-over-bins control is still cheap. |
| Cross-seed pooling of EI data | Lee et al. 2502.01612v2 §5.1 | Done as ladder-A T6. |
| Learned conjecturer head | STP; Minimo | STP's conjecturer is a pretrained 7B LM. Minimo's from-scratch conjecturer found proofs of ≤ 11 steps (§5), below our `L*`. #2 gets the filter without training a generator. |
| Position coupling, Abacus, index hints, RASP-L formats, looped transformers | 2405.20671, 2405.17399, 2310.16028, 2409.15647 | Digit-alignment tricks. Our `n<k>` offset names already act as index hints. Looping needs the step count at training time (2409.15647v5 §7). |
| NoPE instead of RoPE | 2305.19466v2 | Cheap but low value: `L*` stalls in the state arms too, where context length is not the constraint. Keep only as a 6-seed Stage-1 side arm if a pod is idle. |
| Stream of Search, Searchformer | 2404.03683v1, 2402.14083v2 | Need long serialised traces (4,096-token contexts); the gain is over optimal-path imitation, which is not our bottleneck. |
| Self-correction from error messages (Goedel-V2, Seed-Prover, Kimina, Lean-STaR) | 2508.03613v1 §3.5; 2507.23726v2; 2504.11354v1; 2407.10040v5 | Need a pretrained LLM that reads Lean messages. Goedel-V2: removing compiler feedback "significantly lowers performance". |
| RMaxTS as a standalone arm | 2408.08152v1 Table 3 (58.4 → 59.6) | Test-time only, about 1 point, never used as the training expert. |
| DPO from compiler errors | BFS-Prover §2.3 | 70.38 → 70.83 is inside its ±0.89. |
| Entropy regularisers (Clip-Cov / KL-Cov, clip-higher, forking tokens) | 2505.22617v1; 2503.14476v2; 2506.01939v2 | Clip-higher is already proposed (DAPO). The others are tuned for long CoT at 7B–32B; #3 addresses the same collapse more directly. |
| First-error process reward | 2606.20068v1 ablation table: outcome + tactic 59.2 % vs outcome-only GRPO 57.9 % (MiniF2F pass@64, ± 0.5, 7B) | A +1.3-point gain at 7B, within about 2.6 σ of its own error bars, and far below any floor we could resolve. Our env already ends attempts at the first invalid step. |
| Online asynchronous HTPS / ABEL training | ABEL App. C Fig. 2 | Needs distributed prover and trainer infrastructure. The proposed "EI to 24 rounds" is the cheap proxy. |

## (c) Measuring create vs elicit

1. **Per-step base likelihood of newly reached proofs** (do first; $0–1, no sampling).
   - *What.* For each theorem EI reaches and its base did not in 400k attempts (`support-curves`, `support-state`),
     score the found proof under the base: total, minimum and mean per-step log-probability.
   - *How to read it.* Compounding reliability predicts no step near zero. New moves predict at least one step
     below the ε ≈ 3/400k ≈ 7.5e-6 bound. The bound follows Invisible Leash 2507.14843v4 App. C.4, with
     ζ = 0.05; the arithmetic is ours.
   - *Also report* its SRR / NDR (§4.2: all 1.5B–14B models "SRR ≈ 0.93–0.99", "NDR ≤ 0.04").
2. **Power-sampling null.**
   - *What.* Sample the base under p^α with Karan & Du's MCMC (2510.14901v1 Alg. 1) at matched tokens. Their
     Table 1 has it matching GRPO on MATH500 (0.748 vs 0.785) at 8.84× the tokens.
   - *How to read it.* If it recovers EI's new theorems, EI is sharpening. It cannot reach low-likelihood
     proofs (their Fig. 5), so read it together with item 1.
3. **Mode-share tracking per seed during EI and GRPO.** Echo Chamber (2504.07912v2) finds RL "amplifies a
   specific mode from the pretraining mixture while collapsing the others" in from-scratch models. Track
   the share of proofs by style (depth-3 pattern or not) round by round. This may explain the bimodal seeds.
4. **Textbook dose study for B4.** Interplay (2512.07783v1 §4) finds that RL transfers to a new context only
   when it made up ≥ 1 % of pre-training; at 0 % and 0.1 % it does not. Use Stage-1 mixes with 0 / 0.1 / 1 /
   10 % textbook-style theorems (≥ 6 seeds per dose), then identical EI.
5. **Init-seed vs data-order seed.** Zhou et al. 2402.09371v1 §4.2 find "significant variance across training
   data orders even when the weight initialization is constant". A 3 × 3 Stage-1 grid (≈ $1) would tell
   `NOISE_FLOOR.md`'s per-run component apart.

## Pre-registered expectations vs outcomes

| # | expected | outcome |
|---|---|---|
| 1 | search as the EI expert ranks first | **held** |
| 2 | learnability curriculum in the top 3 | **falsified**: already done as T2 / T4; *supply* ranks 2 instead |
| 3 | positional-encoding techniques rank low | **held** (rejected) |
| 4 | pass@k objectives shortlisted, below search | **held** (#3) |
| 5 | ≥ 1 of the brief's identifiers wrong | **falsified**: every arXiv id was right. Corrections were to metadata only: 2509.06941 authors are Song, Kempe, Munos; 2502.12272v6 is titled "LILO: …" in its PDF; AlphaGeometry and ABEL have no arXiv id |
| 6 | ≥ 90 % of cited claims verify | **held**: the agents' ledgers list ≈ 240 claims with 14 marked unverified or partial (§d notes). My independent re-check of 33 claims (list in §d): 32 verified. AlphaProof's Table 1 TTRL percentages were not in my text copy, so they are not cited |

## (d) Screened papers

58 papers, more than the brief's 25–40, because the cluster agents followed citations. The *claims verified*
column is the screening agent's own record. The claim ledgers themselves are in its notes and in `~/lr_out/`
(not committed).

**The executor's independent re-check** (`lit_review/quote.py`; V = verified verbatim at the location given):

| # | claim | location | status |
|---|---|---|---|
| 1 | BFS score Σ log p / L^α | 2502.03438v3 Eq. 1 | V |
| 2 | BFS-Prover 70.83 % ± 0.89 | 2502.03438v3 Table 1 | V |
| 3 | SFT 70.38 vs SFT + DPO 70.83 at 2,048 passes | 2502.03438v3 §3 | V |
| 4 | minimal proofs of all solved nodes 78.1 vs all proofs 40.6 | 2205.11491v1 Table 4 | V |
| 5 | "Learning only from the minimal proofs always leads to improved performance" | 2205.11491v1 §7.2.1 | V |
| 6 | critic soft 78.1 / none 65.6 / hard 63.1 | 2205.11491v1 Table 5 | V |
| 7 | Identities 91.3 % vs supervised never above 36 % | 2205.11491v1 §7.1.3 | V |
| 8 | BF mean found length 1.66 vs critic-guided 4.44 | 2410.15700v2 §3.2 | V |
| 9 | keep conjectures with P̂ ∈ (0, 1/4] | 2502.00212v4 §3.2 | V |
| 10 | 28.5 % vs 13.2 % "achieved through expert iteration" | 2502.00212v4 abstract | V |
| 11 | β_rank = 0.25; zero-advantage samples still skipped | 2506.02355v2 §4.1 | V |
| 12 | 8,065 / 9,600 (+358) | 2506.02355v2 Table 2 | V |
| 13 | Enigmata 12.9/21.3 → 17.9/29.8 | 2508.10751v1 Table 1 | V |
| 14 | `extract_goal` captures unsolved states | 2508.03613v1 §2.2 | V |
| 15 | SRR ≈ 0.93–0.99, NDR ≤ 0.04 | 2507.14843v4 §4.2 | V |
| 16 | unseen-in-k bound (1 − p)^k ≤ ζ | 2507.14843v4 App. C.4 | V (the 7.5e-6 figure is our arithmetic) |
| 17 | "programmatically created variants" | AlphaProof, Methods | V |
| 18 | reward −1 per tactic; value −T_steps | AlphaProof, Methods | V |
| 19 | "More synthetic variants consistently improves the final prove rate" | AlphaProof, ED Fig. 3b | V |
| 20 | "too difficult … is detrimental" | 2502.01612v2 §7.1 | V |
| 21 | RFT on Level 3 "never surpasses 2.6 %" | 2509.25123v3 §4.2 | V |
| 22 | power sampling 8.84× tokens | 2510.14901v1 §5.3 | V |
| 23 | MATH500 0.748 power vs 0.785 GRPO | 2510.14901v1 Table 1 | V |
| 24 | variance across data orders at fixed initialisation | 2402.09371v1 §4.2 | V |
| 25 | RL "amplifies a specific mode … while collapsing the others" | 2504.07912v2 conclusion | V |
| 26 | 0 % / 0.1 % exposure: no transfer; 1 % enhances it | 2512.07783v1 §4 | V |
| 27 | Minimo model ≈ 8.45 M parameters | 2407.00695v2 App. A | V |
| 28 | Minimo propositional longest proofs 5 → 11 steps | 2407.00695v2 §5 | V |
| 29 | single-pass 58.4 vs RMaxTS 59.6 at 4 × 6,400 | 2408.08152v1 Table 3 | V |
| 30 | Darling pass@128 +7.62 / 10.16 % | 2509.02534v1 §5 | V |
| 31 | TS-LLM π_θ1 47.9 vs RFT-50 47.0 / RFT-100 47.5 | 2309.17179v2 Table 4 | V |
| 32 | outcome + tactic 59.2 vs outcome-only 57.9 | 2606.20068v1 ablation table | V |
| 33 | AlphaProof formal-imo TTRL percentages | AlphaProof Table 1 | not found in my copy; not cited |

**Screened-paper table** (*[reviewed: F5]* rows 17, 20, 32 and 37 repaired)

| # | title | year | id | technique | rel. 0–3 | depth read | claims verified | cluster |
|---|---|---|---|---|---|---|---|---|
| 1 | Thinking Fast and Slow with Deep Learning and Tree Search (ExIt) | 2017 | 1705.08439v4 | MCTS expert + NN apprentice; tree-policy targets; policy+value | 3 | full | all (Fig. curves not digitised) | A |
| 2 | HyperTree Proof Search for Neural Theorem Proving (HTPS/Evariste) | 2022 | 2205.11491v1 | AND-OR PUCT search, soft critic token, minimal-proof training data, online training | 3 | full | all | A |
| 3 | DeepSeek-Prover-V1.5 | 2024 | 2408.08152v1 | GRPO on whole proofs (RLPAF); truncate-and-resume; RMaxTS (novelty reward + discounted UCB) | 3 | full | all | A |
| 4 | BFS-Prover | 2025 | 2502.03438v3 | length-normalised log-prob best-first; EI with beam filtering; DPO on Lean errors | 3 | full | all (no ablations exist for α/filter) | A |
| 5 | InternLM2.5-StepProver | 2024 | 2410.15700v2 | critic-guided best-first; preference-trained critic; 21k-CPU-day EI | 3 | full | all | A |
| 6 | Generative Language Modeling for Automated Theorem Proving (GPT-f) | 2020 | 2009.03393v1 | cumulative-logprob best-first; OUTCOME-token value; iterated policy+value | 2 | method + tables 5, 9, 11 | all | A |
| 7 | Formal Mathematics Statement Curriculum Learning | 2022 | 2202.01344v1 | EI with best-first + PROOFSIZE-bucket value; EI vs sample-only | 3 | method + results + discussion | partial (Table 1 garbled in PDF text) | A |
| 8 | Reinforcement Learning of Theorem Proving (rlCoP) | 2018 | 1805.07563v1 | UCT + XGBoost policy/value in a connection-tableau prover | 2 | method + tables 3–6, 8 | all | A |
| 9 | Stream of Search (SoS) | 2024 | 2404.03683v1 | train a 250M LM from scratch on serialized search traces with backtracking; STaR/APA | 1 | abstract + method + main results | all | A |
| 10 | Olympiad-level formal mathematical reasoning with RL (AlphaProof) [@Hubert_2025] | 2025 | DOI 10.1038/s41586-025-09833-y | AlphaZero-style search with −steps value, AND nodes, progressive sampling; TTRL on variants | 2 | full (Nature HTML, open access) | all quoted items | A |
| 11 | ABEL: Sample Efficient Online RL for Neural Theorem Proving (Gloeckle, Limperg, Synnaeve, Hayat) | 2024 | no arXiv; NeurIPS'24 MATH-AI workshop, OpenReview kk3mSjVCUO; PDF cermics.enpc.fr/~hayata/ABEL_pre.pdf | HTPS + online RL on only 244 miniF2F-valid statements | 2 | full (13 pp) | all quoted items | A |
| 12 | AlphaZero-like Tree-Search can Guide LLM Decoding and Training (TS-LLM) — found via citations | 2023 | 2309.17179v2 | one round of MCTS-generated data for policy distillation vs rejection-sampling FT | 2 | method + Table 4 | all | A |
| 13 | Learning Formal Mathematics From Intrinsic Motivation (Minimo) [@poesia2024learningformalmathematicsintrinsic] — found via citations | 2024 | 2407.00695v2 | 8.45M from-scratch LM; MCTS with policy+value tokens; hindsight relabelling | 3 | method + App. A/B | all quoted items | A, B |
| 14 | Beyond A*: Search Dynamics Bootstrapping (Searchformer) — found via citations | 2024 | 2402.14083v2 | train on A* traces, then fine-tune on shorter successful traces | 1 | abstract + method | all quoted items | A |
| 15 | STP: Self-play LLM Theorem Provers with Iterative Conjecturing and Proving | 2025 | 2502.00212v4 | One LLM acts as conjecturer and prover. The conjecturer is trained on conjectures with empirical pass rate in (0, 1/4], after an elegance filter and re-weighting toward the unproved statements | 3 | full (§1–4, App. A.2–A.6, B.1) | all (the equal-budget EI curve is figure-only) | B |
| 16 | Proving Theorems using Incremental Learning and Hindsight Experience Replay (Aygün et al.) | 2021 arXiv / ICML 2022 | 2112.10664v1; PMLR 162 aygun22a | HER on every intermediate clause, goals sampled with weights by size, positives and negatives; implicit curriculum via retries with doubling time budgets | 2 | method + results | all for arXiv v1; ICML version unverified | B |
| 17 | LILO: Learning to Reason at the Frontier of Learnability | 2025 | 2502.12272v6 | Rejection-sample prompts by p̂(1−p̂) and train on the top-\|B\| | 3 | method + results + App. D.1 | all | B |
| 18 | Prioritized Level Replay | 2020 | 2010.03934v4 | Replay levels by rank of a learning-potential score (L1 value loss) mixed with a staleness distribution | 2 | method + headline results | partial (Procgen numbers only as quoted) | B |
| 19 | Online Difficulty Filtering for Reasoning Oriented RL | 2025 | 2504.03380v2 | GRPO batches filled online with prompts whose pass rate lies in (T_low, T_high); compared with offline curation and scheduling | 3 | method + results | all quoted | B |
| 20 | Self-Evolving Curriculum for LLM Reasoning (SEC) | 2025 | 2505.14970v4 | Non-stationary bandit over difficulty levels, with reward = mean \|advantage\| | 3 | method + main results | all quoted; "equal steps" unverified | B |
| 21 | Goedel-Prover-V2 | 2025 | 2508.03613v1 | Scaffolded synthesis: unsolved goals from failed attempts (extract_goal) plus their negations; LLM-made easier or harder variants; RL keeps pass rate in (0, 0.75]; averaging with the base model | 2 | §2.2–2.4, §3.6, App. E | all quoted (self-correction not read) | B, E |
| 22 | Goedel-Prover | 2025 | 2502.07640v3 | Large autoformalised statement set plus 8 rounds of EI retrained from base; more statements and more formalisation styles help | 1 | §3, §5 | partial | B |
| 23 | Learning to Prove Theorems by Learning to Generate Theorems (MetaGen) | 2020 | 2002.07019v2 | Neural forward generator of Metamath theorems (random, IL, or RL with an adversarial "human-likeness" reward) that trains the prover | 2 | abstract + §5.1 Table 3 | partial | B |
| 24 | Training a First-Order Theorem Prover from Synthetic Data | 2021 | 2103.03798v2 | Forward proposer (random linear resolution biased toward small clauses), with N and T grid-searched to maximise E-prover difficulty | 1 | intro + §2 + §5 | partial | B |
| 25 | Solving olympiad geometry without human demonstrations (AlphaGeometry) | 2024 | DOI 10.1038/s41586-023-06747-5 (read via Europe PMC PMC10794143) | Random premises → exhaustive forward deduction → traceback to minimal proofs; premises irrelevant to the statement become "auxiliary constructions" | 2 | main text + Methods (grep) | partial | B |
| 26 | Evolving Curricula with Regret-Based Environment Design (ACCEL) | 2022 | 2203.01302v3 | Edit or mutate levels at the frontier (regret-based) | 1 | abstract | abstract only | B |
| 27 | Rewarding the Unlikely: Lifting GRPO Beyond Distribution Sharpening (He, Fried, Welleck) | 2025 | 2506.02355v2 | Multiplicative within-group rank penalty on correct samples (β_rank 0.25) + more PPO epochs; Lean prover | 3 | full | partial (Fig. 5 large-N gains are plot-only; "outperforms EI" claim has no experiment) | C |
| 28 | Pass@k Training for Adaptively Balancing Exploration and Exploitation of LRMs (Chen, Qin, Wu, Ling, Ye, Zhao, Shi) | 2025 | 2508.10751v1 | Closed-form pass@k group advantage from (N_pos, N_neg, k) | 3 | method + main results | all cited | C |
| 29 | Pass@K Policy Optimization (Walder, Karkhanis) | 2025 | 2505.15201v5 | Unbiased pass@k reward transform for any k ≤ n; anneal k_opt → 1 | 3 | method + results | all cited | C |
| 30 | The Entropy Mechanism of RL for Reasoning LMs (Cui et al.) | 2025 | 2505.22617v1 | Law R = −a·e^H + b; Clip-Cov / KL-Cov on high-covariance tokens | 2 | method + results | partial (32B Clip-Cov/KL-Cov row truncated in extraction) | C |
| 31 | The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning (Zhu, Xia, Wei, Chen, Chen, Meng) | 2025 | 2506.01347v2 | NSR only / W-REINFORCE (λ=0.1 on positives) | 3 | method + results | all cited | C |
| 32 | Understanding R1-Zero-Like Training (Dr. GRPO) | 2025 | 2503.20783v2 | Remove 1/\|o\| and std normalisation from GRPO | 1 | method | all cited | C |
| 33 | DAPO: An Open-Source LLM RL System at Scale | 2025 | 2503.14476v2 | Clip-higher (ε_low 0.2, ε_high 0.28), dynamic sampling, token-level loss, overlong shaping | 1 | method | all cited | C |
| 34 | Outcome-based Exploration for LLM Reasoning (Song, Kempe, Munos) | 2025 | 2509.06941v1 | Historical UCB bonus min{1, 1/√N(x,a)} over final answers (with constant baseline) and a batch-repetition penalty | 2 | method + key figs | partial (test gains figure-only) | C |
| 35 | Beyond the 80/20 Rule: High-Entropy Minority Tokens (Wang et al.) | 2025 | 2506.01939v2 | Policy gradient only on top-20% entropy ("forking") tokens | 1 | abstract + intro | all cited | C |
| 36 | Jointly Reinforcing Diversity and Quality in LM Generations (Darling; Li, Wang et al.) | 2025 | 2509.02534v1 | Reward = quality × Norm(fraction of group in other equivalence classes) | 3 | method + math results | partial (Table 4 column-block labels inferred) | C |
| 37 | Reasoning with Exploration: An Entropy Perspective (Cheng et al.) | 2025 | 2506.14758v4 | Advantage += min(α·H_detach, \|A\|/κ) | 1 | abstract + method | abstract only for results | C |
| 38 | The Invisible Leash: Why RLVR May or May Not Escape Its Origin (Wu et al.) | 2025 | 2507.14843v4 | Empirical-support accounting (preservation / expansion / shrinkage) at k up to 16384 | 2 (measurement, for Q3) | method + Table 1 | all cited | C, E |
| 39 | Self-Improving Transformers Overcome Easy-to-Hard and Length Generalization Challenges (N. Lee, Z. Cai, A. Schwarzschild, …, K. Lee, D. Papailiopoulos) | 2025 | 2502.01612v2 | Label slightly harder problems with the model's own greedy outputs, +1 difficulty per round, keep all rounds; filter by length or by a cross-seed majority vote | 3 | full (main text + App. B.2) | all (appendix hyperparameter tables not read) | D |
| 40 | From f(x) and g(x) to f(g(x)): LLMs Learn New Skills in RL by Composing Old Ones (L. Yuan, W. Chen, Y. Zhang, G. Cui, …) | 2025 | 2509.25123v3 | RL (DAPO) on depth-2 compositions of learned atomic skills generalises to depth 3–6; iterative RFT on the same data does not | 2 | full (§3–4, App. A) | all | D, E |
| 41 | Transformers Can Achieve Length Generalization But Not Robustly (Y. Zhou, U. Alon, X. Chen, X. Wang, R. Agarwal, …, D. Zhou) | 2024 | 2402.09371v1 | FIRE + randomized PE + reversed format + index hints give 40→100 digits; strong seed/data-order fragility | 3 (for B3) | full (§2–5) | all (appendix figures not inspected) | D |
| 42 | Looped Transformers for Length Generalization (Y. Fan, Y. Du, K. Ramchandran, K. Lee) | 2024 | 2409.15647v5 | Weight-tied block looped T(n) times, full-output prediction, step count supervised, adaptive stopping | 1 | full (§3–7) | all | D |
| 43 | Position Coupling: Improving Length Generalization of Arithmetic Transformers Using Task Structure (Cho et al.) | 2024 | 2405.20671v2 | Give "relevant" tokens (same-significance digits) the same position ID | 0–1 | abstract + §6 heads | partial | D |
| 44 | Transformers Can Do Arithmetic with the Right Embeddings (McLeish et al.) | 2024 | 2405.17399v2 | Abacus: per-digit position within a number, random offset U[1,k]; plus input injection and looping | 1 | abstract + method lines | partial | D |
| 45 | What Algorithms can Transformers Learn? A Study in Length Generalization (Zhou et al.) | 2023 | 2310.16028v1 | RASP-L conjecture; index hints; balanced (diverse) training distribution | 1–2 | abstract + §5.1 | partial | D |
| 46 | The Impact of Positional Encoding on Length Generalization in Transformers (Kazemnejad et al.) | 2023 | 2305.19466v2 | NoPE beats APE/T5/ALiBi/Rotary on decoder-only length generalisation (107M, 10 tasks incl. SCAN, PCFG) | 2 (Q4) | abstract + §3 setup | partial | D |
| 47 | Exploring Length Generalization in Large Language Models (Anil et al.) | 2022 | 2207.04901v2 | Fine-tuning fails at length generalisation regardless of scale; hyperparameters change OOD but not in-distribution; scratchpad plus few-shot helps | 1–2 (B3) | abstract + §3.3, §4 lines | partial | D |
| 48 | Transformers Struggle to Learn to Search (Saparov et al., ICLR 2025) | 2024 | 2412.04703v2 | Small transformers learn graph search only with a training distribution balanced over lookahead; converged fraction over seeds falls with graph size; RoPE does not change scaling | 2 | §3.1, §5, App. A.5 heading | partial | D |
| 49 | Faith and Fate: Limits of Transformers on Compositionality (Dziri et al.) | 2023 | 2305.18654v3 | Computation-graph view of compositional tasks; errors compound exponentially with depth; fine-tuned GPT-3 fails OOD on wider/deeper graphs | 1 | abstract + §3.1, §4 statements | partial | D |
| 50 | Process-Verified RL for Theorem Proving via Lean (Kim, Yun et al.) | 2026 | 2606.20068v1 | GRPO plus per-tactic reward from Lean elaboration, with first-error propagation and first-token credit | 3 | full (§3–5, Tables 2–5) | all | E |
| 51 | Reasoning with Sampling: Your Base Model is Smarter Than You Think (Karan & Du) | 2025 | 2510.14901v1 | MH sampling from the power distribution p^α of the base model; matches GRPO pass@1 with no training | 3 | full (§4–5) | all | E |
| 52 | Echo Chamber: RL Post-training Amplifies Behaviors Learned in Pretraining (Zhao et al.) | 2025 | 2504.07912v2 | 150M/1B from-scratch models; PPO/GRPO/EI collapse onto one pretraining format | 3 | method + §3 + App. F.2 | all (for cited items) | E |
| 53 | On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning LMs | 2025 | 2512.07783v1 | 100M from-scratch synthetic study: RL adds pass@128 only at the edge of competence; ≥1% seeds; process rewards | 3 | method + §3–6 + App. obs. 7 | all (for cited items) | E |
| 54 | RL Squeezes, SFT Expands (Matsutani et al.) | 2025 | 2509.21128v2 | counts unique correct/incorrect trajectory clusters (chrF + UPGMA): RL shrinks both, SFT expands correct | 2 | abstract + §3 | all (for cited items) | E |
| 55 | Seed-Prover | 2025 | 2507.23726v2 | lemma-style proving, lemma pool, refinement with Lean feedback plus self-summary | 1 | §2.2–2.2.4 | all (for cited items) | E |
| 56 | Kimina-Prover Preview | 2025 | 2504.11354v1 | 72B long-CoT RL, binary reward; no compiler-feedback refinement | 1 | §RL, conclusion | all (for cited items) | E |
| 57 | Lean-STaR | 2024 | 2407.10040v5 | GPT-4 thoughts interleaved with tactics, then expert iteration | 0 | grep of method | partial | E |
| 58 | Local Look-Ahead Guidance via Verifier-in-the-Loop for ATP (Rajaee et al.) | 2025 | 2503.09730v2 | step-level Lean feedback during training | 1 | abstract only (full text fetch returned HTTP 406) | none beyond abstract | E |
