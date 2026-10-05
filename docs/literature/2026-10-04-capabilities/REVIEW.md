---
written_on: 2026-10-05
written_by: agent:claude
---

> **Reading copy (librarian, 2026-10-05).** Run `capability-defs`'s `capability_defs/lit/REVIEW.md` (fork branch
> `dan_capability-defs`, DONE `645aa4f0`), with the screened table, the 57 in-depth notes, the six readers' claim
> ledgers (`_claims_L*.md`), their screening tables (`_screen_L*.md`), the executor's re-check and the reader
> instructions. The quote checker (`quote.py`, `fetch.py`) stays on the fork. The reviewer's literature sub-audit
> (`review_cd/lit/LIT_AUDIT.md` on the fork) finds **L1 met** and L2 supported. It re-verified 48 / 48 sampled claims
> verbatim, and an automated scan of all 1,180 V rows found one wording change. Its corrections, marked *[review]*
> below or listed here:
> - 185 unique papers counts row 105 (the AIJ version of row 104), which was never read: 184 without it.
> - `screened.md` row 76 says 6 / 6 claims; the ledger has 5. Row 78's escaped `|` splits its cell (fixed here).
> - Two of the three UNVERIFIED ledger rows are placeholders, not claims (`_claims_L2.md` #153, `_claims_L4.md` #80).
> - `_executor_recheck.md` reproduces only with an undocumented "status V" filter.
> - `_claims_L1.md` #149 quotes "stumble in the dark"; the source says "stumbles".
> - Below, three phrases printed as quotations were not verbatim, and one section reference was wrong (marked).
> - Two notes alter paper wording inside quotation marks (`vanderweij2024sandbagging.md`, `voita2020mdlprobing.md`;
>   marked).
> - L2's "a population" (IRT, a game database) and "a difficulty scale" go beyond the pre-registered wording.
>
> Run summary: [experiment-summaries/2026-10-05-capability-defs-…](../../../experiment-summaries/2026-10-05-capability-defs-compute-matched-budget-shrinks-equal-k-creations-rl-still-ahead-lem-teachable/README.md);
> review: [`review_capability-defs.md`](https://github.com/dan-pandori/nd-takehome/blob/da11be78/review_capability-defs.md) § R16. Citekeys in `notes/` are the readers' proposals and are
> not yet in `references.bib` (see `to_add_to_zotero.md`).

# Literature review 3: what is a "capability", and how is a new one told apart from better access to an old one? (run `capability-defs`, 2026-10-05)

Executor: agent:claude. Readers: six subagents (L1–L6), one thread each, reading in parallel (≤ 3 at a time, downloads one at
a time). Instructions: `AGENT_INSTRUCTIONS.md`. Earlier reviews this one extends: nd-rl `docs/literature/2026-09-29-lit-review/`
(58 papers) and `2026-10-02-lit-review-2/` (51 papers). Their 109 papers were excluded from screening
(`_prior_screened.tsv`); they are cited here as "earlier review".

**Counts.**
- 191 screened rows, of which 6 are duplicates across readers: **185 unique papers new to the earlier reviews**
  (`screened.md`).
- **57 read in depth** (`notes/`, TEMPLATE format).
- **1,183 claims** in the readers' ledgers (`_claims_L1.md` … `_claims_L6.md`). Each was checked with `quote.py`
  against the fetched source text, with its location recorded. **3 of 1,183 rows (0.25 %) carry UNVERIFIED** in the status column (no text access:
  Hernández-Orallo's book, the AIJ version of Martínez-Plumed et al., Move 37's prior).
- My independent re-check of 32 randomly drawn claims (`_executor_recheck.md`): **32 / 32 verified** at the stated
  location.

## Bottom line

1. **No field has a k-free, budget-free definition of "new capability".** Every quantitative definition found needs
   one of four anchors:
   - a **budget** (k, compute, fine-tuning data, or "1 % of training cost");
   - a **reference model** (base, null, or a never-had-it control);
   - a **population** (IRT calibration, a database of games);
   - a **difficulty scale.**

   The pre-registered expectation L2 holds. The principled budgets found are tied to compute: the safety-evaluation
   convention of "< 1 % of training cost"; the compute-equivalent gain; the RL query barrier.
2. **Sampling-based capability is coverage at a budget, and the budget is the whole question.** "With unlimited
   samples, any model that assigns a non-zero probability to every sequence will achieve perfect coverage" (Brown et
   al. 2024, Sec. 1). The field's response is to report the single-attempt success probability p (Schaeffer et al.
   2025) and to extrapolate its distribution (Kazdan et al. 2025). No per-problem unbiased pass@k exists beyond the
   sampled n.
3. **Likelihood and sampling meet in rare-event estimation.** The probability of solving is a sum over correct
   outputs.
   - A single output's probability is a lower bound on it.
   - Importance sampling with a stronger policy as the proposal is unbiased for it (Wu & Hilton 2024, Sec. 3.1).
   - Our "sum over known proofs" sits between the two (card `marginal-bracket`): a lower bound in exact arithmetic,
     in practice an estimate (the scorer and the sampler condition on different names) that recovers a median
     0.93–0.98 of the measured p (REPORT §3.1).
4. **The safety-evaluation literature defines capability as the best elicitation within a budget and contrasts it
   with propensity (default behaviour).**
   - "Capability evaluations are about whether an AI system *can* do some task … alignment evaluations are about
     whether an AI system *would* do some task" (van der Weij et al. 2024, Sec. 2).
   - Fine-tuning is the strongest elicitation: "fine-tuning on a single demonstration is often enough to recover most
     of the 'hidden' performance" (Greenblatt et al. 2024, Sec. 1).
5. **The most defensible way to say "it was already there" uses a control that lacks the capability, not a raw
   budget.**
   - Deeb & Roger 2024 reject relearning budgets for want of "a reliable baseline for comparison" and use a T / V
     split with a never-trained control (Sec. 3.2; *[review]* the quoted phrase is in Sec. 2).
   - Hewitt & Liang 2019 replace probe accuracy with selectivity over a control task.
   - Jain et al. 2023 compare revival speed with a never-had-it model.
6. **RL theory gives an exact form of "pure elicitation".**
   - With a binary reward, the KL-regularised RL optimum is the base model conditioned on success (Korbak et al. 2022,
     Sec. 5). It preserves the base's odds between correct outputs.
   - KL(q ‖ π_B) splits into the selection cost log 1 / p_B and a reshaping term (Shenfeld et al. 2025, App. A).
   - Outcome-reward RL needs about 1 / (base success quantile) queries (Mousavi-Hosseini & Erdogdu 2026). Theorems RL
     solves far below that rate are not explained by sharpening.
7. **Philosophy agrees on reliability:** a lucky success is not an ability. Kenny's darts player (SEP "Abilities",
   Sec. 4.3); Harding & Sharadin's conditional analysis, "a machine learning model has a capability to X just when it
   would reliably succeed at doing X if it 'tried'" (abstract). This settles Dan's random-weights objection
   conceptually: random weights never succeed *reliably*.
8. **Latent-variable measurement (IRT, observational scaling, time horizons) defines "more of the same" precisely:**
   calibrate items on one population and test whether another population's successes are predicted (specific
   objectivity). Departures are differential item functioning. This is the closest the literature comes to a test of
   "RL moves the model along the pretraining axis vs off it".
9. **Composition and novelty research measures new relative to training data** (compound divergence, Skill-Mix's
   counting bound, novelty against a game database). That is different from new relative to the base model. Both are
   needed.
10. **Metric choice creates apparent emergence** (Schaeffer et al. 2023). Any "jump" in a thresholded capability must
    be checked against a continuous one (p, log p, θ).

## 1. Sampling-based definitions (L1)

- **Estimator.**
  - Chen et al. 2021's unbiased pass@k, 1 − C(n − c, k) / C(n, k) (Sec. 2.1, Eq. 1).
  - The plug-in 1 − (1 − p̂)^k "results in a consistent underestimate" (App. A).
  - Chen's estimators are "only defined when the number of samples taken for each problem is greater than or equal to
    the number of attempts k" (Kazdan et al. 2025, Sec. 3.1).
- **Scaling with k.**
  - Coverage grows over four orders of magnitude of samples (Brown et al. 2024, abstract). Models in a family differ
    by a constant multiplicative factor in k (Sec. 3.2).
  - The aggregate power law arises iff the distribution of per-problem p has a power-law left tail (Schaeffer et al.
    2025, Thm 3.1). Fitting that tail to the share of problems in (0, 1/n) forecasts the exponent "with an order of
    magnitude lower relative error" (Sec. 5).
  - Kazdan et al. fit a beta-binomial to raw counts (Sec. 4.1). A beta cannot tell "impossible" from "hard", which
    biases forecasts upward (App. D.2).
- **Resolution.** PassUntil samples until r successes. PU = r / K is the MLE and is "designed to be applicable when a
  random baseline achieves P(s)=0" (Hu et al. 2023, Sec. 4.1). It turns emergence into a continuous measurement.
- **Rare events.**
  - Gumbel-tail extrapolation of per-query probabilities forecast deployment-scale risk across up to three orders of
    magnitude (Jones et al. 2025, Sec. 3.3).
  - Importance sampling estimates probabilities "too small to estimate by random sampling" (Wu & Hilton 2024,
    abstract).
- **Compute equivalence.**
  - Davidson et al. 2023: CEG = "how much additional training compute would be needed to improve performance by the
    same amount as the enhancement" (abstract). When no amount of scaling reaches the gain, "the CEG is not
    meaningful" (Sec. 4), the closest the field comes to "new".
  - Hilton et al. 2023: intrinsic performance.
  - Jones 2021: "for each additional 10× of train-time compute, about 15× of test-time compute can be eliminated"
    (Sec. IV-C).
- **Gold-reference likelihood as a covariate.** Schaeffer et al. 2025b fit pass@k as a function of the log likelihood
  of gold reference solutions across ≈ 5 orders of magnitude, and leave open whether it holds "under heavy
  optimization pressure" (Sec. 8). That is our RL case.
- **For us** (cards `passk-budget`, `passk-equal-k`, `marginal-bracket`):
  - report p, not "solved at k";
  - make the budget compute-matched on the theorems being judged;
  - extrapolate only at the set level;
  - bracket single theorems with likelihood (lower) and sampling (upper).

## 2. Elicitation practice, capability vs propensity, philosophy (L2)

- **Definitions in safety evaluation.**
  - Greenblatt et al.: "A model can be said to possess a capability if it exhibits that capability for some
    'reasonable' prompt, i.e. one that doesn't itself effectively provide the capability in question" (Sec. 2).
  - van der Weij et al. define actual capability as "the best performance … given the best currently available
    capability elicitation techniques such as prompting and fine-tuning", with the 1 %-of-training-cost convention
    quoted from Anthropic's RSP (Sec. 2).
  - Hofstätter et al.: "A latent capability … is one that a model exhibits with low probability by default, but can
    be revealed through small changes to the context, activations, or weights" (Sec. 3.3). They recommend fine-tuning
    access for evaluators (Sec. 5).
  - Shevlane et al.: capability vs propensity (abstract, Sec. 1).
- **How much elicitation is allowed?** A convention (1 % of training cost), justified by nothing in the texts read
  (L2: the RSP gives no justification), or by **controls**.
  - Models locked from scratch at 18 M parameters still unlock with a few demonstrations, needing 2–4× the data of
    fine-tune-locked ones (Greenblatt et al. Sec. 7).
  - Deeb & Roger's control model stays at 31.2 % (App. E).
  - Phuong et al. count "expert bits": the help a model needs before it succeeds (Sec. 6.2).
- **Philosophy and measurement theory.**
  - The conditional analysis "(CA) S has the ability to A iff S would A if S tried to A" (SEP "Abilities", Sec. 3.1);
    success-proportion views (Jaster, Sec. 3.4); *[review]* mere possibility is not sufficient
    (not a verbatim quotation; the verified sentence, "this sort of possibility is a sufficient condition", L2 #100, is
    in Sec. 4.1, not 4.3).
  - Harding & Sharadin adapt this to ML (CAMA). They treat decoders and scaffolds as background conditions (Sec. 2.3),
    and say a fine-tuning budget is "a natural proxy for the difficulty of eliciting the capability" (Sec. 5.2.2).
  - Firestone 2020: performance constraints mask competence, so a fair comparison accommodates them.
  - Burden et al. 2023: capabilities are latent levels in a measurement layout. σ(capability − demand) "allows us to
    interpret a capability with value x as consistently succeeding on the demand with meta-feature x in 50 % of
    instances" (Sec. 6).
  - Hernández-Orallo: abilities are constructs, tasks are instruments (1408.6908 Sec. 3.1). Capability is the area
    under the success-vs-difficulty curve (2021).
- **For us** (cards `capability-vs-propensity`, `reliability`, `elicit-finetune`):
  - the pair (capability within budget, propensity) is the central reporting format;
  - the never-had-it control (J6) is what makes an elicitation budget meaningful.

## 3. Information, learnability, likelihood (L3)

- **Description length.**
  - Online / prequential code length measures what a fit had to learn (Blier & Ollivier 2018; Voita & Titov 2020,
    with random-label controls).
  - EDL (earlier review) predicts that elicitation is cheap in bits and teaching expensive.
  - Surplus description length and ε-sample complexity give *[review]* "the number of samples required to reach
    that loss tolerance" (Whitney et al. 2020, Sec. 4; the earlier quotation was not verbatim).
- **Usable information.**
  - V-information is relative to a computationally bounded family. It "can be created through computation" (Xu et al.
    2020), unlike Shannon information.
  - Pointwise V-information gives per-instance difficulty as a log-probability gain over a null input (Ethayarajh et
    al. 2022, Sec. 3, Eq. 4).
- **Size of the update.**
  - Intrinsic dimension (d90) falls over pretraining (Aghajanyan et al. 2020).
  - RL changes "small subnetworks" (Mukherjee et al. 2025), but the sparsity disappears in fp32 (Shenfeld et al.
    Sec. 6).
  - URIAL: "77.7% of the tokens are at such unshifted positions" in aligned vs base models (Lin et al. 2023, Sec. 2.2).
- **RL's Razor.** On-policy RL "is implicitly biased towards KL-minimal solutions" (Shenfeld et al. 2025, abstract).
  The decomposition KL(q ‖ π_B) = log 1 / p_B + KL(q ‖ π_B(· | success)) (App. A) separates selection from reshaping.
- **For us** (cards `sharpen-expand`, `kl-update-size`, `elicit-finetune`): the selection / reshaping split is the most
  useful single idea from this thread.

## 4. Psychometrics (L4)

- **Rasch / 2PL.**
  - Specific objectivity defines "more of the same" (Wright & Linacre).
  - Calibrate on one population, test on a shifted one: tinyBenchmarks (Polo et al. 2024, Sec. 4.2–4.4); 2PL MAP
    abilities for 61–94 pretraining checkpoints (Hofmann et al. 2025).
  - IRT over repeated sampling, with pass@k as a function of θ (Truong et al. 2026, Eq. 4).
  - Observational scaling laws: capabilities are low-dimensional, measured in f-equivalent FLOPs (Ruan et al. 2024).
- **Horizons.**
  - The 50 % time horizon (Kwa et al. 2025).
  - Ability as the demand level at 50 % success on 18 scales (ADeLe, Zhou et al. 2025).
- **Floors and DIF.**
  - Maximum likelihood gives −∞ for a model that solves nothing. The fix is anchor items or concurrent calibration
    (METR App. B.1.3; L4 notes).
  - DIF flags a secondary trait. L4 suggests placebo DIF on pretraining segments and replay-only ladders: we ran this
    (card `irt-ability`).
- **For us:** the IRT card, with its replay-only placebo, is the set-level test "is RL more pretraining?".

## 5. RL theory and composition (L5)

- **KL-RL as Bayesian inference.** The optimum is π_B · exp(r / β) / Z (Korbak et al. 2022, Sec. 4, Eq. 5). With a
  0 / 1 reward, it conditions on success and keeps "all other strings … the original probability π0(x) up to Z"
  (Sec. 5).
- **Coverage.**
  - Best-of-N's reach is bounded by the coverage coefficient (Huang et al. 2025, Sec. 2.1, Eq. 5).
  - XPO adds exploration beyond the reference model; on-policy methods stay suboptimal at low coverage (Xie et al.
    2024, Prop. 2.1).
  - Mousavi-Hosseini & Erdogdu 2026: outcome-reward RL needs ≈ 1 / Q_q(ε) queries, the base's likelihood quantile.
    Process rewards are limited by the worst-token quantile, our worst step w1.
- **Composition.**
  - Systematicity, productivity, substitutivity, localism, overgeneralisation, with a consistency score (Hupkes et al.
    2020).
  - Compound divergence at fixed atom divergence (Keysers et al. 2020).
  - Skill-Mix's counting bound against the corpus (Yu et al. 2023).
  - Multiplicative emergence of composites, P(n) = (1 − (1 − p)^t)^n (Okawa et al. 2023).
  - Counterfactual tasks separate reasoning from reciting (Wu et al. 2023).
- **For us** (cards `sharpen-expand`, `chain-reachability`, `composition`, `transfer-invariance`, `schema-acquisition`).

## 6. Mechanistic and games (L6)

- **Probes and fine-tuning analyses.**
  - Probes need control tasks: a random network matched layer-2 accuracy (96.3 vs 96.6) but not selectivity (20.6 vs
    31.4) (Hewitt & Liang 2019, Table 2).
  - Fine-tuning often adds "wrappers" over existing capabilities. They revive in 0.1–3K iterations vs 4.5K for a
    never-had-it control (Jain et al. 2023, Fig. 9).
  - Fine-tuning enhances an existing entity-tracking circuit (Prakash et al. 2024, Sec. 6).
- **Steering.**
  - Steering vectors computed in the base recover ≈ 76 % of the RL / thinking-model gap (Venhoff et al. 2025, v4; v1's
    "up to 91 %" was walked back).
  - A 2 × 2 steering design separates elicited / repurposed / created (Ward et al. 2025, Sec. 3.2).
- **Diffing.** Crosscoder "model-only" features are mostly artefacts without the Latent Scaling check (Minder et al.
  2025, Sec. 3.1). *[review]* That finding is for L1-loss crosscoders; the cited ledger row (L6 #134) also says
  "most BatchTopK chat-only latents are genuinely chat-specific".
- **Games.**
  - Novelty is measured against a dated database of human games, together with quality (Shin et al. 2023).
  - AlphaGo Zero predicts human moves *worse* than the supervised network while being far stronger (Silver et al.
    2017, Extended Data Table 1).
  - The often-quoted "1 in 10,000" prior of Move 37 is not in the Nature paper; it is UNVERIFIED as a policy-network
    number.
- **For us** (card `latent-probe-steer`, not computed here).

## Pre-registered expectations vs outcomes (Part 1; `preregistration/capability-defs.md`)

| # | expected | outcome |
|---|---|---|
| L1 | ≥ 60 new papers screened, ≥ 20 in depth; ≥ 90 % of claims verified; my re-check of ≥ 30 finds ≤ 2 errors | **held:** 185 / 57; 1,180 of 1,183 ledger claims (99.75 %) verified; 32 / 32 in my re-check |
| L2 | no quantitative definition separates creation from elicitation without a budget, threshold or reference model; the principled budgets are tied to compute | **held** (Bottom line 1). The closest exceptions are controls (Deeb & Roger) and theory (Korbak), each of which still needs a reference model |
| L3 | critics break ≥ 1/3 of the cards | **held: 20 / 20** cards had a failure I accepted (REPORT §2, §3.9) |

## Files

- `screened.md`: merged table.
- `notes/*.md`: 57 notes.
- `_claims_L*.md`: ledgers.
- `_executor_recheck.md`.
- `to_add_to_zotero.md`.
- `fetch.py` / `quote.py`: tools; full texts stay in `~/cd_sources/`, outside git.
