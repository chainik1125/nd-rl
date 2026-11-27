---
written_on: 2026-11-27
written_by: agent:claude (auditor for M3 steer, Claude Opus 5.5)
papers: [heo2025instructionfollowingdimension, venhoffBaseModelsKnow2025, ward2025reasoningfinetuningrepurposes]
reading_level: full text of all three PDFs, every page incl. appendices (Heo 19/19 pages, arXiv 2410.14516v5; Venhoff 45/45 pages, arXiv 2510.07364v4; Ward 8/8 pages, arXiv 2507.12638v1)
---

## Auditor (Claude Opus 5.5), independent read, 2026-11-27 01:47 PST

Written before reading the reader-implementer's notes, the discourse files or the 2026-11-25 review, as instructed.

### What I read and how

| key | arXiv version (from the PDF stamp) | pages read | PDF md5 | my extraction |
|---|---|---|---|---|
| `heo2025instructionfollowingdimension` | 2410.14516v5, 28 Mar 2025 (ICLR 2025) | 1-19 of 19 | 9ee3532a2ed38d5a5eb3dccfaebffdb7 | `/scratch/cowifornia/nd-rl/olmo_elicit/metrics2/audit/heo2025instructionfollowingdimension.txt` |
| `venhoffBaseModelsKnow2025` | 2510.07364v4, 7 Jul 2026 (ICML 2026 header) | 1-45 of 45 | 77933ccd6dfc34faaf3e3a65c66c49ce | `/scratch/cowifornia/nd-rl/olmo_elicit/metrics2/audit/venhoffBaseModelsKnow2025.txt` |
| `ward2025reasoningfinetuningrepurposes` | 2507.12638v1, 16 Jul 2025 (ICML 2025 workshop) | 1-8 of 8 | 14497e34af43623066ba381489ab8ed0 | `/scratch/cowifornia/nd-rl/olmo_elicit/metrics2/audit/ward2025reasoningfinetuningrepurposes.txt` |

Extracted with the CONTEXT.md command (`tools/extract_pdf.py`). Limits of my reading: values that exist only
inside figures (Heo Fig. 4; Venhoff Fig. 3; Ward Figs. 2-5, B.1) cannot be read reliably from extracted text, so I
cite those figures only for what their captions and body text say. I did not download or read the TeX sources.
Venhoff v4 is a 2026 revision; earlier arXiv versions may report different numbers, and I did not read them. Where I
compute a number from a table or state an inference about model structure, I mark it **(my computation)** or
**(my inference)**.

---

### Q1. Construction: how each vector is built and applied

**Heo et al. (probe direction, "representation engineering", §2.2, §3.1, pp. 4, 6-7).**

- *What the vector is.* Not a mean difference. It is the weight vector of a supervised linear probe trained to predict
  instruction-following success: "A simple linear model was trained on instruction-following success outcome,
  optimized for 1000 epochs with AdamW, a 0.001 learning rate, and 0.1 weight decay" (p. 4). For steering: "The
  direction D is the weight of a linear probes trained on all IFEval-simple dataset" (p. 6). Footnote 3 (p. 6): a
  70/30 split was tried and "The results were similar but slightly worse than when the linear probe was trained and RE
  was applied to the entire dataset". So the reported steering is in-sample.
- *Positions.* "first token, LLM(x1, x2, . . . , xn), where xi are the n tokens in the input prompt" (p. 4), i.e. the
  last prompt position before the response, despite the name. Steering: "We focused on the representation of the
  first token, which corresponds to the input embedding before any response generation" (p. 6).
- *Layer.* "This adjustment was applied to the representations in the last layer of the model, as it was more robust
  to variations in α" (p. 6). Probe layers were 14/26/32 (16/32/40 for the 13B) (p. 4), so "last layer" is layer 32
  (40), the final block.
- *Formula and normalisation.* "Rupdated = Roriginal + α × D, where α is a scaling hyper-parameter" (p. 6). No
  normalisation of D is stated, so α is in units of an unreported probe-weight norm.
- *Strength and how chosen.* "We selected α for each model and instruction type using a validation set comprising 10%
  of the instruction data. The selected α values were: 0.3 for Llama-2-chat-13b and Llama-2-chat-7b, 0.1 for Phi-3,
  and 0.15 for Mistral-7B" (p. 7). The text says per model *and* instruction type but reports one value per model, and
  the 10 % validation set lies inside the data the probe was trained on and the steering was scored on.
- *Applied to which prompts.* "We applied this transformation to all input representations, including both success
  and failure cases" (p. 6).
- *Repeated at every generated token?* Not stated. The wording describes one position (the pre-response position) at
  the final layer. **(My inference)** If taken literally, a change to the final block's output at one position can
  only change the logits of the first generated token (no later layer reads it), so with greedy decoding the whole
  effect would run through the first sampled token. If instead it was added at every position after the final block,
  it would be close to a fixed token-level logit bias. Either way the "last layer" choice makes this a much weaker
  analogue of a mid-layer residual intervention than the paper's language suggests.
- *Decoding.* "To avoid randomness in decoding, we employed greedy decoding without sampling" (p. 4).
- *Random-direction control.* "we compared it against random directions" (p. 7). How these were drawn (distribution,
  norm, number) is not stated; Table 4's random SR has a standard deviation "across three runs with different random
  seeds" (p. 6).

**Venhoff et al. (category vectors in a gated "hybrid", §3.3-3.6, App. E.1-E.3, pp. 5-6, 19-20).**

- *What the vectors are.* Trained, not differenced: "Jointly optimize a per-category vector and a small MLP that
  predicts a steering coefficient, minimizing cross-entropy on the thinking model's next token at disagreement
  positions" (p. 5). Parameters: "a matrix of per-category vectors V ∈ R^{K×d}" and an MLP "(Linear(d, 512) → GELU)
  followed by K per-category heads (Linear(512, 1) → Softplus)", "the applied shift at position p with category c is
  α(hp, c) · Vc, where hp is the base model's residual-stream activation at the steering layer" (p. 19). AdamW, lr
  1e-2 (vectors) and 1e-3 (MLP), weight decay 0.01, up to 10 epochs, batch 4, early stopping on validation
  cross-entropy, each category vector kept at its own best epoch (p. 19).
- *Training positions.* Thinking-model rollouts (T 0.6, top-p 0.95, max 2,048 tokens; p. 19) are teacher-forced
  through the base model; positions where the base model's greedy prediction differs from the rollout's next token are
  kept, labelled by the argmax SAE latent of the *thinking* model's hidden state at the SAE layer, capped at 64 per
  (example, category) (p. 19).
- *Layer.* "We optimize category vectors at approximately 37% of model depth, which prior work suggests is most causal
  for behavior modification" (p. 5). The SAE layers chosen per model (App. G: e.g. ORZ-1.5B layer 4, ORZ-7B layer 20,
  ORZ-32B layer 27, R1-Distill-14B layer 38) are different from the steering layer, and the steering layer per model
  is not tabulated.
- *Normalisation and strength.* No normalisation of V; per-position strength comes from the MLP. Reported coefficients
  (Table 5, p. 22) range from 0.25 ± 0.16 (R1-Distill-8B) to 11.28 ± 8.47 (ORZ-32B, GSM8K); vector norms are not
  reported, so the actual shift size relative to the residual stream is unknown.
- *Applied where and when.* Only at disagreement positions during generation, which needs the thinking model running
  alongside: "First, we check whether the base- and the thinking model disagree in their next token prediction. If
  not we generate the jointly predicted tokens. If they disagree, then we obtain the SAE activations at that token
  position and select the highest activating category" (p. 5); "During generation, we compute SAE activations at each
  disagreement position and apply the selected category vector" (p. 6). Steered fraction 4.9-12.2 % of tokens for the
  RL pairs, 12-24 % for the distilled pairs (Table 2, p. 7).
- *Seeds and selection.* Three seeds per pair; the set with the highest gap recovered on a 788-question "holdout mix"
  (gold-answer subset of the training-mix validation split) is used for all evaluations (pp. 5, 19-20). Selection does
  not touch the evaluation benchmarks.
- *Decoding.* Base and hybrid greedy; thinking models 3 rollouts at T 0.6; 2,048-token budget (Table 1 caption, p. 7).
- *Prompts.* "base models receive a simple question–answer completion prompt with no chat template applied" (p. 20);
  thinking models get their chat template plus "Please reason step by step, and put your final answer within
  \boxed{}" (p. 20).

**Ward et al. (difference of means, "base-derived" vector, §2.1, §3, App. A, pp. 2-3, 7).**

- *What the vector is.* Difference of means (Eq. 3, p. 7):
  MeanAct(D) := (1/|D|) Σ_i (1/|S_i|) Σ_{s∈S_i} A(p_i)[s], and "v = MeanAct(D+) − MeanAct(D)", where D+ are traces
  with a GPT-4o-labelled backtracking event and D all traces. Two points matter for us: positions are averaged within
  each prompt first and then across prompts; and the subtraction is "positives minus all", not positives minus
  negatives.
- *Positions.* A negative offset before the backtracking sentence: "the optimal offset for 10th-layer residual stream
  of DeepSeek-R1-Distill-Llama-8B is ∼−13 to −8" (p. 2), chosen from the sweep in Fig. 2 ("We fix this offset for the
  remainder of our analysis", p. 2).
- *Which model's activations.* "base-derived" vectors are the base model's (Llama-3.1-8B) residual activations on the
  *reasoning model's* traces: "we derive steering vectors separately on residual stream activations from both the base
  and reasoning models on the same reasoning traces" (p. 2).
- *Layer.* Layer 10 of 32, after a sweep over all 32 layers: "both base-derived and reasoning-derived steering vectors
  are most effective around layer 10" (p. 3; Fig. B.1).
- *Normalisation and strength.* Not stated whether v is unit-normalised; "Steering Magnitude" runs 0-20 (Fig. 3),
  0-15 (Fig. 2), 0-25 (Fig. 4). Strength is shown as a sweep, not chosen on separate data.
- *Repeated at every generated token?* "examine the effect these vectors have when added to each models' activations
  during generation" (p. 3). Every position is implied but not stated; whether prompt positions are steered too is not
  stated.
- *Data.* 300 prompts in 10 categories generated by Claude Sonnet 3.7; traces from R1-Distill-Llama-8B; GPT-4o sentence
  labels (p. 7).

None of the three papers builds what M3 proposes, a **model-minus-model** mean activation difference on shared inputs.
Ward is the closest in arithmetic (difference of means), Venhoff in purpose (rebuilding a base-to-fine-tuned gap by
steering the earlier model), Heo in benchmark (IFEval-type verifiable instructions).

---

### Q2. Evidence: models, benchmarks, samples, statistics, and what supports each claim

**Heo.**

- Models: LLaMA-2-7B-chat, LLaMA-2-13B-chat, Mistral-7B-Instruct-v0.3, Phi-3-mini-128k-instruct (p. 3). All are
  instruct/chat models; no base, RL or distilled contrast.
- Data: IFEval-simple, 5 instruction types × the same 100 GPT-4-written tasks (p. 3; Table 10's success fractions such
  as 0.79412 = 81/102 suggest about 102 tasks per type **(my computation)**); IFEval-simple-expanded, 23 types × 20
  tasks (App. A.3; §2.3 says 25 types, A.3 says 23). Greedy decoding throughout. Success-rate tables use the IFEval
  "loose" metric (App. A.4, p. 17); Table 4 says only "predefined evaluation functions from the IFEval" (p. 6).
- Statistics: probe AUROC ± sd over five split seeds (Table 1); steering SR ± sd over "three runs with different random
  seeds" (Table 4), where the original SR has ± 0.00 because decoding is greedy, so the seed is presumably the probe or
  random-direction seed (not stated). No confidence intervals or tests.
- Claims and support:
  - Probe predicts success on unseen tasks: Tables 1-2 (first-token AUROC 0.74-0.88 at the early layer, 0.65-0.78 at
    the last layer, the layer later used for steering).
  - No generalisation across instruction types: Tables 1, 3, 7, 8 (AUROC around 0.5, several cells far below 0.5,
    e.g. Phi-3 "detectable content" 0.13/0.11/0.10 and LLaMA-13B "language" 0.12, Table 7).
  - Steering raises success over original and random: Table 4. Original → probe-direction SR: 0.57 → 0.59 (7B),
    0.61 → 0.65 (13B), 0.58 → 0.64 (Mistral), 0.71 → 0.74 (Phi-3); random: 0.55, 0.54 ± 0.12, 0.56 ± 0.02, 0.63 ± 0.04.
  - "consistently outperforms random directions ... across all instruction types and models" (p. 7): **no table or
    figure shows per-instruction-type steering results**; only pooled Table 4 and Fig. 3.
  - "we observe a significant increase in the success rate while keeping the task quality" (p. 9): no test is
    reported; gains are 2-6 points.
  - Direction tied to phrasing more than difficulty: Fig. 4 only (20 prompts, one instruction type, two models;
    cosine similarities shown graphically).
  - Reverse steering hurts: Table 12, with the authors' own caveat "the difference is not significant" (p. 19).
  - Explanations of "why prompt engineering works" (§6.2): interpretation, not supported by an experiment.

**Venhoff.**

- Pairs: RL-zero Open-Reasoner-Zero 0.5B/1.5B/7B/32B on Qwen2.5 bases; SFT-distilled R1-Distill-Qwen-1.5B (base
  Qwen2.5-Math-1.5B), R1-Distill-Llama-8B (Llama-3.1-8B), R1-Distill-Qwen-14B/32B (Qwen2.5-14B/32B); mixed QwQ-32B
  (Qwen2.5-32B) (p. 3, Table 1).
- Data: taxonomy from 12,102 MMLU-Pro prompts, 430,122 sentences (p. 13); vector training mix 9,794 questions (MATH,
  NaturalReasoning, SciBench, TheoremQA) with 995 validation; evaluation GSM8K, MATH500, a 1,000-question
  Hendrycks-MATH holdout (pp. 6, 19). Judge: Claude Sonnet 4.6, κ = 0.88 against humans (App. C).
- Statistics: base and hybrid are single greedy rollouts; "we run 3 judge repetitions per rollout and report mean ±
  std over all rollout-judge pairs" (p. 7). So the ± 0.0-0.3 on base and hybrid reflects judge noise only, not
  sampling or prompt variance. No intervals on Rec. %, no tests. Ablations (Fig. 4): two models, one run each, no error
  bars.
- Claims and support:
  - RL recovers ~76 %, SFT ~11 % (abstract): Table 1. The 12 RL cells average 76.4 % **(my computation)**. The four
    R1-Distill pairs average 8.5 % over their 12 cells; 10.7 % if QwQ's three cells are added **(my computation)**.
    So "11 %" matches only if the mixed QwQ is counted as SFT.
  - RL category vectors reach lower loss: Fig. 3 (curves only; one selected seed per model).
  - Sparse steering (5-12 % of tokens): Tables 2 and 5.
  - Each component necessary: Fig. 4 (full pipeline 77/77; random category 20/28; norm-matched random vectors −28/−13;
    MLP magnitude only −8/−1; random positions −8/28, for ORZ-1.5B/ORZ-32B).
  - "Since these factors are comparable, the recovery gap must stem from the category vectors" (p. 6): an inference.
    Benchmark performance is not comparable at small sizes (ORZ-0.5B 47.0 vs R1-Distill-1.5B 90.8 on GSM8K, Table 1),
    base families differ below 14B, and SAE layer and dictionary size differ per model.
  - "This sparsity rules out the hypothesis that steering simply biases toward specific outputs" (p. 7): no control
    supports this (see Q4).
  - "RL teaches when, not how": an interpretation of the above; no experiment isolates the gate from the next-token
    information it carries.

**Ward.**

- One pair: Llama-3.1-8B (base) and DeepSeek-R1-Distill-Llama-8B (SFT-distilled, not RL) (p. 1). 300 synthetic
  prompts (p. 7). Number of generated traces per cell not stated. Fig. 3 has "error bars of one standard deviation"
  (p. 3) over unstated samples; no seeds and no tests.
- Outcome: a keyword fraction, not task success. Eq. 1 uses B = {wait, hmm} (p. 2); §4.1 uses B = {wait, but}
  (p. 4); Figs. 3-4 plot "Wait" token %. **No accuracy or task-performance measurement anywhere in the paper.**
- Claims and support:
  - Base-derived vectors induce backtracking in the reasoning model, comparable to reasoning-derived: Fig. 3.
  - Neither vector induces backtracking in the base model ("always zero"): Fig. 3.
  - Best offset −13 to −8, best layer ~10: Fig. 2, Fig. B.1 (chosen on the same metric and prompts that are reported).
  - Beats all baselines: Fig. 4; "significantly" without a test; noise scale and norm matching unspecified.
  - Base- and reasoning-derived vectors have cosine ~0.74: stated in text only (p. 3).
  - Not a token-logit boost: Fig. 5 (logit-lens projection).
  - Direction is "densely present" and not the sole mediator: Fig. 6, a single annotated trace.
  - Title claim ("repurposes latent representations"): interpretation from one model pair and a keyword proxy. The
    authors themselves say "Our results should be interpreted primarily as an existence proof" (p. 5).

---

### Q3. The most faithful version of M3 on our checkpoints, and departures that change the meaning

No paper validates a model-minus-model vector, so "faithful" can only mean: use the papers' best-specified
construction choices (Ward's Eq. 3 averaging, a mid-depth layer, every-position addition during generation, held-out
strength choice as in Heo, a norm-matched random control as in Venhoff) and change only what the question requires.

**Proposed construction (my design, for the discourse to attack):**

1. *Text.* The 64 EDL-pool train prompts (arm-A, non-reach) rendered with Dan's `olmo_thinker` template, each followed
   by the final model's stored response (pool responses are truncated at 6,144 tokens). Both `dpo` and `s2300` are
   teacher-forced on identical token sequences in bf16, so the difference comes only from the weights. No reach prompt
   and no evaluation prompt is used.
2. *Activations.* Residual stream after decoder block ℓ for every ℓ (one pass per model stores per-prompt means for
   all 64 layers: 64 × 64 × 5,120 numbers).
3. *Positions.* Two vectors from the same passes, decided between on dev data before any held-out sampling:
   (a) **prompt positions**, the literal reading of "on prompts" and Heo's "first token" logic (RL may change how the
   instruction is encoded); (b) **response positions** (the thinking and answer tokens), the analogue of Ward's
   positions inside reasoning traces, where RL's behavioural change actually happens. Exclude position 0 and the
   template's special tokens from both means. **(My inference, to check on the data)**: attention-sink and
   massive-activation positions can dominate a raw mean, so report the norm share of the top few dimensions and of the
   excluded positions.
4. *Averaging.* Ward Eq. 3: mean over positions within a prompt, then over prompts;
   v_ℓ = MeanAct_s2300,ℓ − MeanAct_dpo,ℓ, accumulated in fp32.
5. *Diagnostics, free from the same passes:* ||v_ℓ|| and ||v_ℓ|| / RMS(h_ℓ); split-half cosine (prompts 1-32 vs 33-64)
   as a reliability check; and the share of the per-position difference explained by its mean,
   ||mean Δ||² / mean ||Δ||², at the candidate layers. If that share is a few percent, a constant vector can only carry
   a small part of what RL changed, and a null is the a-priori expectation (this is Venhoff's "when, not how" in
   measurable form).
6. *Application.* To `dpo` only, residual stream after block ℓ, h ← h + α·v_ℓ at **every position**: prompt tokens
   and every generated token, at every forward step (the CAA/Ward convention; Venhoff's gating is not available to a
   single vector). Sampling exactly as Dan's sampler: T 1.0, top-p 1.0, 32,768 max tokens, same template, same checker.
7. *Strength.* α in units of the vector itself. **α = 1 is the literal "mean difference" and needs no selection**, so
   it can go straight to held-out prompts without touching the dev split. A second α (from {0.5, 2, 4}) may be chosen
   on dev data. Report α·||v_ℓ|| / RMS(h_ℓ) as well, so the size can be compared across layers.
8. *Layer.* One ℓ chosen on dev data from mid-depth candidates. Ward's best was layer 10 of 32 (31 %), Venhoff uses
   ~37 % depth, so for 64 layers about 16-32 (e.g. {16, 20, 24, 28, 32}, plus 40 and 48 to see the trend). **Do not
   use the last layer** (Heo's choice): there the addition is close to a fixed token-logit bias **(my inference)**, which
   belongs in a control, not the main cell. Ward also reports that later-layer vectors "can be attributed to token-level
   logit boosts" but "are less effective when used for steering" (p. 4).
9. *Dev selection without held-out data.* Stage 1 (no sampling): teacher-forced NLL of the final model's dev responses
   under `dpo` + α·v_ℓ over the layer × α grid. This is the same objective Venhoff trains on (cross-entropy on the
   fine-tuned model's next token) and needs no checker results. Stage 2: sample the top one or two settings on the dev
   prompts to check coherence and checker passes.

**Departures that cannot be avoided and change what the result means:**

1. *Model-diff instead of behaviour contrast (Ward), supervised probe (Heo) or trained vectors (Venhoff).* None of the
   papers' positive results is evidence for this construction; they supply hyperparameter conventions and controls
   only. A model-diff mean contains everything RL changed on average (style, length habits, tokens RL prefers), not a
   direction selected for instruction-following success.
2. *Ungated, constant steering.* Venhoff's own ablation of position choice ("Steering at random token positions rather
   than SAE-gated disagreement positions collapses recovery for the smaller model (−8%), while the larger model is more
   robust (28%)", pp. 7-8) is the nearest analogue, and it still had the right category vector at each position and a
   trained coefficient. Our cell tests "how" without "when". A null therefore cannot refute "the pre-RL model already
   has the mechanism"; it can only refute "a context-free shift is enough".
3. *Pre-RL model is already a thinking model (SFT → DPO).* In Venhoff and Ward, "base" means a pretrained model. Our
   difference is RL only, and RL moved the weights by 0.364 % (CONTEXT). Ward's base-vs-reasoning asymmetry (the
   vector works in the fine-tuned model, never in the base) maps onto our dpo-vs-final pair only loosely.
4. *Long sampled generations.* 4-7k tokens at T 1.0, against 2,048-token greedy (Venhoff), short greedy IFEval
   responses (Heo) and unstated lengths (Ward). A constant push over 7k tokens can drift or loop, so length and
   capped-response rates are part of the outcome, and they cost money.
5. *Architecture.* **(My inference from the CONTEXT model facts)**: in Olmo 3 the RMSNorm sits on each sublayer's
   output, and attention and MLP read the raw residual stream. An added vector therefore reaches every later block's
   input without being rescaled by a pre-norm (only QK-norm and the final norm rescale), unlike the pre-norm Llama,
   Mistral and Qwen models in all three papers. Strength calibrations from the papers do not transfer, and norm
   matching of controls must be exact.
6. *Implementation.* vLLM with a patched decoder layer, against the papers' HF hooks. Two concrete hazards: (i) Dan's
   sampler uses prefix caching, and steered prompt positions change the KV cache, so prefix caching must be off or reset
   between α settings, otherwise one cell reuses another cell's cached prompt states; (ii) if the steering tensor is
   swapped in process, it must be updated in place (or CUDA graphs disabled), and every switch must be verified by a
   log-prob check.
7. *Outcome.* Strict IFBench checker at T 1.0, pass rate over draws, against Heo's greedy IFEval success (loose metric
   in his appendix tables), Venhoff's judge-graded math accuracy, and Ward's keyword frequency. The reach set measures
   rescue of prompts the pre-RL model never solved, which none of the papers measures.

---

### Q4. Weakest points that matter for our use

**Heo.**

1. *Location makes the effect nearly trivial.* Steering one pre-response position at the final layer **(my
   inference)** can change only the first output token under greedy decoding. Gains of 2-6 points could come from
   perturbing that one token, and the random control's ± 0.12 (13B) shows how much any perturbation moves things.
2. *In-sample everything.* The probe is trained on all data, α is chosen on a 10 % slice of the same data, and SR is
   scored on the same data (fn. 3, p. 6). The held-out variant is described only as "similar but slightly worse",
   with no numbers.
3. *Unspecified random control.* Distribution, norm and number of random directions are not given, so "better than
   random" cannot be calibrated.
4. *Small effects, no test, a claim without a table.* "across all instruction types" (p. 7) has no per-type table,
   and "significant" (p. 9) has no test.
5. *Probe may encode instruction type or base rate.* Leave-one-type-out AUROCs well below 0.5 (Table 7: 0.10-0.13 in
   some cells) mean the probe is systematically anti-predictive on held-out types. That pattern fits a probe that
   learned type identity times type base rate (types differ widely in success rate, Table 10: 0.14-1.00), not a
   general "following" signal **(my inference)**. For us this argues against expecting one direction to work across
   IFBench's many constraint types, and the paper itself concludes "a 'global' instruction-following dimension ... may
   not exist" (p. 17).
6. *Odd quality metric.* QR is "the number of responses scoring above 7 divided by the total number of responses that
   successfully follow instructions" (p. 7), which is not obviously a proportion.
7. *Small, older models, greedy decoding, simple single-constraint prompts.* Far from a 32B thinking model on
   multi-constraint IFBench prompts.

**Venhoff.**

1. *The hybrid is not "the base model alone".* Every step needs the thinking model's next-token prediction (the
   disagreement gate) and its hidden state (the SAE label). The vectors are trained by cross-entropy toward the thinking
   model's token at exactly those positions, with large coefficients (ORZ-32B α ≈ 10 ± 7, Table 5). Steering 12 % of
   tokens toward the fine-tuned model's choice at its decision points can carry a lot of the fine-tuned model's
   information. "This sparsity rules out the hypothesis that steering simply biases toward specific outputs" (p. 7) is
   asserted, not tested. A control that would test it (the same gate and targets with a token-level logit push in place
   of the category vector) is absent.
2. *Prompting gap counted as "recovered".* Base models get "Answer the following question: Q: ... A:" with no chat
   template and no step-by-step directive; thinking models get the directive (p. 20). Part of the base-to-thinking gap
   is a prompting gap that any CoT-inducing nudge closes, which inflates Rec. %.
3. *Error bars are judge noise only.* Single greedy rollouts, ± over judge repetitions (p. 7); Rec. % has no interval.
   Rec. % of 107.4 (ORZ-0.5B, Hendrycks) shows the comparison of greedy hybrid with T-0.6 thinking-model rollouts is
   loose.
4. *RL-vs-SFT confounds.* Different base families below 14B, different gap sizes, per-model SAE layer and dictionary
   choices, and an abstract figure (11 %) that matches Table 1 only if QwQ is counted as SFT **(my computation)**.
5. *Math only.* "the resulting taxonomies are biased toward mathematical reasoning" (p. 9). Instruction following is
   mentioned only as future scope (p. 9).
6. *Ablations on two models, one run each, no error bars* (Fig. 4).
7. *For our use:* the paper's key ingredient, the gate, is exactly what a single constant vector removes. Its
   relevant numbers are the ablations: random positions (−8 % / 28 %) and norm-matched random vectors (−28 % / −13 %),
   the latter showing that norm-matched random directions at large coefficients actively hurt.

**Ward.**

1. *No task performance at all.* The outcome is a keyword fraction, and the metric changes between sections ({wait,
   hmm} vs {wait, but}).
2. *The metric is partly gamed by the intervention.* Keyword-vs-GPT-4o precision falls from 63.36 % at strength 0 to
   47.66 % at strength 12 (Table C.1, p. 8). Steering raises "wait" tokens that the LLM judge does not call
   backtracking.
3. *In-sample selection.* Layer, offset and magnitude are chosen on the same 300 prompts and the same metric that are
   reported (Figs. 2, B.1).
4. *Baselines under-specified.* Noise scale, norm matching and number of samples are not stated, and "Notably, adding
   Gaussian noise to activations has a nontrivial effect" (p. 3).
5. *"Base never backtracks" is under-specified.* The base model's prompt format is not described. A pretrained base
   given a distill-style template may not produce "Wait" for reasons unrelated to the vector.
6. *One SFT-distilled 8B pair, a workshop paper.* No RL, no scale, no seeds.
7. *For our use:* the base-derived vector came from the earlier model's activations, and it did **not** change the
   earlier model's behaviour ("neither base-derived nor reasoning-derived steering vectors invoke backtracking
   behavior in the base model", p. 3). The fine-tuned model's downstream circuits were what responded.

**Shared weaknesses.** No paper reports prompt-level confidence intervals. None uses sampled long-form generation with
a strict verifier. None steers a model with a vector derived from the difference between two checkpoints of one run.

---

### Q5. Predictions for our setting, what would contradict the papers, and whether a null is interpretable

**What the papers lead me to expect** (my judgement, stated before any data):

- *Held-out arm-A pass@1, dpo + 1·v at a mid layer:* 0 to +5 points over `dpo` (RL's full gain is 29.0 → 68.3, i.e.
  +39 points), so at most about 10-15 % of RL's gain, with the interval including 0. Reasons: Heo's supervised,
  in-sample direction moved greedy success by only 2-6 points on simpler prompts; Venhoff's ungated ("random
  positions") ablation lost most of the recovery even with trained per-category vectors (28 % at 32B, −8 % at 1.5B),
  and a single constant vector is cruder still; Heo finds no direction that transfers across instruction types.
- *Reach prompts:* 0-3 % of draws, at best near the 60-word hint's 6.2 %. Heo frames representation edits as the
  internal analogue of rephrasing ("Our work with representation engineering achieves a similar outcome, but instead
  of modifying the input text...", p. 10), so the prompt-hint result is the natural ceiling to compare against.
- *Larger α (2-4):* more drift, longer responses, more 32,768-token caps, falling pass rates. Ward's sweep is
  non-monotone in magnitude (Fig. 2), and Venhoff's norm-matched random vectors with large coefficients made things
  worse (Fig. 4).
- *Norm-matched random direction:* −5 to +2 points. Heo's random directions lower SR (Table 4); Ward's noise raises
  some behaviours (p. 3).
- *Ward's asymmetry,* applied loosely: if RL changed how later layers read a representation rather than the
  representation itself, adding the representation change to `dpo` will do little. Subtracting v from the final model
  would then hurt more than adding it to `dpo` helps.

**What would contradict them.**

- On held-out arm A, ≥ 15 points (≥ ~40 % of RL's gain) for dpo + v with the random control flat, or reach ≥ 20 % of
  draws. That would contradict the "when, not how" reading of Venhoff as applied to instruction following (a
  context-free shift would be enough) and Heo's "no global direction" across instruction types. It would also be
  remarkable on its own: a 5,120-number intervention recovering a third or more of what the 134 M-weight LoRA recovers.
- On a smaller scale, a clear dose-response (gain rising from α 0.5 to 1 to 2 before breaking) with random flat would
  support a usable "RL direction" even if the effect is modest.

**Is a null interpretable?** Only partly, and only with four supports:

1. a manipulation check showing v moves `dpo` toward `s2300` on something other than the checker (a fall in
   teacher-forced NLL of held-out final-model responses; a shift in response length or thinking style), so the vector
   is shown to act;
2. implementation checks (α = 0 bit- or log-prob-identical to the unpatched sampler; α ≠ 0 matching an HF-hook
   reference on a few prompts);
3. a dose range that reaches visible degradation, so "too weak" is excluded;
4. ideally, the necessity test: `s2300` − v loses performance. If it does, the direction is used by the final model
   but cannot be read by `dpo` (Ward-like). If it does not, the mean direction is irrelevant in both models.

Even with all four, a null rules out only "one constant additive direction at one mid-depth layer, at every position,
recovers RL's gain". It does not rule out gated, position-specific or multi-layer steering (Venhoff), and it says
nothing against "the behaviour was latent": the 64-example LoRA already shows a small weight change suffices. Because
the mean-share diagnostic (Q3, item 5) can be computed before any sampling, it should be reported as a stated
expectation: if the constant part is a few percent of the per-position difference, a null is what we would predict.

---

### Q6. Controls

**Necessary.**

1. *Fresh α = 0 `dpo` draws on the reach prompts, through the patched engine.* The reach set was selected as "0 of
   his 768 stored draws" (2026-11-25 summary), i.e. 0/16 per prompt. The stored 0.0 % is therefore the selection sample
   itself, and a fresh sample can be above 0 by chance (0/16 is compatible with a per-prompt rate up to about 17 %
   **(my computation: one-sided 95 % bound 1 − 0.05^{1/16} ≈ 0.17; two-sided Clopper-Pearson ≈ 0.21)**). `s0100`'s
   0.1 % hints that the fresh rate is small,
   but the comparison baseline must be fresh. For held-out arm-A prompts, the stored 16-draw `dpo` results can serve once
   α = 0 is shown to be equivalent. They are slightly biased upward (non-reach means at least 1 pass in 16), which makes
   the comparison conservative; say so.
2. *Norm-matched random directions at the same layer and the same α·||v||, on the same prompts and draws.* How to
   draw them: **covariance-shaped**, z ~ N(0, diag(σ_ℓ²)) with σ_ℓ the per-dimension standard deviation of `dpo`'s
   residual at layer ℓ (from the extraction pass), rescaled to exactly α·||v_ℓ||. An isotropic Gaussian direction in a
   strongly anisotropic residual stream mostly points where the model is insensitive, which makes it too weak a
   control and flatters v. Use **three** directions (seeds), each on a third of the draws, so the random distribution
   is not one draw (Heo's random SR spread was ± 0.12). Venhoff's "norm-matched random directions" is the precedent for
   norm matching; none of the papers specifies the distribution.
3. *Manipulation check without the checker:* teacher-forced NLL of held-out final-model responses and response-length
   and capped-response shifts for every steered cell.
4. *Implementation equivalence:* α = 0 versus the unpatched sampler (log-probs on a few prompts), α ≠ 0 versus an HF
   forward hook, prefix cache off or reset between settings, and the steering tensor verified after every switch.
5. *Selection hygiene:* v built on the vector prompts, layer and α chosen on the dev prompts, results reported on
   held-out arm-A and reach prompts only. The α = 1 cell is reported whatever the dev result is.

**Nice to have, in order of value.**

1. *Necessity:* `s2300` − v (subtract from the final model). Most useful when the main cell is null, because it
   separates "direction unused" from "direction used but unreadable by `dpo`".
2. *Another checkpoint pair:* v from `s0500` − `dpo`. Its cosine with the final vector across layers is free once the
   `s0500` pass exists (one more download and load). Adding v_s0500 to `dpo` tests whether early-RL directions already
   point the same way. Lower priority than 1.
3. *A realistic-geometry null:* `dpo`'s mean on prompt half A minus half B, rescaled. A natural activation difference
   that is not RL's.
4. *Token-bias control* (Ward §4.1 logic): project v_ℓ through the rest of the network or use a last-layer mean-diff
   logit bias (vLLM supports `logit_bias` without a patch). Run it only if the main cell shows a gain, to test whether
   the gain is a token preference.
5. *Adding +v to `s2300`* ("double dose"): low information; shows only whether the final model is saturated along v.
6. *A prompt-positions vector versus a response-positions vector* as two held-out cells: only if dev data cannot
   separate them.

---

### Q7. Minimum cells and rough cost

**Splits** (176 arm-A non-reach prompts, 48 reach):

- **V** (vector): the 64 EDL train prompts, whose final-model responses are already in `pool/`.
- **D** (dev, layer and α): the 32 EDL test prompts.
- **H** (held-out arm A): the remaining 80 non-reach arm-A prompts.
- **R** (reach): 48 prompts, never used to choose anything.

**Assumptions:**

- Steered `dpo` responses average about 5,000 tokens (between `dpo`'s 4,500 and the final model's 6,700).
- A patched vLLM runs at 1,000-1,400 tokens/s. This is a guess below the unpatched 1,400-1,700, in case CUDA graphs are
  lost; the feasibility agent's numbers override it.
- HF teacher-forced prefill runs at about 5-10k tokens/s on one H200 (a guess).
- H200 at $5.29/h; ×1.3 buffer for repetition loops and stalls.

| stage | cells | work | time | cost |
|---|---|---|---|---|
| A. build + screen (no sampling) | download and load `dpo` and `s2300` in HF; teacher-force V (≈ 64 × 6.5k ≈ 0.42 M tokens per model); store per-layer per-prompt means and per-position activations at ~4 layers; NLL screen on D's final responses, 8 layers × α {0.5, 1, 2, 4} + α 0 (33 forwards × ~0.2 M tokens ≈ 6.6 M tokens); HF-hook reference log-probs | ≈ 7 M prefill tokens | 45-60 min | $4-5.3 |
| B. dev sampling | dpo + 1·v_ℓ* and dpo + α*·v_ℓ* on D, 32 × 4 each; baseline = stored `dpo` draws once α = 0 is shown equivalent | 256 responses ≈ 1.3 M tokens | 15-21 min (+ load) | $1.5-2.5 |
| C1 | dpo + 1·v_ℓ*: H 80 × 2 + R 48 × 4 | 352 responses ≈ 1.8 M | 21-29 min | $1.9-2.6 |
| C2 (only if α* ≠ 1) | dpo + α*·v_ℓ*: same | 352 ≈ 1.8 M | 21-29 min | $1.9-2.6 |
| C3 | 3 covariance-shaped random directions, norm-matched to the largest steered cell: H 80 × 3 + R 48 × 3 (one draw per direction per prompt) | 384 ≈ 1.9 M | 23-32 min | $2.0-2.8 |
| C4 | fresh `dpo` α = 0 on R, 48 × 4, through the patched engine | 192 ≈ 0.9 M (at 4.5k) | 11-15 min | $1.0-1.3 |
| loads, checks, buffer ×1.3 on C | | | | ≈ $3-4 |
| **Total, necessary** | | | ≈ 2.7-3.5 h | **≈ $15-21** ($13-18 if α* = 1) |
| optional N1 | `s2300` − v on H 80 × 2 + R 48 × 4 (≈ 6,800 tokens each) + `s2300` load in vLLM | ≈ 2.4 M | 35-50 min | $3-4.5 |
| optional N2 | `s0500` pass for v_s0500 and cosines (no sampling) | ≈ 0.4 M prefill + load | 10-15 min | ≈ $1 |

**Resolution of this design** (my computation from Bernoulli variance, to be replaced by the prompt bootstrap):

- *H, 80 prompts × 2 draws, paired against stored 16-draw `dpo`:* the 95 % half-width on the pass@1 difference is
  roughly ± 7-8 points. Only effects of about 20 % or more of RL's 39-point gain are detectable. Halving the
  half-width needs about 4× the draws (≈ +$6-8 per cell).
- *R, 48 × 4 = 192 draws:* a steered rate of 3-4 % or more spread over several prompts is distinguishable from a fresh
  baseline near 0.1-0.5 %; 1 % is not.

The proposal should state these resolutions up front, so that "no effect detected" is read as "less than about 20 %
of RL's gain on arm A, less than about 3 % on reach", not as zero.

**The minimal answer to Leon's question:**

- C1 (α = 1, no selection), C3 (random), C4 (fresh reach baseline) and stage A's manipulation check. These four answer
  "is a constant mean-difference direction a cheap recovery of RL's instruction-following gain?"
- C2 and N1 are the first additions, if budget allows.

END OF INDEPENDENT READ
