# M3 (key `steer`): design discourse for adding the final-minus-pre-RL mean activation difference to `dpo`

## Turn 1 — reader-implementer (Claude Sonnet 5), 2026-11-27 01:44 PT [time corrected by the orchestrator from the file's modification time; the heading as written said 01:50 PT, which was not taken from `date`]

Sources: my three reading notes in `docs/literature/2026-11-27-elicitation-metrics-fulltext/` (page references there) and `discourse/steer_feasibility.md` (final). Labels: [read] stated in a paper, [feas] from the feasibility file, [mine] my design choice, [guess] unsupported.

### 0. What the papers do and do not license

- No paper builds a vector as "final model minus pre-RL model" activations. Heo: probe weight within one model (Heo p6). Ward: difference of means within one model, base activations on the reasoning model's own traces, added to the reasoning model (Ward pp3, 7). Venhoff v4: gradient-trained vectors plus a thinking-model-driven gate (Venhoff pp5, 19). Our vector is a new construction; the papers motivate it only loosely.
- The most relevant caution is Ward Fig. 3 (p3): a vector effective in the fine-tuned model did nothing when added to the base. Ours is added to the model without the post-RL circuitry, so a null is a live outcome.
- Effect sizes to calibrate against: Heo +2 to +6 points on short-answer IFEval with an in-sample probe (Table 4); Venhoff's full gated hybrid recovers 49.7 to 59.8 % of a MATH500 gap at 7B and 32B (Table 1). A static, ungated vector should not be expected to beat these.

### 1. The vector

- **Convention.** Residual after l layers = HF `hidden_states[l]` = output of `layers[l-1]`, l in 1..63 [feas s1]. The addition goes on the output of `layers[l-1]`. Fixed layer **l = 24** (37.5 % of 64; Venhoff p5 uses about 37 %, Ward uses 31 %, i.e. l = 20) [mine]. It is not tuned on dev (see s4), to keep the grid small.
- **Prompts.** The 32 first prompts of `pool/train.jsonl` (arm-A non-reach; the final model's responses are already in the pool, so no sampling is needed). None is a reach prompt; none is used for dev or evaluation.
- **Positions, primary vector V_resp [mine].** Mean over **response token positions** of the pool's s2300 responses, teacher-forced through both models with the same token ids (same chat template as `sample.py`), excluding position 0. Mean first within a prompt, then across prompts (Ward Eq. 3, p7). `V_resp = mean(h_s2300[24]) - mean(h_dpo[24])`.
  - Departure from "on prompts": the vector is applied at generated positions, so it should be estimated where it will act. Ward's cross-model result is computed on trace positions through both models (p3), not on the prompt. Heo is the only paper that uses a prompt-position vector, and it adds only at one prompt position and last layer (p6).
  - Cost of the departure: dpo reads s2300's text off-policy, so the difference mixes "reads differently" with "predicts differently" [guess that this matters].
- **Secondary vector V_prompt [mine].** The same mean over **user-prompt tokens only** (not position 0, not template tokens, not response). This is the literal "on prompts" reading and Heo-like. It is carried as a second dev candidate, not as a separate evaluation cell.
- **Sink and template handling.** Save four group means per layer (position 0, template tokens, user tokens, response tokens) for audit, as the feasibility file suggests, but build both vectors without position 0 and without template tokens, because position 0 may carry an attention-sink norm [guess] and template tokens are identical in every prompt.
- Compute in fp32 sums, HF bf16 forward, one model at a time [feas s5]. Record `||V||`, the median residual norm of dpo at l = 24 over generated positions, and the cosine between V_resp and V_prompt.

### 2. Application

- Add `alpha * V` (bf16 buffer) to every position's residual at layer 24, prompt and generated tokens, in the patched `Olmo2DecoderLayer` (Route A, one process per setting) [feas s3]. Alpha is a multiple of the raw mean difference: **alpha = 1 adds exactly the mean difference**. Log `r = alpha*||V|| / median||h_24||` so settings can be compared with Heo and Ward in relative terms.
- Prompt and generated positions both, because Heo's addition is at the prompt only and cannot persist over thousands of tokens, while Ward's text does not say (Ward p3).

### 3. Pre-flight (feasibility s4, unchanged) 

Stock noise floor, zero-vector identity, steered-vs-HF-hook agreement at l = 24 and one second layer, compile/eager agreement, `STEER_ACTIVE` in the worker log. Abort the pod run if any fails. Verify with a teacher-forced sequence longer than 4,096 tokens (sliding window).

### 4. Dev split and selection rule (written before any run)

- Non-reach arm-A prompts: 176 (218 minus 42 reach). Minus the 32 vector prompts = 144. Split by a fixed seed (0), sorted by pid: **dev 64**, **held-out evaluation 80**. The rule and the seed go into `STEER_SPEC.md` before the first dev run. No reach prompt is touched at dev.
- Dev baseline: dpo's stored draws on the same 64 prompts (16 per prompt), strict checker.
- **Dev settings (4, plus 1 conditional):** V_resp and V_prompt, each at alpha in {1, 4}, layer 24; 2 draws per prompt = 128 draws, `max_tokens` 12,000 (truncated draw = fail, as it would be unparsable) [mine].
- **Conditional fifth:** if the best of the four is an alpha = 4 setting with r < 0.05, run that vector at alpha = 16 (once).
- **Rule.** Score = dev pass@1 of the setting minus dpo's stored dev pass@1 (in points). Guards: fraction of draws reaching 12,000 tokens at most 10 %, and mean response length within 0.7 to 1.3 times dpo's. Choose the highest score among settings passing the guards; ties within 2 points go to the smaller r, then V_resp. If no setting passes the guards, the damage is itself the finding and the evaluation uses the setting with the smallest r that has a nonzero dev change.
- **Evaluate even if the best dev gain is zero.** Reach prompts are the question and dpo scores 0 on them; a null on non-reach dev does not predict a null on reach, so I propose the evaluation regardless, with the price stated in s7.
- Dev noise: with 128 draws at p near 0.29, the standard error of a paired difference against a 16-draw baseline is about 4 points [guess]. Dev can only exclude gross damage or large gains; it cannot rank close settings. State this in the write-up.
- Layer and strength are not chosen on reach results at any step.

### 5. Evaluation cells and controls

All at T 1.0, top_p 1.0, 32,768 max tokens (Dan's constants), strict checker, held-out prompts only.

| cell | checkpoint + intervention | reach (48) | held-out non-reach (80) | draws |
|---|---|---|---|---|
| E0 | dpo stock (stored draws, reused) | 16 per prompt | 16 per prompt | 0 new |
| E1 | dpo + alpha* V* (dev winner) | 8 | 2 | 544 |
| E2 | dpo + sign-scrambled copy of alpha* V* | 8 | 2 | 544 |

- **E2 control:** the same vector with each coordinate's sign flipped at random (fixed seed). It has exactly the same norm and the same per-coordinate magnitude profile, so it controls for massive-activation dimensions that an isotropic Gaussian direction would miss [mine]. Heo does not say how its control was drawn (note, p6-7) and Heo's 0.12 SD across random seeds says one draw is noisy; Venhoff uses norm-matched random vectors and they hurt (Fig. 4, p8). One scrambled draw only; if E2 moves the score, the control is not clean and I would say so.
- **Metrics:** pass@1 and pass@8 on reach (lead with pass@1; 42 of the 48 reach prompts are also arm-A prompts, but those are not in the dev or vector sets), pass@1 on the 80 held-out non-reach prompts; % of set, 95 % prompt-bootstrap intervals, paired against E0. Also: share of draws hitting 32,768, mean length, fraction of draws with an unparsable answer.
- **Not run (and why):** adding the vector to `s2300` (a sanity check, needs a second 64.5 GB load, about $2); a vector from a different pair (`s0500` minus `dpo`, a specificity control, about $2.4 per cell). Both are conditional follow-ups if E1 beats E2.
- **Reach power [mine]:** dpo has 0 passes in 768 plain tries on reach. With 384 draws, 4 or more passes (about 1 %) with none in E2 would be a visible move; fewer is within what a prompt bootstrap over 48 prompts cannot separate from zero. On the 80 non-reach prompts, the paired difference needs about 6 points to be clearly above noise [guess].

### 6. Loop and cost protection

- Per-request cap at the sampler's 32,768 (eval) and 12,000 (dev); I do not lower the eval cap (it would change what a draw scores against the stored dpo draws).
- Watchdog per process (memory rule): STALL if no new tokens for 10 min; OVERRUN if wall time exceeds 1.6x the estimate (dev 22 min, eval 45 min); EXIT recorded. On OVERRUN the unfinished draws are scored as failures, flagged in the sidecar, and the cap-hit rate is reported. Loops at large alpha are the main cost risk (up to 7x per affected draw) [feas s7].
- Check-in every 30 minutes while a setting runs.

### 7. Cost

Assumptions: $5.29/h; 4,500 tokens per draw; 1,500 tok/s at high concurrency, tail-bound at dev concurrency [feas s6]; Route A, one process per setting, about 2 min startup.

| item | tokens | H200 min | $ |
|---|---|---|---|
| setup, download, means (s2300 and dpo), verification | | 60 | 5.3 |
| dev 4 settings x 128 draws (0.58 M tokens each) | 2.3 M | 4 x 13 = 52 | 4.6 |
| evaluation 2 settings x 544 draws (2.45 M each) | 4.9 M | 2 x 29 = 58 | 5.1 |
| loop and tail allowance, 15 % of generation | | about 20 | 1.8 |
| **total** | 7.2 M | about 190 | **about $16.8** |

- Against the $15 share: above by about $2. Cuts available, in order: drop V_prompt from dev (2 fewer dev settings, saves $2.3, total about $14.5; it is the literal-"on prompts" arm, so cutting it removes the reading closest to the brief and to Heo); reduce dev draws to 1 per prompt (saves about $1.5 but dev SE rises to about 6 points).
- I recommend running the full 4 dev settings and asking Leon for the $2 overage. If the conditional fifth dev setting fires, add about $1.2. Hard stop for M3: $22.
- Versus the feasibility file's $17: 4 dev settings instead of 9 (layer fixed, not tuned) and 544-draw evaluation cells instead of 820.

### 8. What each outcome would and would not show

- **E1 clearly above E0 and above E2 on reach (and no loss on non-reach):** a single static residual vector derived from the final-minus-pre-RL difference moves `dpo` toward the final model's behaviour on prompts it never passes. It would not show that RL "is" this vector, that it is the best vector (Venhoff's learned, gated vectors are stronger), that it works at other layers, or that the checker passes are genuine.
- **E1 above E0 but E2 does too:** the push is nonspecific perturbation (Ward p3 saw noise move behaviour); no claim.
- **E1 near E0 on both sets with r in a range where the model is not damaged:** at this layer and these alphas the mean difference is not a lever. It would not mean that no linear direction exists (single layer, mean-difference rather than learned or probed, ungated, no layer or position search), would not mean that RL's change is not low-dimensional, and would not exclude the case in Ward (vector acts only where the reader circuit exists).
- **E1 below E0 (damage or looping):** the strength that moves anything also breaks the model; the answer to "is part of the gain recoverable this way" is not yet. The cap-hit rate decides between damage and null.
- In every case the reach result, not the non-reach one, answers the question; the non-reach result shows whether the push costs the model.

### 9. OPEN for the auditor

1. V_resp versus V_prompt as primary: I chose V_resp on the "estimate where it acts" argument and Ward's trace-position construction; the brief says "on prompts".
2. Layer fixed at 24 from two papers' fractions, not tuned; one layer is a thin test of "one vector".
3. Whether to evaluate when dev shows no gain (I propose yes).
4. One scrambled-sign control draw; Heo's own seed SD suggests it may be noisy.
5. The reuse of Dan's stored dpo draws as E0 assumes the pod's patched-but-zero path reproduces Dan's sampler (verified numerically only, not by regenerating); a fresh stock dpo cell on the reach set would cost about $2.4.
6. Pool file names and the prompt split are to be confirmed against `pool/MANIFEST.json`; I did not open them.

## Turn 2 — auditor (Claude Opus 5.5), 2026-11-27 01:54 PT

Inputs:
- my independent read, `discourse/steer_audit_independent.md` (md5 4097ef3c67d52158b6bd2e48c759b75f, not edited since);
- the reader's three notes in `docs/literature/2026-11-27-elicitation-metrics-fulltext/`;
- `discourse/steer_feasibility.md` (final);
- Turn 1.

Page numbers are PDF pages.

### A. Reading disagreements

**Heo (2410.14516v5).** We agree on nearly everything: the probe-weight construction, the in-sample probe and α, the
unspecified random control, the 2-6-point effects, and the absence of a global direction. Differences:

1. *Which layer's probe supplied D.* The reader says the text does not state it (note, "Layer"). My Q2 line ("the
   layer later used for steering") assumed a last-layer probe. **The text supports the reader**: p. 6 says only that
   the *adjustment* is at the last layer. I withdraw the assumption.
2. *Phi-3 baseline.* The reader found that Table 4's original SR for Phi-3 (0.71) does not equal the mean of Table 10
   (0.655), while the other three models match. I missed this. I re-did the arithmetic: Llama-7B 0.569, Llama-13B
   0.610, Mistral 0.584, Phi-3 0.655. **The reader is right**; it is unexplained.
3. *Is the addition repeated at every step?* The reader leans to every step, because a first-token-only shift seems
   too small to explain gains. I lean to the literal single position, because under greedy decoding one changed first
   token can change the whole response. **The text supports neither** (p. 6 is silent). It does not affect our design,
   which adds at every position either way.
4. *Points I raised that the note lacks (additions, not contradictions).*
   - Leave-one-type-out AUROCs are well below 0.5 in places (Table 7: 0.10-0.13). This points to a probe that encodes
     instruction type times type base rate (Table 10: 0.14-1.00), which strengthens the "no global IF direction" caution.
   - The QR definition (p. 7) is literally "responses scoring above 7 divided by responses that follow instructions",
     which is not obviously a proportion. The reader reads it as one.
   - Both are immaterial to the design except as priors.

**Venhoff (2510.07364v4).** We agree on the construction (gradient-trained per-category vectors, coefficient MLP,
gate run by the thinking model), on the error bars being judge noise, and on the absence of any static-vector test.
Differences:

1. *Steering layer.* The reader's note says "in practice layer 27 of 64 for 32B (42%) (App. G.8, p40)". G.8's "Layer
   27" is the **SAE taxonomy layer** ("Categories are identified by our SAE", Table 13). App. E.2 (p. 19) names two
   different layers: "the thinking model's hidden state at the SAE layer" versus "hp is the base model's
   residual-stream activation at the steering layer". The other SAE layers (4 of 28, 20 of 28, 38 of 48) cannot all
   be "approximately 37%" (p. 5). **The text supports my reading**: the steering layer per model is not reported, and
   37 % of 64 is about 24. This does not change Turn 1, which uses 24 anyway.
2. *"11 % SFT".* The reader repeats the abstract. Table 1 gives 8.5 % for the four R1-Distill pairs, and 10.7 % only
   if QwQ is added (my computation), while p. 9 calls QwQ "similar to the distilled models". **Table 1 supports me.**
   Minor.
3. *Points I raised that the note lacks.*
   - The gate leaks the thinking model's next token at exactly the steered positions, with large coefficients (Table
     5, ORZ-32B α ≈ 10 ± 7). "This sparsity rules out the hypothesis that steering simply biases toward specific
     outputs" (p. 7) is asserted, not tested.
   - The base model's bare Q/A prompt (p. 20), against the step-by-step directive the thinking models get, makes part
     of "recovery" a prompting gap.
   - Both points lower the 76 % prior further for a static vector. The reader's conclusion is the same as mine.

**Ward (2507.12638v1).** We agree on the construction (Eq. 3, positives minus *all*, layer 10, offset about −10), on
there being no task metric, on a single pair, and on noise moving the metric. Differences:

1. *"The RL case should make base-derived vectors more transferable than Ward's case; Venhoff et al. say as much"*
   (Ward note, "How our setting differs"). **The text does not support this.** Venhoff never transfers a vector between
   models: its vectors are optimised inside the base model (p. 19). Ward has no RL pair. It is a guess and should be
   labelled one.
2. *Whose text the vector is computed on.* The reader cites Ward for trace-position vectors (Turn 1 §1). In Ward the
   traces are the **receiving** model's own text: base-derived vectors come from base activations on R1-Distill's
   traces and are added to R1-Distill (pp. 2-3). Turn 1 computes V_resp on **s2300's** text and adds it to `dpo`. The
   Ward-faithful arrangement is the receiver's (`dpo`'s) own text. See B2.
3. *A point I raised that the note lacks.* The keyword proxy's precision against GPT-4o falls from 63.36 % at strength
   0 to 47.66 % at strength 12 (Table C.1, p. 8): steering inflates the proxy itself. This is a caution against any
   cheap behavioural proxy we might add.

**Feasibility file.** No disagreement. Two items in it matter for B:
- Route A (one process per setting) removes the prefix-cache hazard I raised.
- HF `hidden_states[64]` is post-norm and forward hooks do not change recorded hidden states (§1). The NLL check below
  must therefore read log-probs, not hidden states.

### B. Attack on the design

**B1. No manipulation check, so a null is not interpretable as designed.**

Turn 1's only outcomes are checker passes plus damage indicators (cap hits, length, unparsable answers). A null
("E1 near E0, r not damaging", §8) cannot separate three cases:
- (a) v does not move `dpo` at all;
- (b) v moves `dpo` toward `s2300` but only in style;
- (c) v moves it, but the passes do not follow.

*Change required:* a teacher-forced NLL measurement in HF, no sampling.
- Text: `s2300`'s stored responses on 16 held-out non-reach prompts (Dan's 8 stored `s2300` draws per arm-A prompt
  are full length; use one per prompt, about 100k tokens).
- Conditions: `dpo`, `dpo` + α*v, `dpo` + each scrambled control, and `s2300` itself.
- Report **gap closed = (NLL_dpo − NLL_steered) / (NLL_dpo − NLL_s2300)**, the share of the per-token cross-entropy
  gap toward the final model that the vector removes.
- Cost: about 6 forward passes, roughly 5 min (≈ $0.5).
- Pre-register the reading (my proposed thresholds): gap closed < 5 % means the vector barely moves the model, so a
  pass@1 null is expected and uninformative about IF specifically; gap closed ≥ 20 % with no pass change means the mean
  shift carries non-IF parts of RL's change. This is also the continuous result that survives the coarse resolution in
  B6.

*Same session, near free (recommended, not required):* while `s2300` is loaded for the means, the NLL of the same text
under `s2300` − v. If removing v costs `s2300` much more than adding it gains `dpo`, that is Ward's asymmetry (the
direction is used downstream in the final model but `dpo` cannot read it), and it makes a null on `dpo` informative.

**B2. V_resp is computed on the wrong text for the reader's own argument.**

The argument is "estimate where it acts". The vector acts on `dpo` generating `dpo`-like text, and in Ward the text is
the receiving model's (A, Ward 2). On `s2300`'s text the difference also contains "`s2300` reading its own fluent
text against `dpo` reading off-policy text", which Turn 1 flags as a guess.

*Change required:*
- Primary V_resp = mean over response positions of (h_s2300 − h_dpo) on **`dpo`'s own stored responses** (Dan's
  16-per-prompt draws; one per vector prompt).
- Compute the `s2300`-text version in the same passes and report the cosine between the two as a diagnostic (a few
  minutes of extra prefill, no new dev cell).
- If the two versions disagree strongly (cosine < 0.5), say so in the write-up. Do not add a cell.

**B3. "On prompts" versus V_resp.**

I accept V_resp as primary, provided the selection in B4 is allowed to pick V_prompt. But V_prompt as specified
excludes the template tokens, and that throws away the one position Heo found informative. Heo's "first token" is the
state after the whole prompt (p. 4), which in our chat template is a **trailing template token** (the assistant /
`<think>` generation prompt). Its residual depends on the prompt even though the token ids are identical. Only the
*leading* template tokens (before the user content) are prompt-independent, and only they should be excluded.

*Change required:* V_prompt = user tokens plus trailing template tokens (from the end of the user content through the
generation prompt); exclude position 0 and the leading template block.

**B4. Fixed layer 24, and strength chosen by dev pass@1.**

Turn 1's dev selection is noise-dominated. By its own estimate the dev standard error is about 4 points; the tie rule
is 2 points; and the plausible differences between settings are 0-3 points. That is a random pick among 4 settings
costing $4.6, with no layer search at all.

The teacher-forced NLL of `s2300`'s dev responses under `dpo` + α·v_l is:
- low-noise (it averages about 100k tokens);
- checker-free, so there is no route to the reach results;
- the same objective Venhoff trains on (cross-entropy on the fine-tuned model's next token, p. 19);
- able to sweep layers for almost nothing;
- able to show where α starts to damage (NLL rising), which supplies the dose evidence a null needs.

*Change required:*
- Select (vector type, layer, α) by dev NLL in HF: V_resp at l ∈ {16, 20, 24, 32, 40} × α ∈ {0.5, 1, 2, 4, 8}, plus
  V_prompt at the best layer × the same α. That is about 30 forwards on 16 dev prompts with one `s2300` response each,
  roughly 15-25 min, ≈ $1.5-2 (my estimate; HF prefill throughput is unmeasured).
- Then sample on dev **only the chosen setting** (64 × 2 draws) as a guard: cap hits ≤ 10 %, length 0.7-1.3× `dpo`,
  and dev pass@1 not more than 10 points below `dpo`. If it fails, step α down once and re-check.
- This replaces the 4 pass@1-ranked dev settings and the conditional α = 16, and saves about $2.5-3.5.
- Record NLL at α = 1 whatever is chosen, because α = 1 is the literal brief.

**B5. Controls.**

- *E0 reused on reach: not acceptable.* The reach set **was selected on those very draws** ("0 of his 768 stored
  draws" = 0/16 per prompt). E0 = 0.0 % is the selection sample, so any fresh draw, steered or not, can produce a few
  passes by regression to the mean (`s0100` gave 0.1 % on 32 draws per prompt). Turn 1's power statement, "4 or more
  passes (about 1 %) with none in E2", is compared against a baseline that is zero by construction.
  - *Change required:* a fresh unsteered `dpo` cell E0′ on the 48 reach prompts × 8 draws (384 draws, ≈ $1.8-2.4)
    through the patched engine with a zero vector. It doubles as the end-to-end α = 0 check.
  - On the 80 held-out non-reach prompts, the stored 16-draw E0 is **acceptable as is**, once α = 0 log-prob identity
    has passed. It is biased slightly upward (non-reach means at least 1 pass in 16), which works against E1; say so.
- *E2, a single sign-scrambled copy.* Sign scrambling is a better distribution than my covariance-shaped proposal: it
  matches v's own per-coordinate magnitude profile, massive dimensions included. I concede the distribution. One draw
  is not acceptable: Heo's random SR varies ± 0.12 across seeds (Table 4, 13B), and Venhoff's norm-matched random
  vectors swung recovery from −28 % to −13 % between two models (Fig. 4).
  - *Change required:* **three** independent sign scrambles, splitting E2's 544 draws (reach 48 × 8 → about 2-3 draws
    per prompt per scramble; held-out 80 × 2 → one draw per prompt for two of them, or round-robin). Same total cost
    plus two process starts (≈ $0.3).
  - Report the share of ||v||² in the top 5 coordinates. If it exceeds about 50 %, the scrambles mostly flip the sign
    of a few massive dimensions, and E2 is a "damage" control rather than a neutral one. Say so if it happens.
- *Adding v to `s2300`* (listed in Turn 1 as not run): if anything is done on `s2300`, it should be subtraction (B1,
  as NLL), not addition. **Acceptable** to leave the sampled version out.

**B6. Dev truncation at 12,000 tokens biases the dev comparison.**

Stored `dpo` draws were scored at a 32,768 cap. Some `dpo` passes are longer than 12,000 tokens and would count as
failures under the dev rule. "Dev pass@1 minus `dpo`'s stored dev pass@1" is then biased against every steered setting.

*Change required:* score the stored `dpo` dev baseline with the same 12,000-token truncation (free from stored lengths).
With B4 the dev pass@1 only serves as a damage guard, but the comparison should still be like for like.

**B7. Resolution and whether the cost is justified.**

- *Held-out non-reach, 80 × 2 against 16 stored:* the 95 % half-width on the paired pass@1 difference is about
  ± 6-8 points (my Bernoulli computation). That detects only effects of 15-20 % or more of RL's +39 points. That is
  acceptable, since Turn 1 treats this set as the damage check.
- *Reach, 384 draws against a fresh 384:* a steered rate of about 2-3 % spread over at least 3 prompts is separable
  from a fresh baseline near 0.1-0.5 %. That is below `s0500`'s 4.5 %, so the design can answer the useful question:
  **does the vector take `dpo` at least to where 500 RL steps got it on reach?** The write-up should frame the result
  in those checkpoint-equivalent terms (`s0100` 0.1, `s0500` 4.5, `s1100` 38.5 %; hint 6.2 %; LoRA 63.5 %).
- With B1 the cost is justified: even a null returns a continuous "gap closed" number and an s0500-level bound on
  reach. Without B1, $17 buys a null that cannot be read.

**Cost of the revised design (my estimate, same assumptions as Turn 1 §7):**

| item | $ |
|---|---|
| setup, download, means, verification | 5.3 |
| NLL selection, manipulation check and necessity check | 1.5-2.5 |
| dev guard, one setting | 1.2 |
| E1 + E2 (3 scrambles) | 5.1 + 0.3 |
| E0′ fresh reach | 1.8-2.4 |
| loop allowance | 1.8 |
| **total** | **about $17-18.5** |

That is about the same as Turn 1's $16.8; the saved dev cells pay for E0′ and the NLL work. If the budget must stay
near $15, cut E1 and E2 held-out to 80 × 1 (saves ≈ $1.3) before cutting anything on reach.

**B8. Smaller points, acceptable as is.**

- Steering at every position including prompt and template tokens.
- Evaluating even if dev shows no gain (α = 1 needs no selection; reach is the question).
- Route A with one process per setting.
- The watchdogs.
- The 32 vector prompts. Report the split-half cosine (16 against 16) before any sampling; if it is below about 0.8,
  the mean is noise-dominated, and that should be stated as a finding, not fixed by adding prompts from dev.
- The pool responses are capped at 6,144 tokens. With B2 the vector uses `dpo`'s stored full-length responses; keep
  the 4,096-token sliding window in mind when comparing positions.

### Required changes

1. Add the teacher-forced NLL manipulation check ("gap closed" toward `s2300`) on 16 held-out non-reach prompts for
   `dpo`, `dpo` + α*v, each scrambled control and `s2300`, with the reading thresholds pre-registered in
   `STEER_SPEC.md` (B1).
2. Compute primary V_resp on `dpo`'s own stored responses; report its cosine with the `s2300`-text version (B2).
3. V_prompt includes the trailing template tokens up to and including the generation prompt; exclude only position 0
   and the leading, prompt-independent template block (B3).
4. Choose vector type, layer (from {16, 20, 24, 32, 40}) and α (from {0.5, 1, 2, 4, 8}) by dev teacher-forced NLL of
   `s2300`'s responses, not by dev pass@1. Sample on dev only the chosen setting, as a damage guard. Drop the 4-setting
   pass@1 grid and the α = 16 conditional. Always report the α = 1 NLL (B4).
5. Add E0′: fresh unsteered `dpo` on the 48 reach prompts × 8 through the patched zero-vector engine, and make it the
   reach baseline. Stored E0 stays the baseline for held-out non-reach, with its upward bias stated (B5).
6. E2 = three independent sign scrambles splitting the same 544 draws. Report the top-5-coordinate share of ||v||²
   (B5).
7. Score the stored `dpo` dev baseline with the same 12,000-token truncation as the dev cells (B6).
8. In `STEER_SPEC.md` and the write-up, state the resolution up front (held-out ± 6-8 points; reach about 2-3 % against
   E0′) and frame reach in checkpoint-equivalents (B7).
9. Correct the reading notes' two text errors: Venhoff's "layer 27" is the SAE layer, not the steering layer; Ward's
   "Venhoff say as much" claim about transferability is unsupported (A).

### Points I would accept being left OPEN

1. Whether E1 uses the NLL-chosen α* or the literal α = 1 when the two differ. I lean to α*, with α = 1's NLL reported.
   Running both is +$2.6.
2. The `s2300` − v necessity check as NLL (near free while `s2300` is loaded). I recommend it but do not require it.
   A sampled `s2300` − v cell stays out.
3. A vector from another checkpoint pair (`s0500` − `dpo`), as a conditional follow-up only if E1 beats E2.
4. Whether Heo's addition was single-position or every step (the text is silent; it does not affect our design).
5. Whether the 32 vector prompts are enough; this is decided by the split-half cosine, reported rather than fixed.
6. Whether scrambling is still a neutral control if the top-5 share of ||v||² is very high. If it is, the write-up
   says E2 is a damage control, and no extra cell is added.

## Turn 3 — reader-implementer (Claude Sonnet 5.5), 2026-11-27 01:57 PT

Labels as in Turn 1: [read], [feas], [mine], [guess]. Page numbers are PDF pages. The three reading notes now carry dated "Correction after audit" sections at their ends; nothing above them was edited.

### A. Reading disagreements

- **Heo, layer of D's probe.** Concede: p6 says only that "this adjustment was applied to the representations in the last layer"; my note already said the probe layer is not stated.
- **Heo, Phi-3 baseline.** Agreed (0.655 from Table 10 against 0.71 in Table 4). Unexplained.
- **Heo, repeat at every step.** Concede the text supports neither reading (p6). My note's "likelier reading is every step" was a guess and is now marked as one in the correction. It does not change the design (we add at every position).
- **Heo, Table 7 AUROCs of 0.10-0.13 and the QR definition.** Accept as added points (p17, p7); the first is in the correction.
- **Venhoff, steering layer.** Concede. Re-check: App. G.8 (p40) title "Open-Reasoner-Zero-32B (Layer 27, Dict Size 15)" with "Categories are identified by our SAE"; App. E.2 (p19) has "the thinking model's hidden state at the SAE layer" and "h_p is the base model's residual-stream activation at the steering layer". The steering layer is not reported. Corrected in the note.
- **Venhoff, "11 %".** Concede. My Table 1 recomputation: R1-Distill pairs, 12 cells (GSM8K -3.0, -3.4, 22.2, 31.8; MATH500 5.7, 1.2, 18.3, 14.8; Hendrycks -8.5, -5.5, 15.4, 12.5) average 8.5 %; with QwQ's three cells (29.1, 14.3, 16.3) 10.7 %. The abstract's 11 % is the second, not "SFT-distilled" alone. Corrected.
- **Venhoff, gate leakage and bare-prompt gap.** Accept both as added caution (p5 gate on disagreement; Table 5 p22, ORZ-32B alpha about 10 +- 7; p20 prompt). They lower the 76 % prior further for a static vector.
- **Ward, "Venhoff say as much".** Concede, unsupported (quote check: Venhoff never moves a vector between models, p19). Marked a guess in the correction.
- **Ward, whose text.** Concede. Quote: "we derive steering vectors separately on residual stream activations from both the base and reasoning models on the same reasoning traces" (p2, section 3), traces generated by R1-Distill (p7, App. A), vectors added to R1-Distill. The text is the receiver's own. Turn 1 used the donor's text and misattributed it to Ward. Accepted into the design (B2).
- **Ward, Table C.1 precision falls 63.36 % to 47.66 %** (p8). Accept: a cheap proxy is itself inflated by steering; no behavioural proxy is added to the design.

### B. Reply to the required changes

1. **NLL manipulation check: accept.** It turns a null into a measured quantity and is the same objective Venhoff trains on (p19). Details and thresholds in C5. I add one control the auditor did not ask for: NLL of `dpo`'s own text under `dpo` + alpha*v, so damage to `dpo`'s fluency is visible without sampling.
2. **V_resp on `dpo`'s own text: accept.** It is Ward's arrangement and it removes the off-policy-reading confound I flagged. The `s2300`-text version is computed in the same passes; cosine < 0.5 is reported, and I do not switch the primary (OPEN 2).
3. **V_prompt keeps trailing template tokens: accept.** Heo's "first token" is the state after the whole prompt (p4), which here ends in the generation prompt; I wrongly treated all template tokens alike. Only the leading, prompt-independent block and position 0 are excluded.
4. **Select by dev NLL, not dev pass@1: accept, with two modifications of mine.**
   (a) NLL text is `s2300`'s strict-pass draws on the dev prompts (one per prompt, prompts where `s2300` has a pass), not any draw. The target is the behaviour we want recovered, not fluent `s2300` text in general. The auditor said one stored draw per prompt; this is a narrowing, not a reversal.
   (b) A damage constraint is part of the rule: NLL of `dpo`'s own dev text may not rise by more than 0.1 nats per token [guess for the threshold].
   I also accept that dev sampling is a guard only, and that the 4-setting grid and the alpha = 16 conditional are dropped.
5. **E0' fresh reach cell: accept.** The reach set was defined by 0 passes in 16 stored draws per prompt (768 draws), so 0.0 % is the selection sample; a fresh baseline is the right comparison. My Turn 1 power statement was against a baseline that is zero by construction. I take the correction.
6. **E2 = three scrambles: accept, but make E2 conditional (argued below, C6).** I concede one draw is too few (Heo's 0.54 +- 0.12, Table 4; Venhoff's -28 % and -13 %, Fig. 4 p8). The conditional run keeps three scrambles when it fires.
7. **Score dev baseline at the same 12,000 cap: accept** (free from stored lengths).
8. **State resolution and frame in checkpoint equivalents: accept.**
9. **Fix the two text errors: done** (corrections appended).
10. **Smaller points: accept** (split-half cosine, 6,144-token pool cap, sliding window). I add one rule: split-half cosine < 0.5 means stop before sampling and ask the orchestrator (saves about $12); 0.5-0.8 proceeds with the flag.

### C. Revised design (self-contained)

**C1. Convention.** Residual after l layers = HF `hidden_states[l]` = output of `layers[l-1]`, l in 1..63 [feas s1]. All additions go on the output of `layers[l-1]`, in bf16, at every position (prompt, template, generated). Alpha multiplies the raw mean difference: alpha = 1 adds exactly the mean difference. Log r = alpha*||v|| / median||h_l|| over `dpo`'s generated positions.

**C2. Vectors.**
- Prompts: the first 32 of `pool/train.jsonl` (arm-A non-reach; to be confirmed against `pool/MANIFEST.json`). Fixed split seed 0 over the other 144 non-reach arm-A prompts: **dev 64**, **held-out 80**. Within dev, 16 prompts (seed 0) are the NLL-selection prompts. Within held-out, 16 prompts (seed 0) are the NLL manipulation-check prompts (their text is used for NLL only; they stay in the sampled held-out set because NLL shows no pass results).
- **V_resp (primary):** for each vector prompt, one stored `dpo` draw (Dan's, chosen by seed 0, truncated to 12,000 tokens). Mean over response positions (excluding position 0) of (`s2300` hidden - `dpo` hidden) at layer l, per prompt, then across prompts (Ward Eq. 3, p7), both models reading `dpo`'s text. fp32 sums.
- **V_resp,s2300text (diagnostic only):** same, on the pool's `s2300` responses (capped 6,144 tokens). Report the cosine with V_resp. Not a cell.
- **V_prompt:** user-content tokens plus trailing template tokens through the generation prompt; excludes position 0 and the leading template block. Positions are identified by index from the tokenizer template and checked by decoding.
- Audit group means (position 0, leading template, user, trailing template, response) are saved. Report: split-half cosine (16 vs 16 prompts) per vector; `||v||`; cosine V_resp vs V_prompt; top-5-coordinate share of `||v||^2`.

**C3. Pre-flight** (unchanged from feasibility s4): stock noise floor, zero-vector identity, steered-vs-HF agreement at l = 24 and one other layer, compile vs eager, `STEER_ACTIVE`. Abort on failure.

**C4. Selection by teacher-forced NLL (HF, no sampling), written before any run.**
- Text: the 16 NLL-selection dev prompts, one strict-pass `s2300` draw each (prompts where `s2300` has none are replaced by the next dev prompt in seed order).
- Grid: V_resp at l in {16, 20, 24, 32, 40} x alpha in {0.5, 1, 2, 4, 8}, then V_prompt at the best V_resp layer x the same five alphas. 30 forwards.
- Score = mean per-token NLL of the `s2300` text under `dpo` + alpha*v (hook on `layers[l-1]`; never read recorded hidden states, [feas s1]).
- Rule: choose the minimum NLL among settings that satisfy the damage constraint (`dpo`-own-text NLL rise <= 0.1 nats/token on the same 16 prompts' `dpo` draws). Ties within 0.005 nats go to smaller alpha, then smaller layer, then V_resp. If the minimum is at alpha = 8, do not extend; report it.
- Always record the alpha = 1 NLL for V_resp at l = 24 (the literal brief).
- Dev sampling guard on the chosen setting only: 64 dev prompts x 2 draws, 12,000-token cap, stored `dpo` dev baseline scored at the same cap. Pass requires: cap hits <= 10 %; mean length 0.7-1.3x `dpo`; dev pass@1 not more than 10 points below `dpo`. On failure, halve alpha once, re-check once; if it fails again, stop and report.

**C5. Manipulation check, pre-set thresholds (16 held-out NLL prompts, `s2300` strict-pass draws; these differ from the selection prompts).**
Conditions: `dpo`; `dpo` + alpha*v (chosen); each of three sign-scrambled copies; `s2300`; and `s2300` - alpha*v.
- **Gap closed g = (NLL_dpo - NLL_steered) / (NLL_dpo - NLL_s2300).**
- g < 5 %: the vector barely moves `dpo` toward `s2300`'s text; a pass@1 null is then expected and says nothing specific about instruction following.
- 5 % <= g < 20 %: moved slightly.
- g >= 20 %: moved substantially; if passes do not follow, the mean shift carries non-instruction-following parts of RL's change (auditor's reading, adopted).
- Necessity R = (NLL_{s2300 - v} - NLL_s2300) / (NLL_dpo - NLL_s2300). R > 2g with g small is read as Ward's asymmetry (the final model uses the direction downstream; `dpo` cannot read it) [guess for the factor 2].
- The scrambled controls' g is reported; a scrambled copy with g >= 0.5 x the real g means the real direction is not special in NLL.
- `dpo`-own-text NLL rise is reported for the chosen setting and the scrambles.
- Cost is small because the `s2300` model is already loaded for the means; the `dpo` HF session is shared with C4.

**C6. Sampled cells** (T 1.0, top_p 1.0, 32,768 tokens; strict checker; Dan's constants; patched engine, one process per cell).

| cell | what | reach (48) | held-out non-reach (80) | new draws |
|---|---|---|---|---|
| E0 | `dpo` stock, Dan's stored draws, 16 per prompt | reused only as a drift check, not the baseline | **baseline** (biased slightly upward, which works against E1) | 0 |
| E0' | `dpo`, patched engine, zero vector | 8 | none | 384 |
| E1 | `dpo` + alpha*v (C4 choice) | 8 | 2 | 544 |
| E2 (conditional) | `dpo` + three independent sign scrambles of alpha*v, same total draws, round-robin | 2-3 per scramble | 1 each, pooled | 544 |

- E0' also serves as the end-to-end alpha = 0 check.
- **Run order:** E0', E1, then E2 only if E1 is not a null: E2 runs when E1's reach passes exceed E0''s by 2 or more, or when E1's held-out pass@1 differs from E0 by 6 points or more in either direction (a gain needs a control; a loss needs to be shown not to be generic perturbation). Otherwise E2 is skipped and the null is reported as a null with the NLL numbers. [mine; this is where I diverge from the auditor's unconditional E2.]
- Metrics: pass@1 and pass@8 on reach; pass@1 on held-out; % of prompt set, 95 % prompt-bootstrap intervals, paired where cells share prompts; cap-hit rate; mean length; unparsable share. Report reach in checkpoint equivalents: `s0100` 0.1 %, `s0500` 4.5 %, `s1100` 38.5 %; 60-word hint 6.2 %; LoRA 63.5 % [from CONTEXT.md].
- **Stated resolution up front:** held-out non-reach, 80 x 2 against 16 stored, is about +-6-8 points on the paired pass@1 difference [auditor's computation, adopted]. Reach: a steered rate of about 2-3 % spread over at least 3 prompts is separable from E0' (expected 0.1-0.5 %); smaller effects are not. So the design answers "does the vector reach at least `s0500`'s reach level"; it does not resolve smaller changes.
- Watchdogs: STALL (no new tokens for 10 min), OVERRUN (1.6x the expected wall time: dev 22 min, E cells 45 min), EXIT; unfinished draws are scored as failures and flagged; check-ins every 30 min.

**C7. Cost (assumptions as Turn 1: $5.29/h, 4,500 tokens per draw, 1,500 tok/s at high concurrency, about 2 min startup per process, HF prefill throughput unmeasured).**

| item | full design $ | capped at $15 $ |
|---|---|---|
| setup, download, means (both texts), verification | 5.3 | 5.3 |
| `s2300` NLL references, necessity (in the same session) | 0.3 | 0.3 |
| `dpo` HF NLL sweep (30 forwards), manipulation check, reload | 2.3 | 2.3 |
| dev guard, chosen setting, 128 draws | 1.2 | 0.65 (64 x 1) |
| E0' reach, 384 draws | 1.9 | 0.95 (192) |
| E1, 544 draws | 2.6 | 1.9 (held-out 80 x 1: 464 draws) |
| E2, 544 draws, 3 processes (conditional) | 2.9 | 2.2 (464 draws) |
| loop and tail allowance | 1.8 | 1.2 |
| **total if E2 runs** | **18.3** | **14.8** |
| **total if E1 is a null and E2 is skipped** | **15.4** | **12.6** |

- The auditor's estimate is $17-18.5; mine is the same at the top. If the orchestrator caps M3 at $15 [guess that this is a hard cap], I cut in this order:
  1. **Keep E2 conditional** (already in the full design; saves $2.9 in the null case). Interpretive cost: in a clean null there are no scramble draws, but there is nothing for the control to control; the NLL scrambles are still reported.
  2. **Held-out non-reach 80 x 2 -> 80 x 1** in E1 and E2 (saves about $1.4). Cost: the damage check on non-reach drops to about +-10 points; NLL damage numbers cover part of it.
  3. **Dev guard 128 -> 64 draws** (saves about $0.55). Cost: a weaker guard (about +-6 points SE); a damaged setting is caught only if gross.
  4. **E0' 384 -> 192 draws** (saves about $0.95). Cost: the reach baseline then has about 0-1 passes expected, so the "separable from E0'" bound loosens to about 3-4 % rather than 2-3 %.
- I would not cut the NLL block (cheapest evidence per dollar) or E0' to zero, and I would not cut reach draws on E1. If the cap is lower than $13, the cheapest honest M3 is: means, NLL sweep and manipulation check only (about $8), with sampling skipped and reported as not done.

**C8. What each outcome shows.**
- g >= 20 % and reach at or above `s0500`'s level, scrambles flat: a static mean difference at one layer moves `dpo` toward `s2300`'s text and unlocks reach-level behaviour. Does not show RL is this vector, that it is optimal (Venhoff's learned, gated vectors are stronger), that it acts at other layers, or that checker passes are genuine.
- g >= 20 % but reach near E0': the shift carries other parts of RL's change (style, length) and not the IF part; this is the case a pass-only design could not separate.
- g < 5 % across the grid: at layer 16-40 and alpha up to 8 the mean difference does not move `dpo`'s predictions; a pass null follows trivially. It does not show that no direction exists (mean difference, not learned; ungated; no position search; one static vector).
- g small with R large: Ward's asymmetry (the final model needs the direction downstream; `dpo` cannot read it). That would be the informative null.
- Damage (NLL rise on `dpo`-own text above 0.1 nats, cap hits, length): the strength that moves anything also breaks the model.
- Any E1 gain with E2 gains too: nonspecific perturbation (Ward p3 saw noise move behaviour); no claim.

### OPEN (still disputed between the auditor and me, or newly raised)

1. **NLL selection text.** I narrow the NLL text to `s2300` strict-pass draws; the auditor asked for one stored `s2300` draw per prompt. Passes-only may make the selection more IF-specific but loses prompts and may overweight the formatting tokens common to passing responses [guess].
2. **Primary vector.** V_resp on `dpo`'s text stays primary even if its cosine with the `s2300`-text version is below 0.5; the auditor said report, not switch. Which one is "right" is unresolved.
3. **Conditional E2.** The auditor's list has E2 unconditional. I argue it is wasted in a clean null; the trigger rule (+2 reach passes over E0' or a 6-point held-out change) is mine.
4. **Split-half stop rule.** I add "stop below 0.5"; the auditor wrote only "state as a finding".
5. **Damage constraint threshold** (0.1 nats/token) and R > 2g are guesses with no calibration.
6. **NLL-optimal is not pass-optimal.** Choosing by NLL, not by checker, may select a setting that is best for the final model's style rather than IF compliance; the design cannot test this without a pass@1 sweep that Turn 2 showed is noise-dominated at this budget.
7. **One layer family only** (grid 16-40); no claim about the other layers or about learned vectors.
8. **E0 for reach** is dropped as the baseline; whether the stored draws also count as a second-order check on drift (they would show 0 by construction) is moot.

## Turn 4 — auditor (Claude Opus 5.5), 2026-11-27 02:00 PT

Round 2. I checked Turn 3 §C as written, and the "Correction after audit" sections appended at 01:56 to the three
notes, against my nine Turn 2 requirements. Then I attacked what is new.

### A. Were the nine required changes implemented?

| # | requirement (Turn 2) | where in Turn 3 | status |
|---|---|---|---|
| 1 | NLL manipulation check, 16 held-out prompts, all conditions, thresholds fixed in advance | C5 | **Implemented**, with one ordering defect: the `s2300` − v term (B4 below). |
| 2 | V_resp on `dpo`'s own stored text; cosine with the `s2300`-text version | C2 | **Implemented.** |
| 3 | V_prompt keeps the trailing template tokens | C2 | **Implemented.** Positions are found by index and checked by decoding. |
| 4 | Select type, layer and α by dev NLL; one dev guard setting; drop the pass@1 grid and α = 16 | C4 | **Implemented.** Two additions are attacked below: pass-only text and the 0.1-nat constraint. α = 1 NLL is recorded only for V_resp at l = 24 on the selection text (B1). |
| 5 | E0′ fresh reach baseline | C6 | **Implemented** (48 × 8; 192 only under the cap). |
| 6 | E2 = three scrambles, unconditional | C6 | **Not implemented as required.** It is now conditional. I accept the condition with two specifications (B5). |
| 7 | Dev baseline scored at the 12,000 cap | C4 | **Implemented.** |
| 8 | Resolution stated up front; checkpoint-equivalent framing | C6 | **Implemented.** |
| 9 | Fix the notes' two text errors | the notes' correction sections | **Implemented.** All three notes carry dated corrections that match the text (Venhoff p. 19 SAE vs steering layer; 8.5 % / 10.7 %; Ward's "say as much" marked unsupported; Ward's receiver-text arrangement). |

### B. Attack on what is new

**B1. NLL selection: over-fitting, and whether NLL-optimal is the right target.**

- *Over-fitting.* Selection uses **16** dev responses (not 32), about 100k tokens, but the effective sample size is 16
  prompts. Over-fitting cannot inflate the reported effect, because g is measured on 16 different held-out prompts
  (C5). The only risk is picking a setting that is not the population optimum. The NLL surface over a 5 × 5 grid is
  probably smooth in α, so a mis-pick most likely lands on a neighbouring setting at small loss. **Acceptable as is.**
- *Is NLL-optimal the right target?* The NLL of `s2300`'s samples under the steered model is the forward
  cross-entropy, so minimising it minimises KL(`s2300` ‖ steered) on `s2300`'s own text. That is the right target for
  "does the vector move `dpo` toward the final model". It is **not** the target "does it recover instruction
  following", for three reasons:
  - it is token-weighted, so thousands of thinking tokens outweigh the few hundred answer tokens the checker reads;
  - it is teacher-forced, so loops and drift that appear only when `dpo` generates its own prefixes are invisible;
  - it is mass-covering: steered text that `s2300` would never write costs nothing on `s2300`'s text.

  Turn 3's damage constraint covers part of the third. The dev guard covers the second. The first is uncovered and
  costs nothing to expose.
  - *Change required:* report every NLL quantity separately for **thinking tokens** and **answer tokens** (after the
    end-of-thinking marker), for the full grid and for g in C5. Same forward passes, a different average. Selection
    stays on total NLL. If g on answer tokens is near 0 while g on thinking tokens is large, the write-up predicts
    "style moved, compliance did not" before the pass results are read.
- *α = 1 on held-out text.* α = 1 is the literal brief. C4 records it only for V_resp at l = 24 on the selection text.
  - *Change required:* also report g at α = 1 at the **chosen** layer and vector type on the held-out NLL text (one
    extra forward). Report the full 30-setting grid on the selection text, not only the winner. The grid shows where
    NLL starts to rise with α, which is the dose evidence a null needs.
- *Pass-only selection text (Turn 3 OPEN 1).* Narrowing to `s2300`'s strict-pass draws aims the target at the
  behaviour we want and uses no `dpo` or reach outcome. **Acceptable as is.** I withdraw my "any draw" wording. Keep the
  same rule for the C5 held-out text, so g's numerator and denominator are on comparable text.

**B2. The 0.1 nats/token damage constraint is uncalibrated.**

The scale that matters is how far RL itself moved the model. If the per-token NLL gap between `dpo` and `s2300` is a
few hundredths of a nat (unknown; the weights moved 0.364 %), then 0.1 nats on `dpo`'s own text would allow damage
several times larger than RL's entire change. If the gap is large, 0.1 might bind too early. A data-defined reference
is available for one extra forward while `s2300` is loaded: **D_rev = NLL_s2300(`dpo` text) − NLL_dpo(`dpo` text)** on
the 16 selection prompts' `dpo` draws. It measures how far the final model itself departs from `dpo`'s own behaviour.

*Change required:*
- The constraint becomes: rise ≤ 2 × D_rev (the factor 2 is my guess, labelled so). Report D_rev.
- If no setting with α ≥ 1 passes at any layer, report that as a finding: a constant shift costs more than RL's whole
  change does. E1 then uses α = 1 at the layer with the smallest rise, with the dev guard as the backstop.

**B3. Manipulation-check thresholds and the 16 held-out prompts.**

The 5 % / 20 % bands are mine and I keep them. But g is a ratio of two prompt-averaged differences over 16 prompts.
If the denominator (the `dpo`→`s2300` gap) is small, g is noisy, and a point estimate can land in the wrong band.

*Change required:*
- Report g with a 95 % prompt-bootstrap interval (16 prompts, resampled jointly for numerator and denominator).
- Assign a band only when the interval lies inside it; otherwise write "between bands".
- 16 prompts is **acceptable** given the interval. Going to 32 held-out prompts costs about 8 more forwards (≈ $0.5),
  which I would accept but do not require.

**B4. Ordering defect in the necessity term.**

C5 computes NLL under `s2300` − α·v "in the same session" as the means. But α and the layer are chosen later, in the
`dpo` NLL sweep, after `s2300` has been unloaded (C7 lists a `dpo` reload, not an `s2300` one; feasibility §5 deletes
`s2300` after the means). As written, R cannot be computed at the chosen setting.

*Change required:* while `s2300` is loaded, compute `s2300` − 1·v_l for V_resp at all five grid layers on the 16
held-out texts (5 forwards). Report R at α = 1 for the chosen layer. That needs no reload and matches the literal
brief. Drop R at α*. The "R > 2g" reading then compares R(α = 1) with g(α = 1) from B1. The factor 2 stays a labelled
guess (OPEN).

**B5. Conditional E2: is a clean null interpretable without it?**

Yes, mostly.
- In a null (E1 ≈ E0′ on reach, |Δ| < 6 points held-out), a sampled scramble could only show something if it *moved*
  while the real vector did not. That would mean the real direction suppresses a generic perturbation benefit; the prior
  is low, and it would not change the answer to Leon's question.
- Specificity in the null case is still covered in NLL by the three scrambles' g in C5.
- The trigger rules (reach +2 passes over E0′, which is lenient given Poisson noise at about 1 expected pass; or a
  held-out move of 6 points or more either way) err toward running E2.

I concede the conditional design, with two specifications:
- *Change required (a):* the trigger is mechanical and decided on the pod. Grade E1 with Dan's checker on the pod's
  CPU immediately after E1 ends and apply the rule without human input. Otherwise E2's saving is spent on pod idle time
  or a second pod start (download plus load, about $1-2).
- *Change required (b):* if E2 is skipped, the write-up states "specificity was tested in NLL only (scrambles' g), not
  in sampled behaviour".

**B6. Split-half stop rule (cosine < 0.5 → stop and ask).**

**Acceptable**, but the threshold is more conservative than it looks. Assume isotropic noise and two independent
half-means with the same noise. A split-half cosine c implies that the full 32-prompt vector has cosine about
√(2c/(1+c)) with the population mean (my computation): c = 0.5 → 0.82; c = 0.3 → 0.68; c = 0.2 → 0.58.

*Change required (small):*
- The stop message to the orchestrator carries c, the converted value, and the same two numbers with the top-5
  coordinates removed, since a few consistent massive dimensions can make c look high while the rest is noise.
- The rule is evaluated at the **chosen** layer and vector type, after selection and before any sampling, not at a
  fixed layer.

**B7. HF hook or patched vLLM with `prompt_logprobs` for the NLL pass.**

- *Use HF with a forward hook on `layers[l-1]` for every NLL number*, and read only log-probs (feasibility §1 trap).
- Reasons:
  - The sweep needs about 40 vector swaps. Route A is one vLLM process per setting, about 2 min of start-up each,
    which is about 80 min of start-up alone.
  - In-process swapping in vLLM is possible (`apply_model` + `copy_`, prefix caching off, `enforce_eager`), but it is
    exactly the stale-state path the feasibility agent advised against.
  - g mixes the `s2300` reference (HF, computed during the means session) with steered `dpo` numbers. Engine-level
    bf16 differences (the feasibility file guesses about 1e-2 nats/token mean) could be a sizeable fraction of a small
    `dpo`→`s2300` gap. **All terms of g, R and D_rev must come from one engine**, and HF is the one already in use for
    the means.
- *Change required:* state "one engine (HF) for all NLL quantities".
- *Change required:* add a throughput rule. HF prefill speed is unmeasured and drives this block's cost, which ranges
  from about $1.5 at 6k tokens/s to about $6 at 1.5k tokens/s (my estimate for about 6M prefill tokens including the
  damage forwards). Time the first forward. If the projected NLL block exceeds $3, shrink the grid in a fixed order:
  drop α = 0.5, then l = 40, then the V_prompt row. Log the shrink.

**B8. Steering-correctness pre-flight.**

C3 keeps feasibility §4 unchanged. It tests steered vLLM against HF at l = 24 and one other layer, but the layer is now
chosen from 16-40 by C4.

*Change required:*
- Run the steered-vs-HF-hook check (feasibility §4 step 3, including a sequence longer than 4,096 tokens) at the
  **chosen** (vector type, layer, α) and on one scramble, after selection and before the dev guard.
- Compute the HF reference log-probs for those four sequences in the same HF `dpo` session as the sweep, so no extra
  load is needed.
- Keep `STEER_ACTIVE` plus the vector hash in every sampling process's sidecar (E0′'s zero vector included).

The resulting order is:
1. HF `dpo` means;
2. HF `s2300` means, references, D_rev and necessity at α = 1;
3. HF `dpo` sweep, manipulation check and pre-flight references;
4. vLLM pre-flight;
5. dev guard;
6. E0′ and E1, then E2 if triggered.

**B9. Cost and cut order under a $15 cap.**

- Turn 3's arithmetic checks out: $18.3 / $15.4 full; $14.8 / $12.6 capped. My changes add about $0.5: 5 necessity
  forwards, 1 D_rev forward, 1 α = 1 held-out forward, and the pre-flight moved, not added.
- The unmeasured HF throughput is a larger risk than any listed cut, hence B7's rule.
- Steered responses may run longer than 4,500 tokens if g is large (toward `s2300`'s 6.7k). E1 and E2 could then cost
  up to 1.5×; the loop allowance covers only part of that.
- *Cut order I would use* (acceptable either way):
  1. E2 conditional;
  2. dev guard → 64 × 1 (the NLL damage constraint now does most of its job; it remains only to catch free-generation
     loops);
  3. held-out 80 × 2 → 80 × 1 in E1 and E2;
  4. the NLL-grid shrink from B7;
  5. E0′ → 192 last.
- Never cut the NLL block to zero, E1's reach draws, or E0′ to zero (agreeing with Turn 3).
- Turn 3's floor (means, NLL sweep and manipulation check only, about $8, no sampling) is the right fallback if the cap
  falls below about $13.

**B10. Smaller points, acceptable as is.**

- Pass-only selection text.
- The tie rule (0.005 nats).
- No extension beyond α = 8.
- The dev-guard rule (halve once, then stop).
- The held-out NLL prompts staying in the sampled held-out set.
- V_resp on draws capped at 12,000 tokens.
- Keeping V_resp primary even if its cosine with the `s2300`-text version is below 0.5 (report it; this is the Ward
  arrangement).

### Required changes

1. Report all NLL quantities split into thinking tokens and answer tokens, for the full selection grid and for g, R
   and the scrambles' g. Selection stays on total NLL (B1).
2. Report the full 30-setting grid on the selection text, and g at α = 1 at the chosen layer and vector type on the
   held-out NLL text (B1).
3. Replace the 0.1 nats/token damage constraint with rise ≤ 2 × D_rev, where D_rev = NLL_s2300(`dpo` text) −
   NLL_dpo(`dpo` text) is measured while `s2300` is loaded. If no α ≥ 1 setting passes, report it and run E1 at α = 1
   on the least-damaging layer (B2).
4. Report g with a 95 % prompt-bootstrap interval; assign a band only when the interval lies inside it (B3).
5. Fix the necessity ordering: compute `s2300` − 1·v_l at all five grid layers while `s2300` is loaded; report
   R(α = 1) at the chosen layer, not R(α*) (B4).
6. Make the E2 trigger mechanical and on the pod (grade E1 on the pod right after it ends). If E2 is skipped, state
   that specificity was tested in NLL only (B5).
7. Evaluate the split-half rule at the chosen layer and vector type after selection. Report c, √(2c/(1+c)) and the
   top-5-removed versions in the stop message (B6).
8. Use one engine (HF, hook on `layers[l-1]`, log-probs only) for every NLL quantity. Time the first forward and shrink
   the grid in the fixed order if the NLL block projects above $3 (B7).
9. Move the steered-vs-HF pre-flight to the chosen (type, layer, α) plus one scramble, after selection and before any
   sampling, with the HF references from the same `dpo` session. Keep `STEER_ACTIVE` and the vector hash in every
   sampling sidecar (B8).

### Points I would accept being left OPEN

1. The factors in "rise ≤ 2 × D_rev" and "R > 2g" are uncalibrated guesses. They are labelled; no pilot to tune them.
2. NLL-optimal versus pass-optimal (Turn 3 OPEN 6). Required change 1 exposes the main failure mode, and a pass-based
   selection is unaffordable at this resolution.
3. The cut order: Turn 3's or mine (B9). Either is acceptable.
4. 16 versus 32 held-out NLL prompts. 16 with an interval is acceptable; 32 costs about $0.5.
5. Which V_resp text is "right" when the two cosines disagree (Turn 3 OPEN 2). Report, do not switch.
6. One layer family (16-40) and one static vector (Turn 3 OPEN 7). This is a stated limit of the measurement, not a
   defect of the design.
7. Whether E2's trigger thresholds (+2 reach passes, ±6 points held-out) are the right ones. They err toward running
   E2, which is the safe direction.

## Turn 5 — reader-implementer (Claude Sonnet 5.5), 2026-11-27 02:03 PT

Labels: [mine], [guess], [aud] = adopted from the auditor's Turn 4. Page references as in the notes. Section F below is the final design and is meant to be read without the earlier turns.

### A. Reply to the nine required changes

1. **Thinking vs answer NLL split: accept.** Every NLL quantity (grid, g, R, scrambles' g, D_rev) is reported on thinking tokens and answer tokens (after the end-of-thinking marker; the build agent reads the exact marker string from the `olmo_thinker` template and records it). Selection stays on total NLL. The auditor's rationale is right: thousands of thinking tokens outweigh the few hundred the checker reads.
2. **Full 30-setting grid, and g at alpha = 1 at the chosen layer and type on the held-out text: accept.** One extra forward.
3. **Damage rule = rise <= 2 x D_rev: accept.** The factor 2 is a guess, labelled. D_rev = NLL_s2300(`dpo` text) - NLL_dpo(`dpo` text) on the 16 selection prompts' `dpo` draws, measured while `s2300` is loaded. If no alpha >= 1 setting passes anywhere, E1 runs at alpha = 1 on the layer with the smallest rise and the finding is reported. I do not add a floor for a very small D_rev: if D_rev is tiny, then "any constant shift costs more than RL's whole change" is itself the result.
4. **g with a 95 % prompt-bootstrap interval, band only if the interval lies inside it: accept.** 16 held-out NLL prompts stay (32 would cost about $0.5; not taken).
5. **Necessity ordering: accept, and one extension.** While `s2300` is loaded, compute `s2300` - 1*v_l on the held-out NLL text for **both** V_resp and V_prompt at all five grid layers (10 forwards, not 5), because V_prompt can win selection and then R(alpha = 1) would otherwise be missing. R is reported at alpha = 1 only, compared with g(alpha = 1).
6. **Mechanical E2 trigger on the pod: accept both specifications.** E1 is graded on the pod's CPU with Dan's grader immediately after it ends; the rule is applied by a script. If E2 is skipped, the write-up says "specificity was tested in NLL only (scrambles' g), not in sampled behaviour".
7. **Split-half rule at the chosen layer and type, with converted numbers and top-5-removed versions: accept.**
8. **One engine (HF, hook on `layers[l-1]`, log-probs only), timing rule, fixed shrink order: accept.** I assume 3,000 prefill tokens/s for the budget [guess]; the first forward is timed and the rule applies.
9. **Pre-flight at the chosen setting plus one scramble, after selection, before sampling, with HF references from the same `dpo` session: accept.** `STEER_ACTIVE` and the vector hash go into every sampling sidecar.

Cut order: I adopt the auditor's (B9): E2 conditional, dev guard to 64 x 1, held-out to 80 x 1, NLL grid shrink, E0' to 192. It differs from my Turn 3 order only in putting the dev guard before the held-out reduction; I agree the NLL damage rule now does most of the dev guard's job.

### F. FINAL DESIGN (self-contained)

**F0. Question and scope.** Does adding one static vector, the mean activation difference between the final model (`s2300`) and the pre-RL model (`dpo`), to `dpo` recover part of the final model's behaviour on IFBench (official strict checker; "checker passes", not verified genuine)? One vector family, layers 16-40, no learned or gated vectors. Never choose any setting on reach results.

**F1. Conventions.**
- Residual after l layers = HF `hidden_states[l]` = output of `layers[l-1]`; l in 1..63 (`hidden_states[64]` is post-final-norm and unusable). Addition = bf16 buffer added to the output of `layers[l-1]` at every position (prompt, template, generated). alpha multiplies the raw mean difference: alpha = 1 adds exactly the mean difference. Log r = alpha*||v|| / median ||h_l|| over `dpo`'s generated positions.
- NLL numbers come only from HF with a forward hook, reading log-probs, never recorded `hidden_states` (feasibility s1). Same token ids as the sampler's `prompt_ids` (chat template included).
- Seeds: split seed 0; scramble seeds 2001, 2002, 2003; engine seeds E0' 1000, dev guard 3000, E1 1001, E2 scrambles 3001, 3002, 3003.

**F2. Prompts (by file and index; manifest written before any run).**
- Arm-A non-reach set N = `pids_armA.txt` minus `pids_reach.txt` = 176 prompts (sorted by pid).
- **Vector prompts V32:** lines 0-31 of `/scratch/cowifornia/nd-rl/olmo_elicit/pool/train.jsonl` (verify membership in N and the pid field against `pool/MANIFEST.json`; if any is not in N, take the next lines until 32 are).
- Remaining 144 = N minus V32, sorted by pid, permuted with `numpy.random.RandomState(0).permutation`. First 64 = **dev**; next 80 = **held-out**.
- **NLL-selection prompts (S16):** the first 16 dev prompts in permutation order that have a strict-pass `s2300` draw. **NLL-check prompts (H16):** the first 16 held-out prompts in permutation order with a strict-pass `s2300` draw. H16 remain in the sampled held-out set (NLL only uses text).
- **Reach set:** `pids_reach.txt` (48). No reach prompt enters V32, dev, S16, H16 or any selection step.
- **Texts.** `dpo` text: for each V32 and S16 prompt, one stored `dpo` draw (draw index chosen by seed 0), truncated at 12,000 tokens. `s2300` text for NLL: for each S16 and H16 prompt, the first strict-pass stored `s2300` draw in draw-index order (Dan's draws in `artifacts/olmo_ifcurve/`, grades from `olmo_ifomega`; the build agent resolves paths and writes them to the manifest). `s2300` text for the diagnostic vector: the pool's `s2300` responses for V32 (capped at 6,144 tokens).

**F3. Vector definitions (computed for all l in 1..63; grid layers L = {16, 20, 24, 32, 40}).**
- **V_resp (primary):** for each V32 prompt, over the `dpo` text's **response positions** (not position 0), mean of (h_s2300[l] - h_dpo[l]) with both models reading the same `dpo` text; mean within a prompt, then across the 32 prompts (Ward Eq. 3, p7). fp32 sums.
- **V_resp,s2300text (diagnostic only):** same on the pool's `s2300` responses. Report its cosine with V_resp (report, never switch primary).
- **V_prompt:** user-content tokens plus **trailing template tokens through the generation prompt**; excludes position 0 and the leading (prompt-independent) template block. Positions identified by index from the tokenizer template and checked by decoding.
- Saved for audit: group means (position 0, leading template, user, trailing template, response) and per-prompt means, for all layers (about 5 groups x 63 x 5,120 fp32). Report: split-half cosine (prompts 0-15 vs 16-31), `||v||`, cos(V_resp, V_prompt), top-5-coordinate share of `||v||^2`.
- **Scrambles:** a sign scramble of alpha*v multiplies each coordinate by an independent random sign (seed 2001/2002/2003); same norm, same coordinate-magnitude profile.

**F4. Order of operations on the pod (one H200, secure $5.29/h; Route A for vLLM).**
1. **Setup.** Download `dpo`; HF stock-vs-vLLM noise floor and zero-vector identity (feasibility s4 steps 1, 2) are deferred to step 9 so they run once.
2. **HF `dpo` session (P1).** Forward V32 `dpo` text and `s2300`-text prompts: accumulate per-layer group sums (fp32) for `dpo`. Unhooked NLL of `dpo`: on S16 `s2300` text, H16 `s2300` text, S16 `dpo` text. Save to disk; unload.
3. **Download `s2300`; HF `s2300` session (P2).** Same sums for `s2300`; form V_resp, V_prompt and the diagnostic vector for all l. Compute while loaded: unhooked NLL of `s2300` on S16 `s2300` text, H16 `s2300` text, S16 `dpo` text; **D_rev = NLL_s2300(S16 dpo text) - NLL_dpo(S16 dpo text)** (total, thinking, answer); **necessity:** NLL of `s2300` - 1*v_l on H16 `s2300` text for V_resp and V_prompt at all five grid layers (10 forwards). Write vectors (1.3 MB each layer set) to disk; unload; delete `s2300` (vectors go back to serrano).
4. **HF `dpo` session (P3).**
   - Time the first hooked forward. Projected NLL-block cost = (forwards remaining x seconds per forward) x $5.29/3,600. **If above $3.00, shrink the grid in this fixed order until the projection is at or below $3.00: drop alpha = 0.5 from all rows, then drop l = 40, then drop the V_prompt row.** Log the shrink.
   - **Sweep:** for each setting, two hooked forwards: total NLL of S16 `s2300` text (the selection score) and NLL of S16 `dpo` text (the damage). Settings: V_resp x L x alpha in {0.5, 1, 2, 4, 8} (25), then V_prompt at the V_resp layer chosen by the rule below x the same alphas (5). 60 forwards. Report the full grid, total / thinking / answer.
   - **Selection rule (written now).** damage_rise(setting) = NLL_steered(S16 dpo text) - NLL_dpo(S16 dpo text). Admissible set A = settings with damage_rise <= 2 x D_rev [guess for the factor]. Among V_resp settings in A, the best layer = the layer holding the minimum-NLL admissible setting; run the V_prompt row at that layer. Over all 30, choose the admissible setting with minimum total NLL on S16 `s2300` text. Ties within 0.005 nats go to smaller alpha, then smaller layer, then V_resp. Do not extend beyond alpha = 8 (report if the minimum is at 8). If A is empty: chosen = V_resp at alpha = 1 on the layer with the smallest damage_rise; record "no admissible setting".
   - **Manipulation check on H16 `s2300` text.** Hooked forwards: chosen setting; V_resp or V_prompt at alpha = 1 at the chosen layer and type; three sign scrambles of the chosen alpha*v; plus unhooked `dpo` numbers from P1 and `s2300` numbers from P2. g = (NLL_dpo - NLL_steered) / (NLL_dpo - NLL_s2300), for total, thinking and answer tokens. Report g with a 95 % prompt-bootstrap interval (B = 10,000, seed 0, prompts resampled jointly for numerator and denominator over the 16). **Bands: g < 5 % "barely moved", 5-20 % "slightly", >= 20 % "substantially"; a band is assigned only if the whole interval lies inside it, else "between bands".** Report R(alpha = 1) = (NLL_{s2300 - v} - NLL_s2300) / (NLL_dpo - NLL_s2300) from P2 at the chosen layer and type, beside g(alpha = 1). R > 2g(alpha = 1) with g small is read as Ward's asymmetry [guess factor]. Report the scrambles' g (a scramble with g >= 0.5 x the real g means the real direction is not special in NLL) and `dpo`-own-text damage rise for the chosen setting and the scrambles.
   - **Split-half rule (at the chosen layer and type, after selection, before sampling).** c = cosine between the 16-prompt half means. If c < 0.5, stop sampling and message the orchestrator with c, sqrt(2c/(1+c)) [aud], and both numbers again with the top-5 coordinates removed. 0.5 <= c < 0.8 proceeds with the flag.
   - **Pre-flight references:** HF hooked log-probs on four fixed sequences (two ~1k tokens, one ~4.3k, one ~5.5k, from stored draws) for the chosen setting and for scramble 2001. Unload.
5. **vLLM pre-flight (patched Route A).** Stock noise floor and zero-vector identity (feasibility s4 steps 1-2). Then steered-vs-HF-hook agreement at the chosen setting and at scramble 2001 within the stock tolerance, while steered-vs-unsteered differences are far larger; compile vs eager at the chosen setting; prefix-cache repeat; `STEER_ACTIVE` plus vector hash in the sidecar. Abort on any failure.
6. **Dev guard.** Chosen setting, 64 dev prompts x 2 draws, 12,000-token cap, engine seed 3000. Stored `dpo` dev baseline scored at the same 12,000 truncation. Pass: cap hits <= 10 %; mean length within 0.7-1.3x of `dpo`; dev pass@1 not more than 10 points below `dpo`. On failure halve alpha once, rerun once (re-run the pre-flight of step 5 for the new alpha); on a second failure stop and report.
7. **E0'.** Zero vector through the patched engine, 48 reach x 8 draws (384), seed 1000, 32,768 cap. This is the reach baseline and the end-to-end alpha = 0 check.
8. **E1.** Chosen setting, 48 reach x 8 + 80 held-out x 2 (544 draws), seed 1001, 32,768 cap. Grade on the pod's CPU with Dan's grader immediately after it ends.
9. **E2 trigger (mechanical, script on the pod).** Run E2 iff **[reach passes in E1 - reach passes in E0' >= 2] OR [|held-out pass@1 of E1 - held-out pass@1 of stored E0| >= 6 points]**. E2 = three processes, one per sign scramble (seeds 3001-3003), same total draws as E1 (reach 8 draws per prompt split 3, 3, 2 across the scrambles; held-out 80 x 2 as 80 x 1 in scramble 1 and 80 x 1 in scramble 2). If not triggered, state in the write-up: "specificity was tested in NLL only, not in sampled behaviour".

**F5. Watchdog (every process).** `timeout` plus STALL (no new tokens for 10 min), OVERRUN (1.6x expected wall time: dev guard 22 min, each E cell 45 min), EXIT logged; per-request caps 12,000 (dev) and 32,768 (E cells); unfinished draws scored as failures and flagged; check-ins every 30 min; hard stop for M3 at $22.

**F6. Analysis (pre-registered).**
- Pass@1 and pass@8 (reach: E1 vs E0', 8 draws per prompt each), pass@1 (held-out: E1 vs stored E0, 16 draws per prompt), as % of the prompt set; 95 % prompt-bootstrap intervals (B = 10,000, seed 0), paired by prompt. Also cap-hit rate, mean length, unparsable share.
- Reach framed in checkpoint equivalents: `s0100` 0.1 %, `s0500` 4.5 %, `s1100` 38.5 %, 60-word hint 6.2 %, LoRA 63.5 %. Stated resolution: held-out about +-6-8 points paired [aud]; reach: a rate of about 2-3 % over at least 3 prompts is separable from E0' (expected 0.1-0.5 %); smaller effects are not resolved.
- Reading table: g >= 20 % and reach at or above `s0500`, scrambles flat: the static difference moves `dpo` toward `s2300` and unlocks reach-level behaviour (does not show RL is this vector, optimality, other layers, or genuine passes). g >= 20 % but reach near E0': the shift carries other parts of RL's change; read the answer-token g. g < 5 % across the grid: the mean difference does not move `dpo` in layers 16-40 at alpha <= 8 (not "no direction exists"). Small g with R > 2g: Ward-type asymmetry. Damage (rise > 2 x D_rev, cap hits, length): the strength that moves anything breaks the model. E1 gain with E2 gain: non-specific perturbation. Answer-token g near 0 with thinking-token g large predicts "style moved, compliance did not" before passes are read.
- All NLL numbers also tabulated at alpha = 1 (literal brief).

**F7. Cost (assumptions: $5.29/h; 4,500 tokens per draw; 1,500 tok/s at high concurrency, 2 min startup per process; HF prefill 3,000 tok/s [guess, timed on the pod]; about 6M NLL tokens).**

| item | full | after cuts |
|---|---|---|
| setup, downloads, P1 sums, vLLM noise floor | 5.3 | 5.3 |
| P2 `s2300` session: sums, references, D_rev, 10 necessity forwards | 0.4 | 0.4 |
| P3 `dpo` session: 60-forward sweep, manipulation check, references, reload | 2.9 | 2.4 (grid shrunk) |
| vLLM pre-flight at chosen setting + scramble | 0.4 | 0.4 |
| dev guard | 1.2 (64 x 2) | 0.65 (64 x 1) |
| E0' | 1.9 (384) | 0.95 (192) |
| E1 | 2.6 (544) | 1.9 (464, held-out 80 x 1) |
| E2 (conditional) | 2.9 (544) | 2.2 (464) |
| loop and tail allowance | 1.8 | 1.2 |
| **total if E2 runs** | **19.4** | **15.4** |
| **total if E2 skipped** | **16.5** | **13.2** |

- Cut order under a $15 cap (applied mechanically by the pod script from spend so far plus projection, in this order, only as far as needed): (1) E2 conditional (already in); (2) dev guard 64 x 2 to 64 x 1; (3) held-out 80 x 2 to 80 x 1 in E1 and E2; (4) NLL grid shrink (F4 step 4 order); (5) E0' 384 to 192. Costs: (2) weaker guard, free-generation loops only caught if gross; (3) held-out resolution about +-10 points; (4) coarser dose picture, possibly missing the true optimum; (5) reach baseline loosened to about 3-4 %.
- Floor if the cap falls below about $13: setup, P1-P3 and manipulation check only (about $9), no sampling, reported as not done.
- Note: the full design at $19.4 (E2 runs) is above the $15 share and above the auditor's $17-18.5; the extra over Turn 3 is the dpo reload, the 10 necessity forwards, the damage forwards and the pre-flight at the chosen setting. [guess: HF throughput could move the NLL block between about $1.5 and $6.]

### OPEN (still disputed or unresolved)

No point is disputed between the auditor and me after this turn. Unresolved limits, all labelled in the design:
1. The factors in "rise <= 2 x D_rev" and "R > 2g" and the band edges are uncalibrated guesses.
2. NLL-optimal is not pass-optimal; the thinking/answer split exposes the main failure mode but does not remove it.
3. HF prefill throughput is unmeasured; the NLL block cost can move by a factor of four, and the shrink rule is the only protection.
4. One layer family (16-40), one static vector, no gate; the result is a bound on this construction only.
5. The E2 trigger leans toward running E2; the thresholds (+2 reach passes, 6 points held-out) are untuned.
6. The full cost ($19.4 with E2, $16.5 without) exceeds the $15 M3 share; the cap handling is mechanical but the orchestrator should decide whether to fund the full design or the capped one.

## Turn 6 — auditor (Claude Opus 5.5), 2026-11-27 02:04 PT

Closing turn (round 3). Checked against section F only. Conceded points are not reopened.

### Turn 4 required changes: implemented in F?

1. **Thinking/answer split: yes.** "Report the full grid, total / thinking / answer" (F4.4); "for total, thinking and answer tokens" (g); "D_rev ... (total, thinking, answer)" (F4.3). R's split is not stated in F, only in A.1. See defect d1.
2. **Full grid and g at α = 1 on held-out text: yes.** "Report the full grid"; "V_resp or V_prompt at alpha = 1 at the chosen layer and type" (F4.4).
3. **Damage rule: yes.** "Admissible set A = settings with damage_rise <= 2 x D_rev"; "If A is empty: chosen = V_resp at alpha = 1 on the layer with the smallest damage_rise" (F4.4).
4. **Bootstrap g, bands only inside the interval: yes.** "a band is assigned only if the whole interval lies inside it, else 'between bands'" (F4.4).
5. **Necessity ordering: yes, extended to V_prompt.** "NLL of `s2300` - 1*v_l on H16 `s2300` text for V_resp and V_prompt at all five grid layers (10 forwards)" (F4.3).
6. **Mechanical E2 trigger on the pod: yes.** "Grade on the pod's CPU with Dan's grader immediately after it ends" (F4.8); "E2 trigger (mechanical, script on the pod)" (F4.9).
7. **Split-half at the chosen setting, with converted numbers: yes.** "c < 0.5, stop sampling and message the orchestrator with c, sqrt(2c/(1+c)), and both numbers again with the top-5 coordinates removed" (F4.4). Idle handling is missing (d4).
8. **One engine and the timing rule: yes.** "NLL numbers come only from HF with a forward hook" (F1); "If above $3.00, shrink the grid in this fixed order" (F4.4).
9. **Pre-flight at the chosen setting plus one scramble: yes.** "Pre-flight references ... for the chosen setting and for scramble 2001" (F4.4); "STEER_ACTIVE plus vector hash in the sidecar" (F4.5). The fallback α is not covered (d3).

### Defects in section F (each with the fix)

- **d1. Undefined quantity: the answer segment.**
  - S16 `dpo` texts are truncated at 12,000 tokens and may contain no end-of-thinking marker, so answer-token D_rev and damage_rise are undefined for those prompts.
  - Fix: a text with no marker counts entirely as thinking and is excluded from answer-token averages. Report the number excluded.
  - Admissibility uses **total** damage_rise against **total** D_rev; say so.
  - Report R with the same split as g.
- **d2. The selection rule is not mechanical in one branch.** "Run the V_prompt row at that layer" is undefined when no V_resp setting is admissible but some V_prompt setting might be.
  - Fix: if no V_resp setting is in A, run the V_prompt row at the V_resp layer with the smallest damage_rise at α = 1. Then apply the rule over all 30, with the existing empty-A fallback.
- **d3. Order of operations: the dev-guard fallback.** F4.6 says "halve alpha once, rerun once (re-run the pre-flight of step 5 for the new alpha)", but the HF references are computed only in P3, which is unloaded by then.
  - Fix: in P3, also compute the four-sequence HF references at the chosen setting with α/2. That is two short forwards, near free.
- **d4. Order of operations: the split-half stop.** "Stop sampling and message the orchestrator" leaves the H200 billing while it waits.
  - Fix: write all NLL results to disk, wait at most 15 min (≈ $1.3) for a reply, then terminate the pod. Resuming costs a new start-up, about $1-2.
- **d5. Text slip.** F4.1 says the noise floor and zero-vector identity are "deferred to step 9"; they are in step 5.
- **d6. Rules broken by the cuts.**
  - Under cut (5), E0′ has 4 draws per prompt. The trigger "reach passes in E1 - reach passes in E0' >= 2" then compares 384 draws with 192, and F6's "pass@8 ... 8 draws per prompt each" cannot be computed for E0′.
  - Fix: the trigger compares E1's pass count with 2 × E0′'s count; reach comparisons use pass@1 and pass@4 when E0′ = 192.
  - Under cut (3), F does not say which scramble gets E2's 80 held-out draws. Fix: prompt i → scramble (i mod 3) + 1.
- **d7. Cost slips** (my recomputation with F7's own assumptions):
  - E1 at 464 draws is about $2.2, not $1.9.
  - E2 at 464 draws in three processes is about $2.6, not $2.2.
  - P2 is about 1.8 M prefill tokens (10 necessity forwards of about 109k each plus references and sums), about 10 min at 3k tokens/s, so about $0.9, not $0.4 (plus the `s2300` HF load if setup does not include it).
  - Corrected totals: full with E2 about $20-20.5, full without about $17-17.5; capped with E2 about $16.6-17.1, capped without about $14.0-14.5.
  - So under a $15 cap with E2 triggered, the cut list is exhausted and the plan still overruns by about $2. All variants stay under F5's $22 hard stop.
  - Fix: add cut (6). If E2 triggers and the projection exceeds the cap, run E2 on reach only (384 draws split 3/3/2, about $1.9 with three start-ups) and drop E2's held-out draws. Reach is the question; held-out is the damage check.

None of d1-d7 changes the design's logic; all are specification fixes the build agent can apply before writing `STEER_SPEC.md`. With them applied, I have no remaining objection to running the design.

### Final OPEN list

1. The factors in "rise ≤ 2 × D_rev" and "R > 2g", and the 5 % / 20 % band edges, are uncalibrated guesses.
2. NLL-optimal is not pass-optimal. The thinking/answer split exposes, but does not remove, the risk that the chosen setting moves style rather than compliance.
3. HF prefill throughput is unmeasured. The NLL block can move by about 4×; the shrink rule protects P3 only, not P2.
4. One static, ungated vector at layers 16-40. The result bounds this construction only.
5. The E2 trigger thresholds (+2 reach passes, 6 points held-out) are untuned and lean toward running E2.
6. Funding: full design about $20 with E2 against the $15 share. Under the cap, E2 only fits with cut (6). This is for the orchestrator or Leon to decide; it is not a design dispute.
