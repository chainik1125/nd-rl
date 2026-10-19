---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - guoDeepSeekR1IncentivizesReasoning2025
---

# LLM-written thinking traces for the proof-state model

Status (updated 2026-10-05, evening):
- **The LLM-written route (§3 (a)) is not pursued.** Claude subagents asked to write these traces
  were blocked by a `[reasoning_extraction]` safeguard. No Claude-written trace is used for training.
- The experiment runs on the programmatic traces (§3 (d)), built for all of K12 by
  `code/thinking/progtrace.py`.
- Current status: [the experiment summary](../../../../experiment-summaries/2026-10-05-thinking-traces/README.md).

The rest of this note is the original proposal. Branch `leon/thinking` (worktree
`/scratch/cowifornia/nd-rl/think/nd-rl`, base `1d13378`, which contains origin/dan `816e553`). Dan's
fork, read-only clone at `4dcd0e4f`.

## Summary

The goal is to pretrain `state_env_october` (ALiBiGPT 9.56M, `lean_staten`, K12) to write a
thinking trace, so that RL can later learn to think longer and better. This note proposes how to
make an LLM write those traces in a language made almost entirely of the model's existing tokens.

- **Where the thinking lives (§1).** Put a think block before every action in the proof-state
  environment. Blocks are Markov (no history) and may be empty:
  `state <think> … </think> action <eos>`. Do not think over the whole proof: that gives up the
  current best interface, and the earlier whole-proof scratchpad lost at matched tokens.
- **The language (§2).** There are 111 existing tokens plus **3 new ones**: `<think>`,
  `</think>` and `<no>`. A think line is one of four things:
  - a goal reduced to typed holes (`⊢ G := SKEL`, where a hole is written `⊢ F`);
  - a recalled hypothesis (`n3 : A`);
  - a candidate action;
  - a rejection (`<no>`).

  These cover subgoals, backward plans, case splits, candidate steps, checks, dead ends and
  backtracks. No `<try>`, `<backtrack>` or `<REVISE>` token is needed (§2.4).
- **Forcing an LLM (§3).**
  - **Primary:** Claude Opus 5.5 through the Batch API. It is given each state and the reference
    action, plus the base model's own typical mistakes as distractors. It writes ASCII DSL inside
    JSON, and a strict validator checks every claim, with up to two retries carrying the
    validator's message.
  - **Fallback and control arm:** programmatic traces from a truth-table-pruned backward search.
    These use the same DSL and the same validator.
  - **Rejected:** grammar-constrained open-weight decoding and NL-plus-translator. Either the
    semantics still need the validator, or the reasoning does not survive translation.
- **Cost (§5).**
  - The estimate is **$0.019 to $0.077 per proof** with Opus 5.5 batch. The midpoint, $0.038,
    assumes 2,000 thinking tokens and 1.3 attempts per proof.
  - That is **$76 for 2,000 proofs and $760 for 20,000**. All 155,000 would cost about $5,900,
    which is not justified.
  - Recommendation: run a calibration of at most about $8, then 2k proofs, then 20k only if a
    cold start shows the model uses its thinking.
- **Pilot (§7, the evidence).** I wrote traces for 8 real problems (6 K12, 2 textbook-train)
  under the production prompt.
  - **8 / 8 pass the strict validator and Lean 4.**
  - Each step is checked against Dan's environment, `step_check`, truth tables and Lean.
  - Validator mutation test: 305 of 337 corrupted traces are rejected. All 32 survivors are
    still true claims.
- **Decisions for Leon:** see §8.

---

## 0. What this builds on

**Observed, from prior work:**

- **r1_cot (2026-09-08/09)** ([summary](../../../../experiment-summaries/2026-09-08-r1-cot/README.md),
  [pair 1 review](../../../../experiment-summaries/2026-09-08-r1-cot/pair1_review.md)).
  - It added `<THINK> </THINK> <FINAL>` to an 8M whole-proof model. The scratchpad was a
    deterministic function of the reference proof (its lines reversed, or a goal-first dependency
    plan, `r1_cot/scratch_variants.py`).
  - Results on 58 textbook dev targets: CoT RL 10/58 against direct RL 11/58 at k = 16, with
    CoT's solved set a subset of direct's. Greedy: 4/58 against 10/58. CoT used about 2× the
    generated tokens, and direct dominated at every matched-token allowance.
  - The `<REVISE>` cue was built into a "correction" cold start: a single-token corruption of the
    proof, then `<ERROR><REVISE>`, then the plan. That cold start was implemented and unit-tested,
    but no trained result is recorded. Its s1-style budget-forcing evaluation
    ([plan](../../../../experiment-summaries/2026-09-08-r1-cot/inference_scaling_plan.md)) was never
    implemented. Pair 2 (85M, dependency plans) was queued; no result is recorded.
  - The support note warned that repair traces without feedback "can teach a model to ignore
    noisy scratch and copy a final answer".
  - **Lesson used here:** the trace must carry computation the answer does not. That means
    rejected candidates, checks, dead ends and alternatives, each true by verification, not a
    re-rendering of the proof.
- **Robbie, test-time scaling** (whole-proof c005, branch `robbie-experiments`,
  `experiment-summaries/2026-09-29-test-time-scaling/`).
  - Resampling invalid lines took pass@1 from 31.2% to 57.4%.
  - Beam search and best-first search were no better than plain sampling at matched tokens.
  - About 12% of theorems stayed unsolved by every method. About 90% of those stayed unsolved at
    k = 1,024.
  - His interpretation, not tested: most failures are local, and the rest are theorems where the
    policy "has no working plan".
- **Dan, guided-tts**
  ([summary](../../../../experiment-summaries/2026-10-03-guided-tts-logical-redraws-beat-plain-on-long-theorems-structural-redraws-do-not/README.md)).
  - Redrawing steps that `step_check` rejects, without replacement, beats plain sampling at
    matched sampled tokens on 90 long theorems. IQM +9.1 pp at cap 12 and +8.3 pp at cap 6, in
    6 of 6 models.
  - Redrawing only structural rejections makes no difference (|Δ| ≤ 1.2 pp).
- **Sibling notes in this folder (other agents, 2026-10-05, not re-verified by me except where
  marked).**
  - [CONTEXT_SURVEY.md](CONTEXT_SURVEY.md) gives the integration constraints:
    - token ids are stored as uint8; I verified `torch.uint8` in `state_train_best.to_device`;
    - every id ≥ ref0 is shifted as a name;
    - EI replay stores only the ND proof;
    - the loss mask covers action plus `<eos>`.
  - [OOD_CHECK.md](OOD_CHECK.md) shows where the current SOTA still has headroom:
    - Charles's long problems, pass@1 7.7–13.0%;
    - textbook_dev, pass@1 40.8%;
    - group C;
    - generated reorder families.

  The second note proposes a testbed (Tier A/B/C) that I reuse in §6.

**Related practice.** DeepSeek-R1 adds a small reasoning cold start before RL
[@guoDeepSeekR1IncentivizesReasoning2025]. DeepSeek-Prover-V2 builds its cold start by
decomposing theorems into subgoals [@ren2025deepseekproverv2advancingformalmathematical]. This
proposal follows the same shape, a verified decomposition trace first and RL second, at 9.56M
parameters and in a formal trace language.

**Interpretation.** Dan's +9 pp shows that a cheap logical check on each step buys accuracy at
matched compute. A thinking model that checks its own candidates internalises that check. Unlike
the external checker, it can also look ahead: does the box I am about to open close? Lookahead is
what the "no working plan" theorems need. This is the hypothesis the TTS phase would test (§6).

## 1. Where the thinking lives

| option | context per sequence | fits the SOTA interface | how RL lengthens thinking |
|---|---|---|---|
| **whole-proof**: think, then write the proof (r1_cot's format) | prompt + think + proof. K12 proofs are about 127 tokens and thinking over a whole proof runs to hundreds or thousands. The state model's training pairs never exceed about 391 tokens (sibling survey), and 4 of its 8 heads are NoPE | **no**: it means returning to `lean_seq` whole-proof decoding, which is weaker (Robbie's whole-proof model scores 30–32 / 72 on textbook72; state_env_october scores 52 / 72 at pass@256; the protocols differ) | one long block per attempt; credit is spread over the whole trace |
| **per-step, Markov** (recommended) | state + think + action. K12 states are at most 386 tokens (mean 76.5, my sample of 6,000). Pilot think blocks are at most 160 tokens; the pilot's largest pair is 264 tokens | **yes**: the action is unchanged and the environment still names steps and checks them; it is stripped of think tokens before `Env.apply` | per-step budget forcing; EI and GRPO in the environment already work per step |
| per-step with history (`SH`-style: earlier think blocks in context) | grows with the proof | partly; arm SH exists but is not the SOTA | the plan persists, but context and memorisation grow |
| root-only (plan once at the first state) | one block | yes | one knob per attempt; nothing at the later decisions where errors happen |

**Recommendation: per-step, Markov, possibly empty.** Every step emits `<think> … </think>`, and
forced steps (premise restatements, `exact` of the line just derived) close it at once, which
costs 2 tokens.

The cost of the Markov choice is that the plan is recomputed at every step. In the pilot this was
cheap: mean 46 think tokens per decision step, against 76 tokens of state.

The benefits:
- each block can be checked against its own state alone;
- the budget knob is local;
- per-step RL credit needs no new machinery;
- context stays bounded. With `max_action` raised to 512, the worst-case pair is about
  386 + 512 tokens, under the 1,024 window.

A "plan memo" that carries the last surviving reduction into the next state is a possible later
extension; it is not part of v0.

## 2. The trace language

### 2.1 Vocabulary

- `lean_staten` has 111 tokens (read from `LeanTokenizer('lean_staten').itos`):
  - `<pad> <eos>`;
  - formulas: `( ) ¬ ∧ ∨ → P Q R S False`;
  - `theorem t : Prop := by`;
  - `have exact ; fun => ⟨ ⟩ , .1 .2 .elim Or.inl Or.inr Or.elim Classical.byContradiction hh`;
  - `<st> <nl> ⊢ <act>`;
  - `h1`–`h8`, `n1`–`n64`.
- The think blocks use **only** these tokens plus `<think> </think> <no>`.
- In a new mode (say `lean_thinkn`) the three tokens must sit before `h1`. That moves ref0 from 47
  to 50. This is harmless from scratch; a cold start from an existing checkpoint needs a row
  permutation of the embedding and the output head.
- LLMs write an ASCII surface, which `trace_dsl.tokens` maps one-to-one onto these tokens:
  - `|-` → `⊢`;
  - `~ & v >` → `¬ ∧ ∨ →`;
  - `F` → `False`;
  - `<| |>` → `⟨ ⟩`.

### 2.2 Grammar (EBNF)

```ebnf
step     = state , think , action , "<eos>" ;            (* state, action: unchanged, STATE_ENV.md *)
think    = "<think>" , { line , "<nl>" } , "</think>" ;
line     = goal | look | action | "<no>" ;               (* a line with ⊢ is a goal; have/exact: action *)
goal     = { name , ":" , formula } , "⊢" , formula , [ ":=" , skel ] , [ "<no>" ] ;
look     = name , ":" , formula ;
skel     = "have" , name , ":" , formula , ":=" , skel , ";" , skel      (* cut / intermediate goal *)
         | lam
         | app , [ ".1" | ".2" | ".elim" ] ;
app      = prim , [ prim ] ;                                              (* application *)
prim     = name | hole | "(" , skel , ")"
         | "⟨" , skel , "," , skel , "⟩"
         | ( "Or.inl" | "Or.inr" | "Classical.byContradiction" ) , prim
         | "Or.elim" , prim , "(" , lam , ")" , "(" , lam , ")" ;
lam      = "fun" , "(" , name , ":" , formula , ")" , "=>" , skel ;
hole     = "⊢" , formula ;
formula  = "P" | "Q" | "R" | "S" | "False" | "(" , "¬" , formula , ")"
         | "(" , formula , ( "∧" | "∨" | "→" ) , formula , ")" ;
name     = "h1" | … | "h8" | "n1" | … | "n64" ;
```

**Semantics. Every line is a claim that the validator checks (§4).**

- **Goal lines.**
  - `ctx ⊢ G := SKEL` claims that SKEL has type G when its holes `⊢ F` are typed metavariables,
    in the state's scope plus the think-local binders listed in ctx.
  - `ctx ⊢ G <no>` claims that G is not a classical consequence of that scope. A countermodel over
    P Q R S exists. The fragment has DN, so "not a consequence" means "not provable".
  - `⊢ G := SKEL <no>` claims that SKEL is ill-typed, or that one of its holes is unprovable.
- **Look.** `n : A` claims that n is in scope with exactly A.
- **Candidate.** An action line is a candidate step in the current state.
  - A `<no>` line rejects the latest candidate. The rejection is true if the environment rejects
    the candidate, or `step_check` rejects it (sound with respect to Lean), or it leaves an
    unprovable goal open.
  - An unrejected candidate must pass all three checks.
- **Commit rule.** The action after `</think>` must equal the last unrejected candidate, if there
  is one.

### 2.3 Which reasoning moves are expressible

The examples below are from the pilot (`pilot_traces.jsonl`), in model tokens.

| move | how | example |
|---|---|---|
| subgoal / intermediate goal | a hole `⊢ F`; a cut `have x : C := ⊢ C ; …`; a goal line refining a hole under its binders | `⊢ ( ¬ ( ¬ Q ) ) := have n4 : Q := ⊢ Q ; fun ( n5 : ( ¬ Q ) ) => n5 n4` (K12 152829, step 2) |
| backward-chaining plan | successive reductions from the goal down to the action | `⊢ ( ( ¬ R ) ∨ R ) := Or.inr ⊢ R` / `⊢ R := n2 ⊢ ( ( ¬ Q ) ∨ ( P ∧ R ) )` / … (89161, step 2) |
| propose a candidate | the action itself | `have n5 : ( ( P → … ) ∧ ( ¬ S ) ) := ⟨ n1 , n2 ⟩ ;` |
| check it | look up the cited names, then the verdict | `n1 : ( ¬ S )` / `<no>` (37120, step 3) |
| dead end | `⊢ G <no>`; `⊢ G := SKEL <no>`; `<no>` after a candidate | `⊢ ( ( ¬ Q ) ∨ ( P ∧ R ) ) <no>` (89161) |
| backtrack | restate an earlier goal with another reduction | `⊢ ( P ∨ Q ) := Or.inl ⊢ P` / `⊢ P <no>` / `⊢ ( P ∨ Q ) := Or.inr ⊢ Q` (tb_bcb4f30a) |
| case-split plan | an `Or.elim` skeleton, then one goal line per branch with its binder in ctx | `⊢ … := have n3 : B := Or.elim n1 ( fun ( n4 : R ) => ⊢ B ) ( fun ( n5 : ( ¬ S ) ) => n2 n5 ) ; n3 .1` / `n4 : R ⊢ B := ⟨ ⟨ n4 , n1 ⟩ , n4 ⟩` (22530) |
| classical detour | `Classical.byContradiction ⊢ ( ¬ ( ¬ G ) )` | Peirce's law, `( ¬ P ) → Q ⊢ P ∨ Q` |
| lookahead into a candidate box | a box candidate, then `x : A ⊢ B <no>`, then `<no>` | `have n3 : ( P → Q ) := ( fun ( n4 : P ) => by` / `n4 : P ⊢ Q <no>` / `<no>` (Peirce, step 1) |
| inconsistent premises | `⊢ False := …` (prove falsity, then `.elim`) | 53241, step 2 |

**Not expressible, by design:**
- "this reduction is circular or makes no progress" (it is true, just useless);
- "the premises are inconsistent", except by deriving False;
- a confidence;
- prose.

Abandonment without a semantic reason is written by restating a parent goal, with no claim.

**A limit found while writing the pilot (observed).** Semantic dead ends only exist in consistent
contexts. Inside a reductio box, and under inconsistent premises (K12 53241), every goal is
provable, so `<no>` can only reject candidates there. In 53241, `⊢ S <no>` would be false. Since
thinking in classical proofs happens inside such boxes, the language's goal-level dead ends help
mostly outside them.

### 2.4 Extra tokens: what was considered

| token | decision | reason |
|---|---|---|
| `<think>`, `</think>` | **add** | They delimit the block and keep it out of `Env.apply`. Suppressing `</think>` is the budget-forcing handle. |
| `<no>` | **add** | Negative judgments are the one move the logical vocabulary cannot say. Reusing `False`, `¬` or `hh` would overload tokens whose meaning is fixed in formulas. Every `<no>` is mechanically checkable. |
| `<try>` / candidate marker | no | A candidate is syntactically identifiable: the line starts with `have` or `exact`. |
| `<backtrack>` | no | Lines are self-describing: each restates its goal and its binders. Restating an earlier goal *is* the backtrack, and it is checkable. |
| `<REVISE>` / "Wait" cue | no, for v0 | r1_cot built it but never evaluated it, and its own support note flagged the copy-the-answer risk. Here, forced continuation can start an in-language line: suppress `</think>` and let the model emit, or force, the next line start, such as `⊢`. The data must contain post-candidate verification, the lookahead pattern in §2.3, for the forced continuation to be in distribution. A dedicated cue is an ablation only if forced continuations turn out to be off-distribution. |
| `<ok>` | no | Passing is implicit, and the commit rule makes it observable. |
| hole ids (`?1`) | no | A hole is addressed by its binders plus its formula. That costs tokens, not vocabulary. |
| countermodel display | optional later | A `<no>`'s countermodel could be written as literals (`P ( ¬ Q )`) with existing tokens. That would make `<no>` checkable by the model itself. |

## 3. How to make an LLM write it

### 3.1 Options compared

| | (a) Claude API + validate/retry | (b) open-weight model + grammar-constrained decoding | (c) NL reasoning + deterministic translator | (d) programmatic search traces |
|---|---|---|---|---|
| cost | $0.019–0.077 per proof (Opus 5.5 batch, §5) | No API cost. GPU time on shared 3090s: for 20k proofs at about 2,600 output tokens each (52M tokens), at roughly 1k tokens/s per GPU, about 14 GPU-h (estimate, unmeasured) | at least (a): NL is longer than the DSL, plus a translator | about $0, CPU |
| fidelity (valid output) | Validator-gated. Acceptance rate unmeasured; the calibration measures it | The grammar guarantees syntax only. Scope, types and truth still need the validator, and a per-state grammar would have to encode the names in scope. Format unfamiliarity costs open models: Qwen3-Coder-30B proves 56% of 236 theorems in Lean against 26% in the take-home token format ([round 2, run 1](../../../../experiment-summaries/2026-09-18-round2-run1-nd-to-lean/README.md)). JSON-constrained output degrades reasoning (Tam et al., arXiv 2408.02442) | Brittle. Most of the NL has no DSL counterpart, and what remains loses the reasoning that produced it | 100% by construction |
| diversity | High across theorems. Opus 5.5 has no temperature (sampling parameters return 400), so diversity comes from prompt variation and distractor seeds | temperature is available; quality is the problem | high before translation, collapsed after | low unless randomised; heuristic-shaped |
| genuine vs post-hoc | Reference-guided mode is retrospective rationalisation, as in Lean-STaR's GPT-4 thoughts (arXiv 2407.10040). Solve mode records a search Claude ran, but Claude's real reasoning is in its hidden thinking; the DSL is a rendering either way. Dead ends are real (verified) in both | genuine only to the extent that a weaker model searches | closest to the LLM's reasoning, lost in translation | a genuine search, by a prover, not by an LLM |
| engineering | 1–2 days: renderer, batch submission, retry loop. The validator is done | 3–5 days: vLLM + xgrammar or outlines, dynamic per-state grammars, model choice, GPU scheduling around shared cards | about 1 week, plus ongoing brittleness | 1–2 days: reuses the validator's type checker, truth tables and environment |

### 3.2 Recommendation

**Primary: (a), Claude Opus 5.5, through the Batch API.** Settings: structured output
(`output_config.format`, a JSON schema `{steps: [{think, action}]}`; forced `tool_choice` returns
400 on Opus 5.5), `effort: low`, 5-minute prompt caching on the system prompt. The prompt is
[production_prompt.md](production_prompt.md). Three modes:

1. **Reference-guided, error-seeded (bulk).**
   - The user turn gives every state of the reference proof (from
     `state_env.decompose(…, canon=True)`) and the reference action.
   - For each decision state it also gives up to 3 of the *base model's own* rejected actions:
     samples from `stage1_best12` at that state, filtered by `step_check` or the environment. That
     takes about a GPU-minute for 2k proofs (estimate from guided-tts throughput); it needs
     approval.
   - Claude must refute each plausible distractor in-trace. This aims the traces' dead ends at
     the model's real failure distribution, not at invented corruptions, which was the r1_cot
     correction arm's weakness.
   - Claude may replace a roundabout reference step (decision 2, §8).
2. **Solve (subset, about 500 proofs).** Claude sees only the theorem and writes states, thinks
   and actions in one call. The validator replays them and retries on error. Use it on the
   hardest strata (classical, ≥10 lines, has a semantic dead end). It measures the post-hoc gap:
   compare cold starts trained on mode-1 vs mode-2 traces for the same theorems.
3. **Environment in the loop (optional, about 100 proofs).** A multi-turn tool loop with the state
   environment as the tool, giving genuine per-step feedback. It costs 3–6× mode 1 and cannot be
   batched, so it is a probe, not a data source.

**How the language is forced** (the core of the question):
- a specification in the system prompt, plus two complete worked traces;
- a Lean-like surface the LLM already knows (`have`, `fun`, `⟨ ⟩`, `Or.elim`, `⊢` holes like
  Lean's `?_`);
- the JSON schema;
- a deterministic renderer;
- the validator, whose error messages are written to be actionable (e.g.
  `false look: n1 : S (in scope: ( ¬ S ))`);
- at most two retries, after which the theorem is dropped.

If format errors dominate the calibration, switch to one JSON object per line
(`{kind: goal|look|candidate|reject, …}`) so that the move type is schema-enforced.

**Fallback and control: (d).** A backward-search trace generator in the same DSL:
1. emit the reduction the reference action implements;
2. enumerate alternative reductions (by the goal's connective, and by hypotheses whose consequent
   is the goal), and mark the dead ones `<no>` by truth table, in random order;
3. add the base model's rejected actions as refuted candidates;
4. commit.

The LLM's only role in (d) would be ordering. (d) fills the 135k–153k proofs with no LLM trace,
and on its own it is the "is LLM authorship the active ingredient?" control.

## 4. Verification and trace quality

**Hard checks (implemented in `trace_dsl.py`, all observed working on the pilot):**
- every token is in `lean_staten` ∪ {`<think>`, `</think>`, `<no>`};
- the grammar holds, and every formula is canonical (fully parenthesised, as `lean_tok.ftoks`);
- every look matches the scope;
- every unmarked reduction type-checks: bidirectional checking with holes as metavariables,
  binders fresh;
- every `<no>` is justified by a truth-table countermodel, an ill-typed reduction, an environment
  rejection, a `step_check` rejection, or an unprovable goal left open;
- every unrejected candidate passes all three checks, with canonical names;
- the commit rule holds;
- the replayed actions finish the proof, and **Lean 4 accepts the assembled text** (`lean_gate`).

**Soft checks** (counted; `--strict` makes them errors, recommended for production):
- goal lines that refine nothing seen;
- vacuous reductions (`⊢ G := ⊢ G`);
- unprovable goals or holes that are explored but never flagged.

**Validator mutation test (observed).** I mutated every think line of the 8 pilot traces: toggled
`<no>`, swapped an atom, a name or a connective, dropped a reject line.

- Under `--strict`, 305 of 337 mutants are rejected.
- I inspected all 32 survivors. Every one is still a true claim:
  - 4 fresh-binder renames;
  - 8 cases of `<no>` added to a reduction whose hole is in fact dead;
  - 20 edits turning one already-rejected candidate into another failing one.
- Without `--strict`, 276 of 337 were rejected. The extra survivors were claim-free edits, such
  as a dropped `<no>`, flagged only as soft findings. That is why production should run strict.

**Informativeness measures** (proposed; deterministic unless marked):
1. **Relevance.** The share of look lines whose name appears elsewhere in the block or in the
   action. Pilot: 11 of 12. The exception, `n3 : ( P → P )` in k12_16472 step 2, is filler (it
   records that n3 is useless without saying so). Flag traces below 0.5.
2. **Decision coverage.** Think length should be near zero on forced steps and positive at
   decisions. In the pilot (in think tokens, tags excluded), `exact` steps were 0 in 21 / 21,
   premise steps 0 in 10 / 12, atomic `have` steps had mean 36 (max 114), and box or case
   openers had mean 72 (max 158).
3. **Dead-end share.** The fraction of decision steps with at least one verified `<no>`, and
   whether `<no>`s target the base model's real errors (mode 1).
4. **Shortcut audits.** The position of the rejected candidate (it must not always come first;
   the pilot has one step with accept-then-reject), the `<no>` rate per position, and the
   identity of the first candidate versus the action. These matter because a statistical
   pattern in the data becomes a learned shortcut.
5. **Causal use (after training, needs GPU).** Corrupt the think block: swap a look's formula,
   drop a `<no>`, paste another state's plan. Measure how often the action changes, and the
   accuracy cost with thinking forced empty versus free.

**Diversity measures:**
- the distribution of move types and skeleton shapes;
- distinct-line n-grams across traces that share a renaming class;
- two traces for a 5% subset of states, with different distractor seeds;
- the share of steps where the trace's action differs from the reference.

**Length against difficulty (needed for TTS):**
- Correlate think tokens per step with the base model's rejection rate at that state (from the
  error-seeding samples) and with "has a semantic dead end".
- In my sample of 20,000 K12 rows, 2,503 of the 11,045 with 7–12 lines (23%) contain one
  (premise-inconsistent rows excluded). The finder script is in the scratchpad; it can be added
  to the folder if wanted.
- Target: Spearman ρ ≥ 0.3 on held-out states. If the LLM's lengths do not track difficulty,
  stratify the prompt: request "minimal" and "extended" thinks for a subset of steps, and drop
  alternative branches at random Dualformer-style (arXiv 2410.09918) to get a controllable length
  distribution.

## 5. Scale and cost

### 5.1 How many traces

- **Training token budget (observed and derived).**
  - Stage 1 of `state_env_october` is 1,200 s on an A40: about 24,000 steps, 3.7 epochs of
    1.24M (state, action) pairs.
  - Per proof, the pilot has 514 state tokens (no loss), 116 action tokens and 232 think tokens.
    Thinking therefore doubles the loss tokens and multiplies the tokens processed per proof by
    862 / 630 ≈ 1.37.
  - At a fixed 1,200 s that is about 2.7 epochs instead of 3.7. Alternatively, give it 1,650 s.
  - The pilot over-represents decision-rich proofs. Assume 150–230 think tokens per K12 proof.
- **What the traces must teach.**
  - The grammar is small and regular: 4 line types, about ten skeleton forms. Its moves are local.
    Interpretation: format should be learnable from tens of thousands of decision steps.
  - Learning *accurate* `<no>`s and useful lookahead needs more. That is where unlimited
    programmatic traces help.
  - A 9.56M model from scratch is not s1's 32B model. Interpretation: s1's 1,000 examples
    elicited a behaviour already present in a pretrained Qwen2.5-32B-Instruct. Here the behaviour has to be created, which is why the
    pause-token result is relevant: delays help only when they are present in pretraining too
    (Goyal et al., arXiv 2310.02226).
- **Recommendation.** Thinking on *every* proof, so that format and length statistics are uniform:
  - **2,000 LLM-traced proofs** first, stratified by length 2–12 and by dead-end presence;
  - up-weighted ×4 in sampling, as EI weights its found proofs;
  - programmatic traces (d) for the other 153,000;
  - 20,000 LLM proofs only if the 2k cold start passes the causal-use test.

  Do not mix LLM traces with *empty* think blocks on the rest: the model would learn that most
  states need no thinking, which confounds length with data source.

### 5.2 Length against context

Observed (pilot):
- think tokens per decision step: mean 46, median 34, max 160;
- largest pair (state + think + action + `<eos>`): 264 tokens, against K12's 391-token largest
  plain pair;
- `max_action` must cover think + action. The default is 256; use 512, which guided-tts already
  uses.

For TTS headroom, budget forcing up to 256 think tokens per step fits the 1,024 window with Tier A
states of at most 317 tokens (sibling OOD note). No sequence in the pilot needs truncation.

### 5.3 API cost (prices fetched from platform.claude.com/docs pricing, 2026-10-05)

| model | input | output | cache read | Batch API |
|---|---|---|---|---|
| Opus 5.5 | $4 / M | $20 / M | $0.20 / M (0.05×) | 50% off: $2 / $10, cache read $0.10 |
| Sonnet 5.5 | $2 / M | $10 / M | $0.20 / M | $1 / $5, cache read $0.10 |
| Haiku 4.5 | $1 / M | $5 / M | $0.10 / M | $0.50 / $2.50. Minimum cacheable prefix is 4,096 tokens, so a 3k-token system prompt does **not** cache |

Batch and caching discounts stack. Opus 5.5's thinking cannot be disabled (default effort
`medium`; set `low`).

**Per-proof model (Claude tokens; to be replaced by calibration measurements):**
- S = 3,000 tokens of cached system prompt and examples;
- U = 1,500 tokens of per-theorem input (states rendered in ASCII; pilot: 514 model-token states
  plus 116 actions per proof, plus distractors);
- O = 600 tokens of visible output (pilot: 311 ASCII words of DSL per proof, plus JSON);
- T = thinking tokens, in three scenarios: 500, 2,000 and 5,000;
- r = 1.3 attempts per proof, i.e. about 70–77% first-pass acceptance.

cost per proof = r × (S·c_cache + U·c_in + (O + T)·c_out)

| model, T | arithmetic (µ$ per attempt) | per proof | 2,000 proofs | 20,000 | 155,000 |
|---|---|---|---|---|---|
| Opus 5.5 batch, T = 500 | 3000×0.10 + 1500×2 + 1100×10 = 14,300 | $0.0186 | $37 | $372 | $2,881 |
| **Opus 5.5 batch, T = 2,000** | 300 + 3,000 + 2600×10 = 29,300 | **$0.0381** | **$76** | **$762** | $5,904 |
| Opus 5.5 batch, T = 5,000 | 300 + 3,000 + 5600×10 = 59,300 | $0.0771 | $154 | $1,542 | $11,949 |
| Sonnet 5.5 batch, same tokens, T = 2,000 | 300 + 1,500 + 2600×5 = 14,800 | $0.0192 | $38 | $385 | $2,980 |
| Haiku 4.5 batch, T = 2,000, no cache | 3000×0.5 + 1500×0.5 + 2600×2.5 = 8,750 | $0.0114 | $23 | $228 | $1,763 |
| (d) programmatic | – | $0 | $0 | $0 | $0 |

Other costs:
- **Solve mode** (about 500 proofs at T = 5,000 and r = 2): 2 × $0.0593 ≈ $0.12 per proof, about
  $60.
- **Calibration** (50 K12 proofs × 3 models, standard API without the batch discount to get
  answers in minutes): about 2 × (50 × ($0.038 + $0.019 + $0.011)) ≈ **$7**.

**Leon's Sonnet rule.** Sonnet 5.5 costs exactly half per token, so it is cheaper only if it uses
under 2× Opus's billed tokens *per accepted trace*, counting retries and thinking. That ratio is
what the calibration measures, on the same 50 theorems. My prior (unmeasured): Opus at low effort
is the safer default for a formal task with a strict checker. Haiku is likely to fail validation
often enough to erase its 4× price advantage, but the calibration will say.

### 5.4 Wall-clock

- **API.** 20k requests fit in one batch (the limit is 100,000 requests or 256 MB). Most batches
  finish within an hour, at most 24 h. With two retry rounds that is typically 3–6 h, at worst 3 days.
- **Validation (observed).** 4.1 ms per trace without Lean. Lean: 400 assembled proofs in 3.7 s in
  one process, 9.3 ms each. For 20k traces, about 1.4 min plus about 3 min, on one core of this
  48-core host.
- **Error seeding.** Minutes of GPU (needs approval).
- **Training arms (§6).** Each Stage 1 is about 20 min on an A40, or an estimated 30–40 min on a
  3090 (the sibling survey says to pass `--best_steps 24000` there).

## 6. How this sets up test-time scaling

**Integration changes in the fork's code (needed before any training; from the sibling survey
plus my reading of `state_sample.py`):**
- a mode `lean_thinkn`, with the 3 tokens before `h1`;
- the sampler splits the sampled sequence at `</think>` and sends only the action to `Env.apply`;
- `max_action` 512;
- `<eos>` forbidden inside think blocks;
- EI replay must **store the sampled think tokens**. Today it stores only the ND proof, so traces
  would silently vanish from RL training data.

**The RL phase.**
- Expert iteration (and later `grpo_state.py`) in the environment, where each step's sample is
  think + action.
- Reward: the final proof is accepted by Lean. No reward for length: r1_cot's strategy note,
  citing Dr. GRPO (arXiv 2503.20783), warns that length can grow without accuracy.
- The thinking knob is per-step budget forcing:
  - a **maximum**: force `</think>` at B tokens;
  - a **minimum**: mask `</think>` until m tokens, so the model must emit another line.

  This is s1's mechanism (arXiv 2501.19393: end thinking early by forcing the delimiter; extend it
  by appending "Wait"). Here the extension is in-language, as described in §2.4.
- During RL, sample the budget per attempt (B ∈ {16, 64, 256}) so that kept trajectories span
  lengths and the model does not collapse to whatever length EI happens to keep.

**What pretraining must teach, for RL to be able to extend thinking:**
1. variable length that tracks difficulty (§4);
2. **causal use**: the action must depend on the think. This is the commit rule plus training
   data where the plan determines a non-obvious action;
3. **calibrated `<no>`**: if `<no>` is accurate, every extra checked candidate raises success; if
   it is noise, extra thinking cannot help;
4. **post-candidate verification**: data where thinking continues *after* a passing candidate,
   with lookahead that sometimes kills it (the Peirce step-1 pattern), so that a forced
   continuation means "double-check", as "Wait" does for s1.

**Evaluation.**

- **Sets:**
  - K12 held-out (in distribution);
  - textbook_dev58 (train14 reported separately if any train14 trace enters training);
  - OOD_CHECK Tier A (80 of Charles's long problems within reach) and Tier B (reorder and commute
    families, n ≤ 6);
  - group C.
- **Curves:**
  1. pass@1 against the per-step think budget B ∈ {0, 16, 32, 64, 128, 256};
  2. pass@k against *total sampled tokens per theorem* (think + action + rejected redraws), and
     against wall-clock.
- **Arms, all from matched pretraining compute:**
  - **T0**: no-think control, trained on *the identical final actions* with think blocks removed;
    plain sampling and guided-logical redraws;
  - **T1**: programmatic traces;
  - **T2**: LLM traces on the subset, programmatic on the rest;
  - **T2 with thinking forced empty**: separates "better data" from "uses thinking";
  - **T2 + guided redraws**: do internal and external checks stack?
- **Pre-registered falsifiers.**
  - Thinking is unused if T2 at B = 0 equals T2 at B = 256 in pass@1 on Tier A.
  - It is not worth its tokens if T2 is at or below T0-guided at matched tokens on Tier A long.
- **Read-outs:** `<no>` precision and recall against the validator on sampled thinks, on held-out
  states; commit consistency; the causal-corruption tests from §4.

**Memorisation risk.**
- The model sees each trace about 2.7 times, and the traces are written for its own training
  theorems. The signatures of memorisation:
  - the validity of sampled thinks (run through `trace_dsl`) drops sharply from training states
    to held-out states;
  - held-out thinks copy training lines verbatim, with wrong names;
  - `<no>` precision is lower on held-out states;
  - length stops tracking held-out difficulty.
- Mitigations:
  - split by renaming class;
  - Dualformer-style random dropping of trace parts;
  - programmatic traces, which can be regenerated with fresh distractors every epoch;
  - in RL, the model's own validated thinks replace the LLM's (STaR-style; validate sampled
    thinks before keeping them).

## 7. Pilot: traces written by Claude under the production prompt (evidence)

**Setup.**
- 8 problems: K12 rows 37120, 16472, 22530, 89161, 53241 and 152829 (5–12 lines; IMPI, ANDI,
  NEGI, DN, ORE, BOTE, an inconsistent-premise case), plus two textbook72 **train-split**
  theorems, `( ¬ P ) → Q ⊢ P ∨ Q` and Peirce's law. Both are classical, with no reference proof.
- Three K12 rows (22530, 89161, 152829) were chosen by a scan for states with semantic dead ends.
- Modes:
  - 3 K12 traces follow the reference exactly;
  - 3 replace a roundabout reference case split with a direct step (89161: 12 → 7 ND lines;
    53241: 9 → 5; 152829: 12 → 7);
  - 2 are solve mode.
- I followed [production_prompt.md](production_prompt.md). Distractor candidates were my guesses
  at plausible model errors, not real model samples.

**Result: 8 / 8 traces valid under `--strict`, and Lean 4 accepts all 8 assembled proofs.**

| trace | steps | think tokens | largest pair | verified `<no>` | notes |
|---|---|---|---|---|---|
| k12_37120 | 6 | 196 | 197 | 1 candidate | whole proof term planned at step 0 |
| k12_16472 | 8 | 180 | 113 | 1 candidate | classical via DN |
| k12_22530 | 10 | 395 | 264 | 1 goal, 2 candidates | case-split plan; accept-then-reject order once |
| k12_89161 | 8 | 179 | 214 | 2 goals, 1 candidate | deviates (12 → 7 lines) |
| k12_53241 | 6 | 158 | 178 | 2 candidates | inconsistent premises: no goal-level `<no>` possible |
| k12_152829 | 8 | 161 | 126 | 1 goal, 2 candidates (one env-rejected `exact`) | deviates; unused premise h3 refuted as a route |
| tb_bcb4f30a | 12 | 306 | 222 | 3 goals, 1 candidate | both disjuncts dead, then reductio |
| tb_58b532c9 (Peirce) | 12 | 281 | 167 | 2 goals, 2 candidates (one dead box via lookahead) | |

Totals: 70 steps, 39 with non-empty thinking (56%). Think tokens per decision step: mean 46,
median 34, max 160. 232 think tokens per proof against 116 action tokens.

**What failed or nearly failed while writing (observed; this is the production error budget).**

The validator rejected nothing on the final run. While writing, I caught three errors before
validating, and the prompt now warns against each:

1. **`<no>` on a goal inside an inconsistent context.** I first planned `⊢ S <no>` in 53241 and
   `n5 : ( ¬ Q ) ⊢ P <no>` in 152829, then saw that every goal is provable there. The validator's
   test now covers this case.
2. **Inference failure.** `( Or.elim … ) .1` cannot be type-inferred. I rewrote it as
   `have n3 : B := Or.elim … ; n3 .1`.
3. **A rule in my first prompt draft was wrong.** "When a hole dies, mark its parent goal `<no>`"
   is false whenever the parent is provable another way. I corrected the prompt before writing
   the traces.

**Validator changes made during the pilot.**
- Box binders of a candidate now count as think-local binders, which enables the lookahead
  pattern.
- Strict mode was added.

**Caveats.**
- The author of the traces designed the validator. 8 / 8 is evidence that the language is
  writable and checkable, **not** an estimate of Claude's production acceptance rate.
- The rejected candidate comes first in all but one step: a shortcut risk (§4).
- 8 traces are too few for length statistics.
- The "solve mode" traces were written after I had worked the proofs out; the replay tool checked
  names. They are not first-attempt records.

## 8. Risks, open questions, decisions

**Risks.**
- **The thinking goes unused.** Post-hoc traces let the model learn the action without reading
  its think block. This happened to r1_cot.
  - Detect: the causal tests and the B = 0 against B = 256 falsifier.
  - Mitigate: error-seeded and solve-mode traces, lookahead content the action depends on.
- **Shortcut statistics:** `<no>` by position, or "the first candidate is always wrong". Audit
  the data and randomise the order.
- **Distribution shift from deviations.** Shorter proofs change K12's flat 2–12 length profile,
  which matters for this project's length work. The T0 control on identical actions isolates the
  effect of thinking, but not the shift itself.
- **Capacity.** A 9.56M model may not learn skeleton type checking. Thinking tokens could also
  hurt action accuracy (T2 at B = 0 against T0 measures that).
- **Names.** Think-local binders use up name tokens. MAXN is 64 and the sampling base is
  U[0, 32]. The sibling OOD note finds failures beyond about 30 names in scope, so cap the
  binders per block.
- **Saturation.** Dan considers the models too strong for TTS. The sibling OOD note finds headroom
  on Tier A/B and group C, and those are the sets to evaluate on.
- **Cost uncertainty.** Opus 5.5's thinking-token use on this task is unknown (a factor of 4
  between scenarios). The calibration resolves it for about $7.

**Open questions.**
- Does a plan memo carried across steps (non-Markov) beat recomputation?
- Should `<no>` carry a countermodel?
- Can the trace-trained model itself produce validated traces for the remaining proofs (STaR
  bootstrapping), replacing (d)?

**Decisions for Leon.**
1. **Format.** Per-step Markov think blocks in the state environment, with exactly three new
   tokens (`<think> </think> <no>`) and no `<try>`/`<backtrack>`/`<REVISE>`. Yes or no?
2. **Deviations.** May reference-guided traces replace roundabout reference steps with shorter
   valid ones? That shifts K12's length profile. The alternative is to force reference actions,
   so that every think must justify the generator's detours.
3. **Spend, in stages.**
   - (i) Calibration, about $7: 50 proofs on Opus 5.5, Sonnet 5.5 and Haiku 4.5, which decides
     the model under the 2× rule.
   - (ii) 2,000 proofs, about $37–154 (Opus batch).
   - (iii) 20,000 proofs, about $370–1,540, only after (ii) passes the causal-use test.
4. **Programmatic traces (d), now.** Build them as the bulk filler and the "is the LLM the active
   ingredient?" control arm: free, 1–2 days.
5. **First training experiment, and approvals.** Either a quick cold start (mid-training the
   existing Stage-1 checkpoint after an embedding permutation; minutes of GPU) or straight to
   from-scratch Stage 1 with traces (4 arms × 3 seeds, about 6–8 GPU-h on 3090s, plus evaluation).
   Also approve the base-model error sampling for error-seeding (GPU minutes).

## Next steps

1. Leon decides items 1–5.
2. If approved: build the renderer and batch submitter around `trace_dsl.validate --strict`, then
   run the calibration and report acceptance, tokens per accepted trace, and the error types.
3. Build (d) and the `lean_thinkn` mode in our own copy of the fork's code. The fork itself is
   never edited.

## Files

- [`trace_dsl.py`](trace_dsl.py): tokens, grammar, type checker, truth tables and validator
  (CLI: `validate`, `show`). It imports the fork read-only and never writes bytecode into it.
- [`pilot_traces.jsonl`](pilot_traces.jsonl): the 8 pilot traces (ASCII surface).
- [`production_prompt.md`](production_prompt.md): the v0 system prompt.
- [`test_trace_dsl.py`](test_trace_dsl.py): 26 tests (vocabulary, pilot strict validity, Lean,
  false-claim rejection, the prompt's worked example, lookahead, inconsistent contexts, the fork
  left unwritten, and the mutation test with a 0.88 rejection floor). Run with
  `code/.venv/bin/python -m pytest code/experiments/current/thinking_traces_20261005/test_trace_dsl.py`.
  It is not under `code/tests`, so `make -C code check` does not collect it; it needs the fork at
  `ND_TAKEHOME`.
- Sibling notes written by other agents the same day: [CONTEXT_SURVEY.md](CONTEXT_SURVEY.md),
  [OOD_CHECK.md](OOD_CHECK.md).

**Checks that ran:**
- `ruff check` and `ruff format --check` with the repository config, on both Python files;
- the 26 pytest tests pass;
- `trace_dsl.py validate pilot_traces.jsonl --strict` with Lean: 8 / 8;
- the mutation test (`test_mutation_sensitivity_strict`): 305 / 337 rejected under `--strict`.

Not run: `make -C code check` (repository-wide).
