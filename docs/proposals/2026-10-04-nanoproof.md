---
written_on: 2026-10-04
written_by: agent:claude
status: shelved (Dan, 2026-10-04) until the nanoproof author replies to Dan's email about the checkpoint; reproducing it (~100× our compute) is out of budget
---

# Proposal 25: nanoproof on our problems, with any `sorry`-free Lean proof allowed, and how much it leans on automation

## Why

[nanoproof](https://github.com/Kripner/nanoproof) (Matěj Kripner; read at commit `4f410c6`, 2026-05-11) is an open,
small-scale AlphaProof: a ~1B-parameter model, AlphaProof-style MCTS with a learned value, and RL against Lean 4 +
Mathlib. It reports **52.7 % on MiniF2F-valid** at 512 simulations. Dan (2026-10-04) wants to know how well it does on
our problems, under a much weaker rule than ours: **any valid Lean proof that does not use `sorry`**, not our one-ND-step
grammar. He also wants to know how much it relies on automation tactics, to judge whether the comparison is apples to
apples.

The second question matters more than it sounds. Our problems are classical propositional tautologies. Mathlib's
`tauto`, `itauto!`, `grind` and `aesop` are built for exactly this kind of goal. Under the weakened rule, one tactic
call may prove most of our sets with no model at all. So the design measures the automation floor first, and scores
nanoproof both with and without automation.

## nanoproof vs `state_env_october`

| | nanoproof (repo defaults) | `state_env_october` / `_r16` |
|---|---|---|
| **parameters** | ≈ 1.0 B (depth 26, width 1,664, 13 heads; my count from the defaults) | 9.56 M (6 × 384): about 100× smaller |
| **architecture** | nanochat GPT: RoPE, QK-norm, ReLU², untied embeddings, sliding-window layers (SSSL), no biases | ALiBiGPT: ALiBi + NoPE heads, Peri-LN; multi-token-prediction head during pretraining |
| **value** | yes: 64 proof-depth bins, emitted as special tokens through the LM head | none |
| **tokenizer** | GPT-2 BPE plus Lean / math special tokens | `lean_seq` (small fixed ND vocabulary) |
| **optimizer** | Muon + AdamW | Muon + AdamW |
| **stage 1: pretraining** | Nemotron-CC-Math, web mathematics (~20B-token corpus) | K12: 155,000 synthetic ND proofs, 2–12 lines, 1,200 s on one A40 |
| **stage 2: midtraining** | Lean code from GitHub (~65M tokens) | — |
| **stage 3: SFT** | LeanTree: ~260k (goal state → tactic) pairs from Mathlib, plus value targets | — (pretraining is already on proofs) |
| **stage 4: RL** | AlphaProof-style: MCTS on NuminaMath-LEAN statements; train on the tactics of found proofs (redundant steps pruned) plus value targets, with 10 % SFT replay. Unlikelihood on failed tactics exists but is off by default | Expert iteration: 32 attempts per target, train on every step of Lean-accepted proofs; 8 rounds (`_r16`: 16) |
| **what the model sees** | Lean goal state: hypotheses and goal, as Lean / Mathlib pretty-prints them (≤ 640 tokens) | proof state: hypotheses in scope and the goal (`lean_staten`) |
| **one action** | any Lean 4 tactic: `intro`, `cases`, `simp`, `tauto`, `grind`, … (≤ 24 generated tokens in RL) | one ND step written as a `have` term; the environment assigns names |
| **search at test time** | MCTS over an AND/OR goal tree: 6 tactics sampled per node, value-guided UCB, 8–512 simulations | none: independent samples (k up to 256), optionally guided redraw |
| **negative signal** | optional unlikelihood (off by default) | none |
| **judge** | Lean v4.27.0 + Mathlib, step by step through the LeanTree REPL | Lean 4 core (4.34.1) alone, on the finished proof |
| **automation** | allowed: `grind`, `lia`, `grobner`, `aesop` can be switched off with `--disable-solvers`, but evaluation then injects `grind` at every node anyway; SFT teaches all of Mathlib's tactics | none: every step is an explicit term |
| **compute** | ≈ 4 × 10²⁰ FLOPs (its README figure's compute axis) | ≈ 10 GPU-hours (A6000 / A40) per seed through round 8; at most ~5 × 10¹⁸ FLOPs even at full utilisation. So **≥ 100× less** |
| **trained on** | competition mathematics (MiniF2F-style), Mathlib | synthetic propositional ND only |

**What is shared:** both work in a proof-state interface (state in, one step out). Both learn from successful proofs
only (outcome-only selection). Both use Lean as the judge.

**What differs most:**
- scale (100× the parameters, ≥ 100× the compute, a general maths corpus);
- search (MCTS with a value head, against our plain sampling);
- the action space (all of Lean's tactics, including decision procedures, against one ND rule per step).

## Where the checkpoint comes from (decision for Dan)

**nanoproof publishes no weights.** The README links none, and Hugging Face has none under the author or the name
(checked 2026-10-04). There are three routes:

1. **Ask the author for the checkpoint behind 52.7 %.** This is cheapest and most faithful: evaluation only, about
   $15–30. Agents do not contact people, so only you can do this.
2. **Reproduce it at the repo defaults.** This is about 4 × 10²⁰ FLOPs, roughly 280 H100-hours at 40 % utilisation,
   or **$700–1,000** of GPU, plus a CPU fleet of Lean + Mathlib servers for RL. I don't recommend it.
3. **Reproduce a small version:** depth 12, about 160M parameters, the same four stages. Estimated **$50–100**. It is
   not "nanoproof as reported", and there is no published small-model number to validate it against. MiniF2F-valid
   would be the sanity check (does RL raise it, as in its figure?).

Phase 0 needs no checkpoint and should run first, whichever route you choose.

## Design

**Judge for this experiment** (Dan's weakened rule):
- Assemble each found proof into Lean source with nanoproof's own `construct_proof_source`.
- Compile it in nanoproof's environment (Lean v4.27.0 + Mathlib).
- Accept it if it elaborates with no error and `#print axioms` ⊆ {`propext`, `Classical.choice`, `Quot.sound`}. So
  `sorry` / `admit` (`sorryAx`) is rejected.
- `native_decide` (`Lean.ofReduceBool`) is accepted but counted separately.
- As a sanity check, the same judge re-checks `state_env_october_r16`'s existing proofs; all must pass.

**Statements:** render each problem as `theorem t (P Q … : Prop) (h1 : A₁) … : goal := by sorry`, with ⊥ as `False`.
Reuse `lean_check`'s renderer; Charles's release ships Lean renderings already.

**Sets** (scoring only; never trained or tuned on):

| set | n | why |
|---|---|---|
| textbook72 | 72 | headline (eval-only copy, manifest rule on `origin/dan_textbook72`) |
| holdout250 | 250 | synthetic, so no contamination risk |
| Charles's release: `candidate_v0`, `candidate_v1`, `batch3`, `hand_proved_14` | 187 | real textbooks, longer (sha256 against `MANIFEST.json`) |
| long pools: L_true 13–16 (400), lengths ≥ 17 (21) | 421 | length |
| `rl-continue`'s group C (union over seeds) | ≤ 99 | what our models cannot reach |

**Arms:**

| arm | what | question |
|---|---|---|
| **S0: automation alone** (no model; CPU) | each of `tauto`, `itauto`, `itauto!`, `grind`, `aesop`, `simp_all`, `solve_by_elim`, `exact?` alone, as the whole proof, 10 s timeout; report each and their union | How much of the weakened proof space is free? |
| **N1: nanoproof as released** | its default evaluation (solvers allowed) at 64 and 512 simulations | Its headline number on our sets |
| **N2: nanoproof without automation** | the S0 tactics plus `simp`/`simpa`, `decide`/`native_decide`, `omega`/`lia`, `linarith`/`nlinarith`, `norm_num`, `grobner`, `apply?` all removed from the model's samples, and **`grind` injection switched off** (a one-flag code change) | What its network and search can do on their own: the apples-to-apples arm |
| **ours** | `state_env_october_r16`, plain and guided, k 256 (existing numbers plus the new judge) | The comparison |

**Automation use** (from N1's found proofs). Each tactic is classified as:
- **heavy:** a decision procedure or search (the S0 / N2 lists);
- **light:** a single lookup (`assumption`, `contradiction`, `trivial`, `rfl`, `exact h`);
- **manual:** `intro` / `rintro`, `apply` / `exact` / `refine` with an explicit term, `constructor`, `left` / `right`,
  `cases` / `rcases` / `obtain`, `by_contra` / `by_cases`, `exfalso` / `absurd`, `have`.

The report gives:
- the share of solved theorems whose proof uses any heavy tactic;
- the share of proofs that are one heavy tactic at the root;
- heavy tactics as a share of all tactic steps;
- per-tactic frequencies;
- proof length in tactics, against our ND line count.

**Compute:** record each arm's generated tokens, Lean calls and wall-clock per theorem (AGENT_POLICY compute rows).
With 100× the parameters, matching compute is not meaningful, so the comparison is at each system's own settings.

## Pre-registration sketch

- **S0:** the union of the automation tactics proves ≥ 95 % of textbook72 and holdout250. If so, the weakened proof
  space is nearly free, and N1's headline says little about the model. It proves fewer of `hand_proved_14` (large
  formulas, timeouts).
- **N1:** proves at least S0's union. ≥ 80 % of its proofs contain a heavy tactic, and most are one-liners.
- **N2:** below `state_env_october_r16` on textbook72 and on problems over 10 lines. Its tactics are capped at 24
  tokens and learned from Mathlib, where propositional goals are closed by `tauto` rather than by hand. Its MCTS and
  100× scale may still win on short problems.
- **Falsifier for "automation decides the comparison":** N2 within 3 problems of N1 on textbook72.
- **Contamination check:** nanoproof's corpora (web maths, GitHub Lean, Mathlib) may contain the public textbooks'
  exercises. If N2 beats our model on Charles's textbook sets but not on holdout250 (synthetic), suspect
  contamination.

## Cost

- **Code:** one executor session. It covers:
  - a RunPod pod with Lean v4.27.0 + Mathlib and a LeanTree server (nanoproof's setup);
  - the statement renderer;
  - the N2 ban list and switch for `grind` injection;
  - the tactic classifier;
  - the whole-proof re-check.
- **Phase 0 (S0 and the judge sanity check):** CPU pod, ≈ $2–5.
- **Phase 1 (N1, N2):**
  - route 1: ≈ $15–30 (one A40 for a 1B model, plus Lean servers; at most ~3,000 tactic calls per theorem at 512
    simulations);
  - route 3: about $50–100 more;
  - route 2: $700–1,000+.
