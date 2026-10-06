# STATE_ENV — an AlphaProof-style proof state for the policy (run `state-env`, proposal 13)

Every other model in this repository writes a whole proof in one pass, conditioned on the theorem and its own text:
π(y_t | theorem, y_<t). This environment instead shows the policy the **tactic state** and asks for **one step**.
Lean still decides whether the finished proof counts (`lean_gate` / `lean_judge`; Dan, 2026-09-27).

## What the policy sees

The focused goal's Lean tactic state and nothing else — no theorem statement (the theorem *is* the initial state),
no proof history (arm S; arm `SH` prepends the actions so far):

```
<st> h1 : ( P ∧ Q ) <nl> n1 : ( P ∧ Q ) <nl> n3 : P <nl> ⊢ ( ¬ ( Q ∧ R ) ) <act>
```

Hypotheses in scope, in Lean's order: the premises `h1..hm`, the `have`s completed at this or an enclosing level, and
the box binders. Then `⊢` and the goal. Formulas are `lean_seq`'s fully parenthesised rendering, one symbol per token.
The tokenizer mode `lean_state` is `lean_seq`'s vocabulary plus exactly four tokens — `<st> <nl> ⊢ <act>` — and
`lean_seq`'s own random name offset; `lean_seq`'s vocabulary is unchanged at 107 tokens, so every on-file `lean_seq`
checkpoint still loads.

## What the policy does

One action, the literal `lean_seq` text of one step:

| action | when |
|---|---|
| `have n<k> : F := <atomic term> ;` | an ordinary line (`PR`, `R`, `∧`-intro/elim, `→`/`¬`-elim, `∨`-intro, `.elim`, `Classical.byContradiction`) |
| `have n<k> : F := ( fun ( n<j> : A ) => by` | opening an implication box (`F = (A → B)`, goal `B`) or a negation box (`F = (¬ A)`, goal `False`) |
| `have n<k> : F := Or.elim n<i> ( fun ( n<j> : A ) => by` | opening an `Or.elim`, first branch |
| `exact n<k>` | closing the focused goal |

The environment applies the action **syntactically** and renders it back into `lean_seq` text. `exact n` is the one
action whose rendered text the environment completes, because the tokens that *close a box* encode box structure, not
a proof step, and a real Lean tactic state does not tell you whether closing this goal also closes a lambda:

| focused frame | `exact n` renders as |
|---|---|
| top level | `exact n` — the proof is finished |
| implication box | `exact n ) ;` |
| negation box | `exact ( n : False ) ) ;` |
| `Or.elim` branch 1 | `exact n ) ( fun ( n<next> : B ) => by` — the second branch is opened, binder named by the environment |
| `Or.elim` branch 2 | `exact n ) ;` |

`n<next>` is `max name index used in this attempt so far + 1`, which is exactly `lean_seq`'s "numbered in order of
first appearance", so replaying a stored proof's actions reproduces its text **byte for byte**.

### Names, and arm SN

In arms S and SH the model writes every name it introduces, exactly as in `lean_seq`. That turned out to matter.
`lean_seq`'s Stage-1 augmentation shifts all names by a random offset (so every name token is trained). As a result
**the first name an attempt introduces is not a function of any state**, since nothing is in scope yet. A greedy policy
takes the mode of a near-uniform distribution. The traced seed-1 model (SN-v1) puts it at `n64` and then cites names that do not exist
(`se_trace.py` shows it). Separately, once a box closes, `lean_seq`'s global first-appearance index is not
determined by the scope either (3.00 % of `have` actions).

Arm **SN** (`lean_staten`) removes both. Training proofs are rewritten by `state_env.canonicalise` so every
introduced name is `max index in scope + 1` (an alpha-variant: same ND proof, same Lean verdict). At sampling time,
`Env(canon=True, base=b, assign=True)` makes the **environment** name what each step introduces, counting from a
per-attempt random base `b ~ U[0, 32]`, as Lean's own `intro`/`have` naming would. The model's name token is
overridden, and its citations name hypotheses the state shows. ("SN-v1" was the same checkpoints sampled without
`assign`; see the pre-registration addenda.)

There is **no symbolic type check**: scope, hypotheses and the goal follow from the actions because every `have`
declares its formula, and a wrong *term* is caught by Lean on the finished proof, exactly as in the whole-proof loop.
The checks the environment does make are the ones `lean_tok.inverse`'s strict grammar makes, so an attempt that
finishes always parses: cited names in scope; `exact n` only if `n` is the focused frame's last statement **and** its
declared formula is the goal; premise lines only at depth 0, in order, before any other line; a box binder whose
formula is the one its `have` declares. An action that fails any of them ends the attempt (counted as `syntax`).

## One proof, end to end

`( ( P ∨ ( S ∨ P ) ) ∨ P ) ⊢ ( P ∨ ( S ∨ P ) )`, whose `lean_seq` text is

```
have n1 : ( ( P ∨ ( S ∨ P ) ) ∨ P ) := h1 ; have n2 : ( P ∨ ( S ∨ P ) ) := Or.elim n1 ( fun ( n3 : ( P ∨ ( S ∨ P ) ) ) => by exact n3 ) ( fun ( n4 : P ) => by have n5 : ( P ∨ ( S ∨ P ) ) := Or.inl n4 ; exact n5 ) ; exact n2
```

| # | state the policy sees | action it writes |
|---:|---|---|
| 1 | `<st> h1 : ((P∨(S∨P))∨P) <nl> ⊢ (P∨(S∨P)) <act>` | `have n1 : ((P∨(S∨P))∨P) := h1 ;` |
| 2 | `… <nl> n1 : ((P∨(S∨P))∨P) <nl> ⊢ (P∨(S∨P)) <act>` | `have n2 : (P∨(S∨P)) := Or.elim n1 ( fun ( n3 : (P∨(S∨P)) ) => by` |
| 3 | `… n1 : … <nl> n3 : (P∨(S∨P)) <nl> ⊢ (P∨(S∨P)) <act>` | `exact n3` |
| 4 | `… n1 : … <nl> n4 : P <nl> ⊢ (P∨(S∨P)) <act>` | `have n5 : (P∨(S∨P)) := Or.inl n4 ;` |
| 5 | `… n4 : P <nl> n5 : (P∨(S∨P)) <nl> ⊢ (P∨(S∨P)) <act>` | `exact n5` |
| 6 | `… n1 : … <nl> n2 : (P∨(S∨P)) <nl> ⊢ (P∨(S∨P)) <act>` | `exact n2` |

Step 3's `exact n3` closes branch 1 and the environment opens branch 2 with `n4 : P` (step 4's state). Step 5 closes
branch 2, and at step 6 the finished `have n2` is a hypothesis of the parent — `n3`, `n4`, `n5` are gone.
(Parentheses are compressed in this table only; the tokens are one symbol each.)

## The three gates

| gate | what it checks | result |
|---|---|---|
| 1 | every one of the 155,000 control proofs (`data/p2/train_depth3_f0_a1.jsonl`) decomposes into actions and the rendered actions reassemble its `lean_seq` text **byte for byte** | **0 failures / 155,000** |
| 1b | a random 5,000 of the reassembled texts, judged by `lean_judge` (nd2lean + Lean 4 core) | **0 rejected / 5,000** |
| 2 | **your state is Lean's state**: cut a proof at a random step, put `trace_state` there and `sorry` in every goal still open, and compare Lean's hypotheses (names and types) and goal with the renderer's, as parsed formulas | **0 mismatches / 2,800** (2,000 random + 500 `Or.elim`-branch + 300 negation-box cuts) |
| 3 | every control proof's actions fed through the environment loop: all syntactic checks pass and every attempt finishes | **0 failures / 155,000** |
| 2b | gate 2 on **7–15-line proofs the model itself found** (T1 S s0, round 3) | **0 mismatches / 1,500** |
| SN | gates 1 / 1b / 2 / 3 on the canonical variant | 0 / 155,000; 0 / 5,000; 0 / 1,300; 0 / 155,000 |
| fuzz | 800,000 random token sequences as actions: the environment never raises | **0 crashes** |

Files: `artifacts/se/gate13.json`, `artifacts/se/gate2.json`, `artifacts/se/gate2_ore.json`, `artifacts/se/gate2_neg.json`,
`artifacts/se/gate2b.json`, `artifacts/se/gate13_canon.json`, `artifacts/se/gate2_canon*.json`,
with the per-case Lean sources in `artifacts/se/gate2*_cases.jsonl`.

Shape statistics from gate 1, which is where the settings come from: **5.000 actions per proof** on the control set
(3/4/5/6/7 for lengths 2/3/4/5/6, 31,000 each), longest single action **220** tokens, longest state **303** tokens,
mean state **67** tokens. A `have`'s name is the global first-appearance index, which the state no longer determines
once a box has closed and taken its names out of scope: this happens for **15,821 of 526,784 `have` actions (3.00 %)**.

## How to run it

```bash
# the gates (CPU; Lean at ~/.elan/bin/lean)
python3 state_gates.py --data data/p2/train_depth3_f0_a1.jsonl --lean 5000 --out artifacts/se/gate13.json
python3 state_gate2.py --n 2000 --out artifacts/se/gate2.json
python3 state_gate2.py --n 500 --kinds or1,or2 --out artifacts/se/gate2_ore.json

# Stage 1 on (state, action) pairs: 128 whole proofs per step -> 640 pairs, loss on the action tokens only
python3 state_train.py --data data/p2/train_depth3_f0_a1.jsonl --heldout data/p2/heldout.jsonl \
    --mode lean_state --steps 6000 --recs 128 --cap 6 --seed 0 --out ckpts/se/stage1_S_s0.pt

# held-out greedy, in the environment
python3 state_eval.py --ckpt ckpts/se/stage1_S_s0.pt --in data/p2/heldout.jsonl --k 1 --temperature 0 \
    --batch 2048 --out artifacts/se/heldout_S_s0.jsonl --summary artifacts/se/heldout_S_s0.json

# expert iteration in the environment: ladder rung T1, and the frozen control at equal attempts
python3 state_ladder_ei.py --init ckpts/se/stage1_S_s0.pt --name la_T1_S_s0 --seed 0 --batch 2048 \
    --train data/p2/train_depth3_f0_a1.jsonl --heldout data/p2/heldout.jsonl
python3 state_ladder_ei.py --init ckpts/se/stage1_S_s0.pt --name la_frozen_S_s0 --seed 0 --batch 2048 \
    --train data/p2/train_depth3_f0_a1.jsonl --heldout data/p2/heldout.jsonl --no_train

# the tables and the figure
python3 se_analysis.py --runs artifacts/se/la_T1_S_s0 artifacts/se/la_frozen_S_s0 \
    --c0 artifacts/dsg/la_T1_c0_s0 artifacts/dsg/la_frozen_c0_s0 --term_size --out artifacts/se/summary.json
python3 se_figures.py --summary artifacts/se/summary.json --out figures/state_env.png
```

Code: `state_env.py` (renderer, decomposition, environment), `lean_tok.py` (`lean_state` / `lean_stateh` modes),
`state_train.py`, `state_sample.py` (the batched environment loop), `state_eval.py`, `state_ladder_ei.py`,
`state_gates.py`, `state_gate2.py`, `se_analysis.py`, `se_tables.py`, `se_figures.py`, `se_trace.py` (step-by-step
rollouts); pod scripts in `pod/se/`. Tests: `tests/test_state_env.py`. Arm SN: add `--mode lean_staten` to
`state_train.py`; sampling picks up canonical, environment-assigned names from the checkpoint's mode. `state_train.py`
micro-batches at `--max_tokens` padded tokens with token-weighted accumulation (same gradient; arm SH's long prompts
need it on a 24 GB card).
