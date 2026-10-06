# Pre-registration — run `state-env` (proposal 13: an AlphaProof-style state for the policy)

Written 2026-09-28, **before any pod exists for this run** (`podbudget state-env` registered at 60 h / $30;
`~/pods.log` has no `se-*` line at this commit). Executor: agent:claude. Branch `dan_state-env` (worktree from the
fork's `origin/dan`). Brief: the run brief for `state-env`; proposal
`~/nd-rl/docs/proposals/2026-09-28-state-conditioned-environment.md`; policy `AGENT_POLICY.md` (nd-rl canonical).
Nothing else of mine is running.

## Question

Every whole-proof model in this project writes π(y_t | theorem, y_<t) and hits the same length wall: on the ladder's
2,285-theorem transfer pool, `L*` = 12 at T1 and **0 theorems solved at `L_true` ≥ 13 in all 11 `ds-generator` ladder
runs**. Does giving the policy the **proof state** — every hypothesis in scope and the current goal, as a
pretty-printed Lean tactic state — move that wall?

## What the policy sees and does (the design I will actually run)

`state_env.py`. The observation is the focused goal's tactic state and nothing else:

```
<st> h1 : ( P ∧ Q ) <nl> n1 : ( P ∧ Q ) <nl> n3 : P <nl> ⊢ ( ¬ ( Q ∧ R ) ) <act>
```

The action is one `lean_seq` step: a `have … ;` line, a box opening up to its `by`, an `Or.elim` opening, or
`exact n`. Actions are applied syntactically (parse; cited names in scope; `exact n` only if `n` is the focused
frame's last statement **and** its declared formula is the goal; premise lines only at depth 0, in order, before any
other line; a box binder whose formula is the one its `have` declares) and rendered back into `lean_seq` text. The
assembled text goes to Lean through `lean_gate` / `lean_judge` — **Lean alone decides** (Dan, 2026-09-27). There is
**no per-step symbolic type check** (the brief allows one only if its agreement with Lean is measured; I do not use
one, so there is nothing to measure).

**Deviation from the brief, decided before any pod.** The brief's action set is the literal text chunks, including
the tokens that close a box (`exact n ) ;`, `exact ( n : False ) ) ;`, and for `Or.elim` the second branch's opener).
I instead let the **environment** supply those tokens; the model's closing action is uniformly `exact n`. Reason:
those tokens encode *box structure*, not a proof step, and `lean_seq` is the only reason they exist — a real Lean
tactic state does not tell you whether closing this goal also closes a lambda, so keeping them in the action would
have forced a frame-kind marker into the observation and made the state *less* Lean-faithful, which is the one thing
this run is testing. The `Or.elim` second-branch binder is named `n<max index used so far + 1>`, which is exactly
`lean_seq`'s "numbered in order of first appearance", so replaying a control proof's actions reproduces its text
**byte for byte** (gate 1, below, is run on the concatenation of the rendered chunks and passes on all 155,000).
The model still writes every `have`'s own name and every implication/negation box binder, so the surface form the
model produces is `lean_seq`'s.

Known imperfection, measured and reported rather than fixed: a `have`'s name is the global first-appearance index,
which the state does not determine once a box has closed and taken its names out of scope. On the control set this
is **15,821 of 526,784 `have` actions (3.00 %)**; in those the target name is not `max index in scope + 1`. A name
that collides with one in scope is a legal Lean shadow and the environment resolves it lexically, as
`lean_tok.inverse` does.

Arms, two Stage-1 seeds each:

- **S — state only** (AlphaProof-faithful; `lean_state`). Primary.
- **SH — proof so far + state** (`lean_stateh`). Run if the budget allows; dropped first.
- **C0 — not re-run.** The whole-proof control on file: `ckpts/lf/stage1_a1_seq_s{0,1}.pt`, 3,214,336 parameters,
  4 layers / d 256, `lean_seq`, **from scratch**, trained on `data/p2/train_depth3_f0_a1.jsonl` (155,000 records,
  cap 6, depth-3 f = 0), `train.py --bs 128 --steps 6000`. Its numbers were measured **under Lean ∧ `nd_verify`**
  (pre-2026-09-27) and at other sampler settings (batch 512, `ND_SAMPLE_COMPACT=0`); per `AGENT_POLICY.md` a
  settings change is a sampling re-draw, not a correctness change, and the `ds-generator` spread on the same two
  checkpoints (T1 856 / 890 / 923 at s0) is what that re-draw looks like.

Everything else is the control's: the same 155,000 proofs, 4 layers / d 256 / 8 heads, 6,000 Stage-1 steps,
`ladder_ei.py`'s rung T1 (8 rounds × k 32, temperature 0.8) on `data/ladder/rl_targets.jsonl` read out on
`data/ladder/transfer.jsonl`, the frozen control at equal attempts, `--retain 20000 --max_per_thm 4 --rl_weight 4
--ft_steps 600 --ft_lr 3e-4`, held-out `data/p2/heldout.jsonl`.

**Batch equivalence.** A Stage-1 step draws **128 whole proofs** and trains on *all* their (state, action) pairs, so
the model sees the same proofs the same number of times as the control. The control set decomposes into exactly
**5.000 actions per proof** on average (3 / 4 / 5 / 6 / 7 for lengths 2 / 3 / 4 / 5 / 6, 31,000 each), so the pair
batch is **640**. Loss is on the action tokens only.

**Sampler settings** (`AGENT_POLICY.md`, 2026-09-28: the fast path, the largest batch that fits, fixed across arms):
`ND_SAMPLE_PATH=fast`, `early='eos'`, compaction on, per-row seeding; env batch **2,048 attempts**, action budget
`--max_action 256`, attempt budget `--max_steps 48`. 256 covers the control set's longest single action (220 tokens)
with margin; the fraction of actions that hit it is reported. Peak `torch.cuda.max_memory_allocated` is recorded in
the first job and reported with the batch. These settings are held fixed across every arm and both seeds. The
control's on-file numbers were taken at whole-proof batch 512 — a different unit; this is a re-draw, not a bias
(`NOISE_FLOOR.md`).

## Gates before any RL (CPU, on the VPS; run at this commit)

| gate | result | file |
|---|---|---|
| 1 round trip: all 155,000 control proofs decompose and the rendered actions reassemble the `lean_seq` text byte for byte | **0 failures / 155,000** | `artifacts/se/gate13.json` |
| 1b a random 5,000 of the reassembled texts accepted by `lean_judge` (nd2lean + Lean core) | **0 rejected / 5,000** | same |
| 2 your state is Lean's state: random proof prefixes, `trace_state` at the cut, hypotheses (names and types) and goal compared | **0 mismatches / 2,000** random + 500 `Or.elim`-branch + 300 negation-box cuts | `artifacts/se/gate2*.json` |
| 3 environment replay: every control proof's actions fed through the environment loop complete and are accepted | **0 failures / 155,000** (the gate-1 walk *is* the replay: every syntactic check passes and the attempt finishes) | `artifacts/se/gate13.json` |

Gate 2's cases come from the control set (cap 6, depth 3), which reaches all five focused-frame kinds
(top / imp / neg / `Or.elim` branch 1 / branch 2). It does not reach the ladder pools' longer proofs, because those
pools carry no reference proof. I will re-run gate 2 on a sample of the proofs the run itself finds (7–14 lines) and
report that as gate 2b.

## Expected results — the numbers I am committing to

`L_true` labels on the ladder pools are **ND-derived upper bounds under Lean** (`minlen.py`; QUESTIONS.md
2026-09-27). All S / SH numbers are under **Lean alone**; the C0 column is under **Lean ∧ `nd_verify`**.

| quantity (per arm × seed) | C0 s0 / s1 (on file) | my prediction for S (both seeds) |
|---|---|---|
| held-out greedy, 5,000, overall | 0.9088 / 0.8968 | **0.85 – 0.95** (point 0.91) |
| held-out greedy, 6-line bin | 0.686 / 0.584 | **≥ 0.60**, and ≥ C0's own seed value on at least one seed |
| T1 transfer solved / 2,285 (round 8) | 890 / 965 | **700 – 1,250** (point 950) |
| T1 transfer `L*` | 12 / 11 | **12** (range 11–13) |
| T1 transfer solved at `L_true` ≥ 13 (of 23) | 0 / 0 (0 in all 11 runs) | **0 – 3** (point 1) |
| T1 textbook schemata solved / 760 | 38 (s0) | **20 – 120** |
| frozen transfer solved / 2,285 | 158 / 114 | **80 – 400** (the frozen run-to-run range is 62–265, `NOISE_FLOOR.md`) |
| frozen `L*` | 9 / 9 | **9** (range 8–10) |
| mean actions per attempt, round 8 | — | **4 – 12** |
| attempts ended by a syntactic error, round 8 | — | **< 45 %** of attempts |
| actions that hit `--max_action` 256 | — | **< 0.1 %** |

**SH relative to S**: I predict SH ≥ S on held-out greedy by ≤ 2 pp and **indistinguishable** from S on T1 transfer
solved (difference inside the frozen ladder's 62–265 run-to-run range, so uninformative at n = 2); i.e. I predict
the history adds nothing once the state is there.

## Falsifiers (proposal 13, verbatim)

- **The state is the wall** — arm S reaches T1 `L*` ≥ 13 **with ≥ 5 solved at `L_true` ≥ 13 on both seeds**.
- **The state is not the wall (at this scale)** — arm S stays at T1 `L*` ≤ 12 **with 0 solved at `L_true` ≥ 13 on
  both seeds**. That points at capacity or search.

Neither outcome is forced: anything in between (e.g. 1–4 solved at `L_true` ≥ 13) is reported as "moved but not
decisive". **My prior, stated before the run: the second falsifier fires** (I put ≈ 0.6 on it, ≈ 0.1 on the first,
≈ 0.3 on in-between).

## Budget, order and stop rule

Budget **$30**, ceiling **60 pod-hours**, registered before the first pod. Balance floor $100 (`rpbalance` was
$136.49 at the start). Order: Stage-1 S s0 → held-out s0 → T1 s0 → frozen s0 → Stage-1 S s1 → held-out s1 → T1 s1 →
frozen s1 → SH. Drop order if the projection does not fit: **SH, then seed-1 frozen, then seed-1 T1**. A ladder run
that is cut off is reported at the round it reached, with the round number stated. GPU class and its real billed rate
are recorded in `numbers.md`; pods are deleted when their work is pulled; `ckpts/`, `artifacts/` and `data/` go to
`hf://buckets/dan-pandori/nd-rl/state-env/` as the run goes, not only at the end.

Stop rule: stop at 60 pod-hours or $30, whichever comes first (`pod_budget_watch` deletes this run's pods at 100 %);
extend only within the declared budget with `podbudget state-env --extend`, and ask in `QUESTIONS.md` beyond it.

---

## Addendum — arm SN (canonical names), added 2026-09-28 ≈ 05:05 UTC

**Added after seeing a result, and labelled as such.** Arm S seed 0's held-out greedy is 0.9580; seed 1's is
**0.8012**, with **581 of 5,000 attempts (11.6 %) ending on `unbound`** — the policy cited a name that is not in
scope. The two Stage-1 models have the same validation loss (0.0701 / 0.0702), so this is not a training failure: it
is the imperfection the pre-registration names. A `have`'s name in `lean_seq` is the **global first-appearance
index**, which the state stops determining as soon as a box closes and takes its names out of scope (3.00 % of the
control set's `have` actions), so the policy must guess it, and one seed guesses worse than the other.

**Arm SN** removes that one non-Markov token and changes nothing else: `state_env.canonicalise` rewrites each proof
so that every name a step introduces is `max index in scope + 1`, which the state does determine. The rewritten text
is an **alpha-variant** — `lean_tok.inverse` returns the same ND proof and Lean's verdict is invariant to hypothesis
renaming — so it is the same proofs, the same set, the same schedule, the same environment, with the policy's naming
decision made a function of the observation. Tokenizer mode `lean_staten`. Gates re-run on the canonical variant
before its first pod's measurements: gate 1 **0 failures / 155,000**, gate 1b **0 Lean rejections / 5,000**,
gate 2 **0 mismatches / 1,300**, and **0 of 526,784 `have` actions** now have a name that is not `max in scope + 1`
(`artifacts/se/gate13_canon.json`, `artifacts/se/gate2_canon*.json`).

SN is a **diagnostic arm, not a falsifier arm**: the pre-registered falsifiers are about arm S and stay about arm S.
Predictions for SN, written now, before any SN measurement exists:

| quantity | prediction for SN (both seeds) |
|---|---|
| held-out greedy, overall | **≥ 0.94**, and the seed-to-seed spread **< 3 pp** (arm S's is 15.7 pp) |
| attempts ending on `unbound` | **< 1 %** (arm S seed 1: 11.6 %) |
| T1 transfer solved / 2,285 | **≥ arm S on the same seed**, and I expect the gain to be large on seed 1 and small on seed 0 |
| T1 transfer solved at `L_true` ≥ 13 | **0 – 5** |

Budget: two more pods (`se-3`, `se-4`), projected ≈ 10 more pod-hours, inside the declared 60 h / $30.

## Addendum 2 — arm SN, second version (environment-assigned names), 2026-09-28 ≈ 05:45 UTC

**Also added after seeing a result.** SN as first built (call it **SN-v1**) scored 0.9716 on seed 0 and **0.623 on
seed 1** on held-out, with 1,424 `unbound` failures, including on 2-line proofs. A step-by-step trace
(`se_trace.py`) showed the cause, which also explains arm S's seed spread: `lean_seq`'s **random name offset** (the
Stage-1 augmentation that makes every name token a trained token) makes the **first** name an attempt introduces
unpredictable from any state, because nothing is in scope yet. Greedy decoding then picks the mode of a near-uniform
distribution. Seed 1's models put it at `n64` and then counted *down*, citing `n62` when `n64` was the only name in
scope. Seed 0's models happened to land low. Canonicalising the later names (SN-v1) could not fix the first one.

**SN-v2 = the same two SN-v1 checkpoints (no retraining), sampled with the environment naming every hypothesis a
step introduces** — `max index in scope + 1`, counted from a per-attempt random base drawn like the training offset,
`U[0, 32]`. The model still writes a name, and the environment overrides it, as Lean's own `intro`/`have` naming
would. Citations are the model's and name hypotheses the state shows. On shifted canonical control proofs the
override is a no-op (0 of 6,370 defining names changed). In SN-v2 the policy's input fully determines everything the
environment accepts from it.

What SN-v2 is and is not: it is a sampling-side change to a diagnostic arm, decided after seeing SN-v1. Its held-out
numbers (**0.9700 / 0.9584**) were measured after this decision, so they do **not** count as confirming addendum 1's
held-out prediction; they are reported as what they are. The addendum-1 ladder predictions (T1 ≥ arm S on the same
seed; 0–5 solved at `L_true` ≥ 13) are carried over unchanged to SN-v2, and no SN-v2 ladder result exists at this
commit. The SN-v1 ladders were stopped before their first round finished. SN-v1 held-out files are kept as
`artifacts/se/heldout_SNv1_s*.json`. **Arm S is left exactly as pre-registered**, flaw included, and remains the arm
the falsifiers are about.
