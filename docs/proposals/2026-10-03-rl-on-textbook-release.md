---
written_on: 2026-10-03
written_by: agent:claude
status: proposed for Dan (2026-10-03); not queued. Needs Dan's confirmation with Charles first (see "Before it runs")
---

# Proposal 23: continue RL on harder, longer targets from Charles's textbook release

## Why

`trajectory`'s cap-12 RL is close to saturation on its own targets: 97–98 % of the 4,495 `rl_targets` are solved,
with 5–12 new per round by round 8. Expert iteration learns only from successes, so the unsolved remainder gives no
signal. The ceiling is the **target pool**. `rl-continue` (running, rounds 9–16 on the same targets) measures the
saturation directly and doubles as this proposal's control.

Charles (nd-rl branch `charles/main`) has built a pool of **harder and longer** problems:
`data/textbook_problem/release_2026-10-02/`, 412 problems from real textbooks.

| file | problems | what it holds |
|---|---|---|
| `candidate_v0` | 87 | forall x, Carnap; minimal proofs of 3–29 lines |
| `candidate_v1` | 72 | Roy, Carr; 3–28 lines |
| `batch3` | 14 | Pelletier, forall x R³, Oxford |
| `hand_proved_14` | 14 | proofs of **20–102 lines**; no search found them |
| `derived` | 225 | variants built by proof surgery: completions, prefixes, curry, permutation, De Morgan, … |

Details:
- Every row carries a reference proof checked by the verifier. 408 of 412 also check in Lean.
- The prompts are in our ND format, so the proof-state environment can use them directly.
- Charles has already scored our best-state r8 model, at k 32, T 1.0, with `nd_verify` (his protocol):
  - candidate_v0: about 75 / 104;
  - candidate_v1: about 50 / 80;
  - batch3: 10–12 / 21;
  - **hand-proved: 0 / 14.**

  So the pool has the headroom our pools no longer do.

## Before it runs (for Dan)

- **It is Charles's evaluation benchmark.** The release is published with baselines, including ours. Training on part
  of it changes what those numbers mean. Please confirm with Charles that RL on a train split is OK, and agree the
  split (or ask him for one). Agents do not contact him.
- The sealed textbook 72 are not in the release, and are never trained on.
- **Licences:** Roy is CC BY-NC-SA (research use is fine). Carr and the Oxford booklet (6 problems) have no stated
  licence, so they stay evaluation-only.

## Design

**Split** (proposed; Charles's view takes precedence):
- Split at the level of **base problems**, by renaming class *and* curried normal form, which is his audit rule.
- **Derived rows go with their base.** Rows marked `curried_twin_of` are never split from it.
- Stratify by file and by `min_lines` bin, 50 / 50.
- `hand_proved_14`: sort by length and alternate, giving 7 train and 7 eval.
- The 6 unlicensed problems are eval-only.

**Starting point:** `trajectory`'s cap-12 r8 checkpoints, 3 seeds (Robbie's recipe, proof-state environment). Every
arm runs rounds 9–16 with `trajectory`'s protocol and **the same attempts per round**.

| arm | targets | what it tests |
|---|---|---|
| **control** | `rl_targets` only | `rl-continue`, reused, not re-run |
| **C1, harder targets** | `rl_targets` + the train split (bases plus plain derived variants), with half of each round's attempts on the train split | Does supplying unsolved, longer targets restart learning? |
| **C2, + reverse curriculum** | C1, plus start states from the train split's reference-proof **prefixes**: start near the end of a long proof, and move the start back 2–4 lines once it is solved at ≥ 25 % | Can RL climb towards 20–100-line proofs with a curriculum (Salimans & Chen 2018; AlphaProof's variants are the same idea)? |

**In C2, train only on the model's own steps from a given start, never on the reference prefix.** Imitating
reference prefixes would be supervised learning on known proofs, not RL.

## Read-outs

- **The eval split** at k 256 (Lean alone) and in Charles's protocol (k 32, T 1.0, `nd_verify`), so the results are
  comparable with his baselines.
- By length bin, with **eval-half hand-proved** as the headline for C2: does anything above 20 lines become solvable?
- **Unaffected sets:** the sealed textbook72 (scoring only), holdout250, `trajectory`'s group C, and the long pools.
- **Retention:** held-out greedy.

## Pre-registration sketch

- **C1:** beats control on the eval split, beyond the seed spread.
- **C2:** beats C1 on eval problems of ≥ 16 lines, and solves ≥ 1 eval hand-proved problem on ≥ 2 seeds.
- **Falsifier:** neither arm beats control on the eval split, so harder targets do not restart learning.

## Cost

- **Code:** one executor session, mostly C2's start-from-a-prefix states. The environment already replays proofs
  (`decompose`).
- **Pods:** 6 new ladders × 8 rounds at about $4–5 each, plus read-outs. **≈ $35** (budget $40).
- **Time:** ≈ 10–12 h.
