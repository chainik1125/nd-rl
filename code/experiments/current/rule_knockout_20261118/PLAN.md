---
written_on: 2026-11-18
written_by: agent:claude
---

# Rule knockouts: remove one inference rule from pretraining, see whether RL brings it back

Plan for the runs launched 2026-11-18 at 3:32 pm PT for that evening's group meeting. Robbie asked for this at
about 3:15 pm and left for another meeting; this file says what is running and why. Results go in `README.md` here
once there are any. Nothing below is a result.

## Question

Robbie's first-principles design for "does RL create a proof strategy": give the model all the strategies but one in
pretraining, run RL, and see whether the missing one appears in the proofs it writes. His hypothesis is that a
heavily nerfed base may pick the rule up fast in RL because it has to.

Dan's knockout (`create-or-elicit`, 2026-10-12) did this for one strategy, classical proof by contradiction, and
removed a pattern (double-negation elimination on a negation the proof built itself), not the rule. This run removes
four rules outright, one per arm.

## Arms

Every pretraining record whose proof uses the rule is dropped. Nothing is refilled, so the arms differ in size.

| arm | rule removed | strategy | pretraining records left (of 154,990) | RL targets whose reference proof uses it (of 13,946 with one) |
| --- | --- | --- | --- | --- |
| `mp` | `IMPE` | modus ponens | 95,561 | 13,774 |
| `cp` | `IMPI` | conditional proof | 97,034 | 4,224 |
| `exfalso` | `BOTE` | ex falso | 152,039 | 4,469 |
| `dn` | `DN` | double-negation elimination (all classical reasoning) | 142,040 | 271 |
| `full` | none | control | 154,990 | n/a |

The other 4,493 RL targets are Dan's ladder theorems, which have no reference proof.

**What "removed" means in the text the model writes.** The model writes Lean, and two of the four rules share their
surface form with a rule that stays:

- `IMPE` and `NEGE` are both written `nF nA` (apply one line to another). With `IMPE` gone the model has still seen
  application, only ever of a negation to its positive.
- `IMPI` and `NEGI` are both a `fun ( n : A ) => by ... ; exact n` box. With `IMPI` gone it has still seen boxes, for
  negations and case splits.
- `BOTE` is `False.elim n`. The `.elim` token never appears in the `exfalso` arm's pretraining.
- `DN` is `Classical.byContradiction ( fun hh => n hh )`. `Classical.byContradiction` and `hh` never appear in the
  `dn` arm's pretraining.

So `mp` and `cp` are pattern knockouts (every token seen, the use never seen), and `exfalso` and `dn` are token
knockouts. Expect the two kinds to behave differently, and read them separately.

**Ex falso has a substitute.** Reductio proves anything from a contradiction, and the `exfalso` arm keeps `DN`. If the
model never writes `BOTE` there, that may be because it never needs to.

## Setup

`../long_rl_20261014`'s harness, `pretrain.py`, tokenizer, Lean-only gate and data, all unchanged (harness
`18d5a3ab54ce`, pretrain `5a341bd06b2a`). `knockout.py` wraps the harness: it filters the pools and logs a rule census
after every oracle call.

- **Model and pretraining:** run 198's untied six blocks, 600 s on one A40, on Dan's cap-6 file with distractor
  premises (proofs of at most 6 body lines).
- **RL targets:** long_rl's pool as it stands, 18,439 theorems of 0 to 80 body lines. No one-step targets were added
  (Robbie, 2026-11-18).
- **Judge:** Lean only.

| recipe | what | RL budget (GPU clock) | seeds per arm |
| --- | --- | --- | --- |
| `ei` | long_rl's expert iteration, sampling at T 0.8 | 3 h | 3 (control: 2) |
| `ei-focus` | the same, on only the targets whose reference proof uses the removed rule | 2 h | 1 (not `dn`) |
| `grpo` | long_rl's v3 GRPO | 2 h | 1 |
| `ei-hot` | expert iteration sampling at T 1.0 | 2 h | not launched |

`ei` is the core: 4 arms x 3 seeds plus one control, 13 pods, launched 3:32 pm. `ei-focus` and `grpo` are 8 more pods
at seed 0, launched once the first pods are seen training.

## Readout

Per oracle call, in each run's `rounds.jsonl` (streamed, so a run that is cut off still has its curve):

1. **What the model did.** Lean-accepted attempts that use the removed rule, and distinct accepted proofs using it so
   far. This is the main measure: rules that appear in proofs the model wrote.
2. **What it solved.** Solved targets whose reference proof uses the removed rule. This is a proxy for "needs the
   rule": a reference proof using it does not show that every proof must.

Both against RL seconds, with the control's rate beside each arm. New accepted proofs are streamed to
`found_stream.jsonl`. The final dev eval is at 16 samples a theorem.

## Where this departs from what Robbie asked

- **Money.** The RunPod balance was $52.80 and an A40 is $0.59 an hour, about 89 pod-hours. Three seeds of every
  recipe on every arm does not fit. So the core recipe gets three seeds and the others one. A top-up buys the rest.
- **Three hours of RL, not four.** A 3 h run ends near 7:25 pm with its final eval; a 4 h run would still be in RL at
  the meeting and would cost a third more. The runs can be resumed (`harness.py --resume`).
- **Two control seeds, not three.** long_rl's own untied expert-iteration runs (2 seeds, same data) are a further
  reference, but on an older harness commit.
- **Dan's and Leon's recipes are not run as such.** This is Robbie's harness with expert iteration and its own GRPO.
  Their code lives in Dan's fork and on Leon's machines.

## What happened on launch day (2026-11-18, times PT)

- **3:32 pm:** the 13 core pods were rented. **3:36 pm:** the 8 secondary pods.
- **About 3:37 pm:** this Mac lost its network while most pods were still receiving their files. 17 launchers died,
  and their cleanup (terminating the pod) failed for the same reason, so 17 pods ran idle until 6:45 pm. Eight of them
  had no self-destruct timer yet. About $30 of the first $38 went to idle pods.
- **Survived the outage:** `cp-ei_s0` (its launcher reconnected) and the `ei-focus` runs of `mp`, `cp` and `exfalso`.
- **6:45 pm:** Robbie topped the balance up to $63.70 and the 17 idle pods were given jobs (`launch.py reuse`):
  every missing `ei` run, the four `grpo` runs, and a second control seed (`full-ei_s1`) on the pod that had been
  meant for `dn-ei-focus`. So `dn-ei-focus` was not run, and the control has two seeds.
- `launch.py` now retries the upload, the address lookup and the terminate call, and `status.py` pulls every live
  run's `rounds.jsonl` and `found_stream.jsonl` to `artifacts/rule_knockout_20261118/live/` every four minutes.

The relaunched runs started RL near 7:00 pm, so at the 8:15 pm meeting they have about an hour of RL each;
`cp-ei_s0` has its full three hours.

## Known limits

- The arms are not size-matched and nothing is refilled, so an arm's base is weaker for two reasons: the missing
  rule and less data (pretraining is time-budgeted, so smaller arms see more epochs of less).
- Expert iteration and GRPO only learn from accepted proofs. If the removed rule is never sampled, a flat line says
  the first success did not happen, not that RL cannot build the rule. The probability each base puts on the removed
  form is not measured during the run; it can be read afterwards from the `pretrain.pt` checkpoints.
- One to three seeds, three hours.

## Files

| file | what |
| --- | --- |
| `knockout.py` | pod side: the wrapper around long_rl's harness (pool filter, per-call census, streamed proofs) |
| `remote_ko.sh` | pod side: long_rl's `remote.sh` with `knockout.py` as the entry point |
| `launch.py` | one A40 per (arm, recipe, seed); output to `artifacts/rule_knockout_20261118/`, checkpoints to HF `runs/rule-knockout-20261118_<arm>-<recipe>_s<seed>` |
| `status.py` | pulls every live run's streamed files and prints one row per run |
| `summary.py` | one row per run at a matched RL time, the control's rate for each rule beside the knockouts |

Checked before launch: a CPU smoke run of `knockout.py` (`--smoke`, arm `mp`) loaded the pools, ran RL rounds with
real Lean and wrote `rounds.jsonl` and `found_stream.jsonl`.
