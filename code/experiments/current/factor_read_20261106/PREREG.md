---
written_on: 2026-11-06
written_by: agent:claude
---

# Pre-registration: factoring the change along the σ route, base to RL end

Frozen before any GPU read. Its sha256 goes into `log.md` first. The reference proofs below are already
frozen (CPU only); no checkpoint has been sampled or scored for this read. No training. No model and
no reference proof changes because of a result. The 106 are Dan's held-out set: read-only inputs.

## Question

On Dan's 106 strict σ held-out theorems, RL raises or fails to raise the solve rate. Between a base
checkpoint and its RL end, how much of the change is the model selecting a route it could already use, and
how much is it executing a link it could not? σ = a DN line citing a NEGI line. This read attributes the
change to groups along one route. It reports whatever the data shows.

**Limits, stated now.**
- It measures the σ route only. Other routes appear only as shares in measure A.
- Shares depend on where the boundaries are. Moving a boundary moves a share.
- Forcing a reference prefix can put the model in states it would not reach on its own.
- C scores one path. B covers every path from a boundary, but only the four boundaries.

## How the σ route looks in Dan's environment (read from `state_env.py`)

The policy sees a tactic state and writes one action. This shapes the groups:
- **The pick and the NEGI statement are one action:** `have x : ( ¬ ( ¬ T ) ) := ( fun ( y : ( ¬ T ) ) => by`.
  So the pick group also writes ¬¬T. There is no separate action that only assumes ¬T.
- **The NEGI link is `exact e`,** where `e` is the box's last line, which must be the ⊥ line. The environment
  renders the rest.
- **The DN link is** `have m : T := Classical.byContradiction ( fun hh => x hh ) ;`.
- **Types are not checked by the env.** A wrong term is only caught by Lean at the end, so an action the env
  accepts can still be false. The classifiers below use the verifier.
- **Names.** The env assigns the names a step introduces (`max in scope + 1`, from a base drawn per attempt,
  U[0, 32]), overriding what the model wrote. Citations stay as written.

**Two places where the brief's parentheticals do not fit this env.** I keep its definitions and
report the difference.
- *"setup: empty when T is the goal; exclude anything the env prefills".* The env prefills nothing: the model
  writes each premise line (`have n : F := h1 ;`). So setup is empty only when there are no premises and
  T is the goal (excluded_middle). Premise lines are reported as a separate part of setup.
- *"finish: empty when T is the goal".* The env needs a closing `exact m`, so finish is one action when T is the
  goal. It is reported as a group anyway, so the groups still add to the path total.

## Reference proofs (frozen)

`run_minlen.py` ran Dan's `minlen` at bound `n_lines + 2`, 60 s per theorem, 8 processes. `make_refs.py` then
accepted a proof only if it replays in his env, the verifier passes it, `coe_sigma.flags` flags the written
proof σ, and Lean accepts the env's text.

- **Result: 106 of 106 from minlen, 0 from the fallback, 0 dropped.** Every proof has `n_lines` lines (Dan's label).
- **File:** `/scratch/cowifornia/nd-rl/factor_read/refs/refs.jsonl`, sha256
  `7074e5ea166ceb1e9f0cb76db4fb290fc4a65a971cdef4aad91f25bf599d1c77`. Proof text stays there; `refs_table.json` in
  this folder holds counts only.

| schema | n | T = goal | premises-only setup | mean actions: setup / pick / reach / NEGI / DN / finish |
|---|---|---|---|---|
| excluded_middle | 37 | 37 | empty | 0 / 1 / 6.0 / 1 / 1 / 1 |
| peirce_sequent | 19 | 19 | 19 | 1 / 1 / 5.9 / 1 / 1 / 1 |
| contraposition_conv | 18 | 0 | 0 | 2 / 1 / 2.0 / 1 / 1 / 2 |
| negated_conditional | 15 | 15 | 15 | 1 / 1 / 8.0 / 1 / 1 / 1 |
| peirce | 12 | 0 | 0 | 1 / 1 / 6.0 / 1 / 1 / 2 |
| demorgan_nand_to_or | 4 | 4 | 4 | 1 / 1 / 9.25 / 1 / 1 / 1 |
| other (no schema label) | 1 | 0 | 0 | 3 / 1 / 2 / 1 / 1 / 2 |
| **all** | **106** | **75** | | |

**Rule for several σ DN lines (frozen).** T comes from the outermost σ DN line (minimum box depth), the last of those
in proof order. An inner reductio is then part of reach ⊥; an earlier one at the same depth is part of setup.
- **Affected: 15 of 106, exactly the negated_conditional schema.** Each has two σ DN lines. Its outer T is the goal,
  and the inner reductio on a subgoal sits inside reach ⊥.
- `analyze.py` splits the Δ of that inner reductio's three actions from the rest of reach ⊥ on those 15.
- No reference has an R copy or any line between the NEGI and the DN, so the DN group is exactly one action.

## Groups (frozen)

T = the formula the chosen σ DN line concludes. Along the reference proof, in actions:
1. **setup:** the actions before the pick. Premise lines count here, and are also split out.
2. **pick:** the action that opens the box on ¬T.
3. **reach ⊥:** the actions after it, through the one that creates the ⊥ line the NEGI closes on.
4. **NEGI:** the `exact` that closes the box.
5. **DN:** the action concluding T.
6. **finish:** the rest, to the end of the proof.

## Checkpoints (19, frozen, each confirmed to load)

Bases: `stage1_{ko,dose01,dose1}_s{0,1,2}.pt` (doses 0 %, 0.1 %, 1 %). RL ends (10); `ckpts.json` holds every md5:

| line | base | file | kind | it | x_dan at that it (latest read at or before it) | line's last it, x_dan |
|---|---|---|---|---|---|---|
| dko_s0 | ko_s0 | seg5_dko_s0_it500_rolling.pt | rolling | 500 | 3.97 (read at it 490) | 580, 5.07 |
| dko_s1 | ko_s1 | seg3_dko_s1_it630_alarm.pt | **alarm** | 630 | 5.25 (read at it 620) | 676, 5.93 |
| dko_s2 | ko_s2 | seg2_dko_s2_it400_rolling.pt | rolling | 400 | 5.94 | 400, 5.94 |
| dd01_s0 | dose01_s0 | seg2_dd01_s0_it695_rolling.pt | rolling | 695 | 4.42 (read at it 690) | 696, 4.49 |
| dd01_s1 | dose01_s1 | seg5_dd01_s1_it400_alarm.pt | **alarm** | 400 | 5.43 | 400, 5.43 |
| dd01_s2 | dose01_s2 | seg4_dd01_s2_it432_alarm.pt | **alarm** | 432 | 5.60 (read at it 430) | 432, 5.63 |
| dd1_s0 | dose1_s0 | seg1_dd1_s0_it60_rolling.pt | rolling | 60 | 0.41 | 64, 0.44 |
| dd1_s1 | dose1_s1 | seg1_dd1_s1_it60_rolling.pt | rolling | 60 | 0.45 | 63, 0.48 |
| dd1_s2 | dose1_s2 | seg2_dd1_s2_it90_rolling.pt | rolling | 90 | 0.69 | 94, 0.72 |
| dd01g0_s2 (the pilot) | dose01_s2 | seg1_dd01g0_s2_it290_rolling.pt | rolling | 290 | 2.94 | 293, 3.02 |

The pilot is one seed, reported beside the 0.1 % lines and never averaged into them. Alarm checkpoints are flagged in
every table. Some "RL ends" are earlier than where the line stopped, most of all dko_s0 (it 500 of 580) and dko_s1 (it 630 of 676).

## Measures per checkpoint, per theorem (frozen)

Sampling as `danread.py`: T 0.8, max_action 256, max_steps 48, Dan's env, Lean decides solves.

Noise seed `n = seed*1000 + it + 500`, as each line's own reads (the base is `it` 0).

**A. Unforced, k 64.** Seed `n`, batch 128 (the lines' reads used 128; a batch change is a re-draw). The first 32 are
Dan's measure. Per draw:
- solved (Lean);
- `coe_sigma.flags` on solved draws: sigma_strict, sigma_chain, dn_derived, dn_any, on the written and pruned proof;
- the furthest stage of its own reductio. A reductio is a NEGI box whose binder is ¬X, for any X:
  1 pick; 2 a valid ⊥ on the box's own level; 3 the NEGI line; 4 a DN line citing it, through R, concluding X.
  Each stage needs every earlier line to pass the verifier;
- the same stage restricted to X = the reference's T.

**B. Sampled from each boundary, reference prefix played through the env, k 64.** Batch 1024. Seeds `n + 10000`, `+20000`,
`+30000`, `+40000` for pick, reach, NEGI, DN. Steps already taken count toward max_steps.
- **pick:** prefix = the actions before the pick. Success: the next action opens a NEGI box declaring ¬¬T with binder ¬T,
  and the attempt so far passes the verifier. Reported beside it: `assume_any`, any box whose binder is ¬T (an implication
  box or an Or.elim branch also assumes ¬T, but NEGI cannot close it).
- **reach ⊥:** prefix through the pick, full continuation. Success: a ⊥ on the box's own level with every line through it
  passing the verifier. Also recorded: goes on to the NEGI closing that box, to a DN citing that line, and a Lean-judged solve;
  draws with a false ⊥ (declared False, verifier rejects); draws with a valid ⊥ only in a nested box.
- **NEGI:** prefix through the ⊥ line. Success: the next action closes that box as a NEGI line concluding ¬¬T, and it passes.
- **DN:** prefix through the NEGI line. Success: the next action is a DN line concluding T and citing that NEGI line, and it
  passes. Reported beside it: `r_copy` (an R copy of the NEGI line) and `dn_other` (DN to T citing another line).

Classification uses the parser and verifier, not string equality: other names, spellings and valid citations count.
`Trace.valid_through` runs Dan's `nd_verify.verify` on the lines so far, with open boxes closed by synthetic IMPI lines, and
the premises written so far checked against the theorem. Next-action stages stop generation after one action.

**C. Exact path probability.** For each reference action, log P(action tokens + `<eos>` | the state the env shows) under
softmax(logits / 0.8), deterministic, bf16 autocast as in the sampler. **Mean over all 33 name bases** (the env's U[0, 32]), so it
is the env's own distribution with no sampling. The introduced names are the canonical ones the env would assign. Summed
per group; the groups add exactly to the path total.

## Analysis (frozen; `analyze.py`)

- **Primary.** For each dose and seed, per theorem and group, Δlog10 = (log P at the RL end − log P at the base) / ln 10. Means per
  theorem, per line, by schema, and split by T = goal vs subgoal. Seeds listed individually: the seed is the unit of replication.
- **Shares.** Each group's mean Δ over the total's mean Δ (signed). Also each group's share of the positive part:
  max(Δ_g, 0) / Σ max(Δ, 0).
- **Sampled rates.** % of attempts with raw counts, pooled per line; Δlog10 with (x + 0.5) / (n + 1), as the mean per theorem and
  pooled; theorems at the floor (0 / 64) at base, at the end, and at both.
- **B vs C.** Listed where they disagree: opposite signs with both |Δ| ≥ 0.1, or apart by ≥ 1 in log10. C covers one path, B all paths.
- **Bottleneck.** For each base, per theorem, the non-empty group with the lowest C probability, and the distribution, overall and by
  schema. Beside it the lowest of the four B rates, since C's longer groups have more tokens and more equivalent paths.
- **Report-only reading.** Per line: pick ≥ 50 % of the positive Δlog10 reads as selection; DN or NEGI ≥ 50 % reads as a candidate gain
  on that link; reach ⊥ ≥ 50 % reads as general execution improving. Setup or finish ≥ 50 %, or no group at 50 %: no reading is set, and
  I say so. These are readings of a path measure, not findings, and nothing rests on one seed. The pilot is a single run.

## Required checks, before the full read (results in `log.md`)

1. An empty forced prefix at the same seed reproduces `env_generate`, byte for byte, on a small sample.
2. A forced prefix shows the state the env builds if the same actions are played through it: at every boundary of every theorem,
   against Dan's `decompose` (names shifted to the attempt's base) and against the scorer's rows; and on sampled, non-reference attempts.
3. On next-action steps with p ≥ about 0.05, the sampled frequency of the exact action at k ≥ 2000 agrees with exp(log p) from C
   within Monte Carlo error (|z| < 3). Also the scorer's full forward against the sampler's left-padded cached path.
4. The unforced read reproduces a known result: `stage1_dose01_s2`, seed 2500, batch 128, against
   `$KD/collected/ynez/dd01g0_s2/seg1/reads/it0000.json` (2 solved, with their proofs).
5. Unit tests of the classifiers, including wrong citation, wrong box depth and a false ⊥ (`test_factor_lib.py`).
6. **Knockout control.** 0 % bases were trained without the DN-of-NEGI composition, so the DN group at base should be lower at the 0 %
   dose than at the 1 % dose. **Passes iff** the mean over the 3 seeds of the mean log10 P(DN group) per theorem is lower at 0 % than at 1 %.
   Per-seed values are reported. If it fails: stop and message the coordinator, change nothing.

One checkpoint runs first (`stage1_dose01_s2`, which also serves check 4); the time is projected from it.

## Stops

- **Budget:** 2 GPU-h expected, hard stop 10,800 s of GPU time by the ledger, summed over every process. If the projection passes it I message the
  coordinator rather than cut the design.
- **Another user on ynez GPU 0,** or any failed check: stop and message.
- **Machines:** ynez GPU 0 only, claimed by `mkdir`, gated in code before launch and before each stage. No other GPU, no temescal, no pods.
