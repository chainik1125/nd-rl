---
written_on: 2026-11-18
written_by: agent:claude
---

# Rule knockouts: results

Remove one inference rule from pretraining, run RL, and count whether the rule appears in the proofs the model
writes. Design, arms and recipes: [`PLAN.md`](PLAN.md), which also has the launch-day log (a network outage cost three
hours). Human-readable summary with the figure:
[`experiment-summaries/2026-11-18-rule-knockouts/`](../../../../experiment-summaries/2026-11-18-rule-knockouts/README.md).
Branch `robbie-experiments`.

Every number below is from the runs' final `rounds.jsonl` and `summary.json`, by `summary.py` and the census
`knockout.py` writes. Lean is the only judge. "Uses the rule" is counted on Lean-accepted proofs only.

## Expert iteration, 3 h of RL (the core runs)

X is the arm's removed rule. "ref-X solved" is solved RL targets whose reference proof uses X. Dev is 843 theorems at
16 samples; old dev is the loops harness's 1,108 at 16.

| arm (rule removed) | seed | targets solved (of 18,439) | accepted attempts | of them using X | distinct proofs with X (targets) | ref-X solved | dev | old dev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `mp` (`IMPE`, modus ponens) | 0 | 1,655 | 120,187 | 0 | 0 | 0 of 13,774 | 0 | 237 |
| | 1 | 3,556 | 174,192 | 37,416 (21.5%) | 10,185 (2,024) | 1,558 of 13,774 | 211 | 358 |
| | 2 | 1,662 | 105,470 | 0 | 0 | 0 of 13,774 | 0 | 245 |
| `cp` (`IMPI`, conditional proof) | 0 | 2,107 | 11,088 | 0 | 0 | 0 of 4,224 | 163 | 101 |
| | 1 | 2,173 | 12,176 | 0 | 0 | 0 of 4,224 | 161 | 96 |
| | 2 | 2,196 | 12,952 | 0 | 0 | 0 of 4,224 | 160 | 94 |
| `exfalso` (`BOTE`, ex falso) | 0 | 7,091 | 192,488 | 0 | 0 | 311 of 4,469 | 240 | 934 |
| | 1 | 6,824 | 139,691 | 0 | 0 | 302 of 4,469 | 227 | 839 |
| | 2 | 7,362 | 148,142 | 0 | 0 | 459 of 4,469 | 249 | 901 |
| `dn` (`DN`, double-negation elimination) | 0 | 5,516 | 178,346 | 0 | 0 | 0 of 271 | 234 | 788 |
| | 1 | 6,670 | 174,435 | 0 | 0 | 0 of 271 | 239 | 789 |
| | 2 | 6,301 | 176,400 | 0 | 0 | 0 of 271 | 272 | 814 |
| `full` (control) | 0 | 7,229 | 219,629 | n/a | n/a | n/a | 247 | 959 |
| | 1 | 7,291 | 229,003 | n/a | n/a | n/a | 256 | 917 |

The controls' numbers for each rule, for comparison (seed 0 / seed 1):

| rule | accepted attempts using it | distinct proofs with it (targets) | ref-X solved |
| --- | --- | --- | --- |
| `IMPE` | 37.6% / 39.6% | 21,442 (4,203) / 18,605 (4,256) | 3,028 / 3,117 of 13,774 |
| `IMPI` | 77.6% / 75.9% | 25,471 (4,402) / 19,027 (4,335) | 553 / 577 of 4,224 |
| `BOTE` | 27.4% / 19.2% | 10,413 (1,722) / 5,740 (1,740) | 352 / 408 of 4,469 |
| `DN` | 9.0% / 8.1% | 5,021 (1,439) / 2,719 (1,093) | 9 / 21 of 271 |

## Observed

- **One of twelve knockout runs wrote its removed rule: `mp` seed 1.** Its first accepted proof with modus ponens
  came at 33.7 min of RL, on a 6-line ladder target with premises `Q > R` and `Q`. Distinct proofs with modus ponens
  then went 1 (36 min), 38 (49), 166 (62), 377 (76), 697 (90), 1,272 (104), 2,398 (118), 3,982 (133), 6,238 (148),
  9,218 (164), 10,185 at the end, on 2,024 targets. Solved targets whose reference proof uses it: 0 at 49 min, 42 at
  90, 385 at 118, 978 at 148, 1,558 at the end.
- **That run's dev eval is 211 of 843, against 0 and 0 for the other two `mp` seeds and 247 and 256 for the
  controls.** 834 of the 843 dev reference proofs use modus ponens.
- **The other two `mp` seeds never wrote it** in 3 h, with 120,187 and 105,470 accepted attempts. They stop near
  1,660 targets, all but two of them in the 0-6 and 7-9 line bands.
- **`cp` never wrote conditional proof**: three seeds, 3 h each, 36,216 accepted attempts in all. These runs still
  solve about 2,150 targets, 95% of their accepted attempts using modus ponens and 78% a case split, and none of the
  4,224 targets whose reference proof uses conditional proof.
- **`exfalso` never wrote ex falso and did not need to.** It solves as many targets as the control (6,824 to 7,362
  against 7,229 and 7,291), including 302 to 459 whose reference proof uses ex falso (control 352 and 408). Its
  accepted attempts use double-negation elimination in 21.8% of cases and negation introduction in 23.3%, against
  the control's 8.5% and 7.9%.
- **`dn` never wrote double-negation elimination** and solved none of the 271 targets whose reference proof uses it
  (control 9 and 21; the `exfalso` arm 28 to 80).
- **GRPO, 2 h, one seed per arm: zero uses in all four.** Targets solved: `mp` 1,446, `cp` 23, `exfalso` 3,651, `dn`
  1,828. The `mp`, `exfalso` and `dn` GRPO pods were gone before their final pull, so those three have `rounds.jsonl`
  to the last call but no `summary.json` and no dev eval.
- **Focused pool, 2 h, one seed (`mp`, `cp`, `exfalso`): zero accepted attempts of any kind.** Every target in those
  pools has a reference proof using the removed rule.

Share of accepted attempts using each rule over the whole run, expert iteration, pooled over seeds:

| arm | accepted | `IMPI` | `IMPE` | `BOTE` | `DN` | `NEGI` | `NEGE` | `ORE` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `full` | 448,632 | 76.7% | 38.6% | 23.2% | 8.5% | 7.9% | 31.1% | 19.3% |
| `mp` | 399,849 | 88.6% | 9.4% | 40.0% | 9.4% | 5.4% | 45.7% | 13.4% |
| `cp` | 36,216 | 0.0% | 95.1% | 13.1% | 0.1% | 2.7% | 15.9% | 78.3% |
| `exfalso` | 480,321 | 74.6% | 43.0% | 0.0% | 21.8% | 23.3% | 23.3% | 20.8% |
| `dn` | 529,181 | 76.2% | 39.1% | 29.6% | 0.0% | 6.6% | 37.3% | 19.1% |

## Interpretation (not shown directly)

- The removed rule appeared once, and then compounded: one sampled proof, expert iteration trains on it, and within
  two hours the run is solving the targets the other seeds cannot touch. That is Robbie's "it picks it up because it
  has to", on one seed of three. The other two seeds are the same base recipe and never drew the first proof, so
  whether a run gets there looks like a lottery on the first sample, not a property of the arm.
- Modus ponens is the knockout closest to something the model had already seen: in the Lean it writes, `IMPE` and
  `NEGE` are both "apply one line to another". `IMPI` also shares its form with a rule that stayed (`NEGI`, a `fun`
  box), yet never appeared. A box is a longer deviation than a two-token term, and conditional-proof goals need it at
  the root. Untested.
- Ex falso shows the other half: where a substitute exists there is no pressure, and the rule does not appear even
  though the base is otherwise as strong as the control. The model's use of reductio more than doubles instead.
- `dn` has both problems. Its tokens were never seen, and only 271 of 18,439 targets reward it.

## Limits

- One to three seeds, 3 h of RL. A run with zero uses at 3 h may ignite later; `mp` seed 1 did so at 34 min.
- The census sees accepted proofs only. A model that tried the removed rule and failed is counted as not trying.
  The probability each base puts on the removed form was not measured; the `pretrain.pt` checkpoints are on HF.
- "ref-X solved" is a proxy for needing X: a reference proof using a rule does not show that every proof must.
- The arms are not size-matched and nothing was refilled (`PLAN.md`).
- All runs but `cp-ei_s0` and the three focused-pool runs started on pods that had sat idle for three hours after
  the outage. The code and data they ran were the same.
- The final dev eval used 16 samples a theorem, not long_rl's 64, so it is not comparable with that experiment's
  dev numbers.

## Where things are

- **Per-run outputs:** `artifacts/rule_knockout_20261118/rule-knockout-20261118_<arm>-<recipe>_s<seed>/`
  (`rounds.jsonl`, `found_stream.jsonl`, `summary.json`, `eval_*.jsonl`, logs). Not committed.
- **Checkpoints:** HF `robbiethompson2018/nd-rl-checkpoints`, `runs/rule-knockout-20261118_<arm>-<recipe>_s<seed>/`.
- **Figure:** `figure.py` writes the summary's `figures/mp_ignition.png`.
- **Cost:** about $79 of RunPod A40 time ($0.59 an hour), about $30 of it on pods idling through the outage.
- **Related, from the parallel session:** `strategies/` here (research notes on what a basis of proof strategies
  could be), the write-up [A generator basis for proof strategies](https://claude.ai/artifact/LTdzaX9RjPZxGZsYeV4w5F)
  and the [live results page](https://claude.ai/artifact/GRRoyzSR77mVWUtQByUx5Y) built by `report.py`.
