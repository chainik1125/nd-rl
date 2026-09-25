---
written_on: 2026-09-25
---

# Design: self-written goal annotations (ab3/goals)

## What is tested

The hypothesis: the prover fails at long proofs because nothing in its own text records what it still has to
prove. The measured failure pattern (lengthexp report, corrected by the coordinator on 2026-09-25):

- **Attempts are too short.** The median attempt has 6-10 written lines at every test length. From L14 on, at
  most 0.3% of attempts reach `L_true` lines.
- **Most failures are ordinary invalid steps**, around lines 4-5. A faked conclusion, where the first rejected
  line states the theorem's conclusion, is only 4-12% of failures at most standard lengths and 1-6% at L15-18
  (`lengthexp/report/fake_finish.md`). The brief's "fakes the conclusion around line 4-5" overstated this.
- **On long tests, surviving attempts die near their arm's training maximum.** See the per-line hazard,
  `lengthexp/report/fig_hazard.png`.

The one change tested here: the model also writes short goal annotations while it writes a proof, so that what it
still owes is on the page when it decides whether to continue.

- The annotations are the model's own text. At test time nothing is given to the model except the usual prompt.
- The verifier never sees them: they are stripped before verification.
- The reward is unchanged: the stripped proof verifies or it does not.

Everything else follows the repr/ pipeline, so the variant compares directly with its abs baseline:

- the 039 pretraining recipe at 2,911 fixed steps, batch 256, seeds 0-2;
- arm-C RL (lengthexp/rl_C.py: targets of true length 6-11, fine-tune proofs of at most 11 lines) at
  DEFAULT_FLOPS, RL seeds 101-103 paired with pretraining seeds 0-2;
- the lengthexp evaluation at lengths 6-18 plus the matched set.

The variant pays for its extra tokens: the steps are the same in pretraining, and the FLOP budget is the same in
RL.

## 1. The annotation

### 1.1 Format

An annotation group is `GOAL <formula>`: one new token, then one formula in the usual fully parenthesised form.
Groups stand only at line boundaries: after the `;` of a line, before the next `N<i>` (or QED). The canonical
annotation writes at most one group per boundary and never one between the last line and QED. Example (L6
modus tollens, cap-6 style):

```
N1 ( P > Q ) : PR ; N2 ( ~ Q ) : PR ; N3 | P : AS ; GOAL F N4 | Q : IMPE N1 N3 ; N5 | F : NEGE N4 N2 ; N6 ( ~ P ) : NEGI N3 N5 ; QED
```

A formula is self-delimiting (parenthesised), so a group needs no end marker. It contains no `N<k>` tokens, so the
abs index shift and the line numbering never touch it, and it adds no `;`, so the line count is unchanged.

### 1.2 The agenda: what the proof still owes

The agenda is computed from a finished, verified proof. Each entry is an **obligation**: a formula that has to be
derived in some context. It becomes owed at a push position and is discharged at an achieve position.

| obligation | formula | pushed | achieved |
|---|---|---|---|
| root | the theorem's conclusion | after the premise block | the last line |
| box goal (one per AS line) | the formula of the box's last line on its own level: the consequent for IMPI, `F` for NEGI, the case target for ORE | at the box's AS line | that last level line |
| conjunct sub-goal (split) | each conjunct of an obligation that is achieved by ANDI and is **not already written** on a line citable when that obligation is pushed; recursive | with its parent | the first line on the parent's level that writes it |

- **Box goals.** The definition "last level line" is the same for all three discharging rules: IMPI's
  consequent, NEGI's `F` and ORE's target are each that line's formula. It also covers the rare box that no
  rule discharges (0 in the cap-6 data, 196 of 130,786 boxes in RL-found proofs); for such a box the goal is simply
  the box's last level line.
- **Splits.** A conjunction is split only when the proof proves it by ANDI. Only the conjuncts that are still owed
  are split off: a conjunct that is a premise, or that the box assumes, or that was derived earlier, is not owed.
  The split is recursive.

**What gets written.** At each boundary the candidate is **next**: the pending obligation that is achieved first,
ties going to the most recently pushed one. A group `GOAL <next>` is written whenever `next` is an obligation
different from the one written last. In practice that means four places:

- after an AS line: the new box's goal, or its first owed conjunct;
- after a line that writes one owed conjunct: the next conjunct;
- after a lemma box closes: the goal the proof returns to (possibly the conclusion again);
- after the premise block: the first owed conjunct, when the conclusion is split.

There are three exceptions:

1. **At a box end nothing is written.** When the last line was a box's last line, the next line must close the
   box, so no goal is needed. A chain of IMPI discharges therefore unwinds silently.
2. **Nothing is written when `next` is ready.** All of its conjuncts are written, so the next line must be the ANDI.
3. **The conclusion is not written at the start.** The prompt already states it. It is written only when the proof
   returns to it after a lemma.

Hand-checked cases are in `tests/test_goals.py` (IMPI chain, NEGI, ORE, conclusion split, a conjunct that is
already available, lemma box, box goal). One more, a lemma box:

```
THM ( ( P > P ) > Q ) SEQ Q PRF
N1 ( ( P > P ) > Q ) : PR ; N2 | P : AS ; GOAL P N3 | P : R N2 ; N4 ( P > P ) : IMPI N2 N3 ; GOAL Q N5 Q : IMPE N1 N4 ; QED
```

A long "joined" theorem `A & B`, the kind that makes up most of the L15-18 sets, reads:

```
(premises) GOAL A (A's lines) GOAL B (B's lines) ANDI
```

Within each part, every box opening states its goal. This is the written record of what is still owed that the
hypothesis says is missing.

### 1.3 Why this choice (token costs measured on the training data)

Measured by `analysis/structure_stats.py` (results in `analysis/structure_stats.json`). The table gives the extra
proof tokens relative to the plain proof.

| policy | cap-6 train (155k) | RL-found proofs (59,885 distinct, L_true 6-11) | constructed L15-18 proofs | groups per line (cap-6 / found) |
|---|---|---|---|---|
| **goals (chosen)** | **+12.7%** | **+24.6%** (L6 +20%, L11 +33%) | **+25-27%** | 0.14 / 0.26 |
| start: also write the conclusion after the premises | +38.4% | +44.0% | +32-35% | 0.42 / 0.38 |
| every: the current goal after every line | +42.6% | +52.3% | +41-46% (4 proofs exceed 1,024 tokens) | 0.47 / 0.48 |
| boxes: no conjunct splits | +11.9% | +25.2% | +31-34% | 0.12 / 0.25 |
| naive: after every AS line and every achievement, no suppression | +61.6% | +82.6% | +61-66% | 0.57 / 0.65 |

- **Why write only on a change.** RL runs under a fixed FLOP budget: about 74% fine-tuning and 26% sampling, both
  proportional to tokens. Every token of annotation costs rounds. The chosen policy still writes the complete
  history of the agenda, because every change of `next` that is not forced by the proof's own structure is
  written. It costs about half as many tokens as writing the goal after every line. A 4-layer model with RoPE
  can attend back to the last group; the AS line it depends on is at a similar distance anyway.
- **Why no copy of the conclusion at the start.** It repeats the prompt verbatim, which is no new record, and it
  alone costs 26 points of overhead on cap-6 data.
- **Why the conjunct split.** It is the one place where the agenda branches: after one conjunct the model still
  owes the other. The composition diagnosis found exactly this failure (joined `A & B` theorems solved at 0% from
  L10 up, although the parts are solvable). Most long test theorems are such conjunctions: 55%, 55%, 79% and 95%
  of the std theorems at L15, 16, 17 and 18 in lengthexp's final draw of 02:27 UTC (38% at L12, 49-55% of the
  matched set at every length); the rest are mostly implications from the chain families.
  Without the split, the annotation of such a proof would be empty (no boxes) or would only restate the
  conclusion.
- **Why only ANDI.** ANDI is the only non-box rule whose premises follow from the goal's shape and are
  independent obligations. IMPE, NEGE and the other eliminations have a premise that the goal does not
  determine: that is forward reasoning, not goal decomposition. ORI turns one obligation into another single one,
  so nothing new is owed alongside it.

### 1.4 Deterministic, derivable from any verified proof

`annotate(prompt, proof)` is a pure function of the verified proof text. The bulk tests (§4) cover every cap-6 and
RL-found proof and check three things:

- it gives the same output twice;
- it gives the same groups when the start index is shifted;
- it never fails on a verified proof. Undischarged boxes, a box cited twice (13 found proofs) and `A & A` are all
  handled.

The whole cap-6 pretraining file (154,990 records) and the held-out file are annotated this way.

### 1.5 What the model could know when it writes a group (lookahead)

Every annotated formula is one of the following:

- (a) the conclusion or a subformula of it: the root, or a conjunct of a split root;
- (b) `F`;
- (c) a formula implied by the goal pending when a box opens together with the box's assumption: goal
  `A > B` and assumption `A` give `B`, and an ORE target is the pending goal;
- (d) a lemma formula: the consequent of an IMPI or ORE box that does not discharge the pending goal, or a conjunct
  of such a goal.

(a)-(c) need nothing beyond the theorem statement and the boxes already open. (d) is lookahead: 3.3% of box goals
in cap-6 data, 2.1% in RL-found proofs. The positions of the groups also depend on the proof's plan:

- which conjunct is written first;
- that a box ends, which is the same information as the depth of the next line, which the model must choose anyway.

Why (d) is the model's own planning choice and not leaked information:

- **Nothing reaches the model at test time except the prompt.** The groups are sampled from the model like every
  other token, and the checker ignores them.
- **In the training data a group reorders information the proof already contains.** A box's goal is the formula
  its last line will write, and the lemma is chosen by the same plan that chose the AS formula before it (itself a
  "future-dependent" token). Writing `GOAL B` right after `| A : AS` states the plan before carrying it out.
- **No count and no length is written.** There is nothing about the remaining lines, `L_true` or the test set.
- **The only extra bit.** The absence of a group at a box end says "the next line closes this box", which the next
  line's depth token says anyway.

I do not see an ambiguity here that changes what is tested, so the design goes ahead.

## 2. Tokenizer

`lib/gtokenizer.py`, `GoalTokenizer`, checkpoint `tok_mode` `abs_goals`:

- **Vocabulary.** The abs vocabulary plus `GOAL`, 100 ids instead of 99. `GOAL` sits between the rules and
  `N1..N64`, so the line-index tokens stay the last 64 ids. Every `x >= tok.ref0` test keeps meaning "line index":
  frozen `shift_abs`, 039's GPU shift, the seed's capped `index_room` shift.
- **Encoding and decoding** are verbatim, as in abs mode.
- **Model size.** The embedding and the head gain one row (+512 parameters). One charged row costs +512 FLOPs, so
  6,342,656 instead of 6,342,144 (+0.008%). The meter charges this, so the variant pays for it too.

## 3. Stripping, verification, checker

- **`strip(body)`** removes every well-formed group at a line boundary: `GOAL`, then one complete formula, at the
  start of the body or right after `;`. Any number may follow each other there.
- **Malformed groups stay in.** A `GOAL` inside a line, or a `GOAL` whose formula does not parse, is left in the
  text, so the verifier rejects the proof like any other malformed output. The model gets no free pass on
  garbage, and malformed groups are counted separately.
- **Verdict.** `nd_verify.verify_text(prompt + ' ' + strip(sample))`, everywhere: RL judging, the harness's
  held-out check and the evaluator.
- **Line checker and first-error position.** `lengthexp/ndcheck.classify(prompt, strip(sample))`: the stripped
  proof has the same lines, so positions mean the same as in the baseline.
- **Tests** (`tests/test_goals.py`). On every sampled proof:
  - `strip(annotate(p)) == p` exactly, and the verdict is unchanged, for every policy;
  - random well-formed groups inserted at random boundaries into valid **and corrupted** proofs strip back
    exactly, so the verdict of the stripped text equals the plain verdict;
  - malformed groups make the proof fail;
  - the tokenizer round-trips the annotated text, and the index shift leaves groups unchanged.

## 4. Pretraining

- **Script.** `pretrain.py` is repr/pretrain.py (039's recipe line for line, fixed steps: 2,911 steps, batch 256,
  frac = step / steps). There are four edits:
  1. the data are the annotated cap-6 records;
  2. the tokenizer is `GoalTokenizer`;
  3. the abs index shift is also applied in mode `abs_goals`;
  4. the held-out checks strip annotations and use max_new 768 instead of 512. Annotated cap-6 proofs are up to
     670 tokens; plain ones are up to 401.
- **Tokens seen.** The step count and batch are equal, so the variant sees more tokens.

  | | plain | annotated |
  |---|---|---|
  | tokens per record | 109.1 | 118.2 (+8.4%) |
  | proof (loss) tokens per record | 72.1 | 81.3 (+12.7%) |
  | sequence tokens over 2,911 x 256 records | 8.13e7 | 8.81e7 |

  51.8% of the records carry at least one group (0.73 per record). The longest sequence is 706 tokens (plain: 516).
- **Losses reported.** `val_loss` over all proof tokens (not comparable with abs), plus `val_loss_lines`: only the
  tokens of proof lines and QED. `val_loss_lines` is closer to comparable, though it is conditioned on the groups.

## 5. RL (arm C)

`rl_goals.py` is lengthexp/rl_C.py (which repr/rl_repr.py equals for abs), with these edits only:

- **Judging.** Each sample is stripped, then verified. The found-proof key is `norm(strip(sample))`, so a proof
  counts once whatever its annotations, as in the baseline. `written`, `pruned` and `max_proof_lines_train` use
  the stripped proof.
- **Training text.** The fine-tune record is **the model's own sampled text, annotations as written**. They are
  not checked or corrected: the reward is still only verification. The cap-6 retention records are the
  canonically annotated pretraining records. Relabelled by-products (hindsight: same premises, other conclusion)
  get the canonical annotation of the stripped proof, because the model's own groups refer to the old conclusion.
  Correction (2026-09-25, from the running jobs): on the 039 base there are about 100-300 of them per run, not
  11-18. base039's seed run had 124 by round 19, and repr's abs_s0 113. Each is repeated twice in the mix, so they
  are about 2-4% of a round's RL records, next to 20,000 canonically annotated retention records per round.
- **Consistency log** (per round, in metrics.jsonl and rl_stats):
  - for verified samples: the share whose annotation equals the canonical annotation of their own stripped proof
    exactly;
  - the share of canonical groups written with the exact goal at the canonical boundary;
  - extra groups;
  - groups naming a goal that is on the pending agenda at that boundary;
  - for all samples: groups per sample and malformed groups.
- **max_new 768 instead of 512.** Plain found proofs reach 440 tokens (86% of 512). Annotated ones reach 677
  tokens at L12 (88% of 768), with p99 532 at L11. At 512, 1-2% of valid long annotated proofs would be cut: a
  handicap that has nothing to do with the hypothesis. Runaway samples cost little: the QED rate is at least 99.4%
  in arm C. The same applies to the harness's final held-out greedy check (768 instead of 512).
- **Budget.** The same DEFAULT_FLOPS (3.9e15), metered by the unchanged frozen meter. Annotated samples and
  packed fine-tune records are longer, so the in-run planner fits fewer rounds. The report states the rounds,
  samples and optimizer steps next to the baseline's. Packing is unchanged: records ≤ 1,024 tokens, whole.
- **Harness.** A copy of ab3/frozen, the repr/make_frozen.py pattern. Its `harness.py` has three edits: the
  tokenizer from the checkpoint's tok_mode (`abs_goals`), the held-out check strips annotations (max_new 768),
  and `train_cap6` points at the annotated file.

## 6. Evaluation

`evaluate.py` is lengthexp/evaluate.py (repr's evaluate.py equals it for abs):

- the same test sets, k = 32, T = 0.8 and max_new 1,024;
- the same sampler (lengthexp/sampler.py) and seeds;
- the same checker.

The edits:

- the model is loaded with `GoalTokenizer`;
- every sample is stripped before `classify`;
- `samples/*.jsonl.gz` store the stripped proof in `body` and the raw text in `raw`, so any analysis written for
  repr/ reads plain proofs;
- each row also gets `n_lines` (written lines of every sample) and annotation statistics: groups, malformed
  groups, and consistency for correct samples.

**Metrics.** The same as the repr/ analysis:

- pass@1 and pass@4 per length;
- the log-odds drop per extra line past 11;
- the share of attempts at least `L_true` long;
- the first-error position;
- in-distribution checks: held-out greedy after pretraining and after RL, and pass@1 at L6-11;
- plus the annotation consistency at each length.

**18-line fit.** The generator-constructed proofs of the long test sets are used for this token count only:

- The longest prompt plus annotated proof is 843 tokens (L16). At L15-18 the 99th percentile is 579-641. None
  exceeds 1,024. "every" would put 4 over.
- max_new 1,024 leaves room for about 40 annotated lines at the observed rate. The sequence then runs to about
  1,100 positions for a runaway, as for the baseline.

## 7. Compute and GPU estimate

- **Pretraining.** 3 x ~12-13 min (2,911 steps; 039's batches are padded to their longest record, which annotation
  lengthens). About 0.65 GPU-h.
- **RL.** 3 x ~33 min (the same FLOPs as arm C, which took 26-36 min). About 1.7 GPU-h.
- **Evaluation.** 3 RL checkpoints x ~16 min (+15-25% sampled tokens over arm C's ~14 min), plus pretrain s0 as
  an anchor. About 1.1 GPU-h.
- **Smoke runs and checks.** About 0.3 GPU-h.

**Total about 3.7 GPU-h**, in jobs of at most ~35 min, all through the gpulock FIFO queue (`--gpu auto`). I hold at
most 2 lock slots at a time.

## 8. Known confounds

- **Token position.** Annotated proofs sit at later token positions: an 18-line proof reaches about 25% further.
  The length-versus-position confound of lengthexp is therefore slightly worse for this variant.
- **Fewer RL rounds.** The variant gets fewer RL rounds for the same FLOPs.
- **Canonical versus raw annotations.** The canonical annotation is taught in pretraining. In RL the model is free
  to drift from it, and the consistency log shows whether it does.
- **Hindsight records.** Relabelled records are the one place where the RL-derived training text carries
  canonical annotations: about 2-4% of RL records (see §5), beside the canonically annotated retention records.
