# Independent verification of exp-00c run_ei (expert iteration), 2026-09-27

Verdict: **no reward bug, verifier loophole, or train/eval leak found.** The claimed numbers reproduce exactly. The length-8 jump is real, but it is narrow. Almost all of it comes from one family of problems: three nested implication introductions around a single trivial inner step. The model builds these by combining two proof shapes that are both in its SFT data. None of the eval proofs appears verbatim in the SFT data. There is one minor data-pipeline note: the "distinct proofs" dedup is applied to raw text, so it does not remove duplicates that differ only in line numbering.

Scripts:
- `verify.py`: checks 1-6. It is `verify_lr3e-5/verify.py` with the run directory made a parameter. Output is in `verify_out.txt` and `verify_out.json`.
- `verify_extra.py`: checks 7-9. Output is in `verify_extra_out.txt`.
- `sft_skeletons.py`: counts proof shapes and box depths in the SFT data.

## 1. Re-verification
- The baseline dump is byte-identical to the run_lr3e-5 baseline.
- Frozen dump: 0 disagreements out of 6400 rows.
- EI dump: **0 disagreements** out of 6400 rows. There are 200 prompts × 32 samples, all at T=1.0.
  - L7: avg 0.5449, solved 118/160.
  - L8: avg 0.2656, solved 19/40.
- EI sampling log (`completions.jsonl`): 0 disagreements between recorded reward and re-verification over 12474 rows (9819 accepted).

## 2. Triviality and loophole scan (3130 accepted EI proofs)
- **0 proofs shorter than L_true.**
- Accepted proof lengths by L_true:
  - L_true 7: 2520 at 7 lines, 269 at 8, 1 at 9.
  - L_true 8: 332 at 8 lines, 8 at 9.
- Zero cases of: conclusion among the premises, F among the premises, final formula that does not match the conclusion, final line inside a box, or text after QED.
- 68 proofs use no premise. As in the lr3e-5 run, these are problems whose conclusion is a tautology, so the proofs are valid.
- Rule histogram: AS 0.34, IMPI 0.34, PR 0.13, ANDI 0.06, ORI1 0.03, DN 0.02, ORI2 0.02, others small. It looks like the GRPO model's histogram. No exotic rules.

## 3. Leaks
- (a) Exact overlap is 0 between eval_set and each of: prompts.jsonl, pool.jsonl, the train split, and the completions prompts.
- (b) Canonical overlap (renaming- and premise-order-invariant) is 0 between eval_set and each of: the train split, pool, and completions. It is also 0 between the two halves of the split. The textbook72 set overlaps pool, completions, the train split and eval_set 0 times each.
- (c) Eval prompts whose conclusion matches a pool conclusion after atom renaming: 6/200. These problems improve from 0.052 to 0.214; the other 154 improve from 0.125 to 0.558. This is generator similarity, not a leak.
- (d) All 200 eval prompts are in the eval split file.

## 4. Newly solved problems
- 98 newly solved: 79 at L7 and 19 at L8. 0 problems were lost.
- The 8 printed examples are all genuine minimal proofs. In each, the frozen model's closest attempt skips one step. Typical misses:
  - an ANDI or IMPE before the step that needs it;
  - an ORI2 inside the innermost box.

## 5. Diversity
- Distinct accepted proofs per newly solved problem: 1 distinct proof for 79 problems, 2 for 15, 3 for 3, 4 for 1.
- There are 45 distinct rule/depth skeletons across 2197 proofs.

## 6. Other
- Renumbering proofs to start at N1 changes 0 of 6207 verdicts.
- The start index varies widely across samples.
- Every dump `length` field equals the eval set's L_true.

## 7. SFT data audit
Across the 3 rounds there are 696, 781 and 786 records.

| Property | Result |
|---|---|
| Prompts not in pool.jsonl | 0 |
| Prompts exactly in eval_set or the eval split | 0 |
| Prompts matching eval_set or the eval split canonically | 0 |
| Completions that fail re-verification against their own prompt | 0 |
| Records per prompt | at most 4 |

- **Dedup note:** 459/696, 520/781 and 528/786 records duplicate another proof for the same prompt once line numbers are normalised. The dedup ran on raw text, so the same proof written from N10 and from N59 was kept twice. This is not a correctness bug. It does mean some proofs are up-weighted, and "distinct" in the method description is weaker than stated.
- Only 297 distinct proofs remain after normalising. By line count: 20 have 8 lines, 181 have 7, 82 have 6.
- SFT records by pool length per round:
  - L6: about 250.
  - L7: 439, 514, 518.
  - L8: 6, 15, 16. These come from only 6 distinct proofs of the 4 length-8 pool prompts.

## 8. Length 8 in detail
- All 19 newly solved L8 problems are listed with one proof each in `verify_extra_out.txt`.
- Line counts: 18 of the listed proofs have 8 lines and 1 has 9. **0 are shorter than L_true.** Each is a valid, natural proof.
- 340 accepted L8 proofs share 10 skeletons. 305 of them (90%) follow one pattern:
  - `PR AS| AS|| AS||| X||| IMPI|| IMPI| IMPI`, with X one of ORI2, ANDI, ANDE1, ANDE2 or ORI1 citing the premise.
  - In words: open three nested assumptions, derive the innermost consequent from the premise in one step, then discharge three times.
- None of the 4 length-8 pool prompts has this shape. Their SFT proofs are shapes like `PR PR AS AS IMPI IMPI NEGE BOTE`.
- The SFT data does contain both halves of the pattern, so the model is combining two shapes it trained on:
  - The two-deep version with a premise-sourced inner step, `PR PR AS| AS|| X|| IMPI| IMPI`, is the most common SFT family.
  - Nine SFT proofs have box depth 3, all of the zero-premise form `AS| AS|| AS||| X||| IMPI|| IMPI| IMPI`.
  - Adding the missing step to a 7-line shape fits: 27 L8 proofs are exactly one line longer than an SFT shape.
- 0 of the 3130 accepted EI eval proofs appear verbatim in the SFT data, even after atom renaming and renumbering.
- Box depth 3 or more:

  | Model | Accepted samples | All parseable samples |
  |---|---|---|
  | Frozen | 0 | 1642 |
  | EI | 1753 | 2531 |

  So the frozen model already attempted deep nesting and never got it right. SFT on correct depth-3 and depth-2 proofs appears to fix that.
- None of the L8 proofs looks suspicious as a verifier artefact.

## 9. EI compared with GRPO lr3e-5

| Length | EI solved | GRPO solved | Both | EI only | GRPO only |
|---|---|---|---|---|---|
| L7 | 118 | 97 | 96 | 22 | 1 |
| L8 | 19 | 3 | 2 | 17 | 1 |

- EI solves every problem the frozen model solved.
- Distinct accepted proofs per solved problem:
  - EI: 1 for 102 problems, 2 for 22, 3 for 11, 4 for 2 (mean 1.36).
  - GRPO: 1 for 76 problems, 2 for 18, 3 for 6 (mean 1.30).
- Both models have collapsed onto one proof per problem to a similar degree.
