# Independent verification of exp-00c run_lr3e-5 (2026-09-27)

Verdict: **no reward bug, verifier loophole, or train/eval leak found.** The claimed numbers reproduce exactly. The gain looks real: the trained model mostly fixes one step that the frozen model systematically skipped. One caveat is low diversity: 54 of the 61 newly solved problems have exactly one distinct accepted proof.

Scripts: `verify.py` (all checks; full output in `verify_out.txt` and `verify_out.json`) and `peek_nopremise.py`. Run from `code/` with `uv run python`. They do no model sampling. They import only `strict_verify.safe_verify`, `nd_data.canonical_theorem`, and the fork's `nd_verify` parser, which is used for prefix and renumbering checks.

## 1. Re-verification
- Frozen dump: 6400 rows, **0 disagreements** with `ok`. There are 200 prompts with 32 samples each, all at T=1.0. L7: avg 0.1225, 39/160 solved. L8: 0/40.
- Trained dump: 6400 rows, **0 disagreements**. L7: avg 0.4299, 97/160 solved. L8: avg 0.0047, 3/40 solved.
- Training rollouts (`completions.jsonl`): 12288 rows, **0** reward/verdict disagreements. 10337 have reward 1 and 1951 have reward 0. The high acceptance rate is expected, because the pool was filtered for prompts the frozen model can solve.

## 2. Triviality and loophole scan (2207 trained and 627 frozen accepted proofs)
- None have the conclusion among the premises, F among the premises, the final line differing textually from the conclusion, the final line inside a box, or text after QED. The same holds for the eval_set prompts themselves.
- **0 accepted proofs are shorter than L_true.** Trained line counts by L_true: 7→7: 2187, 7→8: 14, 8→8: 6. Frozen: 7→7: 625, 7→8: 2.
- 34 trained proofs cite no premise. They cover 2 prompts whose conclusions are tautologies, so the proofs are valid. There are 25 "repeated identical line" cases. These are the same AS line reopened in a separate box, which is valid.
- Rule histogram as fractions of lines, frozen → trained: AS .254→.330, IMPI .249→.322, PR .257→.137, ANDI .108→.058, ORI1 .050→.033, DN .006→.021, BOTE .014→.016, NEGE .014→.018, NEGI 0→.002, others small. There is no exotic rule and no BOTE or R spam. The PR share falls because the model solved more zero- and one-premise implication-chain problems.

## 3. Leaks
- (a) Exact prompt overlap between eval_set and each of the following is **0**: prompts.jsonl, pool.jsonl, the train split (2223), and the completions prompts. prompts.jsonl is identical to pool.jsonl, and the completions prompts are a subset of it.
- (b) `canonical_theorem` overlap (renaming- and premise-order-invariant) is **0** between eval_set and each of: train split, pool, completions prompts. It is also **0** between the whole eval half and the whole train half, and between the 63 length-6 pool prompts and the whole eval half. The 72 textbook problems overlap pool, completions, the train split, and eval_set 0 times each.
- (c) Eval prompts whose renamed conclusion matches the conclusion of a train-split prompt: 18/200. Against the pool, which is what was actually trained on: 6/200. The premises and schemas differ; for example, eval `(~S)v(Q&R), ~~S ⊢ Q&R` vs pool `~(RvP), (~(RvP))>R, P ⊢ R&P`. I judge this to be generator similarity on short conclusions, not a leak. Those 6 problems also do not drive the gain: on them L7 goes from 0.052 to 0.167, and on the other 154 from 0.125 to 0.440.
- (d) All 200 eval_set prompts are in `split/rl_targets_eval.jsonl.gz`. The two split files together have 4495 records, matching the raw fork `rl_targets.jsonl`.

## 4. Newly solved problems
61 problems went from 0/32 to at least 1/32 (58 at L7, 3 at L8). No problem went from solved to unsolved. For each of the 8 printed examples (see `verify_out.txt`), the trained proof is a genuine, minimal Fitch proof. The frozen model's closest attempt usually has the correct overall shape but is missing one intermediate step. Examples:
- It skips the ANDI that builds `((~Q)&S)&S` before IMPE.
- It skips the IMPE that yields `~P` before NEGE.
- It cites the wrong box end in IMPI.

The trained model inserts the missing step, and those proofs are correct.

## 5. Diversity
Distinct accepted proofs per newly solved problem, modulo line numbering: 1 distinct proof for 54 problems, 2 for 6, 3 for 1. Across all 1239 accepted proofs on these problems there are 22 distinct (rule, depth) skeletons. The most common are:
- AS/AS/AS/X/IMPI/IMPI/IMPI curried-implication shapes, with X being DN, ORI1, ANDE2, or ANDI.
- PR/AS/AS/NEGE/BOTE/IMPI/IMPI.

These are several templates, not one. Within a problem, sampling has collapsed onto a single proof. That is typical of GRPO, not a bug.

## 6. Other
- The start index is spread widely for both models; the most common values are 2, 19, 40, 27, 3, 15, 20, 43. Renumbering every parseable trained completion to start at N1 changes **0/6242** verdicts. The verifier checks that indices are consecutive and range-checks refs against the actual starting index.
- In every dump row, the `length` field equals the eval_set L_true.
- My own "valid depth-0 prefix" measure is stricter than the dump's `n_valid_prefix`, because it can only end at depth 0. That is why the two numbers differ in the section 4 examples. This is not a problem.
