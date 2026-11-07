---
written_on: 2026-11-07
written_by: agent:claude
---

# olmo-iffu: on olmo-ifomega's 52 reach prompts (perp_069 only under the doubly-credited reading), the RL run's DPO start `Olmo-3-32B-Think-DPO` becomes able (≥ 2 genuine of 64) on 16 of the 48 instruction-following ones it was tested on when told "Answer in at most 60 words." (4 of 6 IFEval, 12 of 42 IFBench; 15 under the strict title-case reading); all 10 IFBench reach prompts sampled to 1,024 survive with 0 genuine, yet 4 of them are prompt-elicitable; a response budget filling the model's 65,536-token context explains 1 of the 3 geometry prompts (perp_068); `step_0750` is confirmed on 26 of the 43 IFBench reach prompts within 32 draws

Status: completed 2026-11-07 (DONE 09:11:45 UTC) and reviewed 2026-11-07 (final commit 11:29 UTC). **No
hard-constraint violation, nothing quarantined.** These are follow-up measurements: olmo-ifomega's pre-registered
verdicts are unchanged. The review's verdict is **"Stands"**, with two statements not supported (perp_069's "token
budget" and its consequence, "18 are now explained") and six items to reword.

In phase 1 (blind, committed 10:44 UTC), before opening the executor's write-ups, results, status or hand-check
files, the reviewer did the following with their own code (kit `review/olmo-iffu-recount/`, which imports no executor
module):
- re-resolved the three models' Hub revisions, which match the pod logs;
- reproduced both seeded subsets from their seeds and checked that every job sampled exactly its design set;
- checked the hint wording on every hinted row, the prompts and kwargs against their own upstream IFBench / IFEval
  fetch, and the chat template's hash. With each model's tokenizer they re-counted the prompt tokens of all 153
  (job, prompt) pairs (the hint adds 8–9, so it was in the user turn);
- checked the token budgets on all 12,816 rows (65,536 − prompt tokens in test 1, 32,768 elsewhere; `finish ==
  "length"` iff the budget was used up), the engine seeds, the tensor parallelism and the stopping rules;
- computed the three geometry labels two ways (perp_055 = 15, stored 17; 068 = 23; 069 = 14);
- checked that all 17 bulk files match their `MANIFEST.jsonl` sha256 and are in the bucket, that every text equals the
  decode of its token ids (12,816 / 12,816), and scanned all 524 files of the bucket folder for secrets (none);
- re-graded all 12,816 stored texts with the upstream checkers (byte-identical to the vendored copies), test 1 also at
  a 32,768-token truncation, and re-graded olmo-ifomega's raw samples for the before-counts;
- repeated the hand gate blind with their own readers (blinded general-purpose sub-agents; random ids, shuffled across
  arms and models), with second readers on the deciding items (a gap in that set, found in phase 2, was then filled);
- recomputed every outcome by the pre-registered rules, the Clopper–Pearson bounds, the per-job compute and the
  registry rows.

*[librarian]* The review states no limit on its blindness. Two phase-1 sources that it did not exclude named outcomes:
the pre-registration's test-2c addition (a listed input) quotes test 2's "2 of 10 prompt-elicitable", and the run's
commit subjects name others (e.g. `39aaa4a9`, "T3 done: 0 strict in 768 more draws on all 10 prompts").

Phase 2 compared the executor's files with the recount (all 153 rows of `results.csv` equal the re-grade; the 368 hand
items joined by draw key), finished the reviewer's own hand gate (first and second reads disagree on 20 items, all
decided in `hand/confirm.json`: 15 by a rule written before deciding, and 5 from a second-reader gap by the majority of
three readers), and an independent sub-agent's adversarial check of the review found 23
points, all addressed (`out/adversarial_check.md`).

- **What stands (the review's verdict).** The models are `dpo` = `allenai/Olmo-3-32B-Think-DPO` (RL's start), `s2300`
  = `allenai/Olmo-3.1-32B-Think` `main` (= `step_2300`) and `s0750` = its `step_0750`; 32 B, pretrained,
  natural-language chat. What stands:
  - all the sampling, grading, provenance, compute and spend numbers, exactly: every `results.csv` grade count, the
    subsets from their seeds, the hint wording and the 65,536 − prompt budget, registry rows equal to the reviewer's
    per-job sums, $53.87, and the bucket upload with matching sha256 and no secrets;
  - **test 3:** all 10 IFBench prompts survive 1,024 `dpo` draws (≤ 2.92 × 10⁻³ per sample), and 4 of them (78, 133,
    252, 253) are nonetheless prompt-elicitable with the hint. In the review's words, "This is the run's cleanest
    result and it reproduces fully";
  - **test 2 / 2c, `dpo` side:** 4 of 6 IFEval and 12 of 42 IFBench reach prompts are prompt-elicitable (11 under the
    strict title-case reading);
  - **test 4:** `s0750` is confirmed on 26 of 43 within 32 fresh draws (64 pre-registered);
  - **test 1:** every sound credit lies past token 32,768, and perp_068 is "token budget".
- **Not supported.**
  - **"Token budget" on perp_069** (run file, its status line, the figure, the plain summary). perp_069 was reach only
    under `doubly`, and under `doubly` this run gives it one sound success at 65k. Under `any` it was never reach:
    olmo-ifomega's `dpo` already had 2 sound responses in 256. The status count becomes token budget 1, and "one
    success with room or hint" 3 (055, 069, 204).
  - **Its consequence, "Of the 52 reach prompts, 18 are now explained"**, which becomes 17.
- **Must be reworded** (the review's six items):
  1. the geometry result: the plain summary's "solved two puzzles … at least twice … mostly a length limit" becomes
     one of three (068); its "the RL model solves two of these puzzles about 70 % of the time" is `omega` checker
     credit, not hand-sound (on 069 the predecessor review read 6 of 9 such credits sound); the figure and the run
     file mix readings on 069 (a doubly "before" cell beside an any-credit test-1 cell);
  2. "the hint breaks `s2300` on 8 more" becomes 6, or 7 if `ifbench_170` counts as broken. `ifbench_284`'s four exact
     span copies are genuine by rule 5, which names 284; 170 is a reader call. The status line becomes uninformative 4
     (5 with 170) and not elicited with the hint fine 21 (20);
  3. "`s2300`'s strict rate with the hint is at or above its unhinted rate on all 16" is wrong: it is lower on 2 of 16
     against the fixed phase and on 6 of 16 against all 24 draws. None of the drops is significant individually
     (largest: `ifbench_179`, 21 / 24 → 10 / 16, one-sided Fisher p = 0.07); the genuine-rate point stands;
  4. report the missed expectation "gamed share of `dpo`'s strict successes 30 % (10–60 %)": observed 1 of 16 read
     (6 %);
  5. `ifbench_226`: the `s2300` check was left at 5 of 16 against the pre-registered "until 2 genuine or none left".
     Read to the end, the hint breaks `s2300` there by the majority of three readers on each deciding item (1 genuine
     of 16), though not under the first readers alone. `dpo`'s verdict is unaffected;
  6. minor: mark the 28 / 40 as olmo-ifomega's in the run file; "65,290 response tokens" is 65,317 for 069; three
     stated times in the pre-registration commit postdate it; deviations 1, 2 and 4 were adopted after their tests'
     sampling began (disclosed); the log's "`dpo`'s median answer falls from 64–600 words … to 50–95" leaves out
     `ifbench_64` (6 → 12); test 4's 26 of 43 is at a 32-draw cap, so the pre-registered ≤ 64 quantity is unmeasured.

  In the review's words: "No hard constraint is violated. Apart from the two items above, I found nothing in run.md,
  numbers.md or the plain summary unsupported."

Review: [`review_olmo-iffu.md`](https://github.com/dan-pandori/nd-takehome/blob/c0c6e053/review_olmo-iffu.md) (fork
`dan-pandori/nd-takehome`, branch `dan_olmo-iffu`; §Recount, phase 1, blind,
[`a068c686`](https://github.com/dan-pandori/nd-takehome/commit/a068c686), 10:44 UTC; §Compare, phase-2 corrections
and §Verdict [`65aee6a7`](https://github.com/dan-pandori/nd-takehome/commit/65aee6a7), 11:01 UTC; adversarial-check
fixes, a third read on `ifbench_226` and the final verdict
[`c0c6e053`](https://github.com/dan-pandori/nd-takehome/commit/c0c6e053), 11:29 UTC; kit
`review/olmo-iffu-recount/`, whose `README.md` lists every script and output). Run files at the executor's DONE commit
[`e57060d0`](https://github.com/dan-pandori/nd-takehome/commit/e57060d0):
- `run_olmo_iffu.md`, `numbers.md` § olmo-iffu and `log.md` § olmo-iffu;
- `olmo_iffu/`: the code, `manifest.csv` (which prompts, their seeds), `results.csv` (one row per job × prompt),
  `status.csv`, `handcheck.csv` (368 items), `lengths_hint.csv`, `summary_tables.md`, `README.md` and
  `plain_summary.md`. The figure is `figures/olmo_iffu_status.png`;
- the pre-registration `preregistration/olmo-iffu.md`: design and binding expectations
  [`048ef5fe`](https://github.com/dan-pandori/nd-takehome/commit/048ef5fe) (11-06 23:22:37 UTC), the test-2c addition
  [`a9b234f1`](https://github.com/dan-pandori/nd-takehome/commit/a9b234f1) (11-07 02:01:45), and deviations 1–4.

Brief: [`BRIEF_olmo-iffu.md`](../../docs/proposals/followups/BRIEF_olmo-iffu.md);
[proposal 37](../../docs/proposals/2026-11-06-olmo-iffu.md). The run it follows up, on the same models:
[olmo-ifomega](../2026-11-04-olmo-ifomega-reach-on-ifbench-49-strict-43-hand-and-ifeval-6-omega-test-out-inconclusive/README.md)
(its memo's three cheapest tests are this run's tests 1–3, with its review's specification of the first).

Copies here:
- **the plain-language summary** (the brief's ≤ 500-word deliverable), with the review's corrections marked
  *[review]*: [`plain_summary.md`](plain_summary.md);
- **the status figure**, unchanged; its corrections are listed under [Figures](#figures):
  [`olmo_iffu_status.png`](olmo_iffu_status.png).

Question (Dan, 2026-11-06, in the brief): olmo-ifomega found Olmo 3.1 Think 32B's RL (`step_2300`) confirmed where its
DPO start had no success in 256 samples: IFBench (49 prompts strict, 43 by the review's hand check), IFEval (6) and
OMEGA `test_out`, only on perpendicular-diagonal geometry problems where the DPO model is cut at the 32,768-token cap
in 99 % of its failures. Three alternatives to new ability remained: the DPO model needs more room; RL's IF wins are a
brevity habit (its answers are 5–15× shorter on the IFEval reach prompts); 256 samples are too few. Under the
project's capability standard ([REPORT.md](../../docs/project_strategy/capabilities/REPORT.md)), "no success in 256"
means "not reached at 256", not "created". This run tests each alternative on a fixed prompt set.

## Configuration

**Models** (every number here is on one of these; the review re-resolved all three on the Hub):

| label | checkpoint | what it is |
|---|---|---|
| `dpo` | `allenai/Olmo-3-32B-Think-DPO` `main` = `97604023` | 32 B, bf16, pretrained (Olmo 3 32B base → Think SFT → DPO); RL's start |
| `s2300` | `allenai/Olmo-3.1-32B-Think` `main` = `97ade57a` | `dpo` after 2,300 RL steps on Dolci-Think-RL (= `step_2300`) |
| `s0750` | `allenai/Olmo-3.1-32B-Think` `step_0750` = `418fb274` | the same RL run at step 750 (test 4 only) |

**Decoding** as olmo-ifomega: `olmo_thinker` template (sha256 `eba6e269…`), T 1.0, top-p 1.0, no stop strings, vLLM
0.11.0, bf16 weights and KV cache, prefix caching, `generation_config="vllm"`, `max_num_seqs` 512, chunks of 16
samples per request. Responses up to 32,768 tokens, except test 1 (`--fill_context`: vLLM `max_model_len` 65,536 =
the configs' `max_position_embeddings`, YaRN factor 8 over 8,192; `max_tokens` = 65,536 − prompt tokens = 65,290 for
perp_055 and 068, 65,317 for 069). Natural-language output; no proofs, so Lean and `nd_verify` do not apply (the
review: `nd_verify/` unchanged, and no code of the run calls it).

**Prompts:** olmo-ifomega's 52 reach prompts (perp_069 reach only under the doubly-credited reading).
`olmo_iffu/manifest.csv` lists 49 of them; the three all-caps IFEval prompts of test 2b are in
`olmo_iffu/problems_hint.jsonl`:
- OMEGA `test_out` geometry: perp_055 (label corrected to 15), perp_068, perp_069
  (`omega_out_geometry_perpendicular_intersection_test_out_0xx`);
- IFEval: letter avoidance `ifeval_2350` (i < 6), `2447` (w < 2), `3478` (a ≤ 1), and all-caps `1314`, `2247`, `2563`;
- IFBench: the 43 hand-reach prompts of the olmo-ifomega review's list.

**Tests and seeds** (engine seeds as pre-registered, checked by the review):

| test | job | model | prompts | draws per prompt | engine seed | pod |
|---|---|---|---|---|---|---|
| 1, token budget | `T1_dpo` | `dpo` | the 3 geometry prompts | 64 at 65,536 − prompt | 7001 | iffu-a (TP 2) |
| 2, the hint | `T2_dpo` / `T2_s2300` | `dpo` / `s2300` | 3 letter-avoidance IFEval, 3 all-caps IFEval (2b, beyond the brief) and a seeded 10-prompt IFBench subset | 64 / 16 | 7101 / 7102 | iffu-a (TP 2) |
| 2c, the hint (added 02:01) | `T2c_dpo` / `T2c_s2300` | `dpo` / `s2300` | the other 32 eligible IFBench reach prompts | 64 / 16 | 7103 / 7104 | iffu-b (TP 1) |
| 3, deeper sampling | `T3_dpo` | `dpo` | a seeded 10-prompt IFBench subset, no hint | 768 more (to 1,024) | 7201 | iffu-b |
| 4, step 750 | `T4_s0750` + `T4b_s0750` | `s0750` | the 43 IFBench reach prompts | fresh chunks of 16 until 2 genuine, ≤ 32 (deviation 4) | 7301 / 7302 | iffu-b |

- **Subsets** (`olmo_iffu/subsets.py`): test 2's IFBench subset is `random.Random(20261106).sample` of the 42 left
  after excluding `ifbench_281` (its complete answer is 54 country names, more than 60 words): 290, 4, 64, 79, 170, 0,
  188, 207, 179, 268. Test 3's is `random.Random(20261107).sample` of the other 33: 252, 269, 253, 3, 78, 248, 211,
  166, 136, 133 (draw order). Both reproduce when the list is sorted by numeric IFBench index, the order `subsets.py`
  states. The pre-registration text does not name the order, and in the olmo-ifomega review's listed (lexicographic) order
  neither subset reproduces. Test 2c is the 43 minus test 2's 10 and 281; it contains all of test 3.
- **The hint:** `Answer in at most 60 words.`, appended to the user turn after a blank line; grading uses the original
  prompt and kwargs.

**Grades** (`olmo_ifomega/grade.py`, unchanged): IF `strict`, the official IFBench / IFEval checkers on the text after
the last `</think>` (the official `loose` variant is reported, not counted); OMEGA `omega` (OLMES exact match on the
answer) and `rl` (open-instruct `MathVerifier` on the whole response), with `any` = either and `doubly` = both.

**Hand gate (decides).** Blinded sub-agent readers, every label confirmed by the executor. IF rules 1–5
(`olmo_ifomega/handcheck_rule_if.md`) plus rule 6 (`olmo_iffu/handcheck_rule_if_addendum.md`, deviation 2: ALL-CAPS
answers to a title-case request are genuine under the literal reading, and a strict reading is reported too). OMEGA:
**sound** iff the final count comes from a set of pairs equal to the reference set or from a valid counting rule
correctly executed; the right count from a different set is **flawed**; a count not derived is a **guess**. Items:
test 1, every response credited by `omega` or `rl`; tests 2 / 2c, `dpo`'s strict successes in draw order until 2
genuine or none left, and `s2300`'s the same way (deviation 1); test 3, every strict success; test 4, in draw order
until 2 genuine.

**Outcome rules (pre-registered).**
- Test 1: ≥ 2 sound successes under `any` → **"token budget"**; 1 → "one sound success at 65k"; a sensitivity under
  `doubly`.
- Tests 2 / 2c: `dpo` ≥ 2 genuine in 64 → **"prompt-elicitable (brevity)"**; 1 → "one genuine with the hint"; the
  hint "breaks" `s2300` on a prompt with < 2 genuine of 16.
- Test 3: 0 genuine in 1,024 → **"survives at 1,024"** (one-sided 95 % Clopper–Pearson bound 2.92 × 10⁻³ per sample).
- Test 4: "`s0750` confirmed" at ≥ 2 genuine.

**Noise floor and power** (pre-registration): each verdict is one prompt's count. "≥ 2 in 64" has 50 % / 90 % power
at per-sample rates of 2.6 % / 5.9 %, and 13 % at 1 %, so a budget or hint that lifts `dpo` only to ≲ 1 % mostly
reads as "not reached / not elicited". `NOISE_FLOOR.md` (seed variance of the small `lean_seq` models) does not apply;
no seed-level comparison is made.

**Pods and cost.** Two pods, both NVIDIA H200:
- `iffu-a`, 2 × H200 SECURE at $10.58/h (tests 2 and 1, TP 2; 11-06 23:23 – 11-07 01:33 UTC): 2.18 h, $23.08;
- `iffu-b`, 1 × H200 COMMUNITY at $3.59/h (tests 3, 4, 4b and 2c, TP 1; 11-06 23:50 – 11-07 08:24): 8.58 h, $30.79.

Total **10.76 pod-hours, $53.87** of the $60 budget (`podbudget`; the review reproduces it). RunPod billed $54.34 with
disk, which equals the balance drop. The 80 % warning fired at 06:50 UTC ($48.22).

**Bucket** (public): `hf://buckets/dan-pandori/nd-rl/olmo-iffu/`:
- `artifacts/olmo_iffu/`: every sample's text and token ids, grades, metas, logs, and the hand items, keys and
  labels; the 17 bulk files are listed in the fork's `artifacts/MANIFEST.jsonl`;
- `olmo_iffu/`, `pod/olmo_iffu/` and `docs/`.

Registry rows: `hf://buckets/dan-pandori/nd-rl/registry/olmo-iffu/` (56 rows). No weights were stored off the pods.
Recompute, from the fork's root: `python3 fetch_artifacts.py artifacts/olmo_iffu/`, plus olmo-ifomega's four
fixed-phase IF sample files (`F_dpo_ifb`, `F_s2300_ifb`, `F_dpo_ife`, `F_s2300_ife`) for `lengths_hint.py`, then
`olmo_iffu/README.md`'s chain (`handcheck.py table`, `analyze.py`, `status.py`, `lengths_hint.py`, `figures.py`,
`summary.py`). The reviewer's order
is in their kit's `README.md`.

## Observations (the review's numbers)

### Test 1: `dpo` on the geometry prompts at 65,536 − prompt tokens (64 draws each)

| prompt (label) | cut at the new cap | finished length q1 / median / q3 / max (tokens) | credited `omega` / `rl` / `any` / `doubly` | credited at the 32,768 truncation | hand-sound `any` / `doubly` | outcome `any` / `doubly` |
|---|---|---|---|---|---|---|
| perp_055 (15) | 0 / 64 | 43,856 / 46,860 / 50,120 / 56,969 | 1 / 2 / 2 / 1 | 0 | 1 / 1 | one sound success at 65k / same |
| perp_068 (23) | 1 / 64 | 42,814 / 45,521 / 48,794 / 57,305 | 2 / 3 / 3 / 2 | 0 | 3 / 2 | **token budget** / **token budget** |
| perp_069 (14) | 0 / 64 | 40,136 / 45,087 / 47,768 / 60,638 | 2 / 4 / 4 / 2 | 0 | 2 / 1 | not reach under `any` (the rule's "token budget" is not supported) / one sound success at 65k |

- **Lengths.** Pooled, 1 of 192 draws is cut at the new cap (0.5 %), and the finished median is 45,805 tokens. All 192
  run past 32,768 tokens, and none has `</think>` within its first 32,768, so at olmo-ifomega's cap every draw would be
  unfinished and the truncation credits nothing. No prompt is "sampling budget at 32k". (The executor's quartiles in
  `numbers.md` use nearest rank; the medians are 46,865 / 45,521 / 45,299. The review: "reproduces (method)".)
- **Labels.** The stored 17 for perp_055 is exactly the number of perpendicular pairs with the meeting condition
  ignored (two of 055's perpendicular pairs do not meet: 2-12 × 3-5 and 11-1 × 10-8); under it, `omega` credits 0 of
  055's draws. 068 = 23 and 069 = 14 are right.
- **Hand gate (all 9 credited responses).** Final counted pairs equal to the reviewer's reference set: 055 c2.5, 068
  c2.12, 069 c3.4 and 069 c2.8; right class counts: 068 c1.15; flawed: 069 c3.6 (four wrong interior pairs), 055 c3.11
  (it counts a pair that never meets and rejects a valid one; the errors cancel to 15) and 069 c3.11. 068 c0.7 (sound,
  thinking only) rests on the reader, and 068's outcome holds without it. All four `rl`-only credits stop their
  write-up mid-sentence after `</think>`, with the credited `\boxed{}` inside the thinking: 2 sound (068 c0.7, 069
  c2.8) and 2 flawed. All 9 labels agree with the executor's. A second blind reader read two of them, 055 c3.11 and 069
  c2.8 (both `rl`-only). The other items that decide outcomes (055 c2.5, 068 c1.15 / c2.12, 069 c3.4 / c3.6) rest on
  one reader plus the reviewer's pair-set checks.
- **The olmo-ifomega baseline for perp_069** (phase-2 correction). In olmo-ifomega `dpo` already had two
  `rl`-credited responses on 069 within 32,768 tokens (`E1_dpo_oo3` draws c1.4, finished at 30,198 tokens, and
  c4.28, cut), both read sound there. So under `any` 069 was never reach. Under `doubly`, where it was reach, this run
  gives one sound success at 65k. In the review's words, "'Token budget' explains 068 under both readings and 069
  under neither."
- **Rates** (any-credit hand-sound, the executor's Clopper–Pearson intervals, reproduced): 055 0.016 [0.0004, 0.084],
  068 0.047 [0.010, 0.131], 069 0.031 [0.004, 0.108]. For scale, `s2300` had 28 / 40 `omega` credits at 32,768 on 068
  and on 069 in olmo-ifomega. That is checker credit, not hand-sound: the predecessor review read 6 of 9 `s2300`
  credits on 069 sound, so `s2300`'s hand-sound rate is unmeasured and probably lower.

### Tests 2 and 2b: the hint on 16 prompts (`dpo` 64 each, `s2300` 16 each)

| prompt | `dpo` strict / 64 | `dpo` genuine | `dpo` outcome | `s2300` strict / 16 | `s2300` genuine | hint breaks `s2300` | `s2300` strict in olmo-ifomega |
|---|---|---|---|---|---|---|---|
| ifeval_2350 (i < 6) | 0 | 0 | not elicited | 16 | ≥ 2 | no | 24 / 24 |
| ifeval_2447 (w < 2) | 2 | 2 | **prompt-elicitable** | 14 | ≥ 2 | no | 22 / 24 |
| ifeval_3478 (a ≤ 1) | 0 | 0 | not elicited | 15 | ≥ 2 | no | 10 / 24 |
| ifeval_1314 (2b) | 19 | ≥ 2 | **prompt-elicitable** | 12 | ≥ 2 | no | 22 / 24 |
| ifeval_2247 (2b) | 29 | ≥ 2 | **prompt-elicitable** | 16 | ≥ 2 | no | 24 / 24 |
| ifeval_2563 (2b) | 5 | ≥ 2 | **prompt-elicitable** | 15 | ≥ 2 | no | 24 / 24 |
| ifbench_179 | 2 | 2 | **prompt-elicitable** | 10 | ≥ 2 | no | 21 / 24 |
| ifbench_290 | 3 | 2 (ALL CAPS) | prompt-elicitable; **strict reading (rule 6): not elicited** | 13 | ≥ 2 | no | 17 / 24 |
| ifbench_0, 4, 64, 79, 170, 188, 207, 268 | 0 each | 0 | not elicited | 14, 15, 4, 14, 4, 2, 6, 14 | 0, 0, ≥ 2, ≥ 2, ≥ 2 (a reader call, below), 1, ≥ 2, ≥ 2 | **0, 4, 188** (and 170 by the executor's label) | 12, 20, 4, 21, 8, 5, 8, 18 of 24 |

- **Prompt-elicitable:** letter avoidance 1 of 3 (2447); all-caps (2b) 3 of 3; the IFBench subset 2 of 10 (179, 290),
  or 1 of 10 under rule 6's strict reading. Of 290's three strict answers, the first reads gamed (word-fused ALL CAPS
  that hides lowercase starts), and the other two are genuine but ALL CAPS.
- **The hint breaks `s2300`** on `ifbench_0` and `ifbench_4` under every reader, and on `ifbench_188` under the
  reviewer's adjudication (supported by their second and third readers; the first reader alone gives 188 two genuine).
  **`ifbench_170` is a reader call:** both of the reviewer's readers labelled its rank-3 answer genuine ("Franks shaped
  medieval Europe. They set important frameworks. Impact blended varied factors."), but the reviewer's own
  disagreement rule would call it a generic slogan, as they called rank 0. With rank 3 gamed, 170 has 1 genuine and the
  hint breaks `s2300` there, as the executor has it.
- **Answer length.** `dpo`'s median answer is ≤ 80 words on 13 of 16 prompts, and 46 % of its answers are ≤ 60 words;
  `s2300` keeps every answer ≤ 60 words. Over the 48 hinted prompts (tests 2 and 2c), the median of per-prompt median
  answer words goes from 231 to 60 for `dpo` and from 31.5 to 19 for `s2300` (unhinted = olmo-ifomega's fixed-phase
  draws).
- **`s2300`'s strict rate with the hint** is below its unhinted rate on 2 of 16 prompts against the fixed phase
  (`ifbench_179` 10 / 16 against 6 / 8; `ifeval_2563` 15 / 16 against 8 / 8) and on 6 of 16 against all 24 olmo-ifomega
  draws. None is significant individually; the largest drop, 179 (21 / 24 → 10 / 16), has one-sided Fisher p = 0.07.

### Test 2c: the hint on the other 32 eligible IFBench reach prompts

- **Prompt-elicitable on 10 of 32:** `ifbench_19, 78, 133, 173, 186, 187, 220, 226, 252, 253`. One genuine: 136, 204,
  211. The other 19 are not elicited. Four of the 10 are test-3 survivors: 78, 133, 252 and 253 (252: 36 / 64 strict
  and 253: 43 / 64 with the hint, against 0 genuine in 1,024 each without it).
- **The hint breaks `s2300` on 4 of 32** under the reviewer's final labels: `ifbench_3` and `166` under every reader,
  `ifbench_1` under the reviewer's adjudication (two of three readers read its second genuine as keyword stuffing), and
  `ifbench_226` by the majority of three readers on each deciding item (1 genuine of 16; 2 under the first readers
  alone). Under the first readers alone it is 2 (3, 166). `ifbench_284` is **not** broken: its 4 strict `s2300` answers
  are the exact word span, genuine by rule 5, which names 284 (the executor labelled all 4 gamed).
- Per-prompt rows (strict, genuine, `s2300`'s olmo-ifomega strict rate) are in the kit's
  [`out/outcomes.md`](https://github.com/dan-pandori/nd-takehome/blob/c0c6e053/review/olmo-iffu-recount/out/outcomes.md)
  (the reviewer's final labels). The executor's `olmo_iffu/summary_tables.md` gives strict, genuine and answer words
  per prompt with the executor's labels (226's `s2300` "0 (5)", 284 broken).

**Pooled, the 42 eligible IFBench prompts:** 12 prompt-elicitable (11 under the strict title-case reading), 3 with one genuine,
27 not elicited. Over the 48 hinted prompts (42 IFBench, 6 IFEval) the hint breaks `s2300` on 7 (`ifbench_0, 4, 188,
1, 3, 166, 226`), 8 if 170 counts; under the first readers alone, 4. Among the prompts where `dpo` is not elicited it
breaks `s2300` on 6 (0, 4, 188, 1, 3, 166), or 7 with 170: the executor's "8 more" counted 170 and 284.

### Test 3: `dpo` to 1,024 draws on 10 IFBench reach prompts (no hint)

- **0 strict successes in 7,680 new draws** (768 on each of 252, 269, 253, 3, 78, 248, 211, 166, 136 and 133), and 0
  genuine in the 256 prior draws (the one prior strict success, 252's in olmo-ifomega, reads gamed). **All 10 survive
  at 1,024**, each with a one-sided 95 % Clopper–Pearson upper bound of 2.92 × 10⁻³ on `dpo`'s per-sample rate (1 /
  bound = 342.3: the executor's "k-to-solve ≥ 342" and the kit's "k ≥ 343" are the same bound).
- Binding-constraint passes: 0 / 768 on every prompt except 136 (2 and 6 of 768 on its two constraints, never jointly).
- 3 of the 7,680 are cut at 32,768 tokens (all on `ifbench_248`). The official `loose` variant credits 790 of the 7,680
  (765 on `ifbench_133`); loose successes do not count and were not hand-read. The executor's log attributes 133's to
  the loose grader passing answers with no `*` bullets vacuously (not reviewed).
- With the hint (test 2c): 4 prompt-elicitable (78, 133, 252, 253); 136 and 211 one genuine; 248 and 269 survive both,
  with `s2300` unbroken; on 3 and 166 the hint also breaks `s2300`.

### Test 4: `s0750`, fresh draws until 2 genuine, at most 32 per prompt (deviation 4)

**Confirmed (≥ 2 genuine) on 26 of 43**, under both readings of rule 6. Every prompt stopped at 16 draws has 2 genuine.
`ifbench_19`'s first 16 draws give 4 strict, all gamed; the continuation T4b gives 2 genuine, so it is confirmed at 32.

| not confirmed (17) | draws | strict | genuine |
|---|---|---|---|
| `ifbench_0, 1, 79, 253, 264` | 32 | 1 each | 1 each |
| `ifbench_188, 207` | 32 | 2 each | 0 |
| `ifbench_189` | 32 | 3 | 0 |
| `ifbench_3, 67, 166, 169, 170, 203, 252, 268, 284` | 32 | 0 | 0 |

26 is a lower bound for the pre-registered quantity (≤ 64 draws): 17 prompts stopped unconfirmed at 32. For comparison
(the executor's number, reproduced), olmo-ifomega's `s0750` fixed phase had ≥ 2 strict in 8 on 16 of the 43.

### Status of the 52 reach prompts after this run (the review's count)

| status | count | prompts |
|---|---|---|
| token budget | **1** | perp_068 (both readings) |
| one success with room or hint | **3** | perp_055 (1 sound in 64 at 65k, both readings); perp_069 (1 sound at 65k under `doubly`; under `any` it was not reach); `ifbench_204` (1 genuine with the hint) |
| prompt-elicitable (brevity) | **16** | IFEval 2447, 1314, 2247, 2563; IFBench 179, 290 (literal reading of rule 6 only), 19, 78, 133, 173, 186, 187, 220, 226 (the hint also breaks `s2300` there), 252, 253 |
| out of reach at 1,024 and with the hint, `s2300` unbroken | **2** | `ifbench_248`, `ifbench_269` |
| out of reach at 1,024, hint inconclusive | **4** | 136, 211 (one genuine with the hint); 3, 166 (the hint breaks `s2300`) |
| not elicited by the hint, hint fine for `s2300`, sampled only to 256 | **21** (20 if 170 is broken) | IFEval 2350, 3478; IFBench 64, 79, 207, 268, 67, 128, 154, 169, 189, 203, 206, 209, 217, 264, 267, 271, 280, 284, 170 |
| hint uninformative (`dpo` not elicited, the hint breaks `s2300`) | **4** (5 with 170) | `ifbench_0, 4, 188, 1` |
| untested | **1** | `ifbench_281` (its complete answer cannot fit 60 words) |

The executor's counts were token budget 2, one success 2, prompt-elicitable 16, survive both 2, inconclusive 4, not
elicited with the hint fine 19, uninformative 6, untested 1; they differ on 069 and 284, and 170 is a reader call.
Under rule 6's strict reading, 290 moves from prompt-elicitable (16 → 15) to "not elicited, hint fine" (review §5,
§10). "Explained" by the token budget or brevity: **17 of 52** (1 + 16), not 18.

### Hand gate: the review's blind repeat and the comparison

- **Readers and volumes.** The reviewer's blinded sub-agents saw the prompt as sampled, the constraints with the
  checker's own descriptions and the answer only, with the pre-registered rubric (rules 1–5), an ALL-CAPS flag for rule
  6, and, for hinted items, a note that the last paragraph is the experimenter's length limit. Final volumes: 390 items
  read at least once and 316 at least twice. First and second reads disagree on 20, all decided in `hand/confirm.json`
  (18 gamed, 2 genuine). The 15 found in phase 1 were decided by a rule written before deciding: thin hinted answers
  are genuine only if they deliver the request's core content (restating the premise, generic slogans or keyword
  padding is gamed), and hidden sentence breaks and fused words are vacuous satisfaction (rule 1). The 5 from the
  second-reader gap found in phase 2 were decided by the majority of three readers.
- **Against the executor's 368 items** (joined by draw key): 336 shared, 315 agree (93.8 %). All 9 OMEGA labels and
  every test-4 label agree. 20 of the 21 disagreements are `s2300` answers with the hint; the other is T2c `dpo`
  `ifbench_220` rank 0 (the executor applies rule 3, the reviewer rule 1; 220 is prompt-elicitable either way).
- **Outcomes that depend on the reader:** only `s2300` with the hint, on 188, `ifbench_1` and 226 (each "not broken"
  under the first reader alone) and 170 (above). No `dpo` outcome and no test-4 outcome changes between label sources.
- **Incomplete walks in the executor's gate:** 226's `s2300` check stopped at 5 of 16 ("left at 0 genuine of 5 read
  (11 unread)" in the log); T2c `dpo` 78's walk skipped ranks 3, 4 and 6 (gamed in the reviewer's reading; outcome
  unchanged). So "every deciding one included" is "not quite true".
- **Borderline readings.** Inline "- " sub-points on `*` bullets (the `format:sub-bullets` prompts 128 / 133 / 136) were
  read as genuine (format slack); the checker only needs a "-" after each "*", and `dpo`'s 133 prompt-elicitable rests
  partly on such an answer. Rule 6 decides `dpo`'s `ifbench_290`.

### Compute per arm (registry rows = the reviewer's per-job sums; all NVIDIA H200)

| job | GPUs | samples (attempts = verifier checks) | generated tokens | GPU-s ((load + gen) × GPUs) |
|---|---|---|---|---|
| T1_dpo | 2 × H200 secure | 192 | 8,799,941 | 10,276 |
| T2_dpo | 2 × H200 secure | 1,024 | 3,404,930 | 2,397 |
| T2_s2300 | 2 × H200 secure | 256 | 1,351,053 | 1,512 |
| T3_dpo | 1 × H200 community | 7,680 | 33,302,766 | 17,835 |
| T4_s0750 | 1 × H200 community | 1,088 | 7,639,142 | 4,903 |
| T4b_s0750 | 1 × H200 community | 16 | 85,162 | 263 |
| T2c_dpo | 1 × H200 community | 2,048 | 9,156,026 | 5,187 |
| T2c_s2300 | 1 × H200 community | 512 | 2,852,188 | 1,662 |
| **total** | | **12,816** | **66,591,208** | **44,036** (12.2 GPU-h) |

- Training steps, training tokens and Lean checks are 0 in every row (56 rows: 8 jobs × 7 metrics).
- Pod time is 46,580 GPU-s, so set-up, downloads and idle add 2,544 GPU-s (5.5 %) that no job row carries.
- Arms are compared on samples per prompt, fixed by design; no arm is matched on compute (pre-registration).
  *[librarian]* Flagged under the policy's 1.25× rule (unreviewed arithmetic from the rows above): on the hinted
  prompts `dpo` draws 4× `s2300`'s samples (64 against 16), which is 2.5× / 3.2× the generated tokens and 1.6× / 3.1×
  the GPU-seconds in tests 2 / 2c. There `s2300` is the check that the hint does not break it, not a competing arm.
- *[librarian]* Cost per draw at the measured job times (unreviewed arithmetic): test 1 ≈ $15.1 for 192 draws on secure
  H200s (≈ $0.079 per 65k-budget draw, ≈ $1.72 per million tokens); test 3 ≈ $17.8 for 7,680 draws on a community H200
  (≈ $0.0023 per draw, ≈ $0.53 per million tokens).

## Expectations against outcomes

Binding expectations were committed before any pod (`048ef5fe`); the test-2c ones before any of its samples. The
review's scoring (§12 with its phase-2 errata, and §Compare):

| expectation | observed | verdict |
|---|---|---|
| T1 cut share at the new cap 65 % (35–90 %) | 0.5 % (1 / 192) | **miss** |
| T1 ≥ 97 % unfinished at 32,768 | 100 % | holds |
| T1 finished median 48k (38–58k) | 45.8k pooled (45,521, the median of the per-prompt medians) | inside |
| T1 sound per prompt, median 4 (0–15) | 1, 3, 2 → 2 | inside |
| T1 "token budget" on 2 of 3 (1–3) | 1 of 3 under the consistent reading (068; the executor's 2 counts 069 under `any`, where it was never reach) | inside |
| T1 ≥ 90 % of sound successes only after 32,768; no "sampling budget at 32k" | 100 %; none | holds |
| T2 letter avoidance 1 of 3 (0–2), most likely 2447 | 1 (2447) | at the point |
| T2b all-caps 2 of 3 (0–3) | 3 | inside |
| T2 IFBench subset 3 of 10 (1–6) | 2 (strict reading 1) | inside |
| T2 `dpo` median ≤ 80 words on ≥ 12 of 16 | 13 | holds |
| T2 gamed share of `dpo`'s strict successes 30 % (10–60 %) | 1 of the 16 read (6 %), by the executor's labels and the reviewer's; first readers alone 0 % | **miss** (not reported by the run) |
| T2 `s2300` not broken on ≥ 14 of 16 | 13 on the reviewer's final labels (12 if 170 is broken, as the executor has it; 14 under the first readers alone) | **miss** under the executor's, second-reader and final labels; holds under the first readers alone |
| T3 survivors 6 of 10 (3–9); resolved 2 (0–5); exactly one genuine 2 (0–4) | 10; 0; 0 | **miss** on survivors; resolved and one-genuine inside |
| T4 `s0750` confirmed 24 of 43 (16–32) | 26, at a 32-draw cap, not 64 | inside only if at most 6 of the 17 unconfirmed would confirm in 32 more draws |
| T2c prompt-elicitable 6 of 32 (2–12) | 10 | inside |
| T2c hint breaks `s2300` on 10 of 32 (4–16) | 4 on the final labels (1, 3, 166, 226); 2 under the first readers alone | inside at the lower edge on the final labels; a miss under the first readers alone |

The run's write-up reports three misses (the cut share, the test-3 survivors and `s2300` 12 of 16); the review adds the
gamed-share miss.

## The review's corrections to the executor's files

The fork's files are uncorrected; the copy of the plain summary here applies its corrections. By file:
- **`run_olmo_iffu.md`:** finding 1's "token budget" on perp_069 (not supported), "Given 65,290 response tokens"
  (65,317 for 069) and the unmarked "`s2300` 28 / 40 at 32k" (olmo-ifomega's, and `omega` credit beside `dpo`'s
  hand-sound rates); Setup's "52 reach prompts (`dpo` 0 genuine in 256)" needs "(069: under doubly)"; finding 2's "On
  8 more the hint also breaks `s2300`" → 6 (7 with 170); the status line → token budget 1, one success 3, not elicited
  with the hint fine 21 (20), uninformative 4 (5); "Against expectations" lacks the gamed-share miss, and its "hint
  harmless for `s2300` on 12 of 16" is 13 on the review's final labels; within range, "token budget 2 of 3" → 1 of 3
  under the consistent reading, and "`s0750` 26 of 43" is at a 32-draw cap (inside only if at most 6 of the 17
  unconfirmed prompts would confirm in 32 more draws); the figure mixes readings on 069.
- **`numbers.md` § olmo-iffu:** the 2c row "1, 3, 166, 284 (226 undecided …)" → 284 not broken, 226 broken by the
  majority of three readers (not under the first readers alone); "8 where the hint also breaks `s2300`" → 6 (7), so
  *[librarian]* its "19 not elicited with the hint fine for `s2300`" → 21 (20); "`s2300`'s strict rate with the hint
  is at or above its unhinted rate on all 16 test-2 prompts" is wrong (above). Its T1 quartiles are nearest-rank (a
  method difference, not an error).
- **`log.md` § olmo-iffu:** "everywhere" (the same strict-rate claim); 226 "left at 0 genuine of 5 read (11 unread)"
  (incomplete gate); "`dpo`'s median answer falls from 64–600 words … to 50–95" leaves out `ifbench_64` (6 → 12).
  *[librarian]* Its 09:07 entry also carries the status-line counts ("token budget 2 (perp_068, 069)", "not elicited
  by the hint, sampling untested 19", "hint uninformative 6") and "368 hand items, all confirmed" beside the
  incomplete 226 gate.
- **`olmo_iffu/plain_summary.md`:** "18 are now explained" (17), "two puzzles … mostly a length limit" (one of three),
  "about 70 %" (checker credit), "8 more" (6–7), "every deciding one included" (not quite); applied in the copy here.
- **`olmo_iffu/status.csv` and the figure:** 069's `t1_label` and `explanation` "token budget" (its `before_note`
  gives both readings; the figure draws the doubly "before" cell beside the any-credit test-1 cell), 284 counted as
  broken, 170 a reader call, and (*[librarian]*) 226's `s2300` "0/16" from the incomplete reading.
- **`preregistration/olmo-iffu.md`:** the header's end time ("Written 23:20–23:50 UTC") and the config check's
  "23:30", and `olmo_iffu/handcheck_rule_omega.md`'s "written 2026-11-06 23:50 UTC" (the same commit), postdate the
  23:22:37 commit that contains them (wording); deviations 1, 2 and 4 were adopted after their tests' sampling began.
- *[librarian]* `STATUS.md`'s 09:11 "Done" entry repeats three of these: "with 65,290 response tokens" (65,317 for
  069), perp_069's "token budget" (in its test-1 sentence and in "2 are token budget") and "the hint also breaks
  `s2300` on 8".

## Deviations from the pre-registration and the brief

- **Test 2b** (in the original pre-registration, beyond the brief, ≈ $0.5): the three all-caps IFEval reach prompts,
  so all six IFEval reach prompts were hinted.
- **Test 2c** (an addition committed 02:01:45, before T2c's first sample at 06:23:12): the hint on the other 32
  eligible IFBench reach prompts, for ≈ $5, reported separately from test 2's pre-registered subset and then pooled.
- **Deviation 1** (00:24–00:36 UTC): `s2300`'s hinted answers read until 2 genuine or none left, not "the first 2
  strict successes"; adopted after T2_s2300 started (23:53).
- **Deviation 2, rule 6** (00:08 UTC): ALL-CAPS answers to a title-case request are genuine under the literal reading,
  with a strict reading reported; adopted after T2_dpo started (23:33).
- **Deviation 3** (00:41 / 01:05 UTC): test 4 was unqueued when test 1 looked slow and reinstated by rule (it starts
  only if ≤ $50.50 is spent when test 3 ends). The only deviation that preceded its test.
- **Deviation 4** (05:37 UTC): test 4 capped at 32 fresh draws per prompt (64 pre-registered), so that test 2c fitted
  under the $55 gate; adopted after T4 started (04:59, 400 samples in).

The review: "deviations are disclosed, not pre-registered". Expectations were written before the run (pre-registration
committed 23:22:37 < first pod 23:23:56 < first sample 23:33:13), and the original sections are unchanged since.

## Interpretation (labelled; not tested further here)

- **The executor's reading, in the pre-registration's terms:** one fixed hint is a per-prompt elicitation method
  (`capability-vs-propensity`, card 3 of the capability standard), so a prompt-elicitable prompt changes from "not
  reached at 256" to "elicited by prompt", and RL's gain there is not evidence of a created ability. The plain summary
  calls about a third of the instruction-following reach "a verbosity habit that a one-line prompt change unlocks:
  elicited, not created".
- **Test 3 against test 2c** (the review's "cleanest result"): on 4 of the 10 prompts, the DPO model has no genuine
  success in 1,024 unhinted draws but succeeds with the hint (on 252 and 253, 36 and 43 strict of 64). In the plain
  summary's words, "Unfindable in 1,024 tries need not mean absent."
- **`ifbench_248` and `ifbench_269`** survive 1,024 draws and the hint with `s2300` unbroken; the review calls them
  "the run's two candidates for created ability".
- **The geometry gap** is "token budget" on perp_068 only. On 055 and 069 (under `doubly`) the DPO model has one sound
  success in 64 at 65k, which the review's next measurement would decide.

## Limitations

- **One method per alternative.** One hint wording; 65,536 tokens is the model's whole context; test 3 covers 10 of
  the 43 IFBench reach prompts. The 17–19 IFBench prompts (19–21 with `ifeval_2350` and `3478`) that the hint neither
  elicited nor broke were sampled only to 256 (*[librarian]* in the review's range, 17 is the executor's count, with
  284 and 170 broken; the review's own labels give 18–19).
- **Power.** "≥ 2 in 64" has 13 % power at a 1 % per-sample rate, so "not elicited" is weak evidence against a hint
  that lifts `dpo` to ~1 %.
- **Reader-dependent labels.** Four `s2300` outcomes (188, `ifbench_1`, 226, 170) depend on which readers' labels
  are used, and `dpo`'s 290 on rule 6's reading; 133's prompt-elicitable rests partly on the sub-bullet format slack.
  No `dpo` outcome changes between the review's label sources.
- **Test 4** stopped at 32 draws, so 26 of 43 is a lower bound.
- **The `s2300` comparator** for the geometry prompts is `omega` checker credit from olmo-ifomega, not hand-sound.
- **Grading.** Strict checkers plus the hand gate; loose-only successes were neither counted nor read.
- **Pod time outside job rows:** 2,544 GPU-s (5.5 %) of set-up, downloads and idle.

## Next steps (the review's next measurements; not launched)

1. **perp_055 and perp_069:** a further 128 `dpo` draws at 65,536 tokens on each (≈ $20 at test 1's rate) would
   decide "token budget" against "out of reach at 65k" under both graders.
2. **`ifbench_248` and `ifbench_269`:** a second elicitation (a different brevity wording, or a one-shot example) on
   these two, plus `dpo` at 4,096 draws, would test whether they survive more than one method. *[librarian]* At test
   3's average cost per draw, the 6,144 extra `dpo` draws come to ≈ $14 (unreviewed arithmetic).
3. **The IFBench prompts still open on sampling:** the 17–19 (19–21 with `ifeval_2350` and `3478`; see Limitations)
   that the hint did not elicit and did not break need test 3's 1,024-draw treatment "before 'not reached' can mean more than 'not
   reached at 256 plus one hint'". *[librarian]* At test 3's average cost per draw, 768 more draws each comes to
   ≈ $30–37 (unreviewed arithmetic; tokens per draw vary by prompt).

## For olmo-novelty (queued; claim C8 and question 2)

[olmo-novelty](../../docs/proposals/literature/BRIEF_olmo-novelty.md) branches from the fork's `dan_olmo-iffu`, whose
executor files carry the uncorrected wording listed above; quote this summary and the review instead. What this run
measured, in the review's terms, on the same three models:
- **Token budget (C5's OMEGA half):** perp_068 only; perp_055 and perp_069 (under `doubly`) have one sound DPO
  success in 64 at 65k; perp_069 was never reach under `any`.
- **Brevity (C5's IFEval half, and C3):** 4 of 6 IFEval and 12 of 42 IFBench reach prompts are prompt-elicitable (11
  under rule 6's strict title-case reading); 17 of olmo-ifomega's 52 reach prompts are explained by the token budget
  or brevity (*[librarian]* 16 under the strict title-case reading).
- **Larger k:** k = 1,024 on 10 IFBench reach prompts, all 10 surviving (≤ 2.92 × 10⁻³ per sample), 4 of them
  prompt-elicitable.
- **Not tested:** k above 1,024; the other 33 IFBench reach prompts beyond 256 unhinted draws; format-tolerant
  grading. The official `loose` variant was computed but neither counted nor hand-read: in test 3 it credits 790 of
  the 7,680 new draws, 765 of them on `ifbench_133`, where `strict` credits none. The executor's log attributes 133's
  to the loose grader passing answers with no `*` bullets vacuously (not reviewed).
- **The compute axis:** `s0750` is confirmed on 26 of 43 IFBench reach prompts within 32 fresh draws (a lower bound).

## Figures

[`olmo_iffu_status.png`](olmo_iffu_status.png) (the executor's figure, unchanged): per reach prompt, the status
before and after each test. It needs these corrections (§Compare, §Verdict reword item 1, and the 2c findings):
- **perp_069:** the "before" cell (0 / 256) is the `doubly` reading, beside the any-credit test-1 cell (2 / 64, drawn
  as "reached"). Under `any`, `dpo` had 2 / 256 sound before; under `doubly`, test 1 gives 1 / 64 (one sound success).
- **`ifbench_284`:** drawn as "hint also breaks step_2300"; in the review it is not broken (not elicited, hint fine).
- **`ifbench_226`:** drawn as prompt-elicitable, which stands; the hint also breaks `s2300` there (majority of three
  readers).
- **`ifbench_170`:** drawn as broken, a reader call.
- Test-4 cells are at ≤ 32 draws.
