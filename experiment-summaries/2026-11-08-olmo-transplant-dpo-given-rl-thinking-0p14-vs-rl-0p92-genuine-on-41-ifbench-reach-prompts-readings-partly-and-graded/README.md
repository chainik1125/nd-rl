---
written_on: 2026-11-08
written_by: agent:claude
---

# olmo-transplant: handed Olmo 3.1 Think 32B's own successful thinking on 41 IFBench reach prompts, its DPO start writes a genuine answer 0.14 of the time against the RL model's 0.92; without that thinking the RL model scores 0.06–0.09 and the DPO start 0.00, so the thinking carries most of RL's gain, not all of it; pre-registered readings "partly" and "graded"

Status: completed 2026-11-08 (DONE 08:21:03 UTC) and reviewed 2026-11-08 (§Recount, phase 1, blind, 10:09 UTC; final
commit 10:50 UTC). **No hard-constraint violation, nothing quarantined.** The review's verdict is **"Stands"**: no
headline claim is unsupported. Three statements are wrong as stated (the fresh-forcing ratio at 4k, the calibration
cut shares and the job GPU-hours), two pre-registered analyses were omitted (reading 7, and the cut-only budget curve
with its N / c bins), and the review lists seven items to reword or add.

The models are `dpo` = `allenai/Olmo-3-32B-Think-DPO` (the RL run's start) and `s2300` = `allenai/Olmo-3.1-32B-Think`
`main` (= `step_2300`), 32 B, pretrained, natural-language chat; nothing was trained. "RL's thinking" is stored
thinking that `s2300` wrote in olmo-ifomega, re-inserted by token ids; "genuine" is the official IFBench strict checker
plus the blinded hand gate (model readers).

In phase 1 (blind, committed 10:09:29 UTC; the workspace had the executor's write-ups removed), the reviewer used
their own code (kit `review/olmo-transplant-recount/`, which imports no executor module) to:
- find all 328 trace rows in their own bucket copies of olmo-ifomega's four sample files, and check for every trace
  that the inserted thinking equals the stored token ids (sha256 of the ids and of the prefix, the `</think>` position
  and the closing tokens);
- rebuild every pre-registered request from the trace rows (869 + 205 per writer; none missing or extra), and check on
  all 17,184 answer rows that sha256(prompt ids + inserted prefix) equals the row's `input_sha256`. The prompt ids came
  from the reviewer's own tokenizer download at both revisions (the two `tokenizer.json` files are identical);
- read the template boundary on 40 random rows: the prompt ends `<|im_start|>assistant\n<think>`, every prefix ends
  `\n</think>\n\n`, and they saw no tokenisation seam;
- re-grade all 17,472 rows (17,184 answers and 288 fresh-forcing samples) with the upstream IFBench checker (0
  differences from the executor's grades), and olmo-ifomega's 8,512 source rows;
- check the prompt sets against the stated criteria (they did not re-run the executor's seeded draws);
- re-read the deciding successes blind with their own readers (799 distinct items, up to three readers, majority), and
  recompute every rate, contrast, retained gain and reading under the executor's labels and their own, with rule-1,
  rule-6 and sub-bullet variants;
- recompute the compute from every job's start and end lines, check the 17 MANIFEST entries against the bucket
  (sha256), and scan all 629 bucket objects for secrets: one pattern hit, inside a base64 token-id field, not a
  credential.

Phase 2 compared the executor's files with the recount (§Compare, 24 items). In `results.csv` (918 rows), strict 918 /
918 and genuine 845 / 845 (the executor's labels, through the reviewer's code) equal the reviewer's per-prompt tables
to 6 × 10⁻⁵; the 73 calibration B / P cells, strict only by design, are excluded. An independent sub-agent's
adversarial check of the review found 25 points, all addressed (`out/adversarial_check.md`); the errors it found in
the committed §Recount are listed in the review under "Corrections to §Recount".

*[librarian]* The review discloses one phase-1 leak. At 10:08:41, 48 s before the §Recount commit and after every
number had been computed, a `git diff` of `STATUS.md` (an append-only check) printed the executor's status lines with
the T1 / T2 headline. After it the reviewer added the disclosure, rewrote the append-only row of the hard-constraint
table and documented the kit; no number changed. The librarian's check of the reviewer's transcript found three more
phase-1 reads of executor files:
- 08:20:11, the reviewer's first command: the executor's memory note `olmo-transplant-learnings.md`. At that time it
  held infrastructure lessons only (the `</think>` tokens, KV limits, `ot-1`'s dead GPU, hand-gate key collisions), no
  results. The executor added its headline numbers to the note and to its memory index line at 08:21:26, after the
  reviewer's session had started (08:20:05; no compaction followed).
- 08:21:59: the executor's `olmo_transplant/README.md`, which the review lists among the files phase 1 did not open. It
  is the pipeline description; the reviewer's grep of it for percentages and decimal rates, run first, found none.
- 08:38:13: the metric counts and the first row of each registry file, including the headline-rows file, which the
  review also lists as unopened. That row is `dpo`'s T1 strict rate, 0.1392 [0.078, 0.2114] (41 prompts, 920
  answers). The reviewer's own code had printed its T1, 0.139 [0.080, 0.211], 82 s earlier (08:36:51).

Apart from these, every headline figure in the tool outputs before the §Recount commit came from the reviewer's own
scripts. The recount's numbers come from the reviewer's code on the raw rows.

- **What stands (the review's verdict).**
  - "The data are what the pre-registration says": every inserted thinking prefix equals the stored ids, every one of
    the 17,184 answers was sampled after exactly the registered input, and the grades reproduce with 0 differences.
    Every per-prompt rate in `results.csv` and every condition rate, contrast and retained gain in `numbers.md`
    reproduces.
  - The review's blind re-read of 799 deciding items agrees with the executor's labels on 97.1 % of shared samples.
    With the review's labels every reach-41 condition rate is within 0.02 of the executor's (the 6-prompt
    fresh-forcing F4096 rate is not), and no pre-registered reading changes.
  - **The headline:** given `s2300`'s own successful thinking, `dpo` writes a genuine answer 0.14 of the time against
    `s2300`'s 0.92 (R1 = 0.15 [0.09, 0.23]). `s2300` with no thinking or with `dpo`'s failed thinking scores
    0.06–0.09. Readings 1, 2, 4 and 5 fail and 3 and 6 hold, also on `brief20`.
  - "The T1 mechanism counts reproduce exactly: 256 of `dpo`'s 920 T1 answers (28 %) contain RL's drafted answer
    verbatim, and 183 of those fail, every one by adding words."
  - Cost $17.82 of $60.
- **Wrong as stated** (the review's numbers in parentheses: §Verdict reword items, §Compare rows).
  - **Fresh forcing at 4k** (item 1; #8): "0.62 of its natural rate" is 0.46 (the review's labels) to 0.52 (the
    executor's labels with rule 1 applied mechanically), or the strict 0.58. Unread `ifbench_136` answers with no `*`
    (vacuous under rule 1) were counted genuine through their cell's share. F8192's 0.88 is 0.84 with rule 1.
  - **Calibration cut shares** (item 5; #16): three calibration `s2300` cells exceed 1 % of the answers actually
    sampled (P 5 / 184 = 2.7 %, B4096 4 / 136 = 2.9 %, B8192 2 / 64 = 3.1 %), not two (2.7 %, 1.4 %). The executor
    divided B cuts by the primary-curve cell (sampled rows plus substituted full rows).
  - **Job GPU-hours** (item 6; #19): 2.52 GPU-h of job time, not 2.35, which omits R_dpo's killed stage-0 attempt
    (513 s, whose 1,384 rows are kept) and a 12 s failed restart (2.50 h with them; the other 77 s is job overhead).
    Add 525 s to R_dpo's registry `gpu_seconds`.
- **Omitted (pre-registered).**
  - **Reading 7, P** (item 2; #9, #11): `dpo` F(P) = 0.15 ≤ 0.25, "the success rests on the drafted answer"; `s2300`
    0.45, neither threshold as graded, but 0.25–0.26, "on the copying threshold", when answers that re-open thinking
    count as failures. "The draft helps but is not required" needs that qualifier.
  - **The cut-only B curve and the N / c bins** (item 4; #23): on the traces actually cut, `s2300` is 0.19 / 0.32 /
    0.43 / 0.35 at 1k–8k against 0.88–0.93 for the same traces in full (genuine, the review's labels), "so the
    primary curve's 'graded' rise at 4k–8k is mostly uncut traces entering at their full value".
- **Reword or fix.**
  - **"RL's thinking is necessary"** (item 3; #13) becomes "carries most of the gain": `s2300` without any thinking
    still scores 0.08–0.09 where `dpo` scores 0.00.
  - **File defects** (item 6; #24 and §Compare's "Model labels" paragraph): blank `results.csv`'s calibration B / P
    `genuine_rate`, and put `ckpt` on the registry's single-writer contrast rows.
- **Worth saying** (item 7; the review: "not errors"). The forced `</think>` does not cap `s2300`'s thinking: 28 % of
  P and 8–9 % of B1k / B2k answers re-open it, "a confound for every 'capped at N' reading". Reading 3 holds only
  through its "neither 1 nor 2" clause: R1 = 0.15 lies in reading 2's range, far below reading 3's 0.40–0.75 band. The
  draft split pools the P cuts into "without". The plain summary's "256 tries" needs its source (olmo-ifomega).

Review: [`review_olmo-transplant.md`](https://github.com/dan-pandori/nd-takehome/blob/aeb4655d/review_olmo-transplant.md)
(fork `dan-pandori/nd-takehome`, branch `dan_olmo-transplant`; §Recount, phase 1, blind,
[`e80ba754`](https://github.com/dan-pandori/nd-takehome/commit/e80ba754), 10:09:29 UTC; §Compare, corrections to
§Recount and §Verdict [`aeb4655d`](https://github.com/dan-pandori/nd-takehome/commit/aeb4655d), 10:50:48 UTC; kit
`review/olmo-transplant-recount/`, whose `README.md` lists every script and its main outputs). Run files at the reviewed commit
[`1b7fcebe`](https://github.com/dan-pandori/nd-takehome/commit/1b7fcebe):
- `run_olmo_transplant.md`, `numbers.md` § olmo-transplant and `log.md` § olmo-transplant;
- `olmo_transplant/`: the code, `prompts.csv`, `traces.csv` (the trace manifest: source job, chunk and index, ids and
  prefix sha256, `think_k`, `draft_k`), `results.csv` (one row per prompt × cell), `summary.md` / `summary.json`,
  `mechanism.md`, `compute.md`, `handcheck.csv` (425 items), `handcheck_src.csv` (75 calibration source items),
  `README.md` (the pipeline) and `plain_summary.md`. The figure is `figures/olmo_transplant.png`;
- the pre-registration `preregistration/olmo-transplant.md`
  ([`0fea68ab`](https://github.com/dan-pandori/nd-takehome/commit/0fea68ab), 05:35:16 UTC; a one-line header-time fix
  [`15a4522d`](https://github.com/dan-pandori/nd-takehome/commit/15a4522d), 05:35:28).

Brief: [`BRIEF_olmo-transplant.md`](../../docs/proposals/followups/BRIEF_olmo-transplant.md); proposal 40 in
[proposals 39–44](../../docs/proposals/2026-11-07-olmo-followups.md). The runs it builds on (both include these two models):
[olmo-ifomega](../2026-11-04-olmo-ifomega-reach-on-ifbench-49-strict-43-hand-and-ifeval-6-omega-test-out-inconclusive/README.md)
(the stored traces, the hand labels and the 43 hand-reach prompts) and
[olmo-iffu](../2026-11-06-olmo-iffu-brevity-hint-elicits-dpo-on-15-16-of-48-if-reach-prompts-10-of-10-survive-1024-draws-token-budget-on-1-of-3-geometry/README.md)
(the 6 required prompts, and the brevity hint of the review's next measurement 2).

Copies here:
- **the plain-language summary** (the brief's ≤ 500-word deliverable), with the review's corrections marked
  *[review]*: [`plain_summary.md`](plain_summary.md);
- **the figure**, unchanged; its corrections are listed under [Figures](#figures):
  [`olmo_transplant.png`](olmo_transplant.png).

Question (Dan, 2026-11-07, proposal 40, testing claim 4 of the Olmo synthesis): on IFBench, Olmo 3.1 Think 32B
succeeds on prompts where its DPO start has no genuine success in 256 samples (olmo-ifomega), and RL writes thousands
of tokens of thinking before short answers. Claim 4: RL learned a *search procedure* in its chain of thought, not a
better final-answer skill. If so, the DPO model handed RL's thinking should succeed, and RL's advantage should vanish
when its thinking is cut short.

## Configuration

**Models** (every number here is on one of these; the pods fetched these revisions, and the review checked them):

| label | checkpoint | what it is |
|---|---|---|
| `dpo` | `allenai/Olmo-3-32B-Think-DPO` `main` = `97604023` | 32 B, bf16, pretrained (Olmo 3 32B base → Think SFT → DPO); RL's start |
| `s2300` | `allenai/Olmo-3.1-32B-Think` `main` = `97ade57a` | `dpo` after 2,300 RL steps (= `step_2300`) |

**Decoding** as olmo-ifomega and olmo-iffu: `olmo_thinker` template (sha256 `eba6e269…`, the template that generated
the stored traces), T 1.0, top-p 1.0, EOS only, vLLM 0.11.0, bf16, prefix caching, one H200 per job (TP 1). Answer
budget = 32,768 − inserted thinking tokens (RL's response cap on thinking plus answer); every `length` finish is at the
budget. Natural-language output with no proofs, so Lean and `nd_verify` do not apply (the review: `nd_verify/`
unchanged, and no code of the run calls Lean or `nd_verify`).

**Prompts** (`olmo_transplant/select_prompts.py`, seed 20261108 → `prompts.csv`):
- **reach-41:** the olmo-ifomega review's 43 IFBench hand-reach prompts minus `ifbench_226` and `ifbench_252`, which
  have no `s2300` trace labelled genuine. Families (first instruction id): words 15, format 7, count 5, ratio 5,
  sentence 5, custom 2, repeat 2;
- **`brief20`:** 20 prompts nested in reach-41, the brief's seeded set. It includes the 6 prompts that survived 1,024 DPO samples
  and were not hint-elicitable in olmo-iffu (3, 136, 166, 211, 248, 269);
- **calibration (10):** IFBench prompts both models solve (not excluded, not a reach or reverse candidate, `dpo` ≥ 8 / 16
  and `s2300` ≥ 4 / 8 in olmo-ifomega's fixed phase: a pool of exactly 60 by the review's count), disjoint from reach.

**Traces** (stored, never regenerated; `traces.csv`, 328 rows): on reach, 115 `rl` traces (`s2300`'s hand-genuine
successes, min(4, available) per prompt) and 164 `dpo` failures; on calibration, 36 `rl` and 13 `dpo`. RL's reach
thinking has a median of 5,779 tokens (max 17,316); `dpo`'s, 2,786. In 92 of the 115 `rl` reach traces the final
answer's opening already appears in the thinking, at a median 56 % of the way through.

**Conditions** (8 answers per prompt × trace × condition; T0: 4 × 8 per prompt and writer):

| | thinking supplied | answer written by |
|---|---|---|
| T1, transplant | RL's full genuine thinking | `dpo` |
| T2, self | RL's full genuine thinking | `s2300` |
| T3, reverse | `dpo`'s failed thinking | `s2300` |
| T4, reverse-self (added; the control for T3) | `dpo`'s failed thinking | `dpo` |
| T0, none | empty (`\n</think>\n\n`) | each |
| B(N), budget | RL's thinking cut at N ∈ {1,024, 2,048, 4,096, 8,192, 16,384} tokens, then `</think>` | each |
| P, pre-draft (added) | RL's thinking cut just before the final answer first appears (92 of 115 traces; 38 prompts) | each |
| F(N), fresh forcing | `s2300` thinks afresh, capped at N ∈ {2,048, 4,096 (added), 8,192}, then `</think>` is forced | `s2300`, on the 6 required prompts × 16 |

- **The primary B curve** covers every trace: a trace whose thinking is ≤ N enters at its full-condition result. The
  pre-registered secondary curve covers only the traces actually cut (reach traces cut at 1k / 2k / 4k / 8k / 16k:
  112 / 109 / 92 / 20 / 1).
- **Seeds:** engine seeds 8101 (`dpo` writer), 8102 (`s2300` writer), 8103 (F, both phases; 8103 / 8104
  pre-registered), 8111 / 8112 (calibration jobs; not named in the pre-registration).

**Grades.** **Strict:** the official IFBench checker on the text after the last `</think>`
(`olmo_ifomega/grade.py`, unchanged). **Genuine** = strict × the hand gate's genuine share per (prompt, run-condition),
with identity inheritance (an answer equal to an already-labelled answer on the same prompt takes its label).
Calibration B and P cells are strict only, as registered.

**Hand gate.** Blinded Claude sub-agent readers see the prompt, its constraints with the checker's verdicts and the
answer, but not the thinking, the condition or the model; rules 1–6 (`olmo_ifomega/handcheck_rule_if.md` plus the
olmo-iffu addendum; rule 6's strict reading is reported too). Reads per (prompt, cell) are capped: T1 / T2 2;
T3 / T4 / T0 4; B / P 1; F 2; calibration T cells 1. A second reader reads every gamed label and a seeded 15 % of
genuine ones; splits go to a third, and the majority decides. The executor's gate: 425 run items (339 genuine,
86 gamed) and 75 calibration source items (60 genuine).

**Analysis.** A condition's rate is the mean over prompts (equal weight), with a 95 % prompt bootstrap (10,000
resamples). Paired contrasts by prompt: Δ21 = T2 − T1, Δ34 = T3 − T4, Δ0 = T0-`s2300` − T0-`dpo`, Δ30 = T3 −
T0-`s2300`; R1 = T1 / T2. Retained gain F_m(N) = (G_m(N) − G_m(0)) / (G_m(full) − G_m(0)), undefined below a
denominator of 0.10.

**Noise floor.** The pre-registration states a minimum detectable difference of ≈ 0.13 for a paired contrast on 41
prompts (assumed SD 0.30, 80 % power, α 0.05), ≈ 0.19 on `brief20`. The review's realised values (MDD = 2.80 SD / √n,
its labels) are in [Contrasts](#contrasts-paired-by-prompt). `NOISE_FLOOR.md` (seed variance of small `lean_seq`
models) does not apply: there is no training and no seed-level comparison.

**Pods and cost.** Two pods, both NVIDIA H200:
- `ot-1`, 2 × H200 COMMUNITY at $7.18/h (11-08 05:35:50 – 06:48:55 UTC; 05:36:07 in `~/pods.log`; jobs R_dpo and C_dpo): 1.218 h, $8.75. Every job
  ran on GPU 0; the second GPU ran no job (`numbers.md`: "one GPU dead");
- `ot-2`, 1 × H200 SECURE at $5.29/h (≈ 05:43 – 07:26:18; 05:43:40 in `~/pods.log`; R_s2300, F_s2300 and C_s2300): 1.715 h, $9.07.

Total **2.93 pod-hours (4.15 GPU-hours), $17.82** of the $60 budget (`podbudget`, booked hours × rate; the review
reproduces it). *[librarian]* RunPod billed **$17.95** with disk ($8.80 + $9.15 by `runpodctl billing pods`; billed
time equals the booked hours). The
pre-registration projected ≈ $8–20, and the stage-0 re-cost ≈ $12–15.

**Bucket** (public): `hf://buckets/dan-pandori/nd-rl/olmo-transplant/`:
- `artifacts/olmo_transplant/`: every answer's text and token ids (`samples/*.jsonl.gz`), grades, the requests with
  the inserted prefix ids, metas with prompt ids and boundary dumps, pod logs, and the hand items, keys and labels. The
  17 bulk files are listed with sha256 in the fork's `artifacts/MANIFEST.jsonl`;
- `artifacts/olmo-transplant/registry/`, `olmo_transplant/`, `pod/olmo_transplant/`, `figures/`, `preregistration/`
  and `run_olmo_transplant.md`.

The source traces stay in olmo-ifomega's folder (`olmo-ifomega/artifacts/olmo_ifomega/`). Registry rows:
`hf://buckets/dan-pandori/nd-rl/registry/olmo-transplant/` (168 compute and 204 headline rows). No weights were stored
off the pods. Recompute, from the fork's root: `python3 fetch_artifacts.py artifacts/olmo_transplant/`, plus the four
olmo-ifomega sample files and their graded files into `artifacts/olmo_transplant/src/` (`olmo_transplant/README.md`, "Source data"), then
that README's pipeline. The reviewer's order is in their kit's `README.md`.

## Observations (the review's numbers)

### The conditions (reach-41; mean over prompts, 95 % prompt bootstrap)

| condition | strict | genuine, executor's labels | genuine, the review's blinded labels |
|---|---|---|---|
| T2: RL's thinking → `s2300` | 0.935 [0.891, 0.971] | 0.924 [0.873, 0.966] | 0.922 [0.870, 0.964] |
| T1: RL's thinking → `dpo` | 0.139 [0.080, 0.211] | 0.139 [0.080, 0.211] | 0.138 [0.079, 0.211] |
| T3: `dpo`'s failed thinking → `s2300` | 0.101 [0.049, 0.162] | 0.060 [0.023, 0.110] | 0.060 [0.022, 0.111] |
| T4: `dpo`'s failed thinking → `dpo` | 0.001 [0.000, 0.002] | 0.000 | 0.000 |
| T0-`s2300`: no thinking | 0.112 [0.051, 0.188] | 0.078 [0.030, 0.138] | 0.085 [0.037, 0.146] |
| T0-`dpo`: no thinking | 0.003 [0.001, 0.006] | 0.001 [0.000, 0.002] | 0.001 [0.000, 0.002] |
| P-`s2300`: RL's thinking cut before its draft (38 prompts) | 0.511 [0.410, 0.614] | 0.453 [0.345, 0.565] | 0.458 [0.358, 0.564] |
| P-`dpo` (38 prompts) | 0.021 [0.010, 0.035] | 0.020 [0.009, 0.034] | 0.020 [0.009, 0.034] |

- With RL's full thinking, `dpo` is 0 on 20 of the 41 prompts (strict, and under both label sets).
- **Calibration** (10 prompts, strict): T1 0.988, T2 1.000, T0-`dpo` 0.713, T0-`s2300` 0.794; T3 0.769 and T4 0.656
  on the 5 prompts with a `dpo` failure. Given RL's thinking on prompts it can do, `dpo` answers almost as well as
  `s2300`.
- **Rule 6 and the sub-bullet reading:** the verdicts are the same under rule 6's strict reading and under a natural
  sub-bullet reading. With rule 1 applied mechanically (below), no reach-41 condition rate moves by more than 0.01.

### Contrasts (paired by prompt)

The review's labels; the executor's labels in parentheses where they differ. MDD is the review's realised value.

| contrast | reach-41 | MDD | `brief20` |
|---|---|---|---|
| R1 = T1 / T2 | 0.150 [0.086, 0.227] (0.151) | – | 0.130 [0.054, 0.219] |
| Δ21 = T2 − T1 | 0.783 [0.705, 0.853] (0.785) | 0.106 | 0.796 [0.693, 0.886] |
| Δ34 = T3 − T4 | 0.060 [0.022, 0.111] | 0.064 | 0.039 [0.003, 0.089] (0.038) |
| Δ0 = T0-`s2300` − T0-`dpo` | 0.085 [0.036, 0.144] (0.077 [0.030, 0.136]) | 0.077 | 0.056 [0.006, 0.134] (0.047) |
| Δ30 = T3 − T0-`s2300` | −0.025 [−0.061, 0.007] (−0.018) | 0.049 | −0.017 [−0.059, 0.025] (−0.009) |
| T1 − T0-`dpo` | 0.138 [0.079, 0.210] | 0.095 | 0.119 [0.048, 0.202] |

*[librarian]* Δ34's point value (0.060) is below its realised MDD (0.064) and the pre-registered ≈ 0.13, though its
interval excludes 0; Δ0 is above its realised MDD, as the review notes, but inside the pre-registered ≈ 0.13.

### The pre-registered readings (reach-41, genuine, the review's labels)

1. **"The capability is in the thinking"** (R1 ≥ 0.75 and T1 − T0-`dpo` ≥ 0.30): R1 = 0.150, T1 − T0-`dpo` = 0.138:
   **no**.
2. **"RL changed answer writing, not search"** (R1 ≤ 0.40, T3 ≥ 0.22 and Δ34 ≥ 0.15): R1 holds, but T3 = 0.060 and
   Δ34 = 0.060: **no**.
3. **"Partly in the thinking"** (0.40 < R1 < 0.75, or neither 1 nor 2): **holds, through its "neither 1 nor 2"
   clause**; R1 is far below the band.
4. **"The win needs long search":** F_s2300(2,048) = 0.299 (executor's labels 0.328; needs ≤ 0.25) and G_s2300(2,048)
   = 0.336 (needs ≤ T0-`dpo` + 0.10 = 0.101) fail; F_s2300(8,192) = 0.828 ≥ 0.60 and the fresh-forcing clause hold:
   **no**.
5. **"A disposition, not search"** (flat curves): T0-`s2300` = 0.085 against T2 − 0.10 = 0.822: **no**.
6. **"Graded"** (neither 4 nor 5): **yes**.
7. **P, draft or plan** (omitted by the run): on the executor's definition (G(0), G(P), G(full) over the 38 prompts
   with a P cell), F_dpo(P) = 0.146 ≤ 0.25, so for `dpo` "the success rests on the drafted answer" (0.073 if re-opened
   thinking counts as failure). F_s2300(P) = 0.454 (executor's labels 0.453): neither threshold as graded; 0.250
   (executor's labels 0.258) when re-opened thinking counts as failure, on the ≤ 0.25 copying threshold.

On `brief20`: R1 0.130, T3 0.039 (executor's labels 0.038), F_s2300(2,048) 0.282 (0.310), so readings 3 and 6 again.

### The budget curves

| N | G_s2300 strict | G_s2300 genuine (executor's / review's labels) | F_s2300 (review's labels) | G_dpo genuine (review's labels) | F_dpo (review's labels) |
|---|---|---|---|---|---|
| 1,024 | 0.269 | 0.209 / 0.208 | 0.147 | 0.029 | 0.203 |
| 2,048 | 0.392 | 0.355 / 0.336 | 0.299 | 0.037 | 0.266 |
| 4,096 | 0.587 | 0.530 / 0.518 | 0.517 | 0.061 | 0.435 |
| 8,192 | 0.793 | 0.780 / 0.778 | 0.828 | 0.124 | 0.893 |
| 16,384 | 0.931 | 0.919 / 0.917 | 0.995 | 0.138 | 1.000 |

This is the primary curve (every trace; a trace shorter than N enters at its full value). `dpo` never exceeds 0.14.
- **Cut traces only** (the pre-registered secondary curve; omitted by the run): `s2300` genuine (the review's labels) 0.19 / 0.32 / 0.43 /
  0.35 at 1k / 2k / 4k / 8k (40 / 40 / 37 / 11 prompts), against 0.92–0.93 for the same traces in full (0.88 at 8k);
  strict 0.25 / 0.38 / 0.51 / 0.38 against 0.93–0.94 (0.92 at 8k). The primary curve's strict 0.79 at 8k comes from
  the 95 of 115 traces that are not cut there.
- **By N / c**, the cut as a fraction of the trace's thinking (pre-registered "post hoc in bins"; omitted by the run):
  `s2300` strict 0.24 / 0.33 / 0.56 / 0.73 by quarter (pooled over trace × N points); `dpo` 0.004 / 0.007 / 0.034 / 0.048.
- **Re-opened thinking (post hoc).** The forced `</think>` does not cap `s2300`'s thinking. An answer that itself
  contains `</think>` means the writer kept reasoning after the supplied closing, closed again, and was graded on the
  text after the second closing. On reach this happens in 28.4 % of `s2300`'s P samples (209 / 736; median 2,431 extra
  tokens), in 7.8 / 8.8 / 3.7 / 10.0 % of its B1024 / B2048 / B4096 / B8192 samples (median 4,725–8,456 extra
  tokens), in 7.4 % of T0, 0 % of T2 and 0.6 % of T3. Among `s2300`'s strict successes, 152 / 397 in P, 42 / 245 in
  B1024 (17 %), 45 / 345 in B2048 (13 %) and 11 / 147 in T0 are re-opened answers. Counted as failures, the strict
  `s2300` curve is G 0.221 / 0.339 / 0.574 / 0.779 at 1k–8k (F 0.141 / 0.283 / 0.566 / 0.812), and fresh F(2,048)
  falls from 0.240 to 0.177. `dpo` never succeeds this way in B; in P, 8 of its 17 strict successes are re-opened
  answers. In the review's words, "'thinking capped at N' is not enforced by the forced closing for `s2300`; the P
  reading and the small-N end of the curve depend on how re-thought answers are counted."

### The draft

- **Condition P:** RL's thinking cut just before its final answer first appears gives `s2300` 0.453 genuine (the
  review's labels 0.458) and `dpo` 0.020; reading 7 is above.
- **The draft split** (success when the cut prefix contains the final answer's opening, against when it does not;
  strict, pooled over cuts): `s2300` 0.546 (n 808) against 0.390 (n 2,600), +0.16, on the executor's definition, which
  pools the P cuts into "without". Over B cuts only, 0.546 against 0.330 (+0.215; prompt-weighted +0.27; the review's
  labels, genuine, +0.26 / +0.29). `dpo`: +0.03 (+0.026 / +0.020, the review's labels, pooled / prompt-weighted).

### Fresh forcing (`s2300` thinks afresh, capped at N; 6 prompts × 16)

| N | strict | genuine, executor's labels | executor's labels, rule 1 mechanical | the review's labels (rule 1 mechanical) |
|---|---|---|---|---|
| 2,048 | 0.240 | 0.182 | 0.167 | 0.167 (0.167) |
| 4,096 | 0.427 | 0.427 | 0.354 | 0.318 (0.330) |
| 8,192 | 0.615 | 0.604 | 0.573 | 0.604 (0.573) |

- **As a share of `s2300`'s natural genuine rate on the 6 prompts** (0.685, from olmo-ifomega's draws): executor's
  labels 0.27 / 0.62 / 0.88; the review's labels 0.24 / 0.46 / 0.88; executor's labels with rule 1 mechanical
  0.24 / 0.52 / 0.84. Strict over the natural strict rate: 0.33 / 0.58 / 0.83 against all olmo-ifomega `s2300` draws
  (the executor's), 0.38 / 0.68 / 0.98 against the fixed phase (the review's expectations scoring; the pre-registration names no phase).
- **Why 4k moves:** IFBench's `format:sub-bullets` checker splits the answer on `*` and asks only for a `-` in each
  chunk, so an answer with no `*` passes, which is rule 1's own example of gaming. On the three sub-bullet prompts
  (128, 133, 136), 85 of `s2300`'s 411 reach strict successes, 17 of the 37 F successes and 4 of `dpo`'s 51 contain
  no `*`. Every one the executor's readers read was labelled gamed, but unread ones took their cell's share: on 136,
  7 of 14 F4096 and 3 of 12 F8192 successes, all unread, were counted genuine at share 1.0.

### Why the DPO start fails with RL's thinking

- **The executor's mechanism counts** (reproduced exactly): `dpo`'s T1 answers have a median of 126 words against
  26 for RL's own answers; 726 of 920 are more than 1.5× RL's length. 256 of the 920 (28 %) contain RL's drafted
  answer verbatim, and 183 of those fail strict, all of them longer than RL's answer; 43 (4.7 %) are exact copies.
  `s2300` in T2 copies exactly in 611 / 920.
- **The review's copy check** (post hoc; word 3-gram Jaccard ≥ 0.8 = near copy): `s2300` reproduces its trace's final
  answer exactly in 611 / 920 T2 samples (66 %) and nearly in 75 more; 681 of its 867 T2 strict successes (79 %) are
  exact or near copies. `dpo` in T1 copies exactly in 43 and nearly in 11; 53 of its 137 T1 strict successes (39 %) are
  exact or near copies, and 73 (53 %) share < 0.5 of their 3-grams with the trace's answer. The review: "given RL's
  thinking, `dpo` mostly writes its own answer rather than the one drafted."

### The hand gate: the executor's, and the review's blind repeat

- **The executor's protocol, as run:** 425 items (339 genuine, 86 gamed), all strict successes. Every first-reader
  "gamed" got a second reader, and 49 of 324 first-reader "genuine" (15.1 %); every split went to a third reader; no
  read cap was exceeded. Of 2,804 reach strict successes, 348 were read, 1,026 equal a labelled source answer, 261
  equal an answer read in this run and 1,169 are unread (their cell's share applies); 0 label conflicts. Rule 6's
  strict reading changes condition means by ≤ 0.002.
- **The review's readers:** 827 samples → 799 distinct items, selected by the reviewer from the raw texts: every
  reach strict success in the cells the readings rest on (T1, T3, T4, both T0s, `dpo`'s B and P) that is not a
  verbatim copy of a labelled source answer; every item the executor read in T2 and in `s2300`'s B / P cells; seeded
  samples of unread `s2300` successes; and the fresh-forcing items (every F2048 success, every F item the executor
  read, and 15 unread each at F4096 and F8192). Readers A on all 799, B on 270, C on the 21 splits;
  A and B agree on 249 / 270 (92 %). Final labels: 637 genuine, 162 gamed. The adjudication rules were written before
  any third read, and no majority was overridden.
- **Agreement:** on the 376 samples both read, 365 agree (97.1 %): 10 executor-genuine / review-gamed and 1 the other
  way. One item (T3 `ifbench_179`, alliteration) is a judgement call that moves T3 by < 0.001.
- **The share rule:** on 429 unread successes the review read, the executor's share rule predicted 346.3 genuine and
  the readers found 355 (0.807 against 0.828). It is close everywhere except fresh F4096 (15.0 of 15 predicted, 9
  found), T0-`s2300` (65.2 against 76) and R_dpo B4096 (10.0 against 14; F_dpo(4,096) moves from 0.404 to 0.435).
- **Inherited labels:** copies of a source answer carry olmo-ifomega's labels. olmo-ifomega's review re-read 78 of the
  115 reach `rl` trace labels; the other 37 rest on olmo-ifomega's readers alone.

### Compute per arm (all NVIDIA H200, TP 1)

| job (writer, set) | pod | wall GPU-s, every attempt (the review) | samples (= checker calls) | generated tokens | registry `gpu_seconds` |
|---|---|---|---|---|---|
| R_dpo (`dpo`, reach) | `ot-1` | 3,326 (513 killed stage-0 run + 12 failed start + 2,801) | 6,952 | 4,796,394 | 2,770 (omits the 525 s of killed and failed attempts) |
| R_s2300 (`s2300`, reach) | `ot-2` | 2,509 | 6,952 | 3,309,001 | |
| C_dpo (`dpo`, calibration) | `ot-1` | 446 | 1,640 | 823,570 | |
| C_s2300 (`s2300`, calibration) | `ot-2` | 1,646 | 1,640 | 1,631,474 | |
| F_s2300 (`s2300`, fresh) | `ot-2` | 1,153 | 288 | 1,275,080 (1,189,805 thinking + 85,275 answers) | |
| **total** | | **9,080 (2.52 GPU-h)** | **17,472** | **11,835,519** | 8,477.7 (2.35 h) |

- Training steps, training tokens and Lean checks are 0. Input tokens submitted: 2,735,529 per reach job, 581,278 per
  calibration job and 1,210,335 for F.
- Pod time is 4.15 GPU-hours, so set-up, downloads, `ot-1`'s idle second GPU (1.22 GPU-h) and gaps add ≈ 1.63 GPU-h
  outside every job's wall time, or 1.80 outside the registry's `gpu_seconds` (*[librarian]* arithmetic from the
  review's figures).
- Arms are matched on samples per (prompt, trace), 8, not on tokens; RL's thinking is supplied, not generated
  (pre-registration). *[librarian]* Generating it was olmo-ifomega's cost.
- *[librarian]* Flagged under the policy's 1.25× rule (unreviewed arithmetic from the executor's per-arm registry rows,
  whose per-job generated-token sums equal the review's; `gpu_seconds` there is apportioned by generated tokens within a
  job). These are answer lengths (outcomes), not inputs: arms are matched on samples.
  - **Reach jobs, generated tokens:** T1 (`dpo`) used 4.58× T2 (354,230 against 77,277), T4 (`dpo`) 2.46× T3, `dpo`'s
    B cuts 1.53× `s2300`'s, and `s2300`'s P 1.26× `dpo`'s. Within one writer, T0-`s2300` used 1.95× T3 (1,312 samples
    each) and T0-`dpo` 1.81× T1 (1,312 samples against 920; 1.27× per sample).
  - **Reach jobs, apportioned GPU-seconds:** T1 3.50× T2, T4 1.88× T3, T0-`s2300` 1.28× T0-`dpo`, and `s2300`'s P
    1.65× `dpo`'s; the B cuts are 1.17×, under the line.
  - **Calibration jobs:** in generated tokens, `s2300`'s P used 5.11× `dpo`'s and its B cuts 1.98×; `dpo`'s T4 used
    2.44× `s2300`'s T3, its T0 1.53× and its T1 1.35× `s2300`'s T2. In GPU-seconds, `s2300`'s P is 9.84×, B 3.82×, T2
    1.42× and T0 1.26×, and `dpo`'s T4 1.27×.
  - In the reach token flags for T1 / T2, T4 / T3 and the B cuts, the arm that spent more has the lower success rate;
    in P and in T0's GPU-seconds, the `s2300` arm spent more and has the higher rate.

## Expectations against outcomes

Binding expectations were committed before any pod (`0fea68ab`, 05:35:16 < first pod 05:35:50 by the executor's log,
05:36:07 in `~/pods.log`). The review's scoring with its labels (reach-41 unless named):

| quantity | expected (range) | observed | verdict |
|---|---|---|---|
| T1 | 0.55 (0.30–0.80) | 0.138 | **below** |
| R1 = T1 / T2 | 0.70 (0.45–0.95) | 0.150 | **below** |
| G_dpo(4,096) | 0.25 (0.10–0.40) | 0.061 | **below** |
| G_dpo(8,192) | 0.45 (0.30–0.60) | 0.124 | **below** |
| draft in cut prefix − not, `dpo` | ≥ 0.30 (0.15–0.60) | 0.026 pooled, 0.020 prompt-weighted | **below** |
| answers cut at the budget | < 1 % in every condition | three calibration `s2300` cells above 1 %: P 2.7 %, B4096 2.9 %, B8192 3.1 %; every reach cell ≤ 0.27 % | **miss** |
| G_s2300(2,048) | 0.20 (0.05–0.35) | 0.336 (executor's labels 0.355) | inside with the review's labels; **above** with the executor's |
| draft in cut prefix − not, `s2300` | ≥ 0.30 (0.15–0.60) | 0.255 pooled, 0.288 prompt-weighted (executor's definition +0.16) | inside the range |
| T2 | 0.80 (0.65–0.95) | 0.922 | inside |
| T3 / T4 | 0.08 (0.01–0.25) / 0.02 (0.00–0.08) | 0.060 / 0.000 | inside |
| T0-`s2300` / T0-`dpo` | 0.08 (0.00–0.25) / 0.02 (0.00–0.06) | 0.085 / 0.001 | inside |
| G_s2300 at 1k / 4k / 8k | 0.12 / 0.40 / 0.70 (± 0.15) | 0.208 / 0.518 / 0.778 | inside |
| G_dpo at 1k / 2k | 0.05 / 0.10 (± 0.15) | 0.029 / 0.037 | inside |
| F_s2300(2,048) | 0.20 (0.05–0.45) | 0.299 | inside |
| F_s2300(P) / F_dpo(P) | 0.35 / 0.25 (0.10–0.70 each) | 0.446 / 0.140 (G(0) and G(full) over all 41 prompts; 0.454 / 0.146 on the executor's 38-prompt definition) | inside |
| fresh F(2k) / F(4k) / F(8k) as a share of the natural strict rate (fixed phase) | 0.3 / 0.6 / 0.9 (± 0.25) | 0.383 / 0.683 / 0.983 | inside |
| calibration T1 / T2 / T0-`dpo` / T0-`s2300` (strict) | 0.90 / 0.90 / 0.60 / 0.60 (± 0.20) | 0.988 / 1.000 / 0.713 / 0.794 | inside |

The pre-registered most likely outcome was reading 3 or 1 (P(1) ≈ 0.45, P(3) ≈ 0.40, P(2) ≈ 0.15), together with 4 or
6; the outcome is 3 (through its "neither" clause) and 6. The write-up reports the misses it found (T1, R1, `dpo`'s
curve at 4k / 8k, `dpo`'s draft effect, RL at 2k with its own labels, the calibration cuts). It lists RL's draft effect
(+0.16) next to them although it is inside its range, and it under-counts the calibration cut cells.

## The review's corrections to the executor's files

The fork's files are uncorrected; the copy of the plain summary here applies its corrections. By file:
- **`run_olmo_transplant.md`:** finding 3's "0.27 / 0.62 / 0.88 of its natural rate" (4k: 0.46–0.52 genuine or 0.58
  strict; 8k 0.84 with rule 1); finding 4's "The draft helps but is not required" needs reading 7's qualifier, and
  reading 7 is missing; "RL's thinking is necessary but not sufficient" becomes "carries most of the gain";
  "Two calibration cells had > 1 % … (2.7 %, 1.4 %)" becomes three cells, 2.7 / 2.9 / 3.1 %; "Above: RL at 2k" holds
  with the executor's labels only (0.355 against 0.336); RL's draft effect (+0.16) is inside its range;
  "only answers were sampled (17,472)" holds for 17,184 of them (the 288 fresh-forcing samples had their thinking
  sampled too); the cut-only curve and N / c bins are missing; the prose names re-opened thinking for P only, although
  8–9 % of B1k / B2k answers re-open it and carry 17 % / 13 % of the strict successes; the seed deviations are in
  the log, not the deviation line. The write-up names the models without revision hashes (`numbers.md` has them).
- **`numbers.md` § olmo-transplant:** the fresh-forcing genuine ratio at 4k (0.62 [0.37, 0.86]); "calibration up to
  2.7 % (`s2300` P 5 / 184, B4096 4 / 288)" (B4096 is 4 / 136 = 2.9 %, and B8192 2 / 64 = 3.1 %); "2.35
  H200-hours in jobs" (2.52 GPU-h); the draft covariate pools P cuts into "without".
- **`olmo_transplant/results.csv`:** the 118 calibration B / P rows with strict successes carry a `genuine_rate`
  although these cells are strict only by design; in 75 it is below the strict rate (e.g. `ifbench_115` B1024-`dpo`:
  strict 1.0, genuine 0.0, 32 unlabelled). No write-up uses them. Blank them.
- **Registry rows:** R_dpo's `gpu_seconds` (2,770 s) is 16.7 % below its 3,326 s of job wall time; all 96 `contrast`
  rows have `ckpt` empty, including single-writer retained gains; the 168 compute rows name the writer only through
  `arm`.
- **`olmo_transplant/plain_summary.md`:** "The thinking is necessary" (most of the gain, not all); "never succeeded
  genuinely in 256 tries" is olmo-ifomega's number, unattributed; "often copies it" fits "contains" (28 %), not exact
  copies (4.7 %); "Fresh thinking under a hard cap looks similar" holds as a share of the natural rate, not in absolute
  rates. Applied in the copy here.
- *[librarian]* The same figures sit in `olmo_transplant/summary.md` (line 188: the fresh-forcing ratio "0.62 [0.37,
  0.86]") and `olmo_transplant/compute.md` ("2.35 H200-hours in jobs"); `numbers.md`'s "reach 0–0.22 % per cell"
  uses the same primary-curve denominator, and the review gives the reach maximum as 2 / 736 = 0.27 % (B4096-`dpo`).
- **`STATUS.md`:** the result line's "thinking necessary, not sufficient" (the review's #13 cites it with the run
  file). *[librarian]* The same figures sit in `log.md`'s 08:19 entry ("fresh F / natural (genuine) 0.27 / 0.62 /
  0.88") and its 08:21 entry ("two expectation misses (… calibration length cuts 2.7 % / 1.4 % > 1 %)").

## Deviations from the pre-registration and the brief

- **All 41 eligible reach prompts**, not the brief's ≈ 20; `brief20` is nested and agrees (pre-registered).
- **Added conditions** T4 (the control for T3), P (the pre-draft cut) and F(4,096) (pre-registered).
- **Seeds:** F used one engine seed (8103) for both phases, against the pre-registered 8103 / 8104, and the calibration
  jobs used 8111 / 8112, which the pre-registration does not name. The review: "Neither changes the sampling
  distribution." Disclosed in the log, not in the write-up's deviation line.
- **R_dpo's restart:** killed after its stage-0 part (513 s; 1,384 rows kept, 0 partial requests dropped) and resumed
  with 24 trace groups in flight instead of 8, under the same engine seed.
- *[librarian]* **The hand gate** reads a capped sample per (prompt, cell) and applies its share to unread successes,
  as pre-registered, rather than the brief's "every success that a condition-level conclusion rests on". The review
  does not call this a deviation; its share test is above.

## Interpretation (labelled; not tested further here)

- **The review's reading:** RL's thinking "carries most of the gain, not all of it". Without any thinking `s2300`
  still reaches 0.08 genuine where `dpo` has 0.001, and Δ0 is above its realised MDD. The graded rise of the primary
  curve at 4k–8k "is mostly uncut traces entering at their full value", and the P reading and the small-N end of the
  curve depend on how re-opened thinking is counted.
- **The executor's reading** (plain summary): `dpo` "adds an introduction or an explanation … and the extra text
  breaks the rule", and "RL's gain looks like part answer-writing discipline and part longer, more careful thinking."
  The review's copy check: "given RL's thinking, `dpo` mostly writes its own answer rather than the one drafted."
- **Claim 4** (RL learned a search procedure in its chain of thought, not a better final-answer skill): reading 1
  ("the capability is in the thinking") fails at R1 = 0.15, and reading 4 ("the win needs long search") fails.
- *[librarian]* olmo-iffu's 60-word hint made `dpo` prompt-elicitable on 12 of 42 IFBench reach prompts; whether the
  same hint closes the T1 gap is the review's next measurement 2, not tested here.

## Limitations

- **Readers are Claude sub-agents, not people.** Unread successes take their cell's genuine share, which carried
  vacuous sub-bullet answers at F4096; 37 of the 115 `rl` trace labels rest on olmo-ifomega's readers alone.
- **Selected traces.** RL's thinking is from its genuine successes, so T2 (0.92) overstates `s2300`'s everyday rate on
  these prompts, ≈ 0.44 (0.604 × 0.73, from olmo-ifomega; the plain summary's estimate, reproduced by the review).
- **"Capped at N" is not enforced** for `s2300` (above); the P reading and the small-N end of the curve depend on it.
- **Scope:** 41 prompts, one benchmark, one RL run; family means (n = 2–15 prompts) are descriptive.
- **Calibration** B / P cells are strict only, and the calibration genuine columns rest on ≤ 1 executor read per
  (prompt, cell) and on inheritance from the 75 calibration source labels, which the review did not re-read.

## Next steps (the review's next measurements; not launched)

1. **Enforce the budget:** re-sample the B, P and F answers with a second `</think>` banned (or generation stopped at
   it), so that "thinking capped at N" holds, and re-read P under that.
2. **Test the T1 mechanism directly:** T1 with olmo-iffu's brevity instruction ("answer in at most 60 words"), or with
   RL's drafted answer supplied as the answer's first line: "if `dpo` then approaches T2, the transplant gap is
   answer-writing discipline, not missing search."
3. **Close the sub-bullet hole:** read (or rule-1 filter) every unread strict success on the three sub-bullet prompts,
   "so the share rule cannot carry vacuous answers".

*[librarian]* The review gives no costs. This run's 17,472 samples (17,184 answers after supplied thinking and 288 fresh-forcing samples) cost $17.82, so measurements 1 and 2,
which re-sample subsets of its conditions, are likely of that order (unreviewed). Measurement 3 needs no pods. Proposal
41 (an elicitation ladder whose first rung is the same brevity hint) overlaps measurement 2. Nothing is queued after
this run on agent2.

## Figures

[`olmo_transplant.png`](olmo_transplant.png) (the executor's figure, unchanged): (a) genuine success per condition and
writer, (b) the primary B curves with fresh forcing. It needs these corrections:
- **Fresh forcing (b):** the F4096 point (0.43, the executor's labels) is inflated by unread vacuous `ifbench_136`
  answers (7 of the 14 F4096 successes there have no `*`). It is 0.35 with the executor's labels and rule 1 applied
  mechanically, and 0.32 with the review's labels (0.33 with rule 1). F8192 (0.60) is 0.57 with rule 1.
- **The cut-only curve is missing (b).** The plotted curve is the primary one ("a trace shorter than N enters as full",
  in the figure's own footnote); on the traces actually cut, `s2300` is 0.19–0.43 at 1k–8k.
- **The "thinking budget N" is not enforced (a, b):** 28 % of P and 8–9 % of B1k / B2k answers re-open thinking. Counted
  as failures, the P bar for `s2300` (0.45) is 0.28 (the executor's labels, post hoc).
