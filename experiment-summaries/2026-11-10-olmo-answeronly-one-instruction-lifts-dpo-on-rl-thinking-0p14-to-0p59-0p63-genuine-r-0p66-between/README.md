---
written_on: 2026-11-10
written_by: agent:claude
---

# olmo-answeronly: given the RL model's genuine thinking, one plain instruction lifts Olmo 3.1 Think 32B's DPO start from 0.14 to 0.59–0.63 genuine on 41 IFBench reach prompts (R = 0.66–0.68 of RL's rate): pre-registered reading "between" habit and skill

Status: completed 2026-11-10 (DONE 02:15:32 UTC) and reviewed 2026-11-10 (§Recount, phase 1, blind, 02:36:48 UTC;
final 02:39:17 UTC). **No hard-constraint violation, nothing quarantined.** The review's verdict is **"Stands"**, and
under "Not supported" it writes "nothing in the write-up's findings". Six items must be reworded (below); the main one
is that the absolute genuine levels depend on the readers, by about 0.08 for the RL model, while the ratio R does not.

The models are `dpo` = `allenai/Olmo-3-32B-Think-DPO` `main` (`97604023`; the RL run's start) and `s2300` =
`allenai/Olmo-3.1-32B-Think` `main` (`97ade57a` = `step_2300`), 32 B, bf16, pretrained (Olmo 3 32B base → Think SFT
→ DPO → RL), and `lora_main` = `dpo` + olmo-ifladder's rank-16 LoRA on 16 genuine `s2300` trajectories of 16 other
IFBench prompts (adapter sha256 `9921a760…`; olmo-ifladder's review: "Stands"). Natural-language chat, not `lean_seq`;
no proofs, so neither Lean nor `nd_verify` is involved. "Genuine" = the official IFBench strict checker plus a blinded
hand gate (Claude readers). Every number is on the 41 IFBench reach prompts unless it says calibration.

Review: [`review_olmo-answeronly.md`](https://github.com/dan-pandori/nd-takehome/blob/93d36de7/review_olmo-answeronly.md)
(fork `dan-pandori/nd-takehome`, branch `dan_olmo-answeronly`; §Recount, phase 1,
[`b0828c12`](https://github.com/dan-pandori/nd-takehome/commit/b0828c12), 02:36:48 UTC; §Compare and §Verdict
[`93d36de7`](https://github.com/dan-pandori/nd-takehome/commit/93d36de7), 02:39:17 UTC; kit
`review/olmo-answeronly-recount/`). Run files at the reviewed commit
[`49c08ca2`](https://github.com/dan-pandori/nd-takehome/commit/49c08ca2):
- `run_olmo_answeronly.md`, `numbers.md` § olmo-answeronly, `log.md` § olmo-answeronly;
- `olmo_answeronly/`: the code (`build_requests.py`, `build_lora_requests.py`, `sample_prefix.py`, `tok_enc.py`,
  `analyze.py`, `handcheck.py`, `select_reads.py`, `mechanism.py`, `compute_rows.py`, `figures.py`,
  `numbers_section.py`), the exact wordings and token ids `wordings.py`, `lora_traces.csv`, `results.csv`,
  `summary.json`, `handcheck.csv`, `handcheck_prompt.md`, `handcheck_rules.md`, `mechanism.md`, `compute.md`,
  `claim_check{,_second}.md`, `README.md`, `plain_summary.md`; the figure `figures/olmo_answeronly.png`;
- the pre-registration `preregistration/olmo-answeronly.md`
  ([`e2e646f3`](https://github.com/dan-pandori/nd-takehome/commit/e2e646f3), 00:32:14 UTC, 32 s before the pod; the
  amendment [`765702e8`](https://github.com/dan-pandori/nd-takehome/commit/765702e8), 00:44:42, after stage 0, changes
  only the header time and adds the request-file hashes, which match the files used).

Brief: [`BRIEF_olmo-answeronly.md`](../../docs/proposals/followups/BRIEF_olmo-answeronly.md); proposal 45,
[`2026-11-09-olmo-transplant-answer-only.md`](../../docs/proposals/2026-11-09-olmo-transplant-answer-only.md). It builds
on [olmo-transplant](../2026-11-08-olmo-transplant-dpo-given-rl-thinking-0p14-vs-rl-0p92-genuine-on-41-ifbench-reach-prompts-readings-partly-and-graded/README.md)
(the 41 prompts, the 115 reach and 36 calibration traces, the T1 / T2 baselines, the insertion and grading code),
[olmo-ifomega](../2026-11-04-olmo-ifomega-reach-on-ifbench-49-strict-43-hand-and-ifeval-6-omega-test-out-inconclusive/README.md)
(the stored samples, the hand rules) and
[olmo-ifladder](../2026-11-09-olmo-ifladder-lora-on-16-rl-trajectories-elicits-40-41-of-48-if-reach-prompts-control-0/README.md)
(`lora_main` and its stored evaluation draws).

Copies here:
- **the plain-language summary** (the brief's ≤ 500-word deliverable), with the review's corrections marked
  *[review]*: [`plain_summary.md`](plain_summary.md);
- **the figure**, unchanged: [`olmo_answeronly.png`](olmo_answeronly.png). It shows the executor's genuine rates
  (`s2300` ≈ 0.92–0.95); the review's readers give `s2300` 0.83–0.86 and `dpo` A1 0.59 (table below).

Question (Dan, 2026-11-09, proposal 45): olmo-transplant gave `dpo` `s2300`'s full genuine thinking and let it write
only the answer: genuine 0.14 against `s2300`'s 0.92 on the same thinking, with `dpo`'s answers a median of 126 words
against 26, and 28 % containing RL's drafted answer verbatim but failing by adding words. "Is the gap a writing habit
that one instruction removes, or can `dpo` not use RL's thinking at all?"

## Configuration

- **Thinking:** olmo-transplant's 115 stored genuine `s2300` traces on the 41 reach prompts (`traces.csv`) and 36
  traces on 10 calibration prompts, inserted by stored token ids, not regenerated. Each condition adds exactly one
  registered change (`olmo_answeronly/wordings.py`):

  | | change |
  |---|---|
  | A1, instruction | appended to the user turn: "After thinking, reply with only your final answer, exactly as drafted, with no introduction or explanation." |
  | A2, brevity | appended to the user turn: olmo-iffu's "Answer in at most 60 words." |
  | A3, closing thought | a line before `</think>` in the inserted thinking: "I will now output exactly the final draft above and nothing else." (13 ids) |
  | A4, prefill | after `</think>\n\n`, the first min(5, n − 1) tokens of the trace's own final answer; graded on prefill + continuation |

- **Writers:** both `dpo` and `s2300`, 8 answers per prompt × trace (920 per reach cell, 288 per calibration cell).
  T1 (`dpo`) and T2 (`s2300`) with no change are olmo-transplant's rows, not re-sampled.
- **Optional LoRA arm** (run: stage 0 re-costed the required design at ≈ 1 h on `oa-1`, ≈ $7, under the $22 gate): L1 = `lora_main` writing on
  `s2300`'s thinking (all 41 prompts; none is a source prompt of its training manifest); L2 = `dpo` writing on
  `lora_main`'s own hand-genuine thinking from olmo-ifladder's stored draws (116 traces, 35 prompts); L3 = `lora_main`
  on the same thinking.
- **Decoding:** `olmo_thinker` template (one `template_sha256` in all eight meta records), T 1.0, top-p 1.0, budget
  32,768 minus the supplied tokens, vLLM 0.11.0, one H200 (TP 1), the LoRA through vLLM's LoRA path; engine seeds 8201 /
  8202 / 8211 / 8212 / 8221–8223. The writers differ only in weights.
- **Grading:** IFBench official strict (`olmo_ifomega/grade.py`, upstream code byte-identical); genuine = strict × the
  blinded hand gate's genuine share per (prompt, writer, condition), with identity inheritance (`handcheck.csv`: 478
  items read; 441 genuine / 37 gamed). Copies: exact (whitespace-collapsed) and near (word 3-gram Jaccard ≥ 0.8)
  against the trace's own stored answer.
- **Analysis:** rates are means over prompts with a 95 % prompt bootstrap (10,000, seed 1). R_c = mean `dpo` genuine ÷
  mean `s2300` genuine on S_c, the prompts where condition c keeps `s2300` ≥ 0.75. The best of A1–A3 (c\*) is chosen
  on calibration, by `dpo` strict with ties (≤ 0.03) broken by the copy share.
- **Pre-registered readings** (on c\*): R ≥ 0.75 "habit"; R ≤ 0.40 "skill"; between: by constraint family; A4 ≥ 0.75
  with A1–A3 ≤ 0.40: "the habit sits in the first few tokens"; a condition with `s2300` < 0.75 on more than half the
  prompts is "broken" and not read.
- **Pods and cost:** one pod, `oa-1` (`z3efysw7fp9jmc`), 2 × H200 COMMUNITY at $7.18/h, 2026-11-10 00:32:46 – 01:49:55
  UTC, 1.2914 h, **$9.27** of the $30 budget (≈ $9.35 with disk; see [spend](../../docs/infra/spend.md)). GPU 1 was
  unusable ("CUDA-capable device(s) is/are busy or unavailable"; ECC in `log.md`), so every job ran in sequence on GPU 0
  and about half the billed GPU-hours did no work (≈ $4.6 on the pre-registered 1 × H200).
- **Bucket:** `hf://buckets/dan-pandori/nd-rl/olmo-answeronly/`: `artifacts/olmo_answeronly/{requests,samples,graded}/`
  (every answer's text and token ids; `artifacts/MANIFEST.jsonl`), `meta/`, `logs/`, `hand/` (kit and labels),
  `olmo_answeronly/`, `pod/olmo_answeronly/`, registry rows, `docs/`. The review checked the sample files' sizes against
  the local copies (7 / 7).

## Observations

### What the review checked (phase 1, blind, own code)

- **Grades:** all 12,440 new rows and the 17,184 inherited olmo-transplant rows re-graded with upstream
  `allenai/IFBench` `evaluation_lib` directly: 0 strict differences (0 loose on the new rows).
- **Inputs:** 12,440 of 12,440 input sha256 reproduce from a hand-rendered chat template plus the stored trace ids plus
  exactly the registered change, in every condition. Generated ids decode to the stored text (A4: prefill + continuation).
  Every prefix ends after its first and only `</think>`; the template boundary was read by hand for each condition.
- **Own blind hand gate:** 476 distinct items (every distinct `dpo` A1 and T1 success, 80 sampled successes of each
  other main cell, 60 of L1 and L2), 16 Claude readers with the run's reader prompt verbatim, second reads on every
  gamed label and 15 % of genuine ones, third reads on splits. Final 444 genuine / 32 gamed. On the 115 texts both
  gates labelled, they agree on 109 (95 %).
- **Blindness.** The review discloses that its auto-memory index showed one line on this run at 02:16 ("one instruction
  lifts dpo on RL thinking 0.14→0.63 (R 0.68 between)"), and lists the executor files it did not open in phase 1.
  *[librarian]* The librarian's check of the reviewer's transcript (205 tool calls before 02:36:48) found no read of
  the executor's write-ups, `numbers.md`, `results.csv`, `summary.json`, `handcheck.csv`, the memory files or the
  figure, and no headline number in any tool result before the review's own write.

### Results (the review's labels; the executor's in brackets where they differ)

| reach-41 | `dpo` strict | `dpo` genuine [95 %] | `s2300` genuine | R (on S_c, \|S_c\|) [95 %] |
|---|---|---|---|---|
| T1 / T2 (olmo-transplant) | 0.139 | 0.128 [0.071, 0.194] (0.14) | 0.831 (0.92) | 0.157 (31) [0.082, 0.247] (0.159, 37) |
| **A1 instruction = c\*** | 0.633 | **0.593** [0.502, 0.680] (0.626) | 0.848 (0.927) | **0.663** (30) [0.546, 0.769] (0.683 [0.584, 0.778], 35) |
| A2 brevity | 0.325 | 0.303 (0.324) | 0.827 (0.918) | 0.376 (31) [0.250, 0.505] (0.373) |
| A3 closing thought | 0.316 | 0.298 (0.312) | 0.857 (0.951) | 0.347 (34) [0.241, 0.458] (0.336) |
| A4 prefill | 0.223 | 0.216 (0.223) | 0.854 (0.944) | 0.257 (33) [0.166, 0.355] (0.253) |

- **c\* = A1**, by the registered rule (calibration `dpo` strict A1 0.997, A3 0.981, a tie; copy share 0.150 against
  0.101). Both gates agree.
- **Reading: "between"** (0.40 < R_A1 < 0.75). The interval's upper end (0.769) crosses 0.75, so "habit" is not
  excluded at this n; "skill" (≤ 0.40) is. Reading 4 does not hold (R_A4 = 0.26). No condition breaks `s2300` (at most
  11 of 41 prompts below 0.75 on the review's labels, 6 on the executor's; the limit is 20). Calibration strict stays
  0.947–0.997 for `dpo` and 0.975–1.000 for `s2300` in every condition.
- **Paired differences by prompt** (41 prompts, pre-registered MDE ≈ 0.13), review's labels: `dpo` A1 − T1 =
  **+0.465** [+0.383, +0.548] (executor +0.487 [0.406, 0.568]; the review labels T1 and A1 with the same readers);
  A2 − T1 +0.175, A3 − T1 +0.170, A4 − T1 +0.088 (inside the MDE). `s2300`'s differences from T2 are all inside the
  floor (−0.004 to +0.026).
- **"Closes about 60 % of the gap":** (0.663 − 0.157) / (1 − 0.157) = 0.60 (review).
- **Copies:** `dpo` copies the trace's answer exactly in 35.4 % of A1 answers (T1 4.7 %), and exact + near in 39.4 %
  (T1 5.9 %); median 49 words (T1 126; `s2300` 26). Genuine 0 on 2 prompts with A1 against 20 with T1 (A2 10, A3 9,
  A4 17), where 0 means 0 genuine in that prompt's 8–32 answers (8 per trace × 1–4 traces).
- **A4: does the writer stop where the draft stops?** `dpo`: exact copy 0.098, continues past the draft 0.302, diverges
  0.600; `s2300`: 0.711 / 0.024 / 0.265.
- **By constraint family** (descriptive, n = 2–15 prompts; review's labels, `dpo` A1 / T1 / `s2300` A1): words (15)
  0.57 / 0.09 / 0.88, format (7) 0.52 / 0.21 / 0.74, count (5) 0.60 / 0.07 / 0.94, ratio (5) 0.65 / 0.13 / 0.85,
  sentence (5) 0.44 / 0.06 / 0.70, custom (2) 1.00 / 0.06 / 0.97, repeat (2) 0.81 / 0.50 / 1.00.
- **LoRA arm (reading 6):** L1 `lora_main` on `s2300`'s thinking 0.640 genuine (executor 0.684); R_L1 = L1 ÷ T2 =
  0.777 on 31 prompts (executor 0.752 [0.653, 0.843]): **at the threshold**, from one LoRA and one draw of answers, not
  clearly above it. L2 `dpo` on `lora_main`'s thinking 0.179 against T1 0.146 on the same 35 prompts, +0.03, far below
  the registered +0.13: not easier for `dpo` to use than `s2300`'s thinking. L3 `lora_main` on its own thinking 0.732
  (executor 0.773).
- **Cuts and re-opened thinking:** answers cut at the budget ≤ 0.35 % in every cell (they are ≈ 28k-token runaway
  answers); re-opened answers (containing `</think>`) ≤ 0.54 % per A condition (T1 0.76 %, L2 1.08 %), and none of them
  is a strict success, so "re-opened = failure" changes no rate.

### The review's label differences (§Compare)

- **`s2300`'s own drafts.** The executor gave exact copies of a trace's draft that draft's inherited olmo-ifomega label
  (genuine). The review's readers judged 7 of 82 of these drafts gamed (an alphabet-chain answer on `ifbench_220`,
  keyword-quota answers on 203 / 206 / 209, unpunctuated answers on 179 / 188). Because the drafts are copied many
  times, this lowers `s2300`'s level from ≈ 0.93 to ≈ 0.85 and shrinks S_c from ≈ 37 to 30–34 prompts; `dpo`'s copies
  move the same way, so R barely moves.
- **Inline sub-bullets** (`ifbench_128`, 133, 136). The upstream `format:sub-bullets` checker passes any `-` after
  each `*`. The executor ruled inline ` - ` sub-points genuine (rule fixed at 01:57 during the gate, dated and
  disclosed); the review ruled them gamed mechanically (the checker's docstring asks for a line starting with `-`).
  Both readings are defensible; R does not depend on it (executor's drop-both-prompts sensitivity: R_A1 0.669; on the
  review's labels both prompts already fall outside S_c).

### Must be reworded (the review's §Verdict, items 1–6)

1. `numbers.md`'s R table, L2 row: "(0.29 on the same 35 prompts, T1 0.160)" reads as if L2 were 0.29. It should read
   "L2 0.18 vs threshold T1 + 0.13 = 0.29 on the same 35 prompts".
2. The same table's "reading" column gives A2 / A3 / A4 as "skill". The readings are registered for c\* only, and
   R_A2's interval crosses 0.40: say "R ≤ 0.40".
3. "`dpo` A1 above its range (0.63 vs 0.15–0.60)": it sits at the boundary, and the review's readers give 0.593
   (inside).
4. The per-prompt zeros ("genuine 0 on 2 prompts, T1: 20") need their tries: 8–32 answers per prompt.
5. **Genuine levels are reader-dependent by about ±0.08 for `s2300`** (0.92–0.95 on inherited labels, ≈ 0.85 re-read
   blind). R and every paired difference survive; quote the absolute levels with that caveat.
6. Cite the RL-data overlap scan for the inherited prompts. *[librarian]* The 41 reach prompts are a subset of
   olmo-ifomega's headline IFBench set, from which that run excluded every prompt its contamination scan matched against
   all 102,026 `Dolci-Think-RL-32B` prompts (and the DPO and SFT prompts; reviewed there).

### Expectations against results (pre-registered, review's values)

All inside their ranges except one miss: `dpo`'s A4 exact-stop share, 0.098 against 0.10–0.50 (`s2300`'s 0.711 is
inside 0.45–0.85). `dpo` A1 genuine is 0.593 on the review's labels (inside 0.15–0.60, at the top; point expectation
0.35) and 0.626 on the executor's (a miss, reported as one). R_c\* 0.663 (expected 0.40, range 0.20–0.70); most likely
reading 3, "between" (P ≈ 0.5): hit. On the executor's per-(prompt, condition) accounting, the "< 1 % in every cell" cut
bound and the "< 5 % every A cell" re-opened bound miss in 4 and 3 cells (listed in `numbers.md`); they hold per
condition.

### Compute (review §R7, executor `compute.md`; registry rows `artifacts/olmo-answeronly/registry/`)

12,440 samples, 2,068,854 generated tokens, 1.12 H200-h of load + generation in jobs (4,111 s of worker wall time), of
2.58 billed GPU-h; no training, no Lean or environment checks beyond the IFBench checker. Arms are matched on answers
per (prompt, trace) (8), not on tokens. Generated tokens per reach answer: `dpo` T1 385, A1 122, A2 148, A3 274, A4
354, L2 347; `s2300` 75–84; `lora_main` 116–117. Flagged under the 1.25× rule: `dpo` generates 1.5–4.5× `s2300`'s tokens
on A1–A4 (T1 / T2 4.6×), L1 / T2 1.39×, L2 / L3 3.0×. Every `dpo` A condition uses fewer tokens than T1.

## Limitations and next step

- 41 prompts selected because RL beats the start there; the inserted thinking is selected from `s2300`'s successes.
- "Genuine" is judged by Claude readers; two reader sets differ by ≈ 0.08 on `s2300`'s level (not on R). Reader
  noise is large on keyword-stuffed or alphabet-chain fragments and on inline sub-bullets (second readers confirmed 20
  of 39 gamed labels).
- R_A1's interval [0.546, 0.769] reaches past 0.75; "habit" is not excluded at this n.
- The LoRA arm rests on one LoRA and one draw of answers; R_L1 ≈ 0.75–0.78 is borderline.
- The wordings were fixed, not tuned; a different instruction could do better or worse.
- 2 × H200 pod with one dead GPU: about half the billed GPU-hours idle (disclosed as a deviation).

**Next (the review; not launched):**
- **Is the remaining third of the gap one instruction away?** A1 + A3 together (instruction and closing line) on the
  41 prompts, 8 answers per trace; the 0.13 MDE on the paired prompt contrast is enough to see whether R_A1 = 0.66
  passes 0.75.
- **Score the habit / skill readings at a re-read `s2300` level:** re-read all 115 source drafts once with fresh blind
  readers, instead of inheriting their olmo-ifomega labels.

*[librarian]* Not changed: olmo-transplant's summary and STATE line quote `s2300`'s 0.92 on the same thinking from the
inherited labels; this review re-read 82 of those drafts and judged 7 gamed (≈ 0.85 with its readers). The review did
not assess olmo-transplant's document, so it is not marked there.

## Figures

[`olmo_answeronly.png`](olmo_answeronly.png): genuine rate by condition for both writers beside T1 / T2, the
executor's labels (`s2300` 0.92–0.95; the review's readers give 0.83–0.86, and `dpo` A1 0.59 rather than 0.63).
