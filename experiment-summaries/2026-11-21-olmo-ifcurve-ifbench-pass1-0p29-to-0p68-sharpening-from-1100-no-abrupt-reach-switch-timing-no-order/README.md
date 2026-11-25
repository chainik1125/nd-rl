---
written_on: 2026-11-22
written_by: agent:claude
---

# olmo-ifcurve: along Olmo 3.1 Think 32B's RL run, IFBench strict pass@1 rises at every sampled checkpoint (0.29 → 0.68; within noise before step 100) and sharpens from step 1,100 on the checker score; every one of the 48 IF reach prompts gets a first genuine success (three only under a rule reading), none switches on abruptly, and the class timing reads "no order" on the first genuine success ("habit first" only on the second); gaming on random checker wins is flat

Status: completed 2026-11-22 (executor 2026-11-21 05:27 – 11-22 06:24 UTC, after a failed start at 05:20; DONE
06:22:58) and reviewed 2026-11-22 (§Recount, phase 1, blind, 09:53:22 UTC;
final 11:14:56 UTC). **No hard-constraint violation, nothing quarantined** ("No QUARANTINE"). The review's verdict is
**"Stands"**, and under "Not supported" it writes "the run doc states no conclusion the data contradict". Eight items
must be reworded, (a)–(h) below. This summary uses the reworded versions; the plain-summary copy marks the ones that
apply to it *[review]*. The main ones:
- expectation 4 at step 2,300 (≥ 1 / ≥ 2 genuine on ≥ 44 / ≥ 40 reach prompts) is "not established", not a miss;
- the estimated genuine pass@1 is "0.41–0.51 from step 750, label-dependent; no plateau onset established";
- the gradual / intermediate / slow counts depend on the labels and on the cancelled step 350.

The models are `dpo` = `allenai/Olmo-3-32B-Think-DPO` `main` (`97604023`; the RL start, step 0) and `sNNNN` =
`allenai/Olmo-3.1-32B-Think` at `step_NNNN`: `s0050` `2e757870`, `s0100` `e7b80281`, `s0200` `15d5e3cd`, `s0500`
`5bc710ab`, `s0750` `418fb274`, `s1100` `c29a2835`, `s1600` `87749126`, and `s2300` = `main` `97ade57a`, whose 14 weight
shards equal `step_2300` `29884259`. `s0350` (`735c45d5`) was cancelled (amendment 3) and has no draws. All are 32.2 B
parameters, bf16 and pretrained: Olmo 3 32B base → Think SFT → DPO → GRPO-style RL by AI2 on Dolci-Think-RL. The
output is natural-language chat, not `lean_seq`. There are no proofs, so neither Lean nor `nd_verify` is involved.
"Strict" is the official IFBench / IFEval strict checker. "Genuine" is strict plus a blinded hand gate (Claude reader
sub-agents). One RL run (AI2's), so one training seed.

*[olmo-weights review, 2026-11-25]* **`s0200` is off the RL run's lineage.** The review of
[olmo-weights](../2026-11-25-olmo-weights-rl-total-move-31-185x-below-sft-its-8-9x-lr-per-step-holds-only-at-25-50-step-spacing/README.md) computed distances between the released checkpoints from that run's stored
per-tensor statistics. It finds that `step_0150` and `step_0200` lie on a branch the run abandoned: the released line
restarts from `step_0100`, and steps 250–2,300 descend from it. In its words, this run's step-200 point ("pass@1 0.347
and the rug of 7 first-genuine prompts at step 200") "is on the abandoned branch, not on the lineage of steps
250–2,300". It adds that the step labels from 250 up "probably overstate the optimizer steps along the released
lineage by ≈ 100". The review asks for confirmation with direct weight differences and AI2's RL logs.
*[librarian]* The review did not re-derive this run's readings without `s0200`. `s0200` also appears in the copied
figure: the left panel's step-200 point and the right panel's 200 column.

Review: [`review_olmo-ifcurve.md`](https://github.com/dan-pandori/nd-takehome/blob/8e744a89/review_olmo-ifcurve.md)
(fork `dan-pandori/nd-takehome`, branch `dan_olmo-ifcurve`):
- §Recount, phase 1: [`7c23ec4b`](https://github.com/dan-pandori/nd-takehome/commit/7c23ec4b), 09:53:22 UTC;
- §Compare, the corrections to §Recount and §Verdict, after two adversarial checks (33 and 20 findings):
  [`5d2136ec`](https://github.com/dan-pandori/nd-takehome/commit/5d2136ec) (11:14:46 UTC) and
  [`8e744a89`](https://github.com/dan-pandori/nd-takehome/commit/8e744a89) (11:14:56 UTC);
- kit: `review/olmo-ifcurve-recount/`. The review says every number in it is in the kit's `out/`. Two regenerable
  caches there (`out/answers.json.gz`, `out/cells.json`) are not committed, as the kit README says.

Run files at the reviewed commit [`0498d0cf`](https://github.com/dan-pandori/nd-takehome/commit/0498d0cf):
- `run_olmo_ifcurve.md`, `numbers.md` § olmo-ifcurve, `log.md` § olmo-ifcurve, `STATUS.md`, `QUESTIONS.md` (the
  2026-11-21 question on the orphaned pods and Dan's answers);
- `olmo_ifcurve/`:
  - code: `cells.py`, `hand.py`, `adjudicate.py`, `analyze.py`, `figures.py`, `stage0.py`, `compute_rows.py`,
    `numbers_section.py`, `reader_task.sh`;
  - tables: `checkpoints.csv` (commit, shard-LFS digest and blob ids per revision), `reuse_inventory.csv`, `jobs.csv`,
    `compute.csv`, `results.csv`, `passk.csv`, `emergence.csv`, `gaming.csv`, `lengths.csv`, `summary.json`,
    `gate_summary.json`, `handcheck.csv`;
  - the reader prompt `handcheck_prompt_if.md` and the dated rules 13–14 in `handcheck_rule_addendum.md`;
  - `claim_check{,_second}.md`, `README.md`, `plain_summary.md`, `plan/`;
- the figure `figures/olmo_ifcurve.png` and the pod scripts `pod/olmo_ifcurve/`. The registry rows are not tracked in
  the fork. They are in `hf://buckets/dan-pandori/nd-rl/registry/olmo-ifcurve/` (one file, 189 rows, the same size as
  the run worktree's `artifacts/olmo-ifcurve/registry/` copy);
- the pre-registration `preregistration/olmo-ifcurve.md`:
  [`75318677`](https://github.com/dan-pandori/nd-takehome/commit/75318677), 2026-11-21 05:46:36 UTC, before the first
  pod (05:52:14) and the first draw (17:52:44). Its five amendments were each committed before the jobs they admitted:

  | amendment | commit, UTC | change |
  |---|---|---|
  | 1 | [`904399f8`](https://github.com/dan-pandori/nd-takehome/commit/904399f8) 11-21 08:37 | after the $34.84 orphan loss: arm A at 4 draws at 50–500; chunks of 4; safe pod creation |
  | 2 | [`fec95ff4`](https://github.com/dan-pandori/nd-takehome/commit/fec95ff4) 11-21 17:49 | budget $160: all checkpoints restored, secure H200s allowed, the extension size fixed by a spend rule |
  | 3 | [`9eaeb33b`](https://github.com/dan-pandori/nd-takehome/commit/9eaeb33b) 11-21 20:19 | under the $40 balance floor: `s0350` cancelled before its first draw, extension at 64 |
  | 4 | [`acae535c`](https://github.com/dan-pandori/nd-takehome/commit/acae535c) 11-22 01:53 | after the top-up: extension cells back to 128 |
  | 5 | [`e5673e65`](https://github.com/dan-pandori/nd-takehome/commit/e5673e65) 11-22 02:28 | arm A at `s0500` topped up to 8 draws |

Brief: [`BRIEF_olmo-ifcurve.md`](../../docs/proposals/followups/BRIEF_olmo-ifcurve.md); proposal 46,
[`2026-11-20-olmo-rl-curve.md`](../../docs/proposals/2026-11-20-olmo-rl-curve.md) (Dmitry's suggestion, written at
Dan's request; approved by Dan 2026-11-20 at $120, raised to $160 during the run). It builds on:
- [olmo-ifomega](../2026-11-04-olmo-ifomega-reach-on-ifbench-49-strict-43-hand-and-ifeval-6-omega-test-out-inconclusive/README.md):
  the stored `dpo` / `s0100` / `s0750` / `s2300` IFBench samples, the 218-prompt set, the pipeline and hand rules 1–5;
- [olmo-iffu](../2026-11-06-olmo-iffu-brevity-hint-elicits-dpo-on-15-16-of-48-if-reach-prompts-10-of-10-survive-1024-draws-token-budget-on-1-of-3-geometry/README.md):
  the `s0750` reach draws `T4_s0750` and rule 6;
- [olmo-ifladder](../2026-11-09-olmo-ifladder-lora-on-16-rl-trajectories-elicits-40-41-of-48-if-reach-prompts-control-0/README.md):
  the 48 reach prompts, their classes, rules 7–12 and the reader prompt;
- [olmo-answeronly](../2026-11-10-olmo-answeronly-one-instruction-lifts-dpo-on-rl-thinking-0p14-to-0p59-0p63-genuine-r-0p66-between/README.md):
  the answer-habit result behind the "habit first" reading.

Copies here:
- **the plain-language summary** (the brief's ≤ 500-word deliverable), with the review's corrections marked
  *[review]*: [`plain_summary.md`](plain_summary.md);
- **the figure**, unchanged: [`olmo_ifcurve.png`](olmo_ifcurve.png). Its left panel (strict pass@1 / pass@8) does not
  depend on labels. *[librarian]* Its right panel is drawn from the executor's labels. Its F1 dots differ from the
  review's on two prompts: `ifbench_268` at 1,100, the review's 1,600; `ifbench_170` at 1,600, the review's 2,300. The
  colour is an estimated genuine rate: cells read only in part are extrapolated from their first four labels.

Question (the brief): "Every comparison so far on Olmo 3.1 Think 32B used two or three points of one RL run. AI2
released 46 checkpoints of it (`step_0050` … `step_2300`, every 50). This run measures the trajectory:
- IFBench pass@1 and pass@8 by RL step (Dmitry's question);
- when each of the 48 instruction-following reach prompts becomes solvable, and whether gradually or abruptly
  (claim 6);
- whether the prompts that prompting alone recovers in `olmo-ifladder` are reached earlier than the teachable and
  'neither' ones (habit before skill);
- whether the gamed share of checker wins rises or falls over training."

## Configuration

- **Checkpoints:** `dpo` (step 0) and eight RL checkpoints: 50, 100, 200, 500, 750, 1,100, 1,600, 2,300. The brief's
  `s0350` was cancelled before its first draw (amendment 3), when Dan had declined a top-up ("No, run as is") and the
  $40 balance floor bound the plan. The review's own Hub query reproduced every commit hash, the 14-shard count, the
  weight bytes and the `tokenizer.json` blob in `checkpoints.csv`. It could not reproduce the run's `weights_digest`
  column, so it checked the per-shard LFS hashes directly (`main` = `step_2300`; ten distinct weight sets). Every new
  job's `JOB_START` line records its revision's commit.
- **Decoding**, identical for every model and every job (33 metas: 25 new jobs, 2 stage-0 metas, 6 reused jobs):
  - the `olmo_thinker` template (sha256 `eba6e269…`) with the DPO tokenizer;
  - the user turn = the prompt verbatim, no hints;
  - T 1.0, top-p 1.0, `max_tokens` 32,768, EOS only, `generation_config="vllm"`;
  - vLLM 0.11.0, torch 2.8.0+cu128, one H200, TP 1, no LoRA.

  Sampler `olmo_ifladder/sample.py`, unchanged. Grader `olmo_ifomega/grade.py`: IFBench `cd189a9c` / IFEval
  `e6890f85` strict and loose, on the text after the last `</think>`; a draw with no `</think>` fails.
- **Prompts:**
  - arm A: olmo-ifomega's 218 IFBench prompts (300 minus the 82 SFT exclusions). Families: format 65, words 58,
    count 43, ratio 21, sentence 14, custom 11, repeat 6.
  - arm B: olmo-ifladder's 48 reach prompts, 42 IFBench (all inside the 218) and 6 IFEval. The DPO start has 0 genuine
    in ≥ 256 draws on each (the set's definition). Classes from olmo-ifladder: **P** prompt-recoverable 25, **T**
    teachable-only 16, **N** "neither" 7 (`ifeval_3478` and IFBench 3, 67, 166, 188, 268, 284). `ifbench_154` is T
    under olmo-ifladder's rule 7 (primary) and P under rule 3 (the reading of olmo-ifladder's review); every class
    result is also given with it in P.

| arm | what | draws per cell |
|---|---|---|
| A, aggregate curve | 218 IFBench prompts, strict pass@1 / pass@4 / pass@8 (unbiased estimator), 95 % prompt bootstrap (10,000), overall and per family | `dpo` 16 (stored); `s0050`, `s0200` 4 (amendment 1's cut, never restored); `s0100`, `s0750`, `s2300` 8 (stored); `s0500` 8 (amendment 5: 4 + 4 on the 176 non-reach prompts; the first 8 `M_s0500` draws on the 42 IFBench reach prompts); `s1100`, `s1600` 8 |
| B, reach emergence | the 48 reach prompts at the eight RL checkpoints | 32 per base cell; at the checkpoint before each prompt's first genuine success, an extension cell of 128 |
| C, gaming | the hand gate on every deciding success in B, plus a seeded sample of up to 40 strict successes per checkpoint from A | no new draws |
| D, lengths | median thinking and answer tokens / words by step, all prompts and reach prompts | no new draws |

- **Reuse** (the review: same template, decoding and prompts; i.i.d. prefixes). Stored draws enter only where they did
  not select the reach set:
  - arm A at `dpo` / `s0100` / `s0750` / `s2300`: olmo-ifomega's `F_*_ifb`, seeds 3001–3004;
  - arm B at `s0100`: `F_s0100_ifb`'s 8 + 24 new on IFBench; IFEval 32 new;
  - arm B at `s0750`: olmo-iffu's `T4_s0750` chunks 0–1 (seed 7301) and `T4b_s0750` (`ifbench_19`'s chunk 1, seed
    7302), plus 16 new on the 17 IFBench prompts with one chunk; IFEval 32 new;
  - arm B at `s2300`: all new.

  The review found every cell at its pre-registered size and no two draws in a (checkpoint, prompt) with identical
  response ids (34,320 rows).
- **New jobs** (25, seeds distinct within each checkpoint):
  - `M_s0050`, `M_s0200`, `M_s0500`, `M_s1100`, `M_s1600`: arm A plus the reach base cells, seed 46000 + step (+1 for
    the part resumed after stage 0);
  - `B_s0100` (46100), `B_s0750` (46750), `B_s2300` (48300);
  - `T_s0500` (55500);
  - the extension jobs `X_*` (49000 + step), `X2_*` (49500 + step), `X3_*` (52000 + step) and `X4_s1100` (54100).

  Stage 0 measured 296 draws each at `s0050` and `s1600` before the re-cost; its draws are part of those cells.
  Extension: 49 cells of 128 on all 48 prompts. `ifbench_170` has two (`s1600`, and `s1100` from `X4_s1100`;
  observations below).
- **Hand gate**, run by the executor:
  - Blinded Claude readers saw the prompt, the constraints with the checker's verdicts and the answer. They did not see
    the model, the step, the arm or the thinking.
  - The reader prompt is olmo-ifladder's. Rules 1–12 (olmo-ifomega, olmo-iffu, olmo-ifladder) were applied in
    confirmation, plus two rules dated during the gate: 13 (`ifbench_1`) and 14 (`ifbench_0` / `ifbench_4`), both on
    keyword stuffing.
  - A mechanical rule-1 screen and identity inheritance were used.
  - The pre-registered walk: (a) every strict success up to and in each prompt's F1 cell; (b) labelling until 2
    genuine (F2); (c) cells with ≥ 8 strict labelled until 8 genuine or impossible (F25); (d) the first min(strict, 4)
    of every cell; (e) every extension success.
  - Second reads, third reads on splits, adjudication sub-agents and consistency audits u1–u4.
- **Definitions:**
  - **F1** = the first checkpoint with ≥ 1 genuine in its base 32;
  - **F2** = ≥ 2;
  - **F1\*** = the first with any genuine in all its draws (base or extension);
  - **F25** = the first with ≥ 8 genuine of its base 32;
  - "never" = > 2,300, coded 9,999.
- **Pre-registered readings** (thresholds from the proposal):
  1. timing: "habit first" iff median_P < median_T ≤ median_N and the 90 % stratified bootstrap interval of
     median(T ∪ N) − median_P has its 5th percentile > 0;
  2. gradual = ≥ 2 sampled checkpoints strictly between F1\* and F25; abrupt = F1\* = F25 after 0 of 128;
  3. gaming: early (≤ 200) vs late (≥ 1,100) gamed share among strict successes; ± 0.2 for "gaming first" / "gaming
     grows", else "flat". Primary on arm C, secondary on arm B's first ≤ 4 labels per cell;
  4. brevity: B = the first checkpoint from which median answer words stay within ±20 % of `s2300`'s; H = the first by
     which ≥ 24 of 48 have F2 ≤ c; "brevity first" iff B < H;
  5. shape: sharpening from c iff Δpass@8(c → 2,300)'s paired 90 % interval reaches 0 while Δpass@1's lies above 0.
- **Pods and cost:**
  - sampling: eight 1 × H200 pods `ifc-a` … `ifc-h`, 2026-11-21 17:50 – 11-22 04:40 UTC, SECURE at $5.29/h (no
    working community H200 was on offer);
  - three short community starts of `ifc-a` ($3.59/h) on a faulty GPU;
  - **17 orphaned H200 pods**, 05:52–08:32 on 11-21, 7.30 pod-hours, $34.84. RunPod had retired its REST API v1 on
    2026-11-15, and `podnew`'s undo failed silently. The host's pod tools were reported fixed (runpodctl 3.1.1, REST
    v2) at 17:58.
  - Total **$154.67** of the $160 budget over 29.72 pod-hours (`podbudget`). *[librarian]* RunPod billed **$155.72**
    with disk (per-pod records; the spend ledger). The pre-registration projected ≈ $86 for the core at community
    prices; see the [spend ledger](../../docs/infra/spend.md).
- **Bucket:** `hf://buckets/dan-pandori/nd-rl/olmo-ifcurve/` (2,147 objects, 341.9 MB; the review found all of them in
  its own copy, byte-identical, and the 50 MANIFEST sha256 matching):
  - `artifacts/olmo_ifcurve/samples/` and `graded/`: every new draw's text, token ids and grades (25 jobs);
  - `rows/`, `meta/`, `logs/`;
  - `hand/`: every item, batch, reader label and adjudication.

  The reused stored draws are in `…/olmo-ifomega/artifacts/olmo_ifomega/samples/` (`F_*_ifb`) and
  `…/olmo-iffu/artifacts/olmo_iffu/samples/` (`T4*_s0750`). No weights were stored.
- **Recompute** (fork, repository root, from `olmo_ifcurve/README.md`): fetch the samples
  (`python3 fetch_artifacts.py artifacts/olmo_ifcurve/` plus the stored jobs in `reuse_inventory.csv`), then:
  1. `python3 olmo_ifcurve/cells.py rows <job> …` for every job;
  2. `python3 olmo_ifcurve/hand.py screen <job> …`, then `hand.py status` (0 pending), then `hand.py table`;
  3. `python3 olmo_ifcurve/analyze.py --reps 10000`;
  4. `python3 olmo_ifcurve/figures.py`.

  The review's own pipeline is listed in its kit README.

## Observations

### What the review checked (own code; phase 1 blind; gate counts as completed in phase 2)

- **Checkpoints, prompts, reuse:**
  - checked against its own Hub query. `main` = `step_2300` by identical shard names, LFS sha256 and index weight map.
  - `generation_config.json` comes in three variants, not the pre-registration's two. This does not matter, because
    the sampler passes T 1 / top-p 1 explicitly.
  - For every prompt any job used, the prompt, instruction ids and kwargs equal the upstream records.
  - All 1,576 per-(job, prompt) prompt-token hashes equal its own rendering of the template.
- **Re-grade:** all 34,320 rows (25 new jobs + 6 reused) with its own driver around the upstream checkers. Strict and
  loose agree with the run on **34,320 / 34,320**. The length columns differ only by definition: the run counts the
  EOS in answer tokens, counts thinking to the last `</think>`, and counts draws without `</think>` as all thinking.
- **Template boundary:** five requests rebuilt from stored ids and read by hand. No response opens a second
  `<think>`. Every `stop` row ends in EOS, and the 55 other rows are cut at the cap.
- **Its own hand gate, blind in its reads** (its confirmations are qualified; see the blindness note and the
  limitations):
  - **2,134 reads** on 1,524 answer groups: 1,269 + 255 first reads, 570 second, 40 third. First and second reads
    agree on 530 / 570.
  - The pre-registered reader prompt, verbatim; random ids, shuffled across prompts, checkpoints and arms.
  - Selection: the pre-registered walk; each prompt's `s2300` cell read until 2 genuine (for expectation 4); the first
    min(strict, 4) of every arm-B cell; and an arm-C sub-sample, the first 20 of the seeded 40 per checkpoint.
  - 120 mechanical rule-1 screens and 65 identity inheritances.
  - 165 confirmations under rules 1–12, 26 of which change a reader majority or single read. 164 were proposed by an
    adjudicator sub-agent and accepted by the reviewer; 1 came from a mechanical rule-5 pass.
  - **Agreement with the executor's labels:** 0.917 on 1,607 draws (Cohen's κ 0.82). The review's labels are
    stricter: 100 draws gamed where the executor's are genuine, 33 the other way. Agreement is 0.766 on the
    executor's 192 confirmed draws and 0.932 on its reader labels. The deciding items disagree on 32 of 566:
    `ifbench_203` 9, `ifbench_170` 5, `ifbench_268` 3, `ifbench_271` 3, and 1–2 each on nine more prompts.
- **Blindness:**
  - The review discloses the exposures before its §Recount commit, with times. Its phase-1 text named:
    - its auto-memory index line on this run, which includes "results (sharpening from 1,100; genuine plateau
      ≈ 0.50)";
    - the first rows of `jobs.csv`;
    - the first two rows of the executor's `hand/confirm.csv` and the header of `hand/status.csv` (06:56);
    - the commit subjects.

    A phase-2 addendum, added after the two adversarial checks, says the phase-1 paragraph "claims more than
    happened" and adds:
    - while checking formats (06:38–06:55): the headers and line counts of the executor's `key.csv`, `screen.csv`,
      `status.csv` and `confirm.csv` and the names of its label and adjudication files, `reader_task.sh`, the item
      ids of one batch, one item file, and two `screen.csv` rows;
    - its arm-C sampling order, chosen by matching the executor's `key.csv` (06:39; 349 of 360 draws match);
    - "Only the reader prompt and the confirmation rules 1–12 were fixed before my first label. The walk code and
      the adjudication workflow changed during the gate." Its acceptance policy for adjudicator proposals was written
      after the first proposals it rejects were in;
    - from 09:29, the executor's labels, for the agreement table.

    That table showed rule-5 and rule-10 cases its first pass had missed. So it added a rule-5 pass and a
    prompt-level rule-10–12 pass, each applying a pre-registered rule mechanically or through the adjudicator; neither
    copies an executor label. After reading the run's documents and the first adversarial check, it added third reads
    (C7, C8), a second-read round (B9) and a consistent application of its acceptance policy. No reading changed. The
    gate counts did, and so did the `s2300` arm-B gamed count: 68 of 188, so the late share is 0.334, not 0.336.
  - *[librarian]* The librarian parsed the reviewer's transcript: 468 main-session tool calls before 09:53:22.
    - The executor's hand-gate files and outcome tables appear in them only at the disclosed times: 06:24:57
      (`jobs.csv`), 06:38–06:42, 06:55:39, and the comparisons from 09:29:16 to 09:50:04 (`rvc_compare.py` at
      09:29:16 and again at 09:48:54, the second with its printout discarded; its `out/compare.txt` was read at
      09:49:27 and 09:50:04). As part of its job the review also read the pre-registration, `checkpoints.csv`, the `plan/`
      listing and the reader prompt.
    - There is no read of the executor's memory note, transcripts, write-ups, figure or registry rows, or of
      `results.csv`, `passk.csv`, `emergence.csv`, `gaming.csv`, `lengths.csv`, `summary.json`,
      `gate_summary.json` or `handcheck.csv`.
    - The executor's distinctive figures do not appear in any tool result before the commit. The search covered its
      arm-C shares 0.138 / 0.217, its arm-B late share 0.306, the 10 / 26 / 12 split, `s2300` 40 / 36, and its
      estimated genuine pass@1 0.410 / 0.505 / 0.496. The only matches are the review's own shape-interval bound
      0.138 and the stored `s0100` pass@8 0.505 that the pre-registration quotes.
    - The reader and adjudicator sub-agents (3,162 tool calls before the commit) opened only their task and item
      files.

### Arm A: strict pass@k on the 218 IFBench prompts

The review reproduced every strict pass@k value from the stored samples with its own code (the intervals below are
the review's). Estimated genuine pass@1 = strict × (1 − that checkpoint's arm-C gamed share). It is an estimate from 20 (review) or 40 (executor) labels per checkpoint:
each share carries a binomial SE of ≈ 0.05–0.10 at 20 labels and ≈ 0.03–0.07 at 40.

| step | draws / prompt | strict pass@1 [95 %] | pass@4 | pass@8 [95 %] | est. genuine pass@1, review / executor |
|---|---|---|---|---|---|
| `dpo` (0) | 16 | 0.290 [0.239, 0.343] | 0.403 | 0.461 [0.400, 0.524] | 0.246 / 0.254 |
| 50 | 4 | 0.296 [0.243, 0.351] | 0.422 | — | 0.237 / 0.244 |
| 100 | 8 | 0.315 [0.263, 0.368] | 0.441 | 0.505 [0.440, 0.569] | 0.268 / 0.268 |
| 200 | 4 | 0.347 [0.292, 0.405] | 0.482 | — | 0.295 / 0.313 |
| 500 | 8 | 0.412 [0.357, 0.467] | 0.580 | 0.656 [0.592, 0.716] | 0.391 / 0.391 |
| 750 | 8 | 0.512 [0.460, 0.564] | 0.714 | 0.794 [0.739, 0.844] | 0.461 / 0.410 |
| 1,100 | 8 | 0.577 [0.528, 0.626] | 0.795 | 0.872 [0.826, 0.913] | 0.433 / 0.505 |
| 1,600 | 8 | 0.646 [0.598, 0.694] | 0.830 | 0.872 [0.826, 0.913] | 0.485 / 0.485 |
| 2,300 | 8 | 0.683 [0.635, 0.732] | 0.844 | 0.885 [0.844, 0.927] | 0.478 / 0.496 |

- **Strict pass@1 rises at every sampled checkpoint.** All adjacent paired differences are positive; only the first
  two 95 % intervals (0 → 50, 50 → 100) include 0. `dpo`'s first 8 draws give 0.290 / 0.459. *[olmo-weights review,
  2026-11-25]* Step 200 is off the lineage of steps 250–2,300 (see the note under the models).
- **Shape (reading 5): sharpening from 1,100,** on the checker score. The paired 90 % intervals of the remaining gain
  to 2,300 are:
  - from 750: Δpass@1 +0.171 [+0.138, +0.205], Δpass@8 +0.092 [+0.046, +0.138];
  - from 1,100: +0.106 [+0.080, +0.133] and +0.014 [−0.018, +0.046];
  - from 1,600: +0.037 [+0.018, +0.056] and +0.014 [−0.014, +0.041].

  pass@8 at `s2300` is 0.885 against a ceiling of 208 / 218 = 0.954: ten prompts have no strict success in any of the
  72 draws over all nine checkpoints. `s1100` and `s1600` share a pass@8 by coincidence (190 of 218 prompts each, 11
  swapped each way).
- **Per family** (numbers.md; the review reproduced the point values), strict pass@1 at `dpo` → `s0750` → `s2300`:

  | family | n | `dpo` | `s0750` | `s2300` |
  |---|---|---|---|---|
  | count | 43 | 0.573 | 0.666 | 0.852 |
  | format | 65 | 0.309 | 0.637 | 0.737 |
  | words | 58 | 0.185 | 0.418 | 0.713 |
  | sentence | 14 | 0.107 | 0.393 | 0.652 |
  | ratio | 21 | 0.104 | 0.226 | 0.327 |
  | repeat | 6 | 0.302 | 0.500 | 0.542 |
  | custom | 11 | 0.205 | 0.375 | 0.352 |

  The review: "Every family rises; `custom` (11) and `repeat` (6) are flat or noisy after 750."
- **Reword (d), estimated genuine pass@1:** "0.41–0.51 from step 750, label-dependent; no plateau onset
  established", not the run doc's "rises 0.25 → 0.51 by 1,100, then stays near 0.50". From 750 to 1,100 it rises by
  about 0.10 on the executor's labels and falls by about 0.03 on the review's. The review's 20 labels per checkpoint
  are the first half of the executor's 40-draw sample, so they are not independent evidence. On the same 20 `s0750`
  draws the executor labels 5 gamed and the review 2.

### Arm D: lengths (medians over all arm-A draws; the review's definitions)

| step | 0 | 50 | 100 | 200 | 500 | 750 | 1,100 | 1,600 | 2,300 |
|---|---|---|---|---|---|---|---|---|---|
| thinking tokens | 2,114 | 2,166 | 2,239 | 2,147 | 3,039 | 4,412 | 5,307 | 4,501 | 5,470 |
| answer words | 261 | 243 | 219 | 181 | 135 | 93 | 73 | 57 | 50 |
| cut at 32,768 | 0.06 % | 0 | 0 | 0 | 0.17 % | 0.17 % | 0.46 % | **0.63 %** | 0 |

- Thinking is counted to the first `</think>`, over draws that have one. The run counts to the last `</think>` and
  counts draws without one as all-thinking (`s1100` 5,318, `s1600` 4,532). Under the run's convention thinking falls
  14.8 % at `s1600`, against 15.2 % under the review's.
- Draws without `</think>`: at most 0.52 % (`s1600`, arm A) and 0.65 % (`s1100`, reach cells).
- Reach base cells think longer (`s0050` 2,598 … `s2300` 6,059 tokens) and answer shorter late (`s2300` 33 words).
- **Brevity (reading 4): not brevity first.**
  - B = 1,600: from there on, median answer words stay within ±20 % of `s2300`'s 50. At 1,100 the median is 73, 46 %
    above.
  - H = 750: by 750, 27 of 48 reach prompts have F2 on the review's labels (28 per cell and 29 cumulative on the
    executor's).
  - The same holds with answer tokens.

### Arm B: when the 48 reach prompts become solvable (the review's labels; the executor's where they differ)

- **First genuine step F1** (prompts per checkpoint):

  | | 100 | 200 | 500 | 750 | 1,100 | 1,600 | 2,300 | never |
  |---|---|---|---|---|---|---|---|---|
  | review | 2 | 7 | 11 | 15 | 7 | 3 | 3 | 0 |
  | executor | 2 | 7 | 11 | 15 | 8 | 3 | 2 | 0 |

  Two prompts differ:
  - `ifbench_268`: executor 1,100, review 1,600. Its one `s1100` success is gamed in two of the review's reads, and
    the adjudicator's low-confidence "genuine" proposal was not accepted.
  - `ifbench_170`: executor 1,600, review 2,300. The executor's audit u flipped its one `s1600` success, which the
    review's two reads call gamed.

  F1 ≤ 750 on 35 of 48. No prompt has a genuine success at `s0050` in its base 32. *[olmo-weights review,
  2026-11-25]* F1 = 200 (7 prompts) is measured on `s0200`, off the lineage of steps 250–2,300 (see the note under the
  models).
- **Reword (c), "all 48 reached":**
  - `ifbench_64` (`repeat_simple`, 40 successes, one distinct answer) and `ifbench_284` (`repeat_span`, 16 successes,
    all "code and theory behind") are reached only by the fixed-output copy that rule 5 counts as genuine.
    Before the review's rule-5 pass, `ifbench_284` had 0 genuine in 352 RL draws (8 × 32 base + 96 extension at
    `s0750`).
  - `ifbench_189` is reached only through answers its checker reads as one sentence. It is never reached under rule
    7's literal example (variant `r7literal`).
- **Reading 1, timing by class:** **"no order"** on F1. Never = 9,999; stratified bootstrap, 10,000 replicates.

  | statistic | median P | median T | median N | median(T ∪ N) − median(P) | 90 % interval | reading |
  |---|---|---|---|---|---|---|
  | F1 | 500 | 750 | 1,100 | 250 | [0, 600] | **no order** |
  | F1, `ifbench_154` in P | 500 | 750 | 1,100 | 425 | [0, 600] | no order |
  | F2 (per cell) | 500 | 1,100 | 1,600 | 600 | [350, 850] | habit first |
  | F2 (per cell), `ifbench_154` in P | 625 | 1,100 | 1,600 | 475 | [350, 850] | habit first |

  - The order condition holds on F1, but 9.5 % of replicates are ≤ 0, so the 5th percentile is exactly 0. The F1
    reading is "no order" in all seven of the review's label variants, and the F2 reading "habit first" in all
    seven.
  - With the review's code (per-cell F2), the F1 and F2 medians and intervals are identical on the executor's labels
    (`out/phase2_exec.txt`, primary class assignment).
  - **Reword (b):** the run's F2 is the cumulative second genuine success across cells (`hand.py`), which gives T 925
    and [350, 600]. The review's is the first checkpoint with ≥ 2 genuine in its base 32: T 1,100, [350, 850], 28 by
    750 on the executor's labels. The pre-registration is ambiguous between the two, and the readings are the same.
- **Reading 2: gradual or abrupt.** No prompt switches on abruptly.
  - Under the review's labels, 32 prompts reach F25: 7 gradual (0.22), 25 intermediate and 0 abrupt. The other 16
    are slow (genuine by 2,300, no F25).
  - By class: P 6 gradual / 15 intermediate / 4 slow; T 1 / 7 / 8; N 0 / 3 / 4.
  - **Reword (f):** quote this qualitatively: "abrupt 0; gradual a minority on the nine-checkpoint grid". The counts
    depend on the labels. The run's 10 / 26 / 12 is what the review's code gives on the executor's labels, against
    7 / 25 / 16 under the independent gate, and 0.19–0.24 gradual across the review's variants. They also depend on
    the grid: with `s0350` sampled, five more prompts (133, 136, 187, 204, 248; F1\* 200, F25 750) would be gradual,
    12 of 32 under the review's labels.
  - With eight RL checkpoints, "≥ 2 checkpoints strictly between F1\* and F25" is hard to meet; most reached prompts
    go from F1\* to F25 in one or two steps.
  - Three prompts have F1 = F25 (267, 252, 207), but each had 4–7 genuine of 128 in its extension one checkpoint
    earlier.
- **Earliest detectable rate:** the 128-draw cell before F1 already held a genuine success on **17** prompts (the
  executor: 18). At F1\*, the median genuine rate is **1 / 32** over all 48 prompts and 0.039 over the 32 with F25,
  both within expectation 6's ≤ 3 / 32. 17 prompts' F1\* is an extension cell, at 1–7 of 128. A zero bounds a rate
  only per cell: 0 of 128 → < 2.3 %, 0 of 32 → < 8.9 % (95 %).
- **Extension placement:** on 46 of 48 prompts the run's extension cell is the checkpoint before the review's F1.
  - `ifbench_268`: the run extended `s0750` (the checkpoint before the executor's F1 at 1,100). So the cell before
    the review's F1, `s1100`, has only 32 draws.
  - `ifbench_170`: the run extended `s1600` (7 genuine of 96 extension draws under the review's labels). After the
    executor's audit at 02:27 moved its F1 to 1,600, it added a 96-draw extension at `s1100` (`X4_s1100`, started
    04:14, no amendment), with 5 genuine of 96. The review (finding 4): X4 follows the pre-registered placement rule
    for the executor's F1. The `s1600` extension draws then became extra draws in the F1 cell, which the log reports.
    Under the review's labels X4 is extra, and counting it would move F1\* to 1,100.

### Arm C: gaming over training (gamed / labelled among strict successes)

| step | 0 | 50 | 100 | 200 | 500 | 750 | 1,100 | 1,600 | 2,300 |
|---|---|---|---|---|---|---|---|---|---|
| arm C, review (20 / step) | 3/20 | 4/20 | 3/20 | 3/20 | 1/20 | 2/20 | 5/20 | 5/20 | 6/20 |
| arm C, executor (40 / step) | 5/40 | 7/40 | 6/40 | 4/40 | 2/40 | 8/40 | 5/40 | 10/40 | 11/40 |
| arm B first ≤ 4 per cell, review | – | 0/0 | 0/2 | 1/14 | 10/61 | 32/122 | 47/170 | 64/178 | 68/188 |
| arm B first ≤ 4 per cell, executor | – | 0/0 | 0/2 | 1/14 | 11/61 | 27/122 | 42/170 | 55/178 | 67/188 |

- **Primary (arm C): flat.** Early {0, 50, 100, 200} 0.163 (n 80) vs late {1,100, 1,600, 2,300} 0.267 (n 60). Early −
  late = −0.104, prompt-cluster 90 % interval [−0.227, +0.014]. The executor's 40 per step give 0.138 vs 0.217, also
  flat.
- **Secondary (arm B): gaming grows,** on 16 early items. Early {50, 100, 200} 0.062 (1 of 16, from 10 prompts) vs late
  0.334 (179 / 536). The difference is −0.271 [−0.384, −0.137]; the executor's 0.06 vs 0.31 reads the same. The tiny
  early side drives the interval's width.
- *[review, correction to §Recount]* The late set is not "dominated by the slow prompts". They give 169 of the 536 late
  items (32 %), but 105 of the 179 late gamed ones (59 %).

### Expectations (pre-registered 2026-11-21 05:46), scored by the review

| # | expectation | review's values | review's verdict (executor's where different) |
|---|---|---|---|
| 1 | pass@1 50 [0.27, 0.34], 200 [0.31, 0.42], 500 [0.40, 0.51], 1,100 [0.53, 0.64], 1,600 [0.58, 0.68]; pass@8 500 [0.62, 0.78], 1,100 [0.80, 0.89], 1,600 [0.82, 0.90] | 0.296, 0.347, 0.412, 0.577, 0.646; 0.656, 0.872, 0.872 | **hit**. `s0350` and pass@8 at 50 / 200 not measured (amendments 1 and 3) |
| 2 | no pass@1 drop > 0.03 | every adjacent difference positive | **hit** |
| 3 | sharpening from 1,100 or 1,600 | from 1,100 | **hit** |
| 4 | any genuine at 50 / 100 on ≤ 3 of 48; F1 ≤ 750 on 28–36; at 2,300 ≥ 1 genuine on ≥ 44 and ≥ 2 on ≥ 40 | 0 / 2 on base cells (4 at 100 with the extension draws); 35; **44 / 41** with every `s2300` strict success read until 2 genuine (rules 1–12) | first two parts **hit** (the 100 part on base draws only). **Reword (a): the 2,300 part is "not established"**, not "missed": the run's 40 / 36 are lower bounds from partly read cells, and the full read sits exactly at the threshold, 43–44 / 39–41 across the review's variants. It needs rules 1–12 for `ifbench_0` and `ifbench_1` (3 and 6 `s2300` genuines), which rules 13–14 could remove |
| 5 | median F1 P ≤ 500, T ∈ {500, 750}, N ≥ 1,100; reading "habit first" | 500 / 750 / 1,100; "no order" | medians **hit**; reading **miss** ("habit first" only on F2) |
| 6 | of prompts with F25: gradual ≥ 50 %, abrupt ≤ 20 %; median earliest rate ≤ 3 / 32 | 22 % (executor 28 %); 0; 1 / 32 over 48 (0.039 over the 32 with F25) | gradual **miss**; abrupt and rate **hit** |
| 7 | arm C early and late in [0.05, 0.30], "flat"; arm B "gaming first" | 0.163 / 0.267, flat; arm B grows | primary **hit**; secondary **miss** (the opposite direction) |
| 8 | B ∈ {1,100, 1,600}, H ∈ {750, 1,100}; "not brevity first" | 1,600; 750; not brevity first | **hit** |
| 9 | thinking `s0050` [1,900, 2,600], `s1600` [4,500, 5,700], no checkpoint > 15 % below its predecessor; answer words 50 [200, 270], 500 [90, 180], 1,100 [55, 95], 1,600 [45, 75] | 2,166; 4,501; `s1600` 15.2 % below `s1100` (14.7 % counting no-`</think>` draws as all-thinking, to the first `</think>`; the run's count to the last `</think>` gives 14.8 %); 243, 135, 73, 57 | ranges **hit**. **Reword (h):** "held at 14.8 %, with draws without `</think>` counted as all-thinking"; without them it misses by 0.2 pp. The pre-registration does not settle the convention |
| 10 | cap hit by ≤ 0.5 % of arm-A draws at every checkpoint | `s1600` 0.63 % (11 / 1,744); others ≤ 0.46 % | **miss** at 1,600 |

The review on the run's scoring: it "reports its misses as misses, with two overstatements". These are the 2,300 part of
expectation 4, called "missed" rather than "not established", and the run doc's "the rest held", which includes
expectation 4's `s0100` part. That part holds only on base draws (numbers.md says so; the run doc does not).
Readings and thresholds were not changed. Amendments 1, 3, 4 and 5 changed only what could be scored.

### Compute (the review's §11; one H200 per job, TP 1, no training, no Lean)

| arm (new draws only) | draws | generated tokens | GPU-h (`JOB_START` → `JOB_DONE`) |
|---|---|---|---|
| A: the M jobs' non-reach cells and `T_s0500` | 5,632 | 28.6 M | 4.69 |
| B base (includes the 42 IFBench reach prompts' arm-A draws) | 10,880 | 60.5 M | 9.83 |
| B extension | 4,704 | 27.7 M | 5.78 |
| **total, 25 new jobs** | **21,216** | **116,786,901** | **20.31** (the sampler's own load + generation: 20.22) |

- M jobs are split between arms by tokens.
- Reused stored draws; their compute belongs to olmo-ifomega and olmo-iffu. Whole jobs:
  - `F_dpo_ifb` 4,800 draws / 21.1 M tokens; `F_s0100_ifb` 2,400 / 10.4 M; `F_s0750_ifb` 2,400 / 13.6 M;
    `F_s2300_ifb` 2,400 / 15.6 M;
  - `T4_s0750` 1,088 / 7.6 M; `T4b_s0750` 16 / 0.09 M.

  The cells use 218 × 16 / 8 / 8 / 8 arm-A draws of the `F_*` jobs and the reach prefixes.
- Checker calls: 42,432, strict + loose per draw. The registry's `verifier_checks` count one per draw.
- Registry: 189 rows, 27 each (one per job run; `M_s0050` and `M_s1600` ran twice) of `gpu_seconds` (72,808 s),
  `gen_tokens`, `attempts` (21,216), `train_steps` (0), `train_tokens` (0), `lean_checks` (0) and `verifier_checks`.
- Arms C and D use no GPU. The executor's hand gate made 2,104 reads in 64 reader batches, plus 11 adjudication /
  audit tasks (75 Claude sub-agent tasks); the review's gate made 2,134 reads.
- Job wall time is 91 % of the sampling pods' 22.41 pod-hours.

**Reword (e), the 1.25× flag** (AGENT_POLICY), at equal draws, in tokens per draw:
- arm A against `dpo`'s 4,448: `s0750` **1.251**, `s1100` **1.417**, `s1600` 1.245 (the review: 1.246), `s2300`
  **1.411**;
- arm B base cells against `s0050`'s 4,547: `s0500` 1.10, `s0750` **1.42**, `s1100` **1.53**, `s1600` **1.31**,
  `s2300` **1.51**.

So arm A at 750, 1,100 and 2,300 and arm B at 750–2,300 used more than 1.25× the step-0 (arm B: step-50) tokens per
draw. This is the models' own length under one 32,768 cap. The design matches draws, not tokens, and the
pre-registration expected ≈ 1.4×. Neither the run doc nor numbers.md flagged it.

### Other policy checks (the review's §13)

- **Same template, tokenizer, decoding and budget** for every compared model; the only difference per checkpoint is
  the models' own lengths.
- **Token budget and cut share** per model: above (arm D), and in the review's §3.
- **Confirmation in both directions.** The second reads cover the gamed items before F1 and the genuine items that
  set F2. The review added second reads on F25-fragile cells. Extensions sit only before F1, which works against
  "abrupt".
- **RL-data overlap:** inherited from olmo-ifomega's scan against Dolci-Think-RL-32B (4 files, 102,026 documents).
  0 of 218 and 0 of 48 at content containment ≥ 0.5 (maximum 0.382 for arm A, 0.174 for reach). 7 of the 218 and 1 of
  the 48 share a 13-gram with some RL document. It is not a new scan.

### The review's rewordings (§Verdict), all applied above

- (a) Expectation 4 at `s2300`: "not established".
- (b) F2: the run's cumulative definition, or the per-cell recomputation.
- (c) "All 48 reached": qualified for `ifbench_64`, `ifbench_284` and `ifbench_189`.
- (d) Estimated genuine pass@1: "0.41–0.51 from step 750, label-dependent; no plateau onset established".
- (e) The 1.25× token flags.
- (f) Reading 2 quoted qualitatively.
- (g) "Name the DPO repository in the run doc": `allenai/Olmo-3-32B-Think-DPO` (numbers.md names it).
- (h) Expectation 9 under its counting convention.

**The fork's executor files are not corrected.** `run_olmo_ifcurve.md`, `numbers.md` and `plain_summary.md` on
branch `dan_olmo-ifcurve` still carry the wording the review asks to change. The figure and `emergence.csv` are on the
executor's labels. Only the plain-summary copy here is marked.

## Limitations and next step

- **One RL run, one seed:** AI2's Olmo 3.1 Think 32B. The reach set was selected on `dpo` failing and on `s2300`
  succeeding in other draws, so reaching all 48 by 2,300 is largely built in (the run doc says so).
- **Label noise at thresholds.** The two gates agree on 0.917 of the draws both labelled. The review's own gate
  deviates from the pre-registered confirmation in two ways, which weakens its labels as an independent standard
  (finding 6):
  - its adjudicator was a sub-agent whose proposals it accepted by a policy written after the first proposals;
  - its same-shape audit was mechanical and covered only per-sentence and sub-bullet constraints, plus the
    prompt-level rule-10–12 pass.

  `out/phase2_exec.txt` (its code on the executor's labels) shows the readings do not depend on the labels; the counts
  do.
- **Grid resolution.** `s0350` was cancelled, so reading 2's gradual test and the P median rest on a 200 → 500 gap.
  Arm A has only 4 draws at 50 and 200 (no pass@8 there).
- **Rules 13–14** (keyword stuffing on `ifbench_0` / `ifbench_1` / `ifbench_4`) were written during the gate and
  dated, as the pre-registration allows. Rule 14's label changes are not itemised in the addendum. The review's labels
  use rules 1–12 only. Every reading is the same, except that expectation 4's `s2300` count depends on them.
- **Rates near 1 in 1,000** are out of reach: base cells of 32 detect a few per cent, extensions of 128 about 1–2 %.
- **Next measurement** (the review). What remains open sits on label noise and grid resolution at thresholds: the F1
  timing interval's 5th percentile is exactly 0; the `s2300` count is 44 against 44; reading 2's categories change
  with one sampled checkpoint. To settle them:
  1. read the 1,391 still-unread base-cell strict successes at `s1100`–`s2300` with three readers each, under one
     pre-registered confirmation procedure applied to both label sets;
  2. add 32 draws per reach prompt at `s0350`. That restores the grid point between 200 and 500, which reading 2's
     gradual test and the P median depend on.
- **Follow-ups the proposal names** (not tested here):
  - why a prompt switches on, with olmo-ifladder- or olmo-answeronly-style probes at each prompt's switch point;
  - cross-family repeats on other released RL runs: Tülu 3.1 8B (61 checkpoints) and Nemotron-Cascade-8B (stage
    checkpoints) ([open RL checkpoints](../../docs/literature/2026-11-08-open-rl-checkpoints/README.md)).
- **Process.** RunPod's REST v1 retirement (2026-11-15) left the host's `podnew` / `podrm` / watchdogs unable to delete
  pods until runpodctl 3.1.1 was reported installed on 11-21 at 17:58. The run lost $34.84 to 17 orphaned pods, and
  added a REST v2 client (`pod/olmo_ifcurve/rp2.py`), an orphan check after every create and a GPU health gate.
