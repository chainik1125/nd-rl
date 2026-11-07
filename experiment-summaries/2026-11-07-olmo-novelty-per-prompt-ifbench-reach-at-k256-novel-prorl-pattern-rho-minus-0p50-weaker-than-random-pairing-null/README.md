---
written_on: 2026-11-07
written_by: agent:claude
papers:
  - liuProRLProlongedReinforcement2025
  - yueDoesReinforcementLearning2025
---

# olmo-novelty: against 200 screened sources, our per-prompt, confirmed, hand-gated IFBench reach at k = 256 (C3) is new, the gamed share among reach-deciding instruction-following successes (C4) is new only in that narrow form, and the reading of the start's failures (C6, an unreviewed number) is new for instruction following; C1's reach result (under RL's verifier) replicates Yue et al.; ProRL's "largest where the start is weakest" holds on our IFBench prompts (Olmo 3.1 Think 32B step 2,300 against its DPO start, strict pass@1: ρ −0.50) but is weaker than a random-pairing null (−0.65), so it cannot separate creation from sharpening

Status: **completed 2026-11-07 (DONE 14:52:32 UTC) and reviewed 2026-11-07 (final commit 15:51:50 UTC). No
hard-constraint violation, nothing quarantined.** In the review's verdict these **stand**: the correlation; all
loose-credit, gain-share and compute figures; the ledger; the screening counts; REVIEW §7's scoring of every
pre-registered expectation (the run summary's "L-1 and L-5 failed", where L-5 partly held, is the exception); and the
main novelty conclusions. Nothing the executor's REVIEW states as a finding is "not supported". The weakest claim is
C4's "novel", which rests on a narrow definition of the measurement. The review asks for six rewordings and five minor
ones. Those that concern REVIEW.md are applied, marked *[review]*, in the nd-rl reading copy
([`docs/literature/2026-11-06-olmo-novelty/`](../../docs/literature/2026-11-06-olmo-novelty/README.md)); rewording 5
and the `numbers.md` item concern files that were not copied. The fork's `run_olmo_novelty.md`, `numbers.md` and
`STATUS.md` are uncorrected. This was a literature run with one computation on stored results: no pods, no model
trained or sampled and no proofs counted, so no Lean check applies. Where the reviewer's wording differs from the
executor's, this summary uses the reviewer's.

Question (Dan, 2026-11-06; proposal 38): for each finding of our Olmo runs, what is the closest prior result, does it
replicate, extend or contradict ours, and what remains genuinely new? Dan asked in particular for ProRL
[@liuProRLProlongedReinforcement2025] side by side with ours, ProRL's "largest where the start is weakest" correlation
computed on our stored results, and a direct answer to whether our IFBench reach survives an Invisible-Leash-style
objection (format tolerance, much larger k). The claims C1–C8 are those of olmo-rlzero-pilot (C1), olmo-think32b
(C2), olmo-ifomega (C3–C6), the Olmo runs together (C7, methods) and olmo-iffu (C8).

## Configuration

- **Models behind our numbers** (natural-language chat models; no proofs, so the Lean rule does not apply and
  `nd_verify` is not involved):
  - `dpo` = `allenai/Olmo-3-32B-Think-DPO` (32 B; pretrained base → Think SFT → DPO; the RL start);
  - `s0100` / `s0750` / `s2300` = `allenai/Olmo-3.1-32B-Think` at RL steps 100 / 750 / 2,300 (Dolci-Think-RL);
  - C1 only: `allenai/Olmo-3-1025-7B` (base) against `allenai/Olmo-3.1-7B-RL-Zero-Math` (steps 500 / 2,800).

  "Hand" in the Olmo runs means blinded Claude subagent readers plus the executor's confirmation (for C1,
  olmo-rlzero-pilot's reviewer's six reader subagents), not human annotators.
- **Brief and branch.** Brief:
  [`docs/proposals/literature/BRIEF_olmo-novelty.md`](../../docs/proposals/literature/BRIEF_olmo-novelty.md) (Dan,
  2026-11-06; [proposal 38](../../docs/proposals/2026-11-06-olmo-novelty.md)). Fork `dan-pandori/nd-takehome`, branch
  `dan_olmo-novelty`, from `origin/dan_olmo-iffu`; worktree `~/work/olmo-novelty` on agent2. It is not merged into the
  fork's `dan`. Its only code is `olmo_novelty/{correlation,merge_screen,verify_ledger}.py` (review, R1).
- **Pre-registration.** `preregistration/olmo-novelty.md`, one commit,
  [`664aaa93`](https://github.com/dan-pandori/nd-takehome/commit/664aaa93) at 12:17:36 UTC, unchanged since. The
  correlation code and outputs came after it (commit `ffaf3494` at 12:27:37; file times 12:25–12:27), and so did the
  literature agents (launched 12:27). The header's "Written 12:15–12:40 UTC" is a window that ends 22 minutes after
  its own commit; `log.md` explains it, and a working-tree edit of the header was reverted, so the file is as
  pre-registered (review, R1 and Compare).
- **Method** (pre-registered).
  - Relation labels per (claim, paper): **replicates** = same direction, comparable measurement; **extends** = same
    direction, but one side has a setting, axis or control the other lacks; **contradicts** = the opposite finding
    under a comparable measurement. A claim is **novel** if no screened source reports the same measurement on the same
    kind of task and start model, and every "novel" label is re-checked by a fresh search.
  - Verification: every source text is fetched to a file outside the repository, every cited claim is a verbatim
    quote (≤ 25 words) with its location in a ledger, and a script greps each quote in its source, with negative
    controls.
  - The correlation (Q2): Spearman ρ across IFBench's 218 non-excluded prompts, strict grade, `s2300` against `dpo`,
    between x = `dpo` pass@1 on 8 of its 16 fixed draws and the gain = `s2300` pass@1 (its 8 fixed draws) minus `dpo`
    pass@1 on the other 8. The split is a random hypergeometric partition averaged over 2,000 partitions, with a
    2,000-resample prompt bootstrap and a random-pairing null (RL's per-prompt rates permuted across prompts, 2,000
    times). Data: olmo-ifomega's fixed phase (`olmo_ifomega/results.csv`).
- **Executor**, 12:10–14:53 UTC, `--effort max`, one invocation. Six research agents (PR: the ProRL family; IL: its
  critics and What Outcome RL Buys; IF: instruction following; DT: domains and trajectories; GG: gaming, grading and
  Yue's citers; NV: a fresh novelty search) and two adversarial claim checks of the REVIEW (34 findings, then 28 new
  plus a re-check of the first 34). DONE [`60f33d7d`](https://github.com/dan-pandori/nd-takehome/commit/60f33d7d) at
  14:52:32.
- **Reviewer**, 14:56–15:52 UTC, a separate session.
  - §Recount (phase 1, blind) [`5074ca1f`](https://github.com/dan-pandori/nd-takehome/commit/5074ca1f) at 15:11, written
    before opening the executor's REVIEW, README, claim checks, run summary, `numbers.md`, `STATUS.md`, the log's
    section, `correlation.py` or its outputs. Its inputs were the brief, the pre-registration, the claims sheet, the
    agent kit, the raw ledgers and screening rows, the source texts and the stored Olmo results.
  - *[librarian]* The review states no limit on that blindness, but there is one. At 14:56, before writing §Recount,
    the reviewer read the executor's memory note `olmo-novelty-learnings.md` (written 14:29). The note gives the
    correlation (−0.50, against a null of −0.66) and says that L-1 and L-5 failed (HiR; 2611.02574). A 15:08 scan of
    the executor's transcript for outside contact also printed fragments of its edits (e.g. "175 distinct sources
    screened (152 new", "748 / 748; 0 / 735 negative controls"). The write-ups themselves were first opened after the
    §Recount commit (15:11:40), from 15:11:49 on, and the agents' relation-label files were not read in phase 1. The
    recount's numbers come from the reviewer's own code on the raw rows and source texts.
  - A blind fresh search by three agents (RA, RB, RC), launched at 15:00 and told not to open any of the run's files.
  - §Fresh search, §Compare (phase 2) and §Verdict
    [`4263c2af`](https://github.com/dan-pandori/nd-takehome/commit/4263c2af) at 15:26. An adversarial check of the
    review by a fresh subagent flagged 21 statements, of which 20 were fixed and 1 rejected
    ([`81424523`](https://github.com/dan-pandori/nd-takehome/commit/81424523)). Final commit `d2d44fd4` at 15:51:50.
  - Kit `review_onv/` (own code): `corr.py` (the correlation), `ledger_check.py` (its own quote normaliser),
    `refetch.py` and `refetch_html.py` (fresh arXiv copies), `screen_count.py`, `phase2_checks.py`, the fresh-search rows
    `fresh/R{A,B,C}.tsv` and `adv_check.md`.
- **Pods: none.** A $0.50 / 1 h budget was registered only as a guard (`podbudget olmo-novelty`: 0.00 h, **$0.00**).
  No GPU was used. `correlation.py` ran ≈ 80 s on the VPS CPU. There are no arms, so there are no results-registry
  compute rows: nothing was sampled, trained or Lean-checked (0 GPU-seconds, 0 generated tokens, 0 training steps, 0
  Lean checks). Claude time only.
- **Bucket:** nothing uploaded; the run made no checkpoints, artifacts or data. The deliverables are in git. The 102
  fetched source texts (`~/onv_sources/` on agent2; the reviewer's in `~/onv_rev_sources/`) are copyrighted, so they
  are in neither git nor the bucket.
- **Outputs in nd-rl:** [`docs/literature/2026-11-06-olmo-novelty/`](../../docs/literature/2026-11-06-olmo-novelty/README.md):
  - REVIEW.md, with the review's corrections marked;
  - 20 notes on 19 papers, the 748-quote ledgers, the claims sheet and `screened.md` (200 sources);
  - `to_add_to_zotero.md`: 51 identifiers for Dan. `references.bib` is untouched, and `code/tools/literature.py` was
    not run;
  - `correlation.py` with its three outputs, unmarked (one correction is listed in the copy's README).
- Review: [`review_olmo-novelty.md`](https://github.com/dan-pandori/nd-takehome/blob/d2d44fd4/review_olmo-novelty.md).
  Executor summary: [`run_olmo_novelty.md`](https://github.com/dan-pandori/nd-takehome/blob/60f33d7d/run_olmo_novelty.md)
  (uncorrected; see the review's corrections below).

## Observations (the review's numbers and wording)

### Verification

- **The ledger.** All 748 quotes in 7 ledgers are in their named source file under the reviewer's own normaliser: 747
  exactly and one (IF-85, a formula `f(x)` rendering) only after deleting non-alphanumerics. None is over 25 words. The
  negative controls fail as they should: 0 / 748 found with one token altered, 0 / 748 with two adjacent words swapped,
  0 / 748 when looked up in a different paper's file.
- **Authenticity.** The reviewer re-fetched 14 key sources from arXiv, one at a time. 464 of the 465 ledger quotes
  citing them are in the fresh copies; the miss (DT-128) is a LaTeX display formula whose HTML rendering differs. "The
  source files are genuine."
- **Support.** 21 random `supports` sentences (3 per ledger) agree with ±300 characters of source context, 21 / 21. In
  phase 2 the reviewer read all 74 ledger ids that REVIEW.md cites against their quotes, and the key ones against the
  sources: every id exists and its quote supports the sentence that cites it.
- **Screening.** 275 screening rows cover **200 distinct sources** (186 arXiv ids). By the reviewer's own prior list,
  built from every file of the four earlier reviews, **176 are new** and 24 prior; the run's flags agree on every source.
  15 new sources were read in depth or in full, 23 with prior ones. The four required full reads are recorded as full:
  ProRL and Invisible Leash (both prior), What Outcome RL Buys and IFBench.
- **Over the caps.** 176 new sources is about 3× the pre-registered screening stop (60; the brief said 40–60), and 23
  in depth is over the in-depth stop (18) if prior sources count (15 new). The review calls it "thoroughness, not
  harm"; the log records it as a deviation with its reasons (the IF agent covered all 19 awesome-RLVR-boundary entries
  and 50 of IFBench's 225 citers, and in-depth reads were cheap once fetched).

### Q2: ProRL's correlation on our data

`dpo` against an RL checkpoint, olmo-ifomega's fixed phase (`dpo` 16 draws per prompt, each RL checkpoint 8), strict
grade unless marked. Spearman ρ with its 95 % bootstrap interval (for the null, the 2.5–97.5 % range of its 2,000
permutations), from the reviewer's own code; the executor's values
agree to within 0.003, with intervals inside bootstrap noise.

| task / grade / RL checkpoint | n | P1 (split, pass@1) | same draws | pass@8 split | `dpo` ≥ 1 / 16 only | random-pairing null |
|---|---|---|---|---|---|---|
| IFBench strict `s2300` | 218 | **−0.505 [−0.603, −0.396]** | −0.537 | −0.667 [−0.759, −0.573] | n 111: −0.518 [−0.680, −0.339] | **−0.654 [−0.709, −0.600]** |
| IFBench strict `s0750` | 218 | −0.255 [−0.378, −0.124] | −0.266 | −0.494 | −0.478 | −0.624 |
| IFBench strict `s0100` | 218 | +0.014 [−0.105, 0.138] | −0.033 | −0.114 | −0.013 | −0.630 |
| IFBench loose `s2300` | 218 | −0.558 [−0.644, −0.456] | −0.595 | −0.605 | n 135: −0.542 | −0.690 |
| IFEval strict `s2300` | 150 | −0.655 [−0.775, −0.502] | −0.794 | −0.610 | n 139: −0.637 | −0.756 |
| IFEval strict `s0750` | 150 | −0.513 [−0.667, −0.350] | −0.675 | −0.364 | −0.588 | −0.700 |

- The executor's P1 is −0.504 [−0.599, −0.397]. Splitting into the actual draws 1–8 and 9–16 gives −0.503 / −0.523.
  The per-draw grade files of the 7 fixed-phase jobs, fetched from the bucket with matching sha256, give the same
  success counts as `results.csv` on every (model, prompt): 0 mismatches.
- **By bin** (IFBench strict, `s2300`): the mean gain is 0.538 where `dpo`'s half-sample pass@1 is 0 (117 prompts on
  average), 0.455 in (0, .25], 0.444 in (.25, .5], 0.254 in (.5, .75] and −0.000 above .75 (46). 69 % of the summed
  gain (59.62 of 85.81) is on the 107 prompts where `dpo` has 0 of 16; 29.50 is on the 49 strict-reach prompts.
- **Per constraint type** (IFBench strict `s2300`, 47 instruction ids with ≥ 2 prompts, pass@8): −0.803 [−0.906,
  −0.627]. It depends on the definition: one type per prompt gives −0.769, shared draws −0.884, and pass@1 −0.572
  [−0.772, −0.309].
- **ρ(x, RL's own rate):** +0.500 / +0.741 / +0.880 for `s2300` / `s0750` / `s0100`. Pass@1 levels (IFBench strict):
  `dpo` 0.290, `s0100` 0.315, `s2300` 0.683; IFEval strict `dpo` 0.769, `s2300` 0.933.
- *Interpretation (the review's):* the observed P1 is less negative than the random-pairing null, so RL's per-prompt
  success is positively related to `dpo`'s. The "largest where the start is weakest" pattern on our data is what a
  bounded gain produces, and it does not by itself separate creation from sharpening. `s0100` shows no pattern because
  it has barely moved. ProRL's own Fig. 3 regresses a pass@128 *difference* on its base value (legend R² = 0.879, in
  the image only), so the same bounded-difference artefact applies to ProRL's R².

### Q2: ProRL side by side, and the Invisible-Leash objection

- **Setups.** ProRL v1 is R1-Distill-Qwen-1.5B with > 2k RL steps, KL with reference resets, 10k IF prompts with a
  continuous reward, and claims at pass@128 (256 samples). Ours is the 32B SFT + DPO start with 2,300 steps of OlmoRL
  without KL ("No KL loss: We remove the KL loss", the Olmo 3 report [PR-137]) over maths, code, IF and chat. The 32B DPO start was
  sampled to 256 (1,024 on 10 IFBench prompts in olmo-iffu); C1's 7B base went to 512.
- **ProRL's correlation claim (L-2, holds):** ProRL v1 and its NeurIPS camera-ready report "a significant negative
  correlation" with no coefficient, test or p-value in the text. "Significant" is ProRL's word.
- **Its zero-base tasks partly fall to the critics:** in Invisible Leash's re-test, ProRL's start
  (R1-Distill-Qwen-1.5B) reaches 1.000 on dice at k = 8,192 (ProRL's Fig. 4 lists dice among the "Sustained" gains).
  On boxnet, where ProRL says "the base model exhibits no capability" and its final model's pass@k is ≈ 1.0 (ProRL
  §4.3, Fig. 5), Leash's base reaches 0.345 at k = 16,384, and ProRL keeps a lead.
- **Does our IFBench reach survive an Invisible-Leash-style objection? Partly answered** (the review). Leash re-tests
  no IF.
  - *Format tolerance:* open. The checkers' loose grade credits `dpo` on 25 of the 49 strict-reach prompts in its 256
    draws (≥ 2 on 19, ≥ 13 on 4, 256 / 256 on 2; 24 at 0), and 16 of olmo-ifomega's executor's 42 hand-reach prompts
    have ≥ 2. None of these loose-only successes has been read.
  - *Larger k:* olmo-iffu settled k = 1,024 on 10 IFBench reach prompts: 0 genuine in 1,024 `dpo` samples each
    (≤ 2.92 × 10⁻³ per sample), Leash's non-maths budget, yet 4 of the 10 are prompt-elicitable. Untested at 1,024: 33
    of olmo-ifomega's reviewer's 43 hand-reach prompts (34 of its executor's 42). k > 1,024 is untried.
  - *Prompt format,* the analogue of Leash's clarified prompts: olmo-iffu's 60-word hint makes `dpo` prompt-elicitable
    on 12 of the 42 hinted IFBench reach prompts (11 under the strict title-case reading); 2 (`ifbench_248`, `ifbench_269`) survive both the hint and 1,024
    samples.
- **Compute-matched budget** (REVIEW objection 1): K ≈ 7.8 × 10⁵ draws per prompt (Olmo 3.1 Think 32B's RL ≈ 1.4 ×
  10⁵ H100-hours from rl-compute-share; 2.97 H200-seconds per `dpo` draw on olmo-ifomega's reach candidates). It
  reproduces as an order of magnitude; the units are mixed, and those draws averaged 5,442 tokens against a typical
  `dpo` IFBench draw of 4,448, so K is ≈ 20 % low if cost scales with tokens. REVIEW claims only "not reached at 256",
  never "created".

### Q3–Q6

- **Q3, a large-k boundary for instruction following:** found in the aggregate, so L-1 is **falsified** by its own
  falsifier. HiR (2512.23457, Fig. 3a) plots IFBench pass@k for Qwen3-4B-Instruct-2507, initial model against RL, at
  k = 1 … 1,024, and ProRL plots IFEval to k = 256. Both curves are aggregate. No per-prompt IF reach study at k ≥ 256
  was found; the closest per-prompt one (Wei 2608.00220) stops at k = 32.
- **Q4, domains within one RL run (L-3, holds):** Guru (2506.14965, one 7B model: AIME crosses its base while other
  tasks expand), ProRL §4.2 (diminish, plateau or sustain by task within one run), Zhai (2604.14877: agent RL on
  Qwen2.5-7B-Instruct gains on its trained tool task, not on MATH-500), and from the review's fresh search
  AceReason-Nemotron 1.1 (2506.13284: one 7B SFT-start run, from Qwen2.5-Math-7B, whose AIME 2025 gain falls from
  8.3 % at K = 8 to 1.2 % at K = 128, while LiveCodeBench v6 keeps +5 % at K = 128; the benchmark names are from the
  review's `fresh/RC.tsv` and the paper's §4.5.7). The Olmo 3 report's gains of 4+ points on AIME and 20 on IFBench
  (DT-16) are the same RL run's 750 → 2,300 extension measured as benchmark pass@1: a domain contrast in pass@1 gains,
  not reach and not independent.
- **Q5, gaming and grading (L-4, holds narrowly):** no source quantifies gaming among the successes that decide reach
  on IF. The closest is Wei's App. C.2: an LLM judge finds that 73.5 % of an IF-RLVR checkpoint's verifier-passed
  rollouts "rely on superficial shortcuts", and shortcut responses rise from 4.5 % under the base policy to 40.1 % at
  the final checkpoint (7–8B Qwen bases): base against RL, but not at reach. Other screened IF figures are short of
  reach too: IFDecorator's 14.53 % trap hack rate, and LIFT's 38 % of high-scoring completions under 30 words. Maths has
  guessing filters (Yue, Cover@τ) and judged reasoning (CoT-pass@k, about 70 % valid reasoning among RL's correct
  answers), and an audit of CoT-pass@k's judges finds corrupted chains accepted.
- **Q6, trajectories (L-5, partly held):** step-resolved large-k curves exist, so the checkpoint-trajectory method
  "extends". The 0.6 clause failed: 2611.02574 (a COLM 2026 workshop paper on arXiv on 2 Nov 2026, five days before the
  run, so concurrent rather than prior) reads all 13 public OLMo-2-1B RLVR checkpoints, with pass@256 on 5 (GSM8K
  coverage 94.5 → 96.9 %). 2610.31624 uses Olmo 3 7B Think's RLVR checkpoints for behaviour only. Olmo 3.1 Think 32B
  has 46 public `step_*` branches (`step_0050` … `step_2300`, every 50); a LessWrong post reads ten of them (steps 50 to
  2,300) for verbalized eval awareness [DT-97], and no boundary or compute-axis study of them was found.

### The claims ranked (the review's wording, with the closest prior work)

| claim | label | closest prior work | what is new |
|---|---|---|---|
| C3, IFBench reach | **novel**, per prompt | HiR (aggregate IFBench pass@k to 1,024) and ProRL (aggregate IFEval to 256), so not "the first large-k IF comparison"; Wei (per prompt, k = 32); Yoon 2610.19525 (per prompt on IF but at k ≈ 1: its Fail→Pass prompts are in its own IF-RLVR runs; its Olmo 3 DPO→RLVR comparison is aggregate) | a per-prompt, confirmed, hand-gated reach measurement on IF at k = 256 |
| C4, gaming | **novel only in a narrow form**; many readers would say "extends" | Wei's 73.5 % shortcut share among an IF-RL checkpoint's verifier passes; Yoon (constraint satisfaction up and judged quality down on Olmo 3 DPO→RLVR pairs, and a 2.05-point quality loss on Fail→Pass prompts in its own runs); lone-win audits in physics (What Outcome RL Buys) | the gamed share among reach-deciding IF successes, and reach re-counted after it |
| C6, failure reading | **novel for IF** (an unreviewed number); no IFBench precedent | SciIF (2601.04770) reads pre-RL IFEval failures as "conversational packaging and elaboration" [NV-12] (IFEval, not IFBench reach prompts); ProRL's zero-base failures are partly format [PR-66]. The review's blind agent RB found other analogues only for maths or multiple choice, e.g. 2610.20639 ("Same Answer, New Box": 82 % of an out-of-domain multiple-choice RL gain is answer format) | a blind reading of the start's failures on IFBench reach prompts |
| C7, methods | **extends** | every tool has prior use: symmetric fresh-draw confirmation (2609.22547), reader gates (CoT-pass@k, Yue, 2610.20639), guessing baselines (Yue, Cover@τ, 2410.15466), public lab checkpoints (2611.02574, 2610.31624; Olmo 3.1 Think 32B's for behaviour, DT-97) | the combination, and reach at three public steps of a 32B run |
| C8, olmo-iffu | test 1 **replicates**; tests 2–4 **extend** | finishing (What Outcome RL Buys); cue elicitation (Wang 2610.06851) and GEPA's IFBench prompts | tests 2–4 on reach prompts |
| C1, 7B RL-Zero | reach (under RL's verifier) **replicates** Yue et al.; "nothing lost" **contradicted, weakly** | Yue's pretrained-base AIME runs; narrowing in Qwen zero-RL work (2606.15455, 2607.20543); a locked 1–3 % minority predicts ≈ 1–3 reverse problems among 83 (1.1–2.6 by the review's arithmetic); C1, on the Olmo 3.1 7B RL-Zero-Math successor run and under RL's checker, in turn contradicts the Olmo 3 report's reading of its pass@32 gain | hand-read large-k resolutions; reverse 0 at k ≤ 512 |
| C2, 32B AIME | **replicates** (no maths reach) | ProRL's maths "Diminish", What Outcome RL Buys; and the closest prior result for the same model pair, the Olmo 3 report's Table 14 (below) | the 32B SFT + DPO start; three grades |
| C5, brevity and token cap | the token-cap half **replicates**, the brevity half **extends** | finishing (What Outcome RL Buys) and truncation (2609.22547) for the cap; RL answers are short (IFBench paper, 210 tokens) and drop preambles (SciIF), but neither ties brevity to the RL-only prompts | the IF-reach version |

- **C2 against the lab's own numbers.** The Olmo 3 report's Table 14 gives AIME 2025 pass@1 DPO 70.7 → Olmo 3.1 Think
  78.1 (+7.4) and AIME 2024 76.0 → 80.6 (its grade "EM Flex" with Minerva extraction, T 0.6, top-p 0.95, 32 samples).
  olmo-think32b's fixed phase (`olmo32/results.csv`; T 1, top-p 1), `dpo` → `s2300`:

  | grade | AIME 2025 (30) | AIME 2026 (29) | all 59 |
  |---|---|---|---|
  | RL's verifier | 0.696 → 0.742 (+4.6) | 0.737 → 0.750 (+1.3) | 0.716 → 0.746 (+3.0) |
  | final stated answer | 0.667 → 0.738 (**+7.1**) | 0.707 → 0.741 (+3.4) | 0.686 → 0.739 (+5.3) |

  So our own final-answer grade reproduces the lab's AIME 2025 gain. The "barely" of proposal 38's C2 ("RL barely
  moves it") comes from RL's lenient verifier, which credits `dpo` with 0.696 against 0.667 on AIME 2025, and from the
  2026 problems. The reach result (no maths reach; pass@8 ≈ 0.86 for both) is unaffected.
- **C1's labels.** The review's fresh-search agent RC called C1 "novel" because it found no large-k per-problem test of
  this model pair. Under the pre-registered definition (same measurement, same kind of task and start model), Yue's
  pretrained-base AIME runs make the reach result a replication; RC's "novel" concerns the setting.

### The review's blind fresh search

- Three agents saw only the claims C1–C8 with the reviewer's glosses; nothing of the run's write-up, ledgers or
  sources. They returned 54 quotes (all found by the reviewer's normaliser; 0 / 54 altered-quote controls passing)
  from 31 distinct arXiv ids, 27 of them among the run's 200.
- They leave C3 and C6 novel. RB calls C4 "extends", with only its reach-boundary measurement new, which is the
  REVIEW's own narrow form; RB did not find Wei's 73.5 %. RA calls C5 / C8 "extends" (REVIEW had C5 and C8's test 1
  as "replicates"; C5 is now split). RB and RC call C7 "extends", as REVIEW does.
- RA's reply wrongly listed HiR among IF-RL papers with no large-k analysis; the reviewer checked HiR's Fig. 3a, so
  the run's neighbour is right.
- Four papers the run did not screen, none of which changes a label: 2610.20639 (C6 / C7), 2506.13284
  (AceReason-Nemotron 1.1; Q4 evidence the run lacked, since it had dropped AceReason 1.0 for "no common
  denominator"), and the tangential 2505.22203 and 2305.01210.

### Expectations against outcomes (the review's scoring)

| id | pre-registered | outcome |
|---|---|---|
| C-1 | P1 negative, interval excluding 0, ρ ∈ [−0.60, −0.15] | held (−0.505 [−0.603, −0.396]) |
| C-2 | naive more negative than P1 by ≥ 0.03 | held narrowly: 0.032 (Monte Carlo s.e. of P1 ≈ 0.001). REVIEW says "held"; in the review's words, saying "narrowly" would be fair |
| C-3 | `dpo` ≥ 1 / 16 subset ρ ∈ [−0.70, −0.10] | held (−0.518) |
| C-4 | zero-bin gain ≥ 0.35, top-bin gain ≤ 0.15 | held (0.538 / −0.000) |
| C-5 | \|P1\| < \|null ρ\| (p 0.7) | held (0.505 < 0.654) |
| C-6 | IFEval `s2300` ρ ∈ [−0.85, −0.30]; IFBench `s0100` \|ρ\| < 0.20 | held (−0.655; +0.014) |
| C-7 | per constraint type negative, ρ ∈ [−0.80, −0.20], wide interval | **missed narrowly** (−0.803 under the pre-registered pass@8 definition with all ids; −0.769 with one type per prompt). On the boundary; REVIEW reports it as a miss |
| L-1 | no per-prompt start-vs-RL IF boundary at k ≥ 256 (p 0.65); falsified by one source with IF pass@k at k ≥ 256 for a start and its RL model | **failed** by its own falsifier (HiR, aggregate); the per-prompt clause held |
| L-2 | ProRL's correlation qualitative (no coefficient or test in the text); Invisible Leash finds the base recovering at large k "on at least one task ProRL lists among its largest gains" | held (no coefficient in the text; dice, boxnet) |
| L-3 | ≥ 2 papers with within-run domain dependence | held (Guru, ProRL; also AceReason-Nemotron 1.1) |
| L-4 | no source quantifies gaming among reach-deciding IF successes (p 0.6) | held narrowly (Wei's 73.5 % is the closest) |
| L-5 | step-resolved large-k curves exist ("extends"); p 0.6 that no paper uses a lab's public RL checkpoints for a boundary study | **partly held**: the 0.6 clause failed (2611.02574, concurrent) |
| L-6 | ranking; p 0.5 that C3 survives as "novel" | partly held: C3 survives (the p-0.5 clause held); C5 is better split |

The executor first scored L-1 and L-5 "held" and corrected both after its first claim check; the log reports this.
The run's own summary still says "L-1 and L-5 failed" (L-5 partly held).

### The review's corrections to the executor's files

Applied in the reading copy's REVIEW.md, marked *[review]* (the copy's README has the table with locations):
1. Name Wei's 73.5 % IF shortcut audit as C4's closest neighbour in the bottom line (and in the run summary's
   "Neighbours", which is not copied).
2. Add the Olmo 3 report's Table 14 to C2, with the grade split above.
3. Split C5's ranking: the token cap replicates, brevity extends (Matrix B, §4, §7; also wholesale in `STATUS.md`'s
   Done line and the run summary).
4. Relabel DT-16 as the same run's pass@1 gains (Matrix B, §3 Q4).
5. Make the run summary's L-5 "partly held" (REVIEW §7 already says so).
6. Say which hand-reach set each number uses: olmo-ifomega's executor's 42 (`verdict.csv`; REVIEW's "42 hand"), its
   reviewer's 43 (40 shared; "33 of the 43", "26 of 43") or olmo-iffu's hint set of 42 (the reviewer's 43 minus
   `ifbench_281`; "12 of 42"). On the executor's 42, 34 are untested at 1,024, not 33.

Minor: `numbers.md`'s "25 of them" should be "25 of the 200 (24 new)"; §6's "within 512" should be "within 383"; cite
"no KL"; close §3's parenthesis; and one sentence on ProRL's own R² (in the copy). Also: the IFEval per-type interval's
upper bound (25 types) is seed-sensitive (the executor's −0.082; −0.018 to −0.059 over six of the reviewer's seeds).

*[librarian]* For "cite 'no KL'", the review noted that REVIEW §2's "without KL" carried no ledger id; the ledger
already holds the quote as PR-137, which the copy now cites.

## Limitations

- "Novel" means "as far as found": 200 sources screened by the run and 31 by the reviewer's blind search, against a
  pre-registered definition (the same measurement on the same kind of task and start model). C4's novelty rests on
  that definition being narrow.
- The ledger shows that every quote is real and in its source; whether a quote supports its sentence was checked on 21
  random rows and on the 74 ids REVIEW cites. Figure readings are figure-only, and ProRL's R² is in an image.
- Our side of the comparison has its own open points, carried over from the Olmo runs: "hand" means model readers with
  no human sample (REVIEW objection 3), C6's 93 % is the olmo-ifomega executor's number, and the loose-only `dpo`
  successes on 25 strict-reach prompts are unread.
- The correlation is descriptive. Its random-pairing null shows, on our data, that the pattern cannot separate creation
  from sharpening; the review argues that the same bounded-difference artefact applies to ProRL's own R² (a pass@128
  difference regressed on its base value), which it did not test.
- The closest trajectory paper (2611.02574) appeared five days before the run. Later work is not covered.

## Next steps (the review's next measurements; not launched)

1. **Format tolerance:** blind-read `dpo`'s loose-only successes on the 25 strict-reach prompts. They are stored, so
   this costs $0 and needs no pods. It is the open half of the Invisible-Leash answer.
2. **Larger k on the hand-reach prompts never tested at 1,024:** 33 of olmo-ifomega's reviewer's 43 hand-reach
   prompts, or 34 of its executor's 42 (`verdict.csv`).
   That would put every reach prompt under Leash's non-maths budget, and needs pods. *[librarian]* At olmo-iffu test
   3's average cost (≈ $0.0023 per IFBench `dpo` sample on a community H200), 768 more samples on each of 33–34 prompts
   come to ≈ $58–60 (unreviewed arithmetic). olmo-iffu's next step 3 (the 17–19 IFBench prompts the hint neither
   elicited nor broke) covers part of the same prompts.
3. **C2:** the grader part is settled from stored data (+7.1 by final answer on AIME 2025, against the lab's +7.4).
   Only the sampling difference (T 1 / top-p 1 against 0.6 / 0.95) would need pods, and it does not bear on the reach
   claim.
4. **A human read** of a sample of the gamed / genuine labels behind C4, to answer REVIEW objection 3.

For the paper, REVIEW §6 is a ≈ 600-word related-work draft with the review's corrections applied; its citekeys other
than Yue's and ProRL's are proposals until the papers in `to_add_to_zotero.md` are imported.
