---
written_on: 2026-11-28
written_by: agent:claude (independent auditor, Opus 5.5)
---

# Audit of PREREG.md (weight interpolation dpo -> s2300)

## Turn 1 — independent auditor, 2026-11-28 3:15 AM PT

Read: PREREG.md; metrics 2 (`X` = `olmo_metrics2_20261127`) `rank1/{expo_write,common,stats,rules}.py`, `sample_wrap.py`,
`tools/{grade_job.sh,gz_repair.py,answer_style.py,ledger_ok.py}`, `pod/{m2_job,m2_chain,m2_download,m2_setup}.sh`,
`pod/m2_verify_weights.py`, `pod/README.md`, the chain files, `tests/test_rank1_expo.py`, log.md, CONTEXT.md, SPEC.md;
Dan's `sample.py`, `fetch.sh`, `checkpoints.csv`; `oe_setup.sh`; the fetched Xheld and h200 records; `handcheck.py`
`load_cell` (read only). Experiments: CPU scripts in my scratchpad (`audit/t_writer.py`, `t_gap.py`, `t_one.py`, `t_time.py`,
`t_rule.py`, `t_rule_tight.py`, `t_pred.py`), tiny bf16 fixtures, at most 2 threads.

### Findings, by severity

1. **BLOCKER (reading rule, cheap to fix now): "late switch, as in the small provers" is the null outcome of a smooth
   path, so the headline cannot separate a switch from plain log-odds growth.** The note's own model (each prompt's
   log-odds linear in alpha) gives r(0.5) = 0.17-0.18 per prompt with the stored s2300 rates (`t_pred.py`: 12.1 % at
   0.5, 41.6 % at 0.75; 0.10-0.11 only with one pooled 69.4 % rate, which is where the note's 1.4 / 7 / 30 % come from).
   In simulation smooth paths are called "late" in 99-100 % of runs (`t_rule.py`: true r 0.20 and a clipped logistic
   variant with r 0.27; `t_rule_tight.py`: the smoothed logistic, r 0.17). The small provers had r(0.5) = 0.31 / 19.0
   = 0.016, ten times lower. Fix: name the verdicts by what they measure ("midpoint below 35 % of the endpoint") and drop
   "as in the small provers"; pre-register the per-prompt logit-linear reference curve from the stored endpoint rates
   (and its r(0.5), r(0.75)) and report the observed curve against it, so "sharper than logistic" becomes a stated test.
2. **SHOULD FIX: the budget gate makes the last cell unlikely, and writes run before their gate.** The gate passes only if
   spend_now + $7.94 <= $21, that is when a cell starts before 2.47 pod hours; the note expects 3 hours in total, so a025
   (last) is at or past that line in the expected case. "Write, sample, delete, next" serialises two of the three
   writes with the GPU idle. A full 707-tensor verify is also not 4 minutes: the metrics 2 write with 3 tensors checked
   took 4 min (write_g35 runner.log 15:22:03-15:26:11Z), and `verify_tensors` ran at 9 / 15 million elements per second
   on 1 / 2 serrano threads (`t_time.py`), so about 35-60 min, perhaps 20 min on 4 threads (an extrapolation), per
   checkpoint on 32.2e9 elements. Fix: time one full verify early and re-plan; overlap each write with the previous cell
   (two merged copies fit a 500 GB volume, finding 7); put the ledger check before the write and count write + verify +
   the 6,300 s job timeout, not 90 min only; or ask for a cap that fits the expected run.
3. **SHOULD FIX: check 2 ("every element within 1 bf16 step") cannot see the interpolation it is meant to certify.**
   `verify_tensors` passes when `worst <= 1.0` (expo_write.py:78) and only records `n_ne`. A step error is accepted, two
   are not (`t_writer.py` part D). For a tensor whose elements differ by at most one step (likely many norm weights near
   1.0, where a step is 2^-7), writing it wholly at `dpo`, wholly at `s2300` or at 1 - alpha passes: alpha 0.25 with the
   tensor at s2300 gives verify_ok True with n_ne 512 of 512; alpha 0.75 with it at dpo the same (part C). Matrices with
   larger steps do catch a wrong alpha. The writer and the verifier's reference agreed bit for bit on all 20,000 test
   elements at 1/4, 1/2 and 3/4, gap pairs included (`t_gap.py`), so the gate can be exact. Fix: fail on any `n_ne > 0`
   per tensor (report max ulp beside it), and add per class the counts of elements at dpo / at s2300 / between.
4. **SHOULD FIX: the cut-cell rule's two imputations are not bounds on r.** "Missing as fail" and "missing as pass"
   applied to both cells move numerator and denominator together. The extremes of r are (alpha 0.5 missing = pass,
   alpha 1.0 missing = fail) and the reverse. Also define "as sampled" for a prompt with no finished draw: `prompt_rates`
   without `planned_per_prompt` drops such a prompt (common.py:158) and `_aligned` then drops it from every cell
   (stats.py:14), silently. Fix: use the two cross imputations; always call `prompt_rates(..., 4)` and `missing_bound`.
5. **SHOULD FIX: the no-resume rule is not enforced by the tools.** The chain skips only jobs with `state/DONE`
   (m2_chain.sh:27); a cell that ended by its timeout has `state/FAILED` and runs again if the chain is re-submitted
   (for example after a ledger stop), and Dan's sampler then fills only the missing draws (sample.py:189-205, 216):
   the retry bias the note forbids. Fix: each cell command refuses when its `samples.jsonl.gz` exists (or writes a
   `CUT` marker and exits 0 so the job is DONE), and the analysis checks one sidecar `utc_start` per cell.
6. **SHOULD FIX: `ledger_ok.py` defaults belong to yesterday's pod.** `--start-utc 2026-11-27T15:13:17Z`, `--prior 6.40`,
   `--cap 60` (ledger_ok.py:20-23). A line that leaves out `--cap` spends up to $60; one that leaves out `--start-utc`
   skips every cell after paying for set-up and downloads. Fix: pass all four flags in every line and test the chain
   file for them (or a W wrapper with no defaults).
7. **SHOULD FIX: the writer's disk check measures the wrong thing, and the note gives no volume size.** On the earlier
   H200 pod `df` showed `mfs#us-nc-1.runpod.net:9421 1.2P 852T 350T 71% /workspace` (pod_env.txt), and the volume was
   500 GB (log.md:177, 287). `shutil.disk_usage(out).free` (expo_write.py:98) therefore reads about 350 TB, the
   MooseFS cluster, and can never refuse. Needs: 129 GB (dpo, s2300) + 64.5 GB per merged copy; with "a cut cell stays
   on disk, the chain goes on" and overlapped writes, up to 4-5 copies (258-322 GB). Fix: request >= 400 GB (metrics 2
   asked for >= 450, SPEC.md:64), state it in the note, and check `du -sb /workspace` against that number.
8. **SHOULD FIX: report the rule's own error rates or tighten it.** Simulated with the stored per-prompt s2300 rates,
   48 x 4, paired bootstrap (`t_rule.py`, 1,000 data sets x 2,000 resamples): the interval of r(0.5) is 0.14-0.21 wide
   (0.33 if half the prompts are at s2300 and half at 0), so the interval conditions almost never bind and "unclear" was
   never returned. The verdict is the band of the point estimate: true r 0.30 is called "about linear" 14 % of the time,
   0.40 "late" 14 %, 0.60 "early" 17 %. With "late" needing hi < 0.35 and "early" lo > 0.65 (`t_rule_tight.py`), r 0.30
   becomes "unclear" 59 % and "late" 27 %. Either is defensible; the note should state which errors it accepts.
9. **NOTE: what the a100 yardstick leaves uncontrolled is small and checkable on CPU.** It serves s2300's own files; the
   merged folders carry dpo's (expo_write.py:119-129). Checked by blob ids (checkpoints.csv, `git hash-object` of the
   local copies): config.json differs only in `beta_fast` 32 vs 32.0, `beta_slow` 1 vs 1.0, `use_cache`;
   generation_config.json only in `do_sample` (both eos 100257, the only field vLLM takes under
   `generation_config="vllm"`); tokenizer_config.json only in `chat_template` (not used: `--tokenizer
   /workspace/models/tok`, Dan's own template); the index blobs differ (14 shards each; no effect on values). Check 3 names
   config.json only; add generation_config.json. The writer route itself is not exercised at alpha 1: add a pod check
   that `combine(dpo, s2300, 1.0)` equals s2300 bit for bit per tensor. Expect a few exceptions: where |s| << |d| the
   fp32 difference is inexact and alpha 1 does not return s (`t_one.py`: d 1.0, s 1.4e-9 gives 0); count them in the census.
10. **NOTE: equal seeds couple the cells; the note says the opposite.** Dan's requests carry no seed (sample.py:279), so
   vLLM samples every running request with exponential noise from one generator seeded once (`seed=` at
   sample.py:261-265; `random_sample` read in a local vLLM 0.27.1, not in 0.11.0). Same seed, prompts and order give the
   same noise at each step until the batch sizes differ, so draws with the same key share early tokens across cells
   wherever the models agree. Marginal rates and the prompt bootstrap stay valid; do not call the cells independent,
   and tell the blinded reader that identical openings across cells are expected.
11. **NOTE: rounding distorts 0.25 and 0.75 in opposite directions, and the half-gain point inherits it.** Verified on
   fixtures: one-step differences stay at dpo at 0.25, go to s2300 at 0.75, go to the even neighbour at 0.5; two- and
   three-step cases round as expected (`t_writer.py` part A). Inferred: if one-step weights carry a share w1 of
   ||s2300 - dpo||^2, the realised fraction is about 0.25 - 0.25 w1 at 0.25 and 0.75 + 0.25 w1 at 0.75, unbiased at 0.5,
   so the 0.05 flag fires only if w1 > 0.2. Report the half-gain point on the realised axis too, and the at-dpo /
   at-s2300 / between shares per class (norms, embeddings, lm_head, matrices), since norms near 1.0 may barely move.
12. **NOTE: verifier reference.** For alpha 1/4, 1/2, 3/4 the fraction is exact and q a + p (b - a) over q is exact in
   fp64, but `.float().to(bfloat16)` (expo_write.py:70) rounds twice; in gap-and-tie cases it agrees with the writer and
   both are one step from the true rounding (35 and 44 of 400 synthetic gap pairs at 0.25 and 0.75, none of 19,600
   ordinary pairs). Harmless in size; do not call it an independent exact reference there.
13. **NOTE: cut cells lose records the completeness rule needs.** Dan's meta (seed, model_dir, prompt sha256) is written
   only after the last draw (sample.py:340-361); a cut cell has only the wrapper sidecar (argv at start) and a truncated
   gzip. `load_cell` (handcheck.py:122-125) has no guard for a truncated gzip or line: point it at
   `samples_complete.jsonl.gz` from grade_job.sh, and recompute prompt hashes offline for cut cells.
14. **NOTE: run-time hazards.** `write()` does not set torch threads (only `main()` does, expo_write.py:166); imported
   from interp_write it uses all visible cores (192 nproc vs a cgroup quota of about 20 on the earlier pod): set them.
   Peak memory measured at about 2 GB for one 21-million-element tensor, so the verify of lm_head or embeddings
   (513 million elements) needs about 40-50 GB of RAM (extrapolated). The inner 90-min timeout must not be chained
   with `&&` before grading (the Xheld line pattern), or a cut cell is left ungraded.
15. **NOTE: yardstick precision and labels.** r(0.5)'s noise comes partly from the 4-draw a100 denominator (alpha 1.0
   interval about +-10 points, as the note says, confirmed in simulation). Report r against the stored 32-draw s2300
   (69.4 %) as a sensitivity line, not as the verdict. Give the a100 cell a `--name` other than `s2300`, so its rows
   cannot be mixed with Dan's stored rows.

### Short answers

1. Engine: yes for the vLLM build, GPU, sampler, seed, request layout, tokenizer, prompts and grader, which the stored
   69.4 % does not share. It does not control for the writer route (arithmetic, serialisation, dpo's copied files),
   bf16 sub-step rounding, or CPU and disk load from concurrent writes (speed only, hence cut risk). The file
   differences are cosmetic (finding 9); one CPU bit-equality check at alpha 1 closes the writer gap.
2. Writer: valid for alpha in (0, 1). Inputs are bf16 (dpo index total_size = 2 x total_parameters; s2300 has the same
   byte count, and a dtype mismatch would hit the assert). fp32 combine, one RNE rounding: correctly rounded on every
   test element except exponent-gap pairs (at most one step there). Unchanged tensors come back unchanged. This is the
   ExPO route: every tensor in dpo's weight_map goes through `combine`, all 707 (448 matrices, 257 norms, embedding,
   lm_head, counted in the local dpo index); missing names abort the write. The verifier would not catch an
   endpoint-valued one-step tensor (finding 3). On the pod: per-tensor dtype and shape equality, no non-float tensors,
   name-set equality, n_ne = 0 on all 707, the alpha 1 bit check, step histograms per class, the copied file
   diffs, and vLLM's load log with no missing or unexpected weights.
3. Seeds and settings: identical by design (seed 710012, T 1.0 and top_p 1.0 as module constants, 32,768 tokens, same
   prompts, max_model_len from the prompts, gpu_mem 0.92, prefix caching, same eos). Nothing left would move a pass rate
   if the four lines come from one template; check afterwards that the sidecars' argv differ only in model, labels and
   paths. The equal seed couples draws (finding 10); it does not bias rates.
4. Rule: mutually exclusive and exhaustive; zero passes at 0.5 gives r = 0 with interval [0, 0] ("late", defined); a
   zero denominator is covered, a near-zero one leaves bootstrap draws that `paired_boot` silently drops (stats.py:27),
   which must be stated; a non-monotone curve does not affect the verdict. 48 x 4 reaches a definite verdict in nearly
   every simulated case, "unclear" almost never (finding 8). The cut-cell evaluation needs the fix in finding 4.
5. Disk: 194 GB for one merged copy fits a 500 GB volume, and so do the 258-322 GB of the cut-cell and overlap cases,
   but the note states no size and the writer's check reads the 350 TB of free space on the cluster (finding 7).
6. Other: budget arithmetic (2), resume (5), ledger defaults (6), cut-cell records (13), threads and RAM (14), the
   verdict label against a logit-linear null (1). The selection of prompts on dpo failing is stated in the note;
   the s2300 endpoint is unbiased by that selection. ExPO-style unpunctuated passes: compare answer style with the
   a100 cell, not in absolute terms (stored s2300 reach passes are 38.7 % unpunctuated, median 32 words, answer_style.txt).

### Not verified

- vLLM 0.11.0 sources for `random_sample` and `update_from_generation_config` (read in 0.27.1 only); whether the
  same-seed coupling holds in practice on the pod.
- s2300's per-tensor dtypes and the real step histogram (w1, norms): no weights on serrano.
- Pod CPU speed, the container RAM limit, and whether deleting 64.5 GB on the MooseFS volume frees quota at once (trash).
- How often real weight pairs have exponent gaps big enough for the fp32 subtraction to be inexact.
- The new pod's volume size, and the not yet written `interp_write.py`, chain lines and analysis.

## Turn 2 — orchestrator (agent:claude, Fable 5.2), 2026-11-28 3:44 AM PT

Replies by finding. "Applied" means the code or the note changed and a test covers it; the note changes are
Amendment 1 of `PREREG.md` (3:42 am PT, before any pod). Nothing was sampled between the two turns.

1. **Verdict label against a smooth path: accepted in substance, label not changed by me.** I reproduced it from the
   stored per-prompt rates: a per-prompt logit-linear path gives 11.4 % at alpha 0.5 and 40.9 % at 0.75 with a pre-RL
   rate of 1 in 384 (6.2 % and 34.9 % with half a pass in 769), so r(0.5) = 0.09 to 0.17, inside the rule's "late
   switch" band; cells drawn from that path get the "late switch" verdict in my check too. The rule and its labels
   were fixed by the coordinator, so I did not rename them. Applied instead: the reference is pre-registered (A1.5),
   computed by `analyze.smooth_reference`, drawn as a band in the figure, and reported as a second line of equal
   weight ("sharper than the smooth reference" / "compatible with a smooth path" / "earlier than the smooth
   reference"). Only "sharper" supports "as in the small provers" (their r(0.5) was 0.016). The label question is
   put to the coordinator.
2. **Budget gate and idle GPU: applied, with two parts rejected.** Applied: the writes now run in their own CPU chain
   while the cells sample (`make_chain.py --mode lanes`, default), each cell waits for its own write
   (`cell_count.py wait`), and the full exact check became a bit-for-bit recompute on all 707 tensors plus the exact
   verifier on a sample (I measured the verifier at about 45 ns per element on 4 threads, 24 minutes per checkpoint).
   Expected pod time falls to about 1.6 hours, so the last cell starts near 1 hour 15 minutes, well inside the gate's
   2 hours 28 minutes. Rejected: (a) counting the whole job timeout in the gate. The gate is evaluated after the wait,
   so what follows it is the 90-minute sampler plus grading; I kept the coordinator's "90 minutes, $7.94" and state
   that grading and counting (allowed 15 minutes, about 1 minute in practice) are outside it. (b) Asking for a higher
   cap: not mine to ask for; the expected run is far below $21. Not done: timing the check on a pod beforehand; its
   duration there is unknown until the first write (it no longer holds the GPU).
3. **"Within one step" cannot see a one-step tensor at the wrong end: applied, with a small tolerance instead of
   zero.** A tensor now fails when more than 8 elements (or more than 1 in 10 million of it) differ from the
   reference at all, as well as when any element is more than one step away
   (`test_exact_check_counts_mismatches_not_only_their_size` reproduces your case: metrics 2's criterion passes
   it, the new one fails it with 512 of 512). Why not exactly zero: no real weights are on serrano, and one
   fp32-against-fp64 corner element in 32 billion would stop a paid pod for something that is not an error; a wrong
   end or a wrong alpha moves a large share of a tensor. Independently of the verifier, all 707 tensors are compared
   bit for bit with a fresh recomputation and held to two invariants (unchanged where the inputs agree; between the
   inputs elsewhere). The at-dpo / at-s2300 / between shares are in `check.json` per class.
4. **Cut-cell bounds: applied.** The bounds are now crossed (numerator all-fail over denominator all-pass, and the
   reverse); a prompt with no finished draw is counted, kept in both bounds and printed (A1.4,
   `test_a_small_cut_keeps_the_verdict_when_all_bounds_agree`,
   `test_a_prompt_with_no_finished_draw_is_counted_not_dropped`). The analysis has its own per-prompt code and does
   not use metrics 2's `prompt_rates`.
5. **No-resume not enforced: applied.** `pod/wi_cell.sh` refuses to start when rows of an earlier attempt exist
   (exit 2) unless `WI_ALLOW_RESUME=1`, which nothing sets; the chain test re-submits a chain after a cut cell and
   checks that the sampler is not called again. `count.json` also requires exactly one sampler record per cell.
6. **Ledger defaults: applied.** Every cell line carries the pod's start time, rate, prior spend, cap and cell cost;
   `wi_cell.sh` refuses without a start time; the chain-text test checks the flags.
7. **Disk check and volume size: applied.** The volume is 500 GB (RUNBOOK, A1.9). The write step sums file sizes
   under the models folder and refuses if the new checkpoint plus a 20 GB margin would pass that. Everything ever
   written in this run is 322 GB, so the run does not depend on deleted space being released.
8. **Error rates of the rule: stated, rule not tightened.** My simulation agrees with yours (true r 0.30 called
   "about linear" 11 %, 0.40 called "late" 15 %, 0.60 called "early" 17 %, "unclear" never). A1.6 records this, and
   the report adds "near the boundary of the rule" when r(0.5) is within 0.05 of 0.35 or 0.65. Tightening the rule
   is the coordinator's decision.
9. **What a100 leaves uncontrolled: applied.** The token ids of `generation_config.json` must agree
   (`config_check`); the alpha 1 identity is counted with the last write and flagged above 1 in a million. It is
   recorded, not a failure, since it says nothing against a checkpoint with alpha below 1.
10. **Seeds couple the cells: accepted, the note was wrong.** A1.3 corrects it; the hand-off to the blinded readers
    will say that identical openings across cells are expected.
11. **Rounding and the half-gain point: applied.** The half-gain point is also given on the realised axis and the
    realised fraction per tensor class is in `results.json`.
12. **Verifier reference: accepted.** The docs now describe it as the exact fp64 result rounded through fp32 to
    bf16. My own tests check the alpha arithmetic against a separate rational reference with a single rounding.
13. **Cut cells lose records: partly applied.** `cells_manifest.json` points the reader step at the repaired gzip
    for a cut cell, and a cut cell is reported as incomplete with its missing keys. Not done: recomputing prompt
    hashes offline for a cut cell. For such a cell `count.json` says "0 meta lines" and the prompt hashes are
    not checked; the write-up will say so.
14. **Run-time hazards: applied.** Threads are set (`--threads`, 8 beside a sampling cell); every check works in
    chunks of 33.6 million elements; the writer's own three-tensor check draws the same three attention matrices as
    yesterday (same seed and names), not the embedding; grading and counting never sit behind `&&`.
15. **Sensitivity line and labels: applied.** r against the stored 32-draw rate is printed as a sensitivity line;
    the alpha 1.0 rows carry the label `s2300_same_engine`.

Not verifiable before the pod, as in your list: the real step histogram and exponent gaps of the weights, the speed
of the write check on the pod, the container's RAM limit, and whether the same-seed coupling shows in vLLM 0.11.0.

## Turn 3 — coordinator's decisions, recorded by the orchestrator, 2026-11-28 4:01 AM PT

- Finding 1 (verdict labels): the coordinator changed the names. `PREREG.md` Amendment 2 (3:58 am PT, the coordinator's
  text verbatim): "below proportional at the midpoint", "about proportional", "above proportional", "unclear at this
  size"; thresholds unchanged. A switch may be named only when the result is also "sharper than the smooth
  reference". The code, the tests and the runbook use the new names only; a test fails if an old name reappears in
  them. The dated records (the note's text before Amendment 2, and Turns 1 and 2 above) keep the old names.
- Finding 2 (schedule): the coordinator chose the overlapped default, with the sequential schedule as my fallback
  if the pod's memory or disk makes it unsafe (`wi_pod.sh podcheck`; the choice goes in `log.md`).
- Finding 8 (error rates): no change to the thresholds; "near the boundary" stays as recorded in A1.6.
- New since Turn 2, from the coordinator: the pod is shared with a later job. Merged checkpoints now live under this
  run's own folder, the disk budget counts the whole volume, and a release step removes this run's merged
  checkpoints and `s2300`, leaves `dpo`, and writes a "GPU free" marker. The auditor has not reviewed these additions;
  they are covered by tests only.
