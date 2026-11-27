---
written_on: 2026-11-27
written_by: agent:claude (orchestrator, Opus 5.5)
---

# Three elicitation measurements on Olmo 3.1 Think 32B: spec and adjudication

Contract between the orchestrator, the build agents and the pod runs. `CONTEXT.md` is the shared brief; the designs
were argued in `discourse/<key>.md` (reader-implementer: Sonnet; auditor: Opus; each auditor first wrote an
independent read, `discourse/<key>_audit_independent.md`, before it saw the reader's notes). Pre-registrations:
`PREREG_temp.md`, `PREREG_rank1.md`, `PREREG_steer.md`. What ran: `log.md`.

## Adjudication

The design of record for each measurement is the final section named below, with the auditor's closing fixes, and
the orchestrator decisions listed here. Where they disagree, this file wins.

| key | design of record | auditor's closing turn | discourse md5 at adjudication |
|---|---|---|---|
| `temp` (M1) | `discourse/temp.md` Turn 5 Part 3 | Turn 6: fixes 1-8, "no remaining objection" | 9288b782f78db54fdc74939ad8955e4c |
| `rank1` (M2) | `discourse/rank1.md` Turn 5 section B | Turn 6: fixes D1-D8 (PREREG_rank1 Amendment 1), "no remaining objection" | 1d2f46540d6aeee73a1eda5590094095 |
| `steer` (M3) | `discourse/steer.md` Turn 5 section F | Turn 6: fixes d1-d7, "no remaining objection" | 1ff1a983149224275bde6fc2ed6c2590 |

Orchestrator decisions (all three):
1. Every pod step is a separate job started by hand with `pod/m2_pod.sh job` (one GPU job at a time, flock,
   heartbeat, state files), watched from serrano by `pod/m2_watch.sh` (STALL / OVERRUN / EXIT, log only). Scripts
   that apply a pre-set rule print the verdict and exit with a code; the orchestrator starts the next job. No pod ever
   idles waiting for a decision longer than it takes to read a verdict.
2. Draw counts and cut choices are job parameters; the defaults are the designs' full values. Cuts are taken only in
   each design's stated order, before the cell they affect, and named in the write-up.
3. Sampling always goes through `sample_wrap.py` around Dan's unedited `olmo_ifladder/sample.py` (vLLM 0.11.0,
   `olmo_thinker` template, 32,768 max tokens, prefix caching as Dan's); grading with Dan's strict `grade.py` via
   `tools/grade_job.sh`. `--chunk 1` for new cells so rows are written as each draw finishes.
4. Shared cells: M3's E0' (fresh `dpo`, T 1.0, zero steering vector through the patched engine after its exact
   zero-vector identity check, 48 reach x 8 = 384 draws, seed 1000) is also M1's fresh T 1.0 reach baseline, with the
   sidecar checks of `temp.md` Turn 6 and the pre-set proxy fallback. It is never cut.
5. M1's prefill likelihood check (PF) is its own HF job (`tools/hf_score.py`), scheduled next to M3's HF `dpo` session.
   No M1 cell uses a LoRA-enabled engine.
6. M2's single conditional cell (rank-16 adapter versus the lm_head / embeddings / norms reference E) is decided by the
   orchestrator from A1's result, within M2's cap.
7. Trigger-released cells (M1 trigger A: C3 + C4; M1 trigger B: C2; M3's E2) are decided by the orchestrator when the
   trigger script fires, within the budget below; their release rules are pre-registered.
8. Power sampling is not approved and is not built. A proposal may be written only under `temp.md` Part 3 section 9.

## Budget (H200 secure at $5.29/h; estimates, not measurements)

| | planned (no conditional cells) | conditional | cap |
|---|---|---|---|
| shared: pod setup, downloads of dpo, s0100, s0500, s2300, gaps between jobs | about $5 | | |
| M1 temp: C1 (736 draws), S5R (384), PF | about $8.5-9.5 | trigger A (C3 + C4) $6.3; trigger B (C2) $2.9 | $10 without triggered cells |
| M2 rank1: W, gate, A1, A2, A1u, ExPO dev and held-out | about $15.65 (success branch), $11.75 (clean null) + one conditional cell | rank-16 or E about $2.7-3.5 | $15 cumulative (design's cut order) |
| M3 steer: HF P1-P3, pre-flight, dev guard 64 x 1, E0' 384, E1 544 | about $15-16 | E2 $2.6 (reach-only cut (6) if over the cap) | $17 |
| **total** | **about $44** | up to about $12 more | stop and report to the coordinator at $60 of pod spend; the brief's hard limit is $80 |

The total is above the $40 Leon was told; the overrun is the price of the auditors' required controls (M2's random
and unscaled rank-1 cells, M3's fresh reach baseline and manipulation check). The coordinator is told this before any
pod is created.

## Pod needs

- Sampling (all of M1, M2's A1 / A2 / A1u / ExPO cells, M3's dev guard and E cells) needs one H200 (141 GB): a 64.5 GB
  checkpoint on an 80 GB card leaves too little KV cache for 5-7k-token responses.
- HF and weight-space work (M3 P1-P3 and pre-flight, M1 PF, M2 W pass, adapter build and gate) fits an H100 80 GB.
- Disk: dpo, s0100, s0500, s2300 (258 GB) plus two ExPO checkpoints (129 GB) on the sampling pod: volume >= 450 GB.
- Weights never go to serrano. Results are fetched with md5 checks before a pod is released.

## Order (cheapest informative first)

1. M1 C1 and S5R (no build risk; also gives the first measured throughput at T 0.6).
2. M2 W pass and gate (cheap; decides whether the rank-1 cells run), then M2 sampling, ExPO writes in the background.
3. M3 HF P1-P3 (+ M1 PF next to P1), pre-flight, dev guard, E0', E1, trigger, E2 if released.
4. Trigger-released cells last, if released.

If the work is split over two pods (an H100 handed over for step 2's W pass and step 3's HF work, an H200 for
sampling), every step's inputs and outputs are files fetched to serrano and pushed again (vectors and adapters are
small); the vLLM pre-flight of M3 and M2's applied gate are repeated on the sampling pod before its first steered or
adapted cell.

## Where things go

- Code and notes: this folder. Inputs: `/scratch/cowifornia/nd-rl/olmo_elicit/metrics2/inputs/<key>/`. Pod root
  `/workspace/m2` (runs under `runs/<job>/`), weights `/workspace/models/<ckpt>/`. Fetched results:
  `/scratch/cowifornia/nd-rl/olmo_elicit/metrics2/runs/<job>/`.
- Summary: `experiment-summaries/2026-11-27-olmo-elicitation-metrics-2/README.md`.
- Serrano CPU (coordinator, 2026-11-27 about 2:15 am PT): all local work by the orchestrator and its agents together
  within 3 cores, `nice -n 5`, single-threaded BLAS.

## Write-up wording (binding; coordinator's decision of 2026-11-27 after the gate audit, recorded 5:26 am PT)

- Every amended gate is described as **amended after seeing data**, with the original criterion and the failing numbers in
  the same sentence, and as "passed the amended gate (Amendment ...)", never "passed the gate".
  - W: "W failed its exact-SVD check twice (null 5.4 % and 2.5 % low against 0.5 %); per the pre-registered rule the null and
    sigma_1 were then computed exactly". Not an amendment. Never "Wx passed its exact checks".
  - LoRA gate: the original exact-repeat criterion failed for all three adapters (gate2); Amendment 3 (0.1 %) passed gate3
    at 0.082 %; Amendment 3a (0.5 % plus the both-calls token rule) was written before gate4, which measured 0.149 % and
    would have failed Amendment 3; Amendment 4 added fail-closed base exactness, the zero-B control, the A1 10 % rule and
    the decode check. For A1u and A2 the 0.5 % tolerance exceeds their whole teacher-forced effect, so their "applied"
    status rests on the token rule (and the zero-B control).
  - M3 pre-flight: compiled zero identity failed (mean 0.0100 against stock, which itself varies by 0.0101 across
    processes); identity moved to eager (bit-identical at 11,854 / 11,854); the power factor went from 3 to 1.5 after the
    scramble measured 1.71 (chosen 3.06), and was then replaced by the direct dropped-vector test (Amendment 2).
- A1 / A2 / k16 directions come from the randomized SVD (sigma_1 within 0.17 % of exact on all 448 matrices); the exact
  pass changed only the null and A1u's scale.
- A1's change of dpo's NLL on s2300's own text (it raises it: 2,151.6 to about 2,391 nats and 4,495.5 to about 4,612 on the
  two check examples) is a teacher-forced gate observation, never described as movement toward s2300.
- 13 matrices have sigma_2 / sigma_1 > 0.95 (worst layers.52.mlp.gate_proj, 0.984): there "the top direction" is nearly
  degenerate; say so where per-matrix directions are discussed.
