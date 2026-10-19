---
written_on: 2026-10-03
written_by: agent:claude
---

# mcts-b: not run, because mcts-a's gate failed (MCTS as the RL expert, proposal 22 Phase B)

Status: closed as **not run: gate**, and reviewed 2026-10-03. Verdict **stands**. No hard-constraint violation and
nothing quarantined. The run created no pod, spent $0, and changed no code, data or artefacts. The reviewer applied the
gate rule to mcts-a's raw evaluation files and got the same result: **MCTS-A GATE: FAIL**, with Δ ≥ 3 on 0 of 3 seeds.
mcts-a's own reviewer had let that verdict stand. Under the brief, Phase B therefore must not run. Two minor rewordings
are applied below.

Question (proposal 22, Phase B): used as the expert in RL, does MCTS (training on found proofs, B1, or on visit
distributions, B2) teach the policy to solve group-C theorems that expert iteration (EI) does not? It was to run only if
`mcts-a`'s gate passed. Proposal: [`docs/proposals/2026-10-02-mcts.md`](../../docs/proposals/2026-10-02-mcts.md); brief
[`BRIEF_mcts-b.md`](../../docs/proposals/state-env/BRIEF_mcts-b.md); gate:
[mcts-a](../2026-10-02-mcts-a-puct-value-search-gate-fails-on-group-c/README.md).

## Configuration

- **Repository.** Fork `dan-pandori/nd-takehome`, branch `dan_mcts-b` (from the fork's `origin/dan`). Executor commits
  `57f41b9b` and `694bdf37` / `01897f4e` (trims to ≤ 150 words). Files: `run_mcts_b.md` (142 words) and `STATUS.md`
  § mcts-b, which ends `MCTS-B DONE 2026-10-03T05:50:23Z (not run: gate)`.
- **Executor:** Sat 2026-10-03 05:50:04 – 05:50:43 UTC (one invocation). **Reviewer:** 05:52–05:55 UTC.
- **Pods, cost, compute:** none. `podbudget mcts-b` reads 0.00 h, $0.00 of $45 / 90 h, and `~/pods.log` and
  `~/podhours.log` have no `mcts-b` entry. No pre-registration was due, because the brief asks for one only before the
  first pod. No compute rows, no bucket upload: there was nothing to upload.
- **Review:** [`review_mcts-b.md`](https://github.com/dan-pandori/nd-takehome/blob/0ab829fc/review_mcts-b.md)
  ([`c8303819`](https://github.com/dan-pandori/nd-takehome/commit/c8303819) §Recount,
  [`0ab829fc`](https://github.com/dan-pandori/nd-takehome/commit/0ab829fc) §Compare/§Verdict).

## Observations

**The gate, recomputed by the reviewer** from `origin/dan_mcts-a:artifacts/mcts/eval/s{0,1,2}_r8__C__{value,sample,prior}.json`
against `preregistration/mcts-a.md` §Gate. The rule: PASS iff Δ_s ≥ 3 on ≥ 2 of 3 seeds, and Δ is above the re-draw
spread. Model: `trajectory`'s best-cap12 `la_T1_best12_s{0,1,2}_r8.pt` (ALiBiGPT 9.56 M params, `lean_staten`, from
scratch on K12, then T1 EI to r8; *[reviewed: wording 1]*). GPU: A40. Group C: `data/mcts/groupC_s{s}.jsonl`. Sampling
used k 256, T 0.8, seed 2, batch 2,048. The search budget is the sampling job's `wall_s`. Lean alone judges.

| seed | n (group C) | PUCT + value | k 256 sampling | Δ_s | PUCT prior only | budget (s) |
|---|---|---|---|---|---|---|
| s0 | 36 | 2 | 4 | −2 | 2 | 131.4 |
| s1 | 35 | 2 | 2 | 0 | 1 | 160.4 |
| s2 | 28 | 3 | 4 | −1 | 3 | 75.3 |

Δ_s ≥ 3 holds on 0 of 3 seeds, so the gate **FAILS**. Δ ≤ 0 on every seed, so mcts-a's falsifier also fires. "Matched
GPU-seconds" in `run_mcts_b.md` means equal wall clock on the same GPU class. mcts-a's reviewer accepted that.

**What mcts-a suggests instead**, from `run_mcts_b.md` with the reviewer's rewordings:
1. Score the r8 value heads on group-C states directly. An AUC of ≈ 0.5 would mean the value cannot see the rare steps;
   a clearly higher AUC would point to the search budget instead.
2. *[reviewed: wording 2]* PUCT search helped at the end of pretraining (`trajectory`'s
   `stage1_best12_s{0,1,2}_b1200.pt`, "pend") on long proofs: rr600 L13–16 / rrQ100, +14.3 for value-guided search over
   sampling (+19 / +10 / +14). Per mcts-a's reviewer, the value contributes only +5.3 [+0.7, +10.0] of that; the rest
   comes from search with the prior alone. *Interpretation, not measured directly:* EI closes this gap by r8, where every
   pool is within ±3. A search expert at that stage needs a compute-matched comparison with EI's early rounds.
3. Any re-gate needs several draws per arm, or a hard pool of ≥ 150 theorems.

## Limitations and next step

- Nothing was measured here. Every number is mcts-a's, measured after 2026-09-27 and judged by Lean alone.
- Phase B's question (does search used as the RL expert teach the policy something EI does not?) is still untested. The
  gate shows only that, on these checkpoints, search at r8 does not reach theorems that sampling cannot.
- **Next measurement** (reviewer, as for mcts-a): compute the AUC of the r8 value heads on group-C partial proofs, using
  the states from mcts-a's 10× sampling read. Before any Phase B, also run a compute-matched comparison of search
  against EI at pend and in early rounds. Week-40 digest decision item 36 and open question 43.
