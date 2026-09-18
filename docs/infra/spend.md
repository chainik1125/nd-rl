---
written_on: 2026-09-17
written_by: agent:claude
---

# Pod spend ledger

One row per run, appended by the librarian when a run clears review. Dollars come from
RunPod balance deltas where a reading exists and from pod-hours × list price otherwise.
`~/balance.log` on the VPS did not exist when this ledger was created (2026-09-17 22:30 UTC)
and still did not at 2026-09-18 01:45 UTC, so balance readings are the ones the executor
wrote in `log.md` / `QUESTIONS.md` plus live `rpbalance` calls; `~/pods.log` starts at p3 of
round 2 (2026-09-17 19:46 UTC) and records creations only, so deletion times come from
`log.md`. Rows for runs that were not processed by a librarian carry the executor's own
figures and are marked as such. Pods that this VPS did not create (`r4-*`, `r1-a100`, `la-*`;
see `QUESTIONS.md` 23:25 and `~/QUEUE.md`) draw on the same balance and are not in this ledger.
Policy (`AGENT_POLICY.md`): per-run pod budget $50 unless Dan authorises more; the prepaid
balance is the hard cap.

| run | UTC window | pods and GPU types | pod-hours | dollars | pre-registered estimate | source / notes |
|---|---|---|---|---:|---|---|
| take-home baseline | 2026-09-15 03:16–07:51 | 2 pods (A40 class, 46 GB) | not logged | not logged | – | executor log only; no spend line |
| novelty campaign | 2026-09-15 19:04 – 09-16 00:26 | p1–p3, A40 $0.49/h | ≈ 16.5 | ≈ $8 | – | executor's figure (`log.md` 00:50), reviewed run (`review_campaign.md`, PR #2), not librarian-processed |
| follow-up | 2026-09-16 01:29–05:46 | p1–p3, A40 $0.49/h | ≈ 12.9 | ≈ $6.3 | – | executor's figure (`log.md` 05:46), reviewed (`review_followup.md`) |
| ignition study (hand-off task 2) | 2026-09-16 18:28–22:19 | p1 RTX 3090 $0.50/h; p2, p3 A100 80 GB $1.59/h | 9.35 | ≈ $11.5 | ≈ $9, ceiling $25 | executor's figure (`log.md` 21:58: 3.3 h + 3.0 h + 3.05 h + orphan ≈ 2 min); reviewed (`review_ignition.md`) |
| **round2-run5** (required-pattern pools) | 2026-09-17 19:12–22:05 | p1, p2 RTX 3090 $0.50/h (created 19:12 / 19:20; shared with run 2 from 21:34; not deleted at RUN5 DONE, handed to run 2) | 5.6 (2.9 + 2.75) | **≈ $2.8** by list price; balance $121.40 (18:57) → $117.50 (22:04) = $3.90, which also covers run 2's p3 (3090 from 19:46, ≈ $1.15) | 2 × 3090 ≈ 4 h → $4–13, ceiling $50 (`log.md` 19:05, `d0d8f16`) | reviewed (`review_round2-run5.md`); the reviewer did not re-derive spend; executor's own estimate ≈ $3 |
| **round2-run3** (minimum outside data for ignition) | 2026-09-17 22:05 – 09-18 01:06 | p4 RTX 3090 $0.50/h (created 22:05, deleted 01:14), p5 RTX 3090 $0.50/h (created 22:17, handed to run 4); both shared with run 4 from ≈ 23:15 | ≈ 5.0 of wall time with run-3 jobs on the card (p4 22:05–01:01 ≈ 2.9, p5 22:17–00:20 ≈ 2.05), ≈ 2.1 before run 4 shared the cards | **≈ $1.6** (executor's share estimate, `numbers.md` "2 × 1.6 h"); bounds $1.1 (exclusive hours) – $2.5 (all wall hours) at list price. No balance delta: $117.50 (22:04) → ≈ $113 (23:24) covers five own pods plus seven foreign pods, and the account was topped up before 01:44 | p4 ≈ 4 h + p5 ≈ 1.5 h on 3090s ≈ $3, ceiling $50 (`log.md` 22:14, `734d51f`) | reviewed (`review_round2-run3.md`); the reviewer did not re-derive spend; the two OOM re-runs (one round, one arm) are inside these hours |
| **round2-run4** (GRPO vs expert iteration at f = 0) | 2026-09-17 23:09 – 09-18 01:36 | p4 RTX 3090 $0.50/h (created 22:05 for run 3; run-4 jobs from ≈ 23:09; deleted 01:14), p5 RTX 3090 $0.50/h (created 22:17 for run 3; run-4 jobs from ≈ 23:15; deleted ≈ 01:36); both shared with run 3 (p4 until 01:01, p5 until ≈ 00:20) | ≈ 4.5 of wall time with run-4 jobs on the card (p4 ≈ 2.1, p5 ≈ 2.4); the two cards' whole life 22:05 / 22:17 → 01:14 / 01:36 is 6.4 pod-hours = $3.2 for runs 3 and 4 together | **≈ $1.6–2.4**: $1.6 is the executor's split of the two cards with run 3 (`STATUS.md`), $2.4 is all run-4 wall hours at list price (`numbers.md`); the reviewer flagged the $0.8 gap and did not re-derive it. No balance delta: the account was topped up between 23:24 and 01:44 (below) | p4 / p5 after run 3, ≈ 45 min per arm, 2 concurrent, ≈ $5, ceiling $50 (`log.md` 23:07, `cb3c07e`) | reviewed (`review_round2-run4.md`); seven arms died at start with CUDA OOM and were re-queued, one arm re-run from scratch after p4 was deleted before its pull, one arm reported at round-equivalent 7 (round-8 files lost with p5) — all inside these hours |
| **round2-run2** (six new patterns, classes pre-registered) | 2026-09-17 19:46 – 09-18 01:14 (run-2 jobs on p1 / p2 from 21:34) | p1, p2 RTX 3090 $0.50/h (created 19:12 / 19:20 for run 5; run-2 Stage-1 from 21:34; run 5 finished 22:05), p3 RTX 3090 $0.50/h (created 19:46 per `~/pods.log`, run 2 only; `numbers.md` dates it 20:05); all three deleted 01:14 (`log.md`) | ≈ 17.4 of wall time for the three cards (19:12 / 19:20 / 19:46 → 01:14), of which ≈ 5.6 is run 5's share (its row above); ≈ 11.8 exclusive to run 2 | **≈ $6–6.5** (executor's split: ≈ $6.5 in `STATUS.md`, ≈ $6 in `numbers.md`); bounds $5.9 (exclusive hours) – $8.7 (all wall hours) at list price. No usable balance delta: $117.50 (22:04) → ≈ $113 (23:24) covers five own pods plus seven foreign pods, and the account was topped up before 01:44 | three RTX 3090 ≈ 5 h each ≈ $8, ceiling $50 (`log.md` 19:45, `5b3b257`) | reviewed (`review_round2-run2.md`); the reviewer did not re-derive spend; two coverage runs were cut short (impe_chain4 s0 at 96 theorems, nested_ore s1 re-queued at k = 500 × 100 on the contended p3) and one frozen control (`frozen_depth4_c8f0_s0`) has no files — all inside these hours |
| **round2-run1** (ND → Lean, in-context surface forms, novelty by scale) | 2026-09-17 23:22 – 09-18 01:30 (step 1 ran on the VPS from 22:26, no pod) | p6 A100 80 GB PCIe $1.59/h, 250 GB container disk (created 23:22:43 per `~/pods.log`, deleted 01:30 per `log.md`); run 1 only | ≈ 2.1 (A100) | **≈ $3.4** at list price (2.12 h × $1.59 = $3.37), the executor's own figure "≈ $3.4 incl. model downloads" (57 GB Qwen3-Coder-30B-A3B + ≈ 90 GB Qwen3 0.6B–32B downloaded inside those hours). No balance delta: the account was topped up between 23:24 and 01:44 | none written: `log.md` has no run-1 budget line (the other four runs have one); policy ceiling $50 | reviewed (`review_round2-run1.md`, `e1d6a3b` 02:54 UTC; the reviewer flagged `LIMIT_HIT` at 02:17:44, `ff44b50`, and went on to commit both phases at 02:51–02:54); the reviewer did not re-derive spend; no training in this run; Lean 4.34.0 (core) on the VPS for step 1 and all scoring |

Balance readings: $121.40 at 2026-09-17 18:57 UTC (`log.md`); $117.50 at 22:04 (`log.md`);
$116.39 at 22:30 (`rpbalance`); ≈ $113 at 23:24 (`QUESTIONS.md`); $293.59 at 2026-09-18 01:44
UTC (`rpbalance`) — a top-up of ≥ $180 happened between 23:24 and 01:44; $294.12 at 01:47
(executor's close-out, `STATUS.md`); $290.51 at 02:14 UTC (`rpbalance`), a burn of ≈ $3.6
in 27 min with none of this VPS's pods running, consistent with the foreign pods alone;
$289.37 at 02:21 UTC (`rpbalance`), with the same twelve foreign pods running (≈ $7.1/h,
`r4-1`…`r4-6` and `r1-a100` up ≈ 3.1–3.4 h, `la-1`…`la-3` ≈ 1 h, `la-4` / `la-5` ≈ 0.1 h);
**$285.02 at 03:02 UTC (`rpbalance`)**, a burn of ≈ $4.35 in 41 min, with eight foreign pods
running (`la-1`…`la-5`, `r4-3`, `r4-4` on 3090s at $0.50/h and `r1-a100` at $1.59/h, ≈ $5.1/h;
`r4-1`, `r4-2`, `r4-5`, `r4-6` are gone since 02:21) and none of this VPS's. Round 2 by list
price: ≈ $15.4 for this VPS's six pods (p1–p5 3090 ≈ 24 pod-hours, 19:12 → 01:36; p6 A100
2.1 h), all deleted by 01:36 UTC (executor's own total ≈ $15: run 5 ≈ $3, run 2 ≈ $6.5, run 3
≈ $1.6, run 4 ≈ $1.6, run 1 ≈ $3.4); all five round-2 rows are now librarian-processed. Foreign pods at 02:14 UTC: twelve, `r4-1`…`r4-6` (3090,
running ≈ 3 h), `r1-a100` (A100, ≈ 3.2 h), `la-1`…`la-5` (ladder-A, second host; `la-4` and
`la-5` created 02:12–02:13), ≈ $7.1/h together, none costed here. Week 38 (2026-09-14 →
09-20) to date by these rows: ≈ $41 (campaign ≈ $8, follow-up ≈ $6.3, ignition ≈ $11.5,
round 2 ≈ $15.4), plus the unlogged take-home baseline and the foreign pods. No `~/ALERTS.md`
line (balance never below $50); `~/balance.log` and `~/ALERTS.md` still do not exist at 03:02.
Kill switch: cron 2026-09-19 00:00 UTC deletes every pod on the account (`~/bin/killswitch`),
including the foreign ones.
