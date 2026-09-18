---
written_on: 2026-09-17
written_by: agent:claude
---

# Pod spend ledger

One row per run, appended by the librarian when a run clears review. Dollars come from
RunPod balance deltas where a reading exists and from pod-hours × list price otherwise.
`~/balance.log` on the VPS did not exist when this ledger was created (2026-09-17 22:30 UTC),
so balance readings are the ones the executor wrote in `log.md` plus a live `rpbalance` call;
`~/pods.log` starts at p3 of round 2 (2026-09-17 19:46 UTC). Rows for runs that were not
processed by a librarian carry the executor's own figures and are marked as such.
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
| round 2, runs 2 and 3 | 2026-09-17 19:46 – (open) | p1–p3 (run 2), p4, p5 (run 3), all RTX 3090 $0.50/h; p4 22:05, p5 22:17 | open | open | run 2 ≈ $8 (3 × 5 h), run 3 ≈ $3; ceiling $50 each | in progress, not reviewed |

Balance readings: $121.40 at 2026-09-17 18:57 UTC (`log.md`); $117.50 at 22:04 (`log.md`);
**$116.39 at 22:30 (`rpbalance`)**. Round 2 to date: $5.01. Week 38 (2026-09-14 → 09-20)
to date by these rows: ≈ $31, plus the unlogged take-home baseline. No `~/ALERTS.md` line
(balance above $50).
| **round2-run3** (minimum outside data for ignition) | 2026-09-17 22:05 – 09-18 01:06 | p4 RTX 3090 $0.50/h (created 22:05, deleted 01:14), p5 RTX 3090 $0.50/h (created 22:17, handed to run 4); both shared with run 4 from ≈ 23:15 | ≈ 5.0 of wall time with run-3 jobs on the card (p4 22:05–01:01 ≈ 2.9, p5 22:17–00:20 ≈ 2.05), ≈ 2.1 before run 4 shared the cards | **≈ $1.6** (executor's share estimate, `numbers.md` "2 × 1.6 h"); bounds $1.1 (exclusive hours) – $2.5 (all wall hours) at list price. No balance delta: $117.50 (22:04) → ≈ $113 (23:24) covers five own pods plus seven foreign pods, and the account was topped up before 01:44 | p4 ≈ 4 h + p5 ≈ 1.5 h on 3090s ≈ $3, ceiling $50 (`log.md` 22:14, `734d51f`) | reviewed (`review_round2-run3.md`); the reviewer did not re-derive spend; the two OOM re-runs (one round, one arm) are inside these hours |
| **round2-run4** (GRPO vs expert iteration at f = 0) | 2026-09-17 23:09 – 09-18 01:36 | p4 RTX 3090 $0.50/h (created 22:05 for run 3; run-4 jobs from ≈ 23:09; deleted 01:14), p5 RTX 3090 $0.50/h (created 22:17 for run 3; run-4 jobs from ≈ 23:15; deleted ≈ 01:36); both shared with run 3 (p4 until 01:01, p5 until ≈ 00:20) | ≈ 4.5 of wall time with run-4 jobs on the card (p4 ≈ 2.1, p5 ≈ 2.4); the two cards' whole life 22:05 / 22:17 → 01:14 / 01:36 is 6.4 pod-hours = $3.2 for runs 3 and 4 together | **≈ $1.6–2.4**: $1.6 is the executor's split of the two cards with run 3 (`STATUS.md`), $2.4 is all run-4 wall hours at list price (`numbers.md`); the reviewer flagged the $0.8 gap and did not re-derive it. No balance delta: the account was topped up between 23:24 and 01:44 (below) | p4 / p5 after run 3, ≈ 45 min per arm, 2 concurrent, ≈ $5, ceiling $50 (`log.md` 23:07, `cb3c07e`) | reviewed (`review_round2-run4.md`); seven arms died at start with CUDA OOM and were re-queued, one arm re-run from scratch after p4 was deleted before its pull, one arm reported at round-equivalent 7 (round-8 files lost with p5) — all inside these hours |
| round 2, runs 2 and 1 | 2026-09-17 19:12 – 09-18 01:30 | run 2: p1–p3 3090 (19:12 / 19:20 / 19:46 → 01:14); run 1: p6 A100 80 GB $1.59/h (23:22 → 01:30) | run 2 ≈ 17.4 (3090); run 1 ≈ 2.1 (A100) | run 2 ≈ $8.7 by list price (shared with run 5 until 22:05, see above; executor's own split ≈ $6.5); run 1 ≈ $3.4 incl. downloads (executor) | run 2 ≈ $8 (3 × 5 h); run 1 not read by this librarian | runs 2 and 1 DONE 01:30 / 01:32, reviewed 01:40–02:06 (`review_round2-run2.md`, `review_round2-run1.md`), not yet librarian-processed |

Balance readings: $121.40 at 2026-09-17 18:57 UTC (`log.md`); $117.50 at 22:04 (`log.md`);
$116.39 at 22:30 (`rpbalance`); ≈ $113 at 23:24 (`QUESTIONS.md`); $293.59 at 2026-09-18 01:44
UTC (`rpbalance`) — a top-up of ≥ $180 happened between 23:24 and 01:44; $294.12 at 01:47
(executor's close-out, `STATUS.md`); **$290.51 at 02:14 UTC (`rpbalance`)**, a burn of ≈ $3.6
in 27 min with none of this VPS's pods running, consistent with the foreign pods alone. Round 2 by list
price: ≈ $15.4 for this VPS's six pods (p1–p5 3090 ≈ 24 pod-hours, 19:12 → 01:36; p6 A100
2.1 h), all deleted by 01:36 UTC (executor's own total ≈ $15: run 5 ≈ $3, run 2 ≈ $6.5, run 3
≈ $1.6, run 4 ≈ $1.6, run 1 ≈ $3.4). Foreign pods at 02:14 UTC: twelve, `r4-1`…`r4-6` (3090,
running ≈ 3 h), `r1-a100` (A100, ≈ 3.2 h), `la-1`…`la-5` (ladder-A, second host; `la-4` and
`la-5` created 02:12–02:13), ≈ $7.1/h together, none costed here. Week 38 (2026-09-14 →
09-20) to date by these rows: ≈ $41 (campaign ≈ $8, follow-up ≈ $6.3, ignition ≈ $11.5,
round 2 ≈ $15.4), plus the unlogged take-home baseline and the foreign pods. No `~/ALERTS.md`
line (balance never below $50); `~/balance.log` and `~/ALERTS.md` still do not exist at 02:14.
Kill switch: cron 2026-09-19 00:00 UTC deletes every pod on the account (`~/bin/killswitch`),
including the foreign ones.
