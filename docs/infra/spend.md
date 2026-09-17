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
