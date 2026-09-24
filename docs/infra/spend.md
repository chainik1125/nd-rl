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
| **round2-run2** (six new patterns, classes pre-registered) | 2026-09-17 19:46 – 09-18 01:14 (run-2 jobs on p1 / p2 from 21:34) | p1, p2 RTX 3090 $0.50/h (created 19:12 / 19:20 for run 5; run-2 Stage-1 from 21:34; run 5 finished 22:05), p3 RTX 3090 $0.50/h (created 19:46 per `~/pods.log`, run 2 only; `numbers.md` dates it 20:05); all three deleted 01:14 (`log.md`) | ≈ 17.4 of wall time for the three cards (19:12 / 19:20 / 19:46 → 01:14), of which ≈ 5.6 is run 5's share (its row above); ≈ 11.8 exclusive to run 2 | **≈ $6–6.5** (executor's split: ≈ $6.5 in `STATUS.md`, ≈ $6 in `numbers.md`); bounds $5.9 (exclusive hours) – $8.7 (all wall hours) at list price. No usable balance delta: $117.50 (22:04) → ≈ $113 (23:24) covers five own pods plus seven foreign pods, and the account was topped up before 01:44 | three RTX 3090 ≈ 5 h each ≈ $8, ceiling $50 (`log.md` 19:45, `5b3b257`) | reviewed (`review_round2-run2.md`); the reviewer did not re-derive spend; two coverage runs were cut short (impe_chain4 s0 at 96 theorems, nested_ore s1 re-queued at k = 500 × 100 on the contended p3) and one frozen control (`frozen_depth4_c8f0_s0`) has no files — all inside these hours |
| **round2-run1** (ND → Lean, in-context surface forms, novelty by scale) | 2026-09-17 23:22 – 09-18 01:30 (step 1 ran on the VPS from 22:26, no pod) | p6 A100 80 GB PCIe $1.59/h, 250 GB container disk (created 23:22:43 per `~/pods.log`, deleted 01:30 per `log.md`); run 1 only | ≈ 2.1 (A100) | **≈ $3.4** at list price (2.12 h × $1.59 = $3.37), the executor's own figure "≈ $3.4 incl. model downloads" (57 GB Qwen3-Coder-30B-A3B + ≈ 90 GB Qwen3 0.6B–32B downloaded inside those hours). No balance delta: the account was topped up between 23:24 and 01:44 | none written: `log.md` has no run-1 budget line (the other four runs have one); policy ceiling $50 | reviewed (`review_round2-run1.md`, `e1d6a3b` 02:54 UTC; the reviewer flagged `LIMIT_HIT` at 02:17:44, `ff44b50`, and went on to commit both phases at 02:51–02:54); the reviewer did not re-derive spend; no training in this run; Lean 4.34.0 (core) on the VPS for step 1 and all scoring |
| **round3-run2** (8-line reductio stratum; derived-ORE base-rate dependence) | 2026-09-18 05:15–08:36 | r32a, r32b, r32c RTX 3090 24 GB $0.50/h (created 05:15:15 / 05:15:54 / 05:16:20 per `~/pods.log`; r32c deleted 07:56, r32a and r32b 08:36 per `log.md`); run 2 only | ≈ 9.35 (3.35 + 3.35 + 2.65) | **≈ $4.7** at list price, equal to the executor's figure ("≈ 9.4 pod-hours ≈ $4.7"). Balance $274 (05:07) → $239.7 (08:36) = $34.3, which also covers this VPS's run-1 pods r31a–d (≈ 12 pod-hours to 08:36 ≈ $6) and run-3 pods r33a–d (≈ 12 ≈ $6) and the eight foreign pods (≈ $5.1/h ≈ $18): ≈ $34.4 by list price, consistent | three RTX 3090, 14–16 pod-hours ≈ $8, ceiling $50, stop at $30 (`preregistration/round3-run2.md`, `6ffade0`) | reviewed (`review_round3-run2.md`, `3932dbd`); the reviewer found the r32a / r32b queue logs (05:17–08:33) consistent with 3.35 h each, r32c's 2.67 h rests on `log.md` (queue logs not pulled), balance attribution not checkable; no job failures; five extra cap-8 Stage-1 draws (≈ 10–15 min each) and eight 10⁴-sample coverage runs are inside these hours |
| **round3-run1** (pool composition or pattern class) | 2026-09-18 05:16–08:54 | r31a, r31b, r31c, r31d A40 48 GB $0.49/h (created 05:16:09 / 05:28:44 / 05:46:24 / 05:46:15 per `~/pods.log`; all four deleted 08:54 per `log.md`; RTX 3090 stock was Low, A40 Medium at the same price); run 1 only | ≈ 13.3 (3.63 + 3.42 + 3.13 + 3.13) | **≈ $6.5** at list price (13.3 h × $0.49 = $6.52), equal to the executor's figure ("13.3 pod-hours ≈ $6.5"). No separable balance delta: $274 (05:07) → $239.7 (08:36) also covers run 2's and run 3's pods and the eight foreign pods (see the run-2 row: ≈ $34.4 by list price against $34.3 observed, with run 1 at ≈ 12 pod-hours to 08:36) | three RTX 3090 ≈ 26 pod-hours ≈ $13, ceiling $50, stop at $40 (`preregistration/round3-run1.md`, `8c318e7`) | reviewed (`review_round3-run1.md`, `e5bfcc7`); the reviewer found the four creations in `~/pods.log` and no `r31*` pod listed at 09:20, and did not meter hours × price independently ("consistent; cost not derivable beyond that"); pre-RL samples were co-scheduled with arms on the 48 GB cards (disclosed; no OOM, no job failures); one `podnew` call returned nothing and was retried (no pod, no cost); 16 Stage-1 draws, 48 arms, 16 × 600k pre-RL and 60 × 300k drift samples are inside these hours |
| **round3-run3** (base generalisation by pattern class, 72 Stage-1 draws, no RL) | 2026-09-18 05:23–12:48 | r33a, r33b, r33c, r33d RTX 3090 24 GB $0.50/h (created 05:23:43 / 05:24:00 / 05:24:12 / 06:21:26 per `~/pods.log`; deleted 12:45 / 12:25 / 12:38 / 12:48 per `log.md`, each after a remote-vs-local listing diff with 0 missing files; `log.md` dates r33b and r33c 05:26, two minutes after `~/pods.log`); run 3 only | ≈ 28.0 (7.35 + 7.02 + 7.23 + 6.44) | **≈ $14.0** at list price (28.0 h × $0.50), equal to the executor's figure ("≈ 28 pod-hours ≈ $14.0": 3.7 + 3.5 + 3.6 + 3.2). Balance $232.44 (09:30:39) → $205.51 (13:30:38) = $26.93, which covers run 3's last ≈ 12.6 pod-hours (≈ $6.3) and the eight foreign pods (≈ $5.1/h × 4.0 h ≈ $20.4): ≈ $26.7 by list price, consistent. Over the whole round, $274 (05:07) → $205.51 = $68.5 against ≈ $67.9 by list price (this VPS's eleven pods ≈ $25.2 + foreign ≈ $42.7) | three RTX 3090, 36–42 pod-hours ≈ $20, plus ≈ 10 pod-hours ≈ $5 for the deep-pass addendum; stop at $35, ceiling $50 (`preregistration/round3-run3.md`, `16378b0`, addendum `0c62ffb`) | reviewed (`review_round3-run3.md`, `2106258`); the reviewer marks spend "not derivable" in phase 1 and "confirmed" in phase 2 on the pod-log start times and the absence of any `r33*` pod (its "13:50 UTC" postdates its own 13:23 commit; the librarian's `runpodctl pod list` at 13:30 shows none); a fourth pod (r33d, 06:21) was added to halve the depth-3 coverage wall clock; one self-inflicted CUDA OOM at 07:07 (two coverage jobs on one card after a `pkill` self-match; the job was resumed) and one 4-minute mis-launch at 09:47 are inside these hours; 72 Stage-1 draws, 96 main-pass coverage files (24 × 2.0M + 48 × 600k samples) and 48 deep-pass files (24 × 0.9M + 24 × 1.04M) are inside these hours |
| **ladder-A** (RL technique ladder, Phase A, T1–T6) | 2026-09-18 01:15–17:20, resume 17:53–18:22 | la-1…la-5 RTX 3090 $0.50/h (created 01:15:04 / 01:24:27 / 01:26:40 / 02:13:17 / 02:14:00 per the second host's `~/pods.log`; last arm finished 04:42; **idle 04:42–17:20** after the executor session died on an OAuth expiry, deleted by Dan at 17:20 per the resume note); la-6 RTX 3090 $0.50/h (created 17:53:21, deleted 18:22 per `STATUS.md`) for the T3 s0 resume; ladder-A only | ≈ 78.6 (16.08 + 15.93 + 15.89 + 15.11 + 15.10 + 0.48), of which ≈ 15 productive and **≈ 63.2 idle** | **≈ $39.3** at list price (≈ $7.5 pools and arms, ≈ $31.6 idle, ≈ $0.24 resume), equal to the executor's "≈ $39". No separable balance delta: the la-* pods are the "foreign pods" of the rows above and were never isolated in a reading; $205.51 (13:30:38) → $176.94 (19:21:44) = $28.57 against ≈ $28.3 by list price (eight pods at ≈ $5.1/h to 17:20 ≈ $19.5 **if** `r4-3`, `r4-4` and `r1-a100` stopped then too — not verified; la-6 $0.24; `r34b-1`…`-6` and `r4a-1`…`-3` from 17:56 ≈ $8.6) | one RTX 3090 for pools, then up to three for arms, ≈ $12–20, cap $50, hard stop 36 h (`preregistration/ladder-A.md`, `b3845eb`) | reviewed (`review_ladder-A.md`, `ab629c7`, row 19: "reproduces as an estimate; the deletion time comes from the resume note, not a file" — 78.1 pod-h = $39.1 + $0.25); five pods were used, not three (disclosed in `log.md`); a killed round 1 on the v1 target pool (OOM at batch 1,024), one frozen-s0 rerun after an OOM, and the T3 s0 crash in round 6 are inside these hours; `pod_watchdog` did not delete the idle pods because other runs' sessions were live on the host (executor's reading, `QUESTIONS.md` 18:35, not verified) |
| **round3-run4a** (reductio vs model size: 3.2M / 25M / 85M, 19 Stage-1 draws, EI 8 × 32) | 2026-09-18 17:56 – 09-19 00:53 | r4a-1…r4a-6 A100-SXM4-80GB $1.59/h (created 17:56:44 / 18:07:12 / 19:11:31 / 20:09:05 / 20:56:28 / 20:57:04 per `~/pods.log`; deleted 21:51 / 22:31 / 00:18 / 22:30 / 22:31 / 00:53 per `log.md`); run 4a only. Pre-registered: two pods, a third if needed; six used (co-tenancy OOMs, the rule-mandated config-B cells, exploratory extras), logged with reasons | ≈ 21.3 (3.90 + 4.40 + 5.12 + 2.35 + 1.58 + 3.93) | **≈ $33.8** at list price (21.28 h × $1.59 = $33.84, the reviewer's sum, equal to the executor's "21.3 pod-hours ≈ $33.8"; executor's split: ≈ $27 pre-registered plan incl. B cells, ≈ $6.7 exploratory extras). No separable balance delta: $185 (17:52, pre-registration) → $176.94 (19:21:44) → $127.78 (01:12 reviewer, 01:20 `rpbalance`); the $49.16 drop since 19:21 is run 4a's ≈ $29.3 after that time plus ≈ $19.8 for run 4b's nine `r34b-*` pods (≈ 40 pod-hours at $0.49–0.53/h would account for it; run 4b is in review, not verified) | two A100 80 GB, ≈ 18 A100-hours ≈ $29, ceiling $50, stop at $45 (`preregistration/round3-run4a.md`, `da5bdd8`) | reviewed (`review_round3-run4a.md`, `f9e1ce2`, row 22: creation times match `~/pods.log`, deletion times are log-only so the spend is not derivable from repository files; `podls` empty at 01:10Z and at 01:20Z); within the ceiling and the stop rule |
| **round3-run4b** (depth-3 vs model size: 25M / 85M, 12 Stage-1 draws on two schedules, `req` and `mix` EI 8 × 32, pass@10⁴ for three igniters) | 2026-09-18 17:56 – 09-19 00:48 | r34b-1…r34b-4 A40 48 GB $0.49/h (created 17:56:28 / 17:59:59 / 18:01:08 / 18:02:26 per `~/pods.log`), r34b-5, r34b-6 RTX A6000 48 GB $0.53/h (18:04:58 / 18:05:53; A40 stock ran out, the pre-registered fallback card), r34b-7, r34b-8, r34b-9 A40 $0.49/h (20:44:03 / 20:45:00 / 22:12:34; added after out-of-memory failures on shared 85M cards); deleted 00:05 / ≈ 23:47 / ≈ 23:47 / 00:44 / 00:48 / 00:42 / 00:20 / 00:14 / 00:13 per `log.md`; run 4b only. Pre-registered: one A40 per draw, up to six; nine used, logged with reasons | ≈ 46.8 (A40 6.15 + 5.78 + 5.75 + 6.70 + 3.60 + 3.50 + 2.00 = 33.48; A6000 6.72 + 6.60 = 13.32) | **≈ $23.5** at list price (33.48 h × $0.49 + 13.32 h × $0.53 = $23.47, the reviewer's sum, equal to the executor's "46.8 pod-hours ≈ $23.5"). Balance, jointly with run 4a: $185.50 (17:52, pre-registration) → $127.78 (01:20 and 01:33 `rpbalance`) = $57.72 against $57.55 by list price (run 4a $33.84 + run 4b $23.47 + ladder-A's la-6 $0.24), consistent to $0.17; the two runs are not separable in the readings | six A40 draws × ≈ 7 h × $0.49 ≈ $21 plus ≈ $5 for `mix` arms ≈ $26, ceiling $50, stop at $45 (`preregistration/round3-run4b.md`, `c1ad029`; deviates from the brief's two A100s, decided before any pod) | reviewed (`review_round3-run4b.md`, `450c0b2`, row 19: creation times match `~/pods.log`, pod-hours recomputed from `~/pods.log` + the log's deletion times, "reproduces", under the ceiling); the executor's repeated CUDA out-of-memory failures from doubling jobs on one card (18:56, 20:22, 20:46, 21:20, 22:14, 23:06 in `log.md`), the restarts and resumes they caused, a self-matching launcher guard that idled two samplers ≈ 15 min, and two duplicate optional-pool samples killed at 23:42 are inside these hours |
| **lean-format** (proposal 8: Lean as the training format; 7 Stage-1 models, 16 EI / frozen arms on the depth-3 dial and ladder T1, solo timing rounds) | 2026-09-21 03:04–08:03 | lf-1…lf-4 RTX 3090 24 GB $0.50/h (created 03:04:23 / 03:04:49 / 03:05:38 / 03:06:11 per `~/pods.log`; deleted ≈ 04:32 (lf-1, lf-3), 07:37 (lf-2), 08:03 (lf-4) per `log.md`). Pre-registered: two pods, a third only if wall-clock required it; four used, one per scheme × experiment, logged as a deviation | ≈ 12.4 (1.46 + 4.54 + 1.44 + 4.95) | **≈ $6.2**: balance $127.78 (03:03 `rpbalance`, unchanged since 2026-09-19 01:33) → $121.56 (08:04) = $6.22, against $6.19 by list price; no other pod on the account in the window, so the delta is this run's alone | $30 ceiling, planned ≈ 14 pod-hours ≈ $7–12 (`preregistration/lean-format.md`, `c2dcd0a`) | reviewed (`review_lean-format.md`, `0976494`, row 17: "reproduces"; `podls` empty and balance $121.52 at 08:35). The only run launched during the 2026-09-18 pause (authorised exception) |
| **lean-seed2** (second `lean_seq` Stage-1 seed on the ladder rung; BOTE fix; 1 Stage-1 model, ladder T1 + frozen × 2 EI seeds, pass@16) | 2026-09-22 05:46–08:03 | ls2-1, ls2-2 RTX 3090 24 GB $0.50/h (created 05:46:06 / 05:47:40 per `~/pods.log`; job logs 05:48–07:57 / 05:50–07:54; both deleted 08:03 per `log.md`, `podls` showed none of the executor's pods). Pre-registered: two pods; two used | ≈ 4.5 (2.28 + 2.26) | **≈ $2.3** at list price (4.54 h × $0.50 = $2.27), equal to the executor's figure. No usable balance delta: $121.52 → $108.00 over the run (`numbers.md` E9) with twelve other agents' pods (`lo-*`, `dsc-*`, `dsg-*`) on the account | two RTX 3090, ≈ 6 pod-hours ≈ $3; budget $10 (`preregistration/lean-seed2.md`, `123df56`) | reviewed (`review_lean-seed2.md`, `51604b3`); the reviewer checked the pod windows against the job logs and calls the ≈ $2–3 consistent; `~/balance.log` still does not exist |
| **efficiency** (sampler and Lean-gate throughput on a fixed 51,200-sample workload; no training) | 2026-09-23 15:31:50–17:03 | `ef-1` RTX 3090 24 GB $0.50/h (created 15:31:50 per `~/pods.log`; deleted 17:03 per `log.md`; `podls` empty at 17:10 reviewer and at 17:45 librarian). Pre-registered: one pod; one used | **1.5314** (`podbudget efficiency`) | **$0.77** by list price and by `podbudget efficiency`; the reviewer re-ran `podbudget` and got the same 1.53 h / $0.77. No separable balance delta at this size: $221 at 15:25 (executor `log.md`) → $220.36 at ≈ 17:30 (`rpbalance`, reviewer) → $220.36 at 17:45 (librarian) = $0.64 observed against $0.77 list price, with no other pod on the account | 10 pod-hour ceiling, budget $15 (`podbudget efficiency`, set with the brief); the pre-registration states the ceiling and a stop at 8 pod-hours (`preregistration/efficiency.md`, `57bcb1c`) | reviewed (`review_efficiency.md`, `45c7a6e`, §0: "`podbudget efficiency`: 1.53 h, $0.77 of a 10 h / $15 ceiling; `podls` empty (pod deleted); RunPod balance $220.36" — pass). The cheapest run in this ledger: nothing was trained, one checkpoint was pulled from the bucket, and 25 sampler configurations × 51,200 samples plus ten Lean-gate settings and the four co-tenancy jobs are all inside these hours |
| **ds-composition** (proposal 10 run 1: the training set's composition; five 155,000-proof `lean_seq` sets × 2 Stage-1 seeds, held-out greedy, depth-3 dial EI + frozen, ladder T1 + frozen, three coverage pools at pass@2,000, checker of record) | **session 2** 2026-09-23 21:12 – 09-24 04:10 (every reported number comes from this session); **session 1** 2026-09-22 06:09 – 09-23 ≈ 05:17 (superseded, nothing of it is reported) | session 2: `dsc-c0`, `dsc-a1`, `dsc-a3` **A40 48 GB $0.49/h** and `dsc-a2`, `dsc-a4` **RTX A6000 48 GB $0.53/h** (created 21:12:35 / 21:14:04 / 21:17:34 / 21:19:52 / 21:21:24 per `~/pods.log`; deleted 01:05 / 01:19 / 02:10 / 02:59 / 04:10 per `log.md`, each after its files were pulled and the checker of record had run). **No RTX 3090 was in stock at any tier**, a logged hardware deviation from the pre-registration's 3090s at the same price. Session 1: the same five names on RTX 3090 $0.50/h, created 06:09:23 / 06:11:29 / 06:15:28 / 06:18:26 / 06:40:33 per `~/pods.log`; **never deleted by this host** | **session 2: 25.70** (5.02 + 5.99 + 6.98 + 3.97 + 3.74, `~/podhours.log`, under the 36 h ceiling). **Session 1: ≈ 114.9** (23.13 + 23.10 + 23.03 + 22.98 + 22.61 if they billed to the 05:17 balance-floor reading), of which **≈ 21.2 attended** and **≈ 93.7 idle** | **session 2 ≈ $12.9 · session 1 ≈ $57 · run total ≈ $70.** Session 2 at the billed list prices is **$12.89** (A40 17.99 h × $0.49 = $8.81 + A6000 7.71 h × $0.53 = $4.08); `podbudget` and `~/podhours.log` print **$12.84** because their fallback rate is a flat $0.50/h. Not separable in the balance: $220.36 at 21:02 (`log.md`) → **$181.53 at 2026-09-24 04:46:46 (`rpbalance`)** = $38.83, which also covers the sibling `ds-generator`'s `dsg-1` (A6000 $0.53/h) and `dsg-2` (RTX 4090 $0.74/h), still running at ≈ 7.6 h each ≈ $9.7; ≈ $16 of that delta is **not explained by any pod this host knows about** and is not attributed here. Session 1 is derived from `~/balance.log` instead, and the arithmetic closes: $121.5196 (09-21 14:00) → **$21.3117 at 09-23 05:17** (the balance-floor alert, `~/notify.log`) = **$100.21**, against **$92.02** for this host's eight pods (the five `dsc-*` plus `ds-generator`'s three `dsg-*`) billing from creation to 05:17 at $0.50/h (184.0 pod-hours), leaving **$8.19** for the *other* host's `ls2-*` (lean-seed2, its own row ≈ $2.3) and `lo-*` (lean-only, unledgered) pods — and $21.3117 (05:17) → $21.1387 (14:00) = **$0.17 in 8.7 h**, i.e. nothing running after 05:17. ds-composition's five-eighths share of the $92.02 is ≈ 114.9 pod-hours ≈ **$57.4**. This is a derivation, not a reading: no per-pod deletion record exists for session 1 | session 1: five RTX 3090, ≈ 35 pod-hours ≈ **$18**, ceiling **$22**, stop rule at $20 spent (`preregistration/ds-composition.md`, `56c42bb`); session 2 (amendment 3, `b836eef`): **$18**, ceiling **36 pod-hours** (`podbudget ds-composition`, reset at the resume) | reviewed (`review_ds-composition.md`, `4ebc003` phase 1 / `6b99c8b` phase 2; its spend row reads `podbudget` 25.70 h / $12.84 and `list-pods` shows no `dsc-*` — the reviewer costed **session 2 only**, as the executor's `numbers.md` D7 does). **Session 1's ≈ $57 is in no write-up.** It happened because the executor was stopped at 2026-09-22 10:32:48 UTC by a **Claude usage-credit exhaustion** (three consecutive 429 "out of usage credits", `~/runs/ds-composition/executor.driver.log` and `~/notify.log`) and its five pods then billed unattended for ≈ 19 h until the hourly `balance_floor` fired at 05:17 on a $21.31 balance; `pod_watchdog` (every 15 min; deletes this host's pods after 30 min with no Claude process) did **not** fire in that window — the same failure mode as ladder-A's ≈ $32 of idle pods. By the time the kill switch ran every delete returned `pod not found to terminate` and every ssh was refused, so the pods were already gone and the balance arithmetic above is what dates their end. A host cleanup then deleted the session's uncommitted `artifacts/` and `ckpts/`, so **every number was re-measured in session 2** and session 1 bought nothing that is reported. Session 2's only two job failures (a target file missing from the sync list, one CUDA OOM) were re-run inside the 25.70 h |
| **ds-rendering** (proposal 10: does the Lean rendering change what a from-scratch model learns? five renderings × 2 seeds through the full plan, plus a Stage-1 seed sweep to n = 6 on four arms; 26 models, 83,513 counted proofs) | 2026-09-23 21:19:20 – 09-24 04:14 | six **NVIDIA A40 48 GB $0.49/h**, one arm each plus the sweep pod: `dsr-c0` 6.2119 h, `dsr-r1` 6.0619, `dsr-r3` 6.2333, `dsr-r2` 6.8819, `dsr-r4` 5.4628, `dsr-s` 1.8033 (created 21:19:20 / 21:20:24 / 21:20:48 / 21:21:17 / 21:45:53 / 21:59:27 per `~/pods.log`; deleted 03:34:39 / 03:24:07 / 03:34:48 / 04:14:12 / 03:13:39 / 23:47:39 per `~/podhours.log`), **plus one aborted RTX 3090 `dsr-c0`** created 21:19:20 and deleted 21:26, 0.0375 h $0.02 — RunPod had no 3090 capacity left once the two sibling dataset-styles runs had taken theirs, so the arm was recreated on an A40 and **all five arms including the control share one GPU class** (DEVIATION from the brief and the pre-registration, logged with its reason at 21:29:42). Pre-registered: four pods, one per arm; **six used** — a fifth arm R4 (addendum 1, `3a6b390`, before its pod) and the seed-sweep pod `dsr-s` (addendum 2, `14327f9`, before its pod), both logged as deviations with reasons and both raised in `QUESTIONS.md` | **32.6926** (32.655 A40 + 0.0375 3090; `podbudget ds-rendering`, `~/podhours.log`) | **$16.01** by list price and by `podbudget ds-rendering` ($15.99 A40 + $0.02 3090), of an **$18** budget and a **36 h** ceiling. No separable balance delta: $220.36 at 2026-09-23 17:45 (librarian) → **$181.42** at 2026-09-24 04:52 (`rpbalance`, librarian) = $38.94 over a window in which the two sibling proposal-10 runs `ds-composition` and `ds-generator` were also on the account (`dsg-1` / `dsg-2` still RUNNING at 04:52), so the $38.94 covers those as well and this run's dollars rest on `podbudget` and list price | **$18 budget, 36 pod-hour ceiling**; planned four pods × ≈ 5.5 h ≈ 22 pod-hours ≈ **$11** at RTX 3090 $0.50/h, drop order *R2's ladder, then R2 entirely, then R3's ladder*, hard stop 2026-09-25 03:01 UTC (`preregistration/ds-rendering.md`, `bcb0d06`); addendum 1 added ≈ 5 pod-hours ≈ $2.50 and addendum 2 ≈ 1.6 h ≈ $0.80, projecting ≈ 25 h / ≈ $12.50 | reviewed (`review_ds-rendering.md`, `f9b22a3`, §0: *32.69 h of a 36 h ceiling, $16.01 of $18; `BUDGET_WARNING` present (fired at 28.84 h); `podls` — every `dsr-*` pod is gone* — pass, and gate 0 passes three times: each pre-registration commit precedes its own pod by 49 s to 3 m 26 s). **Over the pre-registered ≈ $12.50 projection by ≈ $3.50, inside both hard limits**; the overrun is the ladder, whose round time was estimated before it was measured (1,376 → 1,704 → 1,717 s, plateauing ≈ 25 % above the estimate) — the executor cut the **frozen ladder from all five arms** rather than following the pre-registered drop order, logging the deviation with the arithmetic and the rejected alternative before acting, so **E9 is unscored and no ladder claim has an equal-attempts frozen control** (reviewer's Finding R-4). 41 `LIMIT_HIT` pauses between 21:40 and 23:34 cost the run nothing: the pods' `setsid nohup` job sequencers ran across every pause |

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

Round 3 (2026-09-18): balance $274 at 05:07 UTC (`log.md`, before any round-3 pod); $253.6 at
07:10 and $239.7 at 08:36 (`log.md`); **$233.63 at 09:20 UTC (`rpbalance`)** with twelve pods
running: this VPS's `r33a`…`r33d` (run 3, RTX 3090, $0.50/h) and the same eight foreign pods
(`la-1`…`la-5`, `r4-3`, `r4-4`, `r1-a100`, ≈ $5.1/h). This VPS created eleven round-3 pods
(`~/pods.log`: r31a–d, r32a–c, r33a–d, 05:15–06:21 UTC); only run 2's three have a ledger row.
Run 1's executor reports 13.3 pod-hours ≈ $6.5 (pods deleted 08:54; under review, no row yet);
run 3 is still running. Week 38 to date by these rows: ≈ $46 (≈ $41 above + round3-run2 ≈ $4.7).
The kill switch is now scoped to this host's registered pods (`~/.config/nd-rl/pods/*`; `--all`
only from `balance_floor`) and its cron moved to 2026-09-19 18:00 UTC (`crontab -l`, commit
`e9999e3`); hourly `balance_floor` and daily `balance_alert` crons exist. `~/balance.log` and
`~/ALERTS.md` still do not exist at 09:20.

Update 09:30 UTC (round3-run1 processed): **$232.44 at 09:30:39 UTC (`rpbalance`)**, $1.19 below
the 09:20 reading (≈ $6.8/h over ≈ 10.5 min; list price of the twelve running pods ≈ $7.1/h:
`r33a`…`r33d` at $0.50/h and the same eight foreign pods ≈ $5.1/h). Run 1's row is now in the
ledger (≈ $6.5, replacing the "under review" note above); of this VPS's eleven round-3 pods
only run 3's four have no row, and they are still running. Round 3 to date by rows ≈ $11.2
(run 2 ≈ $4.7 + run 1 ≈ $6.5). Week 38 to date by these rows: ≈ $52 (≈ $41 through round 2 +
≈ $11.2). `~/balance.log` and `~/ALERTS.md` still do not exist at 09:30.

Update 13:30 UTC (round3-run3 processed): **$205.51 at 13:30:38 UTC (`rpbalance`)**, $26.93 below
the 09:30 reading (≈ $6.7/h over 4.0 h; by list price ≈ $26.7: run 3's four pods until their
deletion at 12:25–12:48 ≈ $6.3, the eight foreign pods ≈ $20.4). Eight pods are running, none
of this VPS's: `la-1`…`la-5`, `r4-3`, `r4-4` (RTX 3090, $0.50/h) and `r1-a100` ($1.59/h),
≈ $5.1/h ≈ $122/day, which reaches the `balance_floor` default of $25 in ≈ 35 h (around
2026-09-20 01:00 UTC) if nothing changes. All eleven round-3 pods of this VPS now have a row and
are deleted. Round 3 by rows ≈ $25.2 (run 2 ≈ $4.7 + run 1 ≈ $6.5 + run 3 ≈ $14.0), against
≈ $41 pre-registered for the three runs together (≈ $46 with run 3's deep-pass addendum). Week 38 to date by these rows: ≈ $66 (≈ $41
through round 2 + ≈ $25.2). `~/balance.log` and `~/ALERTS.md` still do not exist at 13:30; the
crontab has the host-scoped kill switch at 2026-09-19 18:00 UTC, hourly `balance_floor` and
daily `balance_alert`.

Update 19:20 UTC (ladder-A processed; written on the second host): **$176.94 at 19:21:44 UTC
(`rpbalance`)**. The `la-*`, `r4-*` and `r1-a100` pods that the paragraphs above call "foreign"
were created on the host this update is written on (its `~/pods.log` lists them, 2026-09-17 23:01
→ 09-18 02:14); `la-*` now has the ladder-A row, `r4-*` (run4-grpo) and `r1-a100` (run1-lean)
belong to runs that are not reviewed and have no row. Nine pods are running at 19:21, none of
them ladder-A's: `r34b-1`…`-6` (four at $0.49/h, two at $0.53/h, since 17:56–18:05) and
`r4a-1`…`-3` ($1.59/h each, since 17:56 / 18:06 / 19:11), **≈ $7.79/h ≈ $187/day**; at that
burn the balance crosses the $50 alert line in ≈ 16 h and the `balance_floor` default of $25 in
≈ 19.5 h (around 2026-09-19 15:00 UTC). Week 38 to date by these rows: **≈ $105** (≈ $66
through round 3 + ladder-A ≈ $39.3); not in any row: the take-home baseline, run4-grpo's and
run1-lean's pods (≈ $5.1/h while eight pods ran; their share is whatever of the $274 → $205.51
→ $176.94 deltas the rows do not explain), and the nine pods now running. `~/balance.log` and
`~/ALERTS.md` do not exist on this host either; its crontab has hourly `balance_floor`,
`pod_watchdog` every 15 min, the orchestrator every 10 min, and the kill switch at 2026-09-20
18:00 UTC.

Update 2026-09-19 01:25 UTC (round3-run4a processed; second host): **$127.78 at 01:20 UTC
(`rpbalance`)**, no pod running on the account (`podls` empty; run 4b's `r34b-*` pods are gone
too). Week 38 to date by these rows: **≈ $139** (≈ $105 above + run 4a ≈ $33.8); not in any
row: the take-home baseline, run4-grpo's and run1-lean's pods, and round-3 run 4b's nine
`r34b-*` pods (in review; ≈ $20–23 by the balance deltas above: $185 → $127.78 = $57.2 less run
4a's $33.8). The balance is ≈ $78 above the $50 alert line with nothing burning. `~/balance.log`
and `~/ALERTS.md` still do not exist on this host.

Update 2026-09-19 01:35 UTC (round3-run4b processed; second host): **$127.78 at 01:33 UTC
(`rpbalance`)**, unchanged since 01:20; no pod running on the account (`podls` empty). Run 4b's
row (≈ $23.5) replaces the "in review; ≈ $20–23 by the balance deltas" note above, and the
joint reading closes: $185.50 (17:52) → $127.78 = $57.72 against $57.55 by list price for run
4a, run 4b and la-6. Round-3 runs 4a + 4b together ≈ $57.3 against ≈ $55 pre-registered (≈ $29 +
≈ $26), each under its $50 ceiling. Week 38 to date by these rows: **≈ $163** (≈ $139 above +
run 4b ≈ $23.5); not in any row: the take-home baseline and run4-grpo's and run1-lean's pods
(not reviewed). The balance is ≈ $78 above the $50 alert line with nothing burning.
`~/balance.log` and `~/ALERTS.md` still do not exist on this host.

Update 2026-09-21 08:55 UTC (lean-format processed): **$121.52 at 08:53 UTC (`rpbalance`)**, no pod
running on the account (`podls` empty). Readings: $127.78 (2026-09-19 01:33 and 2026-09-21 03:03,
nothing ran in between) → $121.56 (08:04) → $121.52 (08:35 reviewer, 08:53), a $0.04 drift with no
pod running (not attributed). Week 39 (2026-09-21 → 09-27) to date by these rows: **≈ $6.2**
(lean-format only). Cumulative by rows since 2026-09-15: **≈ $169** (week 38 ≈ $163 + ≈ $6.2); not
in any row: the take-home baseline and run4-grpo's and run1-lean's pods (not reviewed). The
balance is ≈ $71 above the $50 alert line with nothing burning. `~/balance.log` and `~/ALERTS.md`
still do not exist on this host, so the dollars above come from the executor's and reviewer's
`rpbalance` readings and a live call.

Update 2026-09-22 08:35 UTC (lean-seed2 processed): **$105.29 at 08:33 UTC (`rpbalance`)**. Readings: $121.52 (2026-09-21 08:53) → $108.00 (≈ 08:03 today, executor) → $105.29; the $16.2 since yesterday covers lean-seed2's two pods (≈ $2.3) and the other runs launched this morning, which are not in any row yet: `lo-1`…`lo-6` (lean-only, proposal 9; RTX 3090, created 06:05–06:33 per `~/pods.log`; `lo-1` and `lo-3` still running at 08:33) and the dataset-styles pods created on another host (`dsc-a1`…`dsc-a4`, `dsc-c0`, `dsg-1`…`dsg-3`, all RUNNING at 08:33, proposal 10, $60 authorised). Ten pods are running on the account, none of them lean-seed2's. Week 39 (2026-09-21 → 09-27) to date by these rows: **≈ $8.5** (lean-format ≈ $6.2 + lean-seed2 ≈ $2.3). Cumulative by rows since 2026-09-15: **≈ $171**; not in any row: the take-home baseline, run4-grpo's and run1-lean's pods (not reviewed), and the running lean-only and dataset-styles pods. The balance is ≈ $55 above the $50 alert line with ten pods burning (≈ $5/h by list price if all are 3090-class; the alert line is reached in ≈ 11 h at that rate if nothing is deleted). `~/balance.log` and `~/ALERTS.md` still do not exist on this host.

Update 2026-09-23 17:45 UTC (efficiency processed): **$220.36 (`rpbalance`)**, unchanged from the
reviewer's reading ≈ 17:30, and `podls` is empty — no pod is running on the account. Dan topped
the account up since the last update (**$105.29 on 2026-09-22 08:33 → $221 at 2026-09-23 15:25**,
the executor's reading at run start), so the $50 alert line is far off and the between-update
delta is not attributable to any run. Efficiency's own window is the first since 2026-09-21 with
a single pod on the account, but at $0.77 the balance delta ($221 → $220.36 = $0.64) is within
the rounding of the executor's start reading, so the row's dollars are list price confirmed by
`podbudget`. Week 39 (2026-09-21 → 09-27) to date by these rows: **≈ $9.3** (lean-format ≈ $6.2,
lean-seed2 ≈ $2.3, efficiency $0.77). Cumulative by rows since 2026-09-15: **≈ $172**. Still not
in any row: the take-home baseline, run4-grpo's and run1-lean's pods (not reviewed), and the
`lean-only` (proposal 9) and dataset-styles (`ds-composition`, `ds-generator`, `ds-rendering`,
proposal 10) runs — all of whose pods are now gone from the account but whose runs have not
cleared review, so their spend is still unledgered. `~/balance.log` and `~/ALERTS.md` still do not
exist on this host.

Update 2026-09-24 04:47 UTC (ds-composition processed; written on the host that ran it, which
does have `~/balance.log` and `~/ALERTS.md`): **$181.5288 at 04:46:46 UTC (`rpbalance`)**, two pods
running, both the sibling `ds-generator`'s (`dsg-1` RTX A6000 $0.53/h and `dsg-2` RTX 4090 $0.74/h,
up since 2026-09-23 21:07 / 21:08, ≈ $1.27/h together). Readings on this host since the last
update: `~/balance.log` $121.5196 (09-21 14:00) → $82.6105 (09-22 14:00) → **$21.1387 (09-23
14:00)**, with the balance-floor alert at **$21.3117 on 09-23 05:17**; Dan topped up to $220.36 by
09-23 21:02 (executor's reading). **`~/ALERTS.md` carries the 2026-09-23 line "RunPod balance
$21.14 is below $50".** Week 39 (2026-09-21 → 09-27) to date by these rows: **≈ $79** (lean-format
≈ $6.2, lean-seed2 ≈ $2.3, efficiency $0.77, **ds-composition ≈ $70**). Cumulative by rows since
2026-09-15: **≈ $242**. Still not in any row: the take-home baseline, run4-grpo's and run1-lean's
pods, the `lean-only` (proposal 9) `lo-*` pods, and the sibling dataset-styles runs `ds-generator`
(running) and `ds-rendering` — none of which has cleared review.

Two things about this row that are not about ds-composition alone:

1. **ds-composition spent ≈ $70 against a pre-registered $18 (ceiling $22) and a proposal budget of
   $60 for all three dataset-styles runs**, and ≈ $47 of it is idle pods. Session 2 was run
   properly and came in at $12.84 of $18; the overrun is entirely session 1's unattended pods.
2. **The idle-pod failure mode is now on record twice** (ladder-A ≈ $32, ds-composition ≈ $47). In
   both cases the agent session died and `pod_watchdog` did not delete the pods; in both cases the
   balance, not the watchdog, ended the burn. The two guards that did work here are the hourly
   `balance_floor` (fired at $21.31 < $25) and `balance_alert` (wrote the `~/ALERTS.md` line) —
   both of which act on the *account*, hours after the money is gone, rather than on the *run*.
   The guard that would have caught it — `pod_budget_watch` (cron every 10 min: warn at 80 % of a
   run's pod-hour ceiling, **delete that run's pods at 100 %**, after pulling `artifacts`,
   `numbers.md`, `log.md` and `STATUS.md`) together with `~/podhours.log` and the per-run
   `podbudget` ceiling — was introduced by the `efficiency` run on **2026-09-23 15:19 UTC**, a day
   *after* session 1, and ds-composition's ceiling was registered at 21:00:22 that evening
   (`~/podbudget.log`). It is what makes session 2 auditable to the minute, and on session 1's
   numbers it would have deleted the pods at 35 h instead of ≈ 115. So this failure mode is
   plausibly already closed for runs that register a ceiling; what is *not* covered is a pod whose
   run never registered one, and the watchdog path that failed in both incidents. See the week-39
   digest for what this needs from Dan.
Update 2026-09-24 04:52 UTC (ds-rendering processed): **$181.42 (`rpbalance`)**, down from $220.36
at 2026-09-23 17:45. `podls` shows two pods still RUNNING on the account — `dsg-1` and `dsg-2`
(`ds-generator`, proposal 10, another host) — and **every `dsr-*` pod is gone**. The $38.94 since
Wednesday evening therefore covers ds-rendering's $16.01 **and** the two sibling proposal-10 runs
`ds-composition` and `ds-generator`, which are not in any row (not yet through review); the two are
not separable in the readings, so ds-rendering's row rests on `podbudget ds-rendering` and A40 list
price, which agree at $16.01 / 32.69 h. Week 39 (2026-09-21 → 09-27) to date by these rows:
**≈ $25.3** (lean-format ≈ $6.2, lean-seed2 ≈ $2.3, efficiency $0.77, **ds-rendering $16.01**).
Cumulative by rows since 2026-09-15: **≈ $188**. Still not in any row: the take-home baseline,
run4-grpo's and run1-lean's pods (not reviewed), the `lean-only` run (proposal 9, in review) and
the two remaining dataset-styles runs (`ds-composition`, `ds-generator`, proposal 10) — the last
of which is still burning. The balance is ≈ $131 above the $50 alert line with two pods running
(≈ $1/h if both are 3090-class). `~/balance.log` and `~/ALERTS.md` still do not exist on this host;
`~/podhours.log` now records per-pod hours and dollars at deletion, which is where this row's
per-pod figures come from.
