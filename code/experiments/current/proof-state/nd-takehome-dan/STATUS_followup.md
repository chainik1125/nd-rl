# STATUS — follow-up run (branch dan_novelty)

Started 2026-09-16. First campaign status: STATUS_campaign1.md. Brief: FOLLOWUP_BRIEF.md.

## Plan
Block A (depth-3 replication: 3 new f=0 sets x 2 seeds, 2 new f=0.1 sets x 2 seeds, EI + frozen + novelty) -> Block B (derived-double-negation reductio pool; 3+2 arms, frozen, base pass@1e4) -> Block C (cap-8 strict derived-ORE dial) if A+B done by 2026-09-17 04:00 UTC. Plan + expectations in log.md (01:40).

## Done
- Block A: 13/14 EI arms + 6/10 frozen done; f = 0 (8 arms) mean 0.339 SD 0.029, f = 0.1 (5 arms so far) mean 0.344 — replicates; novelty on every arm (artifacts/fu/blockA_summary.json, figures/followup_depth3_strips.png)
- Block B: done (5 EI + 5 frozen + coverage + novelty; artifacts/fu/blockB_summary.json, figures/followup_reductio.png)
- Block C: done (4 EI, 2 frozen, coverage, novelty; strict acquisition 0.008 / 0.008 / 0.022 / 0.022; artifacts/fu/blockC_summary.json, figures/followup_cap8_dial.png)
- Block A final: 14 EI + 10 frozen arms; f = 0 (8 arms) 0.339 ± 0.029, f = 0.1 (6 arms) 0.307 ± 0.092 — replicates
- Deliverables: followup.md, phase2.md §Follow-up, numbers.md and log.md sections, figures/followup_*.png; all results, checkpoints and pod logs pulled; all pods deleted
- RESUMED 2026-09-16 02:09 UTC after an unexplained session end at ~02:07 (possibly a usage limit; no LIMIT message was seen). Pod jobs unaffected.
- 01:41 five new depth-3 sets assembled and verified (0 written depth-3 in every f = 0 set); block-A queues launched on p1/p2/p3

## Running on pods
- none (p2, p3 deleted 04:16 UTC; p1 deleted 05:45 UTC)

## Next step
- none

DONE 2026-09-16 05:47 UTC
