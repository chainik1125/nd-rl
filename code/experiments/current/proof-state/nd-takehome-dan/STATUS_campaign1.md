# STATUS — novelty campaign (branch dan_novelty)

Started 2026-09-15 19:04 UTC. The take-home run's final status is in STATUS_takehome.md.

## Plan
Phase 1 (now): novelty.py log-probs + reachability; coverage.py base-model sampling (transfer k=1e4, val36 k=1e5) on p1; minlen.py on p2. Then Phase 2 (controlled-coverage pretraining, patterns P1-P3, f dial), Phase 3 if time.

## Done
- Phase 1: phase1.md (log-probs, reachability, empirical check on all 1,638 transfer theorems at k=1e4 and val36 at k=1e5, surprisal loci, minlen labels)
- Phase 2: phase2.md (30 Stage-1 sets/models, 24 EI arms + 6 frozen controls, base pass@1e4 and log-probs for the f=0 arms; figures/phase2_acquisition.png is the main result)
- Phase 3: phase3.md (textbook pool 623, precursor injection, val36 per round by bin, base reachability of newly solved validation theorems)
- campaign.md (≤300 words + 3 figures); numbers.md and log.md complete; all results, checkpoints and pod logs pulled; pods deleted

## Running on pods
- none (p1, p2, p3 deleted 2026-09-16 00:26 UTC)

## Next step
- none

DONE 2026-09-16 00:26 UTC
