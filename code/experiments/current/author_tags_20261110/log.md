---
written_on: 2026-11-10
written_by: agent:claude (orchestrator)
---

# Author tags: run log

Times are PT (UTC−8) unless marked Z.

- **2026-11-10, about 7:20 pm — proposal to Leon.** After he called the experiment "interesting and
  worth running": gate = 3 tagged bases plus the base read under A, B and no tag, about 2.5 GPU-h on
  lab GPUs; then RL under B (about 11 GPU-h) and under A (about 5), which wait for him to see the
  gate result. I said the gate would start when the build passed its tests unless he said otherwise.
- **8:00 pm — pre-registration frozen** (PREREG.md, commit 05d16bd, pushed to `leon/author-tags`),
  before any full-size base existed.
- **8:01 pm — pretraining launched, seed 0, serrano GPU 0** (no reply from Leon in the 40 minutes
  since the proposal; started on his 30-minute rule). Lab state: 8 of 12 GPUs free before, 0 of ours.
- **8:03 pm — launcher bug found and fixed.** The watchdog reported EXIT at once: `tag_launch.py`
  started the job as `nohup setsid python ...` with `start_new_session=True`, so `setsid` forked
  and the PID handed to the watchdog was already dead. The job itself was running. Removed the
  extra `setsid` from both Popen calls, stopped the stale watchdog (PID 1043549) and attached a new
  one to the real job PID (1043548).
- **8:04 pm — seeds 1 and 2 launched on serrano GPUs 1 and 3** with the fixed launcher; all three
  watchdogs report "watching". Seed 0 runs at about 15 steps/s after compile (25,607 steps, about
  29 min). All five serrano GPUs are at the 200 W cap; lab state now 5 of 12 free, 3 of ours.
- Note: the line `params 10734336` in the pretraining log is Dan's `state_train.py` printing the
  size of a placeholder control model before the `best` recipe builds the real one (6 layers,
  d 384, d_ff 1280). Check `n_params` in the saved checkpoint (Dan's 1 % base: 9,560,832).
