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
- **8:31 to 8:34 pm — the three tagged bases finished** (rc 0; 1,812 s, 1,525 s, 1,598 s; locks
  released). Each has 9,562,368 parameters (Dan's 9,560,832 plus two token rows in the embedding and
  the head), ran its matched step count (25,607 / 24,028 / 24,286), final validation loss 0.072 /
  0.079 / 0.075 (Dan's 1 % seed 0: 0.065, on his untagged held-out set, so not the same quantity).
  Registry rows were written offline (`uri: null`); nothing was uploaded. Sanity sample, 64 corpus
  theorems × 4 attempts: Lean accepted 218 to 229 of 256 under each of A, B and no tag.
- **8:37 pm — base read launched, serrano GPU 0** (`read --seeds 0,1,2 --tags A,B,none`; estimate
  1.1 h, cap 5 h; lab state 8 of 12 free before, 0 of ours). Watchdog attached to the real PID.
  To check when it ends: the E arm logs "tag states checked 0"; confirm from the rows that E differs
  between tags, or from the code that E inputs carry the tag.
- **9:11 pm — base read finished** (rc 0, 0.56 GPU-h; GPU 0 released; `pc_gpu` 1,759 s). QC clean
  for all 9 base-and-tag reads. The E arm's inputs carry the tag (its values differ between tags);
  the "tag states checked 0" is the sampler's counter, which E does not use.
- **9:37 pm — gate result: the manipulation check fails on 3 of 3 seeds** (Δ_open − Δ_ctrl =
  +0.309, +0.090, +0.078 against the frozen threshold log 2 = 0.693). Written up in
  `GATE_RESULT.md`; rates from `tag_gate_summary.py` (written after the read). By the frozen rule
  the RL arms are not run on these bases. Reported to Leon with a proposal to vary how strongly
  the tag is tied to the route before any RL.
- **9:40 pm — proposal to Leon:** tie-strength pilot (route records as 10 %, 50 %, 100 % of tag A's
  records; pretraining only, 1 seed each, about 1.5 GPU-h on lab GPUs), to start at 10:10 pm on his
  30-minute rule unless he says otherwise. An Opus agent built the option meanwhile (no GPU):
  `tagcorpus.py --route-share`, variants `p10`/`p50`/`p100` in `AT/data/`, `--variant` for
  pretraining, `tag_launch.py check`, `tag_gate.py`, 29 new tests. I checked the three corpora
  (155,000 keys each; all 1,550 route records under A; A has 15,500 / 3,100 / 1,550 records), that
  the default corpus hash is unchanged, and that `tag_gate.py` reproduces +0.309 / +0.090 / +0.078.
- **10:02 pm — PREREG Amendment 1** (the pilot's rule, confound and predictions), before any
  variant base exists.
- **10:10 pm — tie-strength pilot launched** on Leon's 30-minute rule (proposal sent 9:40 pm, no
  reply): `pretrain --variant p10` on serrano GPU 0 and `--variant p50` on GPU 3, seed 0 each
  (estimate 0.6 h, cap 2 h). Lab state before: 5 of 12 free, 0 of ours. `p100` follows when a
  serrano GPU frees up; then `check` on the three bases.
- **10:40 pm — `p50` base finished** (rc 0, 1,789 s; offline, `uri: null`). Sanity sample: Lean
  accepted 201 / 225 / 220 of 256 attempts under A / B / no tag.
- **10:41 pm — `p100` pretraining launched on serrano GPU 3** (freed by `p50`); `p10` still on GPU 0.
- **10:42 pm — `p10` base finished** (rc 0; offline). Sanity sample: 223 / 226 / 223 of 256 under
  A / B / no tag.
- **10:55–10:57 pm — check on `p10` and `p50`** (serrano GPU 0, 2 min). **Both pass:** Δ_open −
  Δ_ctrl = +3.294 (`p10`, opening 27× as likely under A as under B) and +9.149 (`p50`, about
  9,400×), against +0.309 for the first seed-0 base. Under B the opening falls below the first
  bases' level (mean log P −6.42 and −10.22, against −4.13).
- **11:12 pm — `p100` base finished** (rc 0, 1,811 s; offline). Sanity sample: 144 / 220 / 226 of
  256 under A / B / no tag. **11:20 pm — check on `p100`: passes** (+10.461). Pilot table added to
  `GATE_RESULT.md`. Pilot GPU time: 3 × 0.5 h pretraining + 3 min of checks.
- **11:00 pm — proposal to Leon:** seeds 1 and 2 at p10 plus the full base read on the three p10
  bases (about 1.7 GPU-h), pretraining to start at 11:30 pm on his 30-minute rule. An Opus agent
  made the read, the summary and the RL lines variant-aware meanwhile (no GPU; `--variant` on
  `read` and `tag_gate_summary.py`, lines `t<A|B>_<pNN>_s<seed>`, `preflight()` refusals,
  `tag_audit`; 92 tests pass, default summary output byte-identical).
- **11:30 pm — seeds 1 and 2 at p10 launched** on Leon's 30-minute rule (proposal 11:00 pm, no
  reply): serrano GPUs 0 and 3. Lab state before: 5 of 12 free, 0 of ours.
- **11:59 pm / 12:01 am (Nov 11) — p10 seeds 2 and 1 finished** (rc 0; offline). Sanity samples:
  seed 1 226 / 225 / 218, seed 2 222 / 221 / 214 of 256 under A / B / no tag.
- **12:01 am — full read on the three p10 bases launched, serrano GPU 0**
  (`read --seeds 0,1,2 --tags A,B,none --variant p10`; results to `AT/reads/pc_p10/`; estimate
  0.55 h). A first launch attempt at 12:00 was refused by the gate because seed 1's sample step
  still held GPU 0.
- **12:03 am — the p10 read stopped itself** (rc 3 after 143 s): another user (lukezeng) started on
  serrano GPU 0 and the never-share gate ended our job; lock released. Two arms of `tagAB_p10_s0@A`
  (E and U) were written to `AT/reads/pc_p10/raw/` before that. Not relaunched: the lab has 2 of
  12 GPUs free, and a launch would leave 1, below the minimum of 2. Waiting for the gate.
- **12:45 am — p10 read relaunched on serrano GPU 3** when the gate passed (lab 3 of 12 free
  before, 0 of ours). The allocator's OOM warning in the log is the sampler backing off its batch
  size; the first read logged the same warning 4 times.
- **1:11 am — p10 read finished** (rc 0; `pc_gpu` 1,396 s; checks for seeds 1 and 2). **Gate at
  p10: passes on 3 of 3 seeds** (27×, 39×, 23×). Uses reductio on the 106: about 5 % under A,
  0.03–0.68 % under B. Forced-opening success about 1 % under both. Tables in `GATE_RESULT.md`.
  RL arms wait for Leon's go.
