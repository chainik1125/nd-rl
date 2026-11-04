---
written_on: 2026-11-04
written_by: agent:claude
---

# knockout_danbase: GRPO on Dan's K12⁻ knockout bases

**Question (Dan).** Can GRPO learn proof by contradiction (σ) from his 0 %-σ pretraining, the
create-or-elicit knockout bases on which his expert iteration did not learn it?

**Design** ([`PREREG.md`](PREREG.md), sha256 in [`log.md`](log.md)): his bases, his 4,495
rl_targets, his environment caps, his held-out 106, his measure (cumulative k 32 count over 16
draws matched to his ladder's x) and his bar (14); only the algorithm differs (Leon's GRPO at
f35d3daf). Secondary: the same bases with our σ-rich pool. Reference: his full K12 base.

Artifacts: `/scratch/cowifornia/nd-rl/knockout_danbase/` (KD) on each machine. Branch
`leon/knockout-grpo`.

## Scripts

| file | what | where |
|---|---|---|
| `common.py` | the lines, paths, Dan's compute rule, the GRPO recipe (`grpo_argv`) | — |
| `danread.py` | a held-out read with Dan's own code (generate, judge, σ flags) | GPU |
| `basecheck.sh` | his round-1 draw on the four bases (his seed, batch, set) | GPU |
| `pilot.sh` | the 10-iteration pilot on the full base | GPU |
| `danbase_log_f35d3daf.patch` | logging-only patch to grpo.py (token counts for both x rules, peak memory) | — |
| `dline.py` | one segment of one line: GRPO plus the online reads, control files, heartbeat | GPU host |
| `sup.py` | the supervisor (serrano) and its standby (temescal): liveness, respawns, moves, lease, backups, collection, stop rule | serrano / temescal |
| `gpu.py` | who is on which GPU, per host | — |
| `merge.py` | one line's merged view across segments, Dan's matched draws, the stop rule | CPU |
| `tables.py` | `numbers.json` / `numbers.md` (with the σ share of accepted training proofs) | CPU |
| `fig.py` | the figures (`code/ndbench/figures/style.py`): Dan's layout, his measure against x, σ share | CPU |

## Operating

- Supervisor: `KD/sup/run_sup.sh` (serrano), `KD/sup/run_standby.sh` (temescal); restart with
  `KD/sup/restart_sup.sh`. State `KD/sup/state.json`, events `KD/sup/events.md`, log
  `KD/sup/sup.log`, lease `KD/sup/lease.json`.
- Analysis (serrano, from the worktree root):
  `/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python code/experiments/current/knockout_danbase_20261104/tables.py`,
  then `code/.venv/bin/python -m code.experiments.current.knockout_danbase_20261104.fig`.
