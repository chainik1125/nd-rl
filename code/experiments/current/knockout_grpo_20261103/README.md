---
written_on: 2026-11-03
written_by: agent:claude
---

# knockout_grpo: Dan's σ knockout / dose experiment with GRPO and thinking traces

This replicates Dan's create-or-elicit / coe-dose01 chart: how much RL compute is needed to solve
the full bases' median number of the 106 strict σ-covered held-out theorems, by σ's share of the
pretraining proofs. It uses Leon's GRPO + thinking-trace pipeline instead of T1 expert iteration.
Leon asked for it ("can you rerun this properly to match dans") for the midterm report's section
"How much RL is needed to recover rare or fully excluded proof strategies?".

- **Design and predictions:** [`PREREG.md`](PREREG.md) (sha256 in [`log.md`](log.md), frozen
  before any GPU job), plus Amendment 1 (hold window) and the gate amendment in `log.md`.
- **Run log:** [`log.md`](log.md).
- **Numbers:** [`numbers.md`](numbers.md) / `numbers.json`.
- **Summary:** `experiment-summaries/2026-11-03-knockout-grpo-thinking/README.md`.
- **Branch:** `leon/knockout-grpo` (from `leon/high-rl-paper` at `50d0336`), worktree
  `/scratch/cowifornia/nd-rl/worktrees/knockout-grpo`. Artifacts (corpora, pickles, bases, runs,
  reads, Dan's bucket copies) are in `/scratch/cowifornia/nd-rl/knockout_grpo/` (`KO`) on serrano;
  ynez and temescal hold their own runs under the same path.

## Scripts

| script | what | where it runs |
|---|---|---|
| `common.py` | paths; Dan's σ / σ⁺ / lemma-route labels (his `coe_sigma` / `coe_assemble`, unchanged) | — |
| `sigma.py` | Dan's labels for a corpus of ND proofs; `--name k12` is the positive control | CPU |
| `refill_pool.py` | Dan's eligible refill pool (`coe_assemble.consider` with his helpers) plus our exclusion sets | CPU |
| `assemble.py plan` / `write` | the ko / d01 / d1 corpora from S1g3 (Dan's removal, refill and dose rules) | CPU |
| `base.py ARM SEED` | a Stage-1 base (train.py at `cd089700`) and its Stage-1 token count | GPU |
| `line.py` | one GRPO line: L2 (`fa28f6eb` + `snap_iters_fa28f6eb.patch`), the long L3 (`f35d3daf` + patch), the online reads and the stop rule; `--resume_line`, `--x2`, `--pilot_l3` | GPU |
| `kq.py` | the per-machine GPU-gated queue with the watchdog | — |
| `measure.py` | E32 (unbiased pass@32 summed over the 106) and R | — |
| `dan_axis.py` | Dan's ladders recomputed with our R rule from his registry rows | CPU |
| `pool_sigma.py` | strict σ-covered share of the RL pools | CPU |
| `sigma_proofs.py` | Dan's σ labels on the Lean-accepted proofs of a read | CPU |
| `health.py` / `health_loop.sh` / `watch_all.sh` / `collect.sh` | monitoring | CPU |
| `analyze.py` | `numbers.json` / `numbers.md` (bar, first and held crossings, Dan-matched cumulative y) | CPU |
| `fig.py` | the figures (`code/ndbench/figures/style.py`) | CPU |
| `project.py`, `mkjobs_gate.py`, `mkjobs_bulk.py`, `mkjobs_line2.py`, `initread*.sh` | the gate phase and job lists | — |

## Rerun

From the worktree root, with `ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1`:

```bash
PY=/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python   # the runs' environment
$PY -m code.experiments.current.knockout_grpo_20261103.sigma /scratch/cowifornia/nd-rl/bench/dan_ckpts/data/train_k12.jsonl --name k12
$PY -m code.experiments.current.knockout_grpo_20261103.refill_pool
# search traces of the candidates with searchtrace.py at 79749ce (log.md), then:
$PY -m code.experiments.current.knockout_grpo_20261103.assemble plan
$PY -m code.experiments.current.knockout_grpo_20261103.assemble write
# GPU: KO/tools/{base.py,line.py} through kq.py (log.md has every job); then, CPU:
python3 code/experiments/current/knockout_grpo_20261103/dan_axis.py
python3 code/experiments/current/knockout_grpo_20261103/analyze.py
code/.venv/bin/python -m code.experiments.current.knockout_grpo_20261103.fig
```
