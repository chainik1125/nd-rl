# RL hill-climb A/B (2026-09-23): frozen harness, controller, briefs

Archived sources of the GEAR-versus-orchestration A/B test. `common/SPEC.md` is the contract; `common/harness.py`,
`score.py`, `metrics.py`, `rl.py` are the frozen files each arm started from; `common/score_host.py` and
`score_daemon.py` are the host-side sealed scorer; `gear/` holds the GEAR-Evolve controller as written and
as the workers left it (`controller_evolved_final.py`); `briefs/` and `docker/` are the worker briefs and
container scripts. Results and evidence: `experiment-summaries/2026-09-23-rl-hillclimb-ab/`. Raw runs and
checkpoints stayed in `/scratch/cowifornia/nd-rl/ab/` (not in Git). Data files and the checkpoint are not
copied here; the harness expects them under `frozen/data/` and `frozen/pretrain.pt` as SPEC.md describes.
