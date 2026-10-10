# code/finetuning

Code for the `finetuning` workstream (`docs/finetuning/plan.md`). Run as `uv run python -m
finetuning.<module>` from `code/`, or with `code/.venv/bin/python`.

Planned modules (plan § 7): `passk_area.py` (M3: pass@k curves, area, sharpening / expansion
split, k\*, from Dan's dumps or our evaluator output), `support.py` (M4: teacher-forced
log-probs of proofs under a checkpoint, budget classes), `delta_anatomy.py` (M5), `subset_ft.py`
(M1: parameter masks as gradient hooks around Dan's unmodified `state_train.py`), `edl.py` (M2:
prequential driver), `streams.py` (the data streams and their leak audits), `MANIFEST.json`
(Dan's worktree commit and the hashes of every file imported from it).

Dan's code is imported from `~/.cache/nd-rl/fork_dan` at the pinned commit and never edited;
every proof is judged by the pinned `nd_verify` through `strict_verify.py`.
