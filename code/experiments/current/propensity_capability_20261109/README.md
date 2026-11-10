---
written_on: 2026-11-09
written_by: agent:claude
---

# propensity_capability_20261109

For a trained prover and a proof direction (reductio, negation, conditional, cases), this read separates
**propensity** (how often the model opens with the direction unprompted) from **capability** (how often it finishes
when its first proof action is forced to be that direction's opening). It reads both at a pretrained base and at the
end of its RL line, for Dan's 0 %, 0.1 % and 1 % σ-dose lines (3 seeds each) and his full-corpus base, on the strict
106 and 240 other theorems. A base-to-end change becomes elicitation, a candidate capability gain, a mix or neither
under rules fixed in [PREREG.md](PREREG.md) before any result. A secondary analysis asks whether a rise in propensity
stays where the direction is needed (router) or spreads to theorems where it is not (default shift).

It generalises the one-direction factor read ([`../factor_read_20261106/`](../factor_read_20261106/), summary in
[`experiment-summaries/2026-11-06-factor-read-strict-106/`](../../../../experiment-summaries/2026-11-06-factor-read-strict-106/README.md)).

**Status:** pre-registered, not run. No result exists.

## Files

| file | what | owner |
|---|---|---|
| [SPEC.md](SPEC.md) | the build spec, including the result-file schema (the contract) | coordinator |
| [PREREG.md](PREREG.md) | question, definitions, the reading rule and thresholds, traps, when the split is not informative | analysis |
| `analyze.py` | reads the result files, writes `results/results.json` and `results/tables.md`; `--calibrate` reproduces PREREG's noise numbers | analysis |
| `fig.py` | the propensity-against-capability figure, through `code/ndbench/figures/style.py` | analysis |
| `test_analyze.py` | tests on synthetic result files, one case per reading and per router verdict; also writes a demo set | analysis |
| `pc_lib.py`, `pc_sets.py`, `pc_run.py`, `run_read.sh` | directions, sets, the GPU read and its gated launcher | runner |

Outputs of the read: one `<ckpt_id>.jsonl` per checkpoint in `/scratch/cowifornia/nd-rl/propensity_capability/results/`
(counts only, no proof text). Raw attempts, which hold proof text, stay in `.../raw/` and are never committed.

## Running the analysis

From the repository root, with the project's Python (`code/.venv/bin/python`; on serrano the venv in
`/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv` works):

```bash
D=code/experiments/current/propensity_capability_20261109
PY=code/.venv/bin/python

$PY $D/analyze.py --results /scratch/cowifornia/nd-rl/propensity_capability/results --out $D/results
$PY -m code.experiments.current.propensity_capability_20261109.fig \
    --results $D/results/results.json --out $D/results/propensity_capability
$PY -m pytest -c code/pyproject.toml $D/test_analyze.py
```

`tables.md` opens with the pre-registered verdict on whether the split is informative here, then the dose-level
readings, the per-seed readings, levels per checkpoint, unforced success, and the secondary and exploratory tables.

To see the analysis and figure on invented data: `$PY $D/test_analyze.py /tmp/somewhere/results` writes a synthetic read
in the schema (its numbers mean nothing), then run the two commands above on it.

To recompute the factor-read noise numbers that set the thresholds:
`$PY $D/analyze.py --calibrate /scratch/cowifornia/nd-rl/factor_read/summaries`.

## The figure

One panel per direction (columns) and set (rows). x: the exact first-move propensity; y: the share of forced attempts
solved; both log10, so an arrow's two extents are the two changes the reading uses, and the dotted diagonals are
constant P × C. An arrow per seed from base (hollow) to RL end (filled); colour and line style both encode the dose, so
it reads in greyscale. Below 1 % (grey band, not read) each decade is drawn a third as wide, so propensities down to
10⁻⁶ % stay visible without squeezing the 1–100 % range. The legend gives each dose's x at its RL ends: the doses are
at unequal RL compute and the figure does not compare them.
