---
written_on: 2026-09-26
written_by: agent:claude
---

# Mid-training on long glued proofs: working folder

This folder holds the code of the mid-training experiment as it ran on 2026-09-26 in
`/scratch/cowifornia/nd-rl/ab3/midtrain/`. It asks whether a short supervised stage on long proofs glued from shorter
ones, before the arm-C RL on lengths 6-11, helps the prover past its training length. One seed; four arms (base,
long, short, more).

Results and evidence:
[experiment-summaries/2026-09-26-midtraining-long-glued](../../../../experiment-summaries/2026-09-26-midtraining-long-glued/README.md).

**About the copies.**
- *Byte-identical.* Every file except this README is a copy of a file in the scratch folder, checked with `cmp`. The
  files are not reformatted and are excluded from ruff, like the other frozen snapshots.
- *Renamed on copy* (contents unchanged): the experiment's own `README.md` became `midtrain_README.md`, and
  `build/tools/` became `build_tools/`.
- *Not copied.*
  - `rl_C.py` is byte-identical to
    [`length_generalisation_20260925/rl_C.py`](../length_generalisation_20260925/rl_C.py).
  - Evaluation uses lengthexp's unchanged `evaluate.py`, in the same folder.
  - No checkpoints, data files, evaluation rows or run outputs. The data files' hashes are in `data/SHA256SUMS`.
- *Where they write.* The scripts use hard-coded scratch paths. To re-run anything, work in the scratch copies.

## Files

| file | what it is |
|---|---|
| `midtrain_README.md` | the experiment's own README: question, design, commands, results |
| `build_tools/build_mid.py` | builds the glued proofs: parts from the run-2 reserve, renamed onto disjoint atoms, joined by a random conjunction tree |
| `build_tools/assemble_mid.py`, `supply.py` | assembles the three files from the built pools; counts the parts available per length |
| `build_tools/validate_k3.py` | checks that three-part joins are additive in length on the cases exact search can resolve |
| `build_tools/verify_mid.py` | re-verifies every glued proof and checks disjointness against the existing pools |
| `data/POOLS_MIDTRAIN.md`, `data/SHA256SUMS` | how the data was built and checked; hashes of `sft_long`, `sft_short` and `test_long` |
| `build_long.py` | turns `test_long` into the evaluation's `testsets/long.jsonl` |
| `midtrain.py` | the mid-training stage: the seed recipe's `finetune` on a 10,000-record mix, 3 epochs, identical in all arms |
| `make_frozen.py` | makes a frozen harness copy per arm with only `pretrain.pt` replaced, checked by sha256 |
| `mt_evaluate.py` | runs lengthexp's `evaluate.py` against this folder's test sets |
| `pipeline.py` | the end-to-end runner: mid-training, frozen copies, RL through the GPU lock, evaluation |
| `analyse.py` | writes `report.md`, `results.json` and the figures |
| `testsets/*MANIFEST.json` | the test-set manifests (standard and matched sets are lengthexp's final draw) |
