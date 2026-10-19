# Runnable experiments

Put active experiment-specific code in `current/` and completed implementations in `archived/`.
Reusable components belong in `../shared_code/`, tests in `../tests/`, and dated outcome records in
[`../../experiment-summaries/`](../../experiment-summaries/README.md).

Current experiments:

- [`r1_cot/`](current/r1_cot/README.md): symbolic scratchpad reasoning for natural-deduction RL.
- [`long_proofs_20261013/`](current/long_proofs_20261013/README.md): generators and an exact
  minimal-length certifier for natural-deduction theorems that need 20+ body lines.
- [`long_rl_20261014/`](current/long_rl_20261014/README.md): the whole-proof training pipeline
  (run 198 pretraining, then expert iteration), see [PIPELINES.md](../../docs/PIPELINES.md).
- [`autoresearch/`](current/autoresearch/pod.py): only the RunPod helpers (`pod.py`, `watchdog.sh`)
  that `long_rl_20261014/launch.py` imports; the autoresearch harness is on branch
  `autoresearch-pretrain`.

Archived experiment snapshots:

- [`nd_sprints_20260609_11/`](archived/nd_sprints_20260609_11/IMPORT.md): the connected
  bootstrapping, proof-novelty, and first-order natural-deduction sprints from the earlier
  repository.
- [`olmo_single_gpu_20260909/`](archived/olmo_single_gpu_20260909/IMPORT.md): OLMo-32B ICL, SFT, DPO, and verifier-RL pilot.
- [`cap_comparison_20260910/`](archived/cap_comparison_20260910/IMPORT.md): direct and symbolic-CoT proof-cap comparison.
- [`proof_fragments_20260910/`](archived/proof_fragments_20260910/IMPORT.md): fragment selection and backward-reset curriculum.
- [`flop_nd_20260910/`](archived/flop_nd_20260910/IMPORT.md): symbolic and token-LM FLoP adaptations.

These imported snapshots omit remote data, full checkpoints, and result payloads. Their dated
reports and key figures live in [`../../experiment-summaries/`](../../experiment-summaries/README.md).
