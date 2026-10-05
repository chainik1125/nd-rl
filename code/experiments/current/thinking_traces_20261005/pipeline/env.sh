# Source from the repository root (the nd-rl worktree):
#   source code/experiments/current/thinking_traces_20261005/pipeline/env.sh
# Paths and settings shared by every command in README.md.  Nothing here starts a job.
export REPO=$(pwd)
export PY="uv run --project code --no-sync python"
export PYTHONDONTWRITEBYTECODE=1
export ND_OFFLINE=1            # the entry points force it too: never upload to Dan's bucket
export ND_REGISTRY_SYNC=0
export ND_LEAN=${ND_LEAN:-$HOME/.elan/bin/lean}   # 4.34.0 here; pods run 4.34.1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# inputs (read only)
export FORK=/scratch/cowifornia/nd-rl/think/nd-takehome          # Dan's fork, 4dcd0e4f (eval sets, RL pools)
export K12=/scratch/cowifornia/nd-rl/bench/dan_ckpts/data/train_k12.jsonl
export HELDOUT=/scratch/cowifornia/nd-rl/bench/dan_ckpts/data/heldout.jsonl   # = fork data/p2/heldout.jsonl
export STAGE1=/scratch/cowifornia/nd-rl/bench/dan_ckpts/best-state/stage1/stage1_best12_s0_b1200.pt
export OOD=/scratch/cowifornia/nd-rl/think/ood_check/sets
export TB_DEV=$FORK/data/eval_only/textbook72/textbook_dev.jsonl
export TB_TRAIN=$FORK/data/eval_only/textbook72/textbook_train.jsonl
export TARGETS=$FORK/data/ladder/rl_targets.jsonl                # 4,495 RL targets (the ladder's)
export TRANSFER=$FORK/data/ladder/transfer.jsonl                 # 2,285, never trained on
export PILOT=$REPO/code/experiments/current/thinking_traces_20261005/pilot_traces.jsonl
# traces to come (not produced yet): LLM traces from code/thinking/gen, programmatic traces (PROPOSAL §3.1 d)
export LLM_TRACES=${LLM_TRACES:-/scratch/cowifornia/nd-rl/think/traces/llm_traces.jsonl}
export PROG_TRACES=${PROG_TRACES:-/scratch/cowifornia/nd-rl/think/traces/prog_traces.jsonl}

# eval sets: Tier A (Charles long, within reach), Tier B families, textbook_dev58 (OOD_CHECK.md)
export SETS=$OOD/charles_long_reach.jsonl,$OOD/ood_or_reorder.jsonl,$OOD/ood_and_reorder.jsonl,$OOD/ood_commute.jsonl,$OOD/ood_case_product.jsonl,$TB_DEV
export SETS_FIRST=$OOD/charles_long_reach.jsonl,$TB_DEV       # 138 theorems: the first, cheap read

# outputs (large files stay on /scratch)
export RUNS=${RUNS:-/scratch/cowifornia/nd-rl/think/runs}
# the entry points create their output directories
