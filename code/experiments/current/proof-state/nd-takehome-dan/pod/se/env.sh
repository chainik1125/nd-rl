# sourced by every state-env pod job
export PATH=$HOME/.elan/bin:$PATH
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LEAN_GATE_WORKERS=${LEAN_GATE_WORKERS:-32}
export ND_SAMPLE_PATH=fast
cd /workspace/nd-takehome
mkdir -p artifacts/se/logs ckpts/se/ladder
