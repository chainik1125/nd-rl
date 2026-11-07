# Sourced on ynez before any GPU process of this read. GPU 0 by PCI order, as nvidia-smi numbers it.
export ND_OFFLINE=1 ND_REGISTRY=0 PYTHONDONTWRITEBYTECODE=1
export LEAN_GATE_WORKERS=8 LEAN_GATE_LOG=/scratch/cowifornia/nd-rl/factor_read/logs/lean_gate.jsonl
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export PY=/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python
export FR=/scratch/cowifornia/nd-rl/factor_read
