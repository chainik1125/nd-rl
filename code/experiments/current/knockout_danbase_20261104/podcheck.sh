#!/usr/bin/env bash
# podcheck.sh <pod alias> <base file>: verify a Runpod pod before its line resumes there.
# GPU and torch.cuda in /workspace/venv, Lean v4.34.0 accepts one proof and rejects one, and a tiny
# Dan read (k 2 on the 106) of the line's base runs end to end. Bounded; nothing is written outside
# /workspace/check on the pod.
set -euo pipefail
h=$1
base=$2
timeout 900 ssh -o BatchMode=yes -o ConnectTimeout=15 "$h" bash -s -- "$base" <<'EOF'
set -euo pipefail
base=$1
PY=/workspace/venv/bin/python
mkdir -p /workspace/check
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
$PY -c "import torch; x=torch.randn(512,512,device='cuda'); print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), bool(torch.isfinite(x@x).all()))"
/root/.elan/bin/lean --version
cd /workspace/kd/snap/f35d3daf_danbase
LEAN_GATE_LOG=/workspace/check/lean_gate.jsonl ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 $PY -c "
from code.thinking.lean import lean_check
v = lean_check([('theorem chk (p : Prop) (hp : p) : p :=', 'by exact hp'),
                ('theorem chk (p q : Prop) (hp : p) : q :=', 'by exact hp')])
print('lean verdicts (want [True, False]):', v)
assert v == [True, False], v
"
cd /workspace/exp
ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 LEAN_GATE_WORKERS=8 CUDA_VISIBLE_DEVICES=0 \
  DANBASE_PY=$PY timeout 600 $PY danread.py --ckpt /workspace/kd/bases/$base --seed 0 --round 0 \
  --out /workspace/check/read_k2.json --only106 --k 2 --batch 128 > /workspace/check/read_k2.log 2>&1
$PY -c "
import json; d = json.load(open('/workspace/check/read_k2.json'))
print('tiny read ok:', {k: (len(v) if isinstance(v, list) else v) for k, v in d.items() if k in ('strict_solved', 'first32_solved', 'first32_n', 'e32')})
"
echo PODCHECK_OK
EOF
