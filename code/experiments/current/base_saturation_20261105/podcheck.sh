#!/usr/bin/env bash
# podcheck.sh <alias>: verify a base_saturation pod after runpod_copy.sh. GPU and torch.cuda in
# /workspace/venv, Lean v4.34.0 accepts one proof and rejects one (read snapshot's lean gate),
# every code snapshot on the pod imports its entry point, zoneinfo works, CPU quota and disk.
# Bounded; writes nothing outside /workspace/sat/check.
set -euo pipefail
h=${1:?usage: podcheck.sh <alias>}
timeout 600 ssh -o BatchMode=yes -o ConnectTimeout=15 "$h" bash -s <<'EOF'
set -euo pipefail
source /workspace/sat/exp/steps.sh
mkdir -p $W/check
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
$PY -c "import torch; x=torch.randn(512,512,device='cuda'); print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), bool(torch.isfinite(x@x).all()))"
$PY -c "import datetime, zoneinfo; print('zoneinfo', datetime.datetime.now(zoneinfo.ZoneInfo('America/Los_Angeles')).strftime('%I:%M %p PT'))"
echo "cpu quota $(cpu_quota), nproc $(nproc), disk free $(df -h /workspace | tail -1 | awk '{print $4}'), mem $(free -g | awk '/Mem/{print $2}') GB"
/root/.elan/bin/lean --version
cd $CODE/rd
LEAN_GATE_LOG=$W/check/lean_gate.jsonl $PY -c "
from code.thinking.lean import lean_check
v = lean_check([('theorem chk (p : Prop) (hp : p) : p :=', 'by exact hp'),
                ('theorem chk (p q : Prop) (hp : p) : q :=', 'by exact hp')])
print('lean verdicts (want [True, False]):', v)
assert v == [True, False], v
"
for t in s1 cpt l2 l3 rd; do
  [ -d $CODE/$t ] || continue
  case $t in
    s1) m="code.thinking.train"; [ -f $CODE/s1/code/experiments/current/base_saturation_20261105/anneal.py ] && m="$m code.experiments.current.base_saturation_20261105.anneal"
        [ -f $CODE/s1/code/experiments/current/base_saturation_20261105/stage1_to0.py ] && m="$m code.experiments.current.base_saturation_20261105.stage1_to0" ;;
    cpt) m="code.experiments.current.circuit_diffing_20261101.cpt" ;;
    l2|l3) m="code.thinking.grpo" ;;
    rd) m="code.thinking.eval" ;;
  esac
  (cd $CODE/$t && CIRCUITS_RUNS=$RUNS $PY -c "
import importlib, sys
for m in sys.argv[1:]:
    importlib.import_module(m)
print('$t', open('COMMIT').read()[:8], 'imports ok:', ' '.join(sys.argv[1:]))
" $m)
done
echo PODCHECK_OK
EOF
