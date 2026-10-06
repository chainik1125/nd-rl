#!/usr/bin/env bash
# p6 setup: vLLM + model downloads (Qwen3-Coder-30B-A3B-Instruct for step 2; Qwen3 instruct sizes for step 3).
cd /workspace/nd-takehome
pip install -q --break-system-packages vllm > artifacts/pip_vllm.log 2>&1
python3 -c "import vllm; print('vllm', vllm.__version__)"
pip install -q --break-system-packages -U huggingface_hub >> artifacts/pip_vllm.log 2>&1
python3 - <<'PY'
from huggingface_hub import snapshot_download
for m in ['Qwen/Qwen3-Coder-30B-A3B-Instruct']:
    print(m, snapshot_download(m), flush=True)
PY
echo SETUP_DONE
