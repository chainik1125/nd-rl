#!/usr/bin/env bash
# oe_setup.sh (POD side; started by olmo_pod.sh setup): Dan's pod set-up for the olmo elicitation curves.
# = his pod/olmo_ifcurve/setup.sh (vLLM 0.11.0, transformers 4.57.1 on the image's torch 2.8.0 / CUDA 12.8, the IF
# checkers' dependencies at IFBench's uv.lock versions, their nltk data, tokenizer files from dpo main, GPU health gate)
# plus peft 0.17.1 (his pod/olmo_ifladder/setup.sh) for the EDL runner; huggingface_hub 0.36.2 / hf-xet 1.8.0 as his.
# His code: github.com/dan-pandori/nd-takehome, partial sparse clone pinned at
#   /workspace/nd-takehome    8e744a89 (dan_olmo-ifcurve): olmo_ifladder/sample.py (H, rep), olmo_ifomega/grade.py
#   /workspace/nd-answeronly  93d36de7 (dan_olmo-answeronly, git worktree): olmo_answeronly/sample_prefix.py and
#                             olmo_transplant/grade_rows.py (T); its grade.py / vendor / problems / olmo32 / iffu
#                             sampler are identical to 8e744a89's (checked on serrano, 2026-11-25)
# Writes /workspace/oe/meta/{pod_env.txt,versions.json,pip_freeze.txt,gpu_status,setup.done}; idempotent.
set -uo pipefail
export HF_HOME=/workspace/.hf HF_HUB_DISABLE_TELEMETRY=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0 PIP_ROOT_USER_ACTION=ignore
REPO=https://github.com/dan-pandori/nd-takehome
C_IFC=8e744a89255449b89be89d42a738292e98a1aaf7
C_AO=93d36de764953bc624aafb9edce73d5dc70f0fb4
M=/workspace/oe/meta
mkdir -p $M /workspace/models /workspace/oe/runs
E=$M/pod_env.txt
say() { echo "$*"; }
{ date -u; nvidia-smi --query-gpu=name,memory.total,driver_version,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader
  echo "cpu.max $(cat /sys/fs/cgroup/cpu.max 2>/dev/null) cfs $(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null)"; nproc; free -g | head -2; df -h /workspace / | tail -2
  for i in $(seq 0 $(( $(nvidia-smi -L | wc -l) - 1 ))); do echo "GPU $i remapped:"; nvidia-smi -q -i $i | grep -A3 -i "Remapped Rows" | tail -3; done
  python --version; } > $E 2>&1
cat $E

# --- his code at the pinned commits (blobless clone, sparse: only the paths the jobs use)
PATHS="/olmo_ifladder/sample.py /olmo_ifomega/grade.py /olmo_ifomega/vendor/ /olmo_ifomega/problems.jsonl /olmo32/grade.py
 /olmo32/oi_math_utils.py /olmo_iffu/sample.py /olmo_iffu/problems_hint.jsonl /olmo_iffu/problems_hint2c.jsonl
 /olmo_ifcurve/checkpoints.csv /olmo_ifcurve/pids_armA.txt /olmo_ifcurve/pids_reach.txt /olmo_ifladder/lora.py
 /pod/olmo_ifomega/fetch.sh /olmo_answeronly/ /olmo_transplant/"
if [ ! -d /workspace/nd-takehome/.git ]; then
  git clone -q --filter=blob:none --no-checkout $REPO /workspace/nd-takehome || { echo "CLONE_FAIL"; exit 1; }
fi
cd /workspace/nd-takehome
git sparse-checkout set --no-cone $PATHS && git -c advice.detachedHead=false checkout -q --detach $C_IFC && git reset -q --hard $C_IFC \
  || { echo "CHECKOUT_FAIL ifcurve"; exit 1; }
if [ ! -d /workspace/nd-answeronly ]; then
  git worktree add -q --no-checkout --detach /workspace/nd-answeronly $C_AO || { echo "WORKTREE_FAIL"; exit 1; }
fi
(cd /workspace/nd-answeronly && git sparse-checkout set --no-cone $PATHS && git reset -q --hard $C_AO) || { echo "CHECKOUT_FAIL answeronly"; exit 1; }
H1=$(git -C /workspace/nd-takehome rev-parse HEAD); H2=$(git -C /workspace/nd-answeronly rev-parse HEAD)
[ "$H1" = $C_IFC ] && [ "$H2" = $C_AO ] || { echo "COMMIT_MISMATCH $H1 $H2"; exit 1; }
for f in olmo_ifomega/grade.py olmo_ifomega/problems.jsonl olmo32/grade.py olmo_iffu/sample.py; do
  cmp -s /workspace/nd-takehome/$f /workspace/nd-answeronly/$f || { echo "SHARED_FILE_DIFFERS $f"; exit 1; }; done
say "CODE_OK ifcurve=$H1 answeronly=$H2"

# --- his installs (pod/olmo_ifcurve/setup.sh; + peft as pod/olmo_ifladder/setup.sh)
pip install --break-system-packages -q "huggingface_hub[cli,hf_xet]==0.36.2" "hf-xet==1.8.0" 2>&1 | grep -v -i "warning\|notice" | tail -2
(cd /workspace/nd-takehome && bash pod/olmo_ifomega/fetch.sh allenai/Olmo-3-32B-Think-DPO main tok tokenizer.json tokenizer_config.json \
  special_tokens_map.json vocab.json merges.txt config.json generation_config.json) || { echo "TOKENIZER_FETCH_FAIL"; exit 1; }
pip install --break-system-packages -q "vllm==0.11.0" "transformers==4.57.1" "peft==0.17.1" "antlr4-python3-runtime==4.11" \
  "huggingface_hub[cli,hf_xet]==0.36.2" "hf-xet==1.8.0" pyarrow pandas \
  "nltk==3.9.2" "langdetect==1.0.9" "emoji==2.15.0" "syllapy==0.8.0" "absl-py==2.3.1" "immutabledict==4.2.2" 2>&1 | grep -v -i "warning\|notice" | tail -5
python -c "import nltk
for p in ['punkt', 'punkt_tab', 'stopwords', 'averaged_perceptron_tagger_eng']: nltk.download(p, quiet=True)
print('NLTK_OK', nltk.__version__)"
python - <<'EOF' 2>&1 | tail -1
import json, sys, vllm, torch, transformers, peft, sympy, huggingface_hub
v = {"python": sys.version.split()[0], "vllm": vllm.__version__, "torch": torch.__version__, "transformers": transformers.__version__,
     "peft": peft.__version__, "sympy": sympy.__version__, "huggingface_hub": huggingface_hub.__version__,
     "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "gpus": torch.cuda.device_count()}
json.dump(v, open("/workspace/oe/meta/versions.json", "w"))
print("VERSIONS", json.dumps(v))
EOF
(cd /workspace/nd-takehome && python -c "import sys; sys.path.insert(0, 'olmo_ifomega'); import grade, json
P = [json.loads(l) for l in open('olmo_ifomega/problems.jsonl')]
print('GRADER_OK', sum(grade.success(p['task'], '<think>x</think>Hello.', p) for p in P[:20]))" 2>&1 | tail -2)
# GPU health gate (his 2026-11-21 rule): every GPU must pass an 8192^2 matmul
NG=$(nvidia-smi -L | wc -l)
python -c "
import torch
ok = 0
for i in range(torch.cuda.device_count()):
    try:
        x = torch.randn(8192, 8192, device=f'cuda:{i}'); y = (x @ x).sum().item(); torch.cuda.synchronize(i); ok += 1
    except Exception as e:
        print('GPU_FAIL', i, repr(e)[:120])
print('GPU_TEST', ok, torch.cuda.device_count())
" 2>&1 | tail -3 | tee $M/gpu_test.txt
GT=$(grep GPU_TEST $M/gpu_test.txt | awk '{print $2}')
if [ "$NG" -ge 1 ] && [ "${GT:-0}" = "$NG" ]; then echo "GPU_OK $NG" > $M/gpu_status; else echo "GPU_BAD $NG gpus, ${GT:-0} passed" > $M/gpu_status; fi
cat $M/gpu_status
pip freeze 2>/dev/null > $M/pip_freeze.txt
grep -q '"vllm": "0.11.0"' $M/versions.json && grep -q '"transformers": "4.57.1"' $M/versions.json && grep -q '"peft": "0.17.1"' $M/versions.json \
  || { echo "VERSION_MISMATCH $(cat $M/versions.json)"; exit 1; }
grep -q "^GPU_OK" $M/gpu_status || { echo "GPU_BAD"; exit 1; }
date -u +%FT%TZ > $M/setup.done
echo "SETUP_DONE $(date -u +%FT%TZ)"
