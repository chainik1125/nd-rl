#!/usr/bin/env bash
# p6: step 2 (Qwen3-Coder-30B-A3B in-context, 3 forms) then step 3 (Qwen3 sizes on the RL-found classes, Lean form, k = 16).
cd /workspace/nd-takehome
until grep -q SETUP_DONE artifacts/setup_p6.log; do sleep 30; done
mkdir -p artifacts/r1
python3 run_vllm.py --model Qwen/Qwen3-Coder-30B-A3B-Instruct --prompts data/r1/prompts.jsonl --out artifacts/r1/gens_qwen30b.jsonl --n 8 --temperature 0.7 --max_tokens 1200 --batch 256 > artifacts/r1/step2.log 2>&1
echo STEP2_DONE
for m in Qwen3-0.6B Qwen3-1.7B Qwen3-4B Qwen3-8B Qwen3-14B Qwen3-32B; do
  python3 run_vllm.py --model Qwen/$m --prompts data/r1/scale_prompts.jsonl --out artifacts/r1/gens_scale_$m.jsonl --n 16 --temperature 0.7 --max_tokens 1200 --batch 128 --thinking_off > artifacts/r1/step3_$m.log 2>&1
  echo "STEP3_DONE $m"
done
echo STEP3_ALL_DONE
