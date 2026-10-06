#!/usr/bin/env bash
# THE single test-set run. Refuses to run twice. usage: bash test_run_once.sh ckpts/stage1_abs.pt ckpts/final.pt
set -e
if [ -f artifacts/TEST_RUN_DONE ]; then echo "TEST ALREADY RUN: $(cat artifacts/TEST_RUN_DONE)"; exit 1; fi
S1=$1; FINAL=$2
date -u > artifacts/TEST_RUN_DONE
echo "stage1=$S1 final=$FINAL" >> artifacts/TEST_RUN_DONE
for f in short long; do
  python3 prove.py --ckpt $S1 --in targets/test_${f}_prompts.jsonl --out artifacts/test_${f}_stage1.jsonl --greedy
  python3 prove.py --ckpt $FINAL --in targets/test_${f}_prompts.jsonl --out artifacts/test_${f}_final.jsonl --greedy
done
{
  echo "# score_test.py output, run once at $(cat artifacts/TEST_RUN_DONE | head -1) UTC"
  for m in stage1 final; do for f in short long; do echo -n "$m ($( [ $m = stage1 ] && echo $S1 || echo $FINAL )) : "; python3 score_test.py artifacts/test_${f}_${m}.jsonl; done; done
} | tee artifacts/test_scores.txt
