#!/usr/bin/env bash
# Matched-attempt comparison tables (EI r1/r2 vs GRPO it39/it78; thinking vs no-think), with
# hierarchical-bootstrap intervals from code.thinking.stats. CPU only; run from the repo root.
set -u
M=/scratch/cowifornia/nd-rl/think/runs/eval/matched; E=/scratch/cowifornia/nd-rl/think/runs/eval
D=code/experiments/current/thinking_traces_20261005/matched_eval
export ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
py=code/.venv/bin/python
for S in h12 tb72; do
  if [ $S = h12 ]; then
    base="--group T0_stage1_s12=$E/h12_T0_s1/rows_plain_B0.jsonl,$E/h12_T0_s2/rows_plain_B0.jsonl"
    base="$base --group T1v1_stage1_B0=$E/h12_T1v1_s0/rows_plain_B0.jsonl --group T1v1_stage1_B256=$E/h12_T1v1_s0/rows_plain_B256.jsonl"
  else
    base="--group T0_stage1_s0=$E/tb72_T0_s0/rows_plain_B0.jsonl"
    base="$base --group T1v1_stage1_B0=$E/tb72_T1v1_s0/rows_plain_B0.jsonl --group T1v1_stage1_B256=$E/tb72_T1v1_s0/rows_plain_B256.jsonl"
  fi
  g=""; p=""
  for pt in "r1 it39" "r2 it78"; do set -- $pt; r=$1; i=$2
    g="$g --group EI_T0_$r=$M/${S}_eiT0_$r/rows_plain_B0.jsonl --group GRPO_T0_$i=$M/${S}_grpoT0_$i/rows_plain_B0.jsonl"
    for B in 0 64 256; do
      g="$g --group EI_T1v1e_${r}_B$B=$M/${S}_eiT1v1e_$r/rows_plain_B$B.jsonl --group GRPO_T1v1e_${i}_B$B=$M/${S}_grpoT1v1e_$i/rows_plain_B$B.jsonl"
    done
    p="$p --pair GRPO_T0_$i@vs@EI_T0_$r --pair GRPO_T1v1e_${i}_B256@vs@EI_T1v1e_${r}_B256"
    p="$p --pair EI_T1v1e_${r}_B256@vs@EI_T0_$r --pair GRPO_T1v1e_${i}_B256@vs@GRPO_T0_$i"
    p="$p --pair EI_T1v1e_${r}_B256@vs@EI_T1v1e_${r}_B0 --pair GRPO_T1v1e_${i}_B256@vs@GRPO_T1v1e_${i}_B0"
  done
  # drop groups whose rows file does not exist yet
  args=""; for tok in $base $g; do
    case $tok in --group) continue;; esac; f=${tok#*=}; ok=1
    for x in ${f//,/ }; do [ -f $x ] || ok=0; done; [ $ok = 1 ] && args="$args --group $tok"; done
  pairs=""; for tok in $p; do case $tok in --pair) continue;; esac
    a=${tok%%@vs@*}; b=${tok##*@vs@}; echo "$args" | grep -q -- "--group $a=" && echo "$args" | grep -q -- "--group $b=" && pairs="$pairs --pair \"$a vs $b\""; done
  eval $py -m code.thinking.stats table $args $pairs --k 1,5 --tokens 1000,3000 --title "\"matched $S\"" --out $D/matched_$S.md
done
