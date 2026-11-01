#!/usr/bin/env bash
# Block until serrano has fewer than 3 GPU jobs in total (every user) and GPU $1 is idle.
GPU=$1
while :; do
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u | wc -l)
  mine=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)
  [[ "$busy" -lt 3 && "$mine" -eq 0 ]] && exit 0
  sleep 30
done
