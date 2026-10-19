#!/usr/bin/env bash
# On the pod: terminate this pod after $1 seconds. Pods are otherwise torn down only by the
# machine that rented them; if it sleeps or loses the network, this bounds the bill.
# Uses the pod-scoped RunPod key injected into PID 1's environment, never our account key.
sleep "$1"
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(POD_ID|API_KEY)=' | sed 's/^/export /')"
curl -s -X POST "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
  -H 'Content-Type: application/json' \
  -d "{\"query\": \"mutation { podTerminate(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) }\"}"
