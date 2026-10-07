# Running the test-time-loops loop

Same launcher as the pretraining loop; see [its LOOP.md](../autoresearch/LOOP.md) for how
sessions, restarts, script-owned commits, terminal output and cancellation work. Differences:

- Branch `autoresearch-loops`, from a checkout of its own (`~/repos/nd-rl-11` is the first). One
  agent per checkout; more agents need more checkouts on the same branch.
- Artifacts go to `artifacts/autoresearch_loops/`, pods are named `autoresearch-ndrl-loops-...`,
  experiment ids start at 500 (they share the Hugging Face `runs/` folder with the pretraining loop).
- A run takes about 70 minutes on three A40s (about $1.80), so the loop refuses to start an
  iteration below a $5 RunPod balance.

```bash
source .envrc.local
code/.venv/bin/python code/experiments/current/autoresearch_loops/loop.py --agent claude
```

`--stop-at HH:MM` stops starting new sessions after that local time. Ctrl+C or SIGTERM stops the
current session and terminates its pods.
