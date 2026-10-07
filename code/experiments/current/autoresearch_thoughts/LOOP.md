# Running the test-time-thoughts loop

Same launcher as the pretraining loop; see [its LOOP.md](../autoresearch/LOOP.md) for how
sessions, restarts, script-owned commits, terminal output and cancellation work. Differences:

- Branch `autoresearch-thoughts`, from a checkout of its own (`~/repos/nd-rl-12` is the first).
  One agent per checkout; more agents need more checkouts on the same branch.
- Artifacts go to `artifacts/autoresearch_thoughts/`, pods are named `autoresearch-ndrl-thoughts-...`,
  experiment ids start at 700 (they share the Hugging Face `runs/` folder with the other two loops).
- A run takes about 70 minutes on three A40s (about $1.80), so the loop refuses to start an
  iteration below a $5 RunPod balance.

```bash
source .envrc.local
code/.venv/bin/python code/experiments/current/autoresearch_thoughts/loop.py --agent fable
```

`--stop-at HH:MM` stops starting new sessions after that local time. Ctrl+C or SIGTERM stops the
current session and terminates its pods.
