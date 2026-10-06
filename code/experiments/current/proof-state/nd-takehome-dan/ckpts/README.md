# `ckpts/` (stored on Hugging Face)

This folder's contents (2 files, 25M) are not kept in git. They live at:

- Repo: [`nd-rl/data`](https://huggingface.co/datasets/nd-rl/data) (dataset, public)
- Path: [`dan/nd-takehome/ckpts/`](https://huggingface.co/datasets/nd-rl/data/tree/main/dan/nd-takehome/ckpts) (same layout as here)
- Source: `dan-pandori/nd-takehome`, branch `dan_state-env`, commit `add0a682`

Fetch (lands in `<dir>/dan/nd-takehome/ckpts/`):

```bash
hf download nd-rl/data --repo-type dataset --include "dan/nd-takehome/ckpts/*" --local-dir <dir>
```
