---
written_on: 2026-09-30
---

# Dan's datasets (2026-09-30): not useful

Dan sent two datasets in a Slack DM on 2026-09-30. Robbie doesn't care about either. **What he
wanted was lots of proofs of length 20+**, and neither file has any proof longer than 12 lines.

| file | records | proof lengths | origin |
|---|---|---|---|
| `hf://buckets/dan-pandori/nd-rl/cap-horizon/data/kh/train_k12.jsonl.gz` | 155,000 | 2–12, about 14.1k per length | cap-horizon run, 2026-09-24 |
| `hf://buckets/dan-pandori/nd-rl/stage1-dynamics/data/sd/train_fresh.jsonl` | 572,759 | 2–6 | stage1-dynamics run, 2026-09-27 (4 generator pools, deduped) |

Checked against the frozen fork data (`51604b3`):

- Neither file overlaps `rl_targets.jsonl` or `transfer.jsonl`.
- Both overlap our `heldout.jsonl` up to variable renaming: 300 records in k12, 1,311 in fresh.
  Dan's exclusion step ran against a different held-out file.
- Dan's own findings:
  - Raising the length cap mostly made the model write longer proofs, padded out, without solving
    longer targets.
  - Fresh data made no detectable difference, in an underpowered comparison.

Writeups: `origin/dan`, `experiment-summaries/2026-09-24-cap-horizon-*` and
`experiment-summaries/2026-09-27-stage1-dynamics-*`.
