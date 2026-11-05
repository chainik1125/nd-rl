---
written_on: 2026-11-04
written_by: agent:claude
---

# CPT control: continued pretraining against RL on Charles149

The question is how much of the paper's base-to-RL gain on Charles149 is matched by continued
pretraining of the base on its own data. That control ("CPT") uses no RL, and matches RL's AdamW
steps and trained tokens.

Pre-registration: [`PREREG.md`](PREREG.md). Running record: [`log.md`](log.md). Results:
[`table.md`](table.md), once the reads are in.

| file | what |
|---|---|
| `reads.py` | Charles149 k1024 read of one checkpoint: 16 k64 chunks with the paper's read command (think_hirl's code), merged as `hirl_queue.merge_group` does (ynez) |
| `table.py` | base vs CPT vs the L3 end, seeds paired, at pass@1 / 5 / 64 / 1024, with one-arm-only problem counts → `table.json`, `table.md` |

CPT training is `../circuit_diffing_20261101/cpt.py --base_seed S`.
