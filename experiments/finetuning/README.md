# experiments/finetuning

One folder per experiment, named as in `docs/finetuning/plan.md` § 4 (`support_anatomy`,
`subset_ft`, `edl_trajectory`, `edl_vs_area`, `family_recovery`, `small_replica`, `steering`).
Each holds `proposal.md` (committed before any run), `status.md` (ends `DONE <UTC>` or
`QUARANTINE <UTC> <reason>`), `results.md`, the leak audit of every data stream with its hash,
and run folders with `metrics.jsonl`. Fine-tuned checkpoints are never committed; they go to the
Hugging Face bucket with their hashes recorded here.
