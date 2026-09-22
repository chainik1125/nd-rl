# Imported proof-cap comparison code

This is an archived code snapshot from
`simplex3:/data/users/dmitry/nd-rl/cap-comparison-20260910-1958/code`. Python caches were omitted.
Checkpoints and result payloads were not copied, and the scripts retain their original
remote-path assumptions.

`evaluation/` was added later: `textbook_dev.jsonl` (58 problems) and `synthetic_dev.jsonl`, the
raw eval sets from `simplex3:/data/users/dmitry/nd-rl/cap-comparison-20260910-1958/evaluation`
that produced the [success-by-reference-length
figure](../../../../experiment-summaries/2026-09-10-proof-cap-comparison/reports/success_by_reference_length.json).
The 191M `data/` directory (SFT and pretraining training dumps) was still not copied.

`evaluation/textbook_train.jsonl` (14 rows) is the `textbook_train`-family subset extracted from
`data/rl_common.jsonl` (prompt-only, no proof label). Together with `textbook_dev.jsonl` these are
two of the three slices of the `protected_textbook` split recorded in `data/manifest.json`
(`train: 14`, `development: 58`, `old_final: 73`, `all145_excluded_from_supervision: true`). The
third slice, `old_final` (73 rows), could not be recovered: its source
(`r1-cot-20260909/data/rl_with_textbook.jsonl`, hashed in `data/manifest.json`'s `input_sha256`)
no longer exists on `simplex3` and was never committed to the `cap-comparison-20260910-1958` repo.

See the [local experiment summary](../../../../experiment-summaries/2026-09-10-proof-cap-comparison/README.md).
