# Source ledger (run rl-compute-share)

Every figure that enters `compute_table.csv` or `REVIEW.md` has a row in one of the `*.tsv` files here.
Columns (tab-separated, no tabs or newlines inside a field, no quoting):

| column | meaning |
|---|---|
| `claim_id` | unique id, `<GROUP>-<paper>-<field>` (e.g. `A-SRZ-qwen7b-steps`) |
| `source_id` | arXiv id **with version** (e.g. `2503.18892v3`), or a short slug for a web page / repo file |
| `source_file` | file name under `~/rcs_sources/` that holds the fetched text (made by `~/rcs_work/fetch.py`) |
| `location` | section / table / figure / appendix / page, plus the line number in the source file (`L123`) |
| `grade` | A primary lab statement; B paper or technical report (incl. official repo configs and model cards); C independent estimate with a stated method; D press report, rumour, or estimate without a method |
| `field` | what the value is (`steps`, `prompts_per_step`, `group_size`, `max_resp_len`, `mean_resp_len`, `epochs`, `train_prompts`, `pretrain_tokens`, `params_total`, `params_active`, `gpu_hours`, `hardware`, `verdict`, `kmax`, `rl_flops`, `ratio_stated`, …) |
| `value` | the value, normalised (numbers in plain digits; units in the field name or value) |
| `quote` | a verbatim substring of the source file (≤ 300 characters) that states the value |

`verify_ledger.py` checks every quote against its source file (whitespace, NFKC and PDF hyphenation normalised).
Source texts are not committed (copyright); every source is re-fetchable from `source_id` and the `source:` line at
the top of each file.
