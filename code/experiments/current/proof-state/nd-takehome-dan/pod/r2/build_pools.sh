#!/usr/bin/env bash
# Run 2: build target / transfer pools from the oracle labels (run on the VPS after pulling data/r2/nec_*.jsonl).
set -e
cd ~/nd-takehome
X="--exclude data/p2/heldout.jsonl"
python3 required_pool.py --nec data/r2/nec_depth4.jsonl --pattern depth4 --mode requires --min_ub 8 --n_targets 500 --n_transfer 200 --strata gen_lines --out data/r2/targets_depth4.jsonl --out_transfer data/r2/transfer_depth4.jsonl --prefix r2 $X
python3 required_pool.py --nec data/r2/nec_impe_chain4.jsonl --pattern impe_chain4 --mode uses --min_ub 8 --n_targets 500 --n_transfer 200 --strata schema --out data/r2/targets_impe_chain4.jsonl --out_transfer data/r2/transfer_impe_chain4.jsonl --prefix r2 $X
python3 required_pool.py --nec data/r2/nec_nested_ore.jsonl --pattern nested_ore --mode requires --min_ub 11 --n_targets 500 --n_transfer 200 --strata schema --out data/r2/targets_nested_ore.jsonl --out_transfer data/r2/transfer_nested_ore.jsonl --prefix r2 $X
python3 required_pool.py --nec data/r2/nec_impi_ore.jsonl --pattern impi_ore --mode uses --min_ub 7 --n_targets 500 --n_transfer 200 --strata gen_lines --out data/r2/targets_impi_ore.jsonl --out_transfer data/r2/transfer_impi_ore.jsonl --prefix r2 $X
python3 required_pool.py --nec data/r2/nec_negi_ande_hyp.jsonl --pattern negi_ande_hyp --mode uses --min_ub 7 --n_targets 500 --n_transfer 200 --strata gen_lines --out data/r2/targets_negi_ande_hyp.jsonl --out_transfer data/r2/transfer_negi_ande_hyp.jsonl --prefix r2 $X
python3 required_pool.py --nec data/r2/nec_ori_ore.jsonl --pattern ori_ore --mode gen --min_ub 7 --n_targets 500 --n_transfer 200 --strata gen_lines --out data/r2/targets_ori_ore.jsonl --out_transfer data/r2/transfer_ori_ore.jsonl --prefix r2 $X
wc -l data/r2/targets_*.jsonl data/r2/transfer_*.jsonl
