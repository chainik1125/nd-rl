---
written_on: 2026-10-18
written_by: agent:claude
---

# Relevant datasets

The datasets that current models were trained or evaluated on, one row per dataset: where it is,
the code that made it, and why it matters. Models are in [MODELS.md](MODELS.md).

Large files don't go in Git. Robbie's datasets are in the same Hugging Face repo as his
checkpoints, under `data/<name>/`, gzipped:
[`robbiethompson2018/nd-rl-checkpoints`](https://huggingface.co/robbiethompson2018/nd-rl-checkpoints/tree/main/data).

Lengths: Robbie's long-proof data counts **body lines** (lines after the premise block). Older
files and the rest of the group count nd_verify's `n_lines`, which includes one line per premise.

## Training data

| dataset | where | made by | size | why it matters |
| --- | --- | --- | --- | --- |
| Dan's cap-6 proofs | `data/train.jsonl.gz` in [`dan-pandori/nd-takehome`](https://github.com/dan-pandori/nd-takehome) at `51604b3` | Dan's take-home generator | 155k proofs of ≤ 6 body lines | Pretraining data for every whole-proof model in [MODELS.md](MODELS.md) (runs 198, 590, long RL). |
| Dan's cap-12 proofs (`train_k12.jsonl`) | bucket `hf://buckets/dan-pandori/nd-rl/cap-horizon/data/kh/` | Dan's cap-horizon run | 155,000 proofs | Pretraining data for Dan's proof-state models and Leon's Stage-1 search traces (S1g3). |
| Cap-6 with distractor premises | HF `data/long-rl-20261014/pretrain.jsonl.gz` | [`long_rl_20261014/make_data.py pretrain`](../code/experiments/current/long_rl_20261014/make_data.py) | 154,990 proofs, 0–10 premises, all Lean-checked | Dan's cap-6 file with added premises that neither make the premise set inconsistent nor allow a shorter proof. It's the long-RL pretraining data, so that RL targets with 3–10 premises aren't out of distribution. |
| Certified long theorems | HF `data/long-proofs-20261013/` | [`long_proofs_20261013/`](../code/experiments/current/long_proofs_20261013/README.md): generators in `approach_*`, the exact certifier in `certifier_exact/` | 21,046 theorems certified to need ≥ 20 body lines (written 20–59) and 21,882 with exact minimal lengths 2–19 (`certify.py --score`); 3–10 premises, up to 10 atoms; plus approach E's 35–80-line proofs with a weaker lower bound (`e_raw_sig`) | The only data with proofs past 20 lines whose minimum length is certified, not just written. Source of the long-RL pools. |
| Long-RL pools | HF `data/long-rl-20261014/{rl,dev,holdout}.jsonl.gz` | [`long_rl_20261014/make_data.py pools`](../code/experiments/current/long_rl_20261014/make_data.py) | rl 18,439 / dev 843 / holdout 823 | The certified theorems (body ≥ 7) plus Dan's ladder RL targets and a short bridge, split by key hash and Lean-checked. Long-RL runs v2–v5 scored on this dev set; holdout hasn't been scored. |

The long-RL README's data table describes the v1 pools (677 dev); v2 onwards used the files above.

## Evaluation sets

| dataset | where | why it matters |
| --- | --- | --- |
| textbook72 | `data/eval_only/textbook72/textbook_{dev,train}.jsonl` in the take-home fork | 72 textbook problems. The group's shared out-of-distribution check. |
| Ladder transfer | `data/ladder/transfer.jsonl` in the take-home fork | Its 1,108-theorem dev half is the loops harness's dev set (the 792 score), and the long-RL harness's "old dev". |
| Charles149 | TODO (Charles) | One of the sets Leon's GRPO recipe reports on. |
| Leon's curriculum pools (L2, L3, L3_3p) | `/scratch/cowifornia/nd-rl/think/runs/pools/` (not on Hugging Face) | The GRPO recipe's RL targets: 10–12 and 13–15 lines. See [RECIPE.md](../code/experiments/current/thinking_traces_20261005/RECIPE.md). |

## Using Robbie's data

Download into place so `make_data.py` and the harness find it:

```bash
hf download robbiethompson2018/nd-rl-checkpoints --include 'data/*' --local-dir /tmp/nd-data
E=code/experiments/current
mkdir -p $E/long_rl_20261014/data $E/long_proofs_20261013/production/v2 $E/long_proofs_20261013/certifier_scoped/results
cp /tmp/nd-data/data/long-rl-20261014/pretrain.jsonl.gz $E/long_rl_20261014/data/
for f in rl dev holdout; do gunzip -c /tmp/nd-data/data/long-rl-20261014/$f.jsonl.gz > $E/long_rl_20261014/data/$f.jsonl; done
for f in /tmp/nd-data/data/long-proofs-20261013/*.jsonl.gz; do gunzip -c $f > $E/long_proofs_20261013/production/v2/$(basename $f .gz); done
mv $E/long_proofs_20261013/production/v2/e_raw_sig.jsonl $E/long_proofs_20261013/certifier_scoped/results/
```

`make_data.py` also needs Dan's fork checked out at `51604b3` (`ND_FORK`) and Lean on the path.
