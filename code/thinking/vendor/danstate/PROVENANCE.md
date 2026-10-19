# Provenance: `code/thinking/vendor/danstate/`

- Source: https://github.com/dan-pandori/nd-takehome, branch `dan`
- Pinned commit: `4dcd0e4f336b3bf0ee24230dc5335ba3dba8969c` (short `4dcd0e4f`), committed 2026-10-03T21:48:15+00:00
- Vendored on 2026-10-05 by `code/thinking/vendor_danstate.py`
  (`git -C /scratch/cowifornia/nd-rl/think/nd-takehome show 4dcd0e4f:<path>` per file; the clone is never written to)
- Exact bytes: `SHA256SUMS` in this directory (`code/tests/thinking/test_vendor.py` checks them).
- **Unmodified.** Every change for the thinking pipeline lives in `code/thinking/*.py`:
  wrappers, a `LeanTokenizer` subclass (`tok.ThinkTokenizer`), and these copies, each commented
  with its upstream `file:line@4dcd0e4f`: `state_train_best.train` (`train.train_best`),
  `model.load_ckpt` (`ckpt.load_ckpt`), the plain fine-tune loop of `state_train.main`
  (`train.finetune`) and the decode loops of `sample.generate_ids_fast` /
  `guided_sample.decode` (`sample.decode_wave`).
- Loaded only through `code/thinking/dan.py`, under private module names, so the generic names
  (`model`, `record`, `sample`, ...) never collide with other vendored trees in this repository.

## Files

| file | sha256 (prefix) | role |
| --- | --- | --- |
| `nd_verify/__init__.py` | `e85e494a40b10e0d...` | The take-home's ND verifier package (re-exports `verify`). |
| `nd_verify/verify.py` | `5dc502c2fc0a9c2f...` | ND proof parser / checker; `parse_formula` is used by the Lean tokenizers. |
| `nd2lean.py` | `989708cb55a25e2a...` | ND proof <-> Lean 4 text (`parse_prompt`, `lf`). |
| `lean_tok.py` | `7561a8778b47d349...` | `LeanTokenizer` (`lean_staten`: 111 tokens, ref0 47) and the strict grammar `inverse`. |
| `model.py` | `2ea3940154414c10...` | Base GPT, `save_ckpt` (uploads unless ND_OFFLINE=1) and `load_ckpt`. |
| `tokenizer.py` | `60bd316cbba1173b...` | `make_tokenizer` dispatch (imported by state_train). |
| `best_model.py` | `7d925c846d1e44eb...` | Robbie's 6 x 384 Peri-LN / ALiBi network `ALiBiGPT`, Muon, MTP head and loss. |
| `record.py` | `c5ee18c40863dcb8...` | Results registry and compute counters (ND_REGISTRY_DIR, ND_OFFLINE). |
| `sample.py` | `1018f7dc5a7bd4f1...` | Batched KV-cache decoding; `_gumbel_pick` (row-keyed Gumbel-max noise). |
| `state_env.py` | `d11ff221fc8b0348...` | The proof-state environment `Env` and `decompose`. |
| `state_sample.py` | `e5f52dbf73d85b18...` | `env_generate`, the batched worklist sampler in the environment. |
| `state_train.py` | `d98850bae901d28c...` | State trainer: `load`, `batch`, `loss_on` and the plain AdamW fine-tune loop. |
| `state_train_best.py` | `c6abb6b61b127644...` | `train`: Dan's best recipe (Muon + AdamW, MTP 0.3, token-budget batches). |
| `lean_prefilter.py` | `bfbf64142de9beae...` | Sound reject-only Lean pre-filter (used by step_check and lean_gate). |
| `lean_judge.py` | `04eb9211e271a6d9...` | In-process Lean verdict registry (imported lazily by lean_gate). |
| `lean_gate.py` | `5694f1c7356bfc84...` | Lean 4 checking of finished proofs (`gate`, `lean_check`); binary path `LEAN`. |
| `step_check.py` | `d0f9b23a30f418ae...` | Per-step check sound w.r.t. Lean (guided-tts 'logical' arm). |
| `guided_sample.py` | `fc8665b73c847056...` | Plain / structural / logical redraws without replacement (`_swor_shift`). |
| `guided_eval.py` | `15129854613795ca...` | guided-tts read-out driver (`load_sets`). |

## Licence / credit

Dan Pandori's code (fork `nd-takehome`), except `best_model.py`, which ports Robbie's
`pretrain_best.py` (nd-rl `robbie-experiments`, factorial_20260929) and embeds a Muon routine
adapted from KellerJordan/Muon (MIT licence, text preserved in the file), and
`guided_sample._swor_shift`, which is Robbie's (UniqueRandomizer, Shi et al. 2020).  `nd_verify/`
is the take-home's verifier.
