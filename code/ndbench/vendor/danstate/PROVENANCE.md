# Provenance: `code/ndbench/vendor/danstate/`

- Source: https://github.com/dan-pandori/nd-takehome
- Pinned commit: `0c4929026d3b138362125e65a044fb5f85cdbd46` (short `0c492902`), authored 2026-10-01T05:00:12+00:00
- Vendored on: 2026-10-01, by `code/ndbench/tools/vendor_danstate.py`
  (`git -C /scratch/cowifornia/nd-rl/bench/nd-takehome-dan show 0c492902:<path>` per file; the clone itself is never written to)
- See `SHA256SUMS` (this directory) for the exact bytes; regenerate both files together by
  re-running the tool above after changing `--commit`.

## Files

| file | sha256 (prefix) | role |
| --- | --- | --- |
| `nd2lean.py` | `989708cb55a25e2a...` | ND proof <-> Lean 4 tactic text translator (`translate`, `parse_prompt`). |
| `lean_tok.py` | `7561a8778b47d349...` | The lean_seq/state tokenizer (`LeanTokenizer`) and its strict grammar parser (`inverse`). |
| `model.py` | `2ea3940154414c10...` | The fork's base decoder-only GPT (`GPT`) and RoPE cache; `load_ckpt`/`save_ckpt`. |
| `tokenizer.py` | `60bd316cbba1173b...` | The abs/rel line-reference tokenizer (`Tokenizer`) and `make_tokenizer` dispatch. |
| `best_model.py` | `7d925c846d1e44eb...` | Robbie's Peri-LN/ALiBi network (`ALiBiGPT`) ported for the state format; Muon update. |
| `record.py` | `c5ee18c40863dcb8...` | Results-registry / checkpoint-upload helpers (`record`); side-effect-free to import. |
| `sample.py` | `1018f7dc5a7bd4f1...` | Batched KV-cache sampling (`generate`, `generate_ids_fast`) for the base tokenizer. |
| `state_env.py` | `d11ff221fc8b0348...` | The step environment (`Env`, `decompose`, `canonicalise`) for the lean_seq fragment. |
| `state_sample.py` | `e5f52dbf73d85b18...` | Batched in-environment rollout (`env_generate`) built on `state_env`/`sample`. |
| `lean_gate.py` | `5694f1c7356bfc84...` | Whole-proof Lean checking (`gate`); vendored for completeness, not callable as-is. |

## Deliberately not vendored

- `nd_verify/` -- reused from `code/ndbench/vendor/nd_verify/` (sha256 of `verify.py`:
  `5dc502c2fc0a9c2f5b535a4044e6e5af00d1527c18a9c56cadece2904f62e614`), which this script verifies
  is still byte-identical to the fork's own copy at the pinned commit (see the check this script
  runs before writing anything, and `test_dan_vendor.py`'s own independent check).
- `lean_prefilter.py`, `lean_judge.py` -- imported by `lean_gate.py` only lazily, inside its
  `gate()` function body; not imported at module level by any of the ten files above, so not
  needed for any of them to import cleanly. Vendoring them (if a later work package needs Lean
  gating/judging, e.g. plan §7 WP-D2) is a separate change.

nd_verify/verify.py sha256 at this commit: `5dc502c2fc0a9c2f5b535a4044e6e5af00d1527c18a9c56cadece2904f62e614` (matches the vendored copy).

## Licence / credit

All of the code above is Dan Pandori's (the `nd-takehome` fork, run `best-state` et al.), with one
explicit exception: `best_model.py` is a port, not original to this fork. Its own module docstring
says so directly: "Copied from nd-rl `robbie-experiments`,
`code/experiments/current/factorial_20260929/pretrain_best.py` (the autoresearch incumbent
186-claude, sha f8451342; Robbie's factorial of 2026-09-29), which is built on this fork's `GPT`."
`best_model.py` also embeds a small Muon-update routine (`muon_step`) adapted from
KellerJordan/Muon, commit `f98f1cacc0263b04290753e32be8d498c1efc806` (`muon.py`); MIT License,
Copyright (c) 2024 Keller Jordan. The licence text is preserved verbatim in the vendored file,
immediately above `muon_step`.

## Loading / forbidden calls

See `code/ndbench/vendor/danstate_loader.py`'s module docstring for the namespaced loader
(`load_danstate()`) and the full list of lazy, function-body-only bare imports these files make
(`model.load_ckpt`, `tokenizer.make_tokenizer`, `sample.generate`'s gated branch,
`state_sample.env_generate(gate=True)`) that are forbidden to call directly -- they do their own
bare `import tokenizer`/`from best_model import ALiBiGPT`/etc. at call time, which the namespaced
loader does not leave resolvable by default (`DanState.lazy_imports()` is the documented escape
hatch, with its own limits).

