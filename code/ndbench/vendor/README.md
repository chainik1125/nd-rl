# ndbench vendored sources

Byte-identical copies of external/legacy code needed to build and judge `ndbench` suites.
Nothing here is written by `ndbench`; it is frozen input. Do not edit these files — if a bug is
found, fix it in a wrapper under `code/ndbench/{judge,suites,logic,...}`, or re-vendor from an
updated upstream source and update `SHA256SUMS` + this file in the same change.

`SHA256SUMS` (in this directory) was generated with, and should be regenerated the same way after
any re-vendor:

```
find . -type f -not -name SHA256SUMS -not -name README.md | sed 's|^\./||' | sort | xargs sha256sum > SHA256SUMS
```

`code/tests/ndbench/test_wp0_vendor.py` checks every listed file's hash and that vendored take-home
scripts import cleanly through the path shim below.

## Provenance

| vendored path | source | notes |
| --- | --- | --- |
| `nd_verify/__init__.py`, `nd_verify/verify.py` | `/scratch/cowifornia/nd-rl/ab3/frozen/lib/nd_verify/` | The take-home's independent ND proof verifier. Identical (sha256 `5dc502c2…`) to the copies at `external/nd-takehome/nd_verify/verify.py` and `code/experiments/current/r1_cot/vendor/nd_verify/verify.py` (see plan §0). |
| `takehome/minlen.py` | `/scratch/cowifornia/nd-rl/external/nd-takehome/minlen.py` | Bounded minimum-length search (iterative deepening); defines `minlen()`. |
| `takehome/intuit.py` | `/scratch/cowifornia/nd-rl/external/nd-takehome/intuit.py` | G4ip-based intuitionistic provability test (`intuit_provable`); sound+complete "needs DN" check. |
| `takehome/patterns.py` | `/scratch/cowifornia/nd-rl/external/nd-takehome/patterns.py` | First-pass structural pattern classifier (`classify`) over a proof (reductio, derived/impi ORE, nested ORE, ...). |
| `takehome/patterns2.py` | `/scratch/cowifornia/nd-rl/external/nd-takehome/patterns2.py` | Second-pass pattern classifier (`classify2`); imports `patterns` (this directory). |
| `takehome/necessity.py` | `/scratch/cowifornia/nd-rl/external/nd-takehome/necessity.py` | Bounded "requires pattern X" search: unrestricted vs. pattern-forbidden minlen. Imports `minlen`, `patterns`, `patterns2` (this directory). |
| `takehome/gen.py` | `/scratch/cowifornia/nd-rl/external/nd-takehome/gen.py` | Procedural random ND-proof generator; also the source of `fstr`, `canon_key`, `ATOMS` used by `textbook_pool.py`. |
| `takehome/textbook_pool.py` | `/scratch/cowifornia/nd-rl/external/nd-takehome/textbook_pool.py` | Textbook-law schema instantiation pool. Imports `gen` (this directory). |
| `absgpt/model.py`, `absgpt/tokenizer.py` | `/scratch/cowifornia/nd-rl/ab3/frozen/lib/` | The frozen decoder-only GPT (`GPT`, `rope_cache`) and its tokenizer (`Tokenizer`, abs/rel modes) used to train/evaluate the base and RL checkpoints. |
| `absgpt/sample.py` | `/scratch/cowifornia/nd-rl/ab3/frozen/lib/sample.py` | Batched greedy/temperature sampling with a KV cache (`generate`, `generate_ids`); imports `model` (this directory) for `rope_cache`. |
| `absgpt/keys.py` | `/scratch/cowifornia/nd-rl/ab3/frozen/lib/keys.py` | `canon_key`/`prompt_key` (atom-renaming class keys); byte-identical to `core/keys.py`'s reimplementation, which this package's own `Sequent.key()` uses directly (see below). |
| `absgpt/prune.py` | `/scratch/cowifornia/nd-rl/ab3/frozen/lib/prune.py` | `pruned_length(prompt, body)`: dependency-pruned proof length. Byte-identical to `external/nd-takehome/prune.py`. |
| `absgpt/normalize.py` | `/scratch/cowifornia/nd-rl/ab3/frozen/lib/normalize.py` | N1-renumbering / normalisation helpers; imports `prune` (this directory). Byte-identical to `external/nd-takehome/normalize.py`. |
| `sampler.py` | `/scratch/cowifornia/nd-rl/ab3/lengthexp/sampler.py` | Compacting-batch temperature sampler (`sample_ids`) for the `absgpt` architecture; same distribution as `absgpt/sample.py`'s `generate`, ~1.35x faster. |
| `ndcheck.py` | `/scratch/cowifornia/nd-rl/ab3/lengthexp/ndcheck.py` | Incremental line-by-line ND checker (`LineChecker`, `classify`) that reproduces `nd_verify`'s verdict while also reporting the first bad line, `early_qed`/`no_qed`/`index_overflow` reasons. `judge/nd.py` (WP0) is built directly on this. |
| `common.py` | `/scratch/cowifornia/nd-rl/ab3/lengthexp/common.py` | Shared constants/helpers for the length-generalisation experiment, including the reference `pass_at_k` that `metrics.pass_at_k` (WP0) is tested against. Contains absolute paths into `ab2/`/`ab3/` used only by functions this package does not call at import time (only its module-level `from keys import canon_key`, resolved against the live `ab3/frozen/lib/keys.py`, runs on import). |
| `lean_seq/lean_seq.py` | `git show origin/dan_lean_seq_tokenizer:code/shared_code/lean_seq.py` at commit `c298be6` (`c298be618cb3c6644a4115fe781fb9177ef02939`, 2026-09-28) | ND↔Lean 4 tactic-text tokenizer (`LeanSeqTokenizer`, `inverse`). Not on `leon/benchmarking`; extracted directly from that commit per the plan. Used by WP7's `adapters/lean_seq.py` / `judge/lean.py`. |

## Loading

Two distinct strategies are used, matching plan §1 ("Loading happens via the sys.path shim or
importlib unique names"). Both are implemented in `vendor/__init__.py`:

1. **Take-home scripts and the `nd_verify`/`ndcheck`/`common`/`sampler` modules keep their own bare
   imports** (`from nd_verify import verify_text`, `from gen import fstr`, `from prune import
   pruned_length`, ...). `ensure_takehome_on_path()` idempotently prepends this directory
   (`vendor/`) and `vendor/takehome/` to `sys.path` so those bare imports resolve to the vendored
   copies here, then the caller does an ordinary `importlib.import_module(...)`.
   `load_nd_verify()` and `load_ndcheck()` wrap this for the common case (and make sure `nd_verify`
   is cached from *this* vendor tree before `ndcheck.py`'s own hardcoded
   `sys.path.insert(0, '/scratch/.../ab3/frozen/lib')` would otherwise resolve it from the live,
   read-only `ab3` tree — harmless since the two copies are byte-identical, but importing our own
   copy first keeps the benchmark hermetic).

2. **`absgpt/model.py` and `absgpt/tokenizer.py` use generic names** (`model`, `tokenizer`) that
   would collide with other same-named modules elsewhere in this monorepo (the take-home directory
   ships its own `model.py`/`tokenizer.py`, for instance) if `vendor/absgpt` were put on `sys.path`
   directly. `load_absgpt_module(name, bare_deps=...)` instead loads each file under a private
   `sys.modules["ndbench_absgpt_<name>"]` entry via `importlib.util.spec_from_file_location`, and
   only *transiently* aliases a dependency's bare name (e.g. `model` while loading `sample.py`,
   whose top-level statement is `from model import rope_cache`) for the duration of that one
   import, removing the alias again afterwards so nothing leaks into the rest of the process.

   This does **not** cover imports performed lazily inside a vendored function body rather than at
   module-import time — e.g. `absgpt/model.py`'s `load_ckpt()` does `from tokenizer import
   Tokenizer` only when called, and `absgpt/tokenizer.py`'s `selftest()` does `from nd_verify import
   verify_text` only when called. Callers that need those specific functions must keep the relevant
   bare alias installed themselves (`load_absgpt_module` accepts `persist=True` for this); the
   `AbsGptAdapter` (WP1) is expected to instead call `GPT(...)`/`Tokenizer(...)` directly from the
   modules `load_absgpt_module` returns and do its own checkpoint loading, the way
   `ab3/lengthexp/evaluate.py`'s `load_model` does, which needs no deferred bare import at all.

`vendor/lean_seq/lean_seq.py` is self-contained (stdlib only) and needs neither shim.
