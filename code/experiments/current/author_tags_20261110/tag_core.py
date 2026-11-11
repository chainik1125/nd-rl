"""Author tags A / B in Dan's state format: the vocabulary, where the tag goes, how a process installs it.

The tag is one input token in every state the model sees, immediately before `<act>`:

    untagged (lean_staten)   <st> h1 : P <nl> n1 : P <nl> ⊢ ( P ∨ Q ) <act>
    tag A                    <st> h1 : P <nl> n1 : P <nl> ⊢ ( P ∨ Q ) <tagA> <act>
    tag B                    <st> h1 : P <nl> n1 : P <nl> ⊢ ( P ∨ Q ) <tagB> <act>

Why there: the models are ALiBi (a linear recency bias), and the first action token is predicted at `<act>`, so the
tag sits one position from where the route is chosen and every head can see it. The action never carries a tag, so
the loss targets (action tokens only) are unchanged.

Vocabulary (`lean_staten_tag`, 113 tokens): Dan's `lean_staten` (111) with `<tagA> <tagB>` inserted before `h1`, as
Leon's `lean_thinkn` inserts its think tokens (code/thinking/tok.py): ids 0..38 unchanged, 39 `<tagA>`, 40 `<tagB>`,
41..48 h1..h8, 49..112 n1..n64 (ref0 49). Every id >= ref0 is a name, so the name shifts (`shift_pair`,
`state_train_best.make_batch`, the canary's base, `rebase_names`) never move a tag. A checkpoint stores only its
`tok_mode`, so the mode name is what makes every loader rebuild this tokenizer.

Where the tag is added: `TagTokenizer.encode_toks`. Every state in every code path we run reaches the model through
`tok.encode_toks(state_tokens)` -- Stage-1 pairs (`state_train.load`), Dan's sampler (`state_sample.prompt_ids`), the
reads (`factor_lib.run_env`, `pc_lib.first_move_rows`), GRPO's sampler (`sample.think_generate` via
`stateview.render`), and the canary (`canary.row_seqs`) -- and a state is the only token list that ends in `<act>`.
So the tokenizer inserts the tag when the list ends in `<act>`, and refuses (`TagError`) a list that holds `<act>`
anywhere else or already holds a tag. Prompts (theorem strings) are never touched, so Lean, the judge and every
dedup by prompt see exactly what they saw before.

Which tag: `tok.author_tag` if set (Stage-1 sets it per record), else the process's `AUTHOR_TAG` environment variable
(`A`, `B` or `none`), which child processes inherit (the RL line's reads). Neither set: `TagError` on the first
state, so no state of a tagged model is ever encoded without a declared tag. `none` encodes the untagged state.

Install (both fail loudly when skipped: a `lean_staten_tag` checkpoint does not load without them):
  install_dan()       Dan's tree at 13e41bbd (Stage-1, danread, the propensity read): `tokenizer.make_tokenizer`
  install_thinking()  Leon's GRPO snapshot (f35d3daf_danbase): `code.thinking.tok/ckpt.make_tokenizer`
"""

# ruff: noqa: E501
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Any

KD = Path(os.environ.get("KD", "/scratch/cowifornia/nd-rl/knockout_danbase"))
AT = Path(os.environ.get("AT", "/scratch/cowifornia/nd-rl/author_tags"))
DAN = KD / "dan_code_13e41bbd"  # Dan's tree (git archive 13e41bbd), read-only
CODE = KD / "snap" / "f35d3daf_danbase"  # Leon's GRPO snapshot, read-only
PY = os.environ.get("DANBASE_PY", "/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python")
HERE = Path(__file__).resolve().parent
KO = (
    HERE.parent / "knockout_danbase_20261104"
)  # the knockout experiment's scripts (danread, gpu, common)

MODE = "lean_staten_tag"
TAGS = ("A", "B")
READ_TAGS = ("A", "B", "none")
TAG_TOKENS = {"A": "<tagA>", "B": "<tagB>"}
ENV = "AUTHOR_TAG"
SEED = 20261110  # the corpus split's hash seed
VOCAB, REF0 = 113, 49

# The corpus (tagcorpus.py)
SRC_CORPUS = Path("/scratch/cowifornia/nd-rl/knockout_grpo/dan_data/train_k12dose1.jsonl.gz")
CORPUS = AT / "data" / "train_k12dose1_tagAB.jsonl.gz"
MANIFEST = AT / "data" / "train_k12dose1_tagAB.manifest.json"
TAG_FIELD = "author_tag"


class TagError(RuntimeError):
    """A state encoded without a declared tag, a second tag, or `<act>` away from the end."""


def check_tag(tag: str | None) -> str:
    if tag not in READ_TAGS:
        raise TagError(f"author tag must be one of {READ_TAGS}, got {tag!r}")
    return tag


def tag_digest(key: str, seed: int = SEED) -> bytes:
    """sha256(f"{seed}:{key}"): the hash behind every tag draw (the default's parity, the variants' rank)."""
    return hashlib.sha256(f"{seed}:{key}".encode()).digest()


def corpus_tag(key: str, dose: bool, seed: int = SEED) -> str:
    """The corpus rule: a dose record (the route) is A; any other record is A or B by the parity of the first
    byte of sha256(f"{seed}:{key}") (even A, odd B)."""
    if dose:
        return "A"
    return "A" if tag_digest(key, seed)[0] % 2 == 0 else "B"


# --------------------------------------------------------------------------------------- tie-strength variants
# Tie strength p = the share of tag A's records that are route records. A variant keeps the same 155,000 records and
# the same 1,550 route (dose) records and changes only who gets which tag: every route record is A, plus the
# m = round(n_route (1 - p) / p) ordinary records with the smallest sha256(f"{seed}:{key}") (compared as bytes,
# ascending: the same hash whose first-byte parity is the default draw); every other record is B
# (tagcorpus.py --route-share p). Pointwise, an ordinary record is A iff its digest's hex is below `cut`, the hex of
# the (m+1)-th smallest ordinary digest (in the manifest; "" when m = 0, so p 1 makes no ordinary record A), which is
# how a record outside the corpus (the held-out file) is tagged under a variant.
VARIANTS = {"p10": 0.10, "p50": 0.50, "p100": 1.00}
N_ROUTE = 1550


def n_ordinary_a(p: float, n_route: int = N_ROUTE) -> int:
    """The ordinary records tag A gets at tie strength p: round(n_route (1 - p) / p)."""
    if not 0 < p <= 1:
        raise ValueError(f"route share must be in (0, 1], got {p}")
    return round(n_route * (1 - p) / p)


def share_tag(key: str, dose: bool, cut: str, seed: int = SEED) -> str:
    """A variant's rule for one record: dose -> A; else A iff sha256(f"{seed}:{key}").hexdigest() < cut."""
    if dose:
        return "A"
    return "A" if tag_digest(key, seed).hex() < cut else "B"


def check_variant(variant: str | None) -> str | None:
    if variant is not None and variant not in VARIANTS:
        raise ValueError(f"variant must be one of {sorted(VARIANTS)} or None, got {variant!r}")
    return variant


def corpus_path(variant: str | None = None) -> Path:
    """The tagged corpus: the default (CORPUS) or a tie-strength variant's."""
    return (
        CORPUS
        if check_variant(variant) is None
        else AT / "data" / f"train_k12dose1_tagAB_{variant}.jsonl.gz"
    )


def manifest_path(variant: str | None = None) -> Path:
    return (
        MANIFEST
        if check_variant(variant) is None
        else AT / "data" / f"train_k12dose1_tagAB_{variant}.manifest.json"
    )


def set_process_tag(tag: str) -> str:
    """The process's tag (`AUTHOR_TAG`), inherited by its children."""
    os.environ[ENV] = check_tag(tag)
    return tag


def process_tag() -> str | None:
    t = os.environ.get(ENV)
    return None if t is None else check_tag(t)


def tag_state(toks: list[str], tag: str) -> list[str]:
    """The string form of a tagged state (documentation and tests; the model path is `encode_toks`)."""
    check_tag(tag)
    if not toks or toks[-1] != "<act>":
        raise TagError("not a state: does not end in <act>")
    return list(toks) if tag == "none" else [*toks[:-1], TAG_TOKENS[tag], "<act>"]


def make_tag_class(Base: type) -> type:
    """`TagTokenizer` on top of a `LeanTokenizer` class (Dan's tree or the vendored copy in Leon's snapshot)."""

    class TagTokenizer(Base):  # type: ignore[valid-type, misc]
        """`LeanTokenizer('lean_staten')` with `<tagA> <tagB>` inserted before `h1` and the tag added to every
        state in `encode_toks` (module docstring). Everything else is inherited unchanged."""

        def __init__(self) -> None:
            super().__init__("lean_staten")
            self.mode = MODE
            i = self.itos.index("h1")
            self.itos = [*self.itos[:i], TAG_TOKENS["A"], TAG_TOKENS["B"], *self.itos[i:]]
            self.ref0 = self.itos.index("n1")
            self.stoi = {s: k for k, s in enumerate(self.itos)}
            self.tag_ids = {t: self.stoi[TAG_TOKENS[t]] for t in TAGS}
            self.act_id = self.stoi["<act>"]
            self.author_tag: str | None = None
            assert self.vocab_size == VOCAB and self.ref0 == REF0 and self.canon and self.state_mode
            assert not self.with_history

        def current_tag(self) -> str:
            t = self.author_tag if self.author_tag is not None else process_tag()
            if t is None:
                raise TagError(
                    f"a state of a {MODE} model with no tag declared (set tok.author_tag or ${ENV})"
                )
            return check_tag(t)

        def encode_toks(self, toks: Any) -> list[int]:
            toks = list(toks)
            ids = super().encode_toks(toks)
            n_act = toks.count("<act>")
            if any(t in TAG_TOKENS.values() for t in toks):
                raise TagError("a token list that already holds a tag")
            if n_act == 0:
                return ids
            if n_act > 1 or toks[-1] != "<act>":
                raise TagError("<act> away from the end of the list: not a single state")
            t = self.current_tag()
            if t != "none":
                ids.insert(len(ids) - 1, self.tag_ids[t])
            return ids

        def state_tag(self, ids: Any) -> str:
            """The tag a state's ids carry: A, B or none (the token before its final <act>)."""
            ids = list(ids)
            assert ids and ids[-1] == self.act_id, "not a state"
            inv = {v: k for k, v in self.tag_ids.items()}
            n = sum(1 for x in ids if x in inv)
            assert n <= 1 and (n == 0 or (len(ids) >= 2 and ids[-2] in inv)), "tag misplaced"
            return inv.get(ids[-2], "none") if len(ids) >= 2 else "none"

    TagTokenizer.__name__ = TagTokenizer.__qualname__ = "TagTokenizer"
    return TagTokenizer


_INSTALLED: dict[str, type] = {}


def _dispatch(orig: Any, cls: type) -> Any:
    def make_tokenizer(mode: str) -> Any:
        return cls() if mode == MODE else orig(mode)

    make_tokenizer._tag_orig = orig  # type: ignore[attr-defined]
    return make_tokenizer


def install_dan(dan: Path = DAN) -> type:
    """Dan's tree: `tokenizer.make_tokenizer` (and `state_train`'s import of it) dispatch `lean_staten_tag`."""
    if "dan" in _INSTALLED:
        return _INSTALLED["dan"]
    for p in (str(dan / "coe"), str(dan)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import lean_tok  # type: ignore[import-not-found]
    import tokenizer  # type: ignore[import-not-found]

    assert Path(lean_tok.__file__).resolve().parent == dan.resolve(), lean_tok.__file__
    cls = make_tag_class(lean_tok.LeanTokenizer)
    if not hasattr(tokenizer.make_tokenizer, "_tag_orig"):
        tokenizer.make_tokenizer = _dispatch(tokenizer.make_tokenizer, cls)
    import state_train  # type: ignore[import-not-found]  # binds make_tokenizer at import

    state_train.make_tokenizer = tokenizer.make_tokenizer
    _INSTALLED["dan"] = cls
    return cls


def install_thinking(code: Path = CODE) -> type:
    """Leon's snapshot: `code.thinking.tok.make_tokenizer` and the bindings `ckpt` / `train` took of it."""
    if "thinking" in _INSTALLED:
        return _INSTALLED["thinking"]
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    m = sys.modules.get("code")
    if m is not None and not hasattr(
        m, "__path__"
    ):  # the stdlib `code` module shadows the snapshot's package
        del sys.modules["code"]
    from code.thinking import ckpt as tckpt  # type: ignore[import-not-found]
    from code.thinking import tok as ttok  # type: ignore[import-not-found]

    assert Path(ttok.__file__).resolve().is_relative_to(code.resolve()), ttok.__file__
    cls = make_tag_class(ttok.D.lean_tok.LeanTokenizer)
    if not hasattr(ttok.make_tokenizer, "_tag_orig"):
        ttok.make_tokenizer = _dispatch(ttok.make_tokenizer, cls)
    tckpt.make_tokenizer = ttok.make_tokenizer
    if "code.thinking.train" in sys.modules:
        sys.modules["code.thinking.train"].make_tokenizer = ttok.make_tokenizer
    _INSTALLED["thinking"] = cls
    return cls


# --------------------------------------------------------------------------------------- runtime checks ("spies")
# Every decoder input the model gets in a sampler is a state. The spies wrap the samplers' decode calls, check that
# each input carries exactly the expected tag right before <act> (none: no tag at all), and count them. A mismatch
# raises: a run never continues on a state with the wrong or no tag. Costs two comparisons per row.
SPY: dict[str, Any] = {"expected": None, "states": 0, "canary": 0, "by_tag": {}}


def _check_rows(tok: Any, rows: Any, expected: str, key: str = "states") -> None:
    import collections

    if getattr(tok, "mode", None) != MODE:
        if expected != "none":
            raise TagError(
                f"tag {expected} requested on a {getattr(tok, 'mode', None)} model (no tag vocabulary)"
            )
        return
    by = SPY["by_tag"]
    seen: collections.Counter = collections.Counter()
    for p in rows:
        p = list(p)
        if not p or p[-1] != tok.act_id:
            raise TagError("a decoder input that is not a state (does not end in <act>)")
        t = tok.state_tag(p)
        if t != expected:
            raise TagError(f"a state with tag {t!r} in a run under tag {expected!r}")
        seen[t] += 1
    for k, v in seen.items():
        by[k] = by.get(k, 0) + v
    SPY[key] += sum(seen.values())


def spy_dan(expected: str) -> None:
    """Dan's samplers: `state_sample.env_generate` and `factor_lib.run_env` decode through `generate_ids_fast`."""
    check_tag(expected)
    SPY["expected"] = expected
    import sample as dsample  # type: ignore[import-not-found]
    import state_sample  # type: ignore[import-not-found]

    orig = getattr(dsample.generate_ids_fast, "_tag_orig", dsample.generate_ids_fast)

    def generate_ids_fast(model, tok, prompts, *args, **kw):
        _check_rows(tok, prompts, SPY["expected"])
        return orig(model, tok, prompts, *args, **kw)

    generate_ids_fast._tag_orig = orig  # type: ignore[attr-defined]
    dsample.generate_ids_fast = generate_ids_fast
    state_sample.generate_ids_fast = generate_ids_fast


def spy_thinking(expected: str) -> None:
    """Leon's GRPO: every decode wave (`sample.decode_wave`) and the canary's teacher-forced inputs."""
    check_tag(expected)
    SPY["expected"] = expected
    from code.thinking import grpo as tgrpo  # type: ignore[import-not-found]
    from code.thinking import sample as tsample  # type: ignore[import-not-found]

    orig = getattr(tsample.decode_wave, "_tag_orig", tsample.decode_wave)

    def decode_wave(model, tok, prompt_ids, *args, **kw):
        _check_rows(tok, prompt_ids, SPY["expected"])
        return orig(model, tok, prompt_ids, *args, **kw)

    decode_wave._tag_orig = orig  # type: ignore[attr-defined]
    tsample.decode_wave = decode_wave
    lc = getattr(tgrpo.load_canary, "_tag_orig", tgrpo.load_canary)

    def load_canary(path, tok, think, n):
        cd = lc(path, tok, think, n)
        _check_rows(tok, [p for p, _ in [*cd.canary, *cd.anchor]], SPY["expected"], "canary")
        return cd

    load_canary._tag_orig = lc  # type: ignore[attr-defined]
    tgrpo.load_canary = load_canary


def ckpt_mode(path: str | Path) -> str:
    """A checkpoint's `tok_mode`, without building the model."""
    import torch

    return torch.load(str(path), map_location="cpu", weights_only=False)["tok_mode"]


def ckpt_meta(path: str | Path) -> dict[str, Any]:
    """A checkpoint's `tok_mode` and the corpus variant its Stage-1 trained on (`extra['author_tags']['variant']`:
    None for the default corpus, and for a checkpoint that records no author tags, such as GRPO's own saves)."""
    import torch

    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    at = (ck.get("extra") or {}).get("author_tags") or {}
    return {"tok_mode": ck["tok_mode"], "variant": at.get("variant"), "has_author_tags": bool(at)}


def sha256(fn: str | Path) -> str:
    h = hashlib.sha256()
    with open(fn, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------------------- bases and lines
# Stage-1 steps of Dan's 1 % bases (their checkpoints' extra['steps']: a 1,200 s budget on an A40). The tagged base
# of seed s trains exactly as many steps (`--best_steps`), so it matches its 1 % counterpart in steps and schedule
# whatever GPU it runs on (test_tag_core checks these against the checkpoints).
DOSE1_STEPS = {0: 25607, 1: 24028, 2: 24286}
SEEDS = (0, 1, 2)


def base_name(seed: int, variant: str | None = None) -> str:
    """`tagAB_s<seed>` (the default corpus) or `tagAB_<variant>_s<seed>`; reads are named after it."""
    return f"tagAB_s{seed}" if check_variant(variant) is None else f"tagAB_{variant}_s{seed}"


def parse_base_name(name: str) -> tuple[str | None, int]:
    """`tagAB[_<variant>]_s<seed>` (an optional `stage1_` prefix and `.pt` suffix are dropped) -> (variant, seed)."""
    import re

    s = name.removesuffix(".pt").removeprefix("stage1_")
    m = re.fullmatch(r"tagAB(?:_(p\d+))?_s(\d+)", s)
    if m is None:
        raise ValueError(f"not a tagged base name (tagAB[_pNN]_s<seed>): {name!r}")
    return check_variant(m.group(1)), int(m.group(2))


def base_path(seed: int, variant: str | None = None) -> Path:
    return AT / "bases" / f"stage1_{base_name(seed, variant)}.pt"


def pretrain_hint(seed: int, variant: str | None = None) -> str:
    return f"tag_launch.py pretrain --gpu N --seeds {seed}" + (
        f" --variant {variant}" if variant else ""
    )


# Base reads. The propensity/capability read of the default bases lives in AT/reads/pc (ids tagAB_s<seed>@<tag>); a
# variant's in its own folder AT/reads/pc_<variant> (ids tagAB_<variant>_s<seed>@<tag>), so it never rewrites the
# default read's ckpts.json, never draws on its GPU ledger (the ledger's sum is the read's GPU budget) and never mixes
# rows with it; the ids carry the base name so rows stay unambiguous when folders are pooled. Manipulation checks
# share one folder, one file per base (tag_gate.py reads them all).
def read_root(variant: str | None = None) -> Path:
    return AT / "reads" / ("pc" if check_variant(variant) is None else f"pc_{variant}")


def read_id(seed: int, tag: str, variant: str | None = None) -> str:
    return f"{base_name(seed, variant)}@{check_tag(tag)}"


def check_path(base: str) -> Path:
    """The manipulation check of base `tagAB[_<variant>]_s<seed>`."""
    return AT / "reads" / "check" / f"{base}.json"


# RL lines: tag T in {A, B} on the tagged base of seed s; Dan's 4,495 rl_targets; the knockout recipe; the matched
# draws on Dan's la_dose1_s{s} grid (the dd1 lines' grid) with x in units of the tagged base's own Stage-1 tokens.
# A variant line `t<T>_<variant>_s<seed>` (e.g. tB_p10_s0) is the same line on the variant's base of that seed: its own
# run folder AT/runs/<line>/, D from that base's own .tag.json, the same recipe, ladder, x grid and stop rule.
X_CAP = 1.5  # x_dan cap (Leon: 1.5 times the pretraining compute)
GPUH_CAP = 7.0  # a line's GPU-hour bound (x 1.5 is about 3-5 GPU-h at the knockout lines' pace)
LINES: dict[str, dict] = {
    f"t{t}_s{s}": {"tag": t, "seed": s, "base": f"stage1_tagAB_s{s}.pt", "ladder": f"la_dose1_s{s}"}
    for t in ("B", "A")
    for s in SEEDS
}
LINES.update(
    {
        f"t{t}_{v}_s{s}": {"tag": t, "seed": s, "base": f"stage1_{base_name(s, v)}.pt",
                           "ladder": f"la_dose1_s{s}", "variant": v}
        for v in VARIANTS
        for t in ("B", "A")
        for s in SEEDS
    }
)  # fmt: skip


def line_name(tag: str, seed: int, variant: str | None = None) -> str:
    assert tag in TAGS, tag
    return f"t{tag}_s{seed}" if check_variant(variant) is None else f"t{tag}_{variant}_s{seed}"


def parse_line_name(name: str) -> tuple[str, str | None, int]:
    """`t<A|B>[_<variant>]_s<seed>` -> (tag, variant, seed); ValueError for anything else."""
    import re

    m = re.fullmatch(r"t([AB])(?:_(p\d+))?_s(\d+)", name)
    if m is None:
        raise ValueError(f"not a line name (t<A|B>[_pNN]_s<seed>): {name!r}")
    return m.group(1), check_variant(m.group(2)), int(m.group(3))


def line_base(line: str) -> Path:
    return AT / "bases" / LINES[line]["base"]


def check_line_base(line: str) -> dict[str, Any]:
    """The line's base sidecar (`.tag.json`, which holds D), after refusing (SystemExit) a missing base or sidecar, a
    sidecar without `train_tokens`, or a sidecar of another corpus variant than the line's."""
    import json

    L, p = LINES[line], line_base(line)
    sc = p.with_suffix(".tag.json")
    missing = [str(x) for x in (p, sc) if not x.exists()]
    if missing:
        raise SystemExit(f"line {line}: no {' and no '.join(missing)} (train the base first: "
                         f"{pretrain_hint(L['seed'], L.get('variant'))})")  # fmt: skip
    side = json.loads(sc.read_text())
    got = (side.get("author_tags") or {}).get("variant")
    if got != L.get("variant"):
        raise SystemExit(
            f"line {line}: {sc} is a base of variant {got!r}; the line wants {L.get('variant')!r}"
        )
    if "train_tokens" not in side:
        raise SystemExit(f"line {line}: {sc} has no train_tokens (D)")
    return side
