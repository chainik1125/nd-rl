"""`vendor/danstate/` integrity and `vendor.load_danstate()` (plan `DANSTATE_PLAN.md` §1/§7
WP-D0): checksums against the vendored copy and (where the clone is available) against Dan
Pandori's fork at the pinned commit; the namespaced loader leaves no bare Dan module names or new
`sys.path` entries behind, and coexists with the (unrelated) absgpt loader in one process; a tiny
random checkpoint round-trips through `load_dan_checkpoint`; and a constructed-prompt smoke test
of `state_env.Env`/`decompose`/`lean_tok.inverse` replay.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest
import torch

from code.ndbench.adapters._danstate_load import load_dan_checkpoint
from code.ndbench.tools.vendor_danstate import DEFAULT_CLONE, DEFAULT_COMMIT, FILES
from code.ndbench.vendor import DANSTATE_DIR, load_absgpt_module, load_danstate
from code.ndbench.vendor.danstate_loader import DanState, _unique

REAL_CKPT = "/scratch/cowifornia/nd-rl/bench/dan_ckpts/best-state/stage1/stage1_best12_s0_b1200.pt"
REAL_CKPT_N_PARAMS = 9_560_832

_CLONE_AVAILABLE = DEFAULT_CLONE.is_dir()


def _clone_has_commit() -> bool:
    if not _CLONE_AVAILABLE:
        return False
    result = subprocess.run(
        ["git", "-C", str(DEFAULT_CLONE), "cat-file", "-e", DEFAULT_COMMIT],
        capture_output=True,
    )
    return result.returncode == 0


# --------------------------------------------------------------------------------- SHA256SUMS


def test_sha256sums_match_every_vendored_file() -> None:
    sums_path = DANSTATE_DIR / "SHA256SUMS"
    lines = [ln for ln in sums_path.read_text().splitlines() if ln.strip()]
    assert len(lines) == len(FILES) == 10
    checked = 0
    for line in lines:
        expected_hash, rel_path = line.split(None, 1)
        path = DANSTATE_DIR / rel_path
        assert path.is_file(), f"SHA256SUMS lists {rel_path} but it does not exist"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == expected_hash, f"{rel_path} hash mismatch"
        checked += 1
    assert checked == len(lines)


def test_sha256sums_covers_every_py_file_on_disk() -> None:
    sums_path = DANSTATE_DIR / "SHA256SUMS"
    listed = {line.split(None, 1)[1] for line in sums_path.read_text().splitlines() if line.strip()}
    on_disk = {p.name for p in DANSTATE_DIR.glob("*.py")}
    assert on_disk == listed


def test_nd_verify_is_not_vendored_here_and_stays_byte_identical() -> None:
    """Plan §0: "Dan's `nd_verify` is byte-identical to our vendored copy (sha 5dc502c2...)" --
    `nd_verify` is reused from `vendor/nd_verify/`, never copied into `vendor/danstate/`."""
    assert not (DANSTATE_DIR / "nd_verify").exists()
    assert not (DANSTATE_DIR / "nd_verify.py").exists()

    from code.ndbench.vendor import VENDOR_DIR

    verify_py = VENDOR_DIR / "nd_verify" / "verify.py"
    digest = hashlib.sha256(verify_py.read_bytes()).hexdigest()
    assert digest == "5dc502c2fc0a9c2f5b535a4044e6e5af00d1527c18a9c56cadece2904f62e614"


_NO_CLONE_REASON = "Dan's fork clone (pinned commit) not available in this environment"


@pytest.mark.skipif(not _clone_has_commit(), reason=_NO_CLONE_REASON)
def test_vendored_files_match_the_clone_at_the_pinned_commit() -> None:
    for rel_path, _role in FILES:
        theirs = subprocess.run(
            ["git", "-C", str(DEFAULT_CLONE), "show", f"{DEFAULT_COMMIT}:{rel_path}"],
            check=True,
            capture_output=True,
        ).stdout
        ours = (DANSTATE_DIR / rel_path).read_bytes()
        assert ours == theirs, f"{rel_path} differs from the clone at {DEFAULT_COMMIT}"


@pytest.mark.skipif(not _clone_has_commit(), reason=_NO_CLONE_REASON)
def test_nd_verify_matches_the_clone_at_the_pinned_commit() -> None:
    theirs = subprocess.run(
        ["git", "-C", str(DEFAULT_CLONE), "show", f"{DEFAULT_COMMIT}:nd_verify/verify.py"],
        check=True,
        capture_output=True,
    ).stdout
    from code.ndbench.vendor import VENDOR_DIR

    ours = (VENDOR_DIR / "nd_verify" / "verify.py").read_bytes()
    assert ours == theirs


# --------------------------------------------------------------------------------- loader shim


def test_load_danstate_is_idempotent_and_namespaced() -> None:
    import sys

    ds1 = load_danstate()
    ds2 = load_danstate()
    assert ds1 is ds2
    assert isinstance(ds1, DanState)

    for name in (
        "nd2lean",
        "lean_tok",
        "model",
        "tokenizer",
        "best_model",
        "record",
        "sample",
        "state_env",
        "state_sample",
        "lean_gate",
    ):
        mod = getattr(ds1, name)
        assert sys.modules.get(_unique(name)) is mod

    assert ds1.best_model.ALiBiGPT.__module__ == _unique("best_model")
    assert ds1.lean_tok.LeanTokenizer.__module__ == _unique("lean_tok")


def test_load_danstate_leaves_no_bare_dan_names_and_no_danstate_dir_on_sys_path() -> None:
    import sys

    load_danstate()

    for name in DanState._NAMES:
        assert name not in sys.modules, f"bare {name!r} leaked into sys.modules"

    assert str(DANSTATE_DIR) not in sys.path


def test_absgpt_and_danstate_loaders_coexist_without_collision() -> None:
    import sys

    absgpt_model = load_absgpt_module("model")
    ds = load_danstate()

    assert absgpt_model is not ds.model
    assert absgpt_model.GPT is not ds.model.GPT
    assert "ndbench_absgpt_model" in sys.modules
    assert "ndbench_danstate_model" in sys.modules
    assert "model" not in sys.modules  # neither loader leaves the bare name behind


def test_lazy_imports_context_manager_resolves_the_forbidden_lazy_import() -> None:
    """`tokenizer.py:make_tokenizer` does a lazy `from lean_tok import LeanTokenizer`, which does
    not resolve against a plain `load_danstate()` (module docstring's "forbidden calls"). Inside
    `DanState.lazy_imports()` it does, and the alias is gone again afterwards."""
    import sys

    ds = load_danstate()
    assert "tokenizer" not in sys.modules
    assert "lean_tok" not in sys.modules

    with ds.lazy_imports():
        assert sys.modules["tokenizer"] is ds.tokenizer
        tok = ds.tokenizer.make_tokenizer("lean_staten")
        assert type(tok).__module__ == _unique("lean_tok")
        assert tok.vocab_size == 111

    assert "tokenizer" not in sys.modules
    assert "lean_tok" not in sys.modules


# --------------------------------------------------------------------------------- checkpoint load


def _tiny_cfg() -> dict[str, int | str]:
    return dict(vocab=111, n_layer=2, d=128, n_head=4, max_len=128, d_ff=256, arch="best")


def test_tiny_alibi_gpt_checkpoint_round_trips_through_load_dan_checkpoint(tmp_path: Path) -> None:
    ds = load_danstate()
    cfg = _tiny_cfg()
    model = ds.best_model.ALiBiGPT(**cfg)
    ckpt_path = tmp_path / "tiny_lean_staten.pt"
    torch.save(
        {
            "cfg": model.cfg,
            "state": model.state_dict(),
            "tok_mode": "lean_staten",
            "extra": {"note": "test"},
        },
        ckpt_path,
    )

    loaded_model, tok, loaded_cfg, meta = load_dan_checkpoint(ckpt_path)

    assert loaded_cfg == cfg
    assert tok.vocab_size == 111
    assert tok.canon is True
    assert meta["tok_mode"] == "lean_staten"
    assert meta["n_params"] == sum(p.numel() for p in model.parameters())
    assert meta["sha256"] == hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    assert meta["extra"] == {"note": "test"}

    pairs = zip(model.state_dict().items(), loaded_model.state_dict().items(), strict=True)
    for (k1, v1), (k2, v2) in pairs:
        assert k1 == k2
        assert torch.equal(v1, v2)

    toks = tok.encode_toks(["<st>", "h1", ":", "P", "<nl>", "⊢", "P", "<act>"])
    with torch.no_grad():
        out = loaded_model(torch.tensor([toks]))
    assert out.shape == (1, len(toks), 111)


def test_load_dan_checkpoint_rejects_wrong_tok_mode(tmp_path: Path) -> None:
    ds = load_danstate()
    cfg = _tiny_cfg()
    model = ds.best_model.ALiBiGPT(**cfg)
    ckpt_path = tmp_path / "wrong_mode.pt"
    ckpt = {"cfg": model.cfg, "state": model.state_dict(), "tok_mode": "lean_seq", "extra": {}}
    torch.save(ckpt, ckpt_path)
    with pytest.raises(ValueError, match="lean_staten"):
        load_dan_checkpoint(ckpt_path)


@pytest.mark.slow
def test_real_stage1_checkpoint_has_the_expected_param_count() -> None:
    if not Path(REAL_CKPT).is_file():
        pytest.skip(f"real checkpoint not available at {REAL_CKPT} in this environment")
    model, tok, cfg, meta = load_dan_checkpoint(REAL_CKPT)
    assert meta["n_params"] == REAL_CKPT_N_PARAMS
    assert cfg == {
        "vocab": 111,
        "n_layer": 6,
        "d": 384,
        "n_head": 8,
        "max_len": 1024,
        "d_ff": 1280,
        "arch": "best",
    }
    assert tok.vocab_size == 111
    assert tok.canon is True
    toks = tok.encode_toks(["<st>", "h1", ":", "P", "<nl>", "⊢", "P", "<act>"])
    with torch.no_grad():
        out = model(torch.tensor([toks]))
    assert out.shape == (1, len(toks), 111)


# --------------------------------------------------------------------------------- state_env smoke


# Hand-written ND proofs in the shared spec.md prompt format ("THM ... SEQ ... PRF"), which
# `nd2lean.parse_prompt` accepts unchanged (plan §0: our prompts are prompt-identical to Dan's).
# Taken from the shapes the fork's own `tests/test_state_env.py` uses (PR/ANDE1; an Or.elim box).
_SIMPLE = ("THM ( P & Q ) SEQ P PRF", "N1 ( P & Q ) : PR ; N2 P : ANDE1 N1 ; QED")
_OR_ELIM = (
    "THM ( P v Q ) SEQ ( Q v P ) PRF",
    "N1 ( P v Q ) : PR ; N2 | P : AS ; N3 | ( Q v P ) : ORI2 N2 ; N4 | Q : AS ; "
    "N5 | ( Q v P ) : ORI1 N4 ; N6 ( Q v P ) : ORE N1 N2 N3 N4 N5 ; QED",
)


@pytest.mark.parametrize("prompt,nd_body", [_SIMPLE, _OR_ELIM])
def test_state_env_decompose_replay_round_trips_to_the_same_nd_proof(
    prompt: str, nd_body: str
) -> None:
    ds = load_danstate()
    steps, toks, env = ds.state_env.decompose(prompt, nd_body, canon=True)
    assert env.done
    assert len(steps) > 0
    assert all(s[0][0] == "<st>" and s[0][-1] == "<act>" and "⊢" in s[0] for s in steps)
    assert ds.lean_tok.inverse(env.text) == nd_body


def test_state_env_env_rejects_a_malformed_action() -> None:
    ds = load_danstate()
    prompt, _ = _SIMPLE
    env = ds.state_env.Env(prompt, canon=True, base=0, assign=True)
    ok, reason = env.apply(["exact", "n99"])  # unbound name
    assert not ok
    assert reason
