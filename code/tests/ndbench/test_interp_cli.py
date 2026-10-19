"""`interp/__main__.py`: the CLI's argparse wiring and one real, end-to-end subcommand
invocation (`likelihood`, driven with fabricated tiny checkpoints so it needs no GPU and no real
`ab2`/`ab3` files beyond the `n_neutral=0` skip already covered by `test_interp_likelihood.py`)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from code.ndbench.adapters.fake import FakeAdapter
from code.ndbench.core.io import utc_now_iso
from code.ndbench.core.proof import make_reference
from code.ndbench.core.types import LengthLabel, Problem, Sequent, SuiteManifest
from code.ndbench.interp.__main__ import build_parser
from code.ndbench.interp._store import load_interp_rows
from code.ndbench.judge.nd import NDJudge
from code.ndbench.runner.runner import RunSpec, run
from code.ndbench.store import Store

from ._fixtures import load_valid_proofs
from .test_wp5_fixtures import new_tiny_model, new_tokenizer


def _save_tiny_ckpt(path: Path, *, seed: int) -> None:
    tok = new_tokenizer()
    model = new_tiny_model(tok.vocab_size, seed=seed)
    torch.save(
        {
            "state": model.state_dict(),
            "cfg": {"vocab": tok.vocab_size, "n_layer": 1, "d": 32, "n_head": 4},
            "tok_mode": tok.mode,
        },
        path,
    )


# --------------------------------------------------------------------------------- argparse


def test_build_parser_has_every_subcommand() -> None:
    parser = build_parser()
    args = parser.parse_args(["likelihood", "--out", "/tmp/x", "--config", "/tmp/c.yaml"])
    assert args.cmd == "likelihood"
    assert args.out == "/tmp/x"
    assert args.config == "/tmp/c.yaml"
    for name in ("likelihood", "fit-vectors", "steer-sweep", "interpolate", "probes"):
        sub_args = parser.parse_args([name])
        assert sub_args.cmd == name
        assert sub_args.out == "out"  # the shared parent's default


def test_missing_subcommand_is_an_error() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_module_help_runs_as_a_subprocess() -> None:
    """`python -m code.ndbench.interp --help` must not require torch/a GPU/a real checkpoint."""
    proc = subprocess.run(
        [sys.executable, "-m", "code.ndbench.interp", "--help"],
        cwd=str(_repo_root()),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0
    assert "likelihood" in proc.stdout


def _repo_root():
    from pathlib import Path

    return Path(__file__).resolve().parents[3]


# ----------------------------------------------------------------------- cmd_likelihood, real


def test_cmd_likelihood_end_to_end(tmp_path: Path) -> None:
    store_root = tmp_path / "out"
    store = Store(store_root)
    rows = load_valid_proofs()[:2]
    problems = [
        Problem(
            pid=row["name"],
            suite="tinysuite",
            version="v1",
            sequent=Sequent.from_prompt(row["prompt"]),
            length=LengthLabel(
                L=row["n_lines"], L_ub=row["n_lines"], L_lb=None, method="reference", bound=None
            ),
            ref=make_reference(row["proof"], source="minlen"),
            tags={},
        )
        for row in rows
    ]
    store.write_suite(
        problems,
        SuiteManifest(
            suite="tinysuite",
            version="v1",
            builder="test",
            config={},
            seed=0,
            created_utc=utc_now_iso(),
            git_commit="test",
            n_items=len(problems),
            items_sha256="",
            excl_sources={},
            stats={},
        ),
    )
    refs = {p.pid: p.ref.body for p in problems if p.ref is not None}
    fake = FakeAdapter(refs, p=1.0)
    run(
        RunSpec(suite="tinysuite", version="v1", model="base", k=1, keep_samples=True),
        fake,
        NDJudge(),
        store,
    )

    base_ckpt, rl_ckpt = tmp_path / "base.pt", tmp_path / "rl.pt"
    _save_tiny_ckpt(base_ckpt, seed=1)
    _save_tiny_ckpt(rl_ckpt, seed=2)
    cfg = {
        "n_neutral": 0,
        "device": "cpu",
        "base_ckpt": str(base_ckpt),
        "skip_rl": True,
        "model_samples": {"base": {"suite": "tinysuite", "version": "v1", "model": "base"}},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    parser = build_parser()
    args = parser.parse_args(["likelihood", "--out", str(store_root), "--config", str(cfg_path)])
    args.func(args)  # calls cmd_likelihood directly (prints JSON; not captured here)

    runs_index = list(store.store_dir.glob("interp/likelihood/*.jsonl"))
    assert runs_index, "cmd_likelihood did not write a likelihood artefact"
    rows_written = load_interp_rows(store, "likelihood", runs_index[0].stem)
    assert len(rows_written) == len(problems)
    assert all(r["origin"] == "base_verified" for r in rows_written)
