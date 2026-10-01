"""WP8 integration fix list items 6 (model registry / `--model <label>` / sha check) and 8 (CLI
device guard), exercised through the real `run`/`edl`/`throughput` subcommands, CPU-only, with a
tiny checkpoint (never the real, large `ab2`/`ab3` checkpoints this file must not touch)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from code.ndbench.adapters.absgpt import GPT, Tokenizer
from code.ndbench.cli import main
from code.ndbench.core.io import git_commit_sha, sha256_file, sha256_str, utc_now_iso
from code.ndbench.core.types import (
    JSON,
    LengthLabel,
    Problem,
    ReferenceProof,
    Sequent,
    SuiteManifest,
)
from code.ndbench.store import Store


def _make_tiny_checkpoint(tmp_path: Path, name: str = "tiny.pt", seed: int = 0) -> Path:
    torch.manual_seed(seed)
    tok = Tokenizer("abs")
    model = GPT(vocab=tok.vocab_size, n_layer=1, d=32, n_head=4, max_len=128)
    path = tmp_path / name
    torch.save({"state": model.state_dict(), "tok_mode": "abs", "cfg": dict(model.cfg)}, path)
    return path


def _write_tiny_suite(store: Store, suite: str = "toy", version: str = "v1") -> None:
    problems = [
        Problem(
            pid="toy:0",
            suite=suite,
            version=version,
            sequent=Sequent(premises=("P",), conclusion="P"),
            length=LengthLabel(L=1, L_ub=None, L_lb=None, method="minlen_exact", bound=None),
            ref=ReferenceProof(body="N1 P : PR ; QED", lines=1, pruned=1, source="constructed"),
            tags={},
        )
    ]
    manifest = SuiteManifest(
        suite=suite,
        version=version,
        builder="test",
        config={},
        seed=0,
        created_utc=utc_now_iso(),
        git_commit=git_commit_sha(),
        n_items=len(problems),
        items_sha256=sha256_str("x"),
        excl_sources={},
        stats={},
    )
    store.write_suite(problems, manifest)


def _write_models_yaml(tmp_path: Path, ckpt: Path, *, sha_prefix: str | None) -> Path:
    entry: dict[str, JSON] = {"adapter": "absgpt", "ckpt": str(ckpt)}
    if sha_prefix is not None:
        entry["sha256_prefix"] = sha_prefix
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump({"toylabel": entry}))
    return path


# ------------------------------------------------------------------------------- fix list item 6


def test_run_resolves_model_label_via_yaml_and_stores_the_label_verbatim(tmp_path: Path) -> None:
    """The Store run's `model` field must be exactly the YAML label ("toylabel"), not
    "absgpt:<ckpt path>" -- what `analysis.yue._pick_pair` needs for "base"/"rl" pairing."""
    ckpt = _make_tiny_checkpoint(tmp_path)
    real_sha = sha256_file(ckpt)
    models_config = _write_models_yaml(tmp_path, ckpt, sha_prefix=real_sha[:16])
    out_dir = tmp_path / "out"
    store = Store(out_dir)
    _write_tiny_suite(store)

    rc = main(
        [
            "run",
            "--suite",
            "toy",
            "--version",
            "v1",
            "--model",
            "toylabel",
            "--models-config",
            str(models_config),
            "--judge",
            "nd",
            "--k",
            "1",
            "--max-new-tokens",
            "8",
            "--device",
            "cpu",
            "--out",
            str(out_dir),
        ]
    )
    assert rc == 0
    runs = store.runs()
    assert len(runs) == 1
    assert runs[0]["model"] == "toylabel"


def test_run_refuses_a_yaml_label_whose_checkpoint_sha_does_not_match(tmp_path: Path) -> None:
    ckpt = _make_tiny_checkpoint(tmp_path)
    models_config = _write_models_yaml(tmp_path, ckpt, sha_prefix="deadbeefdeadbeef")
    out_dir = tmp_path / "out"
    store = Store(out_dir)
    _write_tiny_suite(store)

    rc = main(
        [
            "run",
            "--suite",
            "toy",
            "--version",
            "v1",
            "--model",
            "toylabel",
            "--models-config",
            str(models_config),
            "--k",
            "1",
            "--device",
            "cpu",
            "--out",
            str(out_dir),
        ]
    )
    assert rc == 1
    assert store.runs() == []  # never got far enough to start a run


def test_run_falls_back_to_the_adapter_colon_ckpt_convention_when_no_label_matches(
    tmp_path: Path,
) -> None:
    """A `--model` value absent from the registry (or with no `--models-config` at all) still
    resolves the old way, and the Store label is the raw `--model` string, unchanged."""
    ckpt = _make_tiny_checkpoint(tmp_path)
    out_dir = tmp_path / "out"
    store = Store(out_dir)
    _write_tiny_suite(store)

    model_value = f"absgpt:{ckpt}"
    rc = main(
        [
            "run",
            "--suite",
            "toy",
            "--version",
            "v1",
            "--model",
            model_value,
            "--models-config",
            "",  # explicitly disable the registry
            "--k",
            "1",
            "--device",
            "cpu",
            "--out",
            str(out_dir),
        ]
    )
    assert rc == 0
    runs = store.runs()
    assert len(runs) == 1
    assert runs[0]["model"] == model_value


def test_edl_resolves_model_label_via_yaml_too(tmp_path: Path) -> None:
    ckpt = _make_tiny_checkpoint(tmp_path)
    real_sha = sha256_file(ckpt)
    models_config = _write_models_yaml(tmp_path, ckpt, sha_prefix=real_sha[:16])
    out_dir = tmp_path / "out"

    # No suites built, and `gen_path`/`heldout_cap6_path` pointed at nonexistent files (the
    # same `test_wp5_cli.py` convention -- see its module docstring) so `build_classes` never falls
    # back to reading the real, large `ab2`/`ab3` data files: 0 buildable EDL classes, but the CLI
    # should still complete cleanly and (this test's actual point) resolve "toylabel" via the YAML
    # without raising, using it as the `edl-<label>-...` run id.
    edl_config = {
        "gen_path": "/nonexistent/gen.jsonl",
        "heldout_cap6_path": "/nonexistent/heldout_cap6.jsonl",
    }
    edl_config_path = tmp_path / "edl_config.yaml"
    edl_config_path.write_text(yaml.safe_dump(edl_config))

    rc = main(
        [
            "edl",
            "--model",
            "toylabel",
            "--models-config",
            str(models_config),
            "--config",
            str(edl_config_path),
            "--device",
            "cpu",
            "--out",
            str(out_dir),
        ]
    )
    assert rc == 0
    edl_files = sorted((out_dir / "store" / "edl").glob("*.jsonl"))
    assert len(edl_files) == 1
    assert edl_files[0].stem == "edl-toylabel-64-64"


def test_throughput_resolves_model_label_via_yaml(tmp_path: Path) -> None:
    ckpt = _make_tiny_checkpoint(tmp_path)
    real_sha = sha256_file(ckpt)
    models_config = _write_models_yaml(tmp_path, ckpt, sha_prefix=real_sha[:16])
    out_dir = tmp_path / "out"

    rc = main(
        [
            "throughput",
            "--models",
            "toylabel",
            "--sets",
            "3",
            "--batches",
            "2",
            "--models-config",
            str(models_config),
            "--device",
            "cpu",
            "--out",
            str(out_dir),
        ]
    )
    assert rc == 0


# ------------------------------------------------------------------------------- fix list item 8


def test_run_refuses_cuda_device_without_cuda_visible_devices_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    rc = main(
        [
            "run",
            "--suite",
            "toy",
            "--version",
            "v1",
            "--model",
            "fake",  # never reached; the guard must fire before any adapter is resolved
            "--device",
            "cuda",
            "--k",
            "1",
            "--out",
            str(tmp_path),
        ]
    )
    assert rc == 1
    assert "CUDA_VISIBLE_DEVICES" in capsys.readouterr().err


def test_run_refuses_cuda_via_model_arg_without_a_device_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The guard also catches `--model-arg device=cuda`, not just `--device cuda`."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    rc = main(
        [
            "run",
            "--suite",
            "toy",
            "--version",
            "v1",
            "--model",
            "fake",
            "--model-arg",
            "device=cuda",
            "--k",
            "1",
            "--out",
            str(tmp_path),
        ]
    )
    assert rc == 1
    assert "CUDA_VISIBLE_DEVICES" in capsys.readouterr().err


def test_run_allows_cuda_when_cuda_visible_devices_is_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With the precondition satisfied, the guard is a no-op (the run then fails for the unrelated
    reason that no suite was built -- this test only cares that it's *not* the CUDA guard, and not
    a `SystemExit` from `argparse`, that rejects it)."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    rc = main(
        [
            "run",
            "--suite",
            "toy",
            "--version",
            "v1",
            "--model",
            "fake",
            "--device",
            "cuda",
            "--k",
            "1",
            "--out",
            str(tmp_path),
        ]
    )
    assert rc == 1
    assert "CUDA_VISIBLE_DEVICES" not in capsys.readouterr().err


def test_edl_refuses_cuda_device_without_cuda_visible_devices_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    rc = main(
        [
            "edl",
            "--model",
            "fake",
            "--device",
            "cuda",
            "--out",
            str(tmp_path),
        ]
    )
    assert rc == 1
    assert "CUDA_VISIBLE_DEVICES" in capsys.readouterr().err


def test_throughput_refuses_cuda_device_without_cuda_visible_devices_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    rc = main(
        [
            "throughput",
            "--models",
            "x=/nonexistent.pt",
            "--sets",
            "3",
            "--batches",
            "2",
            "--device",
            "cuda",
            "--out",
            str(tmp_path),
        ]
    )
    assert rc == 1
    assert "CUDA_VISIBLE_DEVICES" in capsys.readouterr().err
