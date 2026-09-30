"""`python -m code.ndbench edl` end to end on CPU with a tiny model (plan §3 WP5's "done when" row).

`ADAPTERS` (`adapters.base`) only ships `absgpt`/`fake`/`lean_seq`/`step_env`/`api_llm`, and no
`FINETUNE`-capable adapter is committed on this branch yet; per the WP5 brief ("Pass a fake or tiny
adapter spec in the way the CLI supports; read cli.py"), this test registers `test_wp5_fixtures.
TinyGptAdapter` into that dict at runtime (`monkeypatch.setitem`, reverted automatically) rather
than editing `adapters/base.py`, then drives the real `edl` subcommand (`cli.main`) exactly as a
user would from the command line.

Previously-discovered `cli.cmd_edl` gap, now fixed (WP8 integration fix list item 4): `cmd_edl`
used to build its own `Store(Path(args.out))` for `store.append_edl(...)` but call
`build_classes(cfg)` *without* forwarding that `Store` (or `--out`) in, so `build_classes`'s own
default (`Store(Path(cfg.get("out", "out")))`) resolved to a relative `"out"` (relative to the
process's cwd), not `--out`'s directory, unless the config file passed via `--config` explicitly
set an `out:` key. `cmd_edl` now calls `build_classes(cfg, store=store)`, reusing the same `Store`
instance `--out` built; `test_edl_cli_forwards_out_without_an_out_key_in_the_config` below is the
regression test (every other test here still also sets `out:` in its config, which remains
harmless -- `store=` takes priority since `build_classes` only falls back to `cfg["out"]` when no
`store` is passed). `dev_gen_path`/`heldout_cap6_path` are unaffected by that fix: left unset, they
still default to the real (multi-MB, external, read-only) `ab2`/`ab3` files, which is almost never
what a test (or a real run pointed at a scratch `--out`) wants, so tests here always set them
explicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import yaml

from code.ndbench.adapters.base import ADAPTERS
from code.ndbench.cli import main
from code.ndbench.core.io import git_commit_sha, read_jsonl, sha256_str, utc_now_iso
from code.ndbench.core.types import (
    JSON,
    LengthLabel,
    Problem,
    ReferenceProof,
    Sequent,
    SuiteManifest,
)
from code.ndbench.store import Store


def _family_problems(n: int) -> list[Problem]:
    """`n` distinct, real-vocabulary (tokenizer-encodable) theorems tagged as the `and_tower n=8`
    EDL class, each with a trivially tokenizable (not necessarily minimal) one-line reference
    proof -- enough for `build_classes`/`run_edl` to actually exercise the tokenizer/tiny-GPT
    machinery, which `test_wp5_build_classes.py`'s synthetic (`SYN_...`) sequents deliberately do
    not need to."""
    out = []
    conclusion = "P"
    for i in range(n):
        if i > 0:
            conclusion = f"( {conclusion} & P )"
        seq = Sequent(premises=("P",), conclusion=conclusion)
        ref = ReferenceProof(body="N1 P : PR ; QED", lines=1, pruned=1, source="constructed")
        out.append(
            Problem(
                pid=f"families:{i}",
                suite="families",
                version="v1",
                sequent=seq,
                length=LengthLabel(L=1, L_ub=None, L_lb=None, method="minlen_exact", bound=None),
                ref=ref,
                tags={"family": "and_tower", "n": 8},
            )
        )
    return out


def _manifest(suite: str, version: str, n: int) -> SuiteManifest:
    return SuiteManifest(
        suite=suite,
        version=version,
        builder="test",
        config={},
        seed=0,
        created_utc=utc_now_iso(),
        git_commit=git_commit_sha(),
        n_items=n,
        items_sha256=sha256_str("x"),
        excl_sources={},
        stats={},
    )


def _write_cfg(tmp_path: Path, out_dir: Path) -> Path:
    """A `--config` YAML setting `out`/`dev_gen_path`/`heldout_cap6_path` explicitly, working
    around the `cmd_edl` gap this file's module docstring documents (and, incidentally, exercising
    `build_classes`'s `cfg`-key fallback path directly, complementing `test_wp5_build_classes.py`'s
    keyword-argument-override tests)."""
    cfg = {
        "out": str(out_dir),
        "dev_gen_path": "/nonexistent/dev_gen.jsonl",
        "heldout_cap6_path": "/nonexistent/heldout_cap6.jsonl",
    }
    path = tmp_path / "edl_config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_edl_cli_runs_end_to_end_with_a_tiny_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        ADAPTERS, "tiny_gpt_wp5", "code.tests.ndbench.test_wp5_fixtures:TinyGptAdapter"
    )

    out_dir = tmp_path / "out"
    store = Store(out_dir)
    problems = _family_problems(64)
    store.write_suite(problems, _manifest("families", "v1", len(problems)))
    config_path = _write_cfg(tmp_path, out_dir)

    rc = main(
        [
            "edl",
            "--model",
            "tiny_gpt_wp5",
            "--model-arg",
            "seed=0",
            "--config",
            str(config_path),
            "--lrs",
            "1e-3",
            "--orders",
            "1",
            "--out",
            str(out_dir),
        ]
    )
    assert rc == 0

    edl_files = sorted((out_dir / "store" / "edl").glob("*.jsonl"))
    assert len(edl_files) == 1
    rows: list[dict[str, JSON]] = list(read_jsonl(edl_files[0]))
    # build_classes could only assemble one of the 12 classes from the single "families" suite
    # this test wrote (and_tower n=8, 64 items -> 32/32 split); 1 lr x 1 order -> exactly 1 row.
    assert len(rows) == 1
    row = rows[0]
    assert row["class"] == "and_tower_n8"
    assert row["n"] == 32
    assert row["m"] == 32
    assert row["lr"] == 1e-3
    assert row["order"] == 0
    assert isinstance(row["EDL"], float)
    assert len(cast(list[float], row["online_nats"])) == 32


def test_edl_cli_forwards_out_without_an_out_key_in_the_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test for fix list item 4: `--out` alone (no `out:` key in `--config`) must still
    let `build_classes` see the suite this test writes to that same `--out` directory. Before the
    fix, `cmd_edl` never forwarded `--out`/`store` into `build_classes`, so this config (which omits
    `out:` on purpose) would make `build_classes` look for suites under a relative `"out"` instead
    and find nothing, producing 0 rows instead of 1."""
    monkeypatch.setitem(
        ADAPTERS, "tiny_gpt_wp5", "code.tests.ndbench.test_wp5_fixtures:TinyGptAdapter"
    )
    out_dir = tmp_path / "out"
    store = Store(out_dir)
    problems = _family_problems(64)
    store.write_suite(problems, _manifest("families", "v1", len(problems)))

    cfg = {
        "dev_gen_path": "/nonexistent/dev_gen.jsonl",
        "heldout_cap6_path": "/nonexistent/heldout_cap6.jsonl",
    }
    config_path = tmp_path / "edl_config_no_out.yaml"
    config_path.write_text(yaml.safe_dump(cfg))

    rc = main(
        [
            "edl",
            "--model",
            "tiny_gpt_wp5",
            "--model-arg",
            "seed=0",
            "--config",
            str(config_path),
            "--lrs",
            "1e-3",
            "--orders",
            "1",
            "--out",
            str(out_dir),
        ]
    )
    assert rc == 0
    edl_files = sorted((out_dir / "store" / "edl").glob("*.jsonl"))
    assert len(edl_files) == 1
    rows: list[dict[str, JSON]] = list(read_jsonl(edl_files[0]))
    assert len(rows) == 1
    assert rows[0]["class"] == "and_tower_n8"


def test_edl_cli_with_no_buildable_classes_still_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No suite at all written to the store -> every one of the 12 classes is skipped with a
    warning (per `build_classes`), and `run_edl` gets an empty problem list; the CLI subcommand
    should still complete cleanly (0 rows), not crash."""
    monkeypatch.setitem(
        ADAPTERS, "tiny_gpt_wp5", "code.tests.ndbench.test_wp5_fixtures:TinyGptAdapter"
    )
    out_dir = tmp_path / "out"
    Store(out_dir)  # creates the directory tree; writes nothing
    config_path = _write_cfg(tmp_path, out_dir)

    rc = main(
        [
            "edl",
            "--model",
            "tiny_gpt_wp5",
            "--model-arg",
            "seed=0",
            "--config",
            str(config_path),
            "--out",
            str(out_dir),
        ]
    )
    assert rc == 0
    edl_files = sorted((out_dir / "store" / "edl").glob("*.jsonl"))
    assert len(edl_files) == 1
    assert list(read_jsonl(edl_files[0])) == []
