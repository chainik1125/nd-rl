"""CPU checks of checkpoint boundaries without the external training fork."""

import importlib.util
import json
import random
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

HARNESS = Path(__file__).parents[1] / "experiments/current/autoresearch/harness.py"


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> Any:
    for name, attributes in {
        "sample": {"generate": None},
        "train": {"batch": None, "loss_on": None},
        "eval_set": {"judge": None, "wilson": None},
        "normalize": {"norm": lambda proof: proof},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    spec = importlib.util.spec_from_file_location("checkpoint_test_harness", HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_roundtrip_and_interrupted_write(
    harness: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model = torch.nn.Linear(2, 1)
    args = SimpleNamespace(out=str(tmp_path), seed=7)
    tok = SimpleNamespace(mode="abs")
    harness.save_checkpoint(args, model, tok, "pretrain")
    saved = torch.load(tmp_path / "pretrain.pt", weights_only=True)
    assert saved["stage"] == "pretrain"
    assert saved["seed"] == 7
    assert saved["tok_mode"] == "abs"
    restored = torch.nn.Linear(2, 1)
    restored.load_state_dict(saved["state"])
    assert torch.equal(model.weight, restored.weight)
    original = (tmp_path / "pretrain.pt").read_bytes()

    def interrupted_save(payload: Any, path: Path):
        path.write_bytes(b"partial")
        raise harness.Timeout()

    monkeypatch.setattr(harness.torch, "save", interrupted_save)
    with pytest.raises(harness.Timeout):
        harness.save_checkpoint(args, model, tok, "pretrain")
    assert (tmp_path / "pretrain.pt").read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_completed_rl_round_survives_later_timeout(
    harness: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    args = SimpleNamespace(
        out=str(tmp_path), seed=0, rl_rounds=2, k=1, max_new=1, retain=0, ft_steps=1
    )
    model = torch.nn.Linear(2, 1)
    tok = SimpleNamespace(mode="abs")
    calls = 0

    def sample(*args: Any) -> tuple[list[Any], dict[str, int]]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise harness.Timeout()
        return [], {"samples": 1, "samples_per_sec": 1}

    monkeypatch.setattr(harness, "sample_and_judge", sample)
    with pytest.raises(harness.Timeout):
        harness.stage_rl(args, tok, model, [], [], random.Random(0))
    saved = torch.load(tmp_path / "rl_round_01.pt", weights_only=True)
    assert saved["stage"] == "rl_round_01"
    assert not (tmp_path / "rl_round_02.pt").exists()


@pytest.mark.parametrize("pretrain_only", [False, True])
def test_pretrain_saved_before_eval_timeout(
    harness: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pretrain_only: bool
):
    model = torch.nn.Linear(2, 1)
    tok = SimpleNamespace(mode="abs")
    agent = ModuleType("pretrain")
    agent.__dict__["pretrain"] = lambda *args: (model, tok, {})
    monkeypatch.setitem(sys.modules, "pretrain", agent)
    monkeypatch.setattr(harness, "DEV", "cpu")
    argv = [str(HARNESS), "--out", str(tmp_path)]
    if pretrain_only:
        argv.append("--pretrain-only")
    monkeypatch.setattr(sys, "argv", argv)

    def read(path: str) -> list[dict[str, Any]]:
        if path.endswith("heldout.jsonl"):
            return [{}] * 2000
        return []

    def timeout(*args: Any):
        assert not pretrain_only, "Pretrain-only run must not evaluate or run RL"
        raise harness.Timeout()

    monkeypatch.setattr(harness, "read", read)
    monkeypatch.setattr(harness, "greedy_rate", timeout)
    monkeypatch.setattr(harness, "stage_rl", timeout)
    monkeypatch.setattr(harness.signal, "signal", lambda *args: None)
    monkeypatch.setattr(harness.signal, "alarm", lambda *args: None)
    harness.main()
    assert torch.load(tmp_path / "pretrain.pt", weights_only=True)["stage"] == "pretrain"
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == ("ok" if pretrain_only else "timeout")
    if pretrain_only:
        assert summary["stage"] == "pretrain_only"
        assert summary["metric"] is None
    assert not (tmp_path / "final.pt").exists()
    for filename in ("pretrain.py", "harness.py", "remote.sh"):
        assert (tmp_path / "source" / filename).read_bytes() == (
            HARNESS.parent / filename
        ).read_bytes()
