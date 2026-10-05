"""Submission safety for LLM trace generation: dry run by default, --max-usd refusal, no double
submission, the API key never leaves the request header (code/thinking/gen/batch.py, client.py)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest

from code.thinking.gen import batch as bt
from code.thinking.gen import cli, fakes
from code.thinking.gen import requests as rq
from code.thinking.gen import select as sl
from code.thinking.gen.client import ApiError, HttpBatchClient

FORK = Path(os.environ.get("ND_TAKEHOME", "/scratch/cowifornia/nd-rl/think/nd-takehome"))
pytestmark = pytest.mark.skipif(
    not (FORK / "state_env.py").exists() or not sl.K12.exists(), reason="needs Dan's fork and K12"
)
SENTINEL = "sk-ant-test-0123456789-SENTINEL-abcdef"


def setup_run(tmp_path: Path, n: int = 3) -> tuple[Path, Path]:
    run = tmp_path / "run"
    rq.write_jsonl(run / "selection.jsonl", fakes.pilot_selection(sl.K12)[:n])
    assert cli.main(["build", "--run-dir", str(run), "--models", "claude-opus-5-5"]) == 0
    return run, run / "requests" / "opus55.r1.jsonl"


def fake_client() -> tuple[HttpBatchClient, fakes.FakeBatchAPI]:
    api = fakes.FakeBatchAPI(fakes.pilot_responder(), expect_key=SENTINEL)
    return HttpBatchClient(transport=api), api


def test_submit_is_a_dry_run_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL)
    run, req = setup_run(tmp_path)
    client, api = fake_client()
    assert cli.main(["submit", "--run-dir", str(run), "--requests", str(req)], client=client) == 0
    assert api.calls == []
    assert not (run / "ledger.jsonl").exists()
    est = json.loads(req.with_name("opus55.r1.estimate.json").read_text())
    assert est["n_requests"] == 3 and est["worst_usd"] > est["expected_usd"] > 0


def test_max_usd_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL)
    run, req = setup_run(tmp_path)
    client, api = fake_client()
    worst = rq.estimate_file(req)["worst_usd"]
    argv = ["submit", "--run-dir", str(run), "--requests", str(req), "--submit"]
    assert cli.main([*argv, "--max-usd", f"{worst * 0.9:.6f}"], client=client) == 2
    assert "exceeds --max-usd" in capsys.readouterr().err
    assert cli.main(argv, client=client) == 2
    assert "needs --max-usd" in capsys.readouterr().err
    assert api.calls == [] and not (run / "ledger.jsonl").exists()
    assert cli.main([*argv, "--max-usd", f"{worst * 1.01:.6f}"], client=client) == 0
    assert [m for m, _ in api.calls] == ["POST"]


def test_resubmitting_the_same_file_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL)
    run, req = setup_run(tmp_path)
    client, api = fake_client()
    argv = ["submit", "--run-dir", str(run), "--requests", str(req), "--submit", "--max-usd", "100"]
    assert cli.main(argv, client=client) == 0
    assert cli.main(argv, client=client) == 2
    assert len(api.bodies) == 1
    (row,) = bt.ledger_rows(run / "ledger.jsonl")
    assert row["kind"] == "submit" and row["batch_id"] == "msgbatch_fake0001"
    assert row["model"] == "claude-opus-5-5" and row["n_requests"] == 3
    assert row["estimated_usd"] > 0 and row["max_usd"] == 100


def test_unset_key_refuses_without_sending(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    run, req = setup_run(tmp_path)
    with pytest.raises(bt.Refused, match="not set"):
        bt.submit(req, run, send=True, max_usd=100)
    _, api = fake_client()
    with pytest.raises(ApiError, match="not set"):
        HttpBatchClient(transport=api).retrieve("msgbatch_x")
    assert api.calls == []


def test_key_is_never_sent_elsewhere_and_post_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL)
    seen: list[str] = []

    def transport(
        method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float
    ) -> tuple[int, bytes]:
        seen.append(method)
        return 503, b'{"error": {"type": "overloaded_error", "message": "busy"}}'

    c = HttpBatchClient(transport=transport, get_retries=0)
    with pytest.raises(ApiError, match="refusing"):
        c._call("GET", "https://example.com/v1/messages/batches/x")
    with pytest.raises(ApiError, match="503"):
        c.create([{"custom_id": "a", "params": {"model": "m", "max_tokens": 1, "messages": []}}])
    assert seen == ["POST"]


def test_key_never_written_or_logged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The whole pipeline with the real HTTP client code and a fake server: build, submit, poll,
    collect, retry build.  The key must appear in no file, no output and no log record."""
    caplog.set_level(logging.DEBUG)
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL)
    run, req = setup_run(tmp_path, n=6)
    api = fakes.FakeBatchAPI(
        fakes.pilot_responder({"k12_16472": "false_look"}), expect_key=SENTINEL
    )
    client = HttpBatchClient(transport=api)
    r = str(run)
    assert (
        cli.main(
            [
                "submit",
                "--run-dir",
                r,
                "--requests",
                str(req),
                "--submit",
                "--max-usd",
                "100",
                "--wait",
                "--interval",
                "0",
            ],
            client=client,
        )
        == 0
    )
    assert cli.main(["collect", "--run-dir", r], client=client) == 0
    retry = run / "requests" / "opus55.r2.jsonl"
    assert retry.exists()
    assert (
        cli.main(
            ["submit", "--run-dir", r, "--requests", str(retry), "--submit", "--max-usd", "100"],
            client=client,
        )
        == 0
    )
    assert cli.main(["poll", "--run-dir", r], client=client) == 0
    assert cli.main(["collect", "--run-dir", r], client=client) == 0
    assert cli.main(["report", "--run-dir", r], client=client) == 0
    out = capsys.readouterr()
    assert SENTINEL not in out.out + out.err
    assert all(SENTINEL not in rec.getMessage() for rec in caplog.records)
    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert len(files) > 10
    for p in files:
        assert SENTINEL.encode() not in p.read_bytes(), p
    assert all(SENTINEL not in json.dumps(b) for b in api.bodies)
    stats = json.loads((run / "collect" / "stats.json").read_text())
    assert stats["by_model"]["claude-opus-5-5"]["accepted"] == 6
