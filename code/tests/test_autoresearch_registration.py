"""Launch validation must finish before any paid pod work starts."""

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

SOURCE = Path(__file__).parents[1] / "experiments/current/autoresearch"


@pytest.fixture
def research(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    for filename in (
        "schema.py",
        "params.py",
        "run_idea.py",
        "record.py",
        "table.py",
        "pretrain.py",
        "harness.py",
        "remote.sh",
        "pod.py",
    ):
        shutil.copyfile(SOURCE / filename, tmp_path / filename)
    shutil.copytree(SOURCE / "pools", tmp_path / "pools")
    modules = {}
    for name in ("schema", "params", "run_idea", "table", "record"):
        spec = importlib.util.spec_from_file_location(name, tmp_path / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        modules[name] = module
    params = modules["params"]
    params.git("init", "-q")
    params.git("config", "user.name", "Registration test")
    params.git("config", "user.email", "test@example.invalid")
    params.git("config", "core.hooksPath", "/dev/null")
    params.git("config", "commit.gpgsign", "false")
    params.git("add", ".")
    params.git("-c", "commit.gpgsign=false", "commit", "-qm", "baseline")
    # A candidate is deliberately uncommitted: only its preregistration must be committed.
    with (tmp_path / "pretrain.py").open("a") as f:
        f.write("\n# Candidate change\n")
    registration = json.loads((SOURCE / "results/011_curve-baseline/params.json").read_text())
    registration.update(
        id="999",
        slug="test",
        pretrain_sha=params.sha(tmp_path / "pretrain.py"),
        base_commit=params.git("rev-parse", "HEAD").strip(),
    )
    registration["config"]["seed"] = registration["protocol"]["seed"] = [3, 4, 5]
    directory = tmp_path / "results/999_test"
    directory.mkdir(parents=True)
    (directory / "params.json").write_text(json.dumps(registration))
    (directory / "pretrain.patch").write_text(params.git("diff", "HEAD", "--", "pretrain.py"))
    params.git("add", "results")
    params.git("-c", "commit.gpgsign=false", "commit", "-qm", "preregister")
    return modules


def test_committed_registration_allows_uncommitted_candidate(research: Any):
    research["run_idea"].preflight("999_test", [3, 4, 5])


@pytest.mark.parametrize("filename", ["params.json", "pretrain.patch"])
@pytest.mark.parametrize("state", ["missing", "untracked", "staged", "modified"])
def test_rejects_uncommitted_registration(research: Any, filename: str, state: str):
    params = research["params"]
    relative = f"results/999_test/{filename}"
    path = params.HERE / relative
    if state == "missing":
        path.unlink()
    elif state in ("untracked", "staged"):
        params.git("rm", "--cached", relative)
        params.git("-c", "commit.gpgsign=false", "commit", "-qm", "unregister")
        if state == "staged":
            params.git("add", relative)
    else:
        with path.open("a") as f:
            f.write("\n")
    with pytest.raises(SystemExit, match="commit|missing"):
        research["run_idea"].preflight("999_test", [3, 4, 5])


@pytest.mark.parametrize("filename", ["pretrain.py", "harness.py"])
def test_rejects_source_hash_mismatch_even_if_committed(research: Any, filename: str):
    params = research["params"]
    with (params.HERE / filename).open("a") as f:
        f.write("\n# Changed after registration\n")
    params.git("add", filename)
    params.git("-c", "commit.gpgsign=false", "commit", "-qm", "changed source")
    with pytest.raises(SystemExit, match="sha"):
        research["run_idea"].preflight("999_test", [3, 4, 5])


@pytest.mark.parametrize("seeds", [[0, 1, 2], [3, 3, 4], []])
def test_rejects_wrong_or_duplicate_seeds(research: Any, seeds: list[int]):
    with pytest.raises(SystemExit, match="seed"):
        research["run_idea"].preflight("999_test", seeds)


@pytest.mark.parametrize("section", ["config", "protocol"])
def test_checks_both_registered_seed_fields(research: Any, section: str):
    params = research["params"]
    path = params.HERE / "results/999_test/params.json"
    registration = json.loads(path.read_text())
    registration[section]["seed"] = 0
    path.write_text(json.dumps(registration))
    params.git("add", "results")
    params.git("-c", "commit.gpgsign=false", "commit", "-qm", "wrong seed")
    with pytest.raises(SystemExit, match=rf"{section}\.seed"):
        research["run_idea"].preflight("999_test", [3, 4, 5])


def test_rejects_dirty_frozen_file(research: Any):
    params = research["params"]
    with (params.HERE / "pod.py").open("a") as f:
        f.write("\n# Uncommitted change\n")
    with pytest.raises(SystemExit, match="frozen files"):
        research["run_idea"].preflight("999_test", [3, 4, 5])


def test_invalid_registration_never_starts_pods(research: Any, monkeypatch: pytest.MonkeyPatch):
    params = research["params"]
    path = params.HERE / "results/999_test/params.json"
    registration = json.loads(path.read_text())
    registration["config"]["arch"]["n_layers"] = "TODO"
    path.write_text(json.dumps(registration))
    params.git("add", "results")
    params.git("-c", "commit.gpgsign=false", "commit", "-qm", "invalid registration")
    runner = research["run_idea"]
    monkeypatch.setattr(sys, "argv", ["run_idea.py", "999_test", "--seeds", "3,4,5"])
    monkeypatch.setattr(runner.threading, "Thread", lambda **kw: pytest.fail("pod work started"))
    with pytest.raises(SystemExit, match="invalid"):
        runner.main()


def test_new_registration_records_requested_seeds(research: Any):
    params = research["params"]
    subprocess.run(
        [sys.executable, str(params.HERE / "params.py"), "new", "998", "next", "--seeds", "3,4,5"],
        check=True,
        capture_output=True,
    )
    registration = json.loads((params.HERE / "results/998_next/params.json").read_text())
    assert registration["config"]["seed"] == [3, 4, 5]
    assert registration["protocol"]["seed"] == [3, 4, 5]
    # The checked-in pool histogram describes seed 0, not these target subsets.
    assert registration["protocol"]["rl"]["targets_sampled"] is None


def test_seed_results_are_typed_and_roundtrip(research: Any):
    schema = research["schema"]
    fields = {"seed_metrics", "seed_metric_rl", "seed_status"}
    assert fields <= schema.Results.model_fields.keys()
    summary = {
        "metric": 20,
        "metric_rl": 40,
        "harness_sha": "abc",
        "pretrain_sha": "def",
        "rules": {},
        "total_secs": 100,
        "seed_metrics": {"3": 10, "4": 30},
        "seed_metric_rl": {"3": 20, "4": 60},
        "seed_status": {"3": "ok", "4": "ok", "5": "missing"},
    }
    recorded = research["record"].results_from_summary(summary, "crash", None)
    restored = schema.Results.model_validate_json(recorded.model_dump_json())
    assert restored.seed_metrics == summary["seed_metrics"]
    assert restored.seed_metric_rl == summary["seed_metric_rl"]
    assert restored.seed_status == summary["seed_status"]
    invalid = recorded.model_dump()
    invalid["seed_metrics"] = {"3": "not a metric"}
    with pytest.raises(ValidationError):
        schema.Results.model_validate(invalid)


def test_curve_point_before_first_loss_roundtrips(research: Any):
    schema = research["schema"]
    point = {"secs": 15, "step": 16, "train_loss": float("nan"), "val_loss": 3.14}
    stats = schema.PretrainStats(curve=[schema.CurvePoint(**point)])  # NaN is dumped as null
    restored = schema.PretrainStats.model_validate_json(stats.model_dump_json())
    assert restored.curve is not None and restored.curve[0].train_loss is None


def test_historical_single_seed_records_still_validate(research: Any):
    schema = research["schema"]
    for path in (SOURCE / "results").glob("*/params.json"):
        schema.Params.model_validate_json(path.read_text())
    for path in (SOURCE / "results").glob("*/results.json"):
        schema.Results.model_validate_json(path.read_text())


@pytest.fixture
def publication(research: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Actual local Git commits/pushes; never uses the real repository or network."""
    params = research["params"]
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
    params.git("remote", "add", "origin", str(remote))
    params.git("push", "-u", "origin", "HEAD")
    (tmp_path / "unrelated.txt").write_text("staged user draft\n")
    params.git("add", "unrelated.txt")
    (tmp_path / "unrelated.txt").write_text("unstaged user draft\n")
    artifacts = tmp_path / "artifacts"
    (artifacts / "999_test").mkdir(parents=True)
    monkeypatch.setattr(research["record"], "ART", artifacts)
    session = tmp_path / "session"
    session.mkdir()
    monkeypatch.setenv("ND_AUTORESEARCH_SESSION", str(session))
    summary = {
        "status": "ok",
        "metric": 150,
        "metric_rl": 400,
        "harness_sha": params.sha(tmp_path / "harness.py"),
        "pretrain_sha": params.sha(tmp_path / "pretrain.py"),
        "rules": {k: v for k, v in params.harness_rules().items() if isinstance(v, int)},
        "total_secs": 100,
    }
    (artifacts / "999_test/summary.json").write_text(json.dumps(summary))
    return research


def assert_published_with_drafts_intact(params: Any) -> None:
    assert params.git("rev-parse", "HEAD") == params.git("rev-parse", "@{upstream}")
    assert params.git("show", ":unrelated.txt") == "staged user draft\n"
    assert (params.HERE / "unrelated.txt").read_text() == "unstaged user draft\n"
    assert "unrelated.txt" not in params.git("ls-tree", "-r", "--name-only", "HEAD")


def test_runner_publishes_only_preregistration(publication: Any):
    params, runner = publication["params"], publication["run_idea"]
    params.new("998", "next", [3, 4, 5])
    template = json.loads((params.HERE / "results/999_test/params.json").read_text())
    template.update(id="998", slug="next")
    (params.HERE / "results/998_next/params.json").write_text(json.dumps(template))
    runner.register("998_next", [3, 4, 5])
    assert set(
        params.git("diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD").splitlines()
    ) == {
        "results/998_next/params.json",
        "results/998_next/pretrain.patch",
    }
    assert params.git("diff", "HEAD", "--", "pretrain.py")
    assert_published_with_drafts_intact(params)
    before = params.git("rev-parse", "HEAD")
    runner.register("998_next", [3, 4, 5])
    assert params.git("rev-parse", "HEAD") == before  # Safe retry, no empty commit.


@pytest.mark.parametrize("failure", ["push", "commit"])
def test_publication_failure_never_starts_pods(
    publication: Any, monkeypatch: pytest.MonkeyPatch, failure: str
):
    params, runner = publication["params"], publication["run_idea"]
    if failure == "push":
        params.git("remote", "set-url", "origin", str(params.HERE / "missing.git"))
    else:
        path = params.HERE / "results/999_test/params.json"
        registered = path.read_text()
        path.unlink()
        params.git("rm", "--cached", "results/999_test/params.json")
        params.git("commit", "--only", "-qm", "unregister", "--", "results/999_test/params.json")
        path.write_text(registered)
        hooks = params.HERE / "hooks"
        hooks.mkdir()
        hook = hooks / "pre-commit"
        hook.write_text("#!/bin/sh\necho 'test hook rejected commit' >&2\nexit 1\n")
        hook.chmod(0o700)
        params.git("config", "core.hooksPath", str(hooks))
    monkeypatch.setattr(sys, "argv", ["run_idea.py", "999_test", "--seeds", "3,4,5"])
    monkeypatch.setattr(runner.threading, "Thread", lambda **kw: pytest.fail("pod work started"))
    with pytest.raises(SystemExit, match=f"Git {failure} failed"):
        runner.main()


@pytest.mark.parametrize("status", ["keep", "discard", "crash", "timeout"])
def test_record_publishes_outcome_and_applies_decision(
    publication: Any, monkeypatch: pytest.MonkeyPatch, status: str
):
    params, recorder = publication["params"], publication["record"]
    baseline = params.git("show", "HEAD:pretrain.py")
    candidate = (params.HERE / "pretrain.py").read_text()
    monkeypatch.setattr(sys, "argv", ["record.py", "999_test", status, "test observation"])
    recorder.main()
    expected = candidate if status == "keep" else baseline
    assert (params.HERE / "pretrain.py").read_text() == expected
    assert params.git("show", "HEAD:pretrain.py") == expected
    assert not params.git("status", "--porcelain", "--", "pretrain.py", "results", "RESULTS.md")
    assert "test observation" in params.git("show", "HEAD:RESULTS.md")
    receipt = json.loads((params.HERE / "session/completed.json").read_text())
    assert receipt == {"name": "999_test", "commit": params.git("rev-parse", "HEAD").strip()}
    assert_published_with_drafts_intact(params)


@pytest.mark.parametrize("status", ["keep", "discard"])
def test_outcome_push_failure_can_retry_without_rewriting_result(
    publication: Any, monkeypatch: pytest.MonkeyPatch, status: str
):
    params, recorder = publication["params"], publication["record"]
    remote = params.git("remote", "get-url", "origin").strip()
    params.git("remote", "set-url", "origin", str(params.HERE / "missing.git"))
    monkeypatch.setattr(sys, "argv", ["record.py", "999_test", status, "test observation"])
    with pytest.raises(SystemExit, match="Git push failed"):
        recorder.main()
    assert not (params.HERE / "session/completed.json").exists()
    recorded = (params.HERE / "results/999_test/results.json").read_bytes()
    committed = params.git("rev-parse", "HEAD")
    params.git("remote", "set-url", "origin", remote)
    recorder.main()
    assert (params.HERE / "results/999_test/results.json").read_bytes() == recorded
    assert params.git("rev-parse", "HEAD") == committed
    assert (params.HERE / "session/completed.json").exists()
    assert_published_with_drafts_intact(params)


@pytest.mark.parametrize("status", ["keep", "discard"])
def test_record_accepts_candidate_committed_after_registration(
    publication: Any, monkeypatch: pytest.MonkeyPatch, status: str
):
    params = publication["params"]
    baseline = params.git("show", "HEAD:pretrain.py")
    candidate = (params.HERE / "pretrain.py").read_text()
    params.git("commit", "--only", "-qm", "commit candidate", "--", "pretrain.py")
    monkeypatch.setattr(sys, "argv", ["record.py", "999_test", status, "test observation"])
    publication["record"].main()
    expected = candidate if status == "keep" else baseline
    assert params.git("show", "HEAD:pretrain.py") == expected
    assert (params.HERE / "pretrain.py").read_text() == expected


def test_record_refuses_to_overwrite_new_pretrain_edits(
    publication: Any, monkeypatch: pytest.MonkeyPatch
):
    params = publication["params"]
    candidate = params.HERE / "pretrain.py"
    candidate.write_text(candidate.read_text() + "\n# Someone else's work\n")
    before = candidate.read_bytes()
    monkeypatch.setattr(sys, "argv", ["record.py", "999_test", "discard"])
    with pytest.raises(SystemExit, match="pretrain.py"):
        publication["record"].main()
    assert candidate.read_bytes() == before
    assert not (params.HERE / "results/999_test/results.json").exists()


def test_check_only_never_publishes_or_starts_work(research: Any, monkeypatch: pytest.MonkeyPatch):
    runner = research["run_idea"]
    calls = []
    monkeypatch.setattr(runner, "ensure_idle", lambda name: calls.append(name))
    monkeypatch.setattr(runner, "register", lambda *a: pytest.fail("Git write"))
    monkeypatch.setattr(
        runner, "archive_previous_attempt", lambda *a: pytest.fail("artifact write")
    )
    monkeypatch.setattr(runner.threading, "Thread", lambda **kw: pytest.fail("pod work"))
    monkeypatch.setattr(
        sys, "argv", ["run_idea.py", "999_test", "--seeds", "3,4,5", "--check-only"]
    )
    runner.main()
    assert calls == ["999_test"]


def test_recorded_experiment_cannot_rerun(research: Any):
    (research["params"].HERE / "results/999_test/results.json").write_text("{}")
    with pytest.raises(SystemExit, match="already has a recorded result"):
        research["run_idea"].preflight("999_test", [3, 4, 5])


@pytest.mark.parametrize("active", ["runner", "pod", "none"])
def test_resume_rejects_active_attempt_but_ignores_stale_pid(
    research: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, active: str
):
    runner = research["run_idea"]
    artifacts = tmp_path / "artifacts"
    session = artifacts / "loop/session-old"
    session.mkdir(parents=True)
    (session / "launch-1.json").write_text(json.dumps({"name": "999_test", "pid": 12345678}))
    monkeypatch.setattr(runner, "ART", str(artifacts))

    def fake_run(argv: list[str], **kwargs: Any):
        if argv[0] == "ps":
            command = "python run_idea.py 999_test --seeds 3,4,5" if active == "runner" else ""
            return subprocess.CompletedProcess(argv, 0, command, "")
        assert argv[-1] == "list"  # No up/down/kill commands in a preflight.
        pods = [{"name": "autoresearch-unrelated_s0"}]
        if active == "pod":
            pods.append({"name": "autoresearch-999_test_s3-session-old"})
        return subprocess.CompletedProcess(argv, 0, json.dumps({"pods": pods}), "")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    if active == "none":
        runner.ensure_idle("999_test")
    else:
        with pytest.raises(SystemExit, match="still active"):
            runner.ensure_idle("999_test")


def test_previous_summaries_are_archived_not_reused(
    research: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    runner = research["run_idea"]
    artifacts = tmp_path / "artifacts"
    monkeypatch.setattr(runner, "ART", str(artifacts))
    for name in ("999_test", "999_test_s3", "999_test_s4", "998_other"):
        (artifacts / name).mkdir(parents=True)
        (artifacts / name / "summary.json").write_text('{"old": true}')
    runner.archive_previous_attempt("999_test", [3, 4, 5])
    assert (artifacts / "998_other/summary.json").exists()
    assert len(list(artifacts.glob("previous_attempts/*/*/summary.json"))) == 3
    with pytest.raises(SystemExit, match="no seed produced"):
        runner.merge("999_test", [3, 4, 5])
