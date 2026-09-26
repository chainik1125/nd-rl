#!/usr/bin/env python3
"""Fresh Codex or Claude sessions, one experiment each, with a quiet terminal. See LOOP.md."""

import argparse
import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ART = ROOT / "artifacts/autoresearch"
POLL_SECS = 0.5
# Both agents share one branch, each from its own checkout. Highest effort for both (Robbie).
FABLE = "claude-fable-5-1"
MODELS = {
    "codex": "gpt-6-astra",
    "claude": "claude-opus-5-5",
    "fable": FABLE,
    "fable2": FABLE,
    "fable3": FABLE,
}
agent = "codex"  # set once by main(); tests monkeypatch it
stop_at: float | None = None  # epoch seconds; no new session starts after it (--stop-at)
MIN_BALANCE = 3.0  # USD: one experiment rents 3 A40s at ~$0.49/h for ~25 min
# A session must launch run_idea.py this soon (Robbie, 2026-09-23: Claude at max effort took
# 20-150 min to propose, Codex ~3; raised 20 -> 40 min on 2026-09-25 so bench_pretrain.py
# probes and bigger implementations fit). Runs themselves are bounded by the harness, not this.
PROPOSE_SECS = 40 * 60
MAX_TIMEOUTS = 3  # consecutive proposal timeouts before the loop stops


def emit(message: str) -> None:
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%H:%M:%S %Z")
    print(f"[{stamp}] {message}", flush=True)


def read_json(path: Path):
    # Results can be observed mid-write; manifests are atomically published by run_idea.py.
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def launches(session: Path):
    return [(p, entry) for p in sorted(session.glob("launch-*.json")) if (entry := read_json(p))]


class ExperimentBlocked(RuntimeError):
    """The agent exited without completing an experiment."""


class ProposalTimeout(RuntimeError):
    """The agent did not launch an experiment within PROPOSE_SECS."""


def discard_unlaunched_candidate(session: Path) -> None:
    """Undo a timed-out proposal (pretrain.py edit, untracked registration) so the next fresh
    session does not block on it. The diff is kept in the session directory."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=HERE, capture_output=True, text=True, check=True
        ).stdout

    (session / "discarded.patch").write_text(git("diff", "HEAD", "--", "pretrain.py"))
    git("restore", "--source=HEAD", "--staged", "--worktree", "--", "pretrain.py")
    untracked = git("ls-files", "--others", "--exclude-standard", "-z", "--", "results")
    for relative in filter(None, untracked.split("\0")):
        (HERE / relative).unlink()
        with contextlib.suppress(OSError):
            (HERE / relative).parent.rmdir()


def iteration_prompt(prompt: str) -> str:
    """A cancelled, committed experiment is work to finish, not an unexplained dirty edit."""
    pending = sorted(
        path
        for path in (HERE / "results").glob("*/params.json")
        if not path.with_name("results.json").exists() and owner(path.parent.name) == agent
    )
    if not pending:
        return prompt
    if len(pending) > 1:
        raise ExperimentBlocked(
            "Multiple unfinished experiments: " + ", ".join(path.parent.name for path in pending)
        )
    path = pending[0]
    name = path.parent.name
    seeds = json.loads(path.read_text())["config"]["seed"]
    seed_arg = ",".join(str(seed) for seed in ([seeds] if isinstance(seeds, int) else seeds))
    # Keep this launcher stdlib-only; schema validation lives in the project's own environment.
    checked = subprocess.run(
        [
            str(ROOT / "code/.venv/bin/python"),
            str(HERE / "run_idea.py"),
            name,
            "--seeds",
            seed_arg,
            "--check-only",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=45,
    )
    if checked.returncode:
        detail = textwrap.shorten(checked.stderr or checked.stdout, width=600, placeholder="…")
        raise ExperimentBlocked(f"Cannot resume {name}: {detail}")
    emit(f"Resuming experiment | {name} | seeds {seed_arg} | restarting training from scratch")
    return prompt + (
        f"\n\nRESUME {name}: the launcher validated its committed preregistration, candidate "
        "hash, frozen files, seeds, and absence of an active previous attempt. The matching "
        "pretrain.py edits belong to this experiment, not unrelated work. Do not propose a new "
        "idea or edit its code/registration. Run the existing experiment in the foreground:\n"
        f"code/.venv/bin/python code/experiments/current/autoresearch/run_idea.py {name} "
        f"--seeds {seed_arg}\n"
        "Then record and publish its outcome normally. Complete only this experiment."
    )


def owner(name: str) -> str:
    """The agent suffix of the id (041-claude_slug); unsuffixed ids predate Claude, so Codex."""
    return name.split("_", 1)[0].partition("-")[2] or "codex"


def agent_explanation(session: Path) -> str:
    """Show only the last public agent message, never reasoning or tool output."""
    message = ""
    try:
        with (session / f"{agent}.jsonl").open() as log:
            for line in log:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                if event.get("type") == "result" and isinstance(event.get("result"), str):
                    message = event["result"] or message  # Claude stream-json final answer
                    continue
                if event.get("type") != "item.completed":
                    continue
                item = event.get("item", {})
                if (
                    isinstance(item, dict)
                    and item.get("type") == "agent_message"
                    and isinstance(item.get("text"), str)
                    and item["text"].strip()
                ):
                    message = item["text"]
    except FileNotFoundError:
        pass
    if not message:
        return "No agent explanation was recorded."
    return "Agent: " + textwrap.shorten(message, width=360, placeholder="…")


class Monitor:
    def __init__(self, session: Path):
        self.session = session
        self.offsets: dict[Path, int] = {}
        self.started: set[str] = set()
        self.phases: set[tuple[Path, str]] = set()

    def poll(self) -> None:
        for _, entry in launches(self.session):
            name = entry["name"]
            if name not in self.started:
                params = read_json(HERE / "results" / name / "params.json") or {}
                emit(f"Experiment started | {name} | {params.get('title', '')}")
                self.started.add(name)
            for seed in entry["seeds"]:
                path = Path(entry["logs"]) / f"{name}_s{seed}.driver.log"
                if not path.exists():
                    continue
                with path.open() as stream:
                    stream.seek(self.offsets.get(path, 0))
                    while line := stream.readline():
                        if not line.endswith("\n"):
                            break
                        self.offsets[path] = stream.tell()
                        self.line(path, f"{name} s{seed}", line)

    def line(self, path: Path, label: str, line: str) -> None:
        message = re.sub(r"^\[\d{2}:\d{2}:\d{2}\] ", "", line)
        phase = None
        if message.startswith("pools:"):
            phase = "Pretrain started"
        elif message.startswith("pretrain held-out greedy "):
            phase = "RL started"
        if phase and (path, phase) not in self.phases:
            emit(f"{phase} | {label}")
            self.phases.add((path, phase))
        match = re.match(r"ei r(\d+): acc [\d.]+, \+(\d+) proofs, solved (\d+/\d+),", message)
        if match:
            round_no, proofs, solved = match.groups()
            emit(
                f"RL round {round_no} finished | {label} | {solved} solved cumulatively, "
                f"+{proofs} proofs"
            )

    def finish(self) -> None:
        self.poll()
        entries = launches(self.session)
        names = {entry["name"] for _, entry in entries}
        if not names:
            raise ExperimentBlocked(f"No experiment launched. {agent_explanation(self.session)}")
        if len(names) > 1:
            raise RuntimeError(
                f"Expected one experiment, found {len(names)}: {', '.join(sorted(names))}"
            )
        name = names.pop()
        directory = HERE / "results" / name
        path = directory / "results.json"
        result = read_json(path)
        if not result or path.stat().st_mtime_ns < max(p.stat().st_mtime_ns for p, _ in entries):
            raise ExperimentBlocked(
                f"{name}: no new recorded result. {agent_explanation(self.session)}"
            )
        completed = read_json(self.session / "completed.json") or {}
        if completed.get("name") != name or not completed.get("commit"):
            raise ExperimentBlocked(
                f"{name}: outcome not committed and pushed. {agent_explanation(self.session)}"
            )
        params = read_json(directory / "params.json") or {}
        score, baseline, status = result["metric"], params.get("baseline_metric"), result["status"]
        comparison = ""
        if score is not None and baseline is not None:
            comparison = f" vs baseline {baseline} ({score - baseline:+g})"
        best = status == "keep" and score is not None and baseline is not None and score > baseline
        seeds = ", ".join(
            f"s{s}={value}" for s, value in (result.get("seed_metrics") or {}).items()
        )
        emit(
            f"Experiment finished | {name} | {'NEW BEST' if best else status} | "
            f"score {score}{comparison} | {seeds}"
        )


def stop_processes(process: subprocess.Popen[bytes] | None, session: Path) -> None:
    # Agent tools may have their own process groups. The runner explicitly registers its group.
    groups = {entry["pid"] for _, entry in launches(session) if entry.get("pid")}
    if process is not None:
        groups.add(process.pid)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for group in groups:
            try:
                os.killpg(group, sig)
            except ProcessLookupError:
                pass
            except PermissionError as exc:
                # Do not let a local signaling error skip cleanup of rented pods.
                emit(f"ERROR | could not signal process group {group} ({sig.name}): {exc}")
        if process is not None:
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                if sig == signal.SIGKILL:
                    emit(
                        f"ERROR | {agent.title()} process {process.pid} did not exit; "
                        "continuing pod cleanup"
                    )


def cleanup_pods(session: Path) -> bool:
    # bench-*.json: pods rented by bench_pretrain.py during the proposal phase.
    benches = [read_json(p) or {} for p in session.glob("bench-*.json")]
    owned = {
        f"autoresearch-{name}": name
        for entry in [*(entry for _, entry in launches(session)), *benches]
        for name in entry.get("pods", [])
    }
    if not owned:
        return True

    def call(*args: str):
        completed = subprocess.run(
            [sys.executable, str(HERE / "pod.py"), *args],
            capture_output=True,
            text=True,
            timeout=30,
        )
        with (session / "cleanup.log").open("a") as log:
            log.write(completed.stdout + completed.stderr)
        if completed.returncode:
            raise RuntimeError("pod command failed")
        return completed.stdout

    try:
        # Query exact, unique names too: cancellation can precede the local pod-state write.
        pods = json.loads(call("list"))["pods"]
        failed = False
        for pod in pods:
            if pod["name"] not in owned:
                continue
            try:
                call("kill", pod["id"], "--pod", owned[pod["name"]])
                emit(f"Cleanup | terminated {pod['name']}")
            except (OSError, RuntimeError, subprocess.TimeoutExpired):
                emit(f"ERROR | cleanup failed for {pod['name']}; see {session / 'cleanup.log'}")
                failed = True
        return not failed
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.TimeoutExpired):
        emit(f"ERROR | pod cleanup could not list pods; see {session / 'cleanup.log'}")
        return False


def check_balance() -> None:
    """Stop before renting pods the account cannot pay for; unknown balance is not a blocker."""
    try:
        listed = subprocess.run(
            [sys.executable, str(HERE / "pod.py"), "list"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        balance = float(json.loads(listed.stdout)["clientBalance"])
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired):
        return
    if balance < MIN_BALANCE:
        raise ExperimentBlocked(f"RunPod balance ${balance:.2f} < ${MIN_BALANCE:.2f}; top up")


def agent_vars(session: Path) -> dict[str, str]:
    return {
        "ND_AUTORESEARCH_SESSION": str(session),
        "ND_AGENT": agent,
        "ND_AGENT_MODEL": MODELS[agent],
    }


def agent_argv(command: list[str], session: Path, prompt: str) -> list[str]:
    if agent != "codex":  # claude and fable both run the claude CLI
        return [
            *command,
            "--print",
            "--output-format",
            "stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
            "--model",
            MODELS[agent],
            "--effort",
            "max",
            prompt,
        ]
    return [
        *command,
        "exec",
        "--json",
        "-C",
        str(ROOT),
        "-c",
        'approval_policy="never"',
        *(
            arg
            for k, v in agent_vars(session).items()
            for arg in ("-c", f"shell_environment_policy.set.{k}={json.dumps(v)}")
        ),
        prompt,
    ]


def agent_env(session: Path) -> dict[str, str]:
    # Drop a parent Claude Code session's variables and any API key: Claude must run as a fresh
    # top-level session billed to the logged-in claude.ai subscription, not API credits.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CLAUDE", "ANTHROPIC_API_KEY"))}
    return {**env, **agent_vars(session)}


def run_loop(command: list[str], prompt: str) -> int:
    directory = ART / "loop"
    directory.mkdir(parents=True, exist_ok=True)
    timeouts = 0
    while True:
        if stop_at is not None and time.time() >= stop_at:
            emit(f"STOPPED | --stop-at {time.strftime('%H:%M', time.localtime(stop_at))} reached")
            return 0
        session = Path(tempfile.mkdtemp(prefix="session-", dir=directory))
        monitor = Monitor(session)
        process = None
        resuming = False
        try:
            emit(f"Starting {agent.title()} | checking experiment state | diagnostics: {session}")
            # Re-read per session so prompt fixes reach a running loop (the tests pass a string).
            path = HERE / "iteration.md"
            base_prompt = path.read_text() if path.exists() else prompt
            current_prompt = iteration_prompt(base_prompt)
            resuming = current_prompt != base_prompt
            check_balance()
            with (
                (session / f"{agent}.jsonl").open("w") as output,
                (session / f"{agent}.stderr").open("w") as errors,
            ):
                process = subprocess.Popen(
                    agent_argv(command, session, current_prompt),
                    cwd=ROOT,
                    env=agent_env(session),
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=errors,
                    start_new_session=True,
                )
                started = time.monotonic()
                while process.poll() is None:
                    monitor.poll()
                    if not launches(session) and time.monotonic() - started > PROPOSE_SECS:
                        raise ProposalTimeout(f"no launch within {PROPOSE_SECS // 60} min")
                    time.sleep(POLL_SECS)
                if process.returncode:
                    raise RuntimeError(
                        f"{agent.title()} exited with code {process.returncode}. "
                        + agent_explanation(session)
                    )
                monitor.finish()
                timeouts = 0
        except ProposalTimeout as exc:
            stop_processes(process, session)
            cleanup_pods(session)
            timeouts += 1
            if not resuming:  # A resumed candidate is committed work: keep it.
                discard_unlaunched_candidate(session)
            if timeouts >= MAX_TIMEOUTS:
                emit(f"BLOCKED | {timeouts} proposal timeouts in a row; see {session}")
                return 1
            emit(f"TIMEOUT | {exc}; candidate discarded, fresh session next; see {session}")
        except KeyboardInterrupt:
            emit(f"Cancelled | stopping {agent.title()} and cleaning up owned pods")
            stop_processes(process, session)
            cleanup_pods(session)
            return 130
        except Exception as exc:
            label = "BLOCKED" if isinstance(exc, ExperimentBlocked) else "ERROR"
            emit(f"{label} | {exc}; diagnostics: {session}")
            stop_processes(process, session)
            cleanup_pods(session)
            return 1


def main() -> int:
    global agent, stop_at
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=sorted(MODELS), default="codex")
    parser.add_argument(
        "--stop-at",
        metavar="HH:MM",
        help="local time after which no new experiment starts; the running one finishes",
    )
    parser.add_argument(
        "agent_args",
        nargs=argparse.REMAINDER,
        help="Extra CLI options after -- (Codex: permissions; Claude needs none)",
    )
    args = parser.parse_args()
    agent = args.agent
    if args.stop_at:
        hour, minute = map(int, args.stop_at.split(":"))
        now = datetime.now()
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        stop_at = target.timestamp() + (86400 if target <= now else 0)  # next occurrence
    options = args.agent_args
    if options[:1] == ["--"]:
        options = options[1:]
    if not os.environ.get("RUNPOD_API_KEY"):
        emit("ERROR | RUNPOD_API_KEY is unset; run source .envrc.local from the repository root")
        return 1

    def cancel(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, cancel)
    prompt = (HERE / "iteration.md").read_text()
    if agent == "codex":
        options = ["-m", MODELS["codex"], "-c", 'model_reasoning_effort="max"', *options]
    return run_loop(["codex" if agent == "codex" else "claude", *options], prompt)


if __name__ == "__main__":
    sys.exit(main())
