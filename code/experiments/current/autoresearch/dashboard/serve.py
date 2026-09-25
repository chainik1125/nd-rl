"""Localhost dashboard for the autoresearch loop: theorems solved per experiment.

One page: Lean-protocol runs (since 098, fork 51604b3) on top, token-format runs 000-097 below.

Run from anywhere: `python3 code/experiments/current/autoresearch/dashboard/serve.py`,
then open http://autoresearch.localhost/ (Caddy) or http://127.0.0.1:7102/.
Stdlib only. data.json is rebuilt on every request, so refreshing the page picks up new runs;
there is no build step.

By default it reads results/*/ from the shared branch on origin (`--ref`), fetched at most every
FETCH_SECS, so it shows both agents' pushed registrations and results whichever checkout serves
it. `--results-dir` reads a local directory instead (unpushed work, tests).
"""

import argparse
import io
import json
import subprocess
import sys
import tarfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = 7102
FETCH_SECS = 20
HERE = Path(__file__).parent
results_dir: Path | None = None
ref = "origin/autoresearch-pretrain"
last_fetch = 0.0
fetch_error: str | None = None  # shown on the page: a failing fetch once froze it at run 057


def git(*args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=HERE, capture_output=True, check=True, timeout=60
    ).stdout


def files_from_ref() -> dict[str, dict[str, str]]:
    """{run: {filename: text}} for results/*/ at `ref`, after a throttled fetch."""
    global last_fetch, fetch_error
    if time.monotonic() - last_fetch > FETCH_SECS:
        remote, _, branch = ref.partition("/")
        # Offline: serve the last fetched state rather than an error page, but say so.
        try:
            git("fetch", "--quiet", remote, branch)
            fetch_error = None
        except subprocess.CalledProcessError as e:
            fetch_error = e.stderr.decode(errors="replace").strip()[-300:] or f"exit {e.returncode}"
        except subprocess.TimeoutExpired:
            fetch_error = "git fetch timed out"
        if fetch_error:
            print(f"fetch failed: {fetch_error}", file=sys.stderr, flush=True)
        last_fetch = time.monotonic()
    results = git("-C", str(HERE.parent), "rev-parse", "--show-prefix").decode().strip() + "results"
    root = git("rev-parse", "--show-toplevel").decode().strip()  # archive refuses subdirectories
    archive = git("-C", root, "archive", "--format=tar", f"{ref}:{results}")  # <run>/<file>
    runs: dict[str, dict[str, str]] = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar.getmembers():
            run, _, filename = member.name.partition("/")
            stream = tar.extractfile(member) if filename.endswith(".json") else None
            if stream:
                runs.setdefault(run, {})[filename] = stream.read().decode()
    return runs


def files_from_dir(directory: Path) -> dict[str, dict[str, str]]:
    return {
        run.name: {f.name: f.read_text() for f in run.glob("*.json")}
        for run in directory.iterdir()
        if run.is_dir()
    }


def as_of() -> str:
    if results_dir:
        return f"local directory {results_dir}"
    fmt = ("--format=%h, committed %cd", "--date=format-local:%b %d %H:%M")
    return git("log", "-1", *fmt, ref).decode().strip()


def load_runs() -> list[dict[str, object]]:
    runs = []
    files = files_from_dir(results_dir) if results_dir else files_from_ref()
    for run in sorted(files):
        if "params.json" not in files[run]:
            continue
        params = json.loads(files[run]["params.json"])
        # A pre-registered run has params.json but no results.json until record.py runs.
        results = json.loads(files[run].get("results.json", "{}"))
        budget = params["protocol"]["budget"]
        runs.append(
            {
                "id": params["id"],
                # 041-claude -> claude; ids before the parallel loops have no agent suffix.
                "agent": params["id"].partition("-")[2] or None,
                "slug": params["slug"],
                "title": params["title"],
                "hypothesis": params.get("hypothesis"),
                "tags": params.get("tags", []),
                "model": params.get("model"),
                "budget": f"{budget['total_secs']} s total / {budget['pretrain_secs']} s pretrain",
                "status": results.get("status", "pending"),
                "metric": results.get("metric"),
                # Runs before 012 were single-seed and have no seed_metrics.
                "seed_metrics": results.get("seed_metrics"),
                # Cumulative RL targets solved by the last expert-iteration round.
                "metric_rl": results.get("metric_rl"),
                "seed_metric_rl": results.get("seed_metric_rl"),
                "n_targets": params["protocol"]["rl"]["n_targets"],
                "n_dev": params["protocol"]["eval"]["pool"]["n"],
                "surprise": results.get("surprise"),
                "fork": params["protocol"]["fork_commit"][:7],
                "base_commit": params.get("base_commit"),
                "baseline_id": params.get("baseline_id"),
                "p_keep": params.get("p_keep"),
                "expected_delta": params.get("expected_delta"),
                # Per-seed final pretrain val loss etc. (since 098), and summed Lean-gate counts.
                "seed_pretrain": results.get("seed_pretrain"),
                "lean_gate": results.get("lean_gate"),
                "total_secs": results.get("total_secs"),
                "recorded_at": results.get("recorded_at"),
            }
        )
    return runs


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            body, ctype = (HERE / "index.html").read_bytes(), "text/html"
        elif self.path == "/data.json":
            runs = load_runs()
            payload = {"runs": runs, "as_of": as_of(), "fetch_error": fetch_error}
            body, ctype = json.dumps(payload).encode(), "application/json"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


def main(argv: list[str] | None = None) -> None:
    global results_dir, ref
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, help="read a local directory instead")
    parser.add_argument("--ref", default=ref, help="git ref with results/ (default: %(default)s)")
    args = parser.parse_args(argv)
    ref = args.ref
    if args.results_dir:
        results_dir = local = args.results_dir.resolve()
        if not local.is_dir():
            parser.error(f"results directory does not exist: {local}")
        print(f"Reading results from {local}")
    else:
        print(f"Reading results from {ref} (fetched at most every {FETCH_SECS} s)")
    print("http://autoresearch.localhost/  (direct: http://127.0.0.1:7102/)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
