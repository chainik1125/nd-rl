"""Localhost dashboard for the autoresearch loop: theorems solved per experiment.

Run from anywhere: `python3 code/experiments/current/autoresearch/dashboard/serve.py`,
then open http://autoresearch.localhost/ (Caddy) or http://127.0.0.1:7102/.
Stdlib only. data.json is rebuilt on every request, so refreshing the page picks up new runs;
there is no build step.

By default it reads results/*/ from the shared branch on origin (`--ref`), fetched at most every
FETCH_SECS, so it shows both agents' pushed registrations and results whichever checkout serves
it. `--results-dir` reads a local directory instead (unpushed work, tests).
"""

import argparse
import contextlib
import io
import json
import subprocess
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


def git(*args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=HERE, capture_output=True, check=True, timeout=60
    ).stdout


def files_from_ref() -> dict[str, dict[str, str]]:
    """{run: {filename: text}} for results/*/ at `ref`, after a throttled fetch."""
    global last_fetch
    if time.monotonic() - last_fetch > FETCH_SECS:
        remote, _, branch = ref.partition("/")
        # Offline: serve the last fetched state rather than an error page.
        with contextlib.suppress(subprocess.CalledProcessError, subprocess.TimeoutExpired):
            git("fetch", "--quiet", remote, branch)
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
            }
        )
    return runs


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            body, ctype = (HERE / "index.html").read_bytes(), "text/html"
        elif self.path == "/data.json":
            body, ctype = json.dumps(load_runs()).encode(), "application/json"
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
