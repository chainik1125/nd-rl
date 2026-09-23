"""Localhost dashboard for the autoresearch loop: theorems solved per experiment.

Run from anywhere: `python3 code/experiments/current/autoresearch/dashboard/serve.py`,
then open http://autoresearch.localhost/ (Caddy) or http://127.0.0.1:7102/.
Stdlib only. data.json is rebuilt from results/*/ on every request, so refreshing the
page picks up new runs; there is no build step.
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = 7102
HERE = Path(__file__).parent
results_dir = HERE.parent / "results"


def load_runs() -> list[dict[str, object]]:
    runs = []
    for run_dir in sorted(results_dir.iterdir()):
        params_path, results_path = run_dir / "params.json", run_dir / "results.json"
        if not params_path.exists():
            continue
        params = json.loads(params_path.read_text())
        # A pre-registered run has params.json but no results.json until record.py runs.
        results = json.loads(results_path.read_text()) if results_path.exists() else {}
        budget = params["protocol"]["budget"]
        runs.append(
            {
                "id": params["id"],
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
    global results_dir
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=HERE.parent / "results")
    args = parser.parse_args(argv)
    results_dir = args.results_dir.resolve()
    if not results_dir.is_dir():
        parser.error(f"results directory does not exist: {results_dir}")
    print(f"Reading results from {results_dir}")
    print("http://autoresearch.localhost/  (direct: http://127.0.0.1:7102/)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
