"""Budget gate for conditional cells (orchestrator, 2026-11-27). Exit 0 if prior spend + this pod's hours x rate + the cell's
estimate stays at or below the stop-and-report line, else 9 (the cell is listed as not run). Pod clock from --start-utc.

    ledger_ok.py --start-utc 2026-11-27T15:13:17Z --rate 5.29 --prior 6.40 --cap 60 --cell-usd 4
"""

import argparse
import datetime as dt
import sys


def spend(start_utc: str, rate: float, prior: float, now: dt.datetime | None = None) -> float:
    t0 = dt.datetime.strptime(start_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.UTC)
    now = now or dt.datetime.now(dt.UTC)
    return prior + max(0.0, (now - t0).total_seconds() / 3600.0) * rate


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-utc", default="2026-11-27T15:13:17Z")
    ap.add_argument("--rate", type=float, default=5.29)
    ap.add_argument("--prior", type=float, default=6.40)
    ap.add_argument("--cap", type=float, default=60.0)
    ap.add_argument("--cell-usd", type=float, required=True)
    a = ap.parse_args(argv)
    s = spend(a.start_utc, a.rate, a.prior)
    ok = s + a.cell_usd <= a.cap
    print(
        f"LEDGER spend_now=${s:.2f} cell=${a.cell_usd:.2f} projected=${s + a.cell_usd:.2f} cap=${a.cap:.2f} -> {'RUN' if ok else 'SKIP'}"
    )
    return 0 if ok else 9


if __name__ == "__main__":
    sys.exit(main())
