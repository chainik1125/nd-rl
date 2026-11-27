import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import ledger_ok as L  # noqa: E402


def test_spend_and_gate():
    now = dt.datetime(2026, 11, 27, 17, 13, 17, tzinfo=dt.UTC)
    assert abs(L.spend("2026-11-27T15:13:17Z", 5.29, 6.40, now) - (6.40 + 2 * 5.29)) < 1e-9
    assert L.main(["--start-utc", "2000-01-01T00:00:00Z", "--cell-usd", "1"]) == 9
    assert L.main(["--start-utc", "2999-01-01T00:00:00Z", "--cell-usd", "1"]) == 0
