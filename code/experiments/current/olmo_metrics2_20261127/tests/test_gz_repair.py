import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import gz_repair as G  # noqa: E402


def test_truncated_stream_keeps_complete_lines(tmp_path):
    full = tmp_path / "full.gz"
    with gzip.open(full, "wt") as f:
        for i in range(50):
            f.write(json.dumps({"i": i, "text": "x" * 200}) + "\n")
    raw = full.read_bytes()
    cut = tmp_path / "cut.gz"
    cut.write_bytes(raw[: len(raw) - 30])  # no end marker, last line possibly partial
    r = G.repair(str(cut), str(tmp_path / "out.gz"))
    rows = [json.loads(ln) for ln in gzip.open(tmp_path / "out.gz", "rt")]
    assert not r["complete_stream"] and r["rows"] == len(rows) and 1 <= len(rows) < 50
    assert [x["i"] for x in rows] == list(range(len(rows)))
    r2 = G.repair(str(full), str(tmp_path / "out2.gz"))
    assert r2 == {"rows": 50, "dropped_partial": 0, "complete_stream": True}
