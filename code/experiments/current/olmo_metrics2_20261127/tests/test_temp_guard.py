import subprocess
import sys
import time
from pathlib import Path

X = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(X / "tools"))
import temp_inflight_guard as g  # noqa: E402

V1 = (
    "INFO 11-27 02:00:00 [loggers.py:127] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation "
    "throughput: 1500.5 tokens/s, Running: 256 reqs, Waiting: 0 reqs, GPU KV cache usage: 41.2%, Prefix cache hit rate: 30.0%"
)
OLD = "INFO metrics.py:345] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 800.0 tokens/s, Running: 10 reqs, Swapped: 0 reqs, Pending: 3 reqs"


def test_parse_v1_line():
    assert g.parse_stats(V1) == (1500.5, 256, 0)


def test_parse_tolerates_missing_waiting_and_garbage():
    assert g.parse_stats(OLD) == (800.0, 10, None)
    assert g.parse_stats("INFO nothing here") is None
    assert g.parse_stats("Running: 3 reqs") is None


def new(**kw):
    return g.Guard(plan_draws=100, plan_tokens=100 * 5000, **kw)


def test_rule_a_total_tokens():
    gd = new()
    t = 0.0
    assert gd.step(t, [], 0, 0) == []
    lines = ["Avg generation throughput: 1000.0 tokens/s, Running: 50 reqs, Waiting: 0 reqs"] * 10
    # 10 stat lines in a 1000 s poll -> G = 1000 * 1000 = 1.0e6 > 1.5 * 5.0e5
    out = gd.step(1000.0, lines, 10, 40000)
    assert any(r.startswith("(a)") for r in out)


def test_rule_a_not_after_90_percent_rows():
    gd = new()
    gd.step(0.0, [], 0, 0)
    lines = ["Avg generation throughput: 1000.0 tokens/s, Running: 5 reqs, Waiting: 0 reqs"] * 10
    out = gd.step(1000.0, lines, 95, 400000)
    assert not any(r.startswith("(a)") for r in out)


def test_rule_b_inflight_length():
    gd = new()
    gd.step(0.0, [], 0, 0)
    lines = ["Avg generation throughput: 100.0 tokens/s, Running: 20 reqs, Waiting: 0 reqs"] * 4
    # G = 100 * 4000 = 4.0e5 ; finished 20 rows * 1000 tok = 2.0e4 ; in flight 3.8e5 / 20 = 19000 >= 16384
    out = gd.step(4000.0, lines, 20, 20000)
    assert any(r.startswith("(b)") for r in out)


def test_rule_b_needs_waiting_zero_and_enough_running():
    gd = new()
    gd.step(0.0, [], 0, 0)
    out = gd.step(
        4000.0,
        ["Avg generation throughput: 100.0 tokens/s, Running: 20 reqs, Waiting: 5 reqs"] * 4,
        20,
        20000,
    )
    assert not any(r.startswith("(b)") for r in out)
    gd2 = new()
    gd2.step(0.0, [], 0, 0)
    out = gd2.step(
        4000.0,
        ["Avg generation throughput: 100.0 tokens/s, Running: 5 reqs, Waiting: 0 reqs"] * 4,
        20,
        20000,
    )
    assert not any(r.startswith("(b)") for r in out)


def test_rule_c_stall_with_first_row_grace():
    tick = ["Avg generation throughput: 1.0 tokens/s, Running: 5 reqs, Waiting: 0 reqs"]
    gd = new()
    gd.step(0.0, [], 0, 0)
    assert gd.step(21 * 60, tick, 0, 0) == []  # no row yet: grace of 40 min
    assert any(r.startswith("(c)") for r in gd.step(41 * 60, tick, 0, 0))
    gd2 = new()
    gd2.step(0.0, [], 0, 0)
    gd2.step(60.0, tick, 3, 15000)
    assert any(r.startswith("(c)") for r in gd2.step(60.0 + 21 * 60, tick, 3, 15000))


def test_blind_wall_clock_fallback():
    gd = new(blind_after_s=900.0, load_allow_s=600.0, assumed_tps=1500.0)
    gd.step(0.0, [], 0, 0)
    # cap = 1.5 * 5e5 / 1500 = 500 s of generation, so blind abort at > 600 + 500 s once blind (after 900 s)
    assert gd.step(1000.0, ["unparseable line"], 10, 1000) == []
    out = gd.step(1200.0, ["still nothing"], 10, 1000)
    assert any(r.startswith("(d)") for r in out)


def test_blind_never_uses_rules_a_b():
    gd = new(blind_after_s=900.0)
    gd.step(0.0, [], 0, 0)
    out = gd.step(1000.0, ["x"], 0, 0)
    assert not any(r.startswith(("(a)", "(b)")) for r in out)


def test_read_rows_tolerates_truncated_gzip(tmp_path):
    import gzip
    import json

    p = tmp_path / "s.jsonl.gz"
    with gzip.open(p, "wt") as f:
        for i in range(5):
            f.write(json.dumps({"n_tok": 100 + i}) + "\n")
    assert g.read_rows(str(p)) == (5, 510)
    raw = p.read_bytes()
    p.write_bytes(raw[:-8])  # cut the gzip trailer
    n, tok = g.read_rows(str(p))
    assert n <= 5 and tok == sum(100 + i for i in range(n))
    assert g.read_rows(str(tmp_path / "none.gz")) == (0, 0)


def test_end_to_end_terminates_the_sampler(tmp_path):
    log = tmp_path / "job.log"
    samples = tmp_path / "samples.jsonl.gz"
    log.write_text("")
    sampler = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    guard = subprocess.Popen(
        [
            sys.executable,
            str(X / "tools" / "temp_inflight_guard.py"),
            "--log",
            str(log),
            "--samples",
            str(samples),
            "--plan-draws",
            "10",
            "--plan-tokens-per-draw",
            "10",
            "--target-pid",
            str(sampler.pid),
            "--poll",
            "1",
        ]
    )
    try:
        time.sleep(2.5)
        with open(log, "a") as f:
            f.write("\n".join([V1] * 3) + "\n")
        rc = guard.wait(timeout=60)
        assert rc == 3
        assert sampler.wait(timeout=30) is not None  # SIGTERM reached it
        assert "GUARD_ABORT" in (tmp_path / "job.log.guard").read_text()
    finally:
        for p in (guard, sampler):
            if p.poll() is None:
                p.kill()
