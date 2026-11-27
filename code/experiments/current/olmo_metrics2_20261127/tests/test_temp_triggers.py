import sys
from pathlib import Path

X = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(X / "tools"))
import temp_common as tc  # noqa: E402
import temp_triggers as tt  # noqa: E402

GOOD_SIDE = {
    "status": "done",
    "temperature": 1.0,
    "top_p": 1.0,
    "dan_argv": ["--max_tokens", "32768"],
    "template_sha256": "T",
    "engine_config_sha256": "E",
    "dtype": "bf16",
    "enforce_eager": False,
}
GOOD_PRE = {
    "checks": {"zero_identity": True, "zero_compiled_within_tol": True},
    "pass": True,
    "template_sha256": "T",
    "engine_config_sha256": "E",
    "dtype": "bf16",
    "enforce_eager": False,
}


def rows(pid, passes):
    return [
        {"pid": pid, "chunk": i, "i": 0, "strict": int(s), "n_tok": 100, "finish": "stop"}
        for i, s in enumerate(passes)
    ]


def test_e0_accept_ok_and_each_failure():
    ok, f, u = tt.e0_accept(GOOD_SIDE, GOOD_PRE, "T", grade_rows=384)
    assert ok and not f and not u
    for mut, key in (
        (dict(GOOD_SIDE, temperature=0.6), "T/top_p"),
        (dict(GOOD_SIDE, status="started"), "status"),
        (dict(GOOD_SIDE, dan_argv=["--lora", "x"]), "lora"),
        (dict(GOOD_SIDE, dan_argv=["--max_tokens", "1000"]), "max_tokens"),
        (dict(GOOD_SIDE, template_sha256="X"), "template"),
    ):
        ok, f, _ = tt.e0_accept(mut, GOOD_PRE, "T", grade_rows=384)
        assert not ok and any(key in x for x in f), (key, f)
    ok, f, _ = tt.e0_accept(GOOD_SIDE, {"checks": {"zero_identity": False}}, "T", grade_rows=384)
    assert not ok
    # E0' samples compiled: the compiled zero check and a pass flag are required
    ok, f, _ = tt.e0_accept(
        GOOD_SIDE, dict(GOOD_PRE, checks={"zero_identity": True}), "T", grade_rows=384
    )
    assert not ok and any("zero_compiled_within_tol" in x for x in f)
    ok, f, _ = tt.e0_accept(
        GOOD_SIDE,
        dict(GOOD_PRE, checks={"zero_identity": True, "zero_compiled_within_tol": False}),
        "T",
        grade_rows=384,
    )
    assert not ok
    ok, f, _ = tt.e0_accept(
        GOOD_SIDE, dict(GOOD_PRE, **{"pass": False, "pass_chosen": False}), "T", grade_rows=384
    )
    assert not ok and any("pass_chosen" in x for x in f)
    ok, f, _ = tt.e0_accept(
        GOOD_SIDE, dict(GOOD_PRE, **{"pass": False, "pass_chosen": True}), "T", grade_rows=384
    )
    assert ok and not f
    ok, f, _ = tt.e0_accept(
        GOOD_SIDE, {k: v for k, v in GOOD_PRE.items() if k != "pass"}, "T", grade_rows=384
    )
    assert not ok
    ok, f, _ = tt.e0_accept(GOOD_SIDE, GOOD_PRE, "T", grade_rows=100)
    assert not ok and any("graded rows" in x for x in f)
    assert not tt.e0_accept(None, GOOD_PRE, "T")[0]


def test_e0_unverified_needs_flag():
    side = {
        k: v for k, v in GOOD_SIDE.items() if k not in ("engine_config_sha256", "template_sha256")
    }
    pre = {"checks": {"zero_identity": True, "zero_compiled_within_tol": True}, "pass": True}
    ok, f, u = tt.e0_accept(side, pre, "T", allow_unverified=False)
    assert not ok and u
    ok, f, u = tt.e0_accept(side, pre, "T", allow_unverified=True)
    assert ok and u


def test_reach_rule_k8_with_e0():
    c1b = {
        f"p{i}": rows(
            f"p{i}",
            [1, 1, 0, 0, 0, 0, 0, 0] if i == 0 else [1, 0, 0, 0, 0, 0, 0, 0] if i == 1 else [0] * 8,
        )
        for i in range(4)
    }
    e0 = {f"p{i}": rows(f"p{i}", [0] * 8) for i in range(4)}
    r = tt.reach_rule(c1b, 8, e0)
    assert r["c1b"]["passes"] == 3 and r["c1b"]["prompts_with_pass"] == 2 and r["fires"]
    e0b = dict(e0, p0=rows("p0", [1, 0, 0, 0, 0, 0, 0, 0]))
    assert not tt.reach_rule(c1b, 8, e0b)["fires"]  # difference 2 < 3
    one_prompt = {"p0": rows("p0", [1, 1, 1, 0, 0, 0, 0, 0])}
    assert not tt.reach_rule(one_prompt, 8, {"p0": rows("p0", [0] * 8)})[
        "fires"
    ]  # 3 passes but 1 prompt


def test_reach_rule_k4_uses_first_k_of_e0_and_proxy():
    c1b = {"a": rows("a", [1, 0, 0, 0]), "b": rows("b", [1, 0, 0, 0])}
    e0 = {"a": rows("a", [0, 0, 0, 0, 1, 1, 1, 1]), "b": rows("b", [0] * 8)}
    r = tt.reach_rule(c1b, 4, e0)
    assert (
        r["e0"]["draws"] == 8 and r["fires"]
    )  # first 4 of E0' only: E0' draws are cut to k, passes 0
    assert (
        tt.reach_rule(c1b, 4, None)["baseline"] == "proxy" and tt.reach_rule(c1b, 4, None)["fires"]
    )
    c1b1 = {"a": rows("a", [1, 1, 0, 0])}
    assert not tt.reach_rule(c1b1, 4, None)["fires"]  # 2 passes but only 1 prompt


def test_trigger_b_margin():
    texts = {}
    rs = []
    for i in range(100):
        r = {"pid": f"p{i}", "chunk": 0, "i": 0, "n_tok": 100, "finish": "stop", "strict": 0}
        rs.append(r)
        texts[(f"p{i}", 0, 0)] = (
            "Same sentence repeated here again. " * 12 if i < 4 else "fine text"
        ) + "</think>"
    stored = {"dpo": {"share_pct": 1.0, "cut_pct": 0.06}}
    b = tt.trigger_b(rs, texts, stored)
    assert b["repeat_share_pct"] == 4.0 and b["condition"]
    stored2 = {"dpo": {"share_pct": 3.0, "cut_pct": 0.06}}
    assert not tt.trigger_b(rs, texts, stored2)["condition"]


def test_paired_delta_null_and_effect():
    pids = tc.dc.arm_a()[:60]
    base = {p: [{"strict": 0, "n_tok": 1, "finish": "stop"}] * 4 for p in pids}
    same = {p: rows(p, [0, 0]) for p in pids}
    d = tt.paired_delta(same, base)
    assert d["delta_pts"] == 0 and not d["excludes_zero"]
    up = {p: rows(p, [1, 1]) if i % 2 == 0 else rows(p, [0, 0]) for i, p in enumerate(pids)}
    d = tt.paired_delta(up, base)
    assert abs(d["delta_pts"] - 50) < 1e-9 and d["excludes_zero"]


def test_nothing_is_decided_or_released_automatically():
    # the verdict dict says so, and the module has no subprocess / ssh use
    src = (X / "tools" / "temp_triggers.py").read_text()
    assert "subprocess" not in src and "ssh" not in src and "m2_pod" not in src
