"""tools/release_ok.py: release rules before the first adapter / steered cell (PREREG_rank1 Am. 4, PREREG_steer Am. 2)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import release_ok as R  # noqa: E402

PASS = {"verdict": "PASS"}


def test_lora_rules():
    assert R.lora_ok({"verdict_code": 0}, PASS)[0]
    assert not R.lora_ok({"verdict_code": 32}, PASS)[0]
    assert not R.lora_ok({"verdict_code": 30}, PASS)[0]
    assert R.lora_ok({"verdict_code": 33, "fallback": {"pass": True}}, PASS)[0]
    assert not R.lora_ok({"verdict_code": 33, "fallback": {"pass": False}}, PASS)[0]
    assert not R.lora_ok({"verdict_code": 34}, PASS)[0]
    assert R.lora_ok({"verdict_code": 34, "fallback": {"pass": True}}, PASS)[0]
    assert not R.lora_ok({"verdict_code": 0}, {"verdict": "NO_POWER"})[0]
    assert not R.lora_ok({"verdict_code": 0}, {})[0]


def test_steer_rules():
    assert R.steer_ok({"pass": True, "pass_chosen": True}, PASS)[0]
    assert R.steer_ok({"pass": False, "pass_chosen": True}, PASS)[0]
    assert not R.steer_ok({"pass": False, "pass_chosen": True}, PASS, need_scramble=True)[0]
    assert not R.steer_ok({"pass": False, "pass_chosen": False}, PASS)[0]
    assert not R.steer_ok({"pass": True}, PASS)[
        0
    ]  # an old-format report (no pass_chosen) does not release
    assert not R.steer_ok({"pass": True, "pass_chosen": True}, {"verdict": "FAIL"})[0]


def test_cli_codes(tmp_path):
    g, d = tmp_path / "g.json", tmp_path / "d.json"
    g.write_text(json.dumps({"verdict_code": 0}))
    d.write_text(json.dumps(PASS))
    assert R.main(["lora", "--gate", str(g), "--decode", str(d)]) == 0
    d.write_text(json.dumps({"verdict": "FAIL"}))
    assert R.main(["lora", "--gate", str(g), "--decode", str(d)]) == 7
    assert R.main(["lora", "--gate", str(tmp_path / "missing.json"), "--decode", str(d)]) == 7
