"""Release check before the first adapter or steered cell on the H200 (orchestrator, 2026-11-27; PREREG_rank1 Amendment 4,
PREREG_steer Amendment 2). Reads JSON only; exit 0 = release, 7 = do not release (reason printed).

    release_ok.py lora  --gate runs/gate_h/out/gate.json --decode runs/decode_lora/out/decode.json
    release_ok.py steer --preflight runs/m3_pf_h/out/preflight_report.json --decode runs/decode_steer/out/decode.json [--need-scramble]

lora:  gate verdict 0, or 33 / 34 with fallback.pass true (the pre-stated noise-floor mode); and the decode check PASS.
steer: pre-flight pass_chosen true (pass true with --need-scramble, for E2's scramble cells); and the decode check PASS.
"""

import argparse
import json
import sys


def lora_ok(gate: dict, dec: dict) -> tuple[bool, str]:
    code = gate.get("verdict_code")
    if code == 0:
        g = True
    elif code in (33, 34):
        g = bool((gate.get("fallback") or {}).get("pass"))
    else:
        g = False
    if not g:
        return (
            False,
            f"gate verdict {code} (fallback pass={(gate.get('fallback') or {}).get('pass')})",
        )
    if dec.get("verdict") != "PASS":
        return False, f"decode check {dec.get('verdict')}"
    return True, f"gate verdict {code}, decode PASS"


def steer_ok(pre: dict, dec: dict, need_scramble: bool = False) -> tuple[bool, str]:
    key = "pass" if need_scramble else "pass_chosen"
    if pre.get(key) is not True:
        return False, f"pre-flight {key} is {pre.get(key)!r}"
    if dec.get("verdict") != "PASS":
        return False, f"decode check {dec.get('verdict')}"
    return True, f"pre-flight {key}, decode PASS"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("lora")
    a1.add_argument("--gate", required=True)
    a1.add_argument("--decode", required=True)
    a2 = sub.add_parser("steer")
    a2.add_argument("--preflight", required=True)
    a2.add_argument("--decode", required=True)
    a2.add_argument("--need-scramble", action="store_true")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "lora":
            ok, why = lora_ok(json.load(open(a.gate)), json.load(open(a.decode)))
        else:
            ok, why = steer_ok(
                json.load(open(a.preflight)), json.load(open(a.decode)), a.need_scramble
            )
    except (OSError, ValueError) as e:
        ok, why = False, f"cannot read inputs: {e!r}"
    print(("RELEASE " if ok else "NO_RELEASE ") + why)
    return 0 if ok else 7


if __name__ == "__main__":
    sys.exit(main())
