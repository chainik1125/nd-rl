# ruff: noqa: E501
"""Amendment 9 (PREREG, 2026-11-28): the cells, the plan of the one adapter trained now, the budget decision that is
taken before training and before the engine starts, and the reading rule.

  a9.py plan    --out V2/plan_dpo_a9.json      write the training plan (refuses to change an existing file)
  a9.py plan    --check V2/plan_dpo_a9.json    exit 0 if the file is exactly this plan
  a9.py decide  --phase train|sample --elapsed-s N [--no-n16] [--ref] [--cap-usd 15] [--rate 5.29]
                                               which cells fit inside the money stop; prints JSON, last line CELLS=a,b
  a9.py reading --diff D --lo L --hi H         the reading of one paired difference (points)
  a9.py scan    --stage-dir DIR                refuse sealed-set or credential names in a staged push
  a9.py verify  [--v2 V2] [--ref]              the reused adapters are DONE, from our pool, with the recorded md5

CPU only, stdlib only. Nothing here talks to a pod.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

CK = "dpo"
RANK1 = "dpo__final__O1__n64"  # rank 1 on o_proj, 64 examples (stage 1, rp-om1): reused
N16 = "dpo__final__A16__n16"  # full size, 16 examples: trained on the pod
N4 = "dpo__final__A16__n4"  # full size, 4 examples (stage 1, rp-om1): reused
REF = "dpo__final__A16__n64"  # the reference: full size, 64 examples, same rule (stage 1, rp-om2)
BATCH = "batch_a9"
SEED_SAMPLE = 660009
PLAN_STEM = "plan_dpo_a9"
# where the existing adapters and the stored reference draws are (under V2)
SOURCES = {RANK1: "runs_rp-om1", N4: "runs_rp-om1", REF: "runs_rp-om2"}
REF_SAMPLES = "runs_rp-om2/dpo/sample/batch_b1/" + REF
WEIGHTS_MD5 = {  # adapter_model.safetensors, as fetched in stage 1
    RANK1: "a9f675ad1c2ed182a4c7a6b1b4e5cc13",
    N4: "84d456711ac825cb803345a5cc5e94b0",
    REF: "42facfb95430e81a4791d8903e077a78",
}

# ---- the plan: the stage-1 rule for a full-size cell, with the 16-example schedule of Amendment 9(a)
PLAN = {
    "cells": [
        {
            "cell": N16,
            "pool": "dpo/final",
            "family": "lora:r=16:targets=all",
            "n": 16,
            "lrs": [0.0001, 0.0003, 0.001],
            "steps": 64,
            "eval_steps": [4, 8, 16, 32, 64],
            "edge_extend": True,
        }
    ]
}
TRAIN_ARGS = "--ce-checkpoint --max-memory-gb 78"

# ---- budget (Amendment 9(h), 9(i)): planning numbers, all in seconds of this job's pod time
RATE_USD_H = 5.29
CAP_USD = 15.0
DRAWS = 384
TOKENS_PER_DRAW = 8000  # the reference's mean was 7,953
TOKENS_PER_S = 1450  # measured 1,505 to 1,544 on the H200
CELL_S = DRAWS * TOKENS_PER_DRAW / TOKENS_PER_S  # about 35 min 19 s
# upload of the adapters 3 min, gate 3 min, engine load and grading 4 min, the last fetch 5 min (9(j): 15, was 12)
SAMPLE_OVERHEAD_S = 15 * 60
FETCH_S = 5 * 60  # of the overhead: kept after sampling for the last fetch and its md5
TRAIN_S = (
    32 * 60
)  # the 16-example adapter: model load, up to four lrs at about 7 min each (expected 24 to 31 min)
TRAIN_XFER_S = 5 * 60  # planned: fetch of the new adapter before staging
TRAIN_FETCH_S = 3 * 60  # kept for that fetch when the driver sets how long it waits for training


def stop_s(cap_usd: float = CAP_USD, rate: float = RATE_USD_H) -> float:
    """Seconds of pod time the money stop allows."""
    return cap_usd / rate * 3600


def sample_plan_s(n_cells: int) -> float:
    return n_cells * CELL_S + SAMPLE_OVERHEAD_S


def decide(
    phase: str,
    elapsed_s: float,
    have_n16: bool = True,
    want_ref: bool = False,
    cap_usd: float = CAP_USD,
    rate: float = RATE_USD_H,
) -> dict:
    """Which cells are sampled, decided before the engine starts (never by cutting draws).

    Drop order when the plan does not fit inside the money stop: the optional re-sample of the reference first, then
    the 16-example cell; the rank-1 and 4-example cells go together or not at all. phase "train": the same test with
    the training time in front; if training plus three cells does not fit, the adapter is not trained.
    """
    if phase not in ("train", "sample"):
        raise ValueError(phase)
    limit = stop_s(cap_usd, rate)
    lead = TRAIN_S + TRAIN_XFER_S if phase == "train" else 0.0
    options = []
    if have_n16 and want_ref:
        options.append([RANK1, N16, N4, REF])
    if have_n16:
        options.append([RANK1, N16, N4])
    options.append([RANK1, N4])
    for k, cells in enumerate(options):
        # an option that needs the new adapter pays for its training (phase "train")
        need = (lead if N16 in cells else 0.0) + sample_plan_s(len(cells))
        if elapsed_s + need <= limit:
            full = options[0]
            return {
                "phase": phase,
                "cells": cells,
                "dropped": [c for c in full if c not in cells],
                "train_n16": bool(phase == "train" and N16 in cells),
                "planned_end_s": round(elapsed_s + need),
                "stop_s": round(limit),
                "sample_deadline_s": round(
                    limit - FETCH_S - elapsed_s - (lead if N16 in cells else 0)
                ),
                # how long the driver waits for the training stage: past this, three cells no longer fit, so the
                # stage is stopped and the two reused cells go on without the new adapter
                "train_deadline_s": round(limit - sample_plan_s(3) - TRAIN_FETCH_S - elapsed_s)
                if phase == "train" and N16 in cells
                else 0,
                "option": k,
            }
    return {
        "phase": phase,
        "cells": [],
        "dropped": options[0],
        "train_n16": False,
        "planned_end_s": None,
        "stop_s": round(limit),
        "sample_deadline_s": 0,
        "train_deadline_s": 0,
        "option": None,
    }


# ---- reading rule (Amendment 9(e)); arguments in points of all draws
HOLDS_DIFF, HOLDS_LO = -5.0, -12.0
SHORT_DIFF, SHORT_HI = -10.0, 0.0
READINGS = ("holds", "falls short", "unclear with 48 prompts")


def reading(diff: float, lo: float, hi: float) -> str:
    if diff > HOLDS_DIFF and lo > HOLDS_LO:
        return READINGS[0]
    if diff < SHORT_DIFF and hi < SHORT_HI:
        return READINGS[1]
    return READINGS[2]


# ---- what may not be copied to a pod
FORBIDDEN = re.compile(
    r"textbook72|tb72|test_13_20|test_21_30|(^|[^0-9])832([^0-9]|$)|sealed", re.I
)
CREDENTIAL = re.compile(
    r"(^|/)\.env|\.netrc|(^|/)id_(rsa|ecdsa|ed25519|runpod|ynez|dsa)|\.pem$|\.key$|credential|secret|api[_-]?key"
    r"|hf_token|(^|/)token\.(txt|json)$|(^|/)\.ssh/|authorized_keys",
    re.I,
)
SECRET_TEXT = re.compile(
    rb"hf_[A-Za-z0-9]{30,}|rpa_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9_-]{30,}|BEGIN [A-Z ]*PRIVATE KEY"
)


def scan(stage_dir: Path) -> list[str]:
    """Problems in a staged push: a file whose path names a sealed set or a credential, or whose text holds a token."""
    bad = []
    for p in sorted(stage_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(stage_dir))
        if FORBIDDEN.search(rel):
            bad.append(f"sealed-set name: {rel}")
        if CREDENTIAL.search(rel):
            bad.append(f"credential name: {rel}")
        if p.stat().st_size <= 64 << 20 and SECRET_TEXT.search(p.read_bytes()):
            bad.append(f"token-like text: {rel}")
    return bad


POOL_MANIFEST_SHA256 = (
    "12fc1d237b1c2be2d65d4fa65f5c74d404e541fc8dd333c6b9f42d57ec0b0560"  # v2/pools/dpo/final
)


def file_digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def verify(v2: Path, with_ref: bool = False) -> list[str]:
    """Problems with the adapters that are reused: not DONE, another pool, a selection that is not from the dev-loss
    rule, or weights that differ from the md5 recorded at the stage-1 fetch. Also the pool on disk."""
    bad = []
    man = v2 / "pools" / CK / "final" / "MANIFEST.json"
    if not man.exists() or file_digest(man, "sha256") != POOL_MANIFEST_SHA256:
        bad.append(f"pool manifest differs or is missing: {man}")
    for cell in [RANK1, N4] + ([REF] if with_ref else []):
        d = v2 / SOURCES[cell] / CK / "train" / "cells" / cell
        if not (d / "DONE").exists():
            bad.append(f"{cell}: no DONE under {d}")
            continue
        cj = json.loads((d / "cell.json").read_text())
        if cj.get("pool_manifest_sha256") != POOL_MANIFEST_SHA256:
            bad.append(f"{cell}: trained on another pool")
        if cj.get("selection", "dev loss") != "dev loss" or cj.get("recipe_from"):
            bad.append(f"{cell}: not selected by the dev-loss rule")
        if cj.get("did_not_train"):
            bad.append(f"{cell}: flagged did_not_train")
        if cj.get("reload_check_max_abs_diff") != 0.0:
            bad.append(f"{cell}: the saved adapter does not reproduce the selected scores")
        got = file_digest(d / "adapter" / "adapter_model.safetensors", "md5")
        if got != WEIGHTS_MD5[cell]:
            bad.append(f"{cell}: weights md5 {got}, recorded {WEIGHTS_MD5[cell]}")
    return bad


def plan_text() -> str:
    return json.dumps(PLAN, indent=1) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--out", type=Path)
    p.add_argument("--check", type=Path)
    d = sub.add_parser("decide")
    d.add_argument("--phase", choices=["train", "sample"], required=True)
    d.add_argument("--elapsed-s", type=float, required=True)
    d.add_argument("--no-n16", action="store_true")
    d.add_argument("--ref", action="store_true")
    d.add_argument("--cap-usd", type=float, default=CAP_USD)
    d.add_argument("--rate", type=float, default=RATE_USD_H)
    r = sub.add_parser("reading")
    for k in ("--diff", "--lo", "--hi"):
        r.add_argument(k, type=float, required=True)
    s = sub.add_parser("scan")
    s.add_argument("--stage-dir", type=Path, required=True)
    v = sub.add_parser("verify")
    v.add_argument("--v2", type=Path, default=Path("/scratch/cowifornia/nd-rl/olmo_elicit/v2"))
    v.add_argument("--ref", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        if a.check:
            ok = a.check.exists() and a.check.read_text() == plan_text()
            print("PLAN_OK" if ok else f"PLAN_DIFFERS {a.check}")
            return 0 if ok else 1
        if not a.out:
            print(plan_text(), end="")
            return 0
        if a.out.exists() and a.out.read_text() != plan_text():
            print(f"{a.out} exists with other content: not overwritten")
            return 1
        a.out.write_text(plan_text())
        print(f"wrote {a.out}")
        return 0
    if a.cmd == "decide":
        res = decide(a.phase, a.elapsed_s, not a.no_n16, a.ref, a.cap_usd, a.rate)
        print(json.dumps(res))
        print("CELLS=" + ",".join(res["cells"]))
        return 0 if res["cells"] else 3
    if a.cmd == "reading":
        print(reading(a.diff, a.lo, a.hi))
        return 0
    if a.cmd == "verify":
        bad = verify(a.v2, a.ref)
        for b in bad:
            print("PROBLEM", b)
        print("VERIFY_OK" if not bad else f"VERIFY_FAILED {len(bad)}")
        return 0 if not bad else 1
    bad = scan(a.stage_dir)
    for b in bad:
        print("REFUSED", b)
    print("SCAN_OK" if not bad else f"SCAN_FAILED {len(bad)}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
