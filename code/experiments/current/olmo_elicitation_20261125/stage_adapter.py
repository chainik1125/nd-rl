# ruff: noqa: E501
"""Serrano side of stage L: check a checkpoint's stage-1 adapter and stage it for the pod.

  stage_adapter.py <ckpt> <dest dir> [--runs .../runs] [--grid 1e-4,3e-4,1e-3]

Takes runs/<ckpt>/edl/adapters/best_order0 only when
  - runs/<ckpt>/edl/DONE lists every grid lr with status "ok" for order 0 (and for every other order it lists for
    those lrs), and
  - BEST.json names the order-0 row with the shortest online code length among the grid's order-0 rows in
    edl_rows.jsonl (lr equal, code length equal to 1e-6 relative), and
  - adapter_config.json is PEFT LoRA r 16, alpha 32, Dan's seven target modules (the configuration his vLLM 0.11.0
    LoRA evaluations loaded: olmo-ifladder Leval_main).
Writes <dest>/best_order0/{adapter_config.json, adapter_model.safetensors} (hard links when possible; the optimizer
state is not copied) and <dest>/source.json: the BEST.json and DONE contents, the chosen row's identity, md5 and
sha256 of the two adapter files, and the check examples for lora_check.py (the first two test prompts of the chosen
row with their PEFT nats at n' = 0 and 64). Exit 0 when staged; 1 with a reason otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

RUNS = Path("/scratch/cowifornia/nd-rl/olmo_elicit/runs")
GRID = (1e-4, 3e-4, 1e-3)
TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
FILES = ("adapter_config.json", "adapter_model.safetensors")


def digest(p: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def same_lr(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-9 * max(abs(a), abs(b))


def code_length(r: dict) -> float:
    return sum(r["online"]["think_nats"]) + sum(r["online"]["answer_nats"])


def check(edl: Path, grid: tuple[float, ...]) -> tuple[dict, dict, dict]:
    """(DONE, BEST, chosen row); SystemExit(reason) when the adapter may not be used."""
    if not (edl / "DONE").exists():
        raise SystemExit(f"no {edl}/DONE: the EDL run has not finished")
    done = json.loads((edl / "DONE").read_text())
    st = {}
    for k, v in done.get("status", {}).items():
        lr, order = k.split("/")
        st[(float(lr), int(order))] = v
    for lr in grid:
        ent = {o: v for (x, o), v in st.items() if same_lr(x, lr)}
        if 0 not in ent:
            raise SystemExit(f"DONE lacks lr {lr:g} order 0 (has {sorted(done.get('status', {}))})")
        bad = [f"{lr:g}/{o}" for o, v in sorted(ent.items()) if v != "ok"]
        if bad:
            raise SystemExit(f"DONE status not ok for {bad}")
    best = json.loads((edl / "adapters" / "BEST.json").read_text())
    rows = [
        json.loads(ln) for ln in (edl / "edl_rows.jsonl").read_text().splitlines() if ln.strip()
    ]
    cand = [
        r
        for r in rows
        if str(r.get("schema", "")).startswith("olmo_edl_row")
        and int(r["order"]) == 0
        and any(same_lr(float(r["lr"]), g) for g in grid)
        and r.get("status") == "ok"
        and "64" in r.get("evals", {})
    ]
    if not cand:
        raise SystemExit("no finished order-0 grid row in edl_rows.jsonl")
    win = min(cand, key=code_length)
    if (
        int(best.get("order", -1)) != 0
        or not same_lr(float(best["lr"]), float(win["lr"]))
        or abs(float(best["code_length"]) - code_length(win)) > 1e-6 * code_length(win)
    ):
        raise SystemExit(
            f"BEST.json (lr {best.get('lr')}, code {best.get('code_length')}) is not the shortest order-0 "
            f"grid row (lr {win['lr']}, code {code_length(win):.3f})"
        )
    return done, best, win


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("dest")
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--grid", default=",".join(f"{x:g}" for x in GRID))
    a = ap.parse_args()
    edl = Path(a.runs) / a.ckpt / "edl"
    grid = tuple(float(x) for x in a.grid.split(","))
    try:
        done, best, win = check(edl, grid)
    except SystemExit as e:
        print(f"ADAPTER_REFUSED {a.ckpt}: {e}")
        return 1
    src = edl / "adapters" / "best_order0"
    cfg = json.loads((src / "adapter_config.json").read_text())
    if not (
        cfg.get("peft_type") == "LORA"
        and cfg.get("r") == 16
        and cfg.get("lora_alpha") == 32
        and set(cfg.get("target_modules", [])) == TARGETS
        and not cfg.get("use_dora")
        and not cfg.get("modules_to_save")
    ):
        print(
            f"ADAPTER_REFUSED {a.ckpt}: adapter_config is not r16 / alpha 32 / Dan's 7 targets: {cfg}"
        )
        return 1
    dest = Path(a.dest)
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "best_order0").mkdir(parents=True)
    files = {}
    for f in FILES:
        try:
            os.link(src / f, dest / "best_order0" / f)
        except OSError:
            shutil.copy2(src / f, dest / "best_order0" / f)
        files[f] = {
            "bytes": (src / f).stat().st_size,
            "md5": digest(src / f, "md5"),
            "sha256": digest(src / f, "sha256"),
        }
    ev0, ev64 = win["evals"]["0"]["test"], win["evals"]["64"]["test"]
    pids = win["test"]["pids"]
    examples = [
        {
            "pid": pids[k],
            "peft0_think": ev0["think_nats"][k],
            "peft0_answer": ev0["answer_nats"][k],
            "peft64_think": ev64["think_nats"][k],
            "peft64_answer": ev64["answer_nats"][k],
            "n_think": win["test"]["n_think"][k],
            "n_answer": win["test"]["n_answer"][k],
        }
        for k in range(2)
    ]
    out = {
        "ckpt": a.ckpt,
        "source_dir": str(src),
        "grid": list(grid),
        "DONE": done,
        "BEST": best,
        "row": {
            "lr": win["lr"],
            "order": win["order"],
            "code_length": code_length(win),
            "utc_end": win.get("utc_end"),
            "pool_manifest_sha256": win.get("pool_manifest_sha256"),
            "n": win.get("n"),
        },
        "files": files,
        "check_examples": examples,
        "lora_name": f"{a.ckpt}_edl64",
    }
    (dest / "source.json").write_text(json.dumps(out, indent=1) + "\n")
    print(
        f"ADAPTER_STAGED {a.ckpt}: lr {win['lr']:g} order 0 code {code_length(win):.1f} nats; "
        f"adapter_model md5 {files['adapter_model.safetensors']['md5']} -> {dest}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
