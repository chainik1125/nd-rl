# ruff: noqa: E501
"""Round-5 reader check (PREREG Amendment 9(c) to 9(e) and 9(h)): round 4's protocol on the new cells and on the
reference, with 100 round-4 items mixed in as a drift check.

Builds on handcheck4.py (unchanged, so round 4 stays reproducible): the same screen, item files, batch prompts, read
validation, majority label, per-draw arrays and prompts -> draws bootstrap. What round 5 adds: its own cells and seeds,
the drift items and their test, the paired differences against the reference with the reading rule of a9.py, each
cell's rate as a share of the final model's rate, and bounds for a cell whose sampling was cut. After the audit of
Amendment 9 (PREREG 9(j)): every round-4 item of the final model and of the one-pass adapter is read again, so that the
share and the second reference compare labels of one round.

  handcheck5.py prep   --out V2/handcheck/round5 --items-dir V2/hc5_read --batch-dir V2/runs_<alias>/dpo/sample/batch_a9
                       [--no-reread]   only the 100 drift items from round 4, not all of the two round-4 cells
  handcheck5.py status --out ... --items-dir ... [--tags a00 ta00 ...] [--quarantine]
  handcheck5.py third  --out ... --items-dir ... --kind primary|task
  handcheck5.py table  --out ... --items-dir ...
  handcheck5.py spot   --out ... --items-dir ...
  handcheck5.py salvage --batch-dir <fetched batch_a9> --dest <new dir>   only if sampling was cut (RUNBOOK_A9.md)

The key (item -> cell, draw), the screen, the cell list and the batch manifest are written only under --out.
--items-dir holds item files, batch prompts and reads, and nothing that names a cell.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import a9
import handcheck as hc
import handcheck3 as h3
import handcheck4 as h4
import numpy as np

V2 = h3.V2
ROUND4 = V2 / "handcheck" / "round4"
# name in the key and the summary -> cell id inside the sampled batch (the reference's stored draws are fixed)
NEW = {"rank1_n64": a9.RANK1, "full_n16": a9.N16, "full_n4": a9.N4}
REF, REF_AGAIN = "ref", "ref_again"
R4_CELLS = {
    "final_r4": "s2300",
    "onepass_r4": "D",
}  # read in round 4: the yardstick and the second reference
R5_CELLS = {
    "final_r5": "s2300",
    "onepass_r5": "D",
}  # the same draws, every round-4 item read again in round 5
PLAIN = {
    "rank1_n64": "rank-1 adapter, 64 examples",
    "full_n16": "full-size adapter, 16 examples",
    "full_n4": "full-size adapter, 4 examples",
    REF: "reference: full-size adapter, 64 examples (stored draws)",
    REF_AGAIN: "reference adapter, sampled again in the new engine",
    "final_r4": "final RL model, first 8 stored draws (read in round 4)",
    "onepass_r4": "our data, our seed, one pass (read in round 4)",
    "final_r5": "final RL model, first 8 stored draws (read again in round 5)",
    "onepass_r5": "our data, our seed, one pass (read again in round 5)",
}
SEED_ITEMS, SEED_B, SEED_C, SEED_BOOT = 660051, 660052, 660053, 660054
SEED_TA, SEED_TB, SEED_TC, SEED_SPOT, SEED_DRIFT = 660055, 660056, 660057, 660058, 660059
SEEDS = {"primary": (SEED_ITEMS, SEED_B, SEED_C), "task": (SEED_TA, SEED_TB, SEED_TC)}
N_DRIFT, N_DRAWS, CHUNK, REPS = 100, 8, 4, 10000
RATES = ("strict_rate", "genuine_rate", "task_done_rate")
KEY_COLS = ["item", "cell", "pid", "chunk", "i", "r4_item", "uniform100"]


# ---- cells
def read_rows(path: Path) -> list[dict]:
    """Rows of a samples file; a file cut by a kill gives the rows before the cut (cap_bounds.read_rows)."""
    rows: list[dict] = []
    try:
        with gzip.open(path, "rt") as fh:
            for ln in fh:
                try:
                    rows.append(json.loads(ln))
                except json.JSONDecodeError:
                    break
    except (EOFError, OSError):
        pass
    return rows


def load_dir(samples: Path, graded: Path) -> dict:
    """{(pid, chunk, i): {text, strict, per}} on the reach prompts, complete chunks only (handcheck.load_cell's
    fields). A chunk is one request of 4 draws; a chunk with another row count was cut while being written."""
    reach = {ln.strip() for ln in hc.REACH.read_text().splitlines() if ln.strip()}
    g = {}
    for ln in graded.read_text().splitlines():
        r = json.loads(ln)
        if r["pid"] in reach:
            g[(r["pid"], int(r["chunk"]), int(r["i"]))] = r
    out = {}
    for s in read_rows(samples):
        k = (s["pid"], int(s["chunk"]), int(s["i"]))
        if k in g:
            out[k] = {
                "text": s["text"],
                "strict": hc.truthy(g[k]["strict"]),
                "per": g[k].get("per"),
            }
    per_chunk = Counter((k[0], k[1]) for k in out)
    return {k: d for k, d in out.items() if per_chunk[(k[0], k[1])] == CHUNK}


def find_cells(batch_dir: Path, ref_dir: Path) -> dict[str, dict]:
    """The round-5 cells that have graded draws: {name: {"samples", "graded"}}. The stored reference is required."""
    cells = {}
    for name, cid in {**NEW, REF_AGAIN: a9.REF}.items():
        d = batch_dir / cid
        if (d / "samples.jsonl.gz").exists() and not (d / "graded.jsonl").exists():
            raise SystemExit(
                f"{d} has draws and no grading (was sampling cut?). Salvage and grade it first: "
                "RUNBOOK_A9.md, 'If sampling was cut'. It is not dropped silently."
            )
        if (d / "samples.jsonl.gz").exists() and (d / "graded.jsonl").exists():
            cells[name] = {
                "samples": str(d / "samples.jsonl.gz"),
                "graded": str(d / "graded.jsonl"),
            }
    if not (ref_dir / "graded.jsonl").exists():
        raise SystemExit(f"the stored reference is missing: {ref_dir}")
    cells[REF] = {
        "samples": str(ref_dir / "samples.jsonl.gz"),
        "graded": str(ref_dir / "graded.jsonl"),
    }
    order = [*NEW, REF, REF_AGAIN]
    return {n: cells[n] for n in order if n in cells}


def cmd_salvage(batch_dir: Path, dest: Path) -> dict:
    """After a cut: a copy of the batch in a NEW directory that the grader can read. A cell that the pod graded in full
    is linked as it is. For any other cell the whole requests (4 rows per prompt and chunk) are written to a fresh
    samples file with no graded file; the rows of a request that was being written at the cut are left out. The
    fetched batch itself is not changed. Returns, per cell, what was kept."""
    if dest.exists():
        raise SystemExit(f"{dest} exists; salvage writes a new directory")
    info = {}
    for d in sorted(p for p in batch_dir.iterdir() if (p / "samples.jsonl.gz").exists()):
        rows = read_rows(d / "samples.jsonl.gz")
        per_chunk = Counter((r["pid"], int(r["chunk"])) for r in rows)
        keep = [r for r in rows if per_chunk[(r["pid"], int(r["chunk"]))] == CHUNK]
        graded = (
            (d / "graded.jsonl").read_text().splitlines() if (d / "graded.jsonl").exists() else []
        )
        whole = len(graded) == len(rows) == len(keep) > 0
        (dest / d.name).parent.mkdir(parents=True, exist_ok=True)
        if whole:
            (dest / d.name).symlink_to(d.resolve())
        else:
            (dest / d.name).mkdir()
            with gzip.open(dest / d.name / "samples.jsonl.gz", "wt") as f:
                f.writelines(json.dumps(r) + "\n" for r in keep)
        info[d.name] = {"rows_read": len(rows), "rows_kept": len(keep), "rows_of_cut_requests": len(rows) - len(keep),
                        "graded_on_the_pod": whole, "to_grade": not whole and bool(keep)}  # fmt: skip
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "salvage.json").write_text(
        json.dumps({"from": str(batch_dir), "cells": info}, indent=1) + "\n"
    )
    print(json.dumps(info, indent=1))
    return info


def load_cells(out: Path) -> dict[str, dict]:
    spec = json.loads((out / "cells.json").read_text())["cells"]
    return {n: load_dir(Path(c["samples"]), Path(c["graded"])) for n, c in spec.items()}


def item_id(cell: str, kk: tuple) -> str:
    return "h" + hashlib.sha1(f"{SEED_ITEMS}|{cell}|{kk}".encode()).hexdigest()[:7]


def drift_pick(round4: Path = ROUND4, n: int = N_DRIFT) -> list[dict]:
    """n of round 4's items, uniformly without replacement (seed 660059), as round-4 key rows."""
    rows = sorted(
        csv.DictReader((round4 / "key.csv").read_text().splitlines()), key=lambda r: r["item"]
    )
    return random.Random(SEED_DRIFT).sample(rows, min(n, len(rows)))


# ---- prep
def cmd_prep(
    out: Path,
    items_dir: Path,
    batch_dir: Path,
    ref_dir: Path,
    round4: Path = ROUND4,
    reread: bool = True,
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    for d in ("items", "task_items", "batches", "reads"):
        (items_dir / d).mkdir(parents=True, exist_ok=True)
    if any((items_dir / "items").iterdir()) or (out / "key.csv").exists():
        raise SystemExit("round-5 items exist already: refusing to rebuild")
    spec = find_cells(batch_dir, ref_dir)
    loaded = {n: load_dir(Path(c["samples"]), Path(c["graded"])) for n, c in spec.items()}
    spec = {
        n: c for n, c in spec.items() if loaded[n]
    }  # a cell without one finished chunk counts as not sampled
    if REF not in spec:
        raise SystemExit("the stored reference has no draws on the reach prompts")
    missing = [n for n in NEW if n not in spec]
    (out / "cells.json").write_text(
        json.dumps({"cells": spec, "not_sampled": missing, "batch_dir": str(batch_dir),
                    "reread": sorted(set(R5_CELLS.values())) if reread else []}, indent=1) + "\n"
    )  # fmt: skip
    P = hc.problems()
    screen_rows, summ, key_rows = [], [], []
    for cell in spec:
        draws = loaded[cell]
        passes = [(kk, d) for kk, d in sorted(draws.items()) if d["strict"]]
        n_scr = 0
        for kk, d in passes:
            why = hc.screen_reason(P[kk[0]], hc.answer_of(d["text"]))
            screen_rows.append({"cell": cell, "pid": kk[0], "chunk": kk[1], "i": kk[2], "screen": why or ""})  # fmt: skip
            if why:
                n_scr += 1
                continue
            item = item_id(cell, kk)
            hc.write_item(items_dir / "items" / f"{item}.md", item, P[kk[0]], d)
            h4.write_task_item(items_dir / "task_items" / f"{item}.md", item, P[kk[0]], d)
            key_rows.append({"item": item, "cell": cell, "pid": kk[0], "chunk": kk[1], "i": kk[2], "r4_item": "",
                             "uniform100": ""})  # fmt: skip
        summ.append({"cell": cell, "draws": len(draws), "passes": len(passes), "screened_gamed": n_scr})  # fmt: skip
        print(
            f"{cell}: draws {len(draws)}, passes {len(passes)}, rule-1 screened {n_scr}, items {len(passes) - n_scr}"
        )
    r4_draws: dict[str, dict] = {}
    picked = drift_pick(
        round4
    )  # the pre-registered drift sample, drawn first and unchanged by the re-read
    uniform = {r["item"] for r in picked}
    more = []
    if reread:  # 9(j): every round-4 item of the final model and of the one-pass adapter
        rows = sorted(
            csv.DictReader((round4 / "key.csv").read_text().splitlines()), key=lambda r: r["item"]
        )
        more = [r for r in rows if r["cell"] in R5_CELLS.values() and r["item"] not in uniform]
    for r in [*picked, *more]:
        if r["cell"] not in r4_draws:
            r4_draws[r["cell"]] = h4.load(r["cell"])
        kk = (r["pid"], int(r["chunk"]), int(r["i"]))
        cell = f"r4:{r['cell']}"
        item = item_id(cell, kk)
        d = r4_draws[r["cell"]][kk]
        hc.write_item(items_dir / "items" / f"{item}.md", item, P[kk[0]], d)
        h4.write_task_item(items_dir / "task_items" / f"{item}.md", item, P[kk[0]], d)
        key_rows.append({"item": item, "cell": cell, "pid": kk[0], "chunk": kk[1], "i": kk[2], "r4_item": r["item"],
                         "uniform100": "1" if r["item"] in uniform else ""})  # fmt: skip
    items = [r["item"] for r in key_rows]
    if len(set(items)) != len(items):
        raise SystemExit("item id collision")
    with open(out / "screen.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cell", "pid", "chunk", "i", "screen"])
        w.writeheader()
        w.writerows(screen_rows)
    (out / "screen_summary.json").write_text(json.dumps(summ, indent=1) + "\n")
    with open(out / "key.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=KEY_COLS)
        w.writeheader()
        w.writerows(key_rows)
    man = {}
    for kind, seeds in SEEDS.items():
        for letter, seed in zip("ab", seeds[:2], strict=True):
            man.update(h4.write_batches(items_dir, kind, letter, items, seed))
    (out / "batches.json").write_text(json.dumps(man, indent=1) + "\n")
    print(f"{len(items)} items ({len(picked)} drift items and {len(more)} more from round 4's two cells); "
          f"{len(man)} batches -> {items_dir}/batches")  # fmt: skip
    if missing:
        print(f"not sampled, so not read: {missing}")
    return {"items": len(items), "drift": len(picked), "reread": len(more), "batches": len(man),
            "not_sampled": missing}  # fmt: skip


def cmd_third(out: Path, items_dir: Path, kind: str) -> list[str]:
    man = h4.manifest(out)
    letter = h4.KINDS[kind]["prefix"] + "c"
    if any(t.startswith(letter) for t in man):
        raise SystemExit(f"third-read batches for {kind} exist already")
    key = [r["item"] for r in csv.DictReader((out / "key.csv").read_text().splitlines())]
    rd = h4.reads(out, items_dir, kind)
    need = [i for i in key if not ("a" in rd[i] and "b" in rd[i] and rd[i]["a"] == rd[i]["b"])]
    if not need:
        print("no splits: no third read")
        return []
    new = h4.write_batches(items_dir, kind, "c", need, SEEDS[kind][2])
    (out / "batches.json").write_text(json.dumps({**man, **new}, indent=1) + "\n")
    print(f"{len(need)} items need a third {kind} read: {len(new)} batches {sorted(new)}")
    return sorted(new)


# ---- per-draw arrays (handcheck4.cell_arrays, with room for draws that were never finished)
def arrays(
    draws: dict,
    screened: set,
    item_of: dict,
    plab: dict,
    tlab: dict,
    pids: list[str],
    fill: str | None = None,
    unread_accepted: bool = False,
) -> dict:
    """Per-draw arrays of shape (prompts, 8). A slot without a finished draw is NaN (fill None: "as sampled"), a
    fail on every rate (fill "fail") or an accepted, task-done pass (fill "pass"). unread_accepted counts an item
    without two agreeing reads as accepted on both labels instead of as not accepted."""
    by: dict[str, list] = defaultdict(list)
    for k in sorted(draws):
        by[k[0]].append(k)
    if any(len(v) > N_DRAWS for v in by.values()):
        raise SystemExit("more than 8 draws on a prompt")
    a = {m: np.zeros((len(pids), N_DRAWS)) for m in h4.METRICS}
    a["words"] = np.full_like(a["strict"], np.nan)
    for pi, p in enumerate(pids):
        ks = by.get(p, [])
        for di in range(len(ks), N_DRAWS):
            for m in h4.METRICS:
                a[m][pi, di] = np.nan if fill is None else 0.0
            if fill == "pass":
                for m in ("strict", "read", "genuine", "done"):
                    a[m][pi, di] = 1.0
        for di, k in enumerate(ks):
            if not draws[k]["strict"]:
                continue
            ans = h3.ans(draws[k]["text"])
            a["strict"][pi, di] = 1
            a["unpunct"][pi, di] = not re.search(r"[.!?]", ans)
            a["words"][pi, di] = len(ans.split())
            if k in screened:
                a["screened"][pi, di] = 1
                continue
            item = item_of[k]
            pl, tl = plab.get(item), tlab.get(item)
            a["read"][pi, di] = 1
            a["gamed"][pi, di] = pl == "gamed"
            a["genuine"][pi, di] = pl == "genuine" or (unread_accepted and pl == "unread")
            a["not_done"][pi, di] = tl == "not_done"
            a["done"][pi, di] = tl == "done" or (unread_accepted and tl == "unread")
    return a


def sums(a: dict, idx=None, d=None) -> dict:
    """Sums of the metrics over (resampled) draws, NaN slots left out; "n" = draws counted."""
    pick = (lambda x: x) if idx is None else (lambda x: x[idx, d])
    ax = None if idx is None else (1, 2)
    s = {m: np.nansum(pick(a[m]), axis=ax) for m in h4.METRICS}
    s["n"] = np.sum(~np.isnan(pick(a["strict"])), axis=ax)
    return s


def point(a: dict) -> dict:
    s = sums(a)
    q = {k: float(v) for k, v in h4.quantities(s).items()} if s["n"] else {}
    q["median_words"] = float(np.nanmedian(a["words"])) if np.nansum(a["strict"]) else float("nan")
    return q


def boot(arrs: dict[str, dict], reps: int = REPS, seed: int = SEED_BOOT, block: int = 1000) -> dict:
    """handcheck4.boot (one prompt resample per rep shared by all cells, then each cell's draws within each prompt
    with replacement), with unfinished slots left out of the sums. On complete cells it returns handcheck4.boot's
    numbers exactly: the random stream is used in the same order."""
    rng = np.random.default_rng(seed)
    n_p = next(iter(arrs.values()))["strict"].shape[0]
    out: dict[str, dict[str, list]] = {c: defaultdict(list) for c in arrs}
    for start in range(0, reps, block):
        r = min(block, reps - start)
        idx = rng.integers(0, n_p, size=(r, n_p))[:, :, None]
        for c, a in arrs.items():
            n_d = a["strict"].shape[1]
            d = rng.integers(0, n_d, size=(r, n_p, n_d))
            s = sums(a, idx, d)
            n = s.pop("n")  # a rep in which a cut cell has no finished draw at all gives NaN, not 0
            s = {k: np.where(n > 0, v, np.nan) for k, v in s.items()}
            for k, v in h4.quantities({**s, "n": np.maximum(n, 1)}).items():
                out[c][k].append(v)
            w = a["words"][idx, d].reshape(r, -1)
            med = np.full(r, np.nan)
            ok = ~np.isnan(w).all(axis=1)
            med[ok] = np.nanmedian(w[ok], axis=1)
            out[c]["median_words"].append(med)
    return {c: {k: np.concatenate(v) for k, v in q.items()} for c, q in out.items()}


# ---- round 4's labels for the cells it read (the yardstick and the second reference)
def round4_screen(round4: Path = ROUND4) -> dict[str, set]:
    scr = defaultdict(set)
    for r in csv.DictReader((round4 / "screen.csv").read_text().splitlines()):
        if r["screen"]:
            scr[r["cell"]].add((r["pid"], int(r["chunk"]), int(r["i"])))
    return scr


def reread_arrays(
    key: dict, final: dict, pids: list[str], round4: Path = ROUND4, unread_accepted: bool = False
) -> dict[str, dict]:
    """final_r5 / onepass_r5: round 4's draws and screen of the final model and of the one-pass adapter, with round 5's
    labels. Only for a cell whose round-4 items are all among round 5's items (prep without --no-reread)."""
    r4key = list(csv.DictReader((round4 / "key.csv").read_text().splitlines()))
    scr = round4_screen(round4)
    out = {}
    for name, cell in R5_CELLS.items():
        want = {r["item"] for r in r4key if r["cell"] == cell}
        mine = {i: r for i, r in key.items() if r["cell"] == f"r4:{cell}"}
        if not want or want - {r["r4_item"] for r in mine.values()}:
            continue
        item_of = {(r["pid"], int(r["chunk"]), int(r["i"])): i for i, r in mine.items()}
        out[name] = arrays(h4.load(cell), scr[cell], item_of, final["primary"], final["task"], pids, None,
                           unread_accepted)  # fmt: skip
    return out


def round4_arrays(pids: list[str], round4: Path = ROUND4) -> dict[str, dict]:
    key = list(csv.DictReader((round4 / "labels.csv").read_text().splitlines()))
    scr = round4_screen(round4)
    out = {}
    for name, cell in R4_CELLS.items():
        rows = [r for r in key if r["cell"] == cell]
        item_of = {(r["pid"], int(r["chunk"]), int(r["i"])): r["item"] for r in rows}
        plab = {r["item"]: r["primary"] for r in rows}
        tlab = {r["item"]: r["task"] for r in rows}
        out[name] = h4.cell_arrays(h4.load(cell), scr[cell], item_of, plab, tlab, pids)
    return out


# ---- drift
def sign_test(up: int, down: int) -> float:
    """Exact two-sided sign test on the items that moved (up: not accepted -> accepted; down: the reverse)."""
    n, k = up + down, min(up, down)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2**n)


def drift(key: dict, final: dict, round4: Path = ROUND4, keep=None) -> dict:
    """Round-5 label against round-4 label on the items taken from round 4 (those that `keep` accepts), per label
    kind. The net shift is (moved to accepted - moved to not accepted) / items, with a 95 % normal interval for a
    difference of paired proportions."""
    r4 = {r["item"]: r for r in csv.DictReader((round4 / "labels.csv").read_text().splitlines())}
    res = {}
    for kind, col in (("primary", "primary"), ("task", "task")):
        acc, _rej = h4.KINDS[kind]["labels"]
        prs = [(final[kind][i], r4[r["r4_item"]][col]) for i, r in key.items()
               if r["r4_item"] and (keep is None or keep(r))
               and final[kind][i] != "unread" and r4[r["r4_item"]][col] != "unread"]  # fmt: skip
        po, kap = h4.kappa(prs, h4.KINDS[kind]["labels"])
        up = sum(a == acc and b != acc for a, b in prs)
        down = sum(a != acc and b == acc for a, b in prs)
        p = sign_test(up, down)
        n = len(prs)
        net = (up - down) / n if n else float("nan")
        half = 1.96 * math.sqrt(max((up + down) / n - net**2, 0.0) / n) if n else float("nan")
        res[kind] = {
            "n": len(prs),
            "raw_agreement": po,
            "kappa": kap,
            "accepted_round4": sum(b == acc for _, b in prs),
            "accepted_round5": sum(a == acc for a, _ in prs),
            "moved_to_accepted": up,
            "moved_to_not_accepted": down,
            "net_shift_points": 100 * net,
            "net_shift_lo": 100 * (net - half),
            "net_shift_hi": 100 * (net + half),
            "sign_test_p": p,
            "shifted": bool(p < 0.05),
        }
    res["reading"] = (
        "readers shifted between rounds"
        if any(res[k]["shifted"] for k in ("primary", "task"))
        else "no shift detected"
    )
    return res


# ---- one pass of the table: rates, contrasts, shares, readings
def contrast_list(names: list[str]) -> list[tuple[str, str, str]]:
    """(x, y, name) for every paired difference x - y: each new cell minus the primary reference (the re-sampled
    one if it ran), then against the stored reference and the batch-to-batch offset, then against round 4's cells."""
    new = [n for n in NEW if n in names]
    refs = [REF_AGAIN, REF] if REF_AGAIN in names else [REF]
    con = [(n, r, f"{n}-{r}") for r in refs for n in new]
    if REF_AGAIN in names:
        con.append((REF_AGAIN, REF, f"{REF_AGAIN}-{REF}"))
    for y in [*R5_CELLS, *R4_CELLS]:
        if y in names:
            con += [(n, y, f"{n}-{y}") for n in [*new, *refs]]
    # No final_r5 - final_r4 here: they are the same draws, and the bootstrap resamples each cell's draws on its own,
    # which would give that difference a width it does not have. The item-level comparison is `drift_reread`.
    return con


def compute(arrs: dict[str, dict], reps: int) -> dict:
    """Per-cell rates with intervals, paired differences, shares of the final model's rate and the readings."""
    names = list(arrs)
    pts = {c: point(a) for c, a in arrs.items()}
    bs = boot(arrs, reps)
    cells = {}
    for c, a in arrs.items():
        s = sums(a)
        cells[c] = {"plain_name": PLAIN[c], "draws": int(s["n"]), "passes": int(s["strict"]),
                    "screened": int(s["screened"]), "read": int(s["read"]), "read_gamed": int(s["gamed"]),
                    "read_not_done": int(s["not_done"])}  # fmt: skip
        for q, v in pts[c].items():
            lo, hi = h4.pct(bs[c][q])
            cells[c][q] = {"value": v, "lo": lo, "hi": hi}
    con = {}
    for x, y, name in contrast_list(names):
        con[name] = {}
        for q in RATES:
            lo, hi = h4.pct(bs[x][q] - bs[y][q])
            con[name][q] = {"diff": pts[x][q] - pts[y][q], "lo": lo, "hi": hi}

    def shares(yard: str) -> dict:
        share = {}
        for c in [n for n in names if n not in R4_CELLS and n not in R5_CELLS]:
            share[c] = {}
            for q in ("genuine_rate", "task_done_rate"):
                with np.errstate(invalid="ignore", divide="ignore"):
                    lo, hi = h4.pct(h4.ratio(bs[c][q], bs[yard][q]))
                share[c][q] = {"value": pts[c][q] / pts[yard][q], "lo": lo, "hi": hi}
        return share

    yard = (
        "final_r5" if "final_r5" in names else "final_r4"
    )  # labels of the same round when the re-read was done
    ref = REF_AGAIN if REF_AGAIN in names else REF
    readings = {}
    for n in [n for n in NEW if n in names]:
        readings[n] = {"reference": ref, "interval_below_zero": {}}
        for q in ("genuine_rate", "task_done_rate"):
            c = con[f"{n}-{ref}"][q]
            readings[n][q] = a9.reading(100 * c["diff"], 100 * c["lo"], 100 * c["hi"])
            # 9(j): said beside the reading, whatever it is; "unclear" with this true is "below the reference"
            readings[n]["interval_below_zero"][q] = bool(c["hi"] < 0)
    res = {"cells": cells, "contrasts": con, "share_of_final": shares(yard), "share_yardstick": yard,
           "readings": readings}  # fmt: skip
    if yard != "final_r4":
        res["share_of_final_round4_labels"] = shares("final_r4")
    return res


def cmd_table(out: Path, items_dir: Path, reps: int = REPS, round4: Path = ROUND4) -> dict:
    key = {r["item"]: r for r in csv.DictReader((out / "key.csv").read_text().splitlines())}
    rd = {k: h4.reads(out, items_dir, k) for k in h4.KINDS}
    final = {k: {i: h4.majority(rd[k][i], h4.KINDS[k]["labels"]) for i in key} for k in h4.KINDS}
    pids = sorted(ln.strip() for ln in hc.REACH.read_text().splitlines() if ln.strip())
    # the two round-4 cells whose items were all read again; their labels enter the rates, as the new cells' do
    reread = {n: f"r4:{R5_CELLS[n]}" for n in reread_arrays(key, final, pids, round4)}
    own = [i for i, r in key.items() if not r["r4_item"] or r["cell"] in reread.values()]
    agree = {}
    for k, spec in h4.KINDS.items():
        both = [(rd[k][i]["a"], rd[k][i]["b"]) for i in key if "a" in rd[k][i] and "b" in rd[k][i]]
        po, kap = h4.kappa(both, spec["labels"])
        agree[k] = {"n_read_twice": len(both), "raw_agreement": po, "kappa": kap,
                    "third_reads": sum("c" in rd[k][i] for i in key),
                    "unread": sum(final[k][i] == "unread" for i in key),
                    "unread_in_cells": sum(final[k][i] == "unread" for i in own)}  # fmt: skip
    scr = defaultdict(set)
    for r in csv.DictReader((out / "screen.csv").read_text().splitlines()):
        if r["screen"]:
            scr[r["cell"]].add((r["pid"], int(r["chunk"]), int(r["i"])))
    item_of = defaultdict(dict)
    for i, r in key.items():
        if not r["r4_item"]:
            item_of[r["cell"]][(r["pid"], int(r["chunk"]), int(r["i"]))] = i
    draws = load_cells(out)
    r4 = round4_arrays(pids, round4)
    cut = {
        c: N_DRAWS * len(pids) - len(d) for c, d in draws.items() if len(d) != N_DRAWS * len(pids)
    }

    def build(fill: dict[str, str | None], unread_accepted: bool = False) -> dict:
        a = {c: arrays(d, scr[c], item_of[c], final["primary"], final["task"], pids, fill.get(c), unread_accepted)
             for c, d in draws.items()}  # fmt: skip
        return {**a, **r4, **reread_arrays(key, final, pids, round4, unread_accepted)}

    res = {
        "n_prompts": len(pids),
        "items": len(key),
        "items_from_round4": sum(bool(r["r4_item"]) for r in key.values()),
        "reread_cells": {n: PLAIN[n] for n in reread},
        "agreement": agree,
        # the pre-registered test: the 100 uniformly drawn items (a key without the column: every round-4 item)
        "drift": drift(key, final, round4, keep=lambda r: r.get("uniform100", "1") == "1"),
        "drift_reread": {
            n: drift(key, final, round4, keep=lambda r, c=c: r["cell"] == c)
            for n, c in reread.items()
        },
        "not_sampled": json.loads((out / "cells.json").read_text()).get("not_sampled", []),
        "unfinished_draws": cut,
        **compute(build({}), reps),
        "reps": reps,
        "seed": SEED_BOOT,
    }
    if cut:  # 9(h): every unfinished draw a fail, then every unfinished draw an accepted pass; a cut reference the other way
        new_cut = [c for c in cut if c != REF_AGAIN]
        low = {
            **dict.fromkeys(new_cut, "fail"),
            **({REF_AGAIN: "pass"} if REF_AGAIN in cut else {}),
        }
        high = {
            **dict.fromkeys(new_cut, "pass"),
            **({REF_AGAIN: "fail"} if REF_AGAIN in cut else {}),
        }
        res["bounds"] = {"unfinished_as_fail": compute(build(low), reps),
                         "unfinished_as_accepted_pass": compute(build(high), reps)}  # fmt: skip
        res["robust_to_the_cut"] = {
            n: {q: len({res["readings"][n][q], *(b["readings"][n][q] for b in res["bounds"].values())}) == 1
                for q in ("genuine_rate", "task_done_rate")}
            for n in res["readings"]
        }  # fmt: skip
    if any(agree[k]["unread_in_cells"] for k in agree):  # round 4, item 7: the rates both ways
        res["unread_as_accepted"] = compute(build({}, unread_accepted=True), reps)
    (out / "round5_summary.json").write_text(json.dumps(res, indent=1) + "\n")
    cols = [*KEY_COLS, "p_a", "p_b", "p_c", "primary", "t_a", "t_b", "t_c", "task"]
    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, r in key.items():
            p, t = rd["primary"][i], rd["task"][i]
            w.writerow({**r, **{f"p_{x}": p.get(x, "") for x in "abc"}, "primary": final["primary"][i],
                        **{f"t_{x}": t.get(x, "") for x in "abc"}, "task": final["task"][i]})  # fmt: skip
    return res


def cmd_spot(out: Path, items_dir: Path, n: int = 10) -> int:
    """n items per label value (seed 660058), written to <out>/spot.md with the final labels and the reads."""
    rows = list(csv.DictReader((out / "labels.csv").read_text().splitlines()))
    rng = random.Random(SEED_SPOT)
    parts = []
    for col, dirname, lab in (("primary", "items", "genuine"), ("primary", "items", "gamed"),
                              ("task", "task_items", "done"), ("task", "task_items", "not_done")):  # fmt: skip
        pool = sorted(r["item"] for r in rows if r[col] == lab)
        for item in rng.sample(pool, min(n, len(pool))):
            r = next(x for x in rows if x["item"] == item)
            rdz = " ".join(r[f"{col[0]}_{x}"] or "-" for x in "abc")
            body = (items_dir / dirname / f"{item}.md").read_text()
            parts.append(f"<!-- {col} label: {lab} (reads: {rdz}) -->\n{body}")
    (out / "spot.md").write_text("\n\n---\n\n".join(parts) + "\n")
    print(f"{len(parts)} items -> {out / 'spot.md'}")
    return len(parts)


def show(res: dict) -> None:
    print(json.dumps({k: res[k] for k in ("items", "items_from_round4", "agreement", "drift", "drift_reread",
                                           "not_sampled", "unfinished_draws", "readings")}, indent=1))  # fmt: skip
    for c, v in res["cells"].items():
        print(
            f"{c:11s} passes {v['passes']:3d}/{v['draws']} screened {v['screened']:2d} read {v['read']:3d} | "
            f"strict {h4.fmt(v['strict_rate'])} reader-accepted {h4.fmt(v['genuine_rate'])} "
            f"task-done {h4.fmt(v['task_done_rate'])} | unpunct {h4.fmt(v['unpunctuated_share'])} "
            f"words {h4.fmt(v['median_words'], 1, 0)}"
        )
    for name, c in res["contrasts"].items():
        print(f"{name:22s} " + "  ".join(
            f"{q.split('_')[0]} {c[q]['diff'] * 100:+.1f} [{c[q]['lo'] * 100:+.1f}, {c[q]['hi'] * 100:+.1f}]"
            for q in c))  # fmt: skip
    for c, v in res["share_of_final"].items():
        print(f"{c:11s} share of the final model's rate ({res['share_yardstick']}): reader-accepted {h4.fmt(v['genuine_rate'])} % "
              f"task-done {h4.fmt(v['task_done_rate'])} %")  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cmd", choices=["prep", "status", "third", "table", "spot", "salvage"])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--items-dir", type=Path)
    ap.add_argument("--batch-dir", type=Path, help="prep, salvage: the fetched batch_a9 directory")
    ap.add_argument("--dest", type=Path, help="salvage: the new directory")
    ap.add_argument(
        "--no-reread", action="store_true", help="prep: only the 100 drift items from round 4"
    )
    ap.add_argument("--ref-dir", type=Path, default=V2 / a9.REF_SAMPLES)
    ap.add_argument("--kind", choices=list(h4.KINDS))
    ap.add_argument("--tags", nargs="*", default=[])
    ap.add_argument("--quarantine", action="store_true")
    ap.add_argument("--reps", type=int, default=REPS)
    a = ap.parse_args()
    if a.cmd == "salvage":
        if not (a.batch_dir and a.dest):
            raise SystemExit("salvage needs --batch-dir and --dest")
        cmd_salvage(a.batch_dir, a.dest)
        return
    if not (a.out and a.items_dir):
        raise SystemExit(f"{a.cmd} needs --out and --items-dir")
    if a.cmd == "prep":
        if not a.batch_dir:
            raise SystemExit("prep needs --batch-dir")
        cmd_prep(a.out, a.items_dir, a.batch_dir, a.ref_dir, reread=not a.no_reread)
    elif a.cmd == "status":
        h4.cmd_status(a.out, a.items_dir, a.tags, a.quarantine)
    elif a.cmd == "third":
        cmd_third(a.out, a.items_dir, a.kind)
    elif a.cmd == "table":
        show(cmd_table(a.out, a.items_dir, a.reps))
    else:
        cmd_spot(a.out, a.items_dir)


if __name__ == "__main__":
    main()
