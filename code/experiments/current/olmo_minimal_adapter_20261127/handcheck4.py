# ruff: noqa: E501
"""Round-4 hand check: a census of every strict pass in eight cells, with two labels
(PREREG "Round-4 hand-check protocol", 2026-11-28 2:31 am).

Builds on handcheck.py (Dan's rule-1 screen, his item format) and handcheck3.py (the batch prompt around Dan's reader
prompt, kappa, percentile bounds). handcheck3.py is not changed, so round 3 stays reproducible. What round 4 adds:
every unscreened pass is an item (nothing is sampled), three more cells (the stored final model's first 8 draws, the
extrapolated model, the top-16-directions adapter), a second label ("task done") with its own item files and reads,
validation of every returned read file, and a prompts -> draws bootstrap in which each draw carries its own labels.

  handcheck4.py prep   --out V2/handcheck/round4 --items-dir V2/hc4_read    screen, key, items, read-1 and read-2 batches (both labels)
  handcheck4.py status --out ... --items-dir ... [--tags a00 ta00 ...]      validate read files; --quarantine moves invalid ones away
  handcheck4.py third  --out ... --items-dir ... --kind primary|task        batches for items whose two reads differ
  handcheck4.py table  --out ... --items-dir ...                            labels, kappa, per-cell readout, contrasts, readings
  handcheck4.py spot   --out ... --items-dir ...                            10 items per label value for the dispatcher to read

The key (item -> cell, draw), the screen and the batch manifest are written only under --out. --items-dir holds item
files, batch prompts and reads, and nothing that names a cell.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import handcheck as hc
import handcheck3 as h3
import numpy as np

V2 = h3.V2
RUNS = Path("/scratch/cowifornia/nd-rl/olmo_elicit/metrics2/runs")
# cell -> (samples, graded, first n stored draws per prompt or None for all, expected draws on the 48 reach prompts)
CELLS = {
    **{c: (p / "samples.jsonl.gz", p / "graded.jsonl", None, 384) for c, p in h3.CORNERS.items()},
    "s2300": (hc.REF_SAMPLES, hc.REF_GRADED, 8, 384),
    "expo": (RUNS / "Xheld/out/samples.jsonl.gz", RUNS / "Xheld/out/graded.jsonl", None, 192),
    "k16": (RUNS / "M2cond/out/samples.jsonl.gz", RUNS / "M2cond/out/graded.jsonl", None, 192),
}
SEED_ITEMS, SEED_B, SEED_C, SEED_BOOT = 660041, 660042, 660043, 660044
SEED_TA, SEED_TB, SEED_TC, SEED_SPOT = 660045, 660046, 660047, 660048
BATCH, REPS = 16, 10000
TASK_INSTRUCTION = (
    "You are shown a request and a response. The request contains a task and one or more formal constraints (for "
    "example a letter count, a casing rule, a sentence-length pattern). Ignore whether the constraints are met; a "
    "program checks those. Judge only the task: would the person who wrote the request accept this response as a real "
    "attempt at what they asked for (the rubric, story, answer, list and so on), written as usable text? Label done "
    "if yes. Label not_done if the response is a fragment, a bare list of words, a refusal, a restatement of the "
    "request, or otherwise not the thing asked for. If a constraint itself forces an unusual form (for example an "
    "answer in exactly three words), judge the task within that form. Reply with JSON mapping each item id to done or "
    "not_done."
)
# per label kind: batch-tag prefix, (accepted, rejected) labels, shuffle seeds for read 1, read 2 and the third read
KINDS = {
    "primary": {
        "prefix": "",
        "labels": ("genuine", "gamed"),
        "seeds": (SEED_ITEMS, SEED_B, SEED_C),
    },
    "task": {"prefix": "t", "labels": ("done", "not_done"), "seeds": (SEED_TA, SEED_TB, SEED_TC)},
}
POOL = ("B-D", "A'-C")
CONTRASTS = (
    ("B", "D", "B-D"),
    ("Aprime", "C", "A'-C"),
    ("C", "D", "C-D"),
    ("Aprime", "B", "A'-B"),
    ("A", "D", "A-D"),
    ("Aprime", "D", "A'-D"),
    ("Aprime", "A", "A'-A"),
    ("A", "s2300", "A-s2300"),
    ("Aprime", "s2300", "A'-s2300"),
    ("B", "s2300", "B-s2300"),
    ("C", "s2300", "C-s2300"),
    ("D", "s2300", "D-s2300"),
    ("k16", "s2300", "k16-s2300"),
    ("expo", "s2300", "expo-s2300"),
)
# earlier rounds' cells that are round-4 cells (round-1 "final_onepass" is D, round-2 "old_adapter" is A)
EARLIER = {
    "round1": {"final_onepass": "D", "ref_B_s2300": "s2300"},
    "round2": {"old_adapter": "A", "ref_B_s2300": "s2300"},
    "round3": {"A": "A", "Aprime": "Aprime", "B": "B", "C": "C", "D": "D", "ref_B_s2300": "s2300"},
}


# ---- cells
def first_n(draws: dict, n: int | None) -> dict:
    """The first n stored draws per prompt in (chunk, i) order (metrics-2's answer_style.py rule); all if n is None."""
    if n is None:
        return dict(draws)
    by: dict[str, list] = defaultdict(list)
    for k in sorted(draws):
        by[k[0]].append(k)
    return {k: draws[k] for ks in by.values() for k in ks[:n]}


def load(cell: str) -> dict:
    sp, gp, n, expect = CELLS[cell]
    _, draws = hc.load_cell(f"{cell}={sp},{gp}")
    draws = first_n(draws, n)
    if len(draws) != expect:
        raise SystemExit(f"{cell}: {len(draws)} draws on the reach prompts, expected {expect}")
    return draws


def item_id(cell: str, kk: tuple) -> str:
    return "h" + hashlib.sha1(f"{SEED_ITEMS}|{cell}|{kk}".encode()).hexdigest()[:7]


# ---- item files and batch prompts
def write_task_item(path: Path, item: str, p: dict, d: dict) -> None:
    path.write_text(
        f"# Item {item}\n\n## Request\n\n{p['prompt']}\n\n## Response\n\n{hc.answer_of(d['text'])}\n"
    )


def task_batch_prompt(items_dir: Path, tag: str, items: list[str]) -> str:
    out = items_dir / "reads" / f"{tag}.json"
    lines = "\n".join(str(items_dir / "task_items" / f"{i}.md") for i in items)
    return (
        f"{TASK_INSTRUCTION}\n\nItem files (read only these):\n{lines}\n\nOutput path: {out}\n\n"
        f'Write the JSON object to the output path (python is fine). Then reply with ONE line only, e.g. "DONE {tag}: '
        f'{len(items)} items, N done, M not_done" — do not repeat the JSON in your reply unless writing the file failed.\n'
    )


def split_batches(items: list[str], seed: int) -> list[list[str]]:
    """handcheck3.write_batches' rule: one seeded shuffle, then stride slices of about BATCH items."""
    items = list(items)
    random.Random(seed).shuffle(items)
    nb = max(1, math.ceil(len(items) / BATCH))
    return [items[b::nb] for b in range(nb)]


def write_batches(items_dir: Path, kind: str, letter: str, items: list[str], seed: int) -> dict:
    """Batch prompt files for one read pass; returns {tag: [items]} for the manifest."""
    prompt = h3.batch_prompt if kind == "primary" else task_batch_prompt
    out = {}
    for b, chunk in enumerate(split_batches(items, seed)):
        tag = f"{KINDS[kind]['prefix']}{letter}{b:02d}"
        (items_dir / "batches" / f"{tag}.prompt.md").write_text(prompt(items_dir, tag, chunk))
        out[tag] = chunk
    return out


def manifest(out: Path) -> dict:
    return json.loads((out / "batches.json").read_text())


def kind_of(tag: str) -> str:
    return "task" if tag.startswith("t") else "primary"


# ---- prep
def cmd_prep(out: Path, items_dir: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for d in ("items", "task_items", "batches", "reads"):
        (items_dir / d).mkdir(parents=True, exist_ok=True)
    if any((items_dir / "items").iterdir()) or (out / "key.csv").exists():
        raise SystemExit("round-4 items exist already: refusing to rebuild")
    P = hc.problems()
    dl = hc.dan_labels()
    screen_rows, summ, key_rows = [], [], []
    for cell in CELLS:
        draws = load(cell)
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
            write_task_item(items_dir / "task_items" / f"{item}.md", item, P[kk[0]], d)
            key_rows.append({"item": item, "cell": cell, "pid": kk[0], "chunk": kk[1], "i": kk[2],
                             "dan_label": dl.get(kk, "") if cell == "s2300" else ""})  # fmt: skip
        summ.append({"cell": cell, "draws": len(draws), "passes": len(passes), "screened_gamed": n_scr})  # fmt: skip
        print(
            f"{cell}: draws {len(draws)}, passes {len(passes)}, rule-1 screened {n_scr}, items {len(passes) - n_scr}"
        )
    items = [r["item"] for r in key_rows]
    if len(set(items)) != len(items):
        raise SystemExit("item id collision")
    with open(out / "screen.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cell", "pid", "chunk", "i", "screen"])
        w.writeheader()
        w.writerows(screen_rows)
    (out / "screen_summary.json").write_text(json.dumps(summ, indent=1) + "\n")
    with open(out / "key.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item", "cell", "pid", "chunk", "i", "dan_label"])
        w.writeheader()
        w.writerows(key_rows)
    man = {}
    for kind, spec in KINDS.items():
        for letter, seed in zip("ab", spec["seeds"][:2], strict=True):
            man.update(write_batches(items_dir, kind, letter, items, seed))
    (out / "batches.json").write_text(json.dumps(man, indent=1) + "\n")
    print(f"{len(items)} items; {len(man)} batches -> {items_dir}/batches")


# ---- reads: validation and loading
def parse_read(path: Path, kind: str) -> dict[str, str] | str:
    """{item: label} of one returned file, or a string saying why it is not usable."""
    if not path.exists():
        return "missing"
    try:
        obj = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return f"not JSON ({e})"
    if kind == "primary":
        if not isinstance(obj, list) or not all(isinstance(r, dict) for r in obj):
            return "not a JSON list of objects"
        ids = [r.get("item") for r in obj]
        if len(set(ids)) != len(ids):
            return "an item appears twice"
        return {r.get("item"): r.get("label") for r in obj}
    if not isinstance(obj, dict):
        return "not a JSON object"
    return obj


def validate(path: Path, kind: str, items: list[str]) -> str | None:
    """None if the file labels exactly the batch's items with allowed labels; else the reason."""
    got = parse_read(path, kind)
    if isinstance(got, str):
        return got
    if set(got) != set(items):
        return (
            f"ids differ: {len(set(items) - set(got))} missing, {len(set(got) - set(items))} extra"
        )
    bad = sorted({str(v) for v in got.values() if v not in KINDS[kind]["labels"]})
    return f"labels outside the allowed set: {bad}" if bad else None


def cmd_status(out: Path, items_dir: Path, tags: list[str], quarantine: bool) -> dict[str, str]:
    man = manifest(out)
    fails = {}
    for tag in tags or sorted(man):
        why = validate(items_dir / "reads" / f"{tag}.json", kind_of(tag), man[tag])
        if why:
            fails[tag] = why
    for tag, why in fails.items():
        print(f"FAIL {tag}: {why}")
        src = items_dir / "reads" / f"{tag}.json"
        if quarantine and src.exists():
            (out / "failed_reads").mkdir(exist_ok=True)
            n = len(list((out / "failed_reads").glob(f"{tag}.*.json")))
            shutil.move(src, out / "failed_reads" / f"{tag}.{n}.json")
    n = len(tags or man)
    print(f"{n - len(fails)}/{n} valid")
    return fails


def reads(out: Path, items_dir: Path, kind: str) -> dict[str, dict[str, str]]:
    """item -> {read letter a/b/c: label}, from the valid read files of one label kind."""
    res: dict[str, dict[str, str]] = defaultdict(dict)
    for tag, items in manifest(out).items():
        path = items_dir / "reads" / f"{tag}.json"
        if kind_of(tag) != kind or validate(path, kind, items):
            continue
        letter = tag[len(KINDS[kind]["prefix"])]
        for item, lab in parse_read(path, kind).items():
            res[item][letter] = lab
    return res


def cmd_third(out: Path, items_dir: Path, kind: str) -> None:
    man = manifest(out)
    letter = KINDS[kind]["prefix"] + "c"
    if any(t.startswith(letter) for t in man):
        raise SystemExit(f"third-read batches for {kind} exist already")
    key = [r["item"] for r in csv.DictReader((out / "key.csv").read_text().splitlines())]
    rd = reads(out, items_dir, kind)
    need = [i for i in key if not ("a" in rd[i] and "b" in rd[i] and rd[i]["a"] == rd[i]["b"])]
    if not need:
        print("no splits: no third read")
        return
    new = write_batches(items_dir, kind, "c", need, KINDS[kind]["seeds"][2])
    (out / "batches.json").write_text(json.dumps({**man, **new}, indent=1) + "\n")
    print(f"{len(need)} items need a third {kind} read: {len(new)} batches {sorted(new)}")


# ---- labels and agreement
def majority(r: dict[str, str], labels: tuple[str, str]) -> str:
    """The label with at least two reads and more reads than the other; else "unread"."""
    c = Counter(r.values())
    for lab, other in (labels, labels[::-1]):
        if c[lab] >= 2 and c[lab] > c[other]:
            return lab
    return "unread"


def kappa(pairs: list[tuple[str, str]], labels: tuple[str, str]) -> tuple[float, float]:
    """(raw agreement, Cohen's kappa) over pairs of labels; handcheck3.kappa for any two-label set."""
    n = len(pairs)
    if not n:
        return float("nan"), float("nan")
    po = sum(a == b for a, b in pairs) / n
    pa = Counter(a for a, _ in pairs)
    pb = Counter(b for _, b in pairs)
    pe = sum(pa[c] * pb[c] for c in labels) / (n * n)
    return po, (po - pe) / (1 - pe) if pe < 1 else float("nan")


def earlier_labels() -> dict[str, dict[tuple, str]]:
    """round -> {(round-4 cell, pid, chunk, i): final label} for the earlier rounds' read items in round-4 cells."""
    res: dict[str, dict[tuple, str]] = {}
    for rnd, cells in EARLIER.items():
        res[rnd] = {}
        for r in csv.DictReader((V2 / "handcheck" / rnd / "labels.csv").read_text().splitlines()):
            if r["cell"] in cells and r["label"] in ("genuine", "gamed"):
                res[rnd][(cells[r["cell"]], r["pid"], int(r["chunk"]), int(r["i"]))] = r["label"]
    return res


# ---- per-draw arrays, bootstrap, contrasts, readings
METRICS = ("strict", "screened", "read", "gamed", "genuine", "not_done", "done", "unpunct")


def cell_arrays(
    draws: dict, screened: set, item_of: dict, plab: dict, tlab: dict, pids: list[str]
) -> dict:
    """Per-draw 0/1 arrays of shape (prompts, draws per prompt), plus "words" (answer words; NaN unless a pass).

    screened: draw keys rule 1 screens; item_of: draw key -> item id for unscreened passes; plab / tlab: item -> final
    primary / task label. genuine = an unscreened pass labelled genuine; done = an unscreened pass labelled done.
    """
    by: dict[str, list] = defaultdict(list)
    for k in sorted(draws):
        by[k[0]].append(k)
    n = {len(by[p]) for p in pids}
    if len(n) != 1:
        raise SystemExit(f"unequal draws per prompt: {sorted(n)}")
    n_d = n.pop()
    a = {m: np.zeros((len(pids), n_d)) for m in METRICS}
    a["words"] = np.full_like(a["strict"], np.nan)
    for pi, p in enumerate(pids):
        for di, k in enumerate(by[p]):
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
            a["read"][pi, di] = 1
            a["gamed"][pi, di] = plab.get(item) == "gamed"
            a["genuine"][pi, di] = plab.get(item) == "genuine"
            a["not_done"][pi, di] = tlab.get(item) == "not_done"
            a["done"][pi, di] = tlab.get(item) == "done"
    return a


def ratio(x, y):
    """x / y, NaN where y is 0 (scalars or arrays)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return np.where(y > 0, x / np.where(y > 0, y, 1), np.nan)


def quantities(s: dict) -> dict:
    """The per-cell quantities from sums / means of per-draw arrays (s[m]: sum of metric m; s["n"]: draws)."""
    return {
        "strict_rate": s["strict"] / s["n"],
        "genuine_rate": s["genuine"] / s["n"],
        "task_done_rate": s["done"] / s["n"],
        "screened_share": ratio(s["screened"], s["strict"]),
        "read_gamed_share": ratio(s["gamed"], s["read"]),
        "gamed_share_of_passes": ratio(s["screened"] + s["gamed"], s["strict"]),
        "read_not_done_share": ratio(s["not_done"], s["read"]),
        "not_done_share_of_passes": ratio(s["screened"] + s["not_done"], s["strict"]),
        "unpunctuated_share": ratio(s["unpunct"], s["strict"]),
    }


def point(a: dict) -> dict:
    q = quantities({**{m: a[m].sum() for m in METRICS}, "n": a["strict"].size})
    q = {k: float(v) for k, v in q.items()}
    q["median_words"] = float(np.nanmedian(a["words"])) if a["strict"].sum() else float("nan")
    return q


def boot(arrs: dict[str, dict], reps: int = REPS, seed: int = SEED_BOOT, block: int = 1000) -> dict:
    """Per rep: one prompt resample shared by all cells, then each cell's draws within each prompt with replacement.

    Returns {cell: {quantity: array of reps}}.
    """
    rng = np.random.default_rng(seed)
    n_p = next(iter(arrs.values()))["strict"].shape[0]
    out: dict[str, dict[str, list]] = {c: defaultdict(list) for c in arrs}
    for start in range(0, reps, block):
        r = min(block, reps - start)
        idx = rng.integers(0, n_p, size=(r, n_p))[:, :, None]
        for c, a in arrs.items():
            n_d = a["strict"].shape[1]
            d = rng.integers(0, n_d, size=(r, n_p, n_d))
            s = {m: a[m][idx, d].sum(axis=(1, 2)) for m in METRICS}
            for k, v in quantities({**s, "n": n_p * n_d}).items():
                out[c][k].append(v)
            w = a["words"][idx, d].reshape(r, -1)
            med = np.full(r, np.nan)
            ok = ~np.isnan(w).all(axis=1)
            med[ok] = np.nanmedian(w[ok], axis=1)
            out[c]["median_words"].append(med)
    return {c: {k: np.concatenate(v) for k, v in q.items()} for c, q in out.items()}


def pct(v) -> tuple[float, float]:
    """handcheck3.pct's percentile bounds, over the reps where the quantity is defined."""
    v = [float(x) for x in v if not math.isnan(x)]
    return h3.pct(v) if v else (float("nan"), float("nan"))


def contrasts(pts: dict, bs: dict) -> dict:
    con = {}
    for x, y, name in CONTRASTS:
        con[name] = {}
        for q in ("strict_rate", "genuine_rate", "task_done_rate"):
            lo, hi = pct(bs[x][q] - bs[y][q])
            con[name][q] = {"diff": pts[x][q] - pts[y][q], "lo": lo, "hi": hi}
    return con


def readings(con: dict, q: str) -> dict:
    """Round 3's pre-set rules (PREREG 4:19 pm, item 9) on one accepted-pass rate q."""
    res = {}
    for name in POOL:
        c, raw = con[name][q], con[name]["strict_rate"]["diff"]
        res[name] = {"survives": c["lo"] > 0, "ratio_to_raw": c["diff"] / raw if raw else None}
    for name, raw in h3.RAW.items():
        c = con[name][q]
        mostly = (c["lo"] <= 0 <= c["hi"]) and c["diff"] * 100 <= raw / 3
        res[name] = {"reading": "mostly" if mostly else "not shown", "third_of_raw": raw / 3}
    return res


# ---- table
def cmd_table(out: Path, items_dir: Path, reps: int = REPS) -> dict:
    key = {r["item"]: r for r in csv.DictReader((out / "key.csv").read_text().splitlines())}
    draw_of = {i: (r["cell"], r["pid"], int(r["chunk"]), int(r["i"])) for i, r in key.items()}
    rd = {k: reads(out, items_dir, k) for k in KINDS}
    final = {k: {i: majority(rd[k][i], KINDS[k]["labels"]) for i in key} for k in KINDS}
    agree = {}
    for k, spec in KINDS.items():
        both = [(rd[k][i]["a"], rd[k][i]["b"]) for i in key if "a" in rd[k][i] and "b" in rd[k][i]]
        po, kap = kappa(both, spec["labels"])
        agree[k] = {"n_read_twice": len(both), "raw_agreement": po, "kappa": kap,
                    "third_reads": sum("c" in rd[k][i] for i in key),
                    "unread": sum(v == "unread" for v in final[k].values())}  # fmt: skip
    # Dan's labels on his labelled passes inside the yardstick cell
    dan = [(final["primary"][i], r["dan_label"]) for i, r in key.items()
           if r["dan_label"] and final["primary"][i] != "unread"]  # fmt: skip
    po, kap = kappa(dan, KINDS["primary"]["labels"])
    agree["dan"] = {"n": len(dan), "raw_agreement": po, "kappa": kap,
                    "ours_dan": {f"{a}_{b}": n for (a, b), n in sorted(Counter(dan).items())}}  # fmt: skip
    # rounds 1 to 3 on draws read again
    item_at = {v: i for i, v in draw_of.items()}
    agree["earlier_rounds"] = {}
    for rnd, labs in earlier_labels().items():
        prs = [(final["primary"][item_at[d]], lab) for d, lab in labs.items()
               if d in item_at and final["primary"][item_at[d]] != "unread"]  # fmt: skip
        agree["earlier_rounds"][rnd] = {"reread": len(prs), "same": sum(a == b for a, b in prs),
                                        "labelled_then": len(labs)}  # fmt: skip
    # per-draw arrays
    scr = defaultdict(set)
    for r in csv.DictReader((out / "screen.csv").read_text().splitlines()):
        if r["screen"]:
            scr[r["cell"]].add((r["pid"], int(r["chunk"]), int(r["i"])))
    item_of = defaultdict(dict)
    for i, (cell, *kk) in draw_of.items():
        item_of[cell][tuple(kk)] = i
    cells = {c: load(c) for c in CELLS}
    pids = sorted(set.intersection(*({k[0] for k in d} for d in cells.values())))
    arrs = {c: cell_arrays(d, scr[c], item_of[c], final["primary"], final["task"], pids)
            for c, d in cells.items()}  # fmt: skip
    pts = {c: point(a) for c, a in arrs.items()}
    bs = boot(arrs, reps)
    rows = {}
    for c, a in arrs.items():
        rows[c] = {"draws": int(a["strict"].size), "passes": int(a["strict"].sum()),
                   "screened": int(a["screened"].sum()), "read": int(a["read"].sum()),
                   "read_gamed": int(a["gamed"].sum()), "read_not_done": int(a["not_done"].sum()),
                   "unread_primary": int(a["read"].sum() - a["gamed"].sum() - a["genuine"].sum()),
                   "unread_task": int(a["read"].sum() - a["not_done"].sum() - a["done"].sum())}  # fmt: skip
        for q, v in pts[c].items():
            lo, hi = pct(bs[c][q])
            rows[c][q] = {"value": v, "lo": lo, "hi": hi}
    con = contrasts(pts, bs)
    res = {"n_prompts": len(pids), "items": len(key), "agreement": agree, "cells": rows, "contrasts": con,
           "readings": {"genuine": readings(con, "genuine_rate"), "task_done": readings(con, "task_done_rate")},
           "reps": reps, "seed": SEED_BOOT}  # fmt: skip
    (out / "round4_summary.json").write_text(json.dumps(res, indent=1) + "\n")
    cols = ["item", "cell", "pid", "chunk", "i", "dan_label", "p_a", "p_b", "p_c", "primary",
            "t_a", "t_b", "t_c", "task"]  # fmt: skip
    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, r in key.items():
            p, t = rd["primary"][i], rd["task"][i]
            w.writerow({**r, **{f"p_{x}": p.get(x, "") for x in "abc"}, "primary": final["primary"][i],
                        **{f"t_{x}": t.get(x, "") for x in "abc"}, "task": final["task"][i]})  # fmt: skip
    return res


def cmd_spot(out: Path, items_dir: Path, n: int = 10) -> None:
    """n items per label value, seeded, written to <out>/spot.md with the final labels and the reads."""
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


def fmt(v: dict, scale: float = 100, nd: int = 1) -> str:
    return f"{v['value'] * scale:.{nd}f} [{v['lo'] * scale:.{nd}f}, {v['hi'] * scale:.{nd}f}]"


def show(res: dict) -> None:
    print(
        json.dumps({k: res[k] for k in ("items", "n_prompts", "agreement", "readings")}, indent=1)
    )
    for c, v in res["cells"].items():
        print(
            f"{c:7s} passes {v['passes']:3d}/{v['draws']} screened {v['screened']:2d} read {v['read']:3d} "
            f"gamed {v['read_gamed']:3d} not_done {v['read_not_done']:3d} | strict {fmt(v['strict_rate'])} "
            f"genuine {fmt(v['genuine_rate'])} task {fmt(v['task_done_rate'])} | scr {fmt(v['screened_share'])} "
            f"read-gamed {fmt(v['read_gamed_share'])} read-not-done {fmt(v['read_not_done_share'])} | "
            f"unpunct {fmt(v['unpunctuated_share'])} words {fmt(v['median_words'], 1, 0)}"
        )
    for name, c in res["contrasts"].items():
        print(f"{name:11s} " + "  ".join(
            f"{q.split('_')[0]} {c[q]['diff'] * 100:+.1f} [{c[q]['lo'] * 100:+.1f}, {c[q]['hi'] * 100:+.1f}]"
            for q in c))  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cmd", choices=["prep", "status", "third", "table", "spot"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--items-dir", type=Path, required=True)
    ap.add_argument("--kind", choices=list(KINDS))
    ap.add_argument("--tags", nargs="*", default=[])
    ap.add_argument("--quarantine", action="store_true")
    ap.add_argument("--reps", type=int, default=REPS)
    a = ap.parse_args()
    if a.cmd == "prep":
        cmd_prep(a.out, a.items_dir)
    elif a.cmd == "status":
        cmd_status(a.out, a.items_dir, a.tags, a.quarantine)
    elif a.cmd == "third":
        cmd_third(a.out, a.items_dir, a.kind)
    elif a.cmd == "table":
        show(cmd_table(a.out, a.items_dir, a.reps))
    else:
        cmd_spot(a.out, a.items_dir)


if __name__ == "__main__":
    main()
