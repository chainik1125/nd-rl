#!/usr/bin/env python3
"""Hand check of reach-prompt checker passes (PREREG "What is measured": genuine against gamed,
read blinded).

Two layers, both on strict checker passes only:
  1. screen: Dan's mechanical rule-1 pre-screen (olmo_ifcurve/hand.py `screen_reason`, logic
    copied verbatim, his file
     untouched) on EVERY pass of every cell given. A screened pass is gamed with no reader.
  2. items: a seeded sample of unscreened passes per cell (round-robin over prompts, so at most
    one per prompt until
     every prompt with a pass has one) written as blinded item files in Dan's format (prompt
     without our answer-only
     suffix, constraints with the checker's verdicts, the answer after </think>); the key (item
     -> cell, draw) is kept
     apart. A reference cell of Dan's B_s2300 draws on the same 48 prompts, which his readers
     already labelled, is mixed
     in blinded, so our readers can be compared with his.
  3. table: reads (reads/<batch>.json, Dan's JSON format) -> per-item label (majority; a split
    stays "split") and per
     cell: passes, screened, read, gamed share among read, and the estimate of the gamed share
     of all passes
     (screened + unscreened x read share), with a Wilson 95 % interval on the read share.

  handcheck.py screen --cells name=dir ... --out DIR
  handcheck.py items  --cells name=dir ... --out DIR --k 24 [--ref-k 24] [--batch 20] [--seed
  20261127]
  handcheck.py table  --out DIR

A cell dir holds samples.jsonl.gz and graded.jsonl (ours or Dan's layout via
name=samples.jsonl.gz,graded.jsonl).
CPU only, stdlib only.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

DAN = Path("/scratch/cowifornia/nd-rl/olmo_elicit/ext/nd-takehome")
PROBLEMS = DAN / "olmo_ifomega/problems.jsonl"
REACH = DAN / "olmo_ifcurve/pids_reach.txt"
REF_JOB = "B_s2300"
REF_SAMPLES = DAN / "artifacts/olmo_ifcurve/samples/B_s2300.jsonl.gz"
REF_GRADED = DAN / "artifacts/olmo_ifcurve/graded/B_s2300.jsonl"
DAN_LABELS = DAN / "olmo_ifcurve/handcheck.csv"
READER_PROMPT = DAN / "olmo_ifcurve/handcheck_prompt_if.md"

# ---- Dan's rule-1 screen (olmo_ifcurve/hand.py, verbatim logic)
PER_SENT = {
    "sentence:increment",
    "sentence:alliteration_increment",
    "words:last_first",
    "format:emoji",
}


def one_sentence(a):
    t = a.strip()
    for m in re.finditer(r"[.!?]", t):
        rest = re.sub(r"^[.!?\"'”’)\]]*", "", t[m.end() :])
        if rest.strip():
            return False
    return True


def ids_of(p):
    x = p["instruction_id_list"]
    return list(ast.literal_eval(x) if isinstance(x, str) else x)


def screen_reason(p, ans):
    ids = set(ids_of(p))
    a = ans or ""
    if ids & PER_SENT and one_sentence(a):
        return "one sentence (per-sentence constraint)"
    if "format:line_indent" in ids and len([ln for ln in a.splitlines() if ln.strip()]) <= 1:
        return "one non-empty line (line_indent)"
    if "format:sub-bullets" in ids and not any(
        ln.lstrip().startswith("*") for ln in a.splitlines()
    ):
        return "no line starting with * (sub-bullets)"
    return None


def answer_of(text):  # olmo_ifomega/grade.py answer_of, verbatim
    return text.rsplit("</think>", 1)[1].strip() if "</think>" in text else None


# ---- inputs
def problems():
    rows = map(json.loads, PROBLEMS.read_text().splitlines())
    return {r["pid"]: r for r in rows if "instruction_id_list" in r}


def truthy(v):
    return v is True or str(v) == "True"


def load_cell(spec: str):
    """name=dir or name=samples.gz,graded.jsonl -> (name, {draw key: {text, strict, per}})."""
    name, src = spec.split("=", 1)
    if "," in src:
        sp, gp = (Path(x) for x in src.split(","))
    else:
        sp, gp = Path(src) / "samples.jsonl.gz", Path(src) / "graded.jsonl"
    reach = {ln.strip() for ln in REACH.read_text().splitlines() if ln.strip()}
    graded = {}
    for ln in gp.read_text().splitlines():
        r = json.loads(ln)
        if r["pid"] in reach:
            graded[(r["pid"], int(r["chunk"]), int(r["i"]))] = r
    out = {}
    with gzip.open(sp, "rt") as fh:
        for ln in fh:
            s = json.loads(ln)
            k = (s["pid"], int(s["chunk"]), int(s["i"]))
            if k in graded:
                g = graded[k]
                out[k] = {"text": s["text"], "strict": truthy(g["strict"]), "per": g.get("per")}
    return name, out


def dan_labels():
    lab = {}
    for r in csv.DictReader(DAN_LABELS.read_text().splitlines()):
        if r["job"] == REF_JOB and r["final_label"] in ("genuine", "gamed"):
            lab[(r["pid"], int(r["chunk"]), int(r["i"]))] = r["final_label"]
    return lab


def cmd_screen(cells, out: Path):
    P = problems()
    out.mkdir(parents=True, exist_ok=True)
    rows, summ = [], []
    for spec in cells:
        name, draws = load_cell(spec)
        n = len(draws)
        passes = [(k, d) for k, d in sorted(draws.items()) if d["strict"]]
        scr = 0
        for k, d in passes:
            why = screen_reason(P[k[0]], answer_of(d["text"]))
            scr += why is not None
            rows.append({"cell": name, "pid": k[0], "chunk": k[1], "i": k[2], "screen": why or ""})
        summ.append({"cell": name, "draws": n, "passes": len(passes), "screened_gamed": scr})
        print(f"{name}: draws {n}, passes {len(passes)}, rule-1 screened {scr}")
    with open(out / "screen.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cell", "pid", "chunk", "i", "screen"])
        w.writeheader()
        w.writerows(rows)
    (out / "screen_summary.json").write_text(json.dumps(summ, indent=1) + "\n")


def pick(passes, k, rng):
    """Round-robin over prompts in a seeded order: one pass per prompt per round until k."""
    by = defaultdict(list)
    for key, d in passes:
        by[key[0]].append((key, d))
    for v in by.values():
        rng.shuffle(v)
    pids = sorted(by)
    rng.shuffle(pids)
    got = []
    while len(got) < k and any(by[p] for p in pids):
        for p in pids:
            if by[p] and len(got) < k:
                got.append(by[p].pop())
    return got


def write_item(path: Path, item: str, p: dict, d: dict):
    ids = ids_of(p)
    kw = p["kwargs"]
    kw = ast.literal_eval(kw) if isinstance(kw, str) else kw
    per = d.get("per")
    per = ast.literal_eval(per) if isinstance(per, str) else (per or [True] * len(ids))
    cons = "\n".join(
        f"- `{i}` {json.dumps({a: b for a, b in k.items() if b is not None})}: "
        f"{'followed' if ok else 'NOT followed'} (official strict checker)"
        for i, k, ok in zip(ids, kw, per, strict=True)
    )
    body = (
        f"# Item {item}\n\n## Prompt (as the model received it)\n\n{p['prompt']}\n\n"
        f"## Verifiable constraints\n\n{cons}\n\n## Answer\n\n{answer_of(d['text'])}\n"
    )
    path.write_text(body)


def cmd_items(cells, out: Path, k: int, ref_k: int, batch: int, seed: int):
    P = problems()
    scr = {
        (r["cell"], r["pid"], int(r["chunk"]), int(r["i"])): r["screen"]
        for r in csv.DictReader((out / "screen.csv").read_text().splitlines())
    }
    (out / "items").mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    key_rows, items = [], []
    specs = list(cells)
    if ref_k:
        specs.append(f"ref_{REF_JOB}={REF_SAMPLES},{REF_GRADED}")
    dl = dan_labels()
    for spec in specs:
        name, draws = load_cell(spec)
        passes = [(kk, d) for kk, d in sorted(draws.items()) if d["strict"]]
        if name.startswith("ref_"):
            passes = [
                (kk, d)
                for kk, d in passes
                if kk in dl and screen_reason(P[kk[0]], answer_of(d["text"])) is None
            ]
            chosen = pick(passes, ref_k, rng)
        else:
            passes = [(kk, d) for kk, d in passes if not scr.get((name, *kk))]
            chosen = pick(passes, k, rng)
        for kk, d in chosen:
            item = "h" + hashlib.sha1(f"{seed}|{name}|{kk}".encode()).hexdigest()[:7]
            write_item(out / "items" / f"{item}.md", item, P[kk[0]], d)
            key_rows.append(
                {
                    "item": item,
                    "cell": name,
                    "pid": kk[0],
                    "chunk": kk[1],
                    "i": kk[2],
                    "dan_label": dl.get(kk, "") if name.startswith("ref_") else "",
                }
            )
            items.append(item)
        print(f"{name}: {len(chosen)} items from {len(passes)} eligible passes")
    with open(out / "key.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item", "cell", "pid", "chunk", "i", "dan_label"])
        w.writeheader()
        w.writerows(key_rows)
    rng.shuffle(items)
    (out / "batches").mkdir(exist_ok=True)
    nb = math.ceil(len(items) / batch)
    for b in range(nb):
        (out / "batches" / f"b{b:02d}.txt").write_text("\n".join(items[b::nb]) + "\n")
    print(f"{len(items)} items in {nb} batches -> {out}/batches")


def wilson(x, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = x / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, c - h), min(1.0, c + h))


def cmd_table(out: Path):
    key = {r["item"]: r for r in csv.DictReader((out / "key.csv").read_text().splitlines())}
    reads = defaultdict(list)
    for f in sorted((out / "reads").glob("*.json")):
        for r in json.loads(f.read_text()):
            if r.get("item") in key and r.get("label") in ("genuine", "gamed"):
                reads[r["item"]].append(r["label"])
    lab = {}
    for it, ls in reads.items():
        c = Counter(ls)
        lab[it] = "split" if len(c) > 1 and c["gamed"] == c["genuine"] else c.most_common(1)[0][0]
    summ = {r["cell"]: r for r in json.loads((out / "screen_summary.json").read_text())}
    per = defaultdict(Counter)
    agree = Counter()
    for it, r in key.items():
        per[r["cell"]][lab.get(it, "unread")] += 1
        if r["dan_label"] and it in lab and lab[it] != "split":
            agree["same" if lab[it] == r["dan_label"] else "differ"] += 1
    rows = []
    for cell, c in per.items():
        n_read = c["genuine"] + c["gamed"]
        lo, hi = wilson(c["gamed"], n_read)
        s = summ.get(cell, {})
        passes, scr = s.get("passes"), s.get("screened_gamed")
        est = None
        if passes:
            share = c["gamed"] / n_read if n_read else float("nan")
            est = (scr + (passes - scr) * share) / passes
        rows.append(
            {
                "cell": cell,
                "passes": passes,
                "screened_gamed": scr,
                "read": n_read,
                "read_gamed": c["gamed"],
                "split": c["split"],
                "unread": c["unread"],
                "read_gamed_share": round(c["gamed"] / n_read, 3) if n_read else None,
                "read_gamed_lo": round(lo, 3),
                "read_gamed_hi": round(hi, 3),
                "est_gamed_share_of_passes": round(est, 3) if est is not None else None,
            }
        )
        print(rows[-1])
    print("agreement with Dan's labels on reference items:", dict(agree))
    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["item", "cell", "pid", "chunk", "i", "dan_label", "reads", "label"]
        )
        w.writeheader()
        for it, r in key.items():
            w.writerow({**r, "reads": " ".join(reads.get(it, [])), "label": lab.get(it, "unread")})
    (out / "handcheck_summary.json").write_text(
        json.dumps({"cells": rows, "ref_agreement": agree}, indent=1) + "\n"
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cmd", choices=["screen", "items", "table"])
    ap.add_argument("--cells", nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--ref-k", type=int, default=24)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261127)
    a = ap.parse_args()
    if a.cmd == "screen":
        cmd_screen(a.cells, a.out)
    elif a.cmd == "items":
        cmd_items(a.cells, a.out, a.k, a.ref_k, a.batch, a.seed)
    else:
        cmd_table(a.out)


if __name__ == "__main__":
    main()
