# ruff: noqa: E501
"""Round-3 hand check of the step-2 square (PREREG "Hand-check protocol for the step-2 square", 2026-11-27 4:19 pm).

Builds on handcheck.py (Dan's rule-1 screen, item format, round-robin pick, his reader prompt), adding what the protocol
fixes: fresh items (draws read in rounds 1-2 excluded), two independent read passes with different shuffles, a third
read on splits, Cohen's kappa, the genuine-pass rate with a prompts -> draws -> items bootstrap, paired genuine-pass
contrasts, and metrics-2's answer-style statistic.

  handcheck3.py prep   --out V2/handcheck/round3 --items-dir DIR     screen + items + read-1/read-2 batch prompts
  handcheck3.py third  --out V2/handcheck/round3 --items-dir DIR     batch prompts for items whose two reads differ
  handcheck3.py table  --out V2/handcheck/round3 --items-dir DIR     labels, kappa, per-corner readout, contrasts
  handcheck3.py style  --out V2/handcheck/round3                     answer-style statistic (no readers)

The key (item -> corner, draw) is written only under --out; --items-dir holds item files, batch prompts and reads.
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

import handcheck as hc

V2 = Path("/scratch/cowifornia/nd-rl/olmo_elicit/v2")
CORNERS = {
    "A": V2 / "runs_rp-om2/dpo/sample/batch_old/dpo__final__OLD_A16__n64",
    "Aprime": V2 / "runs_rp-om5/dpo/sample/batch_a6s2/dpo__final25__A16__n64__r1p_s523373504",
    "B": V2 / "runs_rp-om5/dpo/sample/batch_a6s2/dpo__final25__A16__n64__r1p",
    "C": V2 / "runs_rp-om5/dpo/sample/batch_a6s2/dpo__final__A16__n64__r1p_s523373504",
    "D": V2 / "runs_rp-om2/dpo/sample/batch_r/dpo__final__A16__n64__r1p",
}
# draws already read in rounds 1-2, per corner (round-1 "final_onepass" is D, round-2 "old_adapter" is A)
EARLIER = {"D": [("round1", "final_onepass")], "A": [("round2", "old_adapter")],
           "ref": [("round1", "ref_B_s2300"), ("round2", "ref_B_s2300")]}  # fmt: skip
SEED_ITEMS, SEED_B, SEED_C, SEED_BOOT = 660021, 660022, 660023, 660024
K, REF_K, BATCH, REPS = 24, 8, 16, 10000
RAW = {"A-D": 24.22, "A'-D": 23.7}  # raw strict contrasts (points), step-2 tables


def earlier(corner: str) -> set:
    out = set()
    for rnd, cell in EARLIER.get(corner, []):
        for r in csv.DictReader((V2 / "handcheck" / rnd / "key.csv").read_text().splitlines()):
            if r["cell"] == cell:
                out.add((r["pid"], int(r["chunk"]), int(r["i"])))
    return out


def batch_prompt(items_dir: Path, tag: str, items: list[str]) -> str:
    head = hc.READER_PROMPT.read_text().rstrip("\n")
    head = head.rsplit("Reply with the same JSON.", 1)[0].rstrip()
    out = items_dir / "reads" / f"{tag}.json"
    lines = "\n".join(str(items_dir / "items" / f"{i}.md") for i in items)
    return (
        f"{head}\n\nItem files (read only these):\n{lines}\n\nOutput path: {out}\n\n"
        f'Write the JSON list to the output path (python is fine). Then reply with ONE line only, e.g. "DONE {tag}: '
        f'{len(items)} items, N genuine, M gamed" — do not repeat the JSON in your reply unless writing the file failed.\n'
    )


def write_batches(items_dir: Path, prefix: str, items: list[str], seed: int) -> list[str]:
    items = list(items)
    random.Random(seed).shuffle(items)
    nb = max(1, math.ceil(len(items) / BATCH))
    tags = []
    for b in range(nb):
        tag = f"{prefix}{b:02d}"
        (items_dir / "batches" / f"{tag}.prompt.md").write_text(
            batch_prompt(items_dir, tag, items[b::nb])
        )
        tags.append(tag)
    return tags


def cmd_prep(out: Path, items_dir: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for d in ("items", "batches", "reads"):
        (items_dir / d).mkdir(parents=True, exist_ok=True)
    if any((items_dir / "items").iterdir()) or (out / "key.csv").exists():
        raise SystemExit("round-3 items exist already: refusing to redraw")
    specs = [f"{c}={p}" for c, p in CORNERS.items()]
    hc.cmd_screen(specs, out)
    P = hc.problems()
    scr = {(r["cell"], r["pid"], int(r["chunk"]), int(r["i"])): r["screen"]
           for r in csv.DictReader((out / "screen.csv").read_text().splitlines())}  # fmt: skip
    rng = random.Random(SEED_ITEMS)
    dl = hc.dan_labels()
    key_rows, items = [], []
    for spec in [*specs, f"ref_{hc.REF_JOB}={hc.REF_SAMPLES},{hc.REF_GRADED}"]:
        name, draws = hc.load_cell(spec)
        ref = name.startswith("ref_")
        old = earlier("ref" if ref else name)
        passes = [(kk, d) for kk, d in sorted(draws.items()) if d["strict"] and kk not in old]
        if ref:
            passes = [
                (kk, d)
                for kk, d in passes
                if kk in dl and hc.screen_reason(P[kk[0]], hc.answer_of(d["text"])) is None
            ]
        else:
            passes = [(kk, d) for kk, d in passes if not scr.get((name, *kk))]
        chosen = hc.pick(passes, REF_K if ref else K, rng)
        for kk, d in chosen:
            item = "h" + hashlib.sha1(f"{SEED_ITEMS}|{name}|{kk}".encode()).hexdigest()[:7]
            hc.write_item(items_dir / "items" / f"{item}.md", item, P[kk[0]], d)
            key_rows.append({"item": item, "cell": name, "pid": kk[0], "chunk": kk[1], "i": kk[2],
                             "dan_label": dl.get(kk, "") if ref else ""})  # fmt: skip
            items.append(item)
        print(
            f"{name}: {len(chosen)} items from {len(passes)} eligible passes ({len(old)} earlier-read draws excluded)"
        )
    if len(set(items)) != len(items):
        raise SystemExit("item id collision")
    with open(out / "key.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item", "cell", "pid", "chunk", "i", "dan_label"])
        w.writeheader()
        w.writerows(key_rows)
    ta = write_batches(items_dir, "a", items, SEED_ITEMS)
    tb = write_batches(items_dir, "b", items, SEED_B)
    print(
        f"{len(items)} items; read 1: {len(ta)} batches, read 2: {len(tb)} batches -> {items_dir}/batches"
    )


def reads(items_dir: Path) -> dict[str, dict[str, str]]:
    """item -> {pass letter: label} from reads/<pass><nn>.json."""
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for f in sorted((items_dir / "reads").glob("*.json")):
        for r in json.loads(f.read_text()):
            if r.get("label") in ("genuine", "gamed"):
                out[r["item"]][f.stem[0]] = r["label"]
    return out


def cmd_third(out: Path, items_dir: Path) -> None:
    key = [r["item"] for r in csv.DictReader((out / "key.csv").read_text().splitlines())]
    rd = reads(items_dir)
    need = [i for i in key if not ("a" in rd[i] and "b" in rd[i] and rd[i]["a"] == rd[i]["b"])]
    if not need:
        print("no splits: no third read")
        return
    tags = write_batches(items_dir, "c", need, SEED_C)
    print(f"{len(need)} items need a third read: {len(tags)} batches")


def kappa(pairs: list[tuple[str, str]]) -> tuple[float, float]:
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    pa = Counter(a for a, _ in pairs)
    pb = Counter(b for _, b in pairs)
    pe = sum(pa[c] * pb[c] for c in ("genuine", "gamed")) / (n * n)
    return po, (po - pe) / (1 - pe) if pe < 1 else float("nan")


def label(r: dict[str, str]) -> str:
    c = Counter(r.values())
    if c["genuine"] >= 2 and c["genuine"] > c["gamed"]:
        return "genuine"
    if c["gamed"] >= 2 and c["gamed"] > c["genuine"]:
        return "gamed"
    return "unread"


def strict_by_prompt(corner: str) -> dict[str, list[int]]:
    _, draws = hc.load_cell(f"{corner}={CORNERS[corner]}")
    by: dict[str, list[int]] = defaultdict(list)
    for (pid, _, _), d in sorted(draws.items()):
        by[pid].append(int(d["strict"]))
    return by


def boot_genuine(strict: dict, s: dict, labs: dict, reps: int = REPS, seed: int = SEED_BOOT):
    """Per rep: shared prompt resample; per corner Binomial draws per prompt; per corner item resample. Returns {corner: [rates]}."""
    rng = random.Random(seed)
    pids = sorted(set.intersection(*(set(v) for v in strict.values())))
    out: dict[str, list[float]] = {c: [] for c in strict}
    for _ in range(reps):
        idx = [rng.choice(pids) for _ in pids]
        for c, by in strict.items():
            r = 0.0
            for p in idx:
                n = len(by[p])
                ph = sum(by[p]) / n
                r += sum(rng.random() < ph for _ in range(n)) / n
            r /= len(idx)
            lab = labs[c]
            g = sum(rng.choice(lab) == "gamed" for _ in lab) / len(lab) if lab else 0.0
            out[c].append(r * (1 - s[c]) * (1 - g))
    return out


def pct(v: list[float]) -> tuple[float, float]:
    v = sorted(v)
    return v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]


def cmd_table(out: Path, items_dir: Path, reps: int = REPS) -> dict:
    key = {r["item"]: r for r in csv.DictReader((out / "key.csv").read_text().splitlines())}
    rd = reads(items_dir)
    both = [(rd[i]["a"], rd[i]["b"]) for i in key if "a" in rd[i] and "b" in rd[i]]
    po, kap = kappa(both)
    final = {i: label(rd[i]) for i in key}
    ref = [
        (final[i], r["dan_label"])
        for i, r in key.items()
        if r["dan_label"] and final[i] != "unread"
    ]
    summ = {r["cell"]: r for r in json.loads((out / "screen_summary.json").read_text())}
    strict = {c: strict_by_prompt(c) for c in CORNERS}
    labs = {
        c: [final[i] for i, r in key.items() if r["cell"] == c and final[i] != "unread"]
        for c in CORNERS
    }
    s = {c: summ[c]["screened_gamed"] / summ[c]["passes"] for c in CORNERS}
    rows = {}
    for c in CORNERS:
        lab = labs[c]
        g = sum(x == "gamed" for x in lab) / len(lab)
        lo, hi = hc.wilson(sum(x == "gamed" for x in lab), len(lab))
        rate = sum(sum(v) / len(v) for v in strict[c].values()) / len(strict[c])
        rows[c] = {"passes": summ[c]["passes"], "screened": summ[c]["screened_gamed"], "screened_share": s[c],
                   "items": sum(r["cell"] == c for r in key.values()), "read": len(lab), "read_gamed": sum(x == "gamed" for x in lab),
                   "read_gamed_share": g, "read_gamed_lo": lo, "read_gamed_hi": hi,
                   "gamed_share_of_passes": s[c] + (1 - s[c]) * g, "strict_rate": rate,
                   "genuine_rate": rate * (1 - s[c]) * (1 - g)}  # fmt: skip
    bs = boot_genuine(strict, s, labs, reps)
    for c in CORNERS:
        rows[c]["genuine_lo"], rows[c]["genuine_hi"] = pct(bs[c])
    con = {}
    for x, y, name in (
        ("B", "D", "B-D"),
        ("Aprime", "C", "A'-C"),
        ("A", "D", "A-D"),
        ("Aprime", "D", "A'-D"),
        ("Aprime", "A", "A'-A"),
    ):
        d = [a - b for a, b in zip(bs[x], bs[y], strict=True)]
        lo, hi = pct(d)
        pt = rows[x]["genuine_rate"] - rows[y]["genuine_rate"]
        raw = rows[x]["strict_rate"] - rows[y]["strict_rate"]
        con[name] = {"genuine_diff": pt, "lo": lo, "hi": hi, "raw_diff": raw, "genuine_over_raw": pt / raw if raw else None,
                     "survives": lo > 0}  # fmt: skip
        if name in RAW:
            con[name]["mostly_gamed"] = (lo <= 0 <= hi) and pt * 100 <= RAW[name] / 3
    res = {"kappa_read1_read2": kap, "raw_agreement": po, "n_items_read_twice": len(both),
           "third_reads": sum("c" in rd[i] for i in key), "unread": sum(v == "unread" for v in final.values()),
           "ref_agreement": {"same": sum(a == b for a, b in ref), "differ": sum(a != b for a, b in ref)},
           "corners": rows, "genuine_contrasts": con, "reps": reps, "seed": SEED_BOOT}  # fmt: skip
    (out / "round3_summary.json").write_text(json.dumps(res, indent=1) + "\n")
    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "item",
                "cell",
                "pid",
                "chunk",
                "i",
                "dan_label",
                "read_a",
                "read_b",
                "read_c",
                "label",
            ],
        )
        w.writeheader()
        for i, r in key.items():
            w.writerow(
                {
                    **r,
                    "read_a": rd[i].get("a", ""),
                    "read_b": rd[i].get("b", ""),
                    "read_c": rd[i].get("c", ""),
                    "label": final[i],
                }
            )
    return res


# ---- metrics-2's answer-style statistic (tools/answer_style.py: answer after the last </think>; no . ! ? = unpunctuated)
def ans(t: str) -> str:
    i = t.rfind("</think>")
    return t[i + 8 :].strip() if i >= 0 else ""


def style(sp: Path, gp: Path, pids: set, passes_only: bool = True) -> dict:
    g = {}
    for ln in gp.read_text().splitlines():
        r = json.loads(ln)
        if r["pid"] in pids:
            g[(r["pid"], r.get("chunk", 0), r.get("i", 0))] = r
    txt = {}
    with gzip.open(sp, "rt") as f:
        for ln in f:
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                break
            k = (r["pid"], r.get("chunk", 0), r.get("i", 0))
            if k in g:
                txt[k] = r.get("text", "")
    a = [
        ans(txt[k])
        for k, r in g.items()
        if k in txt and (hc.truthy(r["strict"]) or not passes_only)
    ]
    if not a:
        return {"n": 0}
    w = sorted(len(x.split()) for x in a)
    med = (w[(len(w) - 1) // 2] + w[len(w) // 2]) / 2
    return {
        "n": len(a),
        "unpunctuated_pct": 100 * sum(not re.search(r"[.!?]", x) for x in a) / len(a),
        "median_words": med,
    }


def cmd_style(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    reach = {ln.strip() for ln in hc.REACH.read_text().splitlines() if ln.strip()}
    res = {c: style(p / "samples.jsonl.gz", p / "graded.jsonl", reach) for c, p in CORNERS.items()}
    GO = hc.DAN / "artifacts/olmo_ifomega"
    res["s2300 stored (B_s2300, 32/prompt)"] = style(hc.REF_SAMPLES, hc.REF_GRADED, reach)
    res["dpo stored (F_dpo_ifb), passes"] = style(
        GO / "samples/F_dpo_ifb.jsonl.gz", GO / "graded/F_dpo_ifb.jsonl", reach
    )
    res["dpo stored (F_dpo_ifb), all draws"] = style(
        GO / "samples/F_dpo_ifb.jsonl.gz", GO / "graded/F_dpo_ifb.jsonl", reach, False
    )
    (out / "answer_style.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cmd", choices=["prep", "third", "table", "style"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--items-dir", type=Path)
    ap.add_argument("--reps", type=int, default=REPS)
    a = ap.parse_args()
    if a.cmd == "prep":
        cmd_prep(a.out, a.items_dir)
    elif a.cmd == "third":
        cmd_third(a.out, a.items_dir)
    elif a.cmd == "table":
        r = cmd_table(a.out, a.items_dir, a.reps)
        print(json.dumps({k: v for k, v in r.items() if k != "corners"}, indent=1))
        for c, v in r["corners"].items():
            print(c, {k: round(x, 3) if isinstance(x, float) else x for k, x in v.items()})
    else:
        for k, v in cmd_style(a.out).items():
            print(k, v)


if __name__ == "__main__":
    main()
