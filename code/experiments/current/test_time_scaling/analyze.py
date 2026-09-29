#!/usr/bin/env python3
"""Tables from ttscale.py results: pass@k (or solve rate) overall and at reference length > 10,
sampled tokens per theorem, wall-clock, and method 1's resampling and salvage stats.

  python3 analyze.py <results.json>...      # markdown to stdout
  python3 analyze.py --subset <r.json> ...  # score every run on r.json's theorems only

Method 1 (guided) results are re-read at every cap c <= the run's cap: a sample with more than c
rejections counts as failed after the tokens of its (c+1)-th rejection. i.i.d. methods' wall-clock
at k is the k = 64 run's time scaled by k / 64 (sampling cost is linear in the number of samples).
"""

import json
import math
import sys

KS = [1, 2, 4, 8, 16, 32, 64]
CAPS = [0, 1, 2, 4, 10, 32]
LONG = 10  # the "> 10 lines" panel


def pass_at_k(n, c, k):
    """Unbiased estimator (Chen et al., 2021)."""
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


def at_cap(sample, cap):
    """(ok, sampled tokens) of a guided sample had the run stopped at `cap` rejections."""
    ok, tok, rej = sample[0], sample[1], sample[2]
    if len(rej) > cap:
        return 0, rej[cap][1]
    return ok, tok


def iid_rows(d, label, cap=None):
    """pass@k per k for plain samples, or guided samples read at `cap`."""
    ths = d["theorems"]
    per = [[at_cap(s, cap) if cap is not None else (s[0], s[1]) for s in t["samples"]] for t in ths]
    n = len(per[0])
    counted = all(x[1] is not None for p in per for x in p)  # guided_kv counts no tokens
    tok = sum(x[1] for p in per for x in p) / (len(per) * n) if counted else float("nan")
    secs = d["stats"]["method_secs"]
    row = {"label": label, "tok_per_sample": tok}
    for k in KS:
        if k > n:
            break
        pk = [pass_at_k(n, sum(x[0] for x in p), k) for p in per]
        row[k] = {
            "all": 100 * sum(pk) / len(pk),
            "long": sum(q for q, t in zip(pk, ths, strict=True) if t["L"] > LONG),
            "tok": k * tok,
            "secs": secs * k / n,
        }
    return row


def guided_stats(d):
    samples = [s for t in d["theorems"] for s in t["samples"]]
    rej = [r for s in samples for r in s[2]]
    acc = sum(s[3] for s in samples if len(s) > 3)
    causes = {}
    for r in rej:
        causes[r[0]] = causes.get(r[0], 0) + 1
    lines = [
        f"- samples {len(samples)}; rejections per sample {len(rej) / len(samples):.2f}; "
        f"samples with >= 1 rejection {100 * sum(bool(s[2]) for s in samples) / len(samples):.1f}%",
        "- rejection causes: "
        + ", ".join(f"{c} {100 * v / len(rej):.1f}%" for c, v in sorted(causes.items())),
        f"- rejections that repeat a segment already rejected at the same prefix: "
        f"{100 * sum(r[2] for r in rej) / max(len(rej), 1):.1f}%",
    ]
    if acc:
        lines.append(f"- rejected / all checked segments: {100 * len(rej) / (len(rej) + acc):.1f}%")
    lines.append("")
    lines.append("| cap | abandoned % | ok samples | salvaged (ok with >= 1 rejection) |")
    lines.append("| --- | --- | --- | --- |")
    for c in CAPS:
        ok = [at_cap(s, c)[0] for s in samples]
        salv = sum(o and len(s[2]) >= 1 for o, s in zip(ok, samples, strict=True))
        abandoned = sum(len(s[2]) > c for s in samples)
        lines.append(
            f"| {c} | {100 * abandoned / len(samples):.1f} | {sum(ok)} | {salv} "
            f"({100 * salv / max(sum(ok), 1):.1f}% of ok) |"
        )
    return "\n".join(lines)


def search_row(d, label):
    ths = d["theorems"]
    return {
        "label": label,
        "all": 100 * sum(t["solved"] for t in ths) / len(ths),
        "long": sum(t["solved"] for t in ths if t["L"] > LONG),
        "tok": sum(t["tok"] for t in ths) / len(ths),
        "secs": d["stats"]["method_secs"],
    }


def main():
    iid, search, extra = [], [], []
    files, names = sys.argv[1:], None
    if files[0] == "--subset":
        with open(files[1]) as f:
            names = {t["name"] for t in json.load(f)["theorems"]}
        files = files[2:]
    for fn in files:
        with open(fn) as f:
            d = json.load(f)
        if names is not None:
            d["theorems"] = [t for t in d["theorems"] if t["name"] in names]
            d["stats"]["method_secs"] *= len(d["theorems"]) / d["n"]  # pro rata
            d["n"] = len(d["theorems"])
        a, name = d["args"], fn.rsplit("/", 1)[-1].removesuffix(".json")
        n_long = sum(t["L"] > LONG for t in d["theorems"])
        head = f"{name} ({a['pool']}, n {d['n']}, {n_long} with L > {LONG})"
        if a["method"] in ("plain", "guided_kv"):
            iid.append(iid_rows(d, f"{head}: {a['method']}"))
        elif a["method"] == "guided":
            for c in CAPS:
                if c <= a["cap"]:
                    iid.append(iid_rows(d, f"{head}: guided cap {c}", cap=c))
            extra.append(f"### {head}: guided resampling\n\n{guided_stats(d)}\n")
        else:
            search.append(search_row(d, head))
    print("## i.i.d. sampling: pass@k % (all) / expected solved with L > 10 / tokens per theorem\n")
    print("| run | " + " | ".join(f"k={k}" for k in KS) + " |")
    print("| --- |" + " --- |" * len(KS))
    for r in iid:
        cells = [
            f"{r[k]['all']:.1f} / {r[k]['long']:.1f} / {r[k]['tok']:.0f}" if k in r else ""
            for k in KS
        ]
        print(f"| {r['label']} | " + " | ".join(cells) + " |")
    print(
        "\nWall-clock seconds at k = 64: "
        + "; ".join(f"{r['label']} {r[64]['secs']:.0f}" for r in iid if 64 in r)
    )
    if search:
        print("\n## Search\n")
        print("| run | solved % | solved with L > 10 | sampled tokens / theorem | wall s |")
        print("| --- | --- | --- | --- | --- |")
        for r in search:
            cells = [f"{r['all']:.1f}", str(r["long"]), f"{r['tok']:.0f}", f"{r['secs']:.0f}"]
            print(f"| {r['label']} | " + " | ".join(cells) + " |")
    for e in extra:
        print("\n" + e)


if __name__ == "__main__":
    main()
