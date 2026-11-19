# ruff: noqa
# Scratch analysis script written by a research agent on 2026-11-18; not reviewed.
"""Independent re-implementations of the recommended filters straight from the proof text (regex / token level),
compared with motifs.py on all pretraining records, plus random examples to read."""

import re, random, collections
from ndlib import *
from motifs import motifs

LINE = re.compile(r"^N(\d+) ((?:\| )*)(.*) : (\w+)((?: N\d+)*)$")


def plain(proof):
    rows = {}
    for s in proof.split(" ; "):
        if s == "QED":
            continue
        m = LINE.match(s)
        assert m, s
        rows[int(m.group(1))] = (
            m.group(2).count("|"),
            m.group(3),
            m.group(4),
            [int(x[1:]) for x in m.group(5).split()],
        )
    return rows


def f_chain2(rows):
    return any(r == "IMPE" and rows[c[1]][2] == "IMPE" for d, f, r, c in rows.values())


def f_curried(rows):
    return any(r == "IMPE" and rows[c[0]][2] == "IMPE" for d, f, r, c in rows.values())


def f_bote_box(rows):
    return any(r == "BOTE" and d >= 1 for d, f, r, c in rows.values())


def f_depth3(rows):
    return any(d >= 3 for d, f, r, c in rows.values())


def f_ore_work(rows):
    return any(r == "ORE" and (c[1] != c[2] or c[3] != c[4]) for d, f, r, c in rows.values())


def f_ore_derived(rows):
    return any(r == "ORE" and rows[c[0]][2] not in ("PR", "AS") for d, f, r, c in rows.values())


def f_ore_in_impi(rows):
    # an ORE line at depth>=1 whose innermost enclosing box is closed by IMPI
    closers = {c[0]: r for d, f, r, c in rows.values() if r in ("IMPI", "NEGI")}
    for i, (d, f, r, c) in rows.items():
        if r == "ORE" and d >= 1:
            j = (
                c[1] - 1
            )  # walk back from the first branch's AS to the AS that opened the enclosing box
            while j in rows and not (
                rows[j][2] == "AS"
                and rows[j][0] == d
                and all(rows[k][0] >= d for k in range(j, i + 1))
            ):
                j -= 1
            if j in rows and closers.get(j) == "IMPI":
                return True
    return False


def f_negi_in_impi(rows):
    closers = {c[0]: (r, i) for i, (d, f, r, c) in rows.items() if r in ("IMPI", "NEGI")}
    for i, (d, f, r, c) in rows.items():
        if r == "NEGI" and d >= 1:
            j = c[0] - 1
            while j in rows and not (
                rows[j][2] == "AS"
                and rows[j][0] == d
                and all(rows[k][0] >= d for k in range(j, i + 1))
            ):
                j -= 1
            if j in rows and closers.get(j, ("", 0))[0] == "IMPI":
                return True
    return False


CHK = {
    "mp_chain2": f_chain2,
    "mp_curried": f_curried,
    "bote_in_box": f_bote_box,
    "depth>=3": f_depth3,
    "ore_any_work": f_ore_work,
    "ore_on_derived": f_ore_derived,
    "ore_in_impi": f_ore_in_impi,
    "negi_in_impi": f_negi_in_impi,
}
agree = collections.Counter()
dead = 0
n = 0
samples = collections.defaultdict(list)
random.seed(0)
for r in pretrain():
    p = Proof(r["proof"])
    ms = motifs(p)
    rows = plain(r["proof"])
    n += 1
    dead += any(not p.L[i].live for i in p.order if p.L[i].rule != "PR")
    for k, f in CHK.items():
        a, b = k in ms, f(rows)
        agree[(k, a, b)] += 1
        if a and random.random() < 0.02 and len(samples[k]) < 40:
            samples[k].append(r)
print("pretraining records", n, "with a dead (unused) body line:", dead)
for k in CHK:
    print(
        f"{k:16s} both {agree[(k, True, True)]:6d}  detector-only {agree[(k, True, False)]:4d}  text-only {agree[(k, False, True)]:4d}"
    )
for k in CHK:
    print("==", k)
    for r in random.sample(samples[k], min(3, len(samples[k]))):
        p = Proof(r["proof"])
        print("   ", r["name"], "|", " ; ".join(p.L[i].txt for i in p.order if p.L[i].live))
