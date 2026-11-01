"""Blind spot-check of the labels: a stratified sample of distinct model proofs for an independent
labeller who sees TAXONOMY.md but not our labels, then the agreement.

    cd /home/cowifornia/code/research/nd-rl
    ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 code/.venv/bin/python -m \\
        code.experiments.current.proof_strategies_grpo_20261101.spotcheck sample|compare

sample:  OUT/spotcheck/blind.jsonl (id, sequent, numbered Fitch lines; no labels) and
         OUT/spotcheck/key/key.jsonl (our labels; the labeller must not open it).
compare: reads OUT/spotcheck/answers.jsonl ({id, uses: [...], working: [...], primary}) and
         writes spotcheck.json next to this file.
Strata (seed 20261101): 9 positives for each tracked strategy, drawn from the base and L3-end
reads on charles149 and t1320s250 (pruned length <= 16 lines), then random proofs up to 100.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import random
from typing import Any

from . import common as C
from .analyze import TRACK
from .classify import read_name

STRATS = [
    "direct",
    "and_intro",
    "cond_proof",
    "or_intro",
    "cases",
    "disj_syll",
    "neg_intro",
    "mod_tollens",
    "ex_falso",
    "contradiction",
    "dne",
    "lem",
    "lemma_reuse",
]
WORK = {
    "cond_proof": "cond_proof.working",
    "cases": "cases.genuine",
    "neg_intro": "neg_intro.working",
    "contradiction": "contradiction.working",
}
STRATA = ["work_red", "templ_red", "mt", "ds", "work_neg", "gen_cases", "lem", "dne", "efq"]
SOURCES = [
    ("charles149", "s1g3", 1024),
    ("charles149", C.END, 1024),
    ("t1320s250", "s1g3", 512),
    ("t1320s250", C.END, 512),
]
D = C.OUT / "spotcheck"
RULE = {
    "PR": "premise",
    "AS": "assumption",
    "R": "reiterate",
    "ANDI": "∧I",
    "ANDE1": "∧E1",
    "ANDE2": "∧E2",
    "IMPE": "→E",
    "IMPI": "→I",
    "ORI1": "∨I1",
    "ORI2": "∨I2",
    "ORE": "∨E",
    "NEGE": "¬E",
    "NEGI": "¬I",
    "BOTE": "⊥E (ex falso)",
    "DN": "DN (¬¬ elimination)",
}


def fitch(nd: str) -> list[str]:
    """ND body -> numbered lines with box bars, unicode connectives (the spec's rules)."""
    C.dan_strategies()
    from patterns import fstr, parse  # type: ignore[import-not-found]

    out = []
    for ln in parse(nd):
        f = fstr(ln["formula"]).replace("~", "¬").replace("&", "∧").replace(" v ", " ∨ ")
        f = f.replace(">", "→").replace("F", "⊥")
        refs = " ".join(str(r) for r in ln["refs"])
        out.append(
            f"{ln['idx']:>3} {'| ' * ln['depth']}{f}    "
            f"[{RULE[ln['rule']]}{' ' + refs if refs else ''}]"
        )
    return out


def sample() -> None:
    rng = random.Random(20261101)
    pool: list[dict[str, Any]] = []
    seen = set()
    for st, stem, k in SOURCES:
        with gzip.open(C.OUT / "reads" / f"{read_name(st, stem, 0, k)}.json.gz", "rt") as f:
            rows = json.load(f)
        for name, r in sorted(rows.items()):
            for nd, _, lab in r["proofs"]:
                if lab["Lp"] > 16 or (r["prompt"], nd) in seen:
                    continue
                seen.add((r["prompt"], nd))
                pool.append(
                    {
                        "src": f"{st}/{stem}",
                        "name": name,
                        "prompt": r["prompt"],
                        "nd": nd,
                        "lab": lab,
                    }
                )
    rng.shuffle(pool)
    chosen: list[dict[str, Any]] = []
    ids = set()
    for s in STRATA:
        f = TRACK[s][1]
        got = [x for x in pool if f(x["lab"]) and id(x) not in ids][:9]
        for x in got:
            ids.add(id(x))
            chosen.append({**x, "stratum": s})
    for x in pool:
        if len(chosen) >= 100:
            break
        if id(x) not in ids:
            ids.add(id(x))
            chosen.append({**x, "stratum": "random"})
    rng.shuffle(chosen)
    (D / "key").mkdir(parents=True, exist_ok=True)
    with open(D / "blind.jsonl", "w") as fb, open(D / "key/key.jsonl", "w") as fk:
        for i, x in enumerate(chosen):
            seq = x["prompt"].replace("THM ", "").replace(" PRF", "").replace(" SEQ ", " ⊢ ")
            seq = seq.replace("~", "¬").replace("&", "∧").replace(" v ", " ∨ ").replace(">", "→")
            fb.write(
                json.dumps(
                    {"id": i, "sequent": seq, "lines": fitch(x["nd"]), "nd": x["nd"]},
                    ensure_ascii=False,
                )
                + "\n"
            )
            fk.write(
                json.dumps(
                    {
                        "id": i,
                        "stratum": x["stratum"],
                        "src": x["src"],
                        "name": x["name"],
                        "u": x["lab"]["u"],
                        "sub": x["lab"]["sub"],
                        "pri": x["lab"]["pri"],
                    }
                )
                + "\n"
            )
    print(
        f"wrote {len(chosen)} blind items to {D / 'blind.jsonl'}; "
        f"strata {collections.Counter(x['stratum'] for x in chosen)}"
    )


def compare() -> None:
    with open(D / "key/key.jsonl") as f:
        key = {r["id"]: r for r in map(json.loads, f)}
    with open(D / "answers.jsonl") as f:
        ans = {r["id"]: r for r in map(json.loads, f)}
    agree: collections.Counter[str] = collections.Counter()
    pos: collections.Counter[str] = collections.Counter()
    dis: dict[str, list[int]] = collections.defaultdict(list)
    for i, k in key.items():
        a = ans.get(i)
        if a is None:
            dis["missing"].append(i)
            continue
        for s in STRATS:
            ours, theirs = s in k["u"], s in a["uses"]
            agree[f"uses:{s}"] += ours == theirs
            pos[f"uses:{s}"] += ours
            if ours != theirs:
                dis[f"uses:{s}"].append(i)
        for s, w in WORK.items():
            ours, theirs = w in k["sub"], s in a.get("working", [])
            agree[f"working:{s}"] += ours == theirs
            pos[f"working:{s}"] += ours
            if ours != theirs:
                dis[f"working:{s}"].append(i)
        agree["primary"] += k["pri"] == a["primary"]
        if k["pri"] != a["primary"]:
            dis["primary"].append(i)
        agree["all_uses"] += set(k["u"]) == set(a["uses"])
    n = len(key)
    out = {
        "n": n,
        "answered": len(ans),
        "agree": {kk: f"{v}/{n}" for kk, v in sorted(agree.items())},
        "positives_ours": dict(pos),
        "disagreements": dict(dis),
        "strata": dict(collections.Counter(k["stratum"] for k in key.values())),
    }
    (C.HERE / "spotcheck.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["agree"], indent=1))
    print("disagreements", {kk: v for kk, v in dis.items()})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("what", choices=["sample", "compare"])
    a = ap.parse_args(argv)
    sample() if a.what == "sample" else compare()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
