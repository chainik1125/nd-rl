"""3-premise versions of the curriculum pools' problems, by uncurrying (CPU only).

    cd /scratch/cowifornia/nd-rl/think/nd-rl-curr
    export ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= PFG_MEM_GB=3.0
    PY=/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python
    X=code.experiments.current.thinking_traces_20261005
    O=/scratch/cowifornia/nd-rl/think/runs/pools/curriculum_3p
    for b in L2 L3; do
      PFG_MIN_L=6 PFG_MAX_L=15 PFG_MAX_GAP=9 nice -n 10 $PY -m $X.curriculum.uncurry3 --out $O \
          build --band $b --workers 12
      LEAN_GATE_WORKERS=12 nice -n 10 $PY -m $X.curriculum.uncurry3 --out $O lean --band $b
    done
    nice -n 10 $PY -m $X.curriculum.uncurry3 --out $O finish --workers 12
    (cd ../nd-rl-v2 && $PY -m $X.grpo_v2.preflight $O/L2_3p.jsonl $O/L3_3p.jsonl) > $O/preflight.log
    $PY -m $X.curriculum.uncurry3 --out $O report

Every pool problem has 0-2 premises; most of test_13_20 has 3. For a GRPO mechanism test
(`grpo.py --targets_extra`), each row of a band's pool (`curriculum/L2/pool.jsonl`,
`L3/pool.jsonl`; never a test split) is rewritten as follows.

* **Uncurry**: leading antecedents of the goal move into the premises, appended after the
  existing ones in order, until there are exactly 3 (`G |- A > (B > C)` becomes `G, A, B |- C`).
  Skipped: too few leading antecedents (`cannot_reach_3`), a moved antecedent repeats a premise
  (`dup_premise`), the new goal is a premise (`goal_is_premise`). The curried normal form
  (`heldout.curried`) does not change, so the curriculum's curried checks still hold.
* **Constructed proof**: the original reference (shortest) or constructed proof with the moved
  antecedents' IMPI boxes removed (`strip_ref`, `strip_gen`: each box's AS line becomes a premise
  line, its IMPI line goes, the lines inside move out one level), or, when neither proof ends in
  those boxes, the reference proof under the new premises followed by one IMPE per antecedent
  (`impe_ref`). The shortest one nd_verify accepts is kept.
* **Statement gate**: `pfgates.shape_gate` with its premise cap (2) and connective cap (7)
  lifted; its other checks are the pools': no repeated or redundant premise, premises jointly
  satisfiable, the goal not a premise, at most 2 irrelevant atom occurrences, no substitution
  instance of a textbook72 theorem either way, and a valid constructed proof.
* **Shortest proof**: `pfgates.certify`, the exact search the pools used, in the band
  `PFG_MIN_L`-`PFG_MAX_L`. The gap limit `PFG_MAX_GAP` is 9 because an `impe_ref` proof can be 3
  lines per antecedent longer than the shortest one. Rows are kept whatever the target-structure
  verdict (`cert_status` ok / not_needed / no_structure, `families` possibly empty), since this
  file is for premise handling; `finish --require_family` keeps only `ok`, as the pools did.
  `reference_proof` is one of possibly several shortest proofs: the search order depends on
  Python's per-process hash seed, so a re-run can return another proof of the same length, and
  with it other families (`min_lines` cannot change).
* **Lean**: `hard712/run.py`'s `lean` step on both proofs (`lean` here also records its time);
  `finish` keeps rows whose constructed and reference proofs both pass.

`finish` writes `OUT/L2_3p.jsonl` and `OUT/L3_3p.jsonl` in the pools' row format
(`run.py final_row`) with name `u3L2_<orig name>`, `source` `uncurry3_L2` / `uncurry3_L3`, the new
statement's class keys, and `orig_name`, `orig_prompt`, `orig_min_lines`, `orig_n_prem`,
`orig_families`, `n_moved`, `construction`, `cert_status`. A row is dropped if its class, as
written or curried (`heldout.keys`), is in textbook72, Charles's 2026-10-02 release, a sealed
test, a curriculum test split or an eval subset, if it hits the curriculum's exclusion index (as
written, like the pools), or if it repeats another row's class. `report` writes `report.md` from
`report.json` and the GRPO v2 pre-flight's log.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import multiprocessing as mp
import pickle
import time
from pathlib import Path
from typing import Any

from code.experiments.current.thinking_traces_20261005.curriculum import heldout as HO
from code.experiments.current.thinking_traces_20261005.hard712 import pfgates as GT
from code.experiments.current.thinking_traces_20261005.hard712 import run as RN

GC = GT.GC
R = "/scratch/cowifornia/nd-rl/think/runs"
CURR = Path(f"{R}/pools/curriculum")
INDEX = CURR / "work" / "excl_index.pkl"
N_PREM = 3
BANDS = ("L2", "L3")
KEEP = ("ok", "not_needed", "no_structure")  # certify statuses with a certified shortest proof
#: Held-out and evaluation sets beyond `heldout.FILES` and `heldout.SEALED`.
HELD = {
    "charles_clean": f"{R}/pools/charles_1002/charles_clean.jsonl",
    "curriculum_L2_test_1k": str(CURR / "L2" / "test_1k.jsonl"),
    "curriculum_L3_test_1k": str(CURR / "L3" / "test_1k.jsonl"),
    **{Path(p).stem: p for p in sorted(glob.glob(f"{R}/eval/matched/sets/*.jsonl"))},
}
# shape_gate's premise cap (2), which every 3-premise statement fails, and its connective cap (7),
# which cannot bind (uncurrying only removes connectives), are lifted; the rest of the gate
# applies unchanged. Set at import, so in every spawned worker too.
GT.MAX_PREM = N_PREM
GT.MAX_CONN = 10**6


# ---------------------------------------------------------------------- statement and proofs
def uncurry(prem: list[Any], goal: Any) -> tuple[list[Any], Any, list[Any], str | None]:
    """(premises, goal, moved antecedents, skip reason or None) with N_PREM premises."""
    ante = []
    for _ in range(N_PREM - len(prem)):
        if goal[0] != "imp":
            return prem, goal, ante, "cannot_reach_3"
        ante.append(goal[1])
        goal = goal[2]
    new = [*prem, *ante]
    if len(new) != N_PREM:
        return prem, goal, ante, "cannot_reach_3"
    if len(set(new)) < len(new):
        return new, goal, ante, "dup_premise"
    if goal in new:
        return new, goal, ante, "goal_is_premise"
    return new, goal, ante, None


def prompt_of(prem: list[Any], goal: Any) -> str:
    return "THM " + " , ".join(GC.fstr(p) for p in prem) + " SEQ " + GC.fstr(goal) + " PRF"


def proof_text(lines: list[dict[str, Any]]) -> str:
    """Lines {depth, f, rule, refs} (refs: 1-based line numbers) in the fork's text format."""
    segs = []
    for i, ln in enumerate(lines, 1):
        bars = ["|"] * ln["depth"]
        refs = [f"N{r}" for r in ln["refs"]]
        segs.append(" ".join([f"N{i}", *bars, GC.fstr(ln["f"]), ":", ln["rule"], *refs]))
    return " ; ".join(segs) + " ; QED"


def _pr(f: Any) -> dict[str, Any]:
    return {"depth": 0, "f": f, "rule": "PR", "refs": []}


def strip_boxes(proof: str, n_prem: int, ante: list[Any]) -> list[dict[str, Any]] | None:
    """Lines of a proof of `G, A1 .. Ak |- C` from one of `G |- A1 > (.. > (Ak > C))` whose last
    k lines are the IMPI lines closing nested boxes for A1 .. Ak, or None. The boxes' AS lines
    become premise lines after G's, their IMPI lines go, every line inside a box moves out one
    level per box."""
    lines = GC.parse_proof(proof)
    by_n = {ln["n"]: ln for ln in lines}
    last = lines[-1]["n"]
    opens: list[int] = []
    for i, a in enumerate(ante):
        close = lines[-1 - i]
        if close["rule"] != "IMPI" or close["depth"] != i:
            return None
        s, e = close["refs"]
        if e != close["n"] - 1 or (opens and s <= opens[-1]):
            return None
        if by_n[s]["rule"] != "AS" or by_n[s]["depth"] != i + 1 or by_n[s]["f"] != a:
            return None
        opens.append(s)
    drop = {last - i for i in range(len(ante))}
    num = {ln["n"]: ln["n"] for ln in lines[:n_prem]}
    num.update({s: n_prem + 1 + i for i, s in enumerate(opens)})
    body = [ln for ln in lines[n_prem:] if ln["n"] not in drop and ln["n"] not in num]
    num.update({ln["n"]: n_prem + len(ante) + 1 + j for j, ln in enumerate(body)})
    out = [_pr(ln["f"]) for ln in lines[:n_prem]] + [_pr(a) for a in ante]
    for ln in body:
        if any(r not in num for r in ln["refs"]):
            return None
        out_by = sum(s < ln["n"] <= last - 1 - i for i, s in enumerate(opens))
        refs = [num[r] for r in ln["refs"]]
        out.append({"depth": ln["depth"] - out_by, "f": ln["f"], "rule": ln["rule"], "refs": refs})
    return out


def append_impe(proof: str, n_prem: int, ante: list[Any]) -> list[dict[str, Any]]:
    """Lines of a proof of `G, A1 .. Ak |- C`: the proof of `G |- A1 > (.. > (Ak > C))` after the
    new premise lines, then one IMPE per antecedent."""
    lines = GC.parse_proof(proof)
    k = len(ante)
    out = [_pr(ln["f"]) for ln in lines[:n_prem]] + [_pr(a) for a in ante]
    for ln in lines[n_prem:]:
        refs = [r if r <= n_prem else r + k for r in ln["refs"]]
        out.append({"depth": ln["depth"], "f": ln["f"], "rule": ln["rule"], "refs": refs})
    f = lines[-1]["f"]
    for i in range(k):
        f = f[2]
        out.append({"depth": 0, "f": f, "rule": "IMPE", "refs": [len(out), n_prem + 1 + i]})
    return out


def construct(row: dict[str, Any], prompt: str, n_prem: int, ante: list[Any]) -> Any:
    """(kind, proof text, lines) of the shortest valid constructed proof, or None."""
    tries = [
        ("strip_ref", strip_boxes(row["reference_proof"], n_prem, ante)),
        ("strip_gen", strip_boxes(row["gen_proof"], n_prem, ante)),
        ("impe_ref", append_impe(row["reference_proof"], n_prem, ante)),
    ]
    best = None
    for kind, lines in tries:
        if lines is None:
            continue
        text = proof_text(lines)
        ok, _, n = GT.verify_text(prompt + " " + text)
        if ok and n == len(lines) and (best is None or n < best[2]):
            best = (kind, text, n)
    return best


# ---------------------------------------------------------------------- build
def build_one(row: dict[str, Any]) -> dict[str, Any]:
    """One pool row -> a certified candidate (`status` from certify) or a skip record."""
    prem0, goal0 = GT.statement(row["prompt"])
    prem, goal, ante, reason = uncurry(prem0, goal0)
    base = {"orig_name": row["name"], "n_moved": len(ante)}
    if reason:
        return dict(base, status=reason)
    prompt = prompt_of(prem, goal)
    c = construct(row, prompt, len(prem0), ante)
    if c is None:
        return dict(base, status="no_construction")
    kind, text, _ = c
    reason, info = GT.shape_gate(prompt, text)
    if reason:
        return dict(base, status="shape:" + reason, construction=kind)
    k, pk = RN.class_keys(prompt)
    cand = dict(
        base,
        prompt=prompt,
        proof=text,
        **info,
        key=k,
        pkey=pk,
        construction=kind,
        orig_prompt=row["prompt"],
        orig_min_lines=row["min_lines"],
        orig_n_prem=len(prem0),
        orig_families=row["families"],
        recipe=row["recipe"],
        seed=row["seed"],
        subst=row.get("subst"),
        discharged=row.get("discharged", 0),
    )
    out = RN.certify_safe(cand)
    if out["status"] in KEEP:
        out.setdefault("families", [])
        out.setdefault("necessity", {})
    return out


def cmd_build(a: argparse.Namespace) -> None:
    band = (GT.MIN_L, GT.MAX_L, GT.MAX_GAP)
    assert band == (6, 15, 9), "set PFG_MIN_L=6 PFG_MAX_L=15 PFG_MAX_GAP=9"
    work = Path(a.out) / a.band / "work"
    work.mkdir(parents=True, exist_ok=True)
    rows = RN.jl(CURR / a.band / "pool.jsonl")[: a.limit or None]
    c: collections.Counter[str] = collections.Counter(pool_rows=len(rows))
    done: set[str] = set()  # resume: rows an earlier run already wrote (counted again here)
    for name in ("gated.jsonl", "rejects.jsonl"):
        if (work / name).exists():
            prev = RN.read_complete(work / name)[0]
            done |= {r["orig_name"] for r in prev}
            c.update(r["status"] for r in prev)
    c["resumed"] = len(done)
    todo = [r for r in rows if r["name"] not in done]
    t0 = time.time()
    RN.log(f"{a.band}: {len(rows)} pool rows, {len(todo)} to do, {a.workers} workers")
    with (
        mp.get_context("spawn").Pool(
            a.workers, initializer=RN._init, initargs=(RN.MEM_GB,), maxtasksperchild=200
        ) as pool,
        open(work / "gated.jsonl", "a") as fg,
        open(work / "rejects.jsonl", "a") as fr,
    ):
        for i, r in enumerate(pool.imap(build_one, todo, chunksize=1), 1):
            c[r["status"]] += 1
            if r["status"] in KEEP:
                fg.write(json.dumps(r) + "\n")
            else:
                keep = ("orig_name", "status", "n_moved", "construction", "min_lines", "gen_lines")
                fr.write(json.dumps({k: r[k] for k in keep if k in r}) + "\n")
            if i % 100 == 0:
                fg.flush()
                fr.flush()
                RN.log(f"{a.band}: {i}/{len(todo)} {time.time() - t0:.0f}s {dict(c)}")
    c["secs"] = round(time.time() - t0)
    RN.wjson(work / "build_report.json", dict(c))
    RN.log(f"{a.band} build done: {dict(c)}")


def cmd_lean(a: argparse.Namespace) -> None:
    """`run.py lean` on the band's certified rows, timed."""
    t0 = time.time()
    RN.cmd_lean(argparse.Namespace(out=str(Path(a.out) / a.band)))
    work = Path(a.out) / a.band / "work"
    rep = RN.rjson(work / "lean_report.json")
    RN.wjson(work / "lean_report.json", dict(rep, secs=round(time.time() - t0)))


# ---------------------------------------------------------------------- finish
def feats(r: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    return RN._feats(r)


def final(r: dict[str, Any], band: str, tok: Any, D: Any) -> dict[str, Any]:
    """The pools' row format (`run.py final_row`) plus the rewrite's provenance."""
    fam = r["families"]
    # final_row names a primary family; a row without a needed structure has none
    row = RN.final_row(dict(r, families=fam or ["depth2"]), f"u3{band}_{r['orig_name']}", tok, D)
    row["family"], row["families"] = (row["family"] if fam else None), fam
    row["source"] = f"uncurry3_{band}"
    for k in ("orig_name", "orig_prompt", "orig_min_lines", "orig_n_prem", "orig_families"):
        row[k] = r[k]
    row["n_moved"], row["construction"], row["cert_status"] = (
        r["n_moved"],
        r["construction"],
        r["status"],
    )
    return row


def index_set(path: str) -> str:
    """An indexed file's directory, as a short label."""
    return str(Path(path).parent).removeprefix("/scratch/cowifornia/nd-rl/")


def screen(
    rows: list[dict[str, Any]],
    a: argparse.Namespace,
    hidx: dict[str, set[str]],
    idx: dict[str, Any],
    seen: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The Lean, held-out, index and duplicate drops -> (kept rows, counts and hit tallies)."""
    c: collections.Counter[str] = collections.Counter()
    held: collections.Counter[str] = collections.Counter()
    by_set: collections.Counter[str] = collections.Counter()
    by_file: collections.Counter[str] = collections.Counter()
    keep = []
    for r in rows:
        if not r["lean_ref_ok"]:
            c["lean_reference_rejected"] += 1
            continue
        if a.require_family and r["status"] != "ok":
            c["no_needed_structure"] += 1
            continue
        h = HO.hits(hidx, r["prompt"])
        if h:
            c["held_out"] += 1
            held.update(h)
            continue
        h = idx["keys"].get(r["key"], set()) | idx["pkeys"].get(r["pkey"], set())
        if h:
            c["index"] += 1
            by_set.update({index_set(x) for x in h})
            by_file.update(h)
            continue
        ck = HO.keys(r["prompt"])
        if ck & seen:
            c["dup_class"] += 1
            continue
        seen |= ck
        keep.append(r)
    hits = dict(held_out=dict(held), index_by_dir=dict(by_set), index_by_file=dict(by_file))
    return keep, dict(counts=dict(c), **hits)


def hist(xs: list[Any]) -> dict[str, int]:
    return {str(k): v for k, v in sorted(collections.Counter(xs).items())}


def stats(out: list[dict[str, Any]]) -> dict[str, Any]:
    drop = [r["orig_min_lines"] - r["min_lines"] for r in out]
    moved = [r["n_moved"] for r in out]
    return dict(
        n=len(out),
        min_lines=hist([r["min_lines"] for r in out]),
        orig_min_lines=hist([r["orig_min_lines"] for r in out]),
        mean_min_lines=round(sum(r["min_lines"] for r in out) / max(1, len(out)), 2),
        n_prem=hist([r["n_prem"] for r in out]),
        n_moved=hist(moved),
        drop_minus_moved=hist([d - m for d, m in zip(drop, moved, strict=True)]),
        drop_by_moved={
            str(k): hist([d for d, m in zip(drop, moved, strict=True) if m == k])
            for k in sorted(set(moved))
        },
        gen_minus_min=hist([r["gen_lines"] - r["min_lines"] for r in out]),
        construction=hist([r["construction"] for r in out]),
        cert_status=hist([r["cert_status"] for r in out]),
        families=hist([",".join(r["families"]) or "none" for r in out]),
        n_irrelevant=hist([r["n_irrelevant"] for r in out]),
    )


def cmd_finish(a: argparse.Namespace) -> None:
    from code.thinking.dan import load
    from code.thinking.tok import ThinkTokenizer

    t0 = time.time()
    root = Path(a.out)
    D, tok = load(), ThinkTokenizer()
    hidx = HO.heldout_index({**HO.FILES, **HO.SEALED, **HELD})
    with open(INDEX, "rb") as f:
        idx = pickle.load(f)
    seen: set[str] = set()
    rep: dict[str, Any] = {"held_out_sets": sorted({**HO.FILES, **HO.SEALED, **HELD})}
    for band in BANDS:
        work = root / band / "work"
        rows = RN.jl(work / "gated_lean.jsonl")
        keep, scr = screen(rows, a, hidx, idx, seen)
        with mp.get_context("spawn").Pool(a.workers) as p:
            fs = p.map(feats, keep, chunksize=20)
        for r, (t, pf) in zip(keep, fs, strict=True):
            r["attrs"], r["gap_thm"], r["gap_proof"] = RN.attrs(r, t, pf), t, pf
        out = [final(r, band, tok, D) for r in keep]
        assert len({r["name"] for r in out}) == len(out)
        RN.wjl(root / f"{band}_3p.jsonl", out)
        rep[band] = dict(
            build=RN.rjson(work / "build_report.json"),
            lean=RN.rjson(work / "lean_report.json"),
            screen=scr,
            **stats(out),
        )
        RN.log(f"{band}: {len(rows)} certified and Lean-checked, {len(out)} written; {scr}")
    rep["finish_secs"] = round(time.time() - t0)
    RN.wjson(root / "report.json", rep)


# ---------------------------------------------------------------------- report
def fmt(h: dict[str, int]) -> str:
    n = max(1, sum(h.values()))
    return ", ".join(f"{k}: {v:,} ({100 * v / n:.0f}%)" for k, v in h.items())


SKIPS = {
    "cannot_reach_3": "goal has too few leading antecedents to reach 3 premises",
    "dup_premise": "a moved antecedent repeats a premise",
    "goal_is_premise": "the new goal is a premise",
    "no_construction": "no constructed proof passes nd_verify",
}
#: shape_gate reasons; the first five rows of the table always show, the rest when they occur.
SHAPE = {
    "inconsistent": "premises jointly unsatisfiable",
    "redundant_prem": "a premise is redundant (the goal follows without it)",
    "irrelevant": "more than 2 irrelevant atom occurrences",
    "textbook_instance": "an instance of a textbook72 theorem",
    "textbook_general": "a textbook72 theorem is an instance of it",
    "goal_is_prem": "the goal is a premise",
    "dup_prem": "a repeated premise",
    "atoms": "not 2-3 atoms",
    "fdepth": "formula depth over 3",
    "dneg": "a double negation",
    "bot": "F in the statement",
    "degenerate": "a subformula X o X",
    "gen_len": "constructed proof outside the length window",
    "gen_invalid": "constructed proof invalid",
}
#: certify statuses other than KEEP; the first two always show.
CERT = ("search_timeout", "long", "short", "gap", "search_none", "memory")
DROPS = {
    "lean_reference_rejected": "Lean: reference proof rejected",
    "no_needed_structure": "no needed target structure (`--require_family`)",
    "held_out": "class of a held-out / sealed / eval set (as written or curried)",
    "index": "class in the curriculum's exclusion index (as written)",
    "dup_class": "class repeats another row's",
}


def funnel(rep: dict[str, Any]) -> list[tuple[str, list[Any]]]:
    """(label, value per band) rows from pool rows in to rows out."""
    b = {x: rep[x]["build"] for x in BANDS}
    scr = {x: rep[x]["screen"]["counts"] for x in BANDS}
    lean = {x: rep[x]["lean"] for x in BANDS}

    def vals(k: str) -> list[Any]:
        return [b[x].get(k, 0) for x in BANDS]

    rows: list[tuple[str, list[Any]]] = [("pool rows in", vals("pool_rows"))]
    rows += [(f"skip: {label}", vals(k)) for k, label in SKIPS.items()]
    for i, (k, label) in enumerate(SHAPE.items()):
        if i < 5 or any(vals(f"shape:{k}")):
            rows.append((f"statement gate: {label}", vals(f"shape:{k}")))
    errs = sorted({k for x in BANDS for k in b[x] if k.startswith("error")})
    for i, k in enumerate((*CERT, *errs)):
        if i < 2 or any(vals(k)):
            rows.append((f"certifier: {k}", vals(k)))
    rows.append(("**certified** (shortest proof in 6-15 lines)", [lean[x]["n"] for x in BANDS]))
    for k in KEEP:
        rows.append((f"&nbsp;&nbsp;of which `cert_status` {k}", [b[x].get(k, 0) for x in BANDS]))
    rows.append(
        ("Lean: constructed proof rejected", [lean[x]["n"] - lean[x]["kept"] for x in BANDS])
    )
    for k, label in DROPS.items():
        if any(scr[x].get(k) for x in BANDS) or k != "no_needed_structure":
            rows.append((label, [scr[x].get(k, 0) for x in BANDS]))
    rows.append(("**rows out**", [f"**{rep[x]['n']:,}**" for x in BANDS]))
    return rows


def preflight_md(path: Path) -> list[str]:
    if not path.exists():
        return ["Not run (no `preflight.log`)."]
    log = path.read_text().splitlines()
    ok = [x for x in log if x.startswith("ok ")]
    bad = [x for x in log if x.lstrip().startswith("BAD")]
    over = [x for x in bad if "rows outside" not in x]
    out = [
        "`grpo_v2/preflight.py L2_3p.jsonl L3_3p.jsonl` (`preflight.log`): "
        f"{len(ok)} of {len(ok) + len(over)} overlap checks clean (prompt, key, renaming and "
        f"curried class, against each set it checks); verdict `{log[-1]}`."
    ]
    if not any("share" in x for x in over):
        out[0] += " L2_3p and L3_3p share no prompt, key or class."
    if over:
        out += ["", "Overlap failures:", "", *[f"    {x}" for x in over]]
    band = [x.strip() for x in bad if "rows outside" in x]
    if band:
        out += [
            "",
            "Its length-band check (L2 10-12, L3 13-15, the pools' bands) fails by construction, "
            "since uncurrying shortens the proofs:",
            "",
            *[f"    {x}" for x in band],
        ]
    return out


def band_md(band: str, x: dict[str, Any]) -> list[str]:
    s = x["screen"]
    idx = ", ".join(f"{k} {v}" for k, v in sorted(s["index_by_dir"].items(), key=lambda t: -t[1]))
    out = [f"## `{band}_3p.jsonl`: {x['n']:,} rows", ""]
    out.append(
        f"- shortest-proof lines (`min_lines`): {fmt(x['min_lines'])}; mean {x['mean_min_lines']}"
    )
    out.append(f"- their originals' `min_lines`: {fmt(x['orig_min_lines'])}")
    out.append(f"- premises: {fmt(x['n_prem'])}; antecedents moved: {fmt(x['n_moved'])}")
    out.append(f"- drop in `min_lines` minus antecedents moved: {fmt(x['drop_minus_moved'])}")
    for k, h in x["drop_by_moved"].items():
        out.append(f"  - {k} moved, drop: {fmt(h)}")
    out.append(f"- constructed minus shortest proof lines: {fmt(x['gen_minus_min'])}")
    out.append(f"- construction: {fmt(x['construction'])}")
    out.append(f"- `cert_status` (target-structure verdict): {fmt(x['cert_status'])}")
    out.append(f"- needed families: {fmt(x['families'])}")
    out.append(f"- irrelevant atom occurrences: {fmt(x['n_irrelevant'])}")
    out.append(f"- held-out / sealed / eval hits: {s['held_out'] or 'none'}")
    out.append(f"- exclusion-index hits, rows per directory: {idx or 'none'}")
    return [*out, ""]


def cmd_report(a: argparse.Namespace) -> None:
    root = Path(a.out)
    rep = RN.rjson(root / "report.json")
    md = ["# 3-premise (uncurried) curriculum rows", ""]
    md += [
        "Every row of `curriculum/L2/pool.jsonl` and `L3/pool.jsonl` with enough leading goal "
        "antecedents, rewritten with 3 premises and re-certified "
        "(`code/experiments/current/thinking_traces_20261005/curriculum/uncurry3.py`, branch "
        "`leon/thinking-curriculum`). For `grpo.py --targets_extra`; the test splits were not "
        "used as a source.",
        "",
    ]
    md += ["| step | " + " | ".join(BANDS) + " |", "|---|" + "---:|" * len(BANDS)]
    for label, vals in funnel(rep):
        md.append(
            f"| {label} | " + " | ".join(f"{v:,}" if isinstance(v, int) else v for v in vals) + " |"
        )
    md.append("")
    for band in BANDS:
        md += band_md(band, rep[band])
    md += ["## Pre-flight and held-out checks", "", *preflight_md(root / "preflight.log"), ""]
    md.append(
        "Before writing, rows were checked as written and in curried normal form against: "
        + ", ".join(rep["held_out_sets"])
        + ". Exclusion-index hits are possible because a 3-premise row is its original in a "
        "different (partly uncurried) form, and the curriculum compared rows with the index as "
        "written only."
    )
    md += [
        "",
        "`reference_proof`, and with it `families` and `cert_status`, is one of possibly several "
        "shortest proofs: the certifier's search order depends on Python's per-process hash "
        "seed, so a re-run can pick another proof of the same length (`min_lines` cannot change).",
    ]
    secs = [
        f"{x} build {rep[x]['build']['secs']} s, Lean {rep[x]['lean']['secs']} s" for x in BANDS
    ]
    md += [
        "",
        f"Time (CPU, 12 workers, nice 10): {'; '.join(secs)}; finish {rep['finish_secs']} s.",
    ]
    (root / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


# ---------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--band", choices=BANDS, required=True)
    b.add_argument("--workers", type=int, default=12)
    b.add_argument("--limit", type=int, default=0, help="only the first N pool rows (a pilot)")
    ln = sub.add_parser("lean")
    ln.add_argument("--band", choices=BANDS, required=True)
    f = sub.add_parser("finish")
    f.add_argument("--workers", type=int, default=12)
    f.add_argument("--require_family", action="store_true", help="only certify status ok")
    sub.add_parser("report")
    a = ap.parse_args()
    {"build": cmd_build, "lean": cmd_lean, "finish": cmd_finish, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
