"""The measurement sets of the two-authors experiment (SPEC.md), built once from Dan's long generator pool.

    python authors_sets.py build [--out DIR] [--procs 8] [--sizes marker_plain=500,route_unseen=300,marker_heldout_class=300]
    python authors_sets.py check [--out DIR]      # re-verify a built set folder against its manifest and the exclusions

Source: `/scratch/cowifornia/nd-rl/knockout_grpo/dan_data/pool_long_kh.jsonl` (518,843 renaming classes, pruned
length 7-14; Dan's `kh_gen.py`). These are measurement sets, not test sets: they may be copied to pods, and their
theorems are never trained on (no corpus record and no RL-pool theorem shares their class).

Exclusions (a pool record is dropped when its `key` or its premise-order-free key `pkey` matches any of these):
  * every corpus key: Dan's `train_k12dose1.jsonl.gz` (the 155,000 records every two-author corpus rewrites; the four
    corpora keep its statements) and every `TA/data/*.jsonl.gz` corpus present at build time;
  * every theorem of the RL pool `knockout_danbase/pools/dan_rl_targets.jsonl` (4,495);
  * every theorem of Dan's `transfer_track832.jsonl` (it holds the strict 106 of Dan's measure).
`key` is Dan's renaming class (atoms renamed by first appearance); `pkey` is the smallest such key over the premise
orders (at most 3! = 6), so a theorem whose premises are only permuted counts as the same.

Every set: K12's limits on the pool's own proof (at most 12 written lines, box depth at most 2, at most 3 premises),
the proof replays in the state format (`state_env.decompose(canon=True)`), STEP-ORDER ROOM on the pool proof (the
primary marker; `authors_profile`, i.e. `authors_corpus.detect`), stratified by the record's length 7-12 (equal
quotas, remainder to the longest lengths; a short stratum is filled from the same set's spare theorems, nearest length
first, and the manifest says so), drawn in a seeded random order of the pool (seed 20261111).

  marker_plain          needs no reductio: provable intuitionistically (Dan's G4ip `intuit.py`), the pool proof is not
                        sigma+ (neither with R lines followed nor by Dan's `knock`), consistent premises; step-order
                        room and room for at least one other marker habit (reiteration sites or a free conjunct site);
                        no F in the statement. Up to 40 % of each length stratum is drawn from theorems with a free
                        conjunct site (the rarest habit), the rest from the others.
  route_unseen          needs the route: classically valid but not intuitionistically (every proof uses DN) and no
                        double negation anywhere in the statement, so every DN must cite a derived double negation:
                        Dan's criterion for the strict 106 and for his reductio targets (`reductio_nodn`). The pool
                        proof is sigma+ and has step-order room. In this pool every such theorem mentions F (the 106 do
                        not; Dan built them from textbook schemata), so this set is also outside the RL pool's class;
                        232 of the corpus's 405 classical-only route records are of this kind. The in-class route
                        theorems are the strict 106 themselves, profiled from Dan's reads (authors_measure.py danread).
  marker_heldout_class  as marker_plain, but the statement mentions F (falsum). That class is absent from the RL pool
                        (0 of its 4,495 statements mention F; Dan's long-pool mode bans it) and present in K12.

Output (written once; a second build refuses): DIR/<set>.jsonl (one theorem per line: name, set, key, pkey, thm,
prompt, n_lines, n_prem, stratum, ref_room {order, reit, conj (free sites)}, ref_conj_all, ref_reit_fits (A's full
reiteration of the pool proof fits in 12 lines), ref_route, needs_route, has_F, has_nn, pool_index, ref_proof) and
DIR/manifest.json (seed, source and exclusion files with sha256 and counts, criteria, per-length counts, room
composition, the overlap checks, each set file's sha256).
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import authors_profile as AP  # noqa: E402

H = AP.H
TA = Path(os.environ.get("TA", "/scratch/cowifornia/nd-rl/two_authors"))
POOL = Path("/scratch/cowifornia/nd-rl/knockout_grpo/dan_data/pool_long_kh.jsonl")
SRC_CORPUS = Path("/scratch/cowifornia/nd-rl/knockout_grpo/dan_data/train_k12dose1.jsonl.gz")
RL_POOL = Path("/scratch/cowifornia/nd-rl/knockout_danbase/pools/dan_rl_targets.jsonl")
TRANSFER = Path(H.DAN) / "data" / "coe" / "transfer_track832.jsonl"
OUT = TA / "sets"
SEED = 20261111
SETS = ("marker_plain", "route_unseen", "marker_heldout_class")
SIZES = {"marker_plain": 500, "route_unseen": 300, "marker_heldout_class": 300}
PREFIX = {"marker_plain": "mp", "route_unseen": "ru", "marker_heldout_class": "mh"}
LENGTHS = tuple(range(7, 13))
CONJ_SHARE = 0.4  # marker sets: share of each stratum drawn from theorems with conjunct-order room
ATOMS = ("P", "Q", "R", "S")


# ------------------------------------------------------------------------------------------------ keys
def split_thm(thm: str) -> tuple[list[str], str]:
    """'p1 , p2 |- c' -> ([p1, p2], c), splitting premises at top-level commas."""
    left, concl = thm.split("|-")
    toks = left.split()
    prem, cur, depth = [], [], 0
    for t in toks:
        if t == "," and depth == 0:
            prem.append(" ".join(cur))
            cur = []
            continue
        depth += (t == "(") - (t == ")")
        cur.append(t)
    if cur:
        prem.append(" ".join(cur))
    return prem, concl.strip()


def rename(formulas: list[str]) -> list[str]:
    """Atoms renamed P, Q, R, S by first appearance across the list (Dan's renaming class)."""
    m: dict[str, str] = {}
    out = []
    for f in formulas:
        toks = []
        for t in f.split():
            if t in ATOMS:
                if t not in m:
                    m[t] = ATOMS[len(m)]
                t = m[t]
            toks.append(t)
        out.append(" ".join(toks))
    return out


def pkey(thm: str) -> str:
    """The renaming class, minimised over premise orders: two theorems that differ only in premise order and atom
    names get the same pkey."""
    prem, concl = split_thm(thm)
    best = None
    for perm in itertools.permutations(prem):
        r = rename([*perm, concl])
        k = " , ".join(r[:-1]) + " |- " + r[-1]
        best = k if best is None or k < best else best
    assert best is not None
    return best


def _open(fn: str | Path):
    return (
        gzip.open(fn, "rt", encoding="utf-8")
        if str(fn).endswith(".gz")
        else open(fn, encoding="utf-8")
    )


def sha256(fn: str | Path) -> str:
    h = hashlib.sha256()
    with open(fn, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def exclusion_files() -> list[Path]:
    """The files whose theorems no set may contain: the source corpus, any two-author corpus present, the RL pool,
    Dan's transfer_track832."""
    out = [SRC_CORPUS]
    out += sorted(p for p in (TA / "data").glob("*.jsonl.gz") if p.is_file())
    return [*out, RL_POOL, TRANSFER]


def exclusion_keys(files: list[Path]) -> tuple[set[str], set[str], dict[str, Any]]:
    """(keys, pkeys, per-file info) of every record of `files` (each record's `key`, and the pkey of its `thm`)."""
    keys: set[str] = set()
    pkeys: set[str] = set()
    info = {}
    for fn in files:
        n = 0
        with _open(fn) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                keys.add(r["key"])
                pkeys.add(pkey(r["thm"]))
                n += 1
        info[str(fn)] = {"records": n, "sha256": sha256(fn)}
    return keys, pkeys, info


# ------------------------------------------------------------------------------------------------ one record
_INTUIT: Any = None


def intuit_provable(prompt: str) -> bool:
    global _INTUIT
    if _INTUIT is None:
        if str(H.DAN) not in sys.path:
            sys.path.insert(0, str(H.DAN))
        from intuit import intuit_provable as ip  # type: ignore[import-not-found]

        _INTUIT = ip
    return bool(_INTUIT(prompt))


def replays(prompt: str, proof: str) -> bool:
    """The proof replays in Dan's state format (the format the models are trained and sampled in)."""
    from state_env import decompose  # type: ignore[import-not-found]

    try:
        decompose(prompt, proof, canon=True)
        return True
    except Exception:  # noqa: BLE001  (ParseFail, ValueError, AssertionError, IndexError in his code)
        return False


def classify(item: tuple[int, str]) -> dict[str, Any] | None:
    """One pool line -> its labels and the set it qualifies for (None: unusable)."""
    i, line = item
    r = json.loads(line)
    L = r["n_lines"]
    if L not in LENGTHS or r["n_prem"] > 3:
        return None
    try:
        P = H.build(r["proof"])
    except H.Unsupported:
        return None
    if H.n_lines(P) > 12 or H.max_depth(P) > 2:
        return None
    p = AP.profile(r["prompt"], r["proof"])
    if not p["ok"] or p["route"] is None:
        return None
    room = {h: p[h]["sites"] for h in AP.HABITS}
    if not room[
        "order"
    ]:  # step order is the primary marker: every set needs its room on the reference proof
        return None
    has_f = " F " in f" {r['thm']} "
    out = {"i": i, "key": r["key"], "thm": r["thm"], "prompt": r["prompt"], "n_lines": L, "n_prem": r["n_prem"],
           "ref_room": room, "ref_conj_all": p["conj_all"]["sites"], "ref_reit_fits": p["reit_fits"],
           "n_room": p["n_room"], "ref_route": p["route"], "has_F": has_f,
           "has_nn": "( ~ ( ~" in r["thm"], "proof": r["proof"], "set": None}  # fmt: skip
    ip = intuit_provable(r["prompt"])
    out["needs_route"] = not ip
    if ip:
        if (
            p["route"]
            or p["route_knock"]
            or p["n_room"] < 2
            or H.inconsistent([u.f for u in P.prem])
        ):
            return None
        out["set"] = "marker_heldout_class" if has_f else "marker_plain"
    else:
        if not p["route"] or out["has_nn"]:
            return None
        out["set"] = "route_unseen"
    if not replays(r["prompt"], r["proof"]):
        return None
    return out


def _init() -> None:
    os.environ.setdefault("PYTHONHASHSEED", "0")


# ------------------------------------------------------------------------------------------------ build
def quotas(n: int) -> dict[int, int]:
    """Equal quotas over LENGTHS, the remainder to the longest lengths."""
    q = {L: n // len(LENGTHS) for L in LENGTHS}
    for L in sorted(LENGTHS, reverse=True)[: n % len(LENGTHS)]:
        q[L] += 1
    return q


def build(out: Path = OUT, procs: int = 8, sizes: dict[str, int] | None = None, pool: Path = POOL,
          limit: int = 0) -> dict:  # fmt: skip
    sizes = dict(SIZES if sizes is None else sizes)
    if (out / "manifest.json").exists() or any((out / f"{s}.jsonl").exists() for s in SETS):
        raise SystemExit(f"{out} already holds sets: they are written once (use another --out)")
    t0 = time.time()
    exf = exclusion_files()
    keys, pkeys, exinfo = exclusion_keys(exf)
    print(
        f"exclusions: {len(keys):,} keys, {len(pkeys):,} pkeys from {len(exf)} files ({time.time() - t0:.0f} s)",
        flush=True,
    )
    lines = pool.read_text(encoding="utf-8").splitlines()
    order = list(range(len(lines)))
    random.Random(SEED).shuffle(order)
    if limit:
        order = order[:limit]
    want = {s: quotas(sizes[s]) for s in SETS}
    # marker sets: per length, CONJ_SHARE of the quota from conjunct-room theorems, the rest from any
    want_c = {
        s: {L: round(q * CONJ_SHARE) for L, q in want[s].items()}
        for s in ("marker_plain", "marker_heldout_class")
    }
    got: dict[str, dict[int, list]] = {s: collections.defaultdict(list) for s in SETS}
    spare: dict[str, dict[int, list]] = {s: collections.defaultdict(list) for s in SETS}
    stats: collections.Counter = collections.Counter()
    seen_pkeys: set[str] = set()

    def full() -> bool:
        return all(len(got[s][L]) >= want[s][L] for s in SETS for L in LENGTHS)

    def items():
        for i in order:
            line = lines[i]
            r = json.loads(line)
            if r["key"] in keys:
                stats["excluded_key"] += 1
                continue
            pk = pkey(r["thm"])
            if pk in pkeys:
                stats["excluded_pkey"] += 1
                continue
            yield i, line

    with mp.Pool(procs, initializer=_init) as P:
        for c in P.imap(classify, items(), chunksize=64):
            stats["classified"] += 1
            if c is None or c["set"] is None:
                stats["unusable"] += 1
                if full():
                    break
                continue
            s, L = c["set"], c["n_lines"]
            pk = pkey(c["thm"])
            if pk in seen_pkeys:  # the pool is distinct by key; this also drops premise-order twins
                stats["dup_pkey"] += 1
                continue
            stats[f"qualified_{s}"] += 1
            c["pkey"] = pk
            g = got[s][L]
            if len(g) < want[s][L]:
                if s in want_c:
                    n_c = sum(1 for x in g if x["ref_room"]["conj"])
                    need_c = want_c[s][L] - n_c
                    free = want[s][L] - len(g)
                    if c["ref_room"]["conj"] or free > need_c:
                        g.append(c)
                        seen_pkeys.add(pk)
                    else:
                        spare[s][L].append(c)
                else:
                    g.append(c)
                    seen_pkeys.add(pk)
            else:
                spare[s][L].append(c)
            if full():
                break
    # a stratum the pool cannot fill: top up from that set's spares (same length first, then neighbours)
    fills = []
    for s in SETS:
        for L in LENGTHS:
            short = want[s][L] - len(got[s][L])
            if short <= 0:
                continue
            for L2 in sorted(LENGTHS, key=lambda x: (abs(x - L), -x)):
                for c in list(spare[s][L2]):
                    if short <= 0:
                        break
                    if c["pkey"] in seen_pkeys:
                        continue
                    got[s][L].append(c)
                    seen_pkeys.add(c["pkey"])
                    spare[s][L2].remove(c)
                    fills.append({"set": s, "stratum": L, "from_length": L2})
                    short -= 1
    out.mkdir(parents=True, exist_ok=True)
    man: dict[str, Any] = {"written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seed": SEED,
                           "source": {"file": str(pool), "sha256": sha256(pool), "records": len(lines)},
                           "exclusions": exinfo, "n_exclusion_keys": len(keys), "n_exclusion_pkeys": len(pkeys),
                           "criteria": __doc__, "sizes": sizes, "conj_share": CONJ_SHARE,
                           "scan": dict(stats), "fills": fills, "sets": {}}  # fmt: skip
    for s in SETS:
        rows = []
        for L in LENGTHS:
            for j, c in enumerate(got[s][L]):
                rows.append({"name": f"ta_{PREFIX[s]}_L{L:02d}_{j:03d}", "set": s, "key": c["key"], "pkey": c["pkey"],
                             "thm": c["thm"], "prompt": c["prompt"], "n_lines": c["n_lines"], "n_prem": c["n_prem"],
                             "stratum": L, "ref_room": c["ref_room"], "ref_conj_all": c["ref_conj_all"],
                             "ref_reit_fits": c["ref_reit_fits"], "ref_route": c["ref_route"],
                             "needs_route": c["needs_route"], "has_F": c["has_F"], "has_nn": c["has_nn"],
                             "pool_index": c["i"], "ref_proof": c["proof"]})  # fmt: skip
        fn = out / f"{s}.jsonl"
        tmp = fn.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        tmp.replace(fn)
        room = collections.Counter(
            "+".join(h for h in AP.HABITS if r["ref_room"][h]) or "none" for r in rows
        )
        man["sets"][s] = {"file": fn.name, "sha256": sha256(fn), "n": len(rows),
                          "per_length": dict(collections.Counter(r["n_lines"] for r in rows)),
                          "per_n_prem": dict(collections.Counter(r["n_prem"] for r in rows)),
                          "ref_room_pattern": dict(room), "ref_route": sum(r["ref_route"] for r in rows),
                          "needs_route": sum(r["needs_route"] for r in rows), "has_F": sum(r["has_F"] for r in rows),
                          "has_nn": sum(r["has_nn"] for r in rows)}  # fmt: skip
    man["checks"] = check(out, keys, pkeys, write=False)
    man["secs"] = round(time.time() - t0, 1)
    (out / "manifest.json").write_text(json.dumps(man, indent=1, default=str))
    print(
        json.dumps(
            {k: v for k, v in man.items() if k not in ("criteria", "exclusions")},
            indent=1,
            default=str,
        )
    )
    return man


def load_set(name: str, root: Path = OUT) -> list[dict]:
    with open(root / f"{name}.jsonl", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def check(
    out: Path = OUT, keys: set[str] | None = None, pkeys: set[str] | None = None, write: bool = True
) -> dict:
    """Re-verify a set folder: no theorem's key or pkey among the exclusions, the sets pairwise disjoint (key and
    pkey), names unique, each set's criteria flags as stated, and (with a manifest) every file's sha256."""
    if keys is None or pkeys is None:
        keys, pkeys, _ = exclusion_keys(exclusion_files())
    res: dict[str, Any] = {
        "excluded_hits": {},
        "pairwise_overlap": {},
        "names_unique": True,
        "flags_ok": {},
    }
    allsets = {s: load_set(s, out) for s in SETS if (out / f"{s}.jsonl").exists()}
    names: set[str] = set()
    for s, rows in allsets.items():
        res["excluded_hits"][s] = sum(
            1 for r in rows if r["key"] in keys or pkey(r["thm"]) in pkeys
        )
        for r in rows:
            if r["name"] in names:
                res["names_unique"] = False
            names.add(r["name"])
        if s == "route_unseen":
            ok = all(r["needs_route"] and r["ref_route"] and not r["has_nn"] and "( ~ ( ~" not in r["thm"]
                     and r["ref_room"]["order"] for r in rows)  # fmt: skip
        else:
            ok = all(not r["needs_route"] and not r["ref_route"] and r["has_F"] == (s == "marker_heldout_class")
                     and r["ref_room"]["order"] and sum(1 for v in r["ref_room"].values() if v) >= 2 for r in rows)  # fmt: skip
        res["flags_ok"][s] = ok and all(
            r["pkey"] == pkey(r["thm"]) and 7 <= r["n_lines"] <= 12 for r in rows
        )
    for a, b in itertools.combinations(sorted(allsets), 2):
        ka = {r["pkey"] for r in allsets[a]} | {r["key"] for r in allsets[a]}
        kb = {r["pkey"] for r in allsets[b]} | {r["key"] for r in allsets[b]}
        res["pairwise_overlap"][f"{a}|{b}"] = len(ka & kb)
    mp_ = out / "manifest.json"
    if write and mp_.exists():
        man = json.loads(mp_.read_text())
        res["sha256_ok"] = {
            s: sha256(out / v["file"]) == v["sha256"] for s, v in man["sets"].items()
        }
    res["ok"] = (not any(res["excluded_hits"].values()) and not any(res["pairwise_overlap"].values())
                 and res["names_unique"] and all(res["flags_ok"].values()) and all(res.get("sha256_ok", {1: 1}).values()))  # fmt: skip
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cmd", choices=("build", "check"))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument(
        "--sizes",
        default=None,
        help="e.g. marker_plain=500,route_unseen=300,marker_heldout_class=300",
    )
    ap.add_argument("--pool", default=str(POOL))
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="scan only the first N records of the seeded order (tests)",
    )
    a = ap.parse_args(argv)
    if a.cmd == "build":
        sizes = dict(SIZES)
        if a.sizes:
            for kv in a.sizes.split(","):
                k, v = kv.split("=")
                assert k in SETS, k
                sizes[k] = int(v)
        build(Path(a.out), a.procs, sizes, Path(a.pool), a.limit)
    else:
        r = check(Path(a.out))
        print(json.dumps(r, indent=1))
        sys.exit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()
