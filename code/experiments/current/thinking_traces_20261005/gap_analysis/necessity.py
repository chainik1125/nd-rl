"""Are the textbook problems' proof structures necessary? Restricted exact search (CPU only).

    cd /scratch/cowifornia/nd-rl/think/nd-rl
    ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 nice -n 15 code/.venv/bin/python \
        code/experiments/current/thinking_traces_20261005/gap_analysis/necessity.py --workers 8

For each problem in the chosen sets and each structure its shortest proof uses, search again with
that structure forbidden, from the shortest length L up to L + --extra, and record the shortest
length found without it (None = nothing within L + extra). Restrictions:

  * nested_neg: no ¬-intro or reductio box inside another box (NEGI / RAA only at depth 0);
  * no_dn:      no DN at all (so no reductio);
  * no_ore:     no case split;
  * flat:       box depth <= 1 (the fork's `max_depth=1`).

The search is `fastlen.FastSearch` (the fork's `minlen.Search` + a cross-budget memo) with
`_search` copied from the fork's `minlen.Search._search` and one change: the NEGI and RAA options
are skipped at depth >= 1 when `no_nested_neg` is set. A structure is "necessary at L + k" when no
proof of <= L + k lines exists without it (in the search space S; a restricted-space result).

Sets: dev58 7–12 (textbook; aggregate output only) and a random 300 of the pool's rows whose
shortest proof uses the structure (generator control). Output: runs/gap_analysis/necessity.jsonl.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import resource
import sys
import time
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "test_21_30"))
import common as C  # noqa: E402
import fastlen as FL  # noqa: E402
from nd_verify import verify_text  # noqa: E402  (fork, read-only)

BOT = ("bot",)


class Restricted(FL.FastSearch):
    def __init__(self, prem, concl, deadline=None, no_nested_neg=False, **kw):
        super().__init__(prem, concl, deadline=deadline, **kw)
        self.no_nested_neg = no_nested_neg
        self.no_ore = False
        if no_nested_neg:
            self.need_d = True  # memo keyed by depth

    def _search(self, G, A, b, d=0, H=frozenset()):  # noqa: C901  (copied from minlen.Search._search)
        bestc, bestt = b + 1, None
        can_box = self.max_depth is None or d < self.max_depth
        neg_ok = can_box and not (self.no_nested_neg and d >= 1)

        def consider(c, t):
            nonlocal bestc, bestt
            if c < bestc:
                bestc, bestt = c, t

        if b >= 1:
            for Cj in self.conj:
                if Cj[1] == G or Cj[2] == G:
                    if self.ctx_neg and ("#neghyp", Cj) in H:
                        continue
                    r = self.inp(Cj, A, min(b, bestc) - 1, d, H)
                    if r:
                        consider(r[0] + 1, ("ANDE1" if Cj[1] == G else "ANDE2", G, r[1]))
        if b >= 1:
            for Ij in self.imps:
                if Ij[2] == G:
                    r = self.inp(Ij, A, min(b, bestc) - 1, d, H)
                    if r:
                        r2 = self.inp(Ij[1], A | {Ij}, min(b, bestc) - 1 - r[0], d, H)
                        if r2:
                            consider(r[0] + r2[0] + 1, ("IMPE", G, r[1], r2[1]))
        if b >= 1 and not self.forbid_dn:
            nn = ("not", ("not", G))
            r = self.inp(nn, A, min(b, bestc) - 1, d, H)
            if r:
                consider(r[0] + 1, ("DN", G, r[1]))
        if G != BOT and b >= 1:
            r = self.inp(BOT, A, min(b, bestc) - 1, d, H)
            if r:
                consider(r[0] + 1, ("BOTE", G, r[1]))
        if G[0] == "and" and b >= 1:
            r = self.inp(G[1], A, min(b, bestc) - 1, d, H)
            if r:
                r2 = self.inp(G[2], A | {G[1]}, min(b, bestc) - 1 - r[0], d, H)
                if r2:
                    consider(r[0] + r2[0] + 1, ("ANDI", G, r[1], r2[1]))
        if G[0] == "or" and b >= 1:
            r = self.inp(G[1], A, min(b, bestc) - 1, d, H)
            if r:
                consider(r[0] + 1, ("ORI1", G, r[1]))
            r = self.inp(G[2], A, min(b, bestc) - 1, d, H)
            if r:
                consider(r[0] + 1, ("ORI2", G, r[1]))
        if G[0] == "imp" and b >= 2:
            a, c = G[1], G[2]
            if a == c and can_box:
                consider(2, ("IMPI", G, a, None))
            elif b >= 3 and can_box:
                r = self.best(
                    c,
                    A | {a},
                    min(b, bestc) - 2,
                    d + 1,
                    H | {a} | ({("#impi",)} if self.ctx_impi else set()),
                )
                if r:
                    consider(r[0] + 2, ("IMPI", G, a, r[1]))
        if G[0] == "not" and b >= 3 and neg_ok:
            r = self.best(
                BOT,
                A | {G[1]},
                min(b, bestc) - 2,
                d + 1,
                H | {G[1]} | ({("#neghyp", G[1])} if self.ctx_neg else set()),
            )
            if r:
                consider(r[0] + 2, ("NEGI", G, G[1], r[1]))
        if G == BOT and b >= 1:
            for X in self.S:
                NX = ("not", X)
                if NX not in self.S and NX not in A:
                    continue
                if NX in A:
                    r = self.inp(X, A, min(b, bestc) - 1, d, H)
                    if r:
                        consider(r[0] + 1, ("NEGE", BOT, r[1], ("R0", NX)))
                elif X in A:
                    r = self.inp(NX, A, min(b, bestc) - 1, d, H)
                    if r:
                        consider(r[0] + 1, ("NEGE", BOT, ("R0", X), r[1]))
                else:
                    r = self.inp(X, A, min(b, bestc) - 2, d, H)
                    if r:
                        r2 = self.inp(NX, A | {X}, min(b, bestc) - 1 - r[0], d, H)
                        if r2:
                            consider(r[0] + r2[0] + 1, ("NEGE", BOT, r[1], r2[1]))
        if G != BOT and G[0] != "not" and b >= 4 and neg_ok and not self.forbid_dn:
            r = self.best(BOT, A | {("not", G)}, min(b, bestc) - 3, d + 1, H | {("not", G)})
            if r:
                consider(r[0] + 3, ("RAA", G, r[1]))
        if (
            b >= 3
            and can_box
            and not self.no_ore
            and not (self.ctx_impi and ("#impi",) in H)
            and not (self.ctx_ore and ("#ore",) in H)
        ):
            for D in self.disj:
                if D == G:
                    continue
                if self.ore_mode == "loose" and D not in self.prem:
                    continue
                if self.ore_mode == "strict" and D not in self.prem and D not in H and D[1] != D[2]:
                    continue
                r = self.inp(D, A, min(b, bestc) - 3, d, H)
                if not r:
                    continue
                a, c = D[1], D[2]
                A2 = A | {D}
                ca = (1, None) if a == G else None
                if ca is None:
                    ra = self.best(
                        G,
                        A2 | {a},
                        min(b, bestc) - r[0] - 3,
                        d + 1,
                        H | {a} | ({("#ore",)} if self.ctx_ore else set()),
                    )
                    ca = (ra[0] + 1, ra[1]) if ra else None
                if ca is None:
                    continue
                cb = (1, None) if c == G else None
                if cb is None:
                    rb = self.best(
                        G,
                        A2 | {c},
                        min(b, bestc) - r[0] - ca[0] - 1,
                        d + 1,
                        H | {c} | ({("#ore",)} if self.ctx_ore else set()),
                    )
                    cb = (rb[0] + 1, rb[1]) if rb else None
                if cb is None:
                    continue
                consider(r[0] + ca[0] + cb[0] + 1, ("ORE", G, r[1], a, ca[1], c, cb[1]))
        if self.lemmas and b >= 2:
            for Lf in self.S:
                if Lf in A or Lf == G:
                    continue
                r = self.best(Lf, A, min(b, bestc) - 1, d, H)
                if not r:
                    continue
                r2 = self.best(G, A | {Lf}, min(b, bestc) - r[0], d, H)
                if r2:
                    consider(r[0] + r2[0], ("LEMMA", G, Lf, r[1], r2[1]))
        return (bestc, bestt) if bestt is not None else None


def restricted_min(
    prompt: str, kind: str, L: int, extra_lines: int, secs: float, extra=()
) -> dict[str, Any]:
    prem, concl = C.parse_prompt(prompt)
    kw: dict[str, Any] = {}
    if kind == "no_dn":
        kw["forbid"] = ("DN",)
    if kind == "flat":
        kw["max_depth"] = 1
    s = Restricted(
        prem,
        concl,
        deadline=time.time() + secs,
        no_nested_neg=(kind == "nested_neg"),
        extra=extra,
        **kw,
    )
    if kind == "no_ore":
        s.no_ore = True
    try:
        r, lb = s.run_bound(L + extra_lines, start=L)
    except FL.Timeout:
        return {"len": None, "timeout": True}
    if r is None:
        return {"len": None, "timeout": False}
    n, text = r
    ok, why, nl = verify_text(prompt + " " + text)
    pf = C.proof_features(prompt, text)
    viol = {
        "nested_neg": pf["p_negbox_nested"],
        "no_dn": pf["p_has_DN"],
        "no_ore": pf["p_has_ORE"],
        "flat": int(pf["p_box_depth"] >= 2),
    }[kind]
    return {
        "len": n,
        "timeout": False,
        "verified": bool(ok and nl == n),
        "violates": viol,
        "skeleton": pf["p_skeleton"],
    }


HAS = {
    "nested_neg": "p_negbox_nested",
    "no_dn": "p_has_DN",
    "no_ore": "p_has_ORE",
    "flat": None,
}


def has(f: dict[str, Any], kind: str) -> bool:
    if kind == "flat":
        return f["p_box_depth"] >= 2
    return bool(f[HAS[kind]])


def _init(gb: float) -> None:
    b = int(gb * 2**30)
    resource.setrlimit(resource.RLIMIT_AS, (b, b))


def _work(job: dict[str, Any]) -> dict[str, Any]:
    out = {k: job[k] for k in ("set", "name", "kind", "L", "score")}
    try:
        out.update(
            restricted_min(
                job["prompt"],
                job["kind"],
                job["L"],
                job["extra_lines"],
                job["secs"],
                job.get("extra", ()),
            )
        )
    except MemoryError:
        out.update(len=None, timeout=False, error="memory")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--extra", type=int, default=3)
    ap.add_argument("--secs", type=float, default=120.0)
    ap.add_argument("--n_pool", type=int, default=300)
    a = ap.parse_args()
    mins = {}
    for s in ("textbook", "pool"):
        for x in C.jl(C.OUT / f"minproofs_{s}.jsonl"):
            mins[x["name"]] = x
    jobs = []
    for r in C.jl(C.TB / "textbook_dev.jsonl"):
        if not 7 <= r["reference_lines"] <= 12:
            continue
        m = mins[r["name"]]
        f = C.proof_features(r["prompt"], m["proof"])
        for kind in ("nested_neg", "no_dn", "no_ore", "flat"):
            if has(f, kind):
                jobs.append(
                    dict(
                        set="dev58_712",
                        name=r["name"],
                        prompt=r["prompt"],
                        kind=kind,
                        L=m["min_lines"],
                        extra_lines=a.extra,
                        secs=a.secs,
                        score=None,
                    )
                )
    rng = random.Random(20261008)
    pool = [x for x in C.jl(C.POOL)]
    for kind in ("nested_neg", "no_dn", "no_ore", "flat"):
        cand = []
        for r in pool:
            m = mins[r["name"]]
            f = C.proof_features(r["prompt"], m["proof"])
            if has(f, kind):
                cand.append((r, m))
        for r, m in rng.sample(cand, min(a.n_pool, len(cand))):
            extra = tuple(ln["f"] for ln in C.parse_proof(r["gen_proof"]))
            jobs.append(
                dict(
                    set="pool",
                    name=r["name"],
                    prompt=r["prompt"],
                    kind=kind,
                    L=m["min_lines"],
                    extra_lines=a.extra,
                    secs=a.secs,
                    score=None,
                    extra=extra,
                )
            )
    print(len(jobs), "jobs", flush=True)
    t0 = time.time()
    with (
        mp.Pool(a.workers, initializer=_init, initargs=(3.5,), maxtasksperchild=50) as p,
        open(C.OUT / "necessity.jsonl", "w") as fo,
    ):
        for i, r in enumerate(p.imap_unordered(_work, jobs)):
            fo.write(json.dumps(r) + "\n")
            fo.flush()
            if (i + 1) % 100 == 0:
                print(i + 1, f"{time.time() - t0:.0f}s", flush=True)
    print("done", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
