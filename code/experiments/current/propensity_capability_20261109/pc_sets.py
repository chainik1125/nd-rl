"""Build and freeze the theorem list and the checkpoint list of the propensity/capability read.

    python pc_sets.py [--out /scratch/cowifornia/nd-rl/propensity_capability/sets] [--n_other 240] [--seed 0]
                      [--no_load]

Theorems (`theorems.jsonl`, one per line, no proof text: the files it reads hold none):
- `strict106`: the 106 names in `sigma_heldout_strict.txt`, in `transfer_track832.jsonl` order (the factor read's order);
- `transfer_other`: a fixed subsample of the other 726, stratified by (`n_lines`, `source`), seed 0: each stratum gets
  its share of `n_other` by largest remainder (ties to the larger stratum, then the smaller key), and its theorems are
  drawn with `random.Random(f"{seed}:{n_lines}:{source}").sample` from the stratum's names sorted; kept in file order.
Each row carries the applicable directions (from the parsed formulas, `pc_lib.applicable`) and `dir_class`, the
direction's class for SPEC.md's "Router or default shift", fixed here before any read:
- reductio, `required`: the theorem is classical-only, i.e. Dan's G4ip decision procedure (`intuit.intuit_provable`)
  cannot prove it intuitionistically, so every proof uses DN. This is Dan's definition of the strict 106 (all 106 are);
  any of the 240 it also holds for is `required` too.
- reductio on an intuitionistically provable theorem: Dan's bounded shortest-proof search (`minlen.minlen`, the
  `necessity.py` pair) to bound n_lines + 2, once unrestricted and once with `forbid=('DN',)` (DN and the reductio
  template disabled). `helpful`: the unrestricted shortest proof is shorter than any DN-free proof (the DN-free search
  finds a longer one, or none within the bound without a timeout) and that proof carries Dan's σ (`coe_sigma`);
  `idle`: both searches find proofs of the same length; null: a timeout, a memory cap, no unrestricted proof within the
  bound, or a shorter proof whose DN is not σ (DN of a premise).
- negation, conditional, cases, direct: null. `minlen` cannot forbid an introduction on the goal or a premise case
  split, so it cannot say whether a proof avoids them.
The searches' proofs are not kept (only lengths and flags, in `reductio_search`).

Checkpoints (`ckpts.json`), in priority order: Dan's full-corpus base, then the 1 %, 0 % and 0.1 % lines, each as
base then RL end per seed (so a budget that binds leaves whole pairs): group `main`, 19 checkpoints. Then group
`matched` (stage "matched", SPEC.md's decision of 9:05 pm): the 0 % and 0.1 % checkpoints at the RL compute where the
1 % lines end, from the factor read's `ckpts_matched.json` (files under factor_read/matched/ckpts); the 0.1 % seed-2
match is the dd01g0 pilot line (`pilot`); last, group `matched_early`, the two early 0.1 % reads (`*_rolling.pt`,
x below the target) that the factor read kept out of its means. Metadata (dose, seed, x_dan, noise seed, md5)
come from the factor read's `ckpts.json`; every path is checked to exist, its md5 compared, and (unless --no_load)
loaded on the CPU to confirm a canonical `lean_staten` tokenizer.

`manifest.json` records the sha256 of both files and the stratum table. The evaluation files are read only.
"""

# ruff: noqa: E501, SIM115, B905, B007, SIM300
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

from pc_lib import DAN, DIRS, FR_DIR, KD, applicable, theorem

OUT = Path("/scratch/cowifornia/nd-rl/propensity_capability/sets")
TRANSFER = DAN / "data/coe/transfer_track832.jsonl"
STRICT = DAN / "data/coe/sigma_heldout_strict.txt"
FULL_BASE = KD / "bases/stage1_full_s3.pt"
# priority order: (dose, base tag, end tag) per seed
LINES = [("1", "dose1", "dd1"), ("0", "ko", "dko"), ("0.1", "dose01", "dd01")]


def sha256(fn: Path) -> str:
    return hashlib.sha256(fn.read_bytes()).hexdigest()


def md5(fn: Path) -> str:
    h = hashlib.md5()
    with open(fn, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def allocate(sizes: dict, n: int) -> dict:
    """Largest-remainder (Hamilton) allocation of n over strata proportional to their sizes."""
    tot = sum(sizes.values())
    q = {s: n * c / tot for s, c in sizes.items()}
    out = {s: int(q[s]) for s in sizes}
    left = n - sum(out.values())
    order = sorted(sizes, key=lambda s: (-(q[s] - int(q[s])), -sizes[s], s))
    for s in order[:left]:
        out[s] += 1
    return out


def _search(a: tuple) -> dict:
    """One theorem's pair of minlen searches, in a worker capped at `mem_gb` of address space."""
    import resource

    from coe_sigma import flags, has_sigma  # type: ignore[import-not-found]
    from minlen import minlen  # type: ignore[import-not-found]

    name, prompt, bound, tl, mem_gb = a
    resource.setrlimit(resource.RLIMIT_AS, (int(mem_gb * 2**30), int(mem_gb * 2**30)))
    out: dict = {"name": name, "bound": bound, "time_limit": tl}
    for key, fb in (("any", ()), ("no_dn", ("DN",))):
        try:
            r = minlen(prompt, bound, tl, forbid=fb)
            out[f"min_{key}"], out[f"timeout_{key}"], out[f"memory_{key}"] = (
                r["min_lines_ub"],
                bool(r.get("timeout")),
                False,
            )
            if key == "any":
                out["sigma_in_shortest"] = (
                    None if r.get("proof") is None else bool(has_sigma(flags(r["proof"])))
                )
        except MemoryError:
            out[f"min_{key}"], out[f"timeout_{key}"], out[f"memory_{key}"] = None, False, True
    return out


def reductio_class(classical_only: bool, s: dict | None) -> str | None:
    if classical_only:
        return "required"
    if (
        s is None
        or s["min_any"] is None
        or s["timeout_any"]
        or s["memory_any"]
        or s["memory_no_dn"]
    ):
        return None
    if s["min_no_dn"] is None:
        if s["timeout_no_dn"]:
            return None
        return (
            "helpful" if s["sigma_in_shortest"] else None
        )  # a DN-free proof exists (G4ip) but is longer than the bound
    if s["min_any"] < s["min_no_dn"]:
        return "helpful" if s["sigma_in_shortest"] else None
    return "idle"


def classes(thms: list[dict], procs: int, slack: int, tl: float, mem_gb: float) -> dict:
    import multiprocessing as mp
    import time

    from intuit import intuit_provable  # type: ignore[import-not-found]

    co = {t["name"]: not intuit_provable(t["prompt"]) for t in thms}
    jobs = [
        (t["name"], t["prompt"], t["n_lines"] + slack, tl, mem_gb)
        for t in thms
        if not co[t["name"]]
    ]
    t0 = time.time()
    with mp.Pool(procs, maxtasksperchild=1) as pool:
        res = {r["name"]: r for r in pool.imap_unordered(_search, jobs)}
    print(
        f"minlen pairs: {len(jobs)} theorems in {time.time() - t0:.0f} s on {procs} processes",
        flush=True,
    )
    out = {}
    for t in thms:
        s = res.get(t["name"])
        dc = {d: None for d in DIRS}
        dc["reductio"] = reductio_class(co[t["name"]], s)
        out[t["name"]] = {
            "dir_class": dc,
            "reductio_search": {"classical_only": co[t["name"]], **(s or {})},
        }
    return out


def build_theorems(n_other: int, seed: int) -> tuple[list[dict], dict]:
    rows = [json.loads(x) for x in open(TRANSFER)]
    strict = set(STRICT.read_text().split())
    assert len(strict) == 106 and all(isinstance(r["n_lines"], int) for r in rows)
    other = [r for r in rows if r["name"] not in strict]
    by = collections.defaultdict(list)
    for r in other:
        by[(r["n_lines"], r["source"])].append(r["name"])
    sizes = {s: len(v) for s, v in by.items()}
    quota = allocate(sizes, n_other)
    pick = set()
    for s in sorted(by):
        names = sorted(by[s])
        pick |= set(random.Random(f"{seed}:{s[0]}:{s[1]}").sample(names, quota[s]))
    out = []
    for r in rows:
        st = (
            "strict106"
            if r["name"] in strict
            else ("transfer_other" if r["name"] in pick else None)
        )
        if st is None:
            continue
        prem, _ = theorem(r["prompt"])
        assert len(prem) == r["n_prem"], r["name"]
        out.append({"name": r["name"], "set": st, "n_lines": r["n_lines"], "schema": r["schema"],
                    "source": r["source"], "n_prem": r["n_prem"], "prompt": r["prompt"],
                    "applicable": applicable(r["prompt"])})  # fmt: skip
    assert sum(x["set"] == "strict106" for x in out) == 106
    assert sum(x["set"] == "transfer_other" for x in out) == n_other
    table = [
        {"n_lines": s[0], "source": s[1], "size": sizes[s], "drawn": quota[s]}
        for s in sorted(sizes)
    ]
    return out, {"strata": table}


def build_ckpts(load: bool) -> list[dict]:
    fr = {c["tag"]: c for c in json.load(open(FR_DIR / "ckpts.json"))}
    order = [("full_s3", None, "main")]
    for _, b, e in LINES:
        for s in range(3):
            order += [
                (f"{b}_s{s}", fr[f"{b}_s{s}"], "main"),
                (f"{e}_s{s}", fr[f"{e}_s{s}"], "main"),
            ]
    fm = json.load(open(FR_DIR / "ckpts_matched.json"))
    order += [(c["tag"], c, "matched") for c in fm if c["kind"] != "rolling"]
    order += [(c["tag"], c, "matched_early") for c in fm if c["kind"] == "rolling"]
    out = []
    for prio, (cid, c, group) in enumerate(order):
        if c is None:  # Dan's full-corpus base: not in the factor read
            c = {"tag": cid, "role": "base", "dose": "full", "seed": 3, "path": str(FULL_BASE), "file": FULL_BASE.name,
                 "it": 0, "x_dan": 0.0, "md5": None, "noise_seed": 3 * 1000 + 0 + 500}  # fmt: skip
        p = Path(c["path"])
        stage = "base" if c["role"] == "base" else ("rl_end" if group == "main" else "matched")
        row = {"ckpt": cid, "priority": prio, "group": group, "dose": c["dose"].rstrip("%"), "seed": c["seed"],
               "stage": stage, "x_dan": c["x_dan"], "x_target": c.get("x_target"), "pilot": bool(c.get("pilot")), "it": c["it"],
               "path": str(p), "exists": p.exists(), "noise_seed": c["noise_seed"],
               "md5_factor_read": c.get("md5"), "line": c.get("line"), "base": c.get("base")}  # fmt: skip
        if p.exists():
            row["md5"] = md5(p)
            row["md5_match"] = (
                row["md5_factor_read"] is None or row["md5"] == row["md5_factor_read"]
            )
            if load:
                from model import load_ckpt  # type: ignore[import-not-found]

                model, tok, _ = load_ckpt(str(p), "cpu")
                row.update(
                    tok_mode=tok.mode,
                    canon=bool(getattr(tok, "canon", False)),
                    n_params=model.n_params(),
                )
                del model
        print(cid, group, row["stage"], row["dose"], row["seed"], "x", round(row["x_dan"] or 0, 3), "exists", row["exists"],
              "md5ok", row.get("md5_match"), row.get("tok_mode"), flush=True)  # fmt: skip
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--n_other", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_load", action="store_true")
    ap.add_argument("--procs", type=int, default=6, help="minlen workers (at most 8 cores)")
    ap.add_argument("--slack", type=int, default=2, help="minlen bound = n_lines + slack")
    ap.add_argument("--time", type=float, default=20.0, help="minlen seconds per search")
    ap.add_argument("--mem_gb", type=float, default=1.5, help="address-space cap per minlen worker")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    thms, man = build_theorems(a.n_other, a.seed)
    fn = out / "theorems.jsonl"
    if not fn.exists():  # the classes are computed once, when the list is frozen
        cl = classes(thms, a.procs, a.slack, a.time, a.mem_gb)
        for t in thms:
            t.update(cl[t["name"]])
    else:
        old = {x["name"]: x for x in (json.loads(y) for y in open(fn))}
        for t in thms:
            if t["name"] in old:
                t.update(
                    dir_class=old[t["name"]].get("dir_class"),
                    reductio_search=old[t["name"]].get("reductio_search"),
                )
    if fn.exists():
        old = [json.loads(x) for x in open(fn)]
        assert old == thms, f"{fn} exists and differs: the list is frozen; move it away to rebuild"
    else:
        with open(fn, "w") as f:
            for t in thms:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
    ck = build_ckpts(not a.no_load)
    (out / "ckpts.json").write_text(json.dumps(ck, indent=1))
    app = collections.Counter()
    cls = collections.Counter()
    for t in thms:
        for d, v in t["applicable"].items():
            app[(t["set"], d)] += v
        cls[(t["set"], str(t["dir_class"]["reductio"]))] += 1
    man["reductio_class"] = {f"{s_}/{c_}": n for (s_, c_), n in sorted(cls.items())}
    man["minlen"] = {"slack": a.slack, "time": a.time, "mem_gb": a.mem_gb}
    man.update(theorems_sha256=sha256(fn), ckpts_sha256=sha256(out / "ckpts.json"), n_other=a.n_other, seed=a.seed,
               n_theorems=len(thms), applicable={f"{s}/{d}": n for (s, d), n in sorted(app.items())},
               transfer_sha256=sha256(TRANSFER), strict_sha256=sha256(STRICT))  # fmt: skip
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps({k: v for k, v in man.items() if k != "strata"}, indent=1))


if __name__ == "__main__":
    main()
