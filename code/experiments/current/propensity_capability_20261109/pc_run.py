"""The propensity/capability read of one or more checkpoints (SPEC.md): arms U, F and E, then the result rows.

    python pc_run.py gpu  --ckpts IDS [--gpu N] [--root ROOT] [--k 32] [--kf 16] [--batch 1024] [--gpu_budget S]
    python pc_run.py rows --ckpts IDS [--root ROOT]                       # CPU: classify raw attempts -> result rows
    python pc_run.py plan --ckpts IDS [--root ROOT] [--k 32] [--kf 16]    # what `gpu` would do, and how long
    (smoke: --device cpu --limit N [--names a,b] --root SOMEWHERE_ELSE; no gate, lock or ledger on the CPU)

IDS is a comma list of `ckpt` ids from SETS/ckpts.json (pc_sets.py), run in the order given; `all` = priority order.

Arms, per checkpoint (seeds: the checkpoint's noise seed, +10000 for F; the name base and decode noise of each
attempt follow `run_env`, which is Dan's `env_generate` line for line):
- U: k attempts per theorem, no forced prefix: `run_env` with an empty prefix is `env_generate` byte for byte
  (test_pc_lib.py checks it on the CPU, the factor read's check 1 on the GPU), temperature 0.8, 256 tokens per action,
  48 steps, Lean gate.
- F: k_f attempts per theorem and applicable direction, the forced prefix = every premise line then d's opening
  (`pc_lib.forced_prefix`); the prefix counts against the 48 steps.
- E: for each applicable direction, log P(d's opening | the state after every premise line) at temperature 0.8,
  name-free (the tokens the env overrides left out, `factor_lib.name_positions`), at each of the 33 name bases;
  `logp_open` is the log of the mean probability over bases (the sampler draws the base uniformly). Also stored raw:
  the full-token values and log P(each premise line), so P(the attempt restates every premise first) is known.

Raw attempts (proof text) go to ROOT/raw/<ckpt>/{U,F}.jsonl.gz and E.json.gz, with <arm>.stats.json; each arm is
written atomically and skipped when present, so a stopped read resumes. `rows` writes ROOT/results/<ckpt>.jsonl in
SPEC.md's schema and ROOT/results/<ckpt>.qc.json (checks: every forced attempt's first action is its direction, replay
agrees with the env, Lean against the verifier on accepted proofs, premises-first share).

GPU rules, in code (as fr_run.py): the GPU is claimed by `mkdir` of ROOT/locks/<host>_gpu<N>.lock, released at exit;
before launch and before every arm `gpu.gpus` must show no other user on GPU N and nvidia-smi no process but this one,
at launch also under 500 MiB in use, else exit 3; ROOT/gpu_ledger.jsonl sums GPU seconds, at --gpu_budget exit 4.
"""

# ruff: noqa: E501, SIM115, B905, B007, SIM300
from __future__ import annotations

import argparse
import atexit
import collections
import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from pc_lib import (
    BASES,
    DIRS,
    FORCED,
    FR_DIR,
    Attempt,
    attempt_actions,
    classify,
    first_move_rows,
    forced_prefix,
    gen_tokens,
    logmeanexp,
    replay_m,
    run_env,
    score_rows,
)

ROOT = Path("/scratch/cowifornia/nd-rl/propensity_capability")
ARM_SEED = {"U": 0, "F": 10000}
TEMP, MAX_ACTION, MAX_STEPS = 0.8, 256, 48
# throughput model for `plan`: actions per second by stage, the mean of the factor read's forced `reach` stage (ynez
# RTX 3090 at 250 W, batch 1024, Lean gate included: bases 1156, RL ends 921, slowest end 626), and actions per attempt
# per checkpoint from its unforced strict-106 read (the CPU smoke found the 240 no longer); a checkpoint it did not read
# takes the mean of its stage
RATE = {"base": 1150.0, "rl_end": 920.0, "matched": 920.0}
E_SECS, LOAD_SECS = 20.0, 5.0


def say(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


# ------------------------------------------------------------------------------------- GPU rules
def gate(gpu: int, first: bool = False) -> None:
    """Exit 3 unless GPU `gpu` (nvidia-smi index, PCI order) is ours alone. Adapted from fr_run.gate."""
    if str(FR_DIR) not in sys.path:
        sys.path.append(str(FR_DIR))
    import gpu as G  # the factor read's gpu.py

    g = G.gpus(G.HOST)
    if g is None or gpu not in g:
        say(f"GATE: cannot read GPU {gpu} on {G.HOST}")
        sys.exit(3)
    others = g[gpu] - {G.ME}
    if others:
        say(f"GATE: another user on GPU {gpu}: {sorted(others)}")
        sys.exit(3)
    q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_bus_id", "--format=csv,noheader"],
                       capture_output=True, text=True, timeout=30).stdout  # fmt: skip
    bus = subprocess.run(["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader", "-i", str(gpu)],
                         capture_output=True, text=True, timeout=30).stdout.strip().lower()  # fmt: skip
    pids = [
        int(x.split(",")[0])
        for x in q.splitlines()
        if x.strip() and x.split(",")[1].strip().lower() == bus
    ]
    if [p for p in pids if p != os.getpid()]:
        say(f"GATE: another process on GPU {gpu}: {pids}")
        sys.exit(3)
    if first:
        mem = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(gpu)],
                             capture_output=True, text=True, timeout=30).stdout.strip()  # fmt: skip
        if int(mem) >= 500:
            say(f"GATE: {mem} MiB in use on GPU {gpu}")
            sys.exit(3)


def claim(lock: Path) -> None:
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.mkdir(lock)
    except FileExistsError:
        say(f"LOCK held: {lock} ({(lock / 'pid').read_text() if (lock / 'pid').exists() else '?'})")
        sys.exit(3)
    (lock / "pid").write_text(f"{os.uname().nodename} {os.getpid()} {time.time()}")

    def release():
        try:
            (lock / "pid").unlink()
            os.rmdir(lock)
        except OSError:
            pass

    atexit.register(release)


def ledger_sum(fn: Path) -> float:
    if not fn.exists():
        return 0.0
    return sum(json.loads(x)["secs"] for x in fn.read_text().splitlines() if x.strip())


# ------------------------------------------------------------------------------------- io
def write_gz(fn: Path, rows: list[dict]) -> None:
    tmp = fn.with_suffix(fn.suffix + ".tmp")
    with gzip.open(tmp, "wt") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, fn)


def read_gz(fn: Path) -> list[dict]:
    with gzip.open(fn, "rt") as f:
        return [json.loads(x) for x in f]


def write_json(fn: Path, obj) -> None:
    tmp = fn.with_suffix(fn.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, fn)


def load_sets(
    sets: Path, limit: int | None = None, names: list[str] | None = None
) -> tuple[list[dict], dict]:
    thms = [json.loads(x) for x in open(sets / "theorems.jsonl")]
    if names:
        thms = [t for t in thms if t["name"] in set(names)]
    if limit:
        out, seen = [], collections.Counter()
        for t in thms:
            if seen[t["set"]] < limit:
                out.append(t)
                seen[t["set"]] += 1
        thms = out
    cks = {c["ckpt"]: c for c in json.load(open(sets / "ckpts.json"))}
    return thms, cks


def ckpt_ids(arg: str, cks: dict) -> list[str]:
    """A comma list of checkpoint ids and group names, expanded in the order given (a group in priority order):
    `main` (the 19 bases and RL ends), `matched` (the 5 at matched compute), `matched_early` (the 2 early 0.1 %
    reads), `all` (the 26: main, matched, matched_early)."""
    order = sorted(cks.values(), key=lambda c: c["priority"])
    groups = {
        g: [c["ckpt"] for c in order if c.get("group", "main") == g]
        for g in ("main", "matched", "matched_early")
    }
    groups["all"] = [c["ckpt"] for c in order]
    ids: list[str] = []
    for tok in arg.split(","):
        for i in groups.get(tok, [tok]):
            if i not in ids:
                ids.append(i)
    bad = [i for i in ids if i not in cks]
    assert not bad, f"unknown checkpoint ids {bad}"
    return ids


# ------------------------------------------------------------------------------------- arms
def u_items(thms: list[dict], k: int) -> list[dict]:
    return [{"name": t["name"], "prompt": t["prompt"]} for t in thms for _ in range(k)]


def f_items(thms: list[dict], kf: int) -> list[dict]:
    return [{"name": t["name"], "prompt": t["prompt"], "d": d, "prefix": forced_prefix(t["prompt"], d)}
            for t in thms for d in FORCED if t["applicable"][d] for _ in range(kf)]  # fmt: skip


def sample(
    model, tok, items: list[dict], seed: int, batch: int, stats: dict, lean: bool = True
) -> tuple[list, list]:
    """run_env over the items, then Lean's verdict on each (judge_many reads the gate's registry)."""
    atts = run_env(
        model, tok, items, TEMP, MAX_ACTION, MAX_STEPS, batch, seed, stats=stats, gate=lean, log=say
    )
    if not lean:
        return atts, [None] * len(atts)
    from lean_judge import judge_many  # type: ignore[import-not-found]

    oks = [x[0] for x in judge_many([(it["prompt"], a.nd) for it, a in zip(items, atts)])]
    return atts, oks


def raw_rows(items: list[dict], atts: list[Attempt], oks: list, k: int) -> list[dict]:
    return [{"name": it["name"], "d": it.get("d"), "draw": i % k, "base": a.base,
             "raw": [" ".join(x) for x in a.raw], "ended": a.ended, "end": a.end, "nd": a.nd,
             "ok": None if ok is None else bool(ok), "tokens": gen_tokens(a)}
            for i, (it, a, ok) in enumerate(zip(items, atts, oks))]  # fmt: skip


def e_arm(model, tok, thms: list[dict], max_elems: int) -> dict:
    """Exact first-move log-probabilities: per theorem, per applicable direction, per name base."""
    rows, npos, idx = [], [], []
    for t in thms:
        ds = [d for d in FORCED if t["applicable"][d]]
        for b in BASES:
            r, n, lab = first_move_rows(tok, t["prompt"], b, ds)
            rows += r
            npos += n
            idx += [(t["name"], b, x) for x in lab]
    lp, lp_name = score_rows(model, tok, rows, TEMP, max_elems=max_elems, name_pos=npos)
    out: dict = {t["name"]: {"full": {}, "name_free": {}} for t in thms}
    for (nm, b, lab), v, w in zip(idx, lp, lp_name):
        out[nm]["full"].setdefault(lab, [0.0] * len(BASES))[b] = round(v, 6)
        out[nm]["name_free"].setdefault(lab, [0.0] * len(BASES))[b] = round(v - w, 6)
    return {"temperature": TEMP, "bases": list(BASES), "logp": out,
            "n_rows": len(rows), "tokens": sum(len(x[0]) + len(x[1]) for x in rows)}  # fmt: skip


def stats_json(st: dict) -> dict:
    return {
        k: (dict(v) if hasattr(v, "items") else v)
        for k, v in st.items()
        if k not in ("declen", "declen_by_prompt")
    }


def run_gpu(a, thms: list[dict], cks: dict) -> None:
    root = Path(a.root)
    on_gpu = a.device == "cuda"
    ledger = root / "gpu_ledger.jsonl"
    if on_gpu:
        if ledger_sum(ledger) >= a.gpu_budget:
            say(f"BUDGET: {ledger_sum(ledger):.0f} s of GPU time already used")
            sys.exit(4)
        gate(a.gpu, first=True)
        claim(root / "locks" / f"{os.uname().nodename.split('.')[0]}_gpu{a.gpu}.lock")
    import torch  # after the gate: nothing on the GPU yet
    from model import load_ckpt  # type: ignore[import-not-found]

    if not on_gpu:
        torch.set_num_threads(a.threads)
    for cid in ckpt_ids(a.ckpts, cks):
        c = cks[cid]
        d = root / "raw" / cid
        d.mkdir(parents=True, exist_ok=True)
        todo = [
            s
            for s in a.arms.split(",")
            if not (d / ("E.json.gz" if s == "E" else f"{s}.jsonl.gz")).exists()
        ]
        if not todo:
            say(f"{cid}: nothing to do")
            continue
        if not c["exists"]:
            say(f"{cid}: MISSING checkpoint {c['path']}; skipped")
            continue
        if on_gpu:
            gate(a.gpu)
        t00 = time.time()
        model, tok, _ = load_ckpt(c["path"], a.device)
        assert tok.canon and tok.mode == "lean_staten", tok.mode
        write_json(d / "ckpt.json", {**c, "host": os.uname().nodename, "k": a.k, "kf": a.kf, "batch": a.batch,
                                     "n_theorems": len(thms), "device": a.device})  # fmt: skip
        say(f"{cid}: loaded in {time.time() - t00:.1f} s; todo {todo}")
        for s in todo:
            if on_gpu:
                gate(a.gpu)
                used = ledger_sum(ledger)
                if used >= a.gpu_budget:
                    say(f"BUDGET: {used:.0f} s of GPU time used, stopping before {cid} {s}")
                    sys.exit(4)
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            st: dict = {}
            if s == "E":
                res = e_arm(model, tok, thms, a.max_elems)
                tmp = d / "E.json.gz.tmp"
                with gzip.open(tmp, "wt") as f:
                    json.dump(res, f)
                os.replace(tmp, d / "E.json.gz")
                st = {"n_rows": res["n_rows"], "tokens": res["tokens"]}
            else:
                seed = c["noise_seed"] + ARM_SEED[s]
                items = u_items(thms, a.k) if s == "U" else f_items(thms, a.kf)
                kk = a.k if s == "U" else a.kf
                atts, oks = sample(model, tok, items, seed, a.batch, st, lean=not a.no_lean)
                write_gz(d / f"{s}.jsonl.gz", raw_rows(items, atts, oks, kk))
                st = stats_json(st)
                st.update(noise_seed=seed, k=kk, batch=a.batch, n_attempts=len(items),
                          gen_tokens=sum(gen_tokens(x) for x in atts))  # fmt: skip
            secs = time.time() - t0
            if on_gpu:
                st.update(peak_alloc_gb=torch.cuda.max_memory_allocated() / 2**30)
            write_json(
                d / f"{s}.stats.json",
                {**st, "secs": secs, "host": os.uname().nodename, "device": a.device},
            )
            if on_gpu:
                with open(ledger, "a") as f:
                    f.write(json.dumps({"utc": time.strftime("%FT%TZ", time.gmtime()), "ckpt": cid, "arm": s,
                                        "secs": secs}) + "\n")  # fmt: skip
            say(f"{cid} {s}: {secs:.1f} s" + (f"; GPU total {ledger_sum(ledger):.0f} s" if on_gpu else "")
                + (f"; {st.get('n_attempts')} attempts, {st.get('rows')} actions" if s != "E" else f"; {st['n_rows']} rows"))  # fmt: skip
        del model
        if on_gpu:
            torch.cuda.empty_cache()
    say("gpu phase done")


# ------------------------------------------------------------------------------------- rows
def ended_by(end: str) -> str:
    """done | capped | refused (an illegal action, or an action with no <eos> in 256 tokens) | stop."""
    if end == "done":
        return "done"
    if end == "step_cap":
        return "capped"
    if end.startswith("syntax") or end == "truncated":
        return "refused"
    return end


def classify_rows(raw: list[dict], prompts: dict, forced: bool) -> list[dict]:
    out = []
    for r in raw:
        prompt = prompts[r["name"]]
        prefix = forced_prefix(prompt, r["d"]) if forced else []
        att = Attempt(0, r["base"], [x.split() for x in r["raw"]], r["ended"], r["end"])
        acts = attempt_actions(prefix, att)
        c = classify(prompt, r["base"], acts, accepted=bool(r["ok"]))
        if r["ok"]:  # the verifier's own view of a Lean-accepted proof (QC only)
            c["verifier_ok"] = replay_m(prompt, acts, r["base"]).valid_through()
        out.append(c)
    return out


def result_rows(
    c: dict, thms: list[dict], U: list[dict], F: list[dict], E: dict | None, k: int, kf: int
) -> tuple[list, dict]:
    prompts = {t["name"]: t["prompt"] for t in thms}
    cu = classify_rows(U, prompts, False)
    cf = classify_rows(F, prompts, True)
    by_u: dict = collections.defaultdict(list)
    for r, x in zip(U, cu):
        by_u[r["name"]].append((r, x))
    by_f: dict = collections.defaultdict(list)
    for r, x in zip(F, cf):
        by_f[(r["name"], r["d"])].append((r, x))
    head = {
        "ckpt": c["ckpt"],
        "dose": c["dose"],
        "seed": c["seed"],
        "stage": c["stage"],
        "x_dan": c["x_dan"],
    }
    rows = []
    for t in thms:
        th = {
            **head,
            "set": t["set"],
            "name": t["name"],
            "n_lines": t["n_lines"],
            "schema": t["schema"],
            "source": t["source"],
        }
        u = by_u[t["name"]]
        assert len(u) == k, (t["name"], len(u), k)
        ends = [ended_by(r["end"]) for r, _ in u]
        rows.append({**th, "arm": "U", "direction": "any", "applicable": True, "k": k, "dir_class": None,
                     "solved": sum(bool(r["ok"]) for r, _ in u), "used": None, "first": None, "solved_first": None,
                     "capped": ends.count("capped"), "refused": ends.count("refused"), "logp_open": None,
                     "tokens": sum(r["tokens"] for r, _ in u)})  # fmt: skip
        for d in DIRS:
            rows.append({**th, "arm": "U", "direction": d, "applicable": t["applicable"][d], "k": k,
                         "dir_class": t["dir_class"][d], "solved": sum(bool(r["ok"]) and d in x["through"] for r, x in u),
                         "used": sum(d in x["opens"] for _, x in u), "first": sum(x["first"] == d for _, x in u),
                         "solved_first": sum(bool(r["ok"]) and x["first"] == d for r, x in u),
                         "capped": None, "refused": None, "logp_open": None, "tokens": None})  # fmt: skip
        for d in FORCED:
            if not t["applicable"][d]:
                continue
            f = by_f[(t["name"], d)]
            assert len(f) == kf, (t["name"], d, len(f), kf)
            ends = [ended_by(r["end"]) for r, _ in f]
            rows.append({**th, "arm": "F", "direction": d, "applicable": True, "k": kf,
                         "dir_class": t["dir_class"][d], "solved": sum(bool(r["ok"]) for r, _ in f), "used": None, "first": None,
                         "solved_first": None, "capped": ends.count("capped"), "refused": ends.count("refused"), "logp_open": None,
                         "tokens": sum(r["tokens"] for r, _ in f)})  # fmt: skip
        if E is not None:
            for d in FORCED:
                if t["applicable"][d]:
                    rows.append({**th, "arm": "E", "direction": d, "applicable": True, "k": 0,
                                 "dir_class": t["dir_class"][d], "solved": 0,
                                 "used": None, "first": None, "solved_first": None, "capped": None, "refused": None,
                                 "logp_open": round(logmeanexp(E["logp"][t["name"]]["name_free"][d]), 6),
                                 "tokens": None})  # fmt: skip
    # QC
    n_prem = {t["name"]: t["n_prem"] for t in thms}
    qc = {
        "n_U": len(U), "n_F": len(F),
        "U_first": dict(collections.Counter(str(x["first"]) for x in cu)),
        "U_premises_first_share": sum(x["n_pr"] == n_prem[r["name"]] for r, x in zip(U, cu)) / max(1, len(U)),
        "F_first_not_forced": sum(x["first"] != r["d"] for r, x in zip(F, cf)),
        "F_opens_not_forced": sum(r["d"] not in x["opens"] for r, x in zip(F, cf)),
        "F_accepted_not_through_forced": dict(collections.Counter(r["d"] for r, x in zip(F, cf) if r["ok"] and r["d"] not in x["through"])),
        "replay_done_vs_env": sum(x["done"] != (r["end"] == "done") for r, x in zip(U + F, cu + cf)),
        "lean_ok_verifier_rejects": sum(1 for x in cu + cf if x.get("verifier_ok") is False),
        "accepted_U": sum(bool(r["ok"]) for r in U), "accepted_F": sum(bool(r["ok"]) for r in F),
        "ends_U": dict(collections.Counter(r["end"].split(":")[0] for r in U)),
        "ends_F": dict(collections.Counter(r["end"].split(":")[0] for r in F)),
        "actions_per_attempt_U": sum(len(r["raw"]) for r in U) / max(1, len(U)),
        "actions_per_attempt_F": sum(len(r["raw"]) for r in F) / max(1, len(F)),
    }  # fmt: skip
    return rows, qc


SCHEMA = {"ckpt": str, "dose": str, "seed": int, "stage": str, "x_dan": (float, type(None)), "set": str, "name": str,
          "n_lines": int, "schema": (str, type(None)), "source": str, "arm": str, "direction": str, "applicable": bool,
          "k": int, "dir_class": (str, type(None)), "solved": int, "used": (int, type(None)), "first": (int, type(None)), "solved_first": (int, type(None)), "capped": (int, type(None)),
          "refused": (int, type(None)), "logp_open": (float, type(None)), "tokens": (int, type(None))}  # fmt: skip


def check_row(r: dict) -> None:
    """SPEC.md's contract, field by field."""
    assert set(r) == set(SCHEMA), set(r) ^ set(SCHEMA)
    for f, ty in SCHEMA.items():
        assert isinstance(r[f], ty) and not (ty is int and isinstance(r[f], bool)), (f, r[f])
    assert r["dose"] in ("full", "0", "0.1", "1") and r["stage"] in ("base", "rl_end", "matched")
    assert r["set"] in ("strict106", "transfer_other") and r["arm"] in ("U", "F", "E")
    assert r["direction"] in ("any",) + DIRS
    assert r["dir_class"] in ("required", "helpful", "idle", None)
    assert (r["direction"] == "any") == (r["arm"] == "U" and r["used"] is None)
    if r["arm"] == "U" and r["direction"] != "any":
        assert (
            r["used"] is not None
            and r["first"] is not None
            and r["solved"] <= r["used"] <= r["k"]
            and r["first"] <= r["k"]
        )
        assert r["solved_first"] is not None and r["solved_first"] <= r["first"]
        assert r["direction"] == "direct" or r["first"] <= r["used"]
    else:
        assert r["solved_first"] is None
    if r["arm"] in ("U", "F") and r["capped"] is not None:
        assert r["solved"] + r["capped"] + r["refused"] <= r["k"]
    if r["arm"] == "E":
        assert r["logp_open"] is not None and r["logp_open"] <= 1e-6 and r["direction"] in FORCED
    else:
        assert r["logp_open"] is None
    if r["arm"] in ("F", "E"):
        assert r["applicable"] and r["direction"] in FORCED


def run_rows(a, thms: list[dict], cks: dict) -> None:
    root = Path(a.root)
    (root / "results").mkdir(parents=True, exist_ok=True)
    for cid in ckpt_ids(a.ckpts, cks):
        d = root / "raw" / cid
        need = [d / "U.jsonl.gz", d / "F.jsonl.gz", d / "E.json.gz"]
        if not all(p.exists() for p in need):
            say(f"{cid}: raw arms missing ({[p.name for p in need if not p.exists()]}); no rows")
            continue
        t0 = time.time()
        with gzip.open(d / "E.json.gz", "rt") as f:
            E = json.load(f)
        rows, qc = result_rows(
            cks[cid],
            thms,
            read_gz(d / "U.jsonl.gz"),
            read_gz(d / "F.jsonl.gz"),
            E,
            json.loads((d / "U.stats.json").read_text())["k"],
            json.loads((d / "F.stats.json").read_text())["k"],
        )
        for r in rows:
            check_row(r)
        fn = root / "results" / f"{cid}.jsonl"
        tmp = fn.with_suffix(".jsonl.tmp")
        with open(tmp, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        os.replace(tmp, fn)
        write_json(
            root / "results" / f"{cid}.qc.json",
            {**qc, "secs": time.time() - t0, "n_rows": len(rows)},
        )
        say(f"{cid}: {len(rows)} rows in {time.time() - t0:.1f} s; QC forced-first mismatches {qc['F_first_not_forced']}, "
            f"replay mismatches {qc['replay_done_vs_env']}, Lean-ok verifier-rejects {qc['lean_ok_verifier_rejects']}")  # fmt: skip


# ------------------------------------------------------------------------------------- plan
def fr_actions_per_attempt() -> dict:
    """Actions per attempt of each checkpoint's unforced strict-106 read in the factor read (decode rows / 6784)."""
    out = {}
    fr = Path("/scratch/cowifornia/nd-rl/factor_read")
    for p in list((fr / "reads").glob("*/A.stats.json")) + list(
        (fr / "matched/reads").glob("*/A.stats.json")
    ):
        s = json.loads(p.read_text())
        out[p.parent.name] = s["rows"] / (s["k"] * 106)
    return out


def run_plan(a, thms: list[dict], cks: dict) -> None:
    root = Path(a.root)
    apa = fr_actions_per_attempt()
    mean = {st: sum(v for k_, v in apa.items() if cks.get(k_, {}).get("stage") == st) /
            max(1, sum(1 for k_ in apa if cks.get(k_, {}).get("stage") == st)) for st in ("base", "rl_end", "matched")}  # fmt: skip
    n_u = len(thms) * a.k
    n_f = sum(t["applicable"][d] for t in thms for d in FORCED) * a.kf
    say(f"plan: {len(thms)} theorems ({collections.Counter(t['set'] for t in thms)}); per checkpoint U {n_u} attempts "
        f"(k {a.k}), F {n_f} attempts (k_f {a.kf}), E {len(thms) * len(BASES)} states; batch {a.batch}")  # fmt: skip
    tot = 0.0
    sub: dict = collections.Counter()
    for cid in ckpt_ids(a.ckpts, cks):
        c = cks[cid]
        d = root / "raw" / cid
        todo = [
            s
            for s in a.arms.split(",")
            if not (d / ("E.json.gz" if s == "E" else f"{s}.jsonl.gz")).exists()
        ]
        x = apa[cid] if cid in apa else mean[c["stage"]]
        secs = (LOAD_SECS if todo else 0) + (E_SECS if "E" in todo else 0)
        secs += sum((n_u if s == "U" else n_f) * x / RATE[c["stage"]] for s in todo if s != "E")
        tot += secs
        sub[c.get("group", "main")] += secs
        say(f"  {c['priority']:2d} {cid:16s} {c['stage']:7s} dose {c['dose']:4s} s{c['seed']} x {c['x_dan'] or 0:.2f} "
            f"exists {c['exists']} todo {','.join(todo) or '-'}; {x:.1f} actions/attempt{'' if cid in apa else ' (stage mean)'}"
            f" -> ~{secs / 60:.0f} min")  # fmt: skip
    for g, v in sub.items():
        say(f"  group {g}: ~{v / 3600:.2f} GPU-h")
    say(f"plan total ~{tot / 3600:.2f} GPU-h at the factor read's rates ({RATE['base']:.0f} actions/s on bases, "
        f"{RATE['rl_end']:.0f} on RL ends); serrano's 200 W cap may add up to ~25 %")  # fmt: skip


# ------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("gpu", "rows", "plan"))
    ap.add_argument(
        "--ckpts", required=True, help="comma list of ckpt ids, or `all` (priority order)"
    )
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--sets", default=str(ROOT / "sets"))
    ap.add_argument("--gpu", type=int, default=0, help="nvidia-smi index (PCI order) to gate on")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--threads", type=int, default=8, help="torch threads on the CPU")
    ap.add_argument("--arms", default="E,U,F")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--kf", type=int, default=16)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--max_elems", type=int, default=2**30)
    ap.add_argument("--gpu_budget", type=float, default=10800.0)
    ap.add_argument("--limit", type=int, default=None, help="first N theorems of each set (smoke)")
    ap.add_argument("--names", default=None, help="comma list of theorem names (smoke)")
    ap.add_argument(
        "--no_lean", action="store_true", help="skip the Lean gate (tests only; rows need verdicts)"
    )
    a = ap.parse_args()
    if a.device == "cpu":
        assert Path(a.root).resolve() != ROOT.resolve(), "CPU runs write under another --root"
    thms, cks = load_sets(Path(a.sets), a.limit, a.names.split(",") if a.names else None)
    {"gpu": run_gpu, "rows": run_rows, "plan": run_plan}[a.phase](a, thms, cks)


if __name__ == "__main__":
    main()
