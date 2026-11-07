"""Checks 1-4 of the prereg (GPU; check 5 is test_factor_lib.py, check 6 is in analyze.py --control).

    python checks.py --ckpts ckpts.json --refs refs.jsonl --tag dose01_s2 --ckpt_dir DIR --check 1,2,3 --out OUT.json
    python checks.py --check 4 --reads READS --tag dose01_s2 --known KNOWN_it0000.json --out OUT.json   # CPU

1. An empty forced prefix at the same seed reproduces env_generate, byte for byte.
2. A forced prefix shows the state the env builds when the same actions are played through it: against Dan's
   `decompose` (base 0, names shifted to the attempt's base), against `ref_rows` (the path scorer's own states), and on
   sampled, non-reference attempts replayed as forced prefixes.
3. On next-action steps with p of about 0.05 or more, the sampled frequency of the exact action at k >= 2000 agrees
   with exp(log p) of the path scorer within Monte Carlo error. Also the scorer's full forward against the sampler's own
   prefill + cached decode (left-padded), deterministically.
4. The unforced read reproduces a known read given its seed and batch.
"""

# ruff: noqa: E501, SIM115, B905, B007, SIM300
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import time
from pathlib import Path

import gpu as G
from factor_lib import DAN, ref_from_json, ref_rows, replay, run_env, score_rows, shift_action
from fr_run import claim, gate, ledger_sum


def transfer_order() -> list[dict]:
    strict = set((DAN / "data/coe/sigma_heldout_strict.txt").read_text().split())
    rows = [json.loads(x) for x in open(DAN / "data/coe/transfer_track832.jsonl")]
    return [t for t in rows if t["name"] in strict]


def check1(model, tok, transfer: list[dict]) -> dict:
    from state_sample import env_generate  # type: ignore[import-not-found]

    out = {}
    for batch, seed, n_thm, k in ((16, 7, 8, 8), (64, 2500, 12, 8)):
        sub = transfer[:: max(1, len(transfer) // n_thm)][:n_thm]
        prompts = [t["prompt"] for t in sub for _ in range(k)]
        st1: dict = {}
        st2: dict = {}
        a = env_generate(model, tok, prompts, greedy=False, temperature=0.8, max_action=256, max_steps=48,
                         batch=batch, seed=seed, stats=st1)  # fmt: skip
        b = run_env(
            model, tok, [{"prompt": p} for p in prompts], 0.8, 256, 48, batch, seed, stats=st2
        )
        same = a == [x.nd for x in b]
        out[f"batch{batch}_seed{seed}"] = {
            "n": len(prompts), "byte_identical": same,
            "n_differ": sum(x != y.nd for x, y in zip(a, b)),
            "env_end_equal": dict(st1["env_end"]) == dict(st2["env_end"]),
            "env_end": dict(st1["env_end"]),
            "prompt_tokens_equal": st1["prompt_tokens"] == st2["prompt_tokens"],
            "waves_equal": st1["env_waves"] == st2["env_waves"],
            "n_finished_proofs": sum(x.end == "done" for x in b),
        }  # fmt: skip
    return out


def check2(model, tok, refs: list) -> dict:
    from state_env import decompose  # type: ignore[import-not-found]

    # reference prefixes: every boundary of every theorem
    items, cuts = [], []
    for r in refs:
        for name, cut in (
            ("pick", r.pick),
            ("reach", r.pick + 1),
            ("negi", r.bot + 1),
            ("dn", r.dn),
        ):
            items.append({"name": r.name, "prompt": r.prompt, "prefix": r.actions[:cut]})
            cuts.append((r, name, cut))
    atts = run_env(
        model, tok, items, 0.8, 256, 48, 1024, 123, stop_after=1, gate=False, keep_states=True
    )
    bad_dec, bad_rows = 0, 0
    bases = set()
    for (r, _name, cut), a in zip(cuts, atts):
        bases.add(a.base)
        steps, _, _ = decompose(r.prompt, r.nd, canon=True)
        want = tok.encode_toks(shift_action(steps[cut][0], a.base))
        if want != a.states[0]:
            bad_dec += 1
        if ref_rows(tok, r, a.base)[cut][0] != a.states[0]:
            bad_rows += 1
    # sampled attempts, not references: replay k of their actions as a forced prefix at the same base
    sub = [refs[i] for i in range(0, len(refs), 5)]
    its = [{"name": r.name, "prompt": r.prompt} for r in sub for _ in range(4)]
    un = run_env(model, tok, its, 0.8, 256, 48, 256, 321, gate=False, keep_states=True)
    rng = random.Random(0)
    n_rep, bad_rep = 0, 0
    for it, a in zip(its, un):
        ok_acts = [x for x, e in zip(a.raw, a.ended) if e]
        tr = replay(it["prompt"], ok_acts, a.base)
        if len(tr.acts) < 2:
            continue
        j = rng.randrange(1, len(tr.acts))
        un0 = [
            [f"n{int(t[1:]) - a.base}" if t.startswith("n") and t[1:].isdigit() else t for t in x]
            for x in tr.acts[:j]
        ]
        f = run_env(model, tok, [{"name": it["name"], "prompt": it["prompt"], "prefix": un0}], 0.8, 256, 48, 1, 5,
                    stop_after=1, gate=False, keep_states=True, base_fixed=a.base)  # fmt: skip
        n_rep += 1
        bad_rep += f[0].states[0] != a.states[j]
    return {"n_boundaries": len(cuts), "n_bases_seen": len(bases), "differ_from_decompose": bad_dec,
            "differ_from_ref_rows": bad_rows, "n_sampled_replays": n_rep, "differ_sampled_replay": bad_rep}  # fmt: skip


def cached_logp(model, tok, rows: list, temperature: float = 0.8) -> list[float]:
    """Sampler's path: left-padded prefill and cached steps (sample.generate_ids_fast), teacher-forced."""
    import torch
    import torch.nn.functional as F

    dev = next(model.parameters()).device
    B = len(rows)
    L = max(len(r[0]) for r in rows)
    M = max(len(r[1]) for r in rows)
    idx = torch.full((B, L), tok.pad, dtype=torch.long, device=dev)
    keep = torch.zeros((B, L), dtype=torch.bool, device=dev)
    for i, (p, _) in enumerate(rows):
        idx[i, L - len(p) :] = torch.tensor(p, device=dev)
        keep[i, L - len(p) :] = True
    pos = (keep.cumsum(1) - 1).clamp(min=0)
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev))
    mask = causal[None, None] & keep[:, None, None, :]
    mask = mask | torch.eye(L, dtype=torch.bool, device=dev)[None, None]
    caches = [{"max": L + M} for _ in model.blocks]
    out = [0.0] * B
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(idx, pos=pos, mask=mask, caches=caches)[:, -1]
        keepb = torch.ones((B, L + M), dtype=torch.bool, device=dev)
        keepb[:, :L] = keep
        cur = pos[:, -1]
        for t in range(M):
            lp = F.log_softmax(logits.float() / temperature, -1)
            nxt = torch.full((B,), tok.pad, dtype=torch.long, device=dev)
            for i, (_, a) in enumerate(rows):
                if t < len(a):
                    out[i] += float(lp[i, a[t]])
                    nxt[i] = a[t]
            cur = cur + 1
            logits = model(
                nxt[:, None],
                pos=cur[:, None],
                mask=keepb[:, None, None, : L + t + 1],
                caches=caches,
            )[:, -1]
    return out


def check3(model, tok, refs: list, k: int) -> dict:
    # one fixed name base, all reference actions: the deterministic scores
    base = 11
    rows, idx = [], []
    for r in refs:
        rr = ref_rows(tok, r, base)
        rows += rr
        idx += [(r, i) for i in range(len(rr))]
    lp = score_rows(model, tok, rows, 0.8, max_elems=2**30)
    # scorer's full forward vs the sampler's cached path, on a spread of 256 rows
    sel = list(range(0, len(rows), max(1, len(rows) // 256)))[:256]
    lp2 = cached_logp(model, tok, [rows[i] for i in sel])
    diffs = [abs(lp[i] - x) for i, x in zip(sel, lp2)]
    rel = [abs(math.exp(lp[i]) - math.exp(x)) for i, x in zip(sel, lp2)]
    res: dict = {"forward_vs_cached": {"n": len(sel), "max_abs_nats": max(diffs), "mean_abs_nats": sum(diffs) / len(diffs),
                                       "max_abs_prob": max(rel)}, "steps": []}  # fmt: skip
    # sampled frequency at p of about 0.08, 0.2 and 0.5
    p = [math.exp(x) for x in lp]
    used = set()
    for target in (0.08, 0.2, 0.5):
        best = None
        for j, pj in enumerate(p):
            if pj < 0.05 or j in used:
                continue
            if best is None or abs(pj - target) < abs(p[best] - target):
                best = j
        if best is None:
            continue
        used.add(best)
        r, ai = idx[best]
        want = shift_action(r.actions[ai], base)
        item = {"name": r.name, "prompt": r.prompt, "prefix": r.actions[:ai]}
        atts = run_env(
            model,
            tok,
            [item] * k,
            0.8,
            256,
            48,
            1024,
            999 + len(used),
            stop_after=1,
            gate=False,
            base_fixed=base,
        )
        hits = sum(a.raw[0] == want and a.ended[0] for a in atts)
        f = hits / k
        se = math.sqrt(p[best] * (1 - p[best]) / k)
        res["steps"].append({"theorem": r.name, "action_index": ai, "group": r.groups[ai], "k": k, "hits": hits, "freq": f,
                             "p": p[best], "se": se, "z": (f - p[best]) / se})  # fmt: skip
    return res


def check4(reads: Path, tag: str, known: Path) -> dict:
    with gzip.open(reads / tag / "A.jsonl.gz", "rt") as f:
        rows = [json.loads(x) for x in f]
    kn = json.loads(known.read_text())
    kp = {}
    for x in open(known.with_suffix(".proofs.jsonl")):
        y = json.loads(x)
        kp[y["name"]] = y["proofs"]
    st = json.loads((reads / tag / "A.stats.json").read_text())
    solved = sorted({r["name"] for r in rows if r["ok"]})
    first32 = sorted({r["name"] for r in rows if r["ok"] and r["draw"] < 32})
    proofs: dict = {}
    n_ok: dict = {}
    for r in rows:
        if r["ok"]:
            proofs.setdefault(r["name"], [])
            n_ok[r["name"]] = n_ok.get(r["name"], 0) + 1
            if r["nd"] not in proofs[r["name"]]:
                proofs[r["name"]].append(r["nd"])
    ke = kn["env_stats"]
    return {
        "known": str(known), "noise_seed_known": kn["noise_seed"], "noise_seed_here": st["noise_seed"],
        "batch_known": kn["batch"], "batch_here": st["batch"], "host_known": kn["host"], "host_here": st["host"],
        "strict_solved_known": kn["strict_solved"], "strict_solved_here": solved,
        "first32_known": kn["first32_solved"], "first32_here": first32,
        "n_ok_equal": {n: kn["sigma"][n]["n_ok"] == n_ok.get(n, 0) for n in kn["sigma"]},
        "proofs_byte_identical": proofs == kp,
        "env_end_known": ke["env_end"], "env_end_here": st["env_end"],
        "prompt_tokens_known": ke["prompt_tokens"], "prompt_tokens_here": st["prompt_tokens"],
        "waves_known": ke["env_waves"], "waves_here": st["env_waves"],
        "pass": solved == kn["strict_solved"] and first32 == kn["first32_solved"] and proofs == kp
                and ke["env_end"] == st["env_end"] and ke["prompt_tokens"] == st["prompt_tokens"],
    }  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts")
    ap.add_argument("--refs")
    ap.add_argument("--tag", default="dose01_s2")
    ap.add_argument("--ckpt_dir", default=None)
    ap.add_argument("--check", default="1,2,3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k3", type=int, default=4000)
    ap.add_argument("--reads", default=None)
    ap.add_argument("--known", default=None)
    ap.add_argument("--ledger", default=None)
    a = ap.parse_args()
    which = a.check.split(",")
    res: dict = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    if "4" in which:
        res["check4"] = check4(Path(a.reads), a.tag, Path(a.known))
    gpu_checks = [c for c in which if c in ("1", "2", "3")]
    if gpu_checks:
        ledger = Path(a.ledger)
        gate(first=True)
        claim(ledger.parent / f"{G.HOST}_gpu{G_GPU}.lock")
        from model import load_ckpt  # type: ignore[import-not-found]

        cks = {c["tag"]: c for c in json.load(open(a.ckpts))}
        c = cks[a.tag]
        path = str(Path(a.ckpt_dir) / c["file"]) if a.ckpt_dir else c["path"]
        t0 = time.time()
        model, tok, _ = load_ckpt(path, "cuda")
        all_refs = {r.name: r for r in (ref_from_json(json.loads(x)) for x in open(a.refs))}
        transfer = transfer_order()
        refs = [all_refs[t["name"]] for t in transfer if t["name"] in all_refs]
        if "1" in which:
            res["check1"] = check1(model, tok, transfer)
            print("check1", res["check1"], flush=True)
        if "2" in which:
            res["check2"] = check2(model, tok, refs)
            print("check2", res["check2"], flush=True)
        if "3" in which:
            res["check3"] = check3(model, tok, refs, a.k3)
            print("check3", res["check3"], flush=True)
        res["gpu_secs_" + "_".join(gpu_checks)] = time.time() - t0
        with open(ledger, "a") as f:
            f.write(json.dumps({"utc": time.strftime("%FT%TZ", time.gmtime()), "tag": "checks", "stage": a.check,
                                "secs": time.time() - t0}) + "\n")  # fmt: skip
        print(f"GPU total {ledger_sum(ledger):.0f} s", flush=True)
    res.setdefault("host", os.uname().nodename)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k == "check4"}, indent=1, default=str))


G_GPU = 0

if __name__ == "__main__":
    main()
