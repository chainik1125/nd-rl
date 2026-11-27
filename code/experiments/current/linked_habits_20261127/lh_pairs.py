"""Stage 2 inputs of linked habits version 2 (SPEC2.md): the KL anchor's states, the base's own proofs on Dan's RL
pool, and the preference pairs, fine-tune files and reward-model probe twins built from them.

    python lh_pairs.py anchor --corpus C [--n 2000] [--min_hyp 0] [--out FILE] [--dry-run]        # CPU
    python lh_pairs.py sample --corpus C [--k 8] [--batch 256] [--device cuda] [--limit N]         # GPU
    python lh_pairs.py build  --corpus C --rater {reit,atom} [--heldout 200] [--workers 8] [--limit N]   # CPU (Lean)

`anchor`: LH/data/anchor_<c>.jsonl, the `--kl_anchor_data` of both GRPO arms. Builder: `code.thinking.canary.proof_row`
(the row builder of the knockout GRPO snapshot's collapse_audit/build_canary.py, which writes the canary_long file the
trainer reads; that script itself reads fixed long-proof pools and sealed-set indexes, so it is not run here). Input:
`--n` (2,000) records of the corpus drawn with seed 20261127 from its untagged records (`tag_shown` false: the states
as the model sees them without a tag); every step of each proof is kept (`--min_hyp 0`; the canary's default 10 keeps
only long states, which these 4-16-line proofs rarely reach). The trainer's anchor takes `--kl_anchor_n` (512) of these
steps per AdamW step. A meta file beside it gives the counts and the source's sha256. The first 256 of the drawn records
also go to LH/data/klstates_<c>.jsonl ({key, prompt, proof}): the offline trainers' fixed states for their logged KL
to the base (`--kl-states`).

`sample`: the corpus's base (LH/bases/stage1_<c>_s0.pt), untagged, Dan's sampler (T 0.8, 256 tokens per action, step
limit 51, his name-base draw, Lean gate) on every theorem of Dan's RL pool (4,495) x k 8, noise seed 20261127; the
accepted proofs go to LH/pairs/<c>_s0/sampled.jsonl.gz (raw proof text: never committed), counts to sample.json
(attempts, accepted, attempts at the step limit).

`build --rater R` (rated habit: reit -> "reit", atom -> "note_atom"): every distinct accepted proof is read by
`lh_profile.profile`. A proof with a site of the rated habit, all of its sites in one form, gets its twin from the
corpus agent's `lh_twin.twin(prompt, proof, habit, tables)` (the same proof with that habit flipped at every site,
nd_verify-accepted and replayed; None when it cannot), and every twin is then checked by Lean in one batch
(`lh_twin.lean_check_twins`, the corpus builder's Lean check); a twin Lean rejects drops its pair (counted). The twin
is read again: it must have the same number of sites of the rated habit, all in the other form, and the same reading
of every other habit; otherwise the pair is dropped (counted). The A-form member is "preferred". Theorems are shuffled
(seed 20261127) and taken whole into the held-out split until both sampled sides have at least 100 pairs there; the
rest is train. Then (PREREG Amendment 3; design audit F1, F2, F5), written to LH/pairs/<c>_s0/<R>/:
  train.jsonl, heldout.jsonl   {"key", "prompt", "preferred", "other", "habit", "sampled_side": "A" | "B"}: each split
                               subsampled (seeded per corpus, rater and split) to equal numbers of A-sampled and
                               B-sampled pairs (held-out: 100 + 100 at most); train then capped at 4,000 pairs (2,000
                               per side, a seeded draw: PREREG Amendment 7 point 5); DPO, its swap control and the RM
  sft_pref.jsonl               {"key", "prompt", "proof"}: the preferred member of the train pairs before the balancing
                               (the base's own mix), capped at 4,000 by one seeded draw of pairs: the rated fine-tune
  sft_sampled.jsonl            the sampled member of the same pairs: the fine-tune control
  probe_twins.jsonl            pair rows from the held-out theorems' sampled proofs: twins of every OTHER flippable
                               habit (lh_twin.FLIPPABLE minus the rated one) with a site in one form, the same checks
                               and Lean, `preferred` = the A-form member, at most 400 per habit (seeded); plus up to
                               300 `probe:nosite` and 300 `probe:length` rows (Amendment 7 point 6): lh_rm.py probe
  summary.json                 every count above (before / after the balancing and the cap, `cap` with `*_below_cap`,
                               why each proof gave no pair, probe rows per habit), the A-share of every habit in
                               sft_pref, sft_sampled and all the base's readable sampled proofs, and `norm_tokens` (N =
                               64 x the mean action tokens per proof of sft_sampled: lh_sft --norm-tokens).
Proofs whose rated-habit sites are in both forms are skipped (counted): flipping them gives no A-form member.
"""

# ruff: noqa: E501, I001
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lh_core as C  # noqa: E402  (TA/AT = LH, sys.path)


def read_jsonl(fn: Path) -> list[dict[str, Any]]:
    op = gzip.open if str(fn).endswith(".gz") else open
    with op(fn, "rt", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(fn: Path, rows: list[dict[str, Any]]) -> None:
    fn.parent.mkdir(parents=True, exist_ok=True)
    tmp = fn.with_name(fn.name + ".tmp")
    op = gzip.open if str(fn).endswith(".gz") else open
    with op(tmp, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(fn)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ------------------------------------------------------------------------------------------------ anchor
def anchor_main(argv: list[str]) -> dict[str, Any]:
    ap = argparse.ArgumentParser(prog="lh_pairs.py anchor")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--n", type=int, default=C.ANCHOR_PROOFS)
    ap.add_argument("--min_hyp", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else C.anchor_path(a.corpus)
    if out.exists():
        raise SystemExit(f"{out} exists: never overwritten")
    src = C.corpus_path(a.corpus)
    recs = [r for r in read_jsonl(src) if not r.get("tag_shown")]
    rng = random.Random(f"{C.SEED2}/anchor/{a.corpus}")
    pick = rng.sample(range(len(recs)), min(a.n, len(recs)))
    import tag_core as T

    T.install_thinking()
    from code.thinking.canary import proof_row  # type: ignore[import-not-found]

    rows, steps, skipped = [], 0, 0
    for i in pick:
        r = recs[i]
        row = proof_row(f"{a.corpus}:{r.get('key', i)}", r["prompt"], r["proof"], a.min_hyp, source=f"lh_{a.corpus}",
                        n_lines=r.get("n_lines"), key=r.get("key"), author=r.get("author"))  # fmt: skip
        if row is None:
            skipped += 1
            continue
        rows.append(row)
        steps += len(row["steps"])
    kl = [
        {"key": recs[i].get("key", str(i)), "prompt": recs[i]["prompt"], "proof": recs[i]["proof"]}
        for i in pick[: C.KL_STATES]
    ]
    kl_out = out.with_name(out.stem + ".klstates.jsonl") if a.out else C.kl_states_path(a.corpus)
    meta = {"corpus": a.corpus, "source": str(src), "kl_states": str(kl_out), "kl_states_n": len(kl), "source_sha256": C.sha256(src), "untagged_records": len(recs),
            "drawn": len(pick), "seed": f"{C.SEED2}/anchor/{a.corpus}", "rows": len(rows), "steps": steps,
            "no_step": skipped, "min_hyp": a.min_hyp, "builder": "code.thinking.canary.proof_row (knockout GRPO snapshot)",
            "written_utc": now()}  # fmt: skip
    print(
        f"anchor {a.corpus}: {len(rows):,} proofs, {steps:,} steps (of {len(recs):,} untagged records, {len(pick):,} drawn)"
    )
    if not a.dry_run:
        write_jsonl(out, rows)
        write_jsonl(kl_out, kl)
        out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
        print(f"wrote {out}")
    return meta


# ------------------------------------------------------------------------------------------------ sample
def pool_rows(limit: int = 0) -> list[dict[str, Any]]:
    rows = read_jsonl(C.RL_POOL)
    return rows[:limit] if limit else rows


def sample_main(argv: list[str]) -> dict[str, Any]:
    ap = argparse.ArgumentParser(prog="lh_pairs.py sample")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--ckpt", default=None, help="default: the corpus's base")
    ap.add_argument("--k", type=int, default=C.PAIR_K)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--base_seed", type=int, default=0, help="1: the second base (PREREG Amendment 8 point 2)"
    )
    a = ap.parse_args(argv)
    d = C.pairs_dir(a.corpus, seed=a.base_seed)
    out, summ = d / "sampled.jsonl.gz", d / "sample.json"
    if out.exists() or summ.exists():
        raise SystemExit(f"{out} exists: never overwritten")
    ck = Path(a.ckpt) if a.ckpt else C.base_path(a.corpus, a.base_seed)
    import lh_measure as LM

    T = LM.T
    LM.install_step_limit()
    os.environ.setdefault("LEAN_GATE_LOG", str(d / "lean_gate.jsonl"))
    d.mkdir(parents=True, exist_ok=True)
    import torch
    from model import load_ckpt  # type: ignore[import-not-found]
    from state_sample import env_generate, env_stats_json  # type: ignore[import-not-found]

    if a.device == "cpu":
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    elif os.environ.get("DANREAD_MEM_FRAC"):
        torch.cuda.set_per_process_memory_fraction(float(os.environ["DANREAD_MEM_FRAC"]))
    model, tok, _ = load_ckpt(str(ck), a.device)
    thms = pool_rows(a.limit)
    prompts = [t["prompt"] for t in thms for _ in range(a.k)]
    T.set_process_tag("none")
    T.spy_dan("none")
    s0 = T.SPY["states"]
    st: dict = {}
    t0 = time.time()
    nds = env_generate(model, tok, prompts, greedy=False, temperature=0.8, max_action=256, max_steps=C.STEP_LIMIT,
                       batch=a.batch, seed=C.SEED2, stats=st, gate=True)  # fmt: skip
    rows = []
    for j, t in enumerate(thms):
        for i, x in enumerate(nds[j * a.k : (j + 1) * a.k]):
            if not x.startswith(("LEANREJ", "LEANPARSE")):
                rows.append(
                    {
                        "theorem": t["name"],
                        "key": t.get("key"),
                        "prompt": t["prompt"],
                        "attempt": i,
                        "proof": x,
                    }
                )
    write_jsonl(out, rows)
    doc = {"corpus": a.corpus, **({"base_seed": a.base_seed} if a.base_seed else {}), "ckpt": str(ck), "ckpt_sha256": C.sha256(ck),
           "pool": str(C.RL_POOL), "pool_sha256": C.sha256(C.RL_POOL),
           "theorems": len(thms), "k": a.k, "attempts": len(nds), "accepted": len(rows),
           "accepted_pct": round(100 * len(rows) / max(1, len(nds)), 2), "theorems_with_accepted": len({r["theorem"] for r in rows}),
           "step_limit": C.STEP_LIMIT, "step_cap": sum(LM.at_limit(x) for x in nds),
           "step_cap_pct": round(100 * sum(LM.at_limit(x) for x in nds) / max(1, len(nds)), 2), "noise_seed": C.SEED2,
           "temperature": 0.8, "max_action": 256, "tag": "none", "states_checked": T.SPY["states"] - s0,
           "spy_by_tag": dict(T.SPY["by_tag"]), "env": env_stats_json(st), "secs": round(time.time() - t0, 1), "utc": now()}  # fmt: skip
    summ.write_text(json.dumps(doc, indent=1, default=str))
    print(f"sample {a.corpus}: {len(rows):,} / {len(nds):,} accepted ({doc['accepted_pct']} %), at the step limit "
          f"{doc['step_cap_pct']} %; {doc['secs']:.0f} s", flush=True)  # fmt: skip
    return doc


# ------------------------------------------------------------------------------------------------ build
def form_of(h: dict[str, Any]) -> str | None:
    """'A' / 'B' when every site of a habit is in that form; 'mixed' when both; None when no site."""
    s, a, b = int(h.get("sites") or 0), int(h.get("a") or 0), int(h.get("b") or 0)
    if not s:
        return None
    if a == s and not b:
        return "A"
    if b == s and not a:
        return "B"
    return "mixed"


def reading(p: dict[str, Any], skip: str) -> dict[str, tuple[int, int, int]]:
    return {
        h: (int(v.get("sites") or 0), int(v.get("a") or 0), int(v.get("b") or 0))
        for h, v in p["habits"].items()
        if h != skip
    }


def hab_ab(p: dict[str, Any]) -> dict[str, tuple[int, int]]:
    return {h: (int(v.get("a") or 0), int(v.get("b") or 0)) for h, v in p["habits"].items()}


def one(job: tuple[dict[str, Any], str]) -> dict[str, Any]:
    """One sampled proof -> its pair (or why not), with its reading (A, B per habit) when it is readable."""
    r, habit = job
    import lh_profile as P
    import lh_twin as TW

    tab = _TAB.get("t")
    if tab is None:
        tab = _TAB["t"] = C.profile_tables()
    out: dict[str, Any] = {
        "theorem": r["theorem"],
        "key": f"{r['theorem']}#{r['attempt']}",
        "prompt": r["prompt"],
    }
    p = P.profile(r["prompt"], r["proof"], tab)
    if not p.get("ok"):
        return {**out, "why": "unread"}
    out["hab"] = hab_ab(p)
    out["lines"] = int(p.get("lines") or 0)
    out["reit_form"] = form_of(
        p["habits"][C.RATERS["reit"]]
    )  # the probe rows of Amendment 7 point 6
    f = form_of(p["habits"][habit])
    if f is None:
        return {**out, "why": "no_site"}
    if f == "mixed":
        return {**out, "why": "mixed"}
    tw, why = checked_twin(P, TW, tab, r["prompt"], r["proof"], p, habit, f)
    if tw is None:
        return {**out, **why, "sampled_side": f}
    a_proof, b_proof = (r["proof"], tw) if f == "A" else (tw, r["proof"])
    return {**out, "why": "pair", "sampled_side": f, "preferred": a_proof, "other": b_proof}


def checked_twin(P: Any, TW: Any, tab: Any, prompt: str, proof: str, p: dict[str, Any], habit: str,
                 f: str) -> tuple[str | None, dict[str, Any]]:  # fmt: skip
    """lh_twin.twin, read again: `habit` flipped at every site, every other habit's reading unchanged (Lean later)."""
    try:
        tw = TW.twin(prompt, proof, habit, tab)
    except Exception as e:  # noqa: BLE001  (the twin must fail closed; a crash is counted, never fatal)
        return None, {"why": "twin_error", "err": f"{type(e).__name__}: {e}"[:200]}
    if tw is None:
        return None, {"why": "no_twin"}
    q = P.profile(prompt, tw, tab)
    g = form_of(q["habits"][habit]) if q.get("ok") else "unread"
    if g != {"A": "B", "B": "A"}[f] or q["habits"][habit]["sites"] != p["habits"][habit]["sites"]:
        return None, {"why": "twin_rated_mismatch"}
    if reading(q, habit) != reading(p, habit):
        return None, {"why": "twin_other_mismatch"}
    return tw, {}


def probe_one(job: tuple[dict[str, Any], str]) -> list[dict[str, Any]]:
    """One held-out sampled proof -> twins of every other flippable habit with a site in one form (design audit F2):
    pair rows with `preferred` the A-form member."""
    r, rated = job
    import lh_profile as P
    import lh_twin as TW

    tab = _TAB.get("t")
    if tab is None:
        tab = _TAB["t"] = C.profile_tables()
    p = P.profile(r["prompt"], r["proof"], tab)
    if not p.get("ok"):
        return []
    out = []
    for h in TW.FLIPPABLE:
        if h == rated or h not in p["habits"]:
            continue
        f = form_of(p["habits"][h])
        if f not in ("A", "B"):
            continue
        tw, why = checked_twin(P, TW, tab, r["prompt"], r["proof"], p, h, f)
        row = {"theorem": r["theorem"], "key": f"{r['theorem']}#{r['attempt']}#{h}", "prompt": r["prompt"], "habit": h,
               "sampled_side": f}  # fmt: skip
        if tw is None:
            out.append({**row, "why": why["why"]})
        else:
            a_proof, b_proof = (r["proof"], tw) if f == "A" else (tw, r["proof"])
            out.append({**row, "why": "pair", "preferred": a_proof, "other": b_proof})
    return out


_TAB: dict[str, Any] = {}


def split(
    pairs: list[dict[str, Any]], n_held: int, seed: str
) -> tuple[list[dict], list[dict], set[str]]:
    """(train, heldout, held-out theorems): whole theorems (shuffled, seeded) into held-out until both sampled sides
    have at least n_held / 2 pairs there, so the balanced subsample can hold n_held; the rest is train."""
    by: dict[str, list[dict]] = {}
    for p in pairs:
        by.setdefault(p["theorem"], []).append(p)
    names = sorted(by)
    random.Random(seed).shuffle(names)
    held_names: set[str] = set()
    n = {"A": 0, "B": 0}
    for t in names:
        if min(n.values()) >= (n_held + 1) // 2:
            break
        held_names.add(t)
        for p in by[t]:
            n[p["sampled_side"]] += 1
    train = [p for t in names if t not in held_names for p in by[t]]
    held = [p for t in names if t in held_names for p in by[t]]
    return train, held, held_names


def balance(pairs: list[dict[str, Any]], seed: str, cap: int | None = None) -> list[dict[str, Any]]:
    """Equal numbers of A-sampled and B-sampled pairs (PREREG Amendment 3, point 2; design audit F5): a seeded sample
    of the larger side, `cap` / 2 of each at most; file order kept."""
    rng = random.Random(seed)
    idx = {k: [i for i, x in enumerate(pairs) if x["sampled_side"] == k] for k in ("A", "B")}
    n = min(len(v) for v in idx.values())
    if cap is not None:
        n = min(n, cap // 2)
    keep = sorted(i for k in ("A", "B") for i in rng.sample(idx[k], n))
    return [pairs[i] for i in keep]


def shares(readings: list[dict[str, tuple[int, int]]]) -> dict[str, float | None]:
    tot: dict[str, list[int]] = {}
    for hb in readings:
        for h, (a, b) in hb.items():
            v = tot.setdefault(h, [0, 0])
            v[0] += a
            v[1] += b
    return {
        h: (round(100 * a / (a + b), 2) if a + b else None) for h, (a, b) in sorted(tot.items())
    }


def lean_filter(rows: list[dict[str, Any]], workers: int) -> None:
    """Lean on every twin (lh_twin checks nd_verify and replay only), one batch: a rejected twin drops its pair."""
    import lh_twin as TW

    idx = [i for i, x in enumerate(rows) if x["why"] == "pair"]
    txt = [
        rows[i]["other"] if rows[i]["sampled_side"] == "A" else rows[i]["preferred"] for i in idx
    ]
    ok = (
        TW.lean_check_twins(
            [(rows[i]["prompt"], t) for i, t in zip(idx, txt, strict=True)], workers
        )
        if idx
        else []
    )
    for i, v in zip(idx, ok, strict=True):
        if not v:
            x = rows[i]
            rows[i] = {k: v2 for k, v2 in x.items() if k not in ("preferred", "other")} | {
                "why": "twin_lean_rejected"
            }


def pool_map(fn: Any, jobs: list[Any], workers: int) -> list[Any]:
    if workers > 1 and len(jobs) > 1:
        import multiprocessing as mp

        with mp.get_context("fork").Pool(workers) as P:
            return P.map(fn, jobs, chunksize=16)
    return [fn(j) for j in jobs]


def norm_tokens(path: Path, batch: int) -> dict[str, Any]:
    """The fine-tune normalizer (PREREG Amendment 7 point 4, audit T8): N = batch x the mean action tokens per readable
    proof of `path`, from the trainer's own function (lh_tune.mean_action_tokens, as `lh_tune.py norm-tokens` prints
    it: action ids and <eos> of every step). N is kept unrounded; lh_sft takes it as a float."""
    import lh_tune as U
    import tag_core as T

    U.setup("cpu", 1)
    from tokenizer import make_tokenizer  # type: ignore[import-not-found]

    tok = make_tokenizer(T.MODE)
    mean, n = U.mean_action_tokens(path, tok)
    rep = U.load_proofs(path, tok)[1]
    return {"file": path.name, "batch": batch, "proofs": n, "unreadable": rep["unreadable"],
            "action_tokens": int(round(mean * n)), "mean_action_tokens_per_proof": mean, "N": batch * mean,
            "rule": "lh_tune.mean_action_tokens on sft_sampled.jsonl x batch (lh_tune.py norm-tokens)"}  # fmt: skip


def extra_probe_rows(res: list[dict[str, Any]], proofs: dict[str, str], held: set[str], seed: str,
                     cap: int) -> tuple[list[dict[str, Any]], dict[str, int]]:  # fmt: skip
    """Amendment 7 point 6 (audit T1): reward-model probe rows from the held-out theorems' readable sampled proofs.
    probe:nosite: a proof without a reiteration site against one with a site, same theorem (preferred = with a site);
    probe:length: two proofs with the same reiteration form (both A or both B) and different numbers of lines, same
    theorem (preferred = the longer). Within a theorem, proofs are shuffled (seeded) and paired without reuse; then a
    seeded draw of at most `cap` rows of each kind."""
    rng = random.Random(seed)
    by_t: dict[str, list[dict[str, Any]]] = {}
    for x in res:
        if x["theorem"] in held and "hab" in x and x["key"] in proofs:
            by_t.setdefault(x["theorem"], []).append(x)
    kinds: dict[str, list[dict[str, Any]]] = {"probe:nosite": [], "probe:length": []}

    def row(kind: str, i: int, w: dict[str, Any], o: dict[str, Any]) -> dict[str, Any]:
        return {"key": f"{w['theorem']}#{kind.split(':')[1]}{i}", "prompt": w["prompt"], "preferred": proofs[w["key"]],
                "other": proofs[o["key"]], "habit": kind, "sampled_side": None,
                "lines": [w["lines"], o["lines"]], "reit_form": [w["reit_form"], o["reit_form"]]}  # fmt: skip

    for t in sorted(by_t):
        xs = by_t[t][:]
        rng.shuffle(xs)
        no = [x for x in xs if x["reit_form"] is None]
        yes = [x for x in xs if x["reit_form"] is not None]
        for i, (w, o) in enumerate(zip(yes, no, strict=False)):
            kinds["probe:nosite"].append(row("probe:nosite", i, w, o))
        i = 0
        for f in ("A", "B"):
            grp = [x for x in xs if x["reit_form"] == f]
            while len(grp) >= 2:
                u = grp.pop()
                j = next((k for k, v in enumerate(grp) if v["lines"] != u["lines"]), None)
                if j is None:
                    continue
                v = grp.pop(j)
                w, o = (u, v) if u["lines"] > v["lines"] else (v, u)
                kinds["probe:length"].append(row("probe:length", i, w, o))
                i += 1
    out, counts = [], {}
    for k, xs in kinds.items():
        keep = rng.sample(xs, cap) if len(xs) > cap else xs
        counts[k] = len(keep)
        counts[f"{k} (before the cap)"] = len(xs)
        out += keep
    return out, counts


def build_main(argv: list[str]) -> dict[str, Any]:
    ap = argparse.ArgumentParser(prog="lh_pairs.py build")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--rater", required=True, choices=list(C.RATERS))
    ap.add_argument("--heldout", type=int, default=C.HELDOUT_PAIRS)
    ap.add_argument("--probe_per_habit", type=int, default=C.PROBE_TWINS_PER_HABIT)
    ap.add_argument(
        "--no_probe",
        action="store_true",
        help="no reward-model probe twins (a rater without a GRPO line)",
    )
    ap.add_argument("--workers", type=int, default=int(os.environ.get("LEAN_GATE_WORKERS", "8")))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--base_seed", type=int, default=0,
                    help="1: the second base's proofs and pairs (LH/pairs/<c>_s1; the same seeded draws as the first)")  # fmt: skip
    a = ap.parse_args(argv)
    habit = C.RATERS[a.rater]
    d = C.pairs_dir(a.corpus, a.rater, a.base_seed)
    if (d / "summary.json").exists():
        raise SystemExit(f"{d / 'summary.json'} exists: never overwritten")
    rows = read_jsonl(C.pairs_dir(a.corpus, seed=a.base_seed) / "sampled.jsonl.gz")
    seen: set[tuple[str, str]] = set()
    uniq = []
    for r in rows:
        if (r["theorem"], r["proof"]) not in seen:
            seen.add((r["theorem"], r["proof"]))
            uniq.append(r)
    uniq = uniq[: a.limit or None]
    t0 = time.time()
    res = pool_map(one, [(r, habit) for r in uniq], a.workers)
    lean_filter(res, a.workers)
    why: dict[str, int] = {}
    for x in res:
        why[x["why"]] = why.get(x["why"], 0) + 1
    pairs = [x for x in res if x["why"] == "pair"]
    seed = f"{C.SEED2}/split/{a.corpus}/{a.rater}"
    train_all, held_all, held_names = split(pairs, a.heldout, seed)
    train_bal = balance(train_all, f"{C.SEED2}/balance/{a.corpus}/{a.rater}/train")
    # Amendment 7 point 5: a seeded draw of at most 2,000 per sampled side after the balancing
    train = balance(train_bal, f"{C.SEED2}/cap/{a.corpus}/{a.rater}/train", cap=C.TRAIN_PAIR_CAP)
    held = balance(held_all, f"{C.SEED2}/balance/{a.corpus}/{a.rater}/heldout", cap=a.heldout)

    def prow(x: dict[str, Any], h: str = habit) -> dict[str, Any]:
        return {"key": x["key"], "prompt": x["prompt"], "preferred": x["preferred"], "other": x["other"],
                "habit": h, "sampled_side": x["sampled_side"]}  # fmt: skip

    def sides(xs: list[dict[str, Any]]) -> dict[str, int]:
        return {k: sum(x["sampled_side"] == k for x in xs) for k in ("A", "B")}

    # the fine-tune files (PREREG Amendment 3, point 5; design audit F1): the members of the train pairs before the
    # balancing; Amendment 7 point 5: a seeded draw of at most 4,000 of those pairs, the same pairs in both files
    rs = random.Random(f"{C.SEED2}/cap/{a.corpus}/{a.rater}/sft")
    sft_src = (
        [train_all[i] for i in sorted(rs.sample(range(len(train_all)), C.SFT_PROOF_CAP))]
        if len(train_all) > C.SFT_PROOF_CAP
        else train_all
    )
    sft_pref = [{"key": x["key"], "prompt": x["prompt"], "proof": x["preferred"]} for x in sft_src]
    sft_sam = [{"key": x["key"], "prompt": x["prompt"],
                "proof": x["preferred"] if x["sampled_side"] == "A" else x["other"]} for x in sft_src]  # fmt: skip
    pref_read = [{**x["hab"], habit: (x["hab"][habit][0] + x["hab"][habit][1], 0)} for x in sft_src]
    comp = {"base_sampled": shares([x["hab"] for x in res if "hab" in x]), "sft_pref": shares(pref_read),
            "sft_sampled": shares([x["hab"] for x in sft_src])}  # fmt: skip
    # held-out twins of every other flippable habit, for the reward-model probe (design audit F2)
    held_rows = [] if a.no_probe else [r for r in uniq if r["theorem"] in held_names]
    probe = [
        y for ys in pool_map(probe_one, [(r, habit) for r in held_rows], a.workers) for y in ys
    ]
    lean_filter(probe, a.workers)
    probe_why: dict[str, dict[str, int]] = {}
    for x in probe:
        w = probe_why.setdefault(x["habit"], {})
        w[x["why"]] = w.get(x["why"], 0) + 1
    rng = random.Random(f"{C.SEED2}/probe/{a.corpus}/{a.rater}")
    probe_rows = []
    for h in sorted(probe_why):
        ok = [x for x in probe if x["habit"] == h and x["why"] == "pair"]
        probe_rows += [
            prow(x, h)
            for x in (rng.sample(ok, a.probe_per_habit) if len(ok) > a.probe_per_habit else ok)
        ]
    extra, extra_n = ([], {}) if a.no_probe else extra_probe_rows(
        res, {f"{r['theorem']}#{r['attempt']}": r["proof"] for r in uniq}, held_names,
        f"{C.SEED2}/probe_extra/{a.corpus}/{a.rater}", C.PROBE_EXTRA_ROWS)  # fmt: skip
    doc = {"corpus": a.corpus, **({"base_seed": a.base_seed} if a.base_seed else {}), "rater": a.rater, "habit": habit, "sampled": len(rows), "distinct": len(uniq), "why": why,
           "pairs": len(pairs), "sampled_side": sides(pairs),
           "sampled_side_A_pct": round(100 * sides(pairs)["A"] / max(1, len(pairs)), 2),
           "split": {"rule": f"whole theorems (seeded) into held-out until both sides have >= {(a.heldout + 1) // 2}",
                     "seed": seed, "heldout_theorems": len(held_names), "train_theorems": len({x['theorem'] for x in train_all})},
           "balance": {"train_before": sides(train_all), "train_after": sides(train),
                       "heldout_before": sides(held_all), "heldout_after": sides(held)},
           "train_pairs": len(train), "heldout_pairs": len(held),
           "cap": {"train_pairs_cap": C.TRAIN_PAIR_CAP, "train_balanced": len(train_bal), "train_capped": len(train),
                   "train_below_cap": len(train) < C.TRAIN_PAIR_CAP, "sft_cap": C.SFT_PROOF_CAP,
                   "sft_before_cap": len(train_all), "sft_capped": len(sft_src), "sft_below_cap": len(sft_src) < C.SFT_PROOF_CAP,
                   "rule": "PREREG Amendment 7 point 5: seeded draws; train 2,000 per sampled side after the balancing"},
           "sft": {"pref": len(sft_pref), "sampled": len(sft_sam),
                   "source": "the train pairs before the balancing, capped (a seeded draw), the same pairs in both files"},
           "a_share": comp, "probe_twins": {h: sum(1 for x in probe_rows if x["habit"] == h) for h in sorted(probe_why)},
           "probe_why": probe_why, "probe_per_habit_cap": a.probe_per_habit, "probe_extra": extra_n,
           "twin_errors": [x["err"] for x in res if x["why"] == "twin_error"][:20],
           "lh_profile_sha256": C.sha256(C.HERE / "lh_profile.py"), "lh_twin_sha256": C.sha256(C.HERE / "lh_twin.py"),
           "secs": round(time.time() - t0, 1), "utc": now()}  # fmt: skip
    write_jsonl(d / "train.jsonl", [prow(x) for x in train])
    write_jsonl(d / "heldout.jsonl", [prow(x) for x in held])
    write_jsonl(d / "sft_pref.jsonl", sft_pref)
    write_jsonl(d / "sft_sampled.jsonl", sft_sam)
    write_jsonl(d / "probe_twins.jsonl", probe_rows + extra)
    doc["norm_tokens"] = norm_tokens(
        d / "sft_sampled.jsonl", C.OFFLINE_BATCH
    )  # Amendment 7 point 4
    (d / "summary.json").write_text(json.dumps(doc, indent=1))
    print(f"pairs {a.corpus} {a.rater}: {len(pairs):,} pairs of {len(uniq):,} distinct proofs ({why}); sampled A "
          f"{doc['sampled_side_A_pct']} %; train {len(train_all):,} -> {len(train):,} balanced, held-out {len(held_all)} -> "
          f"{len(held)} ({len(held_names)} theorems); capped train {len(train):,}{' (BELOW the cap)' if len(train) < C.TRAIN_PAIR_CAP else ''}; "
          f"fine-tune files {len(sft_pref):,} (norm tokens {doc['norm_tokens']['N']}); probe twins {doc['probe_twins']} + {extra_n}; "
          f"{doc['secs']:.0f} s", flush=True)  # fmt: skip
    return doc


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    fn = {"anchor": anchor_main, "sample": sample_main, "build": build_main}.get(cmd)
    if fn is None:
        raise SystemExit(__doc__)
    fn(sys.argv[2:])


if __name__ == "__main__":
    main()
