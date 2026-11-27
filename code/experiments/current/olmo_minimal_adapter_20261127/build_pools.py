# ruff: noqa: E501
"""Matched example pools per checkpoint for the minimal-adapter experiment (DESIGN_DRAFT.md, "Example sources").

    build_pools.py report [--max-tok 6144]
    build_pools.py build --ckpt <dpo|s0100|s0500|s1100> [--n-train 64] [--n-dev 32] [--allow-short-dev] [--max-tok 6144]
                         [--pick-seed 20261125] [--split-seed 20261127] [--tokenizer DIR] [--out DIR]

Prompts: Dan's arm-A prompts minus the 48 reach prompts. Checker-only selection (design discourse Turn 2/3): NO hand labels
in any selection. A draw is eligible when it is a strict pass with finish == stop, has `</think>`, n_tok <= cap; the pick per
prompt is uniform (random.Random(f"{pick_seed}:{pid}").choice over draws sorted by (job, chunk, i)). Hand labels are only
recorded per pick (descriptive). A prompt is matched when it has an eligible draw from the checkpoint AND from s2300. Matched prompts are sorted, shuffled with random.Random(f"{split_seed}:split"); train = first n_train (this
order is the nested order: n = 1, 4, 16, 64 are prefixes), dev = next n_dev. Sources over the same prompts: `own`
(the checkpoint's eligible draws), `final` (s2300's eligible draws) and `unsel` (the checkpoint's draws ignoring the checker:
finish == stop, has `</think>`, n_tok <= cap). Per prompt and source the filter's acceptance rate is a_i = (k_i + 0.5)/(N_i + 1)
(N_i the source model's stored draws on the prompt, k_i those passing the source's whole filter); the manifest holds
sum log2(1/a_i) over the train prefixes n = 1, 4, 16, 64 ("checker bound" bits).
Output per source: <out>/<source>/{train,test,reach}.jsonl + MANIFEST.json (test = the dev set, so olmo_edl.load_pool reads it).
Job discovery is the old one generalised to a model name (same exclusions: problems file olmo_ifomega/problems.jsonl, no LoRA,
samples and grades fetched). Row schema is the old pool's.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "olmo_elicitation_20261125"
for _p in (str(HERE), str(OLD)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import build_pool as bp  # noqa: E402
from olmo_template import TEMPLATE_SHA256, prompt_ids  # noqa: E402

DAN = bp.DAN
V2 = Path("/scratch/cowifornia/nd-rl/olmo_elicit/v2")
OLD_POOL = Path("/scratch/cowifornia/nd-rl/olmo_elicit/pool")
TOK = "/scratch/cowifornia/nd-rl/olmo_elicit/tok/s2300"
CKPTS = ("dpo", "s0100", "s0500", "s1100")
FINAL = "s2300"
PICK_SEED = 20261125
SPLIT_SEED = 20261127


def discover_jobs(dan: Path, name: str) -> tuple[list[tuple[str, str, dict]], list[dict]]:
    """bp.discover_jobs for model `name`: plain-prompt jobs [(run dir, job, meta)] and those left out with the reason."""
    keep, skipped = [], []
    for mf in sorted(dan.glob("artifacts/*/meta/*.json")):
        try:
            m = bp.meta_of(mf)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if m.get("name") != name or "job" not in m:
            continue
        run, job = mf.parent.parent.name, m["job"]
        why = None
        if m.get("problems") != bp.PROBLEMS:
            why = f"problems file {m.get('problems')!r}"
        elif m.get("lora"):
            why = "LoRA"
        elif not (dan / "artifacts" / run / "samples" / f"{job}.jsonl.gz").exists():
            why = "samples not fetched"
        elif not (dan / "artifacts" / run / "graded" / f"{job}.jsonl").exists():
            why = "no graded file"
        if why:
            skipped.append({"run": run, "job": job, "why": why})
        else:
            keep.append((run, job, m))
    return keep, skipped


def gather_model(dan: Path, name: str, pids: set[str], labels: dict | None = None):
    jobs, skipped = discover_jobs(dan, name)
    draws, st = bp.load_draws(dan, pids, jobs)
    st.update(bp.apply_labels(draws, labels if labels is not None else bp.load_labels(dan)))
    used = {d.job for d in draws}
    skipped += [
        {"run": r, "job": j, "why": "no draws on the prompts"} for r, j, _m in jobs if j not in used
    ]
    jobs = [x for x in jobs if x[1] in used]
    by_pid: dict[str, list] = defaultdict(list)
    for d in draws:
        by_pid[d.pid].append(d)
    return jobs, skipped, by_pid, st


def passes_checker(d, cap: int) -> bool:
    """Checker-only eligibility: strict and finish == stop, has `</think>`, within the cap. No hand labels."""
    return bp.strict_pass(d) and d.think_end is not None and d.n_tok <= cap


def passes_unsel(d, cap: int) -> bool:
    """Unselected: the checker is ignored; finish == stop, has `</think>`, within the cap."""
    return d.finish == "stop" and d.think_end is not None and d.n_tok <= cap


# source -> (which model's draws, filter)
SOURCES = {
    "own": ("ckpt", passes_checker),
    "final": ("final", passes_checker),
    "unsel": ("ckpt", passes_unsel),
}
BOUND_NS = (1, 4, 16, 64)


def eligible_at(by_pid: dict, pid: str, cap: int, filt=passes_checker) -> list:
    return [d for d in by_pid.get(pid, []) if filt(d, cap)]


def uniform_pick(cands: list, seed: int, pid: str):
    group = sorted(cands, key=lambda d: (d.job, d.chunk, d.i))
    return random.Random(f"{seed}:{pid}").choice(group)


def accept_rate(by_pid: dict, pid: str, cap: int, filt) -> tuple[float, int, int]:
    ds = by_pid.get(pid, [])
    k = sum(filt(d, cap) for d in ds)
    return (k + 0.5) / (len(ds) + 1), k, len(ds)


def split_prompts(matched: list[str], split_seed: int, n_train: int, n_dev: int, allow_short: bool):
    order = sorted(matched)
    random.Random(f"{split_seed}:split").shuffle(order)
    if len(order) < n_train + n_dev and not (allow_short and len(order) >= n_train):
        raise SystemExit(f"only {len(order)} matched prompts; need {n_train} + {n_dev}")
    return order[:n_train], order[n_train : n_train + n_dev]


def prompts_of(dan: Path) -> tuple[list[str], list[str]]:
    arm_a = (dan / "olmo_ifcurve/pids_armA.txt").read_text().split()
    reach = (dan / "olmo_ifcurve/pids_reach.txt").read_text().split()
    rs = set(reach)
    return [p for p in arm_a if p not in rs], reach


def matched_for(dan: Path, ckpt: str, cap: int):
    pool_p, reach = prompts_of(dan)
    pids = set(pool_p) | set(reach)
    labels = bp.load_labels(dan)
    own = gather_model(dan, ckpt, pids, labels)
    fin = gather_model(dan, FINAL, pids, labels)
    matched = [p for p in pool_p if eligible_at(own[2], p, cap) and eligible_at(fin[2], p, cap)]
    return pool_p, reach, own, fin, matched


def report(a) -> None:
    out = {}
    for ck in CKPTS:
        pool_p, _r, own, fin, matched = matched_for(a.dan, ck, a.max_tok)
        nown = sum(bool(eligible_at(own[2], p, a.max_tok)) for p in pool_p)
        nfin = sum(bool(eligible_at(fin[2], p, a.max_tok)) for p in pool_p)
        lo = [min(d.n_tok for d in eligible_at(own[2], p, a.max_tok)) for p in matched]
        nuns = sum(bool(eligible_at(own[2], p, a.max_tok, passes_unsel)) for p in matched)
        out[ck] = {
            "matched_with_unsel_draw": nuns,
            "pool_prompts": len(pool_p),
            "own_eligible": nown,
            "final_eligible": nfin,
            "matched": len(matched),
            "own_jobs": [j for _, j, _ in own[0]],
        }
        print(
            f"{ck}: arm-A non-reach {len(pool_p)}; checker-eligible own {nown}, s2300 {nfin}, matched {len(matched)} (matched with an unselected draw: {nuns}; own shortest-draw median {int(np.median(lo)) if lo else '-'}); jobs {out[ck]['own_jobs']}"
        )
    if a.json:
        a.json.write_text(json.dumps(out, indent=1) + "\n")


def row_of(d, pr: dict, k: int, name: str, tok, meta: dict, stats: Counter, acc=None) -> dict:
    iid = pr["instruction_id_list"]
    iid = list(json.loads(iid.replace("'", '"')) if isinstance(iid, str) else iid)
    ids = bp.decode_ids(d.ids_b64)
    assert ids[-1] == bp.EOS and d.think_end is not None and d.mark_pos is not None
    row = {
        "set": name,
        "idx": k,
        "pid": d.pid,
        "task": pr["task"],
        "prompt": pr["prompt"],
        "ids": d.ids_b64,
        "ids_sha256": hashlib.sha256(base64.b64decode(d.ids_b64)).hexdigest(),
        "n_tok": d.n_tok,
        "think_end": d.think_end,
        "n_think": d.think_end,
        "n_answer": d.n_tok - d.think_end,
        "think_marker_pos": d.mark_pos,
        "think_marker": ids[d.mark_pos : d.think_end],
        "n_think_markers": d.n_marks,
        "job": d.job,
        "run": d.run,
        "chunk": d.chunk,
        "i": d.i,
        "finish": d.finish,
        "strict": d.strict,
        "label": d.label or "unlabelled",  # hand label, descriptive only (never used in selection)
        "label_source": d.label_source,
        "instruction_id_list": iid,
        "text_sha256": hashlib.sha256(d.text.encode()).hexdigest(),
        "prompt_sha256_dan": (meta.get(d.job, {}).get("prompt_sha256_by_pid") or {}).get(d.pid),
    }
    if acc is not None:
        row["accept_rate"], row["accept_k"], row["accept_N"] = acc
    if tok is not None:
        pids_ = prompt_ids(tok, pr["prompt"])
        arr = np.asarray(pids_, dtype=np.int32)
        sha = hashlib.sha256(arr.tobytes()).hexdigest()
        if row["prompt_sha256_dan"] is not None:
            assert sha == row["prompt_sha256_dan"], (d.pid, "prompt ids differ from the sampler's")
            stats["prompt_hash_checked"] += 1
        row["prompt_ids"] = base64.b64encode(arr.tobytes()).decode()
        row["prompt_ids_sha256"] = sha
        row["n_prompt"] = len(pids_)
    return row


def write_jsonl(path: Path, rows: list[dict]) -> dict:
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as g:
        for r in rows:
            g.write(json.dumps(r) + "\n")
    tmp.replace(path)
    return {"sha256": bp.sha256_file(path), "rows": len(rows)}


def build(
    dan: Path,
    ckpt: str,
    out: Path,
    *,
    cap: int,
    n_train: int,
    n_dev: int,
    allow_short_dev: bool,
    pick_seed: int,
    split_seed: int,
    tokenizer: str | None,
    sources: tuple[str, ...] = ("own", "final", "unsel"),
    old_pool: Path | None = None,
    reach_src: Path | None = None,
) -> dict:
    pool_p, reach, own, fin, matched = matched_for(dan, ckpt, cap)
    train_p, dev_p = split_prompts(matched, split_seed, n_train, n_dev, allow_short_dev)
    models = {"ckpt": own, "final": fin}
    probs = {r["pid"]: r for r in map(json.loads, (dan / bp.PROBLEMS).read_text().splitlines())}
    tok = None
    if tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tokenizer)
    try:
        commit = subprocess.run(
            ["git", "-C", str(dan), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        commit = None
    # descriptive: how many of the old pool's picks the new (label-free, uniform) final-source pick still equals
    repro = None
    if old_pool is not None and (old_pool / "train.jsonl").exists():
        repro = {}
        for nm in ("train", "test"):
            old = [json.loads(x) for x in (old_pool / f"{nm}.jsonl").read_text().splitlines()]
            same = ov = 0
            for r in old:
                if r["pid"] in matched:
                    ov += 1
                    c = eligible_at(fin[2], r["pid"], cap)
                    p = uniform_pick(c, pick_seed, r["pid"])
                    same += p.key == (r["job"], r["pid"], r["chunk"], r["i"])
            repro[nm] = {
                "old_rows": len(old),
                "old_prompts_in_matched": ov,
                "pick_reproduced": same,
            }
    result = {}
    for source in sources:
        which, filt = SOURCES[source]
        jobs, skipped, by_pid, st = models[which]
        d_out = out / source
        d_out.mkdir(parents=True, exist_ok=True)
        pick = {
            p: uniform_pick(eligible_at(by_pid, p, cap, filt), pick_seed, p)
            for p in train_p + dev_p
        }
        acc = {p: accept_rate(by_pid, p, cap, filt) for p in train_p + dev_p}
        meta = {j: m for _, j, m in jobs}
        stats: Counter = Counter()
        files = {}
        sets = {"train": train_p, "test": dev_p}
        for nm, ps in sets.items():
            rows = [
                row_of(pick[p], probs[p], k, nm, tok, meta, stats, acc[p]) for k, p in enumerate(ps)
            ]
            files[f"{nm}.jsonl"] = write_jsonl(d_out / f"{nm}.jsonl", rows)
        if reach_src is not None and reach_src.exists():
            shutil.copyfile(reach_src, d_out / "reach.jsonl")
            files["reach.jsonl"] = {
                "sha256": bp.sha256_file(d_out / "reach.jsonl"),
                "rows": len((d_out / "reach.jsonl").read_text().splitlines()),
                "copied_from": str(reach_src),
            }

        def tstats(ps: list[str], pick=pick, acc=acc) -> dict:
            return {
                "n_tok": bp.pctl([pick[p].n_tok for p in ps]),
                "n_think": bp.pctl([pick[p].think_end for p in ps]),
                "n_answer": bp.pctl([pick[p].n_tok - pick[p].think_end for p in ps]),
                "label_counts": dict(Counter(pick[p].label or "unlabelled" for p in ps)),
                "strict_picks": sum(bp.strict_pass(pick[p]) for p in ps),
                "mean_accept_rate": float(np.mean([acc[p][0] for p in ps])),
                "total_response_tokens": sum(pick[p].n_tok for p in ps),
            }

        man = {
            "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "builder": "build_pools.py (olmo_minimal_adapter_20261127)",
            "ckpt": ckpt,
            "source": source,
            "source_model": FINAL if which == "final" else ckpt,
            "filter": {
                "own": "strict, finish == stop, </think>, n_tok <= cap",
                "final": "same, on s2300's draws",
                "unsel": "finish == stop, </think>, n_tok <= cap (checker ignored)",
            }[source],
            "checker_bound_bits": {
                str(n): sum(math.log2(1.0 / acc[p][0]) for p in train_p[:n])
                for n in BOUND_NS
                if n <= len(train_p)
            },
            "checker_bound_note": "sum over the first n train prompts of log2(1/a_i), a_i = (k_i + 0.5)/(N_i + 1) over the source model's stored draws (N_i) passing the source's filter (k_i)",
            "dan_clone": str(dan),
            "dan_commit": commit,
            "pick_seed": pick_seed,
            "split_seed": split_seed,
            "max_tok": cap,
            "n_train": len(train_p),
            "n_dev": len(dev_p),
            "dev_short": len(dev_p) < n_dev,
            "rules": "no hand labels in selection; eligible = strict and finish == stop, has </think>, n_tok <= cap (unsel: strict not required); uniform seeded pick over draws sorted by (job, chunk, i); hand label of each pick recorded only",
            "matching": "prompts (arm A minus reach) with a checker-eligible draw from both the checkpoint and s2300; sorted, shuffled with Random(f'{split_seed}:split'); train = prefix (nested order), dev = next block",
            "n_pool_prompts": len(pool_p),
            "n_matched": len(matched),
            "matched_prompts": sorted(matched),
            "train_pids": train_p,
            "dev_pids": dev_p,
            "jobs": [
                {
                    "run": r,
                    "job": j,
                    "seed": m.get("seed"),
                    "rev": m.get("rev"),
                    "problems": m.get("problems"),
                    "template_sha256": m.get("template_sha256"),
                    "vllm": m.get("vllm"),
                }
                for r, j, m in jobs
            ],
            "skipped_jobs": skipped,
            "draw_stats": dict(st),
            "label_sources": [s for s, _ in bp.LABEL_SOURCES],
            "template_sha256": TEMPLATE_SHA256,
            "tokenizer": tokenizer,
            "prompt_hash_checked_against_sampler": stats["prompt_hash_checked"],
            "sets": {"train": tstats(train_p), "test": tstats(dev_p)},
            "old_pool_reproduction": repro if source == "final" else None,
            "files": files,
        }
        tmp = d_out / "MANIFEST.json.tmp"
        tmp.write_text(json.dumps(man, indent=1) + "\n")
        tmp.replace(d_out / "MANIFEST.json")
        result[source] = man
    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cmd", choices=["report", "build"])
    ap.add_argument("--ckpt", choices=CKPTS)
    ap.add_argument("--dan", type=Path, default=DAN)
    ap.add_argument("--max-tok", type=int, default=6144)
    ap.add_argument("--n-train", type=int, default=64)
    ap.add_argument("--n-dev", type=int, default=32)
    ap.add_argument("--allow-short-dev", action="store_true")
    ap.add_argument("--pick-seed", type=int, default=PICK_SEED)
    ap.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    ap.add_argument("--sources", default="own,final,unsel")
    ap.add_argument("--tokenizer", default=TOK)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    if a.cmd == "report":
        return report(a)
    if not a.ckpt:
        ap.error("build needs --ckpt")
    out = a.out or V2 / "pools" / a.ckpt
    res = build(
        a.dan,
        a.ckpt,
        out,
        cap=a.max_tok,
        n_train=a.n_train,
        n_dev=a.n_dev,
        allow_short_dev=a.allow_short_dev,
        pick_seed=a.pick_seed,
        split_seed=a.split_seed,
        tokenizer=a.tokenizer,
        sources=tuple(a.sources.split(",")),
        old_pool=OLD_POOL,
        reach_src=OLD_POOL / "reach.jsonl",
    )
    for s, m in res.items():
        print(
            f"{a.ckpt}/{s}: matched {m['n_matched']} train {m['n_train']} dev {m['n_dev']} (short={m['dev_short']}) n_tok median train {m['sets']['train']['n_tok'].get('p50')} dev {m['sets']['test']['n_tok'].get('p50')}; labels {m['sets']['train']['label_counts']}; prompt hashes checked {m['prompt_hash_checked_against_sampler']}"
        )
    for s_, m in res.items():
        print(
            f"{a.ckpt}/{s_}: checker bound bits {json.dumps({k: round(v, 1) for k, v in m['checker_bound_bits'].items()})}; strict picks train {m['sets']['train']['strict_picks']}"
        )
    if "final" in res:
        print(
            "old-pool reproduction (descriptive):",
            json.dumps(res["final"]["old_pool_reproduction"]),
        )


if __name__ == "__main__":
    main()
