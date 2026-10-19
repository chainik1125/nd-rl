"""Decontaminate the generated candidates and attach shape features.

    python -m code.experiments.current.thinking_traces_20261005.pools.prep \
        --cands /scratch/cowifornia/nd-rl/think/runs/pools/raw/cands.jsonl \
        --out /scratch/cowifornia/nd-rl/think/runs/pools/cands_clean.jsonl

A candidate is dropped if its renaming class (`data.class_keys`: the fork's `canon_key`, or the
premise-order-invariant `pkey`) is in any exclusion file (`exclusion_files()`):
  * this project's eval sets: `data.DEFAULT_EXCLUDE` (the five OOD sets, textbook_dev, the K12
    held-out set) + textbook72 train;
  * all training data: K12 train (every trace file's prompts are K12 prompts, checked);
  * the RL pools: the ladder's `rl_targets.jsonl` and the take-home `data/rl_targets.jsonl`;
  * every other jsonl under the fork's `data/` (the ladder's and take-home's `transfer.jsonl`,
    `transfer_long*`, the p2 / r1-r3 / bs / gt / lj pools and eval sets): Dan's long-pool rule.
Candidates are also deduplicated among themselves by `pkey`.  Kept rows gain `name` (`rh_<i>`),
`pkey`, `prompt_tokens`, `formula_depth` (deepest parenthesis nesting of the prompt), `max_state` /
`max_action` (tokens over the reference proof's state-env decomposition, `lean_thinkn` vocabulary)
and `ref_steps`.  A report goes to <out>.report.json.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
import time
from pathlib import Path
from typing import Any

from code.ndbench.core.keys import canon_key, pkey, prompt_to_thm, split_thm
from code.thinking.dan import load
from code.thinking.data import DEFAULT_EXCLUDE, FORK, MAX_PREM_FOR_PKEY, read_jsonl
from code.thinking.tok import ThinkTokenizer

D = load()
K12 = "/scratch/cowifornia/nd-rl/bench/dan_ckpts/data/train_k12.jsonl"
TEXTBOOK_TRAIN = f"{FORK}/data/eval_only/textbook72/textbook_train.jsonl"
TRACES = sorted(glob.glob("/scratch/cowifornia/nd-rl/think/traces/*_traces*.jsonl"))


def exclusion_files() -> list[str]:
    fork = sorted(glob.glob(f"{FORK}/data/**/*.jsonl", recursive=True))
    out = [*DEFAULT_EXCLUDE, TEXTBOOK_TRAIN, K12, *fork]
    return list(dict.fromkeys(out))


def thm_prompt(r: dict[str, Any]) -> str | None:
    if "prompt" in r:
        return str(r["prompt"])
    if "thm" in r:
        prem, _, concl = str(r["thm"]).partition("|-")
        prem = prem.strip()
        return f"THM {prem} SEQ {concl.strip()} PRF" if prem else f"THM SEQ {concl.strip()} PRF"
    return None


def keys_of(prompt: str) -> tuple[str, str | None]:
    thm = prompt_to_thm(prompt)
    prem, _ = split_thm(thm)
    return canon_key(thm), (pkey(thm) if len(prem) <= MAX_PREM_FOR_PKEY else None)


def index(files: list[str]) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, Any]]:
    """class key -> files, pkey -> files."""
    by_k: dict[str, set[str]] = collections.defaultdict(set)
    by_pk: dict[str, set[str]] = collections.defaultdict(set)
    rep: dict[str, Any] = {}
    for f in files:
        n = bad = 0
        with open(f) as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                p = thm_prompt(r)
                if p is None:
                    bad += 1
                    continue
                try:
                    k, pk = keys_of(p)
                except Exception:  # a malformed row in someone's pool: counted, not fatal
                    bad += 1
                    continue
                by_k[k].add(f)
                if pk is not None:
                    by_pk[pk].add(f)
                n += 1
        rep[f] = {"rows": n, "unkeyed": bad}
    return by_k, by_pk, rep


def depth(prompt: str) -> int:
    d = m = 0
    for t in prompt.split():
        if t == "(":
            d += 1
            m = max(m, d)
        elif t == ")":
            d -= 1
    return m


def shape(r: dict[str, Any], tok: Any) -> dict[str, Any]:
    steps, _, _ = D.state_env.decompose(r["prompt"], r["proof"], canon=True)
    return {
        "max_state": max(len(tok.encode_toks(s)) for s, _, _ in steps),
        "max_action": max(len(tok.encode_toks(a)) + 1 for _, a, _ in steps),
        "ref_steps": len(steps),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cands", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    files = exclusion_files()
    by_k, by_pk, files_rep = index(files)
    k12 = {r["prompt"] for r in read_jsonl(K12)}
    traces_in_k12 = {}
    for f in TRACES:
        ps = [r["prompt"] for r in read_jsonl(f)]
        traces_in_k12[f] = {"n": len(ps), "in_k12": sum(p in k12 for p in ps)}
    print(
        f"index: {len(by_k)} classes from {len(files)} files, {time.time() - t0:.0f}s", flush=True
    )
    tok = ThinkTokenizer()
    hits: collections.Counter[str] = collections.Counter()
    st: collections.Counter[str] = collections.Counter()
    seen_pk: set[str] = set()
    n_out = 0
    with open(a.cands) as fi, open(a.out, "w") as fo:
        for line in fi:
            r = json.loads(line)
            st["in"] += 1
            k, pk = keys_of(r["prompt"])
            hit = by_k.get(k, set()) | (by_pk.get(pk, set()) if pk else set())
            if hit:
                st["excluded"] += 1
                for f in hit:
                    hits[f] += 1
                continue
            if pk is not None and pk in seen_pk:
                st["dup_pkey"] += 1
                continue
            if pk is not None:
                seen_pk.add(pk)
            try:
                sh = shape(r, tok)
            except Exception as e:  # noqa: BLE001 -- counted by type
                st[f"decompose_failed:{type(e).__name__}"] += 1
                continue
            row = {
                "name": f"rh_{n_out}",
                **r,
                "key": k,
                "pkey": pk,
                "prompt_tokens": len(r["prompt"].split()),
                "formula_depth": depth(r["prompt"]),
                **sh,
            }
            fo.write(json.dumps(row) + "\n")
            n_out += 1
            st["out"] += 1
    rep = {
        "cands": a.cands,
        "stats": dict(st),
        "excluded_by_file": dict(hits.most_common()),
        "exclusion_files": files_rep,
        "traces_in_k12": traces_in_k12,
        "secs": time.time() - t0,
    }
    Path(a.out + ".report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({"stats": dict(st), "top_hits": dict(hits.most_common(8))}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
