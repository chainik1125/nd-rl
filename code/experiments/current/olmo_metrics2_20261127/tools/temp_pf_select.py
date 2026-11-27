"""Build the PF manifest for hf_score.py (CPU, serrano; needs the tokenizer files, no weights).

Per reach prompt (48): up to --s2300-n (4) STORED s2300 draws that pass Dan's strict checker (first passing draws in draw
order of the 32 stored B_s2300 draws), --dpo-n (8; fallback 4) STORED dpo T 1.0 draws (first of the 16, draw order,
olmo_ifomega F_dpo_ifb + F_dpo_ife), and --c1-n (4) draws of the NEW C1 cell (first in draw order, pass or fail; a
descriptive scale, enters no gate). Prompt ids are rebuilt with Dan's own function and template (his sample.py is imported,
never edited) and their sha256 is compared with the prompt_sha256_by_pid of Dan's stored meta (olmo_ifcurve B_s2300, the 48 reach prompts) -- any mismatch
aborts. Response ids come from the stored base64 int32 `ids` (response only, ending in EOS 100257 when finish == stop).
`split_at` = number of response tokens up to and including the `</think>` text (smallest prefix whose decode, with
skip_special_tokens=True as in Dan's `text`, contains it); -1 when there is none.
Writes <out>.jsonl (manifest) and <out>.summary.json (counts per group, prompts dropped, checks).
Usage: temp_pf_select.py --tok-dir DIR --out PREFIX [--c1-samples S.jsonl.gz] [--dpo-n 8] [--s2300-n 4] [--c1-n 4]
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import temp_common as tc

ART = tc.ART


def decode_ids(b64: str) -> list[int]:
    return np.frombuffer(base64.b64decode(b64), dtype=np.int32).tolist()


def find_split(tok, ids: list[int], needle: str = "</think>") -> int:
    if needle not in tok.decode(ids, skip_special_tokens=True):
        return -1
    lo, hi = 0, len(ids)
    while lo < hi:
        mid = (lo + hi) // 2
        if needle in tok.decode(ids[:mid], skip_special_tokens=True):
            hi = mid
        else:
            lo = mid + 1
    return lo


def load_dan(dan_root: Path):
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(
        "dan_sample_pf", dan_root / "olmo_ifladder" / "sample.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tok-dir", required=True)
    ap.add_argument(
        "--out", required=True, help="prefix; writes PREFIX.jsonl and PREFIX.summary.json"
    )
    ap.add_argument("--dan-root", default=str(tc.DAN))
    ap.add_argument("--c1-samples")
    ap.add_argument("--dpo-n", type=int, default=8)
    ap.add_argument("--s2300-n", type=int, default=4)
    ap.add_argument("--c1-n", type=int, default=4)
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer

    dan = load_dan(Path(a.dan_root))
    tok = AutoTokenizer.from_pretrained(a.tok_dir)
    probs = dan.load_problems(Path(a.dan_root) / "olmo_ifomega" / "problems.jsonl")
    meta = json.load(open(ART / "olmo_ifcurve/meta/B_s2300.json"))
    reach = tc.dc.reach()
    # stored s2300: strict pass flags from Dan's rows, ids from the samples
    R = tc.dc.Rows(ART / "olmo_ifcurve" / "rows")
    s_samples = tc.by_pid(
        [r for r in tc.jl(ART / "olmo_ifcurve/samples/B_s2300.jsonl.gz") if r["pid"] in set(reach)]
    )
    d_samples: dict[str, list[dict]] = {}
    for j in ("F_dpo_ifb", "F_dpo_ife"):
        for r in tc.order(tc.jl(ART / f"olmo_ifomega/samples/{j}.jsonl.gz")):
            if r["pid"] in set(reach):
                d_samples.setdefault(r["pid"], []).append(r)
    c1_samples = (
        tc.by_pid([r for r in tc.jl(a.c1_samples) if r["pid"] in set(reach)])
        if a.c1_samples
        else {}
    )
    rows, summ = (
        [],
        {"dropped_prompts": [], "counts": {}, "prompt_sha_check": "ok", "template_sha256": None},
    )
    summ["template_sha256"] = hashlib.sha256(dan.TEMPLATE.encode()).hexdigest()
    if summ["template_sha256"] != meta["template_sha256"]:
        raise SystemExit("template sha differs from Dan's stored meta")
    for p in reach:
        pr = probs[p]
        pid_ids = dan.prompt_ids(
            tok, pr.get("messages_sample") or pr.get("prompt_sample", pr["prompt"])
        )
        sha = hashlib.sha256(np.asarray(pid_ids, dtype=np.int32).tobytes()).hexdigest()
        if sha != meta["prompt_sha256_by_pid"][p]:
            raise SystemExit(f"prompt sha mismatch for {p}")
        passing = {(r["chunk"], r["i"]) for r in R.draws("s2300", p, "B") if r["strict"]}
        sel = []
        sel += [
            ("s2300", r) for r in s_samples.get(p, []) if (int(r["chunk"]), int(r["i"])) in passing
        ][: a.s2300_n]
        sel += [("dpo", r) for r in d_samples.get(p, [])[: a.dpo_n]]
        sel += [("c1", r) for r in c1_samples.get(p, [])[: a.c1_n]]
        if not any(g == "s2300" for g, _ in sel):
            summ["dropped_prompts"].append(p)
            continue
        for g, r in sel:
            ids = decode_ids(r["ids"])
            assert len(ids) == int(r["n_tok"]), (p, g, len(ids), r["n_tok"])
            rows.append(
                {
                    "id": f"{g}|{p}|{r['job']}|{r['chunk']}|{r['i']}",
                    "pid": p,
                    "group": g,
                    "job": r["job"],
                    "chunk": int(r["chunk"]),
                    "i": int(r["i"]),
                    "prompt_ids": pid_ids,
                    "response_ids": ids,
                    "split_at": find_split(tok, ids),
                    "finish": r.get("finish"),
                }
            )
    for g in ("s2300", "dpo", "c1"):
        summ["counts"][g] = sum(1 for r in rows if r["group"] == g)
    summ["n_sequences"] = len(rows)
    summ["n_prompts"] = len({r["pid"] for r in rows})
    summ["params"] = {"dpo_n": a.dpo_n, "s2300_n": a.s2300_n, "c1_n": a.c1_n}
    Path(a.out + ".jsonl").parent.mkdir(parents=True, exist_ok=True)
    with open(a.out + ".jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    json.dump(summ, open(a.out + ".summary.json", "w"), indent=1)
    print(json.dumps(summ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
