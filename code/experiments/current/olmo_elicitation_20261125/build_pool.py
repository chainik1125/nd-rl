# ruff: noqa: E501
"""Stage 1 data pool: one stored strict-pass response of the final model (s2300) per prompt.

    build_pool.py report [--caps 2048,3072,4096,6144,8192]   length distribution and survivors per cap (stdout + JSON)
    build_pool.py build  [--max-tok 6144] [--tokenizer DIR]   -> <out>/{train,test,reach}.jsonl + MANIFEST.json

Source draws (SPEC.md "Stage 1", data). Every stored sampling job of `allenai/Olmo-3.1-32B-Think` revision `main`
(= `step_2300` by weight-shard hashes, Dan's `olmo_ifcurve/checkpoints.csv`) whose meta says it sampled the plain
prompt: model name `s2300`, problems file `olmo_ifomega/problems.jsonl` (its `prompt` field is the user turn; that
file has no `prompt_sample`), no LoRA. Hinted (olmo-iffu), rung (olmo-ifladder) and transplant jobs sent other
prompts and are left out. Jobs are discovered from the meta files and listed in the manifest.

Rules, in order, per prompt (arm-A prompts and the 48 reach prompts only):
  1. a draw is a strict pass if Dan's grade says `strict` and `finish == "stop"` (no length cut);
  2. it must contain the end-of-thinking marker `</` `think` + a `>`-token (ids 524, 27963, then 1363 `>\\n\\n`,
     397 `>\\n` or 29 `>`); the last marker counts, as in Dan's grader (`text.rsplit("</think>", 1)`); passes without
     one are excluded and counted;
  3. hand labels (genuine / gamed) from every gate Dan ran on these stored draws (LABEL_SOURCES); a draw with any
     gamed label is gamed (two draws carry a later mechanical rule-1 "gamed" over an earlier "genuine"); a draw with
     no label of its own inherits the label of an identical whitespace-collapsed answer on the same prompt among the
     labelled s2300 draws (Dan's identity inheritance, `olmo_ifcurve/hand.py`), unless those labels disagree;
  4. gamed passes are excluded; a prompt whose strict passes are all gamed is dropped ("only gamed");
  5. passes longer than the response-token cap (`--max-tok`, on stored response ids, thinking + answer + EOS) are
     excluded; a prompt left with none is dropped ("over cap");
  6. one response per prompt: among the remaining passes a genuine-labelled one if any, else any (label
     "strict-only"); the pick within that group is `random.Random(f"{seed}:{pid}").choice` over the draws sorted by
     (job, chunk, i);
  7. sets: `train` (n) and `test` (m) from the arm-A prompts that are not reach prompts, by a seeded shuffle
     (stratified by benchmark when both occur; arm A is IFBench only), train first; `reach` = the reach prompts
     whose pick is genuine (SPEC: "the 48 reach prompts that have a genuine s2300 response"; `--reach-any` keeps
     strict-only picks too).
Response ids are stored exactly as Dan stores them (base64 int32, thinking + `</think>` + answer + EOS). With
`--tokenizer`, the prompt ids are also built here with Dan's encoding (olmo_thinker template, add_generation_prompt,
pad ids dropped), stored, and checked against the sampler's own `prompt_sha256_by_pid` where a job meta recorded it.
"""

from __future__ import annotations

import argparse
import base64
import csv
import gzip
import hashlib
import json
import random
import re
import subprocess
import sys
import time
from array import array
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from olmo_template import TEMPLATE_SHA256, prompt_ids  # noqa: E402

DAN = Path("/scratch/cowifornia/nd-rl/olmo_elicit/ext/nd-takehome")
OUT = Path("/scratch/cowifornia/nd-rl/olmo_elicit/pool")
SEED = 20261125
CAPS = (2048, 3072, 4096, 6144, 8192)
MARK = (524, 27963)  # "</" "think"
MARK_CLOSE = {1363: ">\n\n", 397: ">\n", 29: ">"}  # the token after "</" "think"
EOS = 100257
PROBLEMS = "olmo_ifomega/problems.jsonl"
# (path relative to Dan's clone, how it keys a draw); final_label in {genuine, gamed} counts, other labels are OMEGA's
LABEL_SOURCES = (
    ("olmo_ifcurve/handcheck.csv", "cols"),
    ("olmo_ifomega/handcheck.csv", "cols"),
    ("olmo_ifladder/handcheck_src.csv", "cols"),
    ("olmo_transplant/handcheck_src.csv", "key"),
    ("artifacts/olmo_ifcurve/hand/screen.csv", "screen"),
)


@dataclass
class Draw:
    job: str
    pid: str
    chunk: int
    i: int
    n_tok: int
    finish: str
    strict: bool
    task: str
    ids_b64: str
    text: str
    run: str
    mark_pos: int | None = None  # index of "</" of the last marker
    think_end: int | None = None  # first answer token index (after the ">" token)
    n_marks: int = 0
    labels: list[tuple[str, str]] = field(default_factory=list)  # (label, source)
    label: str | None = None
    label_source: str = ""

    @property
    def key(self) -> tuple[str, str, int, int]:
        return (self.job, self.pid, self.chunk, self.i)


def answer_of(text: str) -> str | None:  # olmo_ifomega/grade.py answer_of, verbatim
    return text.rsplit("</think>", 1)[1].strip() if "</think>" in text else None


def ans_key(text: str) -> str | None:  # olmo_ifcurve/cells.py ans_key of the answer
    a = answer_of(text)
    if a is None:
        return None
    return hashlib.sha1(re.sub(r"\s+", " ", a).strip().encode()).hexdigest()[:16]


def decode_ids(b64: str) -> list[int]:
    a = array("i")
    a.frombytes(base64.b64decode(b64))
    return a.tolist()


def find_marker(ids: Sequence[int]) -> tuple[int | None, int | None, int]:
    """(position of "</" of the last `</think>` marker, first answer index, number of markers)."""
    hits = [
        k
        for k in range(len(ids) - 2)
        if ids[k] == MARK[0] and ids[k + 1] == MARK[1] and ids[k + 2] in MARK_CLOSE
    ]
    if not hits:
        return None, None, 0
    k = hits[-1]
    return k, k + 3, len(hits)


def read_jsonl_gz(path: Path) -> Iterable[dict]:
    with gzip.open(path, "rt") as f:
        for ln in f:
            try:
                yield json.loads(ln)
            except json.JSONDecodeError:
                break  # truncated tail of a killed job (Dan's read_rows)


def meta_of(path: Path) -> dict:
    """A job meta file: one JSON object per line (resumed jobs append); merged, later lines win, per-pid dicts merged."""
    out: dict = {}
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        m = json.loads(ln)
        for k, v in m.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = {**out[k], **v}
            else:
                out[k] = v
    return out


def discover_jobs(dan: Path) -> tuple[list[tuple[str, str, dict]], list[dict]]:
    """Plain-prompt s2300 jobs: [(run dir, job, meta)], and the s2300 jobs left out with the reason."""
    keep, skipped = [], []
    for mf in sorted(dan.glob("artifacts/*/meta/*.json")):
        try:
            m = meta_of(mf)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if m.get("name") != "s2300" or "job" not in m:
            continue
        run, job = mf.parent.parent.name, m["job"]
        why = None
        if m.get("problems") != PROBLEMS:
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


def load_labels(dan: Path) -> dict[tuple[str, str, int, int], list[tuple[str, str]]]:
    lab: dict[tuple[str, str, int, int], list[tuple[str, str]]] = defaultdict(list)
    for rel, kind in LABEL_SOURCES:
        for r in csv.DictReader((dan / rel).read_text().splitlines()):
            if kind == "key":
                j, p, c, i = r["key"].split("|")
            else:
                j, p, c, i = r["job"], r["pid"], r["chunk"], r["i"]
            lbl = "gamed" if kind == "screen" else r["final_label"]
            if lbl in ("genuine", "gamed"):
                lab[(j, p, int(c), int(i))].append((lbl, rel))
    return lab


def load_draws(
    dan: Path, pids: set[str], jobs: Sequence[tuple[str, str, dict]]
) -> tuple[list[Draw], Counter]:
    stats: Counter = Counter()
    draws = []
    for run, job, _m in jobs:
        grades = {}
        for ln in (dan / "artifacts" / run / "graded" / f"{job}.jsonl").read_text().splitlines():
            g = json.loads(ln)
            if g["pid"] in pids:
                grades[(g["pid"], int(g["chunk"]), int(g["i"]))] = g
        for r in read_jsonl_gz(dan / "artifacts" / run / "samples" / f"{job}.jsonl.gz"):
            if r["pid"] not in pids:
                continue
            g = grades.get((r["pid"], int(r["chunk"]), int(r["i"])))
            if g is None:
                stats["sample_without_grade"] += 1
                continue
            d = Draw(
                job=job,
                pid=r["pid"],
                chunk=int(r["chunk"]),
                i=int(r["i"]),
                n_tok=int(r["n_tok"]),
                finish=r["finish"],
                strict=bool(g["strict"]),
                task=g.get("task", ""),
                ids_b64=r["ids"],
                text=r["text"],
                run=run,
            )
            ids = decode_ids(r["ids"])
            assert len(ids) == d.n_tok, (d.key, len(ids), d.n_tok)
            d.mark_pos, d.think_end, d.n_marks = find_marker(ids)
            draws.append(d)
            stats[f"draws:{job}"] += 1
    return draws, stats


def apply_labels(draws: list[Draw], labels: dict) -> Counter:
    st: Counter = Counter()
    for d in draws:
        d.labels = labels.get(d.key, [])
        kinds = {lbl for lbl, _ in d.labels}
        if kinds:
            d.label = "gamed" if "gamed" in kinds else "genuine"
            d.label_source = ";".join(sorted({src for _, src in d.labels}))
            if len(kinds) > 1:
                d.label_source = "conflict->gamed:" + d.label_source
                st["label_conflicts"] += 1
    inh: dict[tuple[str, str], set[str]] = defaultdict(set)
    for d in draws:
        if d.label is not None and (k := ans_key(d.text)):
            inh[(d.pid, k)].add(d.label)
    for d in draws:
        if d.label is None and (k := ans_key(d.text)) and (s := inh.get((d.pid, k))):
            if len(s) == 1:
                d.label = next(iter(s))
                d.label_source = "inherited (identical answer)"
                st[f"inherited_{d.label}"] += 1
            else:
                st["inherit_conflict_unlabelled"] += 1
    return st


def strict_pass(d: Draw) -> bool:
    return d.strict and d.finish == "stop"


def eligible(d: Draw) -> bool:
    """Rules 1, 2 and 4: a strict pass with a thinking end that is not labelled gamed (no cap)."""
    return strict_pass(d) and d.think_end is not None and d.label != "gamed"


def choose(cands: list[Draw], seed: int, pid: str) -> Draw:
    gen = [d for d in cands if d.label == "genuine"]
    group = sorted(gen or cands, key=lambda d: (d.job, d.chunk, d.i))
    return random.Random(f"{seed}:{pid}").choice(group)


def prompt_status(ds: list[Draw], cap: int) -> str:
    sp = [d for d in ds if strict_pass(d)]
    if not sp:
        return "no strict pass"
    wt = [d for d in sp if d.think_end is not None]
    if not wt:
        return "no </think> in any strict pass"
    ng = [d for d in wt if d.label != "gamed"]
    if not ng:
        return "only gamed"
    if not [d for d in ng if d.n_tok <= cap]:
        return "over cap"
    return "ok"


def stratified_order(pids: list[str], task: dict[str, str], seed: int) -> list[str]:
    """Seeded shuffle within each benchmark, then interleaved in proportion (train = prefix, test = next block)."""
    by: dict[str, list[str]] = defaultdict(list)
    for p in sorted(pids):
        by[task[p]].append(p)
    keyed = []
    for t, ps in sorted(by.items()):
        random.Random(f"{seed}:split:{t}").shuffle(ps)
        keyed += [((k + 0.5) / len(ps), t, p) for k, p in enumerate(ps)]
    return [p for *_, p in sorted(keyed)]


def pctl(xs: Sequence[float]) -> dict:
    if not xs:
        return {"n": 0}
    a = np.asarray(xs, dtype=float)
    q = np.percentile(a, [10, 25, 50, 75, 90, 95, 99])
    return {
        "n": len(xs),
        "min": int(a.min()),
        **{f"p{k}": int(v) for k, v in zip((10, 25, 50, 75, 90, 95, 99), q, strict=True)},
        "max": int(a.max()),
        "mean": round(float(a.mean()), 1),
    }


def gather(dan: Path):
    arm_a = [x for x in (dan / "olmo_ifcurve/pids_armA.txt").read_text().split()]
    reach = [x for x in (dan / "olmo_ifcurve/pids_reach.txt").read_text().split()]
    probs = {r["pid"]: r for r in map(json.loads, (dan / PROBLEMS).read_text().splitlines())}
    jobs, skipped = discover_jobs(dan)
    draws, st = load_draws(dan, set(arm_a) | set(reach), jobs)
    st.update(apply_labels(draws, load_labels(dan)))
    used = {d.job for d in draws}
    skipped += [
        {"run": r, "job": j, "why": "no draws on the arm-A / reach prompts"}
        for r, j, _m in jobs
        if j not in used
    ]
    jobs = [x for x in jobs if x[1] in used]
    by_pid: dict[str, list[Draw]] = defaultdict(list)
    for d in draws:
        by_pid[d.pid].append(d)
    return arm_a, reach, probs, jobs, skipped, draws, by_pid, st


def length_report(dan: Path, caps: Sequence[int], seed: int) -> dict:
    arm_a, reach, probs, jobs, skipped, draws, by_pid, st = gather(dan)
    rset = set(reach)
    tt = [p for p in arm_a if p not in rset]
    sets = {"train+test pool (arm A minus reach)": tt, "reach": reach}
    out: dict = {
        "jobs": [j for _, j, _ in jobs],
        "skipped_jobs": skipped,
        "draw_stats": dict(st),
        "sets": {},
    }
    for name, pids in sets.items():
        ds = [d for p in pids for d in by_pid.get(p, [])]
        el = [d for d in ds if eligible(d)]
        rec: dict = {
            "prompts": len(pids),
            "draws": len(ds),
            "strict_pass_draws": sum(strict_pass(d) for d in ds),
            "strict_pass_no_think_end": sum(strict_pass(d) and d.think_end is None for d in ds),
            "strict_pass_gamed": sum(strict_pass(d) and d.label == "gamed" for d in ds),
            "eligible_draws": len(el),
            "eligible_genuine": sum(d.label == "genuine" for d in el),
            "n_tok_eligible": pctl([d.n_tok for d in el]),
            "n_tok_eligible_genuine": pctl([d.n_tok for d in el if d.label == "genuine"]),
            "n_tok_prompt_min": pctl(
                [
                    min(d.n_tok for d in by_pid[p] if eligible(d))
                    for p in pids
                    if any(eligible(d) for d in by_pid.get(p, []))
                ]
            ),
            "status_nocap": dict(Counter(prompt_status(by_pid.get(p, []), 10**9) for p in pids)),
            "caps": {},
        }
        for cap in caps:
            picks = []
            for p in pids:
                c = [d for d in by_pid.get(p, []) if eligible(d) and d.n_tok <= cap]
                if c:
                    picks.append(choose(c, seed, p))
            rec["caps"][cap] = {
                "prompts_ok": len(picks),
                "prompts_genuine_pick": sum(d.label == "genuine" for d in picks),
                "pick_n_tok": pctl([d.n_tok for d in picks]),
                "pick_think_frac_mean": round(
                    float(np.mean([d.think_end / d.n_tok for d in picks])), 3
                )
                if picks
                else None,
            }
        out["sets"][name] = rec
    return out


def print_report(rep: dict) -> None:
    print("jobs:", ", ".join(rep["jobs"]))
    print("skipped s2300 jobs:", "; ".join(f"{s['job']} ({s['why']})" for s in rep["skipped_jobs"]))
    for name, r in rep["sets"].items():
        print(
            f"\n== {name}: {r['prompts']} prompts, {r['draws']} draws, {r['strict_pass_draws']} strict passes "
            f"({r['strict_pass_no_think_end']} without </think>, {r['strict_pass_gamed']} gamed); "
            f"eligible {r['eligible_draws']} ({r['eligible_genuine']} genuine-labelled)"
        )
        print("   status without cap:", r["status_nocap"])
        e = r["n_tok_eligible"]
        print(
            "   n_tok of eligible draws: "
            + " ".join(
                f"{k}={e[k]}"
                for k in ("p10", "p25", "p50", "p75", "p90", "p95", "max", "mean")
                if k in e
            )
        )
        e = r["n_tok_prompt_min"]
        print(
            "   per-prompt shortest eligible: "
            + " ".join(f"{k}={e[k]}" for k in ("p10", "p25", "p50", "p75", "p90", "max") if k in e)
        )
        print("   cap   prompts  genuine-pick  pick n_tok p50 / mean / max   think share")
        for cap, c in r["caps"].items():
            pk = c["pick_n_tok"]
            print(
                f"   {cap:5d} {c['prompts_ok']:7d} {c['prompts_genuine_pick']:12d}   {pk.get('p50', '-'):>6} / {pk.get('mean', '-'):>7} / {pk.get('max', '-'):>5}   {c['pick_think_frac_mean']}"
            )


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build(
    dan: Path,
    out: Path,
    cap: int,
    n_train: int,
    n_test: int,
    seed: int,
    tokenizer: str | None,
    reach_any: bool,
) -> dict:
    arm_a, reach, probs, jobs, skipped, draws, by_pid, st = gather(dan)
    meta = {j: m for _, j, m in jobs}
    rset = set(reach)
    status: dict[str, str] = {}
    pick: dict[str, Draw] = {}
    for p in dict.fromkeys(arm_a + reach):
        status[p] = prompt_status(by_pid.get(p, []), cap)
        if status[p] == "ok":
            pick[p] = choose([d for d in by_pid[p] if eligible(d) and d.n_tok <= cap], seed, p)
    tt = [p for p in arm_a if p not in rset and p in pick]
    order = stratified_order(tt, {p: probs[p]["task"] for p in tt}, seed)
    if len(order) < n_train + n_test:
        raise SystemExit(
            f"only {len(order)} train/test prompts survive at cap {cap}; need {n_train + n_test}"
        )
    sets = {"train": order[:n_train], "test": order[n_train : n_train + n_test]}
    reach_kept = [p for p in reach if p in pick and (reach_any or pick[p].label == "genuine")]
    sets["reach"] = reach_kept

    tok = None
    if tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tokenizer)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    n_prompt_hash_checked = 0
    for name, pids in sets.items():
        rows = []
        for k, p in enumerate(pids):
            d = pick[p]
            pr = probs[p]
            iid = pr["instruction_id_list"]
            iid = list(json.loads(iid.replace("'", '"')) if isinstance(iid, str) else iid)
            ids = decode_ids(d.ids_b64)
            assert ids[-1] == EOS and d.think_end is not None and d.mark_pos is not None
            row = {
                "set": name,
                "idx": k,
                "pid": p,
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
                "label": "genuine" if d.label == "genuine" else "strict-only",
                "label_source": d.label_source,
                "instruction_id_list": iid,
                "text_sha256": hashlib.sha256(d.text.encode()).hexdigest(),
                "prompt_sha256_dan": (meta.get(d.job, {}).get("prompt_sha256_by_pid") or {}).get(p),
            }
            if tok is not None:
                pids_ = prompt_ids(tok, pr["prompt"])
                sha = hashlib.sha256(np.asarray(pids_, dtype=np.int32).tobytes()).hexdigest()
                if row["prompt_sha256_dan"] is not None:
                    assert sha == row["prompt_sha256_dan"], (
                        p,
                        "prompt ids differ from the sampler's",
                    )
                    n_prompt_hash_checked += 1
                row["prompt_ids"] = base64.b64encode(
                    np.asarray(pids_, dtype=np.int32).tobytes()
                ).decode()
                row["prompt_ids_sha256"] = sha
                row["n_prompt"] = len(pids_)
            rows.append(row)
        f = out / f"{name}.jsonl"
        tmp = f.with_suffix(".jsonl.tmp")
        with open(tmp, "w") as g:
            for r in rows:
                g.write(json.dumps(r) + "\n")
        tmp.replace(f)
        files[f.name] = {"sha256": sha256_file(f), "rows": len(rows)}

    def tok_stats(pids: list[str]) -> dict:
        return {
            "n_tok": pctl([pick[p].n_tok for p in pids]),
            "n_think": pctl([pick[p].think_end for p in pids]),
            "n_answer": pctl([pick[p].n_tok - pick[p].think_end for p in pids]),
            "genuine": sum(pick[p].label == "genuine" for p in pids),
            "strict_only": sum(pick[p].label != "genuine" for p in pids),
            "total_response_tokens": sum(pick[p].n_tok for p in pids),
        }

    try:
        dan_commit = subprocess.run(
            ["git", "-C", str(dan), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        dan_commit = None
    man = {
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "builder": "build_pool.py (olmo_elicitation_20261125)",
        "dan_clone": str(dan),
        "dan_commit": dan_commit,
        "seed": seed,
        "max_tok": cap,
        "n_train": n_train,
        "n_test": n_test,
        "reach_rule": "any strict-only or genuine" if reach_any else "genuine pick only",
        "source_model": {
            "repo": "allenai/Olmo-3.1-32B-Think",
            "revision": "main (= step_2300)",
            "label": "s2300",
        },
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
        "skipped_s2300_jobs": skipped,
        "label_sources": [s for s, _ in LABEL_SOURCES]
        + ["identity inheritance among labelled s2300 draws"],
        "template_sha256": TEMPLATE_SHA256,
        "think_marker": {
            "ids": list(MARK),
            "close_ids": MARK_CLOSE,
            "segment_rule": "think = ids[:think_end] (through the '>' token of the last </think>), answer = ids[think_end:] (EOS included)",
        },
        "rules": __doc__.split("Rules, in order, per prompt")[1]
        .split("Response ids are stored")[0]
        .strip(),
        "draw_stats": dict(st),
        "prompt_status": {
            k: dict(Counter(status[p] for p in ps))
            for k, ps in (
                ("arm A minus reach", [p for p in arm_a if p not in rset]),
                ("reach", reach),
            )
        },
        "reach_dropped": {
            p: (
                status[p]
                if p not in pick
                else f"pick is {pick[p].label or 'unlabelled'} (strict-only)"
            )
            for p in reach
            if p not in reach_kept
        },
        "train_test_spare": len(order) - n_train - n_test,
        "sets": {k: {"pids": v, **tok_stats(v)} for k, v in sets.items()},
        "tokenizer": tokenizer,
        "prompt_hash_checked_against_sampler": n_prompt_hash_checked,
        "files": files,
    }
    mf = out / "MANIFEST.json"
    tmp = mf.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(man, indent=1) + "\n")
    tmp.replace(mf)
    return man


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cmd", choices=["report", "build"])
    ap.add_argument("--dan", type=Path, default=DAN)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--caps", default=",".join(map(str, CAPS)))
    ap.add_argument("--max-tok", type=int, default=6144)
    ap.add_argument("--n-train", type=int, default=64)
    ap.add_argument("--n-test", type=int, default=32)
    ap.add_argument(
        "--tokenizer",
        default=None,
        help="tokenizer dir: store prompt ids built with Dan's encoding",
    )
    ap.add_argument(
        "--reach-any", action="store_true", help="keep reach prompts whose pick is strict-only too"
    )
    ap.add_argument("--json", type=Path, default=None, help="report: also write the report here")
    a = ap.parse_args()
    if a.cmd == "report":
        rep = length_report(a.dan, [int(x) for x in a.caps.split(",")], a.seed)
        print_report(rep)
        if a.json:
            a.json.write_text(json.dumps(rep, indent=1) + "\n")
    else:
        man = build(a.dan, a.out, a.max_tok, a.n_train, a.n_test, a.seed, a.tokenizer, a.reach_any)
        print(
            json.dumps(
                {
                    k: man[k]
                    for k in (
                        "max_tok",
                        "seed",
                        "files",
                        "prompt_status",
                        "train_test_spare",
                        "prompt_hash_checked_against_sampler",
                    )
                },
                indent=1,
            )
        )
        for k, v in man["sets"].items():
            print(k, len(v["pids"]), "genuine", v["genuine"], "n_tok", v["n_tok"])


if __name__ == "__main__":
    main()
