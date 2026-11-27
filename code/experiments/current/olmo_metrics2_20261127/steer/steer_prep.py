#!/usr/bin/env python3
"""M3 steering inputs (serrano; reads Dan's clone and the 2026-11-25 pool read-only; writes only under --out).

    python -I steer_prep.py build [--out DIR]         prompt manifest of F2, texts, pre-flight sequences, pid files
    python -I steer_prep.py pids  --out DIR --manifest DIR/manifest.json [--reach-n 8 --held-n 2 --dev-n 2 ...]
                                                       (pure python; also runs on the pod) pid files for the sampled cells

Design: discourse/steer.md Turn 5 section F.  Sets (F2): N = arm-A prompts minus reach (176); V32 = lines 0-31 of
pool/train.jsonl (members of N, else the next lines); the other 144 sorted by pid and permuted with
RandomState(0).permutation: first 64 = dev, next 80 = held-out.  S16 / H16 = the first 16 dev / held-out prompts in
permutation order that have an eligible stored s2300 strict pass.  No reach prompt enters any set except the reach cells.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import steer_common as C  # noqa: E402

DAN = Path("/scratch/cowifornia/nd-rl/olmo_elicit/ext/nd-takehome")
POOL = Path("/scratch/cowifornia/nd-rl/olmo_elicit/pool")
TOK = Path("/scratch/cowifornia/nd-rl/olmo_elicit/tok/s2300")
OUT = Path("/scratch/cowifornia/nd-rl/olmo_elicit/metrics2/inputs/steer")
BUILD_POOL = HERE.parent.parent / "olmo_elicitation_20261125" / "build_pool.py"
PROBLEMS = "olmo_ifomega/problems.jsonl"
CAP_TEXT = 12000  # dpo texts truncated here (F2); s2300 NLL texts must not be longer
N_VEC, N_DEV, N_HELD, N_SEL, N_CHK = 32, 64, 80, 16, 16
SPLIT_SEED = 0
PREFLIGHT_TARGETS = (1000, 1000, 4300, 5500)
BASELINE_JOB = "F_dpo_ifb"  # the only stored plain dpo job with exactly 16 non-sequential draws for every prompt;
# the other dpo jobs are sequential (stop_after) or reach top-ups and would bias per-prompt rates


def load_module(path: Path, name: str):
    sys.dont_write_bytecode = True  # never write __pycache__ into Dan's tree or the 20261125 folder
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_pids(p: Path) -> list[str]:
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]


# --- sets ---------------------------------------------------------------------------------------------------------------
def make_sets(arm_a: list[str], reach: list[str], pool_train_pids: list[str]) -> dict:
    """The disjoint prompt sets of F2 (pure function of the pid lists; unit-tested)."""
    reach_set = set(reach)
    N = sorted(p for p in arm_a if p not in reach_set)
    Nset = set(N)
    vec = []
    for p in pool_train_pids:
        if p in Nset and p not in vec:
            vec.append(p)
        if len(vec) == N_VEC:
            break
    assert len(vec) == N_VEC, f"only {len(vec)} pool-train prompts in N"
    rest = sorted(p for p in N if p not in set(vec))
    perm = np.random.RandomState(SPLIT_SEED).permutation(len(rest))
    ordered = [rest[i] for i in perm]
    dev, held = ordered[:N_DEV], ordered[N_DEV : N_DEV + N_HELD]
    assert len(dev) == N_DEV and len(held) == N_HELD, (len(rest), len(dev), len(held))
    return {
        "N": N,
        "vec": vec,
        "dev": dev,
        "held": held,
        "reach": list(reach),
        "unused": ordered[N_DEV + N_HELD :],
    }


def check_disjoint(sets: dict) -> dict:
    """Assertions of F2: reach is outside N, vec/dev/held pairwise disjoint, no reach pid anywhere else."""
    r = set(sets["reach"])
    out = {}
    for k in ("N", "vec", "dev", "held", "unused"):
        assert not (r & set(sets[k])), f"reach prompt in {k}"
    assert not (set(sets["vec"]) & set(sets["dev"])) and not (set(sets["vec"]) & set(sets["held"]))
    assert not (set(sets["dev"]) & set(sets["held"]))
    out["reach_disjoint_from_all_non_reach_sets"] = True
    out["vec_dev_held_pairwise_disjoint"] = True
    return out


# --- stored draws -------------------------------------------------------------------------------------------------------
def discover_plain(bp, dan: Path, name: str):
    """Plain-prompt stored jobs of one checkpoint name (problems file olmo_ifomega/problems.jsonl, no LoRA)."""
    keep = []
    for mf in sorted(dan.glob("artifacts/*/meta/*.json")):
        try:
            m = bp.meta_of(mf)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if (
            m.get("name") != name
            or "job" not in m
            or m.get("problems") != PROBLEMS
            or m.get("lora")
        ):
            continue
        run, job = mf.parent.parent.name, m["job"]
        if (dan / "artifacts" / run / "samples" / f"{job}.jsonl.gz").exists() and (
            dan / "artifacts" / run / "graded" / f"{job}.jsonl"
        ).exists():
            keep.append((run, job, m))
    return keep


def dedupe(draws):
    seen, out = set(), []
    for d in draws:
        if d.key in seen:
            continue
        seen.add(d.key)
        out.append(d)
    return out


def text_record(tid, pid, model, role, prompt_ids, resp_ids, tok, src, cap=None):
    lead_end, trail_start = C.prompt_segments(tok, prompt_ids)
    n_orig = len(resp_ids)
    truncated = cap is not None and n_orig > cap
    if truncated:
        resp_ids = resp_ids[:cap]
    return {
        "tid": tid,
        "pid": pid,
        "model": model,
        "role": role,
        "prompt_ids": C.b64_ids(prompt_ids),
        "resp_ids": C.b64_ids(resp_ids),
        "n_prompt": len(prompt_ids),
        "n_resp": len(resp_ids),
        "n_resp_stored": n_orig,
        "truncated": truncated,
        "think_end": C.find_marker(resp_ids),
        "lead_end": lead_end,
        "trail_start": trail_start,
        "src": src,
    }


def pick_dpo(draws, pid, seed=SPLIT_SEED):
    ds = sorted([d for d in draws if d.pid == pid], key=lambda d: (d.job, d.chunk, d.i))
    return random.Random(f"{seed}:{pid}").choice(ds) if ds else None


def first_s2300_pass(draws, pid, cap=CAP_TEXT):
    """F2: the first strict-pass stored s2300 draw in draw-index order (job, chunk, i): finish == stop, with an
    end-of-thinking marker, not hand-labelled gamed (the pool's rules 1, 2, 4), at most `cap` response tokens."""
    ds = sorted([d for d in draws if d.pid == pid], key=lambda d: (d.job, d.chunk, d.i))
    for d in ds:
        if (
            d.strict
            and d.finish == "stop"
            and d.think_end is not None
            and d.label != "gamed"
            and d.n_tok <= cap
        ):
            return d
    return None


# --- build -------------------------------------------------------------------------------------------------------------
def build(out: Path, dan: Path = DAN, pool: Path = POOL, tokdir: Path = TOK) -> dict:
    t0 = time.time()
    bp = load_module(BUILD_POOL, "build_pool_ro")
    dan_sample = load_module(dan / "olmo_ifladder" / "sample.py", "dan_sample_ro")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(tokdir))

    arm_a = read_pids(dan / "olmo_ifcurve" / "pids_armA.txt")
    reach = read_pids(dan / "olmo_ifcurve" / "pids_reach.txt")
    pool_manifest = json.loads((pool / "MANIFEST.json").read_text())
    pool_rows = [
        json.loads(ln) for ln in (pool / "train.jsonl").read_text().splitlines() if ln.strip()
    ]
    # the manifest's own train pid list must match the file
    assert [r["pid"] for r in pool_rows] == pool_manifest["sets"]["train"]["pids"], (
        "pool train.jsonl vs MANIFEST.json"
    )
    sets = make_sets(arm_a, reach, [r["pid"] for r in pool_rows])
    disj = check_disjoint(sets)
    problems = {}
    for ln in (dan / PROBLEMS).read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            problems[r["pid"]] = r
    missing = [p for k in ("vec", "dev", "held", "reach") for p in sets[k] if p not in problems]
    assert not missing, f"pids not in Dan's problems file: {missing[:5]}"

    # stored draws -------------------------------------------------------------------------------------------------
    need = set(sets["vec"]) | set(sets["dev"]) | set(sets["held"])
    dpo_jobs = [j for j in discover_plain(bp, dan, "dpo") if j[1] == BASELINE_JOB]
    assert len(dpo_jobs) == 1, "baseline job not found"
    dpo_draws, dpo_stats = bp.load_draws(dan, need, dpo_jobs)
    dpo_draws = dedupe(dpo_draws)
    per = Counter(d.pid for d in dpo_draws)
    assert all(per[p] == 16 for p in need), {p: per[p] for p in need if per[p] != 16}
    s_jobs, s_skipped = bp.discover_jobs(dan)
    cand = set(sets["dev"]) | set(sets["held"])
    s_draws, s_stats = bp.load_draws(dan, cand, s_jobs)
    s_draws = dedupe(s_draws)
    bp.apply_labels(s_draws, bp.load_labels(dan))

    def prompt_ids_of(pid):
        return dan_sample.prompt_ids(tok, problems[pid]["prompt"])

    texts, notes = [], Counter()
    # V32: dpo text (V_resp, V_prompt) and the pool's s2300 text (diagnostic V_resp,s2300text)
    pool_by_pid = {r["pid"]: r for r in pool_rows}
    for pid in sets["vec"]:
        pids_ = prompt_ids_of(pid)
        pr = pool_by_pid[pid]
        assert (
            C.sha256_bytes(np.asarray(pids_, dtype=np.int32).tobytes()) == pr["prompt_ids_sha256"]
        ), f"prompt ids of {pid} differ from the pool's"
        d = pick_dpo(dpo_draws, pid)
        assert d is not None, f"no stored dpo draw for {pid}"
        texts.append(
            text_record(
                f"vec_dpo|{pid}",
                pid,
                "dpo",
                "vec_dpo",
                pids_,
                C.ids_from_b64(d.ids_b64),
                tok,
                {"job": d.job, "chunk": d.chunk, "i": d.i, "run": d.run, "n_tok": d.n_tok},
                cap=CAP_TEXT,
            )
        )
        texts.append(
            text_record(
                f"vec_s2300|{pid}",
                pid,
                "s2300",
                "vec_s2300",
                pids_,
                C.ids_from_b64(pr["ids"]),
                tok,
                {
                    "pool": "train.jsonl",
                    "idx": pr["idx"],
                    "job": pr["job"],
                    "chunk": pr["chunk"],
                    "i": pr["i"],
                    "n_tok": pr["n_tok"],
                },
            )
        )

    # S16 / H16
    def first_n(order, n, tag):
        got, skipped = [], []
        for pid in order:
            d = first_s2300_pass(s_draws, pid)
            if d is None:
                skipped.append(pid)
                continue
            got.append((pid, d))
            if len(got) == n:
                break
        assert len(got) == n, f"{tag}: only {len(got)} prompts with an eligible s2300 pass"
        return got, skipped

    sel, sel_skip = first_n(sets["dev"], N_SEL, "S16")
    chk, chk_skip = first_n(sets["held"], N_CHK, "H16")
    for pid, d in sel:
        pids_ = prompt_ids_of(pid)
        src = {
            "job": d.job,
            "chunk": d.chunk,
            "i": d.i,
            "run": d.run,
            "n_tok": d.n_tok,
            "label": d.label,
        }
        texts.append(
            text_record(
                f"sel_s2300|{pid}",
                pid,
                "s2300",
                "sel_s2300",
                pids_,
                C.ids_from_b64(d.ids_b64),
                tok,
                src,
            )
        )
        dd = pick_dpo(dpo_draws, pid)
        assert dd is not None, f"no stored dpo draw for S16 prompt {pid}"
        texts.append(
            text_record(
                f"sel_dpo|{pid}",
                pid,
                "dpo",
                "sel_dpo",
                pids_,
                C.ids_from_b64(dd.ids_b64),
                tok,
                {"job": dd.job, "chunk": dd.chunk, "i": dd.i, "run": dd.run, "n_tok": dd.n_tok},
                cap=CAP_TEXT,
            )
        )
    for pid, d in chk:
        pids_ = prompt_ids_of(pid)
        src = {
            "job": d.job,
            "chunk": d.chunk,
            "i": d.i,
            "run": d.run,
            "n_tok": d.n_tok,
            "label": d.label,
        }
        texts.append(
            text_record(
                f"chk_s2300|{pid}",
                pid,
                "s2300",
                "chk_s2300",
                pids_,
                C.ids_from_b64(d.ids_b64),
                tok,
                src,
            )
        )
    for t in texts:
        notes[f"texts:{t['role']}"] += 1
        notes["truncated_dpo_texts"] += int(t["truncated"])
        notes["texts_without_marker"] += int(t["think_end"] is None)

    # pre-flight sequences: stored dpo draws on dev prompts (never reach), total length near the targets -------------
    dev_dpo = [
        d for d in dpo_draws if d.pid in set(sets["dev"]) and d.finish == "stop" and d.n_tok > 0
    ]
    used, seqs = set(), []
    for tgt in PREFLIGHT_TARGETS:
        best = None
        for d in sorted(dev_dpo, key=lambda d: (d.job, d.pid, d.chunk, d.i)):
            if d.pid in used:
                continue
            n_prompt = len(prompt_ids_of(d.pid))
            gap = abs(n_prompt + d.n_tok - tgt)
            if best is None or gap < best[0]:
                best = (gap, d, n_prompt)
        gap, d, n_prompt = best
        used.add(d.pid)
        ids = prompt_ids_of(d.pid) + C.ids_from_b64(d.ids_b64)
        seqs.append(
            {
                "tid": f"pf{len(seqs)}",
                "pid": d.pid,
                "target": tgt,
                "n": len(ids),
                "n_prompt": n_prompt,
                "ids": C.b64_ids(ids),
                "src": {"job": d.job, "chunk": d.chunk, "i": d.i},
            }
        )

    # stored dpo baselines on dev (cap-12k scoring) and held-out (E0) ------------------------------------------------
    base = {"dev": {}, "held": {}}
    for k, pids in (("dev", sets["dev"]), ("held", sets["held"])):
        for pid in pids:
            ds = sorted([d for d in dpo_draws if d.pid == pid], key=lambda d: (d.job, d.chunk, d.i))
            base[k][pid] = [[bool(d.strict), int(d.n_tok), d.finish] for d in ds]
    nper = {k: Counter(len(v) for v in base[k].values()) for k in base}

    # write ------------------------------------------------------------------------------------------------------------
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "texts.jsonl", "w") as f:
        for t in texts:
            f.write(json.dumps(t) + "\n")
    C.dump_json(seqs, out / "preflight_seqs.json")
    C.dump_json(base, out / "dpo_baseline.json")
    inputs = {}
    for label, p in (
        ("pids_armA", dan / "olmo_ifcurve" / "pids_armA.txt"),
        ("pids_reach", dan / "olmo_ifcurve" / "pids_reach.txt"),
        ("pool_train", pool / "train.jsonl"),
        ("pool_manifest", pool / "MANIFEST.json"),
        ("problems", dan / PROBLEMS),
        ("dan_sample_py", dan / "olmo_ifladder" / "sample.py"),
        ("build_pool_py", BUILD_POOL),
        ("tokenizer_json", tokdir / "tokenizer.json"),
        ("tokenizer_config", tokdir / "tokenizer_config.json"),
    ):
        inputs[label] = {"path": str(p), "sha256": C.sha256_file(p)}
    for run, job, _ in dpo_jobs:
        p = dan / "artifacts" / run / "samples" / f"{job}.jsonl.gz"
        inputs[f"dpo_samples:{run}/{job}"] = {"path": str(p), "sha256": C.sha256_file(p)}
        g = dan / "artifacts" / run / "graded" / f"{job}.jsonl"
        inputs[f"dpo_graded:{run}/{job}"] = {"path": str(g), "sha256": C.sha256_file(g)}
    used_s = {
        (t["src"].get("run"), t["src"].get("job"))
        for t in texts
        if t["model"] == "s2300" and "run" in t["src"]
    }
    for run, job in sorted(used_s):
        for sub, ext in (("samples", ".jsonl.gz"), ("graded", ".jsonl")):
            p = dan / "artifacts" / run / sub / f"{job}{ext}"
            inputs[f"s2300_{sub}:{run}/{job}"] = {"path": str(p), "sha256": C.sha256_file(p)}
    manifest = {
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "builder": "steer_prep.py build",
        "design": "discourse/steer.md Turn 5 section F (+ Turn 6 d1-d7)",
        "split_seed": SPLIT_SEED,
        "dan_commit": pool_manifest["dan_commit"],
        "template_sha256": C.sha256_bytes(dan_sample.TEMPLATE.encode()),
        "pool_template_sha256": pool_manifest["template_sha256"],
        "sets": {
            "vec32": sets["vec"],
            "dev": sets["dev"],
            "held": sets["held"],
            "reach": sets["reach"],
            "S16": [p for p, _ in sel],
            "H16": [p for p, _ in chk],
        },
        "sizes": {
            "N": len(sets["N"]),
            "vec": len(sets["vec"]),
            "dev": len(sets["dev"]),
            "held": len(sets["held"]),
            "reach": len(sets["reach"]),
            "unused": len(sets["unused"]),
        },
        "S16_skipped_no_eligible_pass": sel_skip,
        "H16_skipped_no_eligible_pass": chk_skip,
        "checks": disj,
        "cap_text_tokens": CAP_TEXT,
        "text_sources": {t["tid"]: t["src"] for t in texts},
        "draw_counts_per_prompt_dpo": {k: dict(v) for k, v in nper.items()},
        "dpo_jobs": [f"{r}/{j}" for r, j, _ in dpo_jobs],
        "dpo_load_stats": dict(dpo_stats),
        "s2300_load_stats": dict(s_stats),
        "s2300_jobs_skipped_count": len(s_skipped),
        "notes": dict(notes),
        "marker_ids": {"open": list(C.MARK), "close": list(C.MARK_CLOSE)},
        "seconds": round(time.time() - t0, 1),
        "inputs": inputs,
        "texts_jsonl_sha256": C.sha256_file(out / "texts.jsonl"),
    }
    C.dump_json(manifest, out / "manifest.json")
    write_pids(out, manifest)
    return manifest


# --- pid files --------------------------------------------------------------------------------------------------------
def write_pid_file(path: Path, spec: dict) -> None:
    path.write_text("\n".join(f"{p}:{n}" for p, n in spec.items() if n > 0) + "\n")


def write_pids(out: Path, manifest: dict, reach_n=8, held_n=2, dev_n=2, e0_reach_n=None) -> dict:
    """Pid files in Dan's sampler format ('pid:n' per token, passed as --pids @file).  Draw counts are parameters (the
    full-design values are the defaults); the build is idempotent."""
    s = manifest["sets"]
    e0_reach_n = reach_n if e0_reach_n is None else e0_reach_n
    files = {}
    write_pid_file(out / "pids_dev_guard.txt", {p: dev_n for p in s["dev"]})
    files["dev_guard"] = dev_n
    write_pid_file(out / "pids_e0p_reach.txt", {p: e0_reach_n for p in s["reach"]})
    files["e0p_reach"] = e0_reach_n
    e1 = {p: reach_n for p in s["reach"]}
    e1.update({p: held_n for p in s["held"]})
    write_pid_file(out / "pids_e1.txt", e1)
    files["e1"] = {"reach": reach_n, "held": held_n}
    for k, spec in enumerate(C.e2_pid_specs(s["reach"], s["held"], reach_n, held_n), start=1):
        write_pid_file(out / f"pids_e2_s{k}.txt", spec)
    files["e2"] = {
        "reach_split": C.split_draws(reach_n),
        "held": held_n,
        "assignment": "held-out prompt i -> scramble (i mod 3) + 1",
    }
    C.dump_json(files, out / "pids_params.json")
    return files


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", default=str(OUT))
    b.add_argument("--dan", default=str(DAN))
    b.add_argument("--pool", default=str(POOL))
    b.add_argument("--tok", default=str(TOK))
    p = sub.add_parser("pids")
    p.add_argument("--out", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--reach-n", type=int, default=8)
    p.add_argument("--held-n", type=int, default=2)
    p.add_argument("--dev-n", type=int, default=2)
    p.add_argument("--e0-reach-n", type=int, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        m = build(Path(a.out), Path(a.dan), Path(a.pool), Path(a.tok))
        print(
            json.dumps(
                {
                    "sizes": m["sizes"],
                    "notes": m["notes"],
                    "checks": m["checks"],
                    "seconds": m["seconds"],
                },
                indent=1,
            )
        )
    else:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        m = json.loads(Path(a.manifest).read_text())
        print(json.dumps(write_pids(out, m, a.reach_n, a.held_n, a.dev_n, a.e0_reach_n)))


if __name__ == "__main__":
    main()
