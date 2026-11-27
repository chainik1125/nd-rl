#!/usr/bin/env python3
"""M3 vLLM pre-flight (pod; F4 step 5): the patched engine against the HF references, before any sampling.

    steer_preflight.py run       --mode stock|zero|chosen|chosen_half|scramble2001 [--eager] [--suffix S] --model M --p3 P3DIR --out DIR
    steer_preflight.py summarize --p3 P3DIR --dir DIR --out report.json          (exit 0 = PASS, 4 = PASS_CHOSEN_ONLY, 1 = FAIL)

`run` is one vLLM process per mode (one vector per process).  Each run scores the four stored sequences with
`prompt_logprobs` (max_tokens 1) and repeats sequence 0 once in the same engine (prefix-cache repeat).  `summarize`
compares against the HF log-probs of steer_hf.py p3 (preflight_refs.json):
  1. stock noise floor        vLLM stock vs HF stock (mean / p99 / max |delta logprob|); the tolerance comes from this
  2. zero-vector identity     patched zero vector vs vLLM stock: bit-identical in eager mode (Amendment 1: compiled stock
                              is not bit-reproducible across processes), within tolerance compiled
  3. steering as intended     vLLM chosen (and scramble2001) vs HF hook: mean <= tol_mean and p99 <= tol_p99, AND
                              the drop test (Amendment 2): vLLM stock scored against the HF steered reference must FAIL
                              that match, so a vector dropped by the engine would be detected
  4. compile vs eager         chosen (compile + graphs) vs chosen --eager within tolerance
  5. prefix-cache repeat      the second pass over sequence 0 equals the first
  6. STEER_ACTIVE             a record with the right layer and vector sha256 in the active file of every steered run
Amendment 2 (2026-11-27): (a) the power check is replaced by the direct drop test above (effect over tolerance is only
recorded); (b) eager zero identity is exact, or, if the eager stock run does not reproduce itself across processes (optional
second run `--mode stock --eager --suffix 2` = stock_eager2), zero_eager must lie within EAGER_SPREAD_FACTOR (1.5) x the
stock_eager vs stock_eager2 spread in mean and p99 (path recorded in zero_identity_path); (c) the prefix-cache repeat passes
if exact (max <= 1e-3) or, failing that, within tol_mean / tol_p99 (prefix_cache_repeat_path); (d) `pass_chosen` = every
check except those starting `scramble2001_`: if only scramble checks fail, the chosen-vector cells may run and E2's scrambles
are verified only by STEER_ACTIVE sha256. Exit 0 PREFLIGHT_PASS, 4 PREFLIGHT_PASS_CHOSEN_ONLY, 1 PREFLIGHT_FAIL.
Positions beyond 4,096 (sliding-window layers) are reported separately.  Thresholds are set from data: tol_mean =
max(3 x floor mean, 0.01), tol_p99 = max(3 x floor p99, 0.1) [guesses for the factor 3 and the floors].
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import steer_common as C  # noqa: E402

MODES = ("stock", "zero", "chosen", "chosen_half", "scramble2001")
EAGER_SPREAD_FACTOR = 1.5  # PREREG_steer Amendment 2 (2026-11-27)
FACTOR, FLOOR_MEAN, FLOOR_P99 = 3.0, 0.01, 0.1
SLIDING = 4096


def delta_stats(a: list, b: list, start: int = 0) -> dict:
    """|delta logprob| statistics over a list of per-sequence arrays (equal lengths), tokens from index `start`."""
    d = (
        np.concatenate(
            [
                np.abs(
                    np.asarray(x[start:], dtype=np.float64)
                    - np.asarray(y[start:], dtype=np.float64)
                )
                for x, y in zip(a, b)
            ]
        )
        if a
        else np.array([])
    )
    if d.size == 0:
        return {"n": 0, "mean": float("nan"), "p99": float("nan"), "max": float("nan")}
    return {
        "n": int(d.size),
        "mean": float(d.mean()),
        "p99": float(np.percentile(d, 99)),
        "max": float(d.max()),
    }


def beyond_window(seqs, logprobs, start=SLIDING) -> list:
    """Per sequence, the log-probs at positions >= SLIDING (logprobs[k] belongs to token k + 1)."""
    return [lp[max(start - 1, 0) :] if s["n"] > start else [] for s, lp in zip(seqs, logprobs)]


def summarize_runs(refs: dict, runs: dict, active: dict, expect: dict, require_eager=True) -> dict:
    """Pure comparison logic (unit-tested).  refs = preflight_refs.json; runs = {name: {"logprobs": [...], "repeat_seq0": [...]}}
    with names stock, zero, chosen, chosen_half, scramble2001, chosen_eager; active = {name: [records]};
    expect = {"layer": l, "chosen_sha": ..., "scramble_sha": ..., "half_sha": ...}."""
    seqs, hf = refs["seqs"], refs["refs"]
    res = {"checks": {}, "stats": {}}
    need = ["stock", "zero", "chosen", "scramble2001", "stock_eager", "zero_eager"] + (
        ["chosen_eager"] if require_eager else []
    )
    missing = [m for m in need if m not in runs]
    res["missing_runs"] = missing
    floor = delta_stats(runs["stock"]["logprobs"], hf["stock"]) if "stock" in runs else None
    res["stats"]["stock_vs_hf"] = floor
    if floor:
        tol_mean = max(FACTOR * floor["mean"], FLOOR_MEAN)
        tol_p99 = max(FACTOR * floor["p99"], FLOOR_P99)
    else:
        tol_mean, tol_p99 = FLOOR_MEAN, FLOOR_P99
    res["tolerance"] = {"mean": tol_mean, "p99": tol_p99}

    def ok(st):
        return st["n"] > 0 and st["mean"] <= tol_mean and st["p99"] <= tol_p99

    # PREREG_steer Amendment 1 (2026-11-27): compiled vLLM does not reproduce itself across processes on rp-m2a (stock vs
    # stock: 93 % of positions differ, mean 0.010), so bit identity of the zero vector is tested in eager mode (stock --eager
    # vs zero --eager), and the compiled zero run must agree with compiled stock within the tolerance. The compiled exact
    # comparison stays in the stats as zero_identity_compiled_exact (information only).
    if "zero" in runs and "stock" in runs:
        z = delta_stats(runs["zero"]["logprobs"], runs["stock"]["logprobs"])
        res["stats"]["zero_vs_stock"] = z
        res["stats"]["zero_identity_compiled_exact"] = bool(z["n"] > 0 and z["max"] == 0.0)
        res["checks"]["zero_compiled_within_tol"] = bool(ok(z))
    if "zero_eager" in runs and "stock_eager" in runs:
        ze = delta_stats(runs["zero_eager"]["logprobs"], runs["stock_eager"]["logprobs"])
        res["stats"]["zero_eager_vs_stock_eager"] = ze
        if ze["n"] > 0 and ze["max"] == 0.0:
            res["checks"]["zero_identity"], res["stats"]["zero_identity_path"] = True, "exact"
        else:
            spread = None
            if "stock_eager2" in runs:
                spread = delta_stats(
                    runs["stock_eager2"]["logprobs"], runs["stock_eager"]["logprobs"]
                )
                res["stats"]["stock_eager2_vs_stock_eager"] = spread
                res["stats"]["eager_stock_repeat_exact"] = bool(
                    spread["n"] > 0 and spread["max"] == 0.0
                )
            if spread and spread["n"] > 0 and spread["max"] != 0.0:
                res["checks"]["zero_identity"] = bool(
                    ze["n"] > 0
                    and ze["mean"] <= EAGER_SPREAD_FACTOR * spread["mean"]
                    and ze["p99"] <= EAGER_SPREAD_FACTOR * spread["p99"]
                )
                res["stats"]["zero_identity_path"] = "fallback_eager_spread"
            else:
                res["checks"]["zero_identity"], res["stats"]["zero_identity_path"] = (
                    False,
                    "exact_failed",
                )
    elif "stock_eager2" in runs and "stock_eager" in runs:
        sp = delta_stats(runs["stock_eager2"]["logprobs"], runs["stock_eager"]["logprobs"])
        res["stats"]["stock_eager2_vs_stock_eager"] = sp
        res["stats"]["eager_stock_repeat_exact"] = bool(sp["n"] > 0 and sp["max"] == 0.0)
    for name, hfkey, tag in (
        ("chosen", "chosen", "chosen"),
        ("scramble2001", "scramble2001", "scramble"),
    ):
        if name not in runs:
            continue
        st = delta_stats(runs[name]["logprobs"], hf[hfkey])
        res["stats"][f"{name}_vs_hf"] = st
        res["checks"][f"{name}_matches_hf"] = bool(ok(st))
        if "stock" in runs:
            eff = delta_stats(runs[name]["logprobs"], runs["stock"]["logprobs"])
            res["stats"][f"{name}_vs_stock_vllm"] = eff
            res["stats"][f"{name}_effect_over_tol_mean"] = (
                eff["mean"] / tol_mean if tol_mean else float("nan")
            )
            # drop test: stock scored against the HF steered reference must fail the HF match
            drop = delta_stats(runs["stock"]["logprobs"], hf[hfkey])
            res["stats"][f"{name}_stock_vllm_vs_hf_steered"] = drop
            res["checks"][f"{name}_drop_detected"] = bool(not ok(drop))
        far = delta_stats(
            beyond_window(seqs, runs[name]["logprobs"]), beyond_window(seqs, hf[hfkey])
        )
        res["stats"][f"{name}_vs_hf_beyond_{SLIDING}"] = far
        if far["n"] > 0:
            res["checks"][f"{name}_matches_hf_beyond_{SLIDING}"] = bool(ok(far))
    if "chosen" in runs and "chosen_eager" in runs:
        ce = delta_stats(runs["chosen"]["logprobs"], runs["chosen_eager"]["logprobs"])
        res["stats"]["compile_vs_eager"] = ce
        res["checks"]["compile_vs_eager"] = bool(ok(ce))
    if "chosen_half" in runs:
        st = delta_stats(runs["chosen_half"]["logprobs"], hf["chosen_half"])
        res["stats"]["chosen_half_vs_hf"] = st
        res["checks"]["chosen_half_matches_hf"] = bool(ok(st))
    if "stock" in runs and runs["stock"].get("repeat_seq0") is not None:
        r = delta_stats([runs["stock"]["logprobs"][0]], [runs["stock"]["repeat_seq0"]])
        res["stats"]["prefix_cache_repeat"] = r
        if r["n"] > 0 and r["max"] <= 1e-3:
            res["checks"]["prefix_cache_repeat"], res["stats"]["prefix_cache_repeat_path"] = (
                True,
                "exact",
            )
        elif ok(r):
            res["checks"]["prefix_cache_repeat"], res["stats"]["prefix_cache_repeat_path"] = (
                True,
                "fallback_tol",
            )
        else:
            res["checks"]["prefix_cache_repeat"], res["stats"]["prefix_cache_repeat_path"] = (
                False,
                "failed",
            )
    # STEER_ACTIVE records
    for name, sha_key in (
        ("chosen", "chosen_sha"),
        ("chosen_eager", "chosen_sha"),
        ("scramble2001", "scramble_sha"),
        ("chosen_half", "half_sha"),
        ("zero", "zero_sha"),
        ("zero_eager", "zero_sha"),
    ):
        if name in runs and expect.get(sha_key):
            recs = active.get(name, [])
            res["checks"][f"{name}_steer_active"] = bool(
                any(
                    r.get("layer") == expect["layer"] and r.get("vec_sha256") == expect[sha_key]
                    for r in recs
                )
            )
    res["checks"]["no_missing_required_runs"] = not missing
    res["pass"] = bool(res["checks"]) and all(res["checks"].values())
    res["pass_chosen"] = bool(res["checks"]) and all(
        v for k, v in res["checks"].items() if not k.startswith("scramble2001_")
    )
    return res


# --- run ----------------------------------------------------------------------------------------------------------------
def vllm_logprobs(llm, seqs, repeat_first=True):
    from vllm import SamplingParams

    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)

    def one(ids):
        out = llm.generate([{"prompt_token_ids": ids}], sp, use_tqdm=False)[0]
        pl = out.prompt_logprobs
        return [float(pl[t][ids[t]].logprob) for t in range(1, len(ids))]

    ids_list = [C.ids_from_b64(s["ids"]) for s in seqs]
    lps = [one(ids) for ids in ids_list]
    rep = one(ids_list[0]) if repeat_first else None
    return lps, rep


def cmd_run(a):
    p3 = Path(a.p3)
    chosen = json.loads((p3 / "chosen.json").read_text())
    refs = json.loads((p3 / "preflight_refs.json").read_text())
    mode = a.mode
    vec_file = {
        "stock": None,
        "zero": chosen["files"]["zero"],
        "chosen": chosen["files"]["chosen"],
        "chosen_half": chosen["files"]["chosen_half"],
        "scramble2001": chosen["files"]["scramble2001"],
    }[mode]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = mode + ("_eager" if a.eager else "") + a.suffix
    active = out / f"active_{name}.jsonl"
    if active.exists():
        active.unlink()
    if vec_file:
        os.environ["STEER_LAYER"] = str(chosen["layer"])
        os.environ["STEER_VEC"] = vec_file
        os.environ["STEER_ACTIVE_FILE"] = str(active)
    else:
        os.environ.pop("STEER_LAYER", None)
    if os.environ.get("STEER_INPROC") == "1":
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    import steer_patch

    steer_patch.install_from_env()
    from vllm import LLM

    llm = LLM(
        model=a.model,
        tokenizer=a.model,
        dtype="bfloat16",
        generation_config="vllm",
        max_model_len=8192,
        gpu_memory_utilization=a.gpu_mem,
        enable_prefix_caching=True,
        seed=0,
        enforce_eager=a.eager,
    )
    lps, rep = vllm_logprobs(llm, refs["seqs"])
    (out / f"pf_{name}.json").write_text(
        json.dumps({"mode": mode, "eager": a.eager, "logprobs": lps, "repeat_seq0": rep})
    )
    print(f"PREFLIGHT_RUN_DONE {name}", flush=True)


def cmd_summarize(a):
    p3, d = Path(a.p3), Path(a.dir)
    refs = json.loads((p3 / "preflight_refs.json").read_text())
    chosen = json.loads((p3 / "chosen.json").read_text())
    runs, active = {}, {}
    for f in d.glob("pf_*.json"):
        name = f.stem[3:]
        runs[name] = json.loads(f.read_text())
        af = d / f"active_{name}.jsonl"
        active[name] = [json.loads(ln) for ln in af.read_text().splitlines()] if af.exists() else []
    sha = chosen["sha256"]
    expect = {
        "layer": chosen["layer"],
        "chosen_sha": sha["chosen"],
        "scramble_sha": sha["scramble2001"],
        "half_sha": sha["chosen_half"],
        "zero_sha": sha["zero"],
    }
    rep = summarize_runs(refs, runs, active, expect, require_eager=not a.no_eager)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    verdict, code = (
        ("PREFLIGHT_PASS", 0)
        if rep["pass"]
        else ("PREFLIGHT_PASS_CHOSEN_ONLY", 4)
        if rep["pass_chosen"]
        else ("PREFLIGHT_FAIL", 1)
    )
    print(verdict + " " + json.dumps(rep["checks"]))
    return code


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--mode", required=True, choices=MODES)
    r.add_argument("--eager", action="store_true")
    r.add_argument(
        "--suffix", default="", help="appended to the output name, e.g. 2 -> stock_eager2"
    )
    r.add_argument("--model", required=True)
    r.add_argument("--p3", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--gpu-mem", type=float, default=0.90)
    s = sub.add_parser("summarize")
    s.add_argument("--p3", required=True)
    s.add_argument("--dir", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--no-eager", action="store_true", help="do not require the chosen --eager run")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        cmd_run(a)
        return 0
    return cmd_summarize(a)


if __name__ == "__main__":
    sys.exit(main())
