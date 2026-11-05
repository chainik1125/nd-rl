"""Stage 3 sequences: the L3 end's own Lean-accepted attempts on the G and S problems, with traces.

    python -m code.experiments.current.circuit_diffing_20261101.traces --gpu 2

G: every G problem of `stage2_problems.json`, `--kG` attempts. S: a fixed random subset of `--nS`
S problems (seed 0), `--kS` attempts. The read setting of Stage 2 (plain arm, B64, temperature
0.8, max_action 512, max_steps 96), seed 11, judged by Lean. Per problem, the first `--keep`
accepted attempts (by attempt index) are kept; each accepted step is one sequence (the step's
decoder input followed by every sampled token of the accepted draw).

`metric` marks the tokens whose log-prob is the Stage 3 metric M: the step's action and its
terminating token (the last len(action) + 1 sampled tokens), not the think block.

Writes `/scratch/cowifornia/nd-rl/circuits/stage3/seqs_L3.npz` (tokens, offsets, metric, prob,
cat (0 G, 1 S), attempt, step) and `seqs_L3.json` (the problems and per-problem counts).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np

from .common import ART, HERE, assert_gpu_free, pt_now, registry

OUT = Path(os.environ.get("CIRCUITS_STAGE3", str(ART / "stage3")))


def problem_list(n_s: int) -> list[dict]:
    probs = json.loads((HERE / "stage2_problems.json").read_text())
    s_idx = sorted(random.Random(0).sample(range(len(probs["S"])), n_s))
    return [dict(p, cat="G") for p in probs["G"]] + [dict(probs["S"][i], cat="S") for i in s_idx]


def run(a: argparse.Namespace) -> None:
    from code.thinking.ckpt import run_env
    from code.thinking.sample import judge, think_generate

    from .common import load_model

    run_env(ART / "runenv")
    model, tok, _ = load_model(registry()[("L3", 0)].path, a.device)
    recs = problem_list(a.nS)
    ks = [a.kG if r["cat"] == "G" else a.kS for r in recs]
    owner = [i for i, k in enumerate(ks) for _ in range(k)]
    prompts = [recs[i]["prompt"] for i in owner]
    t0 = time.time()
    res = think_generate(
        model,
        tok,
        prompts,
        arm="plain",
        budgets=64,
        temperature=0.8,
        max_action=512,
        max_steps=96,
        max_rej=10,
        batch=a.batch,
        seed=a.seed,
        keep_trace=True,
    )
    res = judge(tok, prompts, res, None)
    print(
        f"{pt_now()} sampled and judged {len(res)} attempts in {time.time() - t0:.0f}s", flush=True
    )
    toks: list[np.ndarray] = []
    rows: dict[str, list[int]] = {k: [] for k in ("prob", "cat", "attempt", "step")}
    metric: list[np.ndarray] = []
    kept = [0] * len(recs)
    n_ok = [0] * len(recs)
    bad = 0
    for i, r in enumerate(res):
        p = owner[i]
        if not r["ok"]:
            continue
        n_ok[p] += 1
        if kept[p] >= a.keep:
            continue
        kept[p] += 1
        for j, e in enumerate(r["trace"] or []):
            assert not e.get("rejected"), "an accepted attempt has a rejected step"
            pr, g, act = list(e["prompt"]), list(e["gen"]), list(e["action"])
            n_m = len(act) + 1
            if g[-n_m:-1] != act:  # the action must be the sampled tail before its terminator
                bad += 1
            s = np.array(pr + g, dtype=np.uint8)
            m = np.zeros(len(s), dtype=np.uint8)
            m[len(s) - n_m :] = 1
            toks.append(s)
            metric.append(m)
            rows["prob"].append(p)
            rows["cat"].append(0 if recs[p]["cat"] == "G" else 1)
            rows["attempt"].append(i)
            rows["step"].append(j)
    off = np.zeros(len(toks) + 1, dtype=np.int64)
    off[1:] = np.cumsum([len(s) for s in toks])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT / "seqs_L3.npz",
        tokens=np.concatenate(toks),
        offsets=off,
        metric=np.concatenate(metric),
        **{k: np.array(v, dtype=np.int32) for k, v in rows.items()},
    )
    info = dict(
        problems=[dict(name=r["name"], set=r["set"], cat=r["cat"]) for r in recs],
        k=ks,
        n_ok=n_ok,
        kept=kept,
        action_mismatch=bad,
        sequences=len(toks),
        tokens=int(off[-1]),
        metric_tokens=int(sum(int(m.sum()) for m in metric)),
        args=vars(a),
    )
    (OUT / "seqs_L3.json").write_text(json.dumps(info, indent=1))
    for cat in ("G", "S"):
        idx = [i for i, r in enumerate(recs) if r["cat"] == cat]
        print(
            f"{pt_now()} {cat}: {len(idx)} problems, solve "
            f"{100 * np.mean([n_ok[i] / ks[i] for i in idx]):.1f}%, "
            f"{sum(kept[i] > 0 for i in idx)} with a kept attempt, "
            f"{sum(kept[i] for i in idx)} kept",
            flush=True,
        )
    print(
        f"{pt_now()} {len(toks)} sequences, {off[-1] / 1e3:.0f}k tokens, action mismatches {bad}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--device", default="cuda", help="cpu: smoke tests only")
    ap.add_argument("--kG", type=int, default=8)
    ap.add_argument("--kS", type=int, default=4)
    ap.add_argument("--nS", type=int, default=80)
    ap.add_argument("--keep", type=int, default=2)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args(argv)
    if a.device == "cuda":
        assert a.gpu is not None, "--gpu"
        assert_gpu_free(a.gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    print(f"{pt_now()} start {vars(a)}", flush=True)
    run(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
