"""Stage 2: causal tests with crosscoder patching during sampling (seed 0, base vs L3 end).

    python -m code.experiments.current.circuit_diffing_20261101.stage2 --gpu 0 \
        --cc DIR --conditions rl_clean,rl_ablate_rlonly,rl_ablate_rand0 [--k 32]

Conditions (the latent sets come from the Stage 1 classification of the crosscoder in DIR):
  rl_clean            the RL model alone
  rl_recon            the RL model with the layer replaced by its crosscoder reconstruction
  rl_ablate_rlonly    the RL model with every strict RL-only latent ablated (patching.ablate)
  rl_ablate_rand{i}   the same number of shared latents, drawn at random within the RL-only
                      latents' deciles of RL energy (seed i); the matched control
  base_clean          the base alone
  base_rescale        the base with every shared latent rescaled to the RL model's strength
  base_inject         the base with the strict RL-only latents' RL decoders added
  base_replace        the base with the layer replaced by the RL model's residual (positive control)
  base_transplant     the base with every live latent switched to its RL version (norm, direction)
  rl_revert           the RL model with every live latent switched back to its base version
  rl_unscale          the RL model with shared latents scaled back to the base's strength
Problems: `stage2_problems.json` (G: RL-gained, S: shared). Every attempt is the read setting of
the reads (plain arm, B64, temperature 0.8, max_action 512, max_steps 96), judged by Lean. Writes
`/scratch/cowifornia/nd-rl/circuits/stage2/<cc dir>/<condition>.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .common import ART, HERE, assert_gpu_free, pt_now, registry

OUT = ART / "stage2"


def latent_sets(cc_dir: Path, n_rand: int = 3) -> dict[str, np.ndarray]:
    import torch

    from .stage1_analyze import arrays, classify

    L = arrays(torch.load(cc_dir / "latents.pt", weights_only=False))
    c = classify(L)
    rl = np.flatnonzero(c["rl"])
    shared = np.flatnonzero(c["shared"])
    out = {"rlonly": rl, "shared": shared, "live": np.flatnonzero(c["live"])}
    e = L["energy_b_b_roll"]
    if len(rl):
        edges = np.quantile(e[rl], np.linspace(0, 1, 11))
        edges[0], edges[-1] = -np.inf, np.inf
        for i in range(n_rand):
            rng = np.random.default_rng(i)
            pick: list[int] = []
            for b in range(10):
                need = int(((e[rl] >= edges[b]) & (e[rl] < edges[b + 1])).sum())
                pool = shared[(e[shared] >= edges[b]) & (e[shared] < edges[b + 1])]
                pool = np.setdiff1d(pool, pick)
                if len(pool) < need:  # widen to the nearest energies
                    order = np.argsort(
                        np.abs(
                            np.log(e[shared] + 1e-30)
                            - np.log(
                                np.median(e[rl][(e[rl] >= edges[b]) & (e[rl] < edges[b + 1])])
                                + 1e-30
                            )
                        )
                    )
                    pool = np.setdiff1d(shared[order[: 4 * need + 10]], pick)
                pick += rng.choice(pool, size=need, replace=False).tolist()
            out[f"rand{i}"] = np.array(sorted(pick))
    return out


def run(a: argparse.Namespace) -> None:
    import torch

    from code.thinking.ckpt import run_env
    from code.thinking.sample import judge, think_generate

    from .common import load_model
    from .patching import PairPatched, load_cc

    run_env(ART / "runenv")
    reg = registry()
    base, _, _ = load_model(reg[("base", 0)].path, a.device)
    rl, tok, _ = load_model(reg[(a.b, 0)].path, a.device)
    cc_dir = Path(a.cc)
    cc, meta = load_cc(cc_dir, a.device)
    assert meta["b"] == f"{a.b}:0" and meta["a"] == "base:0", meta
    sets = latent_sets(cc_dir)
    probs = json.loads((HERE / "stage2_problems.json").read_text())
    recs = [dict(p, cat=cat) for cat in ("G", "S") for p in probs[cat]]
    if a.limit:
        recs = recs[: a.limit] + [r for r in recs if r["cat"] == "S"][: a.limit]
    prompts = [r["prompt"] for r in recs for _ in range(a.k)]
    out_dir = Path(a.out) if a.out else OUT / cc_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for cond in a.conditions.split(","):
        path = out_dir / f"{cond}.json"
        if path.exists() and not a.force:
            print(f"skip {cond}: exists", flush=True)
            continue
        drv, _, what = cond.partition("_")
        lat = None
        if what == "clean":
            model = rl if drv == "rl" else base
        else:
            mode, _, which = what.partition("_")
            if mode in ("rescale", "unscale"):
                lat = torch.from_numpy(sets["shared"])
            elif mode in ("transplant", "revert"):
                lat = torch.from_numpy(sets["live"])
            elif which:
                lat = torch.from_numpy(sets[which])
            model = PairPatched(base, rl, cc, meta, mode, lat)
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
            keep_trace=False,
        )
        t_s = time.time() - t0
        res = judge(tok, prompts, res, None)
        rows = []
        for i, r in enumerate(recs):
            att = res[i * a.k : (i + 1) * a.k]
            rows.append(
                dict(
                    name=r["name"],
                    set=r["set"],
                    cat=r["cat"],
                    n_ok=sum(bool(x["ok"]) for x in att),
                    n_done=sum(x["status"] == "done" for x in att),
                    k=a.k,
                    tokens=sum(x["tokens"] for x in att),
                )
            )
        st: dict[str, Any] = dict(getattr(model, "stats", {}))
        summ = {
            cat: 100 * np.mean([r["n_ok"] / r["k"] for r in rows if r["cat"] == cat])
            for cat in ("G", "S")
        }
        doc = dict(
            condition=cond,
            cc=str(cc_dir),
            b=a.b,
            n_latents=None if lat is None else int(len(lat)),
            k=a.k,
            seed=a.seed,
            sample_secs=t_s,
            stats=st,
            solve_pct=summ,
            rows=rows,
            utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        path.write_text(json.dumps(doc, indent=1))
        rel = st.get("edit_norm", 0) / max(1e-9, st.get("h_norm", 0)) if st else 0.0
        print(
            f"{pt_now()} {cond}: G {summ['G']:.1f}%, S {summ['S']:.1f}% "
            f"(latents {doc['n_latents']}, edit/h {rel:.3f}, {t_s:.0f}s)",
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--device", default="cuda", help="cpu: smoke tests only")
    ap.add_argument("--limit", type=int, default=0, help="smoke tests: problems per category")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cc", required=True)
    ap.add_argument("--b", default="L3")
    ap.add_argument("--conditions", required=True)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--force", action="store_true")
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
