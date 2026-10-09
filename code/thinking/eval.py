"""pass@k over a grid of think budgets, plain and guided, with full token accounting.

    python -m code.thinking.eval --ckpt DIR/model.pt --sets a.jsonl,b.jsonl --out EVALDIR \
        --k 64 --budgets 0,16,64,256 --arms plain,logical

For each (arm, budget B): k attempts per theorem (`sample.think_generate`), Lean on every finished
proof (`lean.gate`, the literal text).  Writes, in EVALDIR:
  rows_<arm>_B<B>.jsonl  one row per theorem: file, name, prompt, min_lines and per-attempt
                         arrays ok, status, steps (accepted steps), fail_step (accepted steps
                         before the failing one; null if done), tokens / think / close / action /
                         rejected / prefill / draws / rej, budget; plus totals
  summary.json           per (arm, B) and per file: solved, pass@k for k = 1, 2, 4, ... (unbiased
                         estimator, as % of the set), attempt status counts, and tokens per theorem
                         and in total (think, `</think>`, action, rejected redraws, prefill), so
                         curves can be read at matched sampled tokens.
A checkpoint without think tokens (T0 / `none` format, or a `lean_staten` checkpoint) runs once with
B = 0 and no think block.  ND_OFFLINE=1 is forced; logs and registry rows stay in EVALDIR.

The prompt layout follows the checkpoint's `think_format` (`sample.step_prompt`): per step, R
("root": B caps ROOT, the one block at step 0; B = 0 is `<think> </think>` there) or M ("memo": B
caps each memo; B = 0, every memo empty).  `--force_mode line|exact` (default line: a block over B
ends at its line's end, hard cap B + 64).  `--intervene empty|corrupt|swap|filler` (plain arm)
replaces each sampled block before the action is decoded (`sample.intervene_thinks`); the cells
keep their `<arm>_B<B>` names, so use one EVALDIR per intervention -- every row and the summary
record it.  States are rendered in the checkpoint's state format (`stateview`, set by `load_ckpt`;
the summary's `state_format`).

`--budgets auto` (a `lean_thinke` checkpoint, plain arm): the model picks its level at every step
among `--auto_levels` (default 0,64,256; `sample.choose_levels`).  The cell is `<arm>_Bauto`
(rows_plain_Bauto.jsonl, readable by `stats.load_arm`); its rows add per attempt `levels` (the level
of every draw) and `choice` (sampled effort tokens, already in `tokens`), and the summary the shares
of the picked levels.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

from . import stateview
from .ckpt import load_ckpt, run_env
from .dan import bare, load
from .effort import has_effort
from .sample import (
    ARMS,
    AUTO,
    AUTO_LEVELS,
    FORCE_MODES,
    INTERVENTIONS,
    judge,
    sample_format,
    think_generate,
)
from .tok import has_think

D = load()
COUNTERS = ("tokens", "think", "close", "action", "rejected", "prefill", "draws", "rej", "forced")


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n attempts with c successes (Chen et al. 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def ks_upto(n: int) -> list[int]:
    out, k = [], 1
    while k <= n:
        out.append(k)
        k *= 2
    if out[-1] != n:
        out.append(n)
    return out


def summarize(rows: list[dict[str, Any]], k: int) -> dict[str, Any]:
    by_file: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        by_file[r["file"]].append(r)
    by_file["all"] = rows
    out = {}
    for f, rs in by_file.items():
        n = len(rs)
        tot = {c: sum(r["total"][c] for r in rs) for c in COUNTERS}
        pk = {str(j): 100.0 * sum(pass_at_k(k, r["n_ok"], j) for r in rs) / n for j in ks_upto(k)}
        status = collections.Counter(s for r in rs for s in r["status"])
        out[f] = {
            "n": n,
            "solved": sum(r["n_ok"] > 0 for r in rs),
            "solved_pct": 100.0 * sum(r["n_ok"] > 0 for r in rs) / n,
            "pass_at_k_pct": pk,
            "tokens_total": tot,
            "tokens_per_theorem": {c: v / n for c, v in tot.items()},
            "tokens_per_attempt": {c: v / (n * k) for c, v in tot.items()},
            "status": dict(status.most_common()),
        }
    return out


def rows_for(
    recs: list[dict[str, Any]], res: list[dict[str, Any]], k: int, intervene: str = "none"
) -> list[dict[str, Any]]:
    rows = []
    for j, r in enumerate(recs):
        att = res[j * k : (j + 1) * k]
        row = dict(
            file=r["file"],
            name=r["name"],
            prompt=r["prompt"],
            min_lines=r.get("min_lines"),
            k=k,
            n_ok=sum(bool(x["ok"]) for x in att),
            ok=[int(bool(x["ok"])) for x in att],
            status=[x["status"] for x in att],
            steps=[x["steps"] for x in att],
            fail_step=[x["fail_step"] for x in att],
            budget=[x["budget"] for x in att],
            intervene=intervene,
            **{c: [x[c] for x in att] for c in COUNTERS},
            total={c: sum(x[c] for x in att) for c in COUNTERS},
            causes=[x["causes"] for x in att if x["causes"]],
            accepted=dict(collections.Counter(x["text"] for x in att if x["ok"])),
        )
        if att and "levels" in att[0]:  # auto effort: the level picked at every draw
            row["levels"] = [x["levels"] for x in att]
            row["choice"] = [x["choice"] for x in att]
        rows.append(row)
    return rows


def level_shares(res: list[dict[str, Any]]) -> dict[str, float]:
    """Auto effort: % of draws at each picked level (all attempts, rejected draws included)."""
    c = collections.Counter(b for x in res for b in x.get("levels", []))
    n = sum(c.values())
    return {str(b): 100.0 * c[b] / n for b in sorted(c)} if n else {}


def parse_budgets(spec: str, think: bool) -> list[int | str]:
    """--budgets -> the cells' budgets: ints, or 'auto' (auto effort)."""
    if not think:
        return [0]
    out: list[int | str] = []
    for b in spec.split(","):
        b = b.strip()
        out.append(AUTO if b == AUTO else int(b))
    return out


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt", required=True)
    ap.add_argument(
        "--sets", required=True, help="comma-separated jsonl files with a 'prompt' field"
    )
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--budgets", default="0,16,32,64,128,256")
    ap.add_argument("--arms", default="plain,logical")
    ap.add_argument("--min_think", type=int, default=0)
    ap.add_argument("--think", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--greedy", action="store_true", help="greedy decoding (use with --k 1)")
    ap.add_argument("--max_action", type=int, default=512)
    ap.add_argument("--max_steps", type=int, default=96)
    ap.add_argument("--max_rej", type=int, default=10)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--lean", default=None, help="Lean binary (default $ND_LEAN or ~/.elan/bin/lean)"
    )
    ap.add_argument(
        "--mask_action_think", action="store_true", help="ban think tokens inside actions"
    )
    ap.add_argument(
        "--dump_traces", action="store_true", help="also write accepted attempts' traces"
    )
    ap.add_argument(
        "--force_mode",
        choices=FORCE_MODES,
        default="line",
        help="budget forcing: close at the end of the line past B (cap B + 64), or at exactly B",
    )
    ap.add_argument(
        "--intervene",
        choices=INTERVENTIONS,
        default="none",
        help="replace each sampled think block before the action (plain arm)",
    )
    ap.add_argument(
        "--think_format",
        choices=["auto", "think", "root", "memo"],
        default="auto",
        help="prompt layout (auto: the checkpoint's)",
    )
    ap.add_argument(
        "--wave_tokens",
        type=int,
        default=0,
        help="rows x (prompt + decode room) per decode wave (0: one wave; R: 2^19)",
    )
    ap.add_argument(
        "--auto_levels",
        default=",".join(str(b) for b in AUTO_LEVELS),
        help="--budgets auto: the effort levels the model picks from at every step",
    )
    ap.add_argument("--device", default=None)
    return ap


def think_mode(choice: str, tok: Any, extra: dict[str, Any]) -> bool:
    if choice != "auto":
        return choice == "on"
    return has_think(tok) and extra.get("think_format", "think") != "none"


def run(a: argparse.Namespace) -> dict[str, Any]:
    out = Path(a.out)
    run_env(out)
    with bare():
        D.record.save_config(vars(a), str(out) + "/", role="eval")
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, extra = load_ckpt(a.ckpt, dev)
    think = think_mode(a.think, tok, extra)
    fmt = sample_format(
        model, extra.get("think_format") if a.think_format == "auto" else a.think_format
    )
    if a.intervene != "none" and set(a.arms.split(",")) != {"plain"}:
        raise SystemExit(f"error: --intervene {a.intervene} runs the plain arm only")
    budgets = parse_budgets(a.budgets, think)
    arms = a.arms.split(",")
    assert all(x in ARMS for x in arms), arms
    auto_levels = [int(b) for b in a.auto_levels.split(",") if b.strip()]
    if AUTO in budgets:
        if not has_effort(tok):
            raise SystemExit("error: --budgets auto needs a lean_thinke (effort) checkpoint")
        if set(arms) != {"plain"} or a.intervene != "none" or fmt != "think":
            raise SystemExit("error: --budgets auto: plain arm, per-step format, no intervention")
    recs = D.guided_eval.load_sets(a.sets.split(","))
    if a.limit:
        recs = recs[: a.limit]
    prompts = [r["prompt"] for r in recs for _ in range(a.k)]
    summary: dict[str, Any] = dict(
        ckpt=a.ckpt,
        tok_mode=tok.mode,
        think=think,
        think_format=extra.get("think_format"),
        state_format=stateview.current(),  # the checkpoint's (`load_ckpt` sets it)
        layout=fmt,
        max_len=getattr(model, "cfg", {}).get("max_len"),
        force_mode=a.force_mode,
        intervene=a.intervene,
        n_theorems=len(recs),
        k=a.k,
        config={k: v for k, v in vars(a).items()},
        cells={},
    )
    for arm in arms:
        for B in budgets:
            st: dict[str, Any] = {}
            t0 = time.time()
            auto = B == AUTO
            with D.record.compute(phase="read", arm=f"{arm}_B{B}", ckpt=a.ckpt):
                res = think_generate(
                    model,
                    tok,
                    prompts,
                    arm,
                    0 if isinstance(B, str) else B,
                    0 if isinstance(B, str) else min(a.min_think, B),
                    think,
                    a.temperature,
                    a.greedy,
                    a.max_action,
                    a.max_steps,
                    a.max_rej,
                    a.batch,
                    a.seed,
                    st,
                    keep_trace=a.dump_traces,
                    mask_action_think=a.mask_action_think,
                    think_format=fmt,
                    force_mode=a.force_mode,
                    intervene=a.intervene,
                    wave_tokens=a.wave_tokens,
                    auto_levels=auto_levels if auto else None,
                )
                t_loop = time.time() - t0
                res = judge(tok, prompts, res, a.lean)
            rows = rows_for(recs, res, a.k, a.intervene)
            name = f"{arm}_B{B}"
            with open(out / f"rows_{name}.jsonl", "w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            if a.dump_traces:
                with open(out / f"traces_{name}.jsonl", "w") as f:
                    for j, x in enumerate(res):
                        if x["ok"]:
                            steps = [
                                {
                                    k2: v
                                    if isinstance(v, (bool, int))
                                    else " ".join(tok.itos[i] for i in v)
                                    for k2, v in s.items()
                                }
                                for s in x["trace"]
                            ]
                            f.write(
                                json.dumps(
                                    {"prompt": prompts[j], "base": x["base"], "steps": steps}
                                )
                                + "\n"
                            )
            cell = summarize(rows, a.k)
            cell["intervene"] = a.intervene
            cell["wall_s"] = time.time() - t0
            cell["loop_s"] = t_loop
            cell["stats"] = {
                k: (dict(v) if isinstance(v, collections.Counter) else v) for k, v in st.items()
            }
            if auto:
                cell["auto_levels"] = auto_levels
                cell["level_share_pct"] = level_shares(res)
            summary["cells"][name] = cell
            allc = cell["all"]
            tpa = allc["tokens_per_attempt"]
            shares = ""
            if auto:
                shares = ", levels " + " ".join(
                    f"e{b} {v:.1f}%" for b, v in cell["level_share_pct"].items()
                )
            print(
                f"[{name}] solved {allc['solved']}/{allc['n']} ({allc['solved_pct']:.1f}%), "
                f"pass@1 {allc['pass_at_k_pct']['1']:.1f}%, tokens/attempt "
                f"{tpa['tokens']:.1f} (think {tpa['think']:.1f}){shares}, "
                f"{cell['wall_s']:.0f}s",
                flush=True,
            )
            (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


def main(argv: list[str] | None = None) -> int:
    run(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
