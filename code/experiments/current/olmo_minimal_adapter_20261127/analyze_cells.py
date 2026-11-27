# ruff: noqa: E501, SIM115
"""Pass-rate readout per sampled cell (SERRANO side) -> v2/cells_summary.{csv,json} and v2/cells_contrasts.{csv,json}.

Cells: runs/<ckpt>/sample/<batch>/<cell>/{samples.jsonl.gz, graded.jsonl} (adapter cells) and
runs/<ckpt>/plain_<name>/{samples.jsonl.gz, graded.jsonl} (no adapter: the prompt control; cell id "plain:<name>", source
"plain", no cell.json). Graded by Dan's strict IFBench checker, 48 reach prompts x 8 draws.

PRIMARY (design discourse, Turns 2-3): G = (cell - ckpt plain) / (reference - ckpt plain), rates = mean over prompts of the
prompt's strict pass share; reference = the final model's plain rate (Dan's stored arm B for s2300, 32 draws per prompt),
ckpt plain = the checkpoint's plain rate (analyze_arms.b_cells; at dpo his stored plain draws, 0 % on these prompts).
Interval: two-level paired bootstrap, 10,000 reps, fixed seed: resample the prompts (one index per rep, shared by the cell,
the reference and the checkpoint plain), then within every resampled prompt resample that series' draws with replacement
(for 0/1 draws this is Binomial(n, p_hat), done that way; the checkpoint plain is resampled too, a no-op at dpo where it is all 0).
"reaches" = lower bound of G >= 0.80 (steers the search only). Also: the cell rate with its two-level interval (rate2_lo /
rate2_hi), prompts with >= 1 pass in 8 (solved_pct).
SECONDARY (kept from the first version): strict_pct with the prompt-only bootstrap interval (lo95 / hi95, as the old
analyze_arms), diff_final / diff_ckpt (paired, prompts resampled), reaches_interval (interval includes or lies above the
final plain), reaches_strict (lower bound of the paired difference vs final > -10 points), gap_share (prompt-only bootstrap).
Contrasts (cells_contrasts): paired difference of two cells' rates on the same prompts, points, two-level bootstrap; default
list own:A16:64 - unsel:A16:64, final:A16:64 - own:A16:64, and every adapter cell minus the prompt control (plain:<--control>);
--contrasts "a-b,c-d" replaces the first two (a token is source:family:n, matched on the cell id <ckpt>__<source>__<fam>__n<n>,
or a full cell id; "unsel" matches any source starting with it).
Instruction-type coverage: for every reach prompt, are all / some / none of its instruction ids among those of the cell's n training
prompts (pool train.jsonl, nested prefix by idx; Dan's problems.jsonl for the reach prompts)? rate_cov_all / some / none and the prompt
counts. (Computed here from the pool; types_table.json is not read.) Plain cells have no training prompts: no split.
Length: median response tokens, median thinking tokens (before the `</` `think` marker, ids 524 27963; draws with the marker), median
answer words, share of draws cut at the length limit; tokens in training: tokens_one_pass (the n examples), mdl_tokens (cell.json),
tokens_seen_est = tokens_one_pass x selected step / n (an estimate: the cycling order is not read).
Usage: python analyze_cells.py [--runs .../v2/runs] [--out .../v2/cells_summary] [--control answeronly] [--contrasts ...]
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "olmo_elicitation_20261125"
sys.dont_write_bytecode = True  # never write __pycache__ into the old folder
if str(OLD) not in sys.path:
    sys.path.insert(0, str(OLD))
import analyze_arms as AA  # noqa: E402
import dan_curve as dc  # noqa: E402

V2 = Path("/scratch/cowifornia/nd-rl/olmo_elicit/v2")
FINAL = "s2300"
REPS = 10000
SEED = 20261127
G_REACH = 0.80
STRICT_MARGIN_PTS = 10.0
THINK_MARK = (524, 27963)  # `</` `think`
DEFAULT_CONTRASTS = ["own:A16:64-unsel:A16:64", "final:A16:64-own:A16:64"]
KEYS = [
    "ckpt", "batch", "cell", "family", "P", "n", "source", "lr", "step", "did_not_train", "lora_r", "gate",
    "mdl_nats", "edl_bits_per_trained_weight", "tokens_one_pass", "mdl_tokens", "tokens_seen_est",
    "n_prompts", "n_draws", "complete", "strict_pct", "rate2_lo", "rate2_hi", "solved_pct",
    "final_pct", "ckpt_pct", "g", "g_lo", "g_hi", "reaches",
    "med_resp_tokens", "med_think_tokens", "ans_words_median", "length_cut_share",
    "rate_cov_all", "n_cov_all", "rate_cov_some", "n_cov_some", "rate_cov_none", "n_cov_none",
    "lo95", "hi95", "final_lo95", "final_hi95", "ckpt_lo95", "ckpt_hi95", "diff_final", "diff_final_lo", "diff_final_hi",
    "diff_ckpt", "diff_ckpt_lo", "diff_ckpt_hi", "reaches_interval", "reaches_strict", "gap_share", "gap_lo", "gap_hi",
]  # fmt: skip
CONTRAST_KEYS = ["ckpt", "name", "a", "b", "n_prompts", "diff", "lo", "hi", "excludes_zero"]


# ----------------------------------------------------------------------------------------- bootstrap
def counts(cells: dict, pids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per prompt: successes and draws."""
    s = np.array([sum(x for x, _ in cells[p]) for p in pids], dtype=float)
    n = np.array([len(cells[p]) for p in pids], dtype=float)
    return s, n


def boot_means(
    series: list[tuple[np.ndarray, np.ndarray]], reps: int = REPS, seed: int = SEED
) -> np.ndarray:
    """Two-level paired bootstrap. series: [(successes, draws) per prompt]. Returns (len(series), reps) means over prompts:
    one prompt index matrix shared by all series, then each series' draws resampled within the chosen prompts (0/1 draws:
    Binomial(n, s/n)). The first series' stream does not depend on the later ones."""
    rng = np.random.default_rng(seed)
    n_p = len(series[0][0])
    idx = rng.integers(0, n_p, size=(reps, n_p))
    out = []
    for s, n in series:
        ni, pi = n[idx].astype(int), (s / n)[idx]
        out.append((rng.binomial(ni, pi) / ni).mean(axis=1))
    return np.array(out)


def pct_ci(x: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def rates(cells: dict, pids: list[str]) -> np.ndarray:
    return np.array([np.mean([s for s, _ in cells[p]]) for p in pids], dtype=float)


def contrast(a: dict, b: dict, reps: int = REPS) -> dict | None:
    """Rate of a minus rate of b on the prompts both have, points, two-level paired bootstrap."""
    pids = sorted(p for p in a if p in b)
    if not pids:
        return None
    m = boot_means([counts(a, pids), counts(b, pids)], reps)
    lo, hi = pct_ci(m[0] - m[1])
    pt = float(rates(a, pids).mean() - rates(b, pids).mean())
    return {
        "n_prompts": len(pids),
        "diff": round(100 * pt, 2),
        "lo": round(100 * lo, 2),
        "hi": round(100 * hi, 2),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def analyze_cell(cell: dict, final: dict, ckpt: dict, n_set: int = 48, reps: int = REPS) -> dict:
    """cell / final / ckpt: pid -> [(strict, answer words | None), ...]. Returns the readout columns (percent where named pct)."""
    keys = ("g", "g_lo", "g_hi", "reaches", "rate2_lo", "rate2_hi", "diff_final", "diff_final_lo", "diff_final_hi", "diff_ckpt", "diff_ckpt_lo", "diff_ckpt_hi", "reaches_interval", "reaches_strict", "gap_share", "gap_lo", "gap_hi", "final_pct", "ckpt_pct", "final_lo95", "final_hi95", "ckpt_lo95", "ckpt_hi95")  # fmt: skip
    out = {k: None for k in keys}
    base = AA.summarize(
        cell, 8, n_set, reps
    )  # strict_pct, lo95 / hi95 (prompt-only bootstrap), solved, answer words
    out.update(base)
    pids = sorted(p for p in cell if p in final and p in ckpt)
    if not pids:
        return out
    c, f, k = rates(cell, pids), rates(final, pids), rates(ckpt, pids)
    # prompt-only bootstrap (the old readout): same index matrix as dc.boot_mean_ci
    idx = np.random.default_rng(0).integers(0, len(pids), size=(reps, len(pids)))
    mc, mf, mk = c[idx].mean(1), f[idx].mean(1), k[idx].mean(1)
    out["final_pct"], out["ckpt_pct"] = (
        round(100 * float(f.mean()), 2),
        round(100 * float(k.mean()), 2),
    )
    out["final_lo95"], out["final_hi95"] = (round(100 * v, 2) for v in pct_ci(mf))
    out["ckpt_lo95"], out["ckpt_hi95"] = (round(100 * v, 2) for v in pct_ci(mk))
    out["diff_final"] = round(100 * float(c.mean() - f.mean()), 2)
    out["diff_final_lo"], out["diff_final_hi"] = (round(100 * v, 2) for v in pct_ci(mc - mf))
    out["diff_ckpt"] = round(100 * float(c.mean() - k.mean()), 2)
    out["diff_ckpt_lo"], out["diff_ckpt_hi"] = (round(100 * v, 2) for v in pct_ci(mc - mk))
    out["reaches_interval"] = bool(base["hi95"] >= 100 * float(f.mean()))
    out["reaches_strict"] = bool(out["diff_final_lo"] > -STRICT_MARGIN_PTS)
    plain_gap = float(f.mean() - k.mean())
    gap = mf - mk
    ok = gap >= 0.01
    if plain_gap >= 0.01 and ok.mean() > 0.5:
        out["gap_share"] = round(float((c.mean() - k.mean()) / plain_gap), 4)
        lo, hi = pct_ci(((mc - mk)[ok]) / gap[ok])
        out["gap_lo"], out["gap_hi"] = round(lo, 4), round(hi, 4)
    # primary: two-level paired bootstrap of G
    bc, bf, bk = boot_means([counts(cell, pids), counts(final, pids), counts(ckpt, pids)], reps)
    out["rate2_lo"], out["rate2_hi"] = (round(100 * v, 2) for v in pct_ci(bc))
    gap2 = bf - bk
    ok2 = gap2 >= 0.01
    if plain_gap >= 0.01 and ok2.mean() > 0.5:
        out["g"] = round(float((c.mean() - k.mean()) / plain_gap), 4)
        lo, hi = pct_ci((bc - bk)[ok2] / gap2[ok2])
        out["g_lo"], out["g_hi"] = round(lo, 4), round(hi, 4)
        out["reaches"] = bool(out["g_lo"] >= G_REACH)
    return out


# ----------------------------------------------------------------------------------------- loading
def think_tokens(ids_b64: str) -> int | None:
    """Tokens before the last `</` `think` marker (None: no marker)."""
    ids = np.frombuffer(base64.b64decode(ids_b64), dtype=np.int32)
    hit = np.flatnonzero((ids[:-1] == THINK_MARK[0]) & (ids[1:] == THINK_MARK[1]))
    return int(hit[-1]) if len(hit) else None


def load_detail(samples: Path, graded: Path, reach: list[str]) -> dict:
    """pid -> [ {strict, words, n_tok, think, length_cut} ] in (chunk, i) order, for the graded draws of the reach prompts."""
    srow = {(r["pid"], int(r["chunk"]), int(r["i"])): r for r in AA.jl(samples)}
    out: dict[str, list] = {}
    for r in sorted(AA.jl(graded), key=lambda r: (r["pid"], int(r["chunk"]), int(r["i"]))):
        key = (r["pid"], int(r["chunk"]), int(r["i"]))
        if r["pid"] not in reach or key not in srow:
            continue
        s = srow[key]
        out.setdefault(r["pid"], []).append(
            {
                "strict": int(bool(r["strict"])),
                "words": AA.answer_words(s["text"]),
                "n_tok": s.get("n_tok"),
                "think": think_tokens(s["ids"]) if s.get("ids") else None,
                "length_cut": s.get("finish") == "length",
            }
        )
    return out


def to_cells(detail: dict) -> dict:
    return {p: [(d["strict"], d["words"]) for d in v] for p, v in detail.items()}


def load_cell_draws(samples: Path, graded: Path, reach: list[str]) -> dict:
    return AA.h_cells([(samples, graded)], reach)


def length_stats(detail: dict) -> dict:
    ds = [d for v in detail.values() for d in v]

    def med(xs):
        return float(np.median(xs)) if xs else None

    return {
        "med_resp_tokens": med([d["n_tok"] for d in ds if d["n_tok"] is not None]),
        "med_think_tokens": med([d["think"] for d in ds if d["think"] is not None]),
        "length_cut_share": round(sum(d["length_cut"] for d in ds) / len(ds), 4) if ds else None,
    }


def read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


_POOL_CACHE: dict = {}


def pool_train(pool: Path, n: int) -> list[dict]:
    """The first n training rows (nested order = idx) of a pool's train.jsonl."""
    key = str(pool)
    if key not in _POOL_CACHE:
        rows = []
        try:
            for ln in open(pool / "train.jsonl"):
                if ln.strip():
                    r = json.loads(ln)
                    rows.append(
                        {k: r.get(k) for k in ("idx", "pid", "n_tok", "instruction_id_list")}
                    )
        except OSError:
            pass
        _POOL_CACHE[key] = sorted(rows, key=lambda r: r["idx"])
    return _POOL_CACHE[key][:n]


def reach_ids(problems: Path | None = None) -> dict[str, set]:
    p = problems or dc.DAN / "olmo_ifomega" / "problems.jsonl"
    out = {}
    for ln in open(p):
        r = json.loads(ln)
        if r.get("instruction_id_list"):
            out[r["pid"]] = set(r["instruction_id_list"])
    return out


def coverage_split(cells: dict, train_ids: set, rids: dict[str, set]) -> dict:
    """Rate over the reach prompts by whether all / some / none of their instruction ids occur in the training prompts."""
    groups: dict[str, list] = {"all": [], "some": [], "none": []}
    for p, v in cells.items():
        ids = rids.get(p)
        if not ids:
            continue
        seen = len(ids & train_ids)
        groups["all" if seen == len(ids) else "some" if seen else "none"].append(
            float(np.mean([s for s, _ in v]))
        )
    out = {}
    for g, xs in groups.items():
        out[f"rate_cov_{g}"] = round(100 * float(np.mean(xs)), 2) if xs else None
        out[f"n_cov_{g}"] = len(xs)
    return out


def cell_meta(runs: Path, ckpt: str, cell: str, root: Path = V2) -> tuple[dict, set | None]:
    cj = read_json(runs / ckpt / "train" / "cells" / cell / "cell.json")
    sel = cj.get("selected") or {}
    train = None
    meta = {
        "family": cj.get("family"),
        "P": cj.get("P"),
        "n": cj.get("n"),
        "source": cj.get("source"),
        "lr": sel.get("lr"),
        "step": sel.get("step"),
        "did_not_train": cj.get("did_not_train"),
        "lora_r": cj.get("lora_r"),
        "mdl_nats": cj.get("mdl_nats"),
        "mdl_tokens": cj.get("mdl_tokens"),
        "edl_bits_per_trained_weight": cj.get("edl_tok_bits_per_trained_weight"),
    }
    if cj.get("n"):
        pool = Path(cj["pool"]) if cj.get("pool") else root / "pools" / ckpt / str(cj.get("source"))
        # cell.json records the pod's path; use serrano's copy of the same pool
        if not pool.exists():
            pool = root / "pools" / ckpt / str(cj.get("source"))
        rows = pool_train(pool, int(cj["n"]))
        if rows:
            train = set().union(*(set(r["instruction_id_list"] or []) for r in rows))
            one = sum(r["n_tok"] or 0 for r in rows)
            meta["tokens_one_pass"] = one
            if sel.get("step"):
                meta["tokens_seen_est"] = round(one * sel["step"] / len(rows))
    return meta, train


def references(ckpt: str, reach: list[str]) -> tuple[dict, dict]:
    return AA.b_cells(FINAL, reach), AA.b_cells(ckpt, reach)


# ----------------------------------------------------------------------------------------- tree
def discover(runs: Path, ck: str):
    """Yield (cell id, batch, directory, is_plain) for every sampled cell of a checkpoint."""
    for bdir in sorted((runs / ck / "sample").glob("*/")):
        for cdir in sorted(p for p in bdir.iterdir() if p.is_dir()):
            yield cdir.name, bdir.name, cdir, False
    for pdir in sorted((runs / ck).glob("plain_*/")):
        yield f"plain:{pdir.name[6:]}", pdir.name, pdir, True


def resolve(token: str, ids: list[str]) -> str | None:
    if token in ids:
        return token
    parts = token.split(":")
    if len(parts) == 3:
        src, fam, n = parts
        for c in ids:
            q = c.split("__")
            if len(q) == 4 and q[1].startswith(src) and q[2] == fam and q[3] == f"n{n}":
                return c
    return None


def analyze(
    runs: Path,
    ckpts: list[str] | None = None,
    reps: int = REPS,
    reach: list[str] | None = None,
    control: str = "answeronly",
    contrasts: list[str] | None = None,
    root: Path = V2,
    rids: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    reach = reach or dc.reach()
    rids = rids if rids is not None else reach_ids()
    rows, crows = [], []
    names = ckpts or sorted(
        p.name for p in runs.iterdir() if (p / "sample").is_dir() or list(p.glob("plain_*"))
    )
    for ck in names:
        final, plain = references(ck, reach)
        allc: dict[str, dict] = {}
        for cid, batch, cdir, is_plain in discover(runs, ck):
            gp, sp = cdir / "graded.jsonl", cdir / "samples.jsonl.gz"
            if not gp.exists() or not sp.exists():
                continue
            detail = load_detail(sp, gp, reach)
            cells = to_cells(detail)
            if not cells:
                continue
            allc[cid] = cells
            gate = (
                {}
                if is_plain
                else read_json(runs / ck / "gate" / f"{batch}.json").get("verdicts", {})
            )
            meta, train = (
                ({"source": "plain", "family": None}, None)
                if is_plain
                else cell_meta(runs, ck, cid, root)
            )
            r = analyze_cell(cells, final, plain, len(reach), reps)
            rows.append(
                {
                    "ckpt": ck,
                    "batch": batch,
                    "cell": cid,
                    **meta,
                    "gate": gate.get(cid),
                    **r,
                    **length_stats(detail),
                    **(coverage_split(cells, train, rids) if train is not None else {}),
                    "complete": r["n_draws"] == 8 * len(reach),
                }
            )
        ids = list(allc)
        pairs = []
        for spec in contrasts if contrasts is not None else DEFAULT_CONTRASTS:
            if "-" in spec:
                a, b = (resolve(t, ids) for t in spec.split("-", 1))
                if a and b:
                    pairs.append((spec, a, b))
        if contrasts is None and f"plain:{control}" in allc:
            pairs += [
                (f"{c}-plain:{control}", c, f"plain:{control}")
                for c in ids
                if not c.startswith("plain:")
            ]
        for spec, a, b in pairs:
            d = contrast(allc[a], allc[b], reps)
            if d:
                crows.append({"ckpt": ck, "name": spec, "a": a, "b": b, **d})
    return rows, crows


def clean(v):
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, np.generic):
        return v.item()
    return v


def write_table(rows: list[dict], keys: list[str], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out}.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: "" if clean(r.get(k)) is None else clean(r.get(k)) for k in keys})
    Path(f"{out}.json").write_text(
        json.dumps([{k: clean(r.get(k)) for k in keys} for r in rows], indent=1) + "\n"
    )


def write(rows: list[dict], out: Path) -> None:
    write_table(rows, KEYS, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(V2 / "runs"))
    ap.add_argument("--out", default=str(V2 / "cells_summary"))
    ap.add_argument("--contrasts-out", default=str(V2 / "cells_contrasts"))
    ap.add_argument(
        "--ckpt", default=None, help="comma list; default every checkpoint with samples"
    )
    ap.add_argument(
        "--control",
        default="answeronly",
        help="name of the plain prompt-control cell (plain:<name>)",
    )
    ap.add_argument(
        "--contrasts",
        default=None,
        help="comma list 'a-b,c-d' (tokens source:family:n or cell ids); replaces the defaults",
    )
    ap.add_argument("--reps", type=int, default=REPS)
    a = ap.parse_args()
    rows, crows = analyze(
        Path(a.runs),
        a.ckpt.split(",") if a.ckpt else None,
        a.reps,
        control=a.control,
        contrasts=a.contrasts.split(",") if a.contrasts else None,
    )
    write(rows, Path(a.out))
    write_table(crows, CONTRAST_KEYS, Path(a.contrasts_out))
    print(
        f"wrote {a.out}.csv / .json: {len(rows)} cells; {a.contrasts_out}: {len(crows)} contrasts"
    )
    for r in rows:
        print(
            f"{r['ckpt']:5} {r['cell']:30} {r['strict_pct']:6}% G {r['g']} [{r['g_lo']}, {r['g_hi']}] reaches={r['reaches']} solved {r['solved_pct']}% gate={r['gate']} draws={r['n_draws']}"
        )
    for c in crows:
        print(
            f"  {c['name']}: {c['diff']} pts [{c['lo']}, {c['hi']}] excludes0={c['excludes_zero']}"
        )


if __name__ == "__main__":
    main()
