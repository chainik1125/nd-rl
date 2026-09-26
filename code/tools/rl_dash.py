#!/usr/bin/env python3
"""RL run dashboards for the natural-deduction prover: static HTML, inline SVG, no network.

Reads run directories written by the ab3 harness (metrics.jsonl and summary.json; examples.jsonl,
found_targets.jsonl and an evaluation directory when present) and writes one HTML page per
dashboard:

  1_limiting.html     what limits the run: solve rate against the frozen control, group
                      fractions, new proofs, compute and time per phase, loss and grad norm
  2_diversity.html    pass@1/4/32 on the sampling window, distinct proofs per solved target,
                      token entropy
  3_integrity.html    written against pruned length, failure reasons, accepted against rejected
                      length, truncation, example proofs
  4_retention.html    held-out greedy overall and by length, KL to the start model, RL records
                      against retained records
  5_comparison.html   every run side by side, joined with final scores and evaluation pass@1
  6_length_wall.html  attempts long enough for L_true, first-error hazard by line, attempts that
                      end on the conclusion, citation distance, ignition-set log-probability, and
                      the test-time length wall when an evaluation directory exists

Usage (from the repository root):
  python code/tools/rl_dash.py render RUN [RUN ...] --out DIR [--control RUN] [--eval LABEL=DIR]
  python code/tools/rl_dash.py fields RUN [RUN ...]
  python code/tools/rl_dash.py wandb RUN [RUN ...] --project P [--mode dry-run|offline|online]

RUN is a run directory, optionally labelled: `C_s101=/scratch/.../runs/C_s101`. An evaluation
directory is found automatically at `<experiment>/eval/<label>` next to `<experiment>/runs/<label>`
when its meta.json names the run's final.pt.

Fields a run did not log are shown as missing, never as zero. A few are derived exactly from
older fields (pass@1 = accept_rate, all-fail share = 1 - solved / window, truncation = 1 -
qed_rate, ...). The field-coverage table at the foot of every page says which.
"""

from __future__ import annotations

import argparse
import functools
import gzip
import html
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Rec = dict[str, Any]
DASHBOARDS = (
    ("1_limiting", "Limiting factor"),
    ("2_diversity", "Diversity"),
    ("3_integrity", "Reward integrity"),
    ("4_retention", "Retention"),
    ("5_comparison", "Comparison"),
    ("6_length_wall", "Length wall"),
)
N_SLOTS = 6  # categorical slots used (validated palette, both modes)

# ---------------------------------------------------------------- loading


def read_jsonl(path: Path) -> list[Rec]:
    out: list[Rec] = []
    if not path.exists():
        return out
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # a line being written right now
    return out


def read_json(path: Path) -> Rec:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


@dataclass
class Run:
    label: str
    path: Path
    records: list[Rec]
    summary: Rec
    examples: dict[int, Rec]
    found: list[Rec]
    eval_dir: Path | None = None
    sample: dict[int, Rec] = field(default_factory=dict)
    round_end: dict[int, Rec] = field(default_factory=dict)
    steps: list[Rec] = field(default_factory=list)
    dash_eval: dict[int, Rec] = field(default_factory=dict)
    final_eval: Rec | None = None
    rows: dict[int, Rec] = field(default_factory=dict)
    source: dict[str, str] = field(default_factory=dict)

    @property
    def rounds(self) -> list[int]:
        return sorted(self.rows)

    @property
    def group(self) -> str:
        return re.sub(r"_s\d+$", "", self.label)


def find_eval_dir(path: Path, label: str) -> Path | None:
    cand = path.parent.parent / "eval" / label
    meta = read_json(cand / "meta.json")
    if meta and Path(str(meta.get("ckpt", ""))).parent == path.resolve():
        return cand
    return None


def load_run(spec: str, eval_dirs: dict[str, Path] | None = None) -> Run:
    label, _, p = spec.rpartition("=") if "=" in spec else ("", "", spec)
    path = Path(p).resolve()
    label = label or path.name
    recs = read_jsonl(path / "metrics.jsonl")
    if not recs:
        raise SystemExit(f"rl_dash: no metrics.jsonl records in {path}")
    ex = {int(e["round"]): e for e in read_jsonl(path / "examples.jsonl") if "round" in e}
    run = Run(
        label,
        path,
        recs,
        read_json(path / "summary.json"),
        ex,
        read_jsonl(path / "found_targets.jsonl"),
    )
    run.eval_dir = (eval_dirs or {}).get(label) or find_eval_dir(path, label)
    for r in recs:
        ph, rnd = r.get("phase"), r.get("round")
        if ph == "sample" and rnd is not None:
            run.sample[int(rnd)] = r
        elif ph == "train" and r.get("round_end"):
            run.round_end[int(rnd)] = r
        elif ph == "train" and rnd is not None:
            run.steps.append(r)
        elif ph == "eval" and r.get("source") == "dash":
            run.dash_eval[int(rnd)] = r
        elif ph == "eval" and r.get("source") == "harness":
            run.final_eval = r
    build_rows(run)
    return run


# ---------------------------------------------------------------- per-round table

# canonical field -> (description, logged key or None). "derived" entries are filled in build_rows.
FIELDS: dict[str, str] = {
    "solve_rate": "targets solved this round / window (= window pass@k at k samples)",
    "solved_cum_frac": "targets solved so far / run subset",
    "new_proofs": "new distinct verified proofs this round",
    "frac_all_fail": "window targets whose k samples all failed",
    "frac_mixed": "window targets with some but not all samples verified",
    "frac_all_pass": "window targets whose k samples all verified",
    "pass1": "unbiased pass@1 on the window (T = sampling temperature)",
    "pass4": "unbiased pass@4 on the window",
    "pass32": "unbiased pass@32 on the window",
    "distinct_per_solved": "distinct verified proofs per target solved this round",
    "entropy": "mean token entropy on sampled proofs (nats)",
    "len_written": "mean written lines of verified samples",
    "len_pruned": "mean pruned lines of verified samples",
    "len_accepted": "mean written lines of verified samples",
    "len_rejected": "mean written lines of rejected samples",
    "trunc_rate": "samples that hit max_new without QED",
    "n_rl_records": "found-proof records in the fine-tune mix",
    "n_retained": "cap-6 retention records in the fine-tune mix",
    "sample_flops": "sampling FLOPs this round",
    "ft_flops": "fine-tune FLOPs this round",
    "dash_flops": "instrumentation FLOPs this round",
    "eval_flops": "in-run evaluation FLOPs this round",
    "sample_secs": "sampling seconds",
    "judge_secs": "verification seconds",
    "ft_secs": "fine-tune seconds",
    "other_secs": "other seconds in the round (mix, checkpoint, checks)",
    "loss": "fine-tune loss per proof token",
    "grad_norm": "mean gradient norm over the round's steps",
    "grad_norm_max": "largest gradient norm in the round",
    "gpu_mem_gb": "peak GPU memory allocated in the round (GiB)",
    "heldout": "held-out cap-6 greedy accuracy after the round (fixed 200-record slice)",
    "kl_to_init": "KL(current || start) per token on a fixed held-out set",
    "kl_to_init_sampled": "KL(current || start) per token on this round's samples",
    "ref_logp_unsolved": "mean per-token log-prob of the ignition set's known proofs",
    "ref_tok_acc": "share of ignition-proof tokens that are the model's argmax",
    "long_enough": "attempts with at least L_true written lines",
    "ends_on_conclusion": "attempts whose last line states the conclusion at depth 0",
    "cite_dist_max": "largest citation distance in verified samples",
}
BY_LEN_FIELDS = {
    "solve_rate_by_len": "targets solved this round / window, by L_true",
    "pass1_by_len": "pass@1 on the window by L_true",
    "cum_by_len": "targets solved so far / subset, by L_true",
    "long_enough_by_len": "attempts with >= L_true written lines, by L_true",
    "ends_on_conclusion_by_len": "attempts ending on the conclusion, by L_true",
    "heldout_by_len": "held-out cap-6 greedy accuracy by proof length",
    "ref_logp_by_len": "ignition-set log-prob per token by L_true",
}


def _num(x: Any) -> float | None:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _ratio(a: Any, b: Any) -> float | None:
    a, b = _num(a), _num(b)
    return a / b if a is not None and b else None


def _by_len_ratio(num: Any, den: Any) -> dict[str, float]:
    if not isinstance(num, dict) or not isinstance(den, dict):
        return {}
    return {str(k): num.get(k, 0) / v for k, v in den.items() if v}


def build_rows(run: Run) -> None:
    """Per-round canonical fields; run.source[field] is logged or derived (absent = missing)."""
    per_round = {
        int(x["round"]): x for x in (run.summary.get("rl_stats") or {}).get("per_round", [])
    }
    n_targets = _num((run.summary.get("rl_stats") or {}).get("n_targets"))
    last_summary = next((r for r in reversed(run.records) if r.get("phase") == "summary"), {})
    n_targets = n_targets or _num(last_summary.get("n_targets"))
    tgt_by_len = (run.summary.get("rl_stats") or {}).get("targets_by_len") or last_summary.get(
        "targets_by_len"
    )
    src: dict[str, str] = {}

    def put(row: Rec, key: str, val: Any, how: str) -> None:
        if val is None or (isinstance(val, dict) and not val):
            return
        row[key] = val
        if src.get(key, "logged") == "logged" or how == "logged":
            src.setdefault(key, how)

    for rnd, s in sorted(run.sample.items()):
        e = run.round_end.get(rnd, {})
        pr = per_round.get(rnd, {})
        row: Rec = {"round": rnd}
        k = _num(s.get("k"))
        sr = _ratio(s.get("targets_solved_round"), s.get("n_prompts"))
        put(row, "solve_rate", sr, "logged")
        put(row, "solved_cum_frac", _ratio(s.get("targets_solved_cum"), n_targets), "logged")
        put(row, "new_proofs", s.get("new_proofs"), "logged")
        pa = s.get("pass_at") or {}
        if "frac_all_fail" in s:
            for f in ("frac_all_fail", "frac_mixed", "frac_all_pass"):
                put(row, f, s.get(f), "logged")
        elif sr is not None:
            put(row, "frac_all_fail", 1 - sr, "derived: 1 - targets_solved_round / n_prompts")
        if pa:
            put(row, "pass1", pa.get("1"), "logged")
            put(row, "pass4", pa.get("4"), "logged")
            put(row, "pass32", pa.get("32"), "logged")
        else:
            put(row, "pass1", s.get("accept_rate"), "derived: accept_rate (k samples per target)")
            if k == 32:
                put(row, "pass32", sr, "derived: targets_solved_round / n_prompts (k = 32)")
        for f in ("distinct_per_solved", "entropy", "len_rejected", "long_enough"):
            put(row, f, s.get(f), "logged")
        put(row, "ends_on_conclusion", s.get("ends_on_conclusion"), "logged")
        put(row, "cite_dist_max", s.get("cite_dist_max"), "logged")
        put(row, "kl_to_init_sampled", s.get("kl_to_init_sampled"), "logged")
        put(row, "len_written", s.get("mean_written_len"), "logged")
        put(row, "len_pruned", s.get("mean_pruned_len"), "logged")
        if "len_accepted" in s:
            put(row, "len_accepted", s.get("len_accepted"), "logged")
        else:
            put(row, "len_accepted", s.get("mean_written_len"), "derived: mean_written_len")
        if "trunc_rate" in s:
            put(row, "trunc_rate", s.get("trunc_rate"), "logged")
        elif _num(s.get("qed_rate")) is not None:
            put(row, "trunc_rate", 1 - s["qed_rate"], "derived: 1 - qed_rate")
        put(row, "sample_flops", s.get("sample_flops"), "logged")
        put(row, "sample_secs", s.get("sample_secs"), "logged")
        put(row, "judge_secs", s.get("judge_secs"), "logged")
        put(row, "fail_reasons", s.get("fail_reasons"), "logged")
        for f in ("first_error_line_hist", "lines_at_risk_hist", "cite_dist_hist"):
            put(row, f, s.get(f), "logged")
        # training side
        if "n_rl_records" in e:
            put(row, "n_rl_records", e.get("n_rl_records"), "logged")
            put(row, "n_retained", e.get("n_retained"), "logged")
        elif "mix_rl_records" in e:
            put(row, "n_rl_records", e.get("mix_rl_records"), "derived: mix_rl_records")
            nr = _num(e.get("n_records"))
            if nr is not None:
                put(
                    row,
                    "n_retained",
                    nr - e["mix_rl_records"],
                    "derived: n_records - mix_rl_records",
                )
        put(row, "ft_flops", e.get("ft_flops", pr.get("ft_flops")), "logged")
        pf = e.get("phase_flops") or {}
        put(row, "dash_flops", pf.get("dash"), "logged")
        put(row, "eval_flops", pf.get("eval"), "logged")
        if "ft_secs" in e:
            put(row, "ft_secs", e.get("ft_secs"), "logged")
        else:
            put(row, "ft_secs", pr.get("ft_secs"), "derived: summary.json rl_stats.per_round")
        ps = e.get("phase_secs") or {}
        if ps:
            other = sum(v for kk, v in ps.items() if kk not in ("sample", "judge", "finetune"))
            put(row, "other_secs", other, "logged")
        elif pr.get("secs") is not None:
            other = (
                pr["secs"]
                - (s.get("sample_secs") or 0)
                - (s.get("judge_secs") or 0)
                - (pr.get("ft_secs") or 0)
            )
            put(row, "other_secs", max(0.0, other), "derived: per_round secs - sample - judge - ft")
        put(row, "loss", e.get("loss"), "logged")
        if "grad_norm_mean" in e:
            put(row, "grad_norm", e.get("grad_norm_mean"), "logged")
            put(row, "grad_norm_max", e.get("grad_norm_max"), "logged")
        else:
            gs = [x["grad_norm"] for x in run.steps if x.get("round") == rnd and "grad_norm" in x]
            if gs:
                put(row, "grad_norm", sum(gs) / len(gs), "derived: mean of logged step records")
                put(row, "grad_norm_max", max(gs), "derived: max of logged step records")
        put(row, "gpu_mem_gb", e.get("gpu_mem_gb"), "logged")
        # by length
        if "pass_at_by_len" in s:
            p1 = {L: v.get("1") for L, v in s["pass_at_by_len"].items() if v.get("1") is not None}
            put(row, "pass1_by_len", p1, "logged")
        else:
            put(
                row,
                "pass1_by_len",
                _by_len_ratio(s.get("accepted_by_len"), s.get("samples_by_len")),
                "derived: accepted_by_len / samples_by_len",
            )
        put(
            row,
            "solve_rate_by_len",
            _by_len_ratio(s.get("solved_by_len_round"), s.get("window_by_len")),
            "logged",
        )
        if isinstance(tgt_by_len, dict):
            put(
                row,
                "cum_by_len",
                _by_len_ratio(s.get("solved_cum_all_by_len"), tgt_by_len),
                "logged",
            )
        put(row, "long_enough_by_len", s.get("long_enough_by_len"), "logged")
        put(row, "ends_on_conclusion_by_len", s.get("ends_on_conclusion_by_len"), "logged")
        run.rows[rnd] = row
    # end-of-round checks (round 0 = the start model)
    for rnd, d in sorted(run.dash_eval.items()):
        row = run.rows.setdefault(rnd, {"round": rnd})
        put(row, "heldout", d.get("heldout_greedy_dash"), "logged")
        put(row, "heldout_by_len", d.get("heldout_greedy_by_len"), "logged")
        for f in ("kl_to_init", "ref_logp_unsolved", "ref_tok_acc"):
            put(row, f, d.get(f), "logged")
        put(row, "ref_logp_by_len", d.get("ref_logp_by_len"), "logged")
    run.source = src


def series_of(
    run: Run | None, key: str, rounds: list[int] | None = None
) -> list[tuple[float, float]]:
    if run is None:
        return []
    out: list[tuple[float, float]] = []
    for r in rounds if rounds is not None else run.rounds:
        v = _num(run.rows.get(r, {}).get(key))
        if v is not None:
            out.append((float(r), v))
    return out


# ---------------------------------------------------------------- evaluation directories


@functools.lru_cache(maxsize=64)
def load_eval(ev: Path) -> Rec:
    """Per-length test-time numbers of a lengthexp-style evaluation directory (std set)."""
    out: Rec = {"pass1": {}, "pass4": {}, "pass32": {}, "long_enough": {}, "hazard": {}}
    for sf in sorted(
        ev.glob("std_L*.summary.json"), key=lambda p: int(re.findall(r"\d+", p.name)[0])
    ):
        L = int(re.findall(r"\d+", sf.name)[0])
        s = read_json(sf)
        for k in ("pass1", "pass4", "pass32"):
            v = s.get(k.replace("pass", "pass@"))
            if v is not None:
                out[k][L] = v
        rows = read_jsonl(ev / f"std_L{L}.jsonl")
        fails, risk = Counter(), Counter()
        for r in rows:
            for n in r.get("ok_lines", []):
                for p in range(1, n + 2):
                    risk[p] += 1
            for kind, line, _reason, _start in r.get("fails", []):
                if kind == "bad_line":
                    fails[line] += 1
                    for p in range(1, line + 1):
                        risk[p] += 1
                elif kind == "early_qed":
                    fails[line + 1] += 1
                    for p in range(1, line + 2):
                        risk[p] += 1
                elif kind == "no_qed":
                    for p in range(1, line + 1):
                        risk[p] += 1
        out["hazard"][L] = {p: fails[p] / risk[p] for p in sorted(risk) if risk[p] >= 100}
        n = enough = 0
        for smp in read_jsonl(ev / "samples" / f"std_L{L}.jsonl.gz"):
            n += 1
            enough += smp.get("body", "").split().count(";") >= L
        if n:
            out["long_enough"][L] = enough / n
    return out


# ---------------------------------------------------------------- formatting


def fmt(v: Any, kind: str = "g") -> str:
    x = _num(v)
    if x is None:
        return "missing"
    if kind == "pct":
        return f"{100 * x:.1f}%"
    if kind == "int":
        return f"{x:,.0f}"
    if kind == "sci":
        return f"{x:.3g}"
    if kind == "secs":
        return f"{x:.1f} s"
    if kind == "f3":
        return f"{x:.3f}"
    return f"{x:.4g}"


def tick_fmt(t: float, step: float, kind: str) -> str:
    """Tick label with as many decimals as the tick step needs."""
    if kind == "pct":
        d = max(0, -math.floor(math.log10(step * 100) + 1e-9)) if step > 0 else 0
        return f"{100 * t:.{d}f}%"
    if kind == "sci" and (abs(t) >= 1e5 or 0 < abs(t) < 1e-3):
        return f"{t:.2g}"
    d = max(0, -math.floor(math.log10(step) + 1e-9)) if step > 0 else 0
    return f"{t:,.{d}f}"


def nice_ticks(lo: float, hi: float, n: int = 5, integer: bool = False) -> list[float]:
    if not math.isfinite(lo) or not math.isfinite(hi):
        return [0.0, 1.0]
    if hi <= lo:
        hi = lo + (abs(lo) or 1.0)
    raw = (hi - lo) / max(n - 1, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    if integer:
        step = max(1.0, round(step))
    t0 = math.floor(lo / step) * step
    ticks = []
    t = t0
    while t <= hi + step * 0.5:
        ticks.append(round(t, 12))
        t += step
    return ticks


def esc(s: Any) -> str:
    return html.escape(str(s), quote=True)


# ---------------------------------------------------------------- charts (inline SVG)

W, H = 640, 250
WW = 1280  # viewBox width of a full-row chart (text keeps its size)
ML, MR, MT, MB = 58, 14, 10, 38
_ids = Counter()


def _cid(prefix: str) -> str:
    _ids[prefix] += 1
    return f"{prefix}{_ids[prefix]}"


@dataclass
class Series:
    name: str
    pts: list[tuple[float, float]]
    color: str  # a CSS value, e.g. var(--s1)
    muted: bool = False


def _legend(series: list[Series], kind: str = "line") -> str:
    if len(series) < 2:
        return ""
    items = []
    for s in series:
        key = "lkey" if kind == "line" else "bkey"
        items.append(
            f'<span class="li"><span class="{key}" style="background:{s.color}"></span>'
            f"{esc(s.name)}</span>"
        )
    return f'<div class="legend">{"".join(items)}</div>'


def _table(headers: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in rows)
    return (
        '<details><summary>Table</summary><div class="tw">'
        f"<table><tr>{th}</tr>{body}</table></div></details>"
    )


def figure(title: str, sub: str, body: str, note: str = "", wide: bool = False) -> str:
    n = f'<p class="note">{note}</p>' if note else ""
    cls = "card wide" if wide else "card"
    cap = f'<figcaption><h3>{esc(title)}</h3><p class="sub">{esc(sub)}</p></figcaption>'
    return f'<figure class="{cls}">{cap}{body}{n}</figure>'


def missing_fig(title: str, what: str) -> str:
    return figure(
        title,
        what,
        '<div class="missing">missing: not logged (shown as missing, not as zero)</div>',
    )


def line_chart(
    title: str,
    sub: str,
    series: list[Series],
    *,
    x_label: str = "round",
    y_fmt: str = "g",
    y_min: float | None = None,
    y_max: float | None = None,
    note: str = "",
    wide: bool = False,
    x_fmt: str = "int",
) -> str:
    empty = [s for s in series if not s.pts]
    series = [s for s in series if s.pts]
    if not series:
        return missing_fig(title, sub)
    if empty:
        miss = ", ".join(s.name for s in empty)
        note = f"Missing (not logged): {esc(miss)}." + (f" {note}" if note else "")
    W = WW if wide else globals()["W"]  # noqa: N806
    xs = sorted({x for s in series for x, _ in s.pts})
    ys = [y for s in series for _, y in s.pts]
    lo = min(ys) if y_min is None else y_min
    hi = max(ys) if y_max is None else y_max
    if y_min is None and lo > 0 and lo < 0.35 * hi:
        lo = 0.0
    yt = nice_ticks(lo, hi)
    lo, hi = min(yt[0], lo), max(yt[-1], hi)
    x0, x1 = xs[0], xs[-1] if xs[-1] > xs[0] else xs[0] + 1
    pw, ph = W - ML - MR, H - MT - MB

    def px(x: float) -> float:
        return ML + (x - x0) / (x1 - x0) * pw

    def py(y: float) -> float:
        return MT + ph - (y - lo) / ((hi - lo) or 1) * ph

    g = []
    ystep = yt[1] - yt[0] if len(yt) > 1 else 1.0
    for t in yt:
        g.append(f'<line class="grid" x1="{ML}" x2="{W - MR}" y1="{py(t):.1f}" y2="{py(t):.1f}"/>')
        g.append(
            f'<text class="tick" x="{ML - 6}" y="{py(t) + 4:.1f}" text-anchor="end">'
            f"{esc(tick_fmt(t, ystep, y_fmt))}</text>"
        )
    for t in nice_ticks(x0, x1, 12 if wide else 7, integer=x_fmt == "int"):
        if x0 <= t <= x1:
            g.append(
                f'<text class="tick" x="{px(t):.1f}" y="{H - MB + 16}" text-anchor="middle">'
                f"{esc(fmt(t, x_fmt))}</text>"
            )
    g.append(f'<line class="axis" x1="{ML}" x2="{W - MR}" y1="{MT + ph}" y2="{MT + ph}"/>')
    g.append(
        f'<text class="alabel" x="{ML + pw / 2}" y="{H - 4}" text-anchor="middle">'
        f"{esc(x_label)}</text>"
    )
    marks = []
    for s in series:
        segs: list[list[tuple[float, float]]] = [[]]
        prev = None
        xi = {x: i for i, x in enumerate(xs)}
        for x, y in s.pts:
            if prev is not None and xi[x] - xi[prev] > 1 and x - prev > 1:
                segs.append([])
            segs[-1].append((x, y))
            prev = x
        cls = "ln muted" if s.muted else "ln"
        for seg in segs:
            if len(seg) == 1:
                marks.append(
                    f'<circle class="dot" cx="{px(seg[0][0]):.1f}" cy="{py(seg[0][1]):.1f}" r="4" '
                    f'style="fill:{s.color}"/>'
                )
            else:
                d = " ".join(
                    f"{'M' if i == 0 else 'L'}{px(x):.1f},{py(y):.1f}"
                    for i, (x, y) in enumerate(seg)
                )
                marks.append(f'<path class="{cls}" d="{d}" style="stroke:{s.color}"/>')
    data = {
        "xl": x_label,
        "xs": [fmt(x, x_fmt) for x in xs],
        "px": [round(px(x), 1) for x in xs],
        "s": [
            {"n": s.name, "c": s.color, "v": [fmt(dict(s.pts).get(x), y_fmt) for x in xs]}
            for s in series
        ],
    }
    svg = (
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}">'
        f"{''.join(g)}{''.join(marks)}"
        f'<line class="cross" x1="0" x2="0" y1="{MT}" y2="{MT + ph}" visibility="hidden"/>'
        f'<rect class="hit" x="{ML}" y="{MT}" width="{pw}" height="{ph}"/></svg>'
    )
    blob = json.dumps(data).replace("</", "<\\/")
    js = f'<script type="application/json">{blob}</script>'
    table = _table(
        [x_label] + [s.name for s in series],
        [[fmt(x, x_fmt)] + [fmt(dict(s.pts).get(x), y_fmt) for s in series] for x in xs],
    )
    leg = _legend(series + [Series(f"{s.name} (missing)", [], "transparent") for s in empty])
    body = f'{leg}<div class="plot" data-chart="line">{svg}{js}</div>{table}'
    return figure(title, sub, body, note, wide)


def heatmap(
    title: str,
    sub: str,
    grid: dict[int, dict[str, float]],
    *,
    y_label: str = "L_true",
    v_fmt: str = "pct",
    v_max: float | None = None,
    note: str = "",
    log: bool = False,
    wide: bool = False,
) -> str:
    """grid[round][length] = value. Rows = lengths (short at the bottom), columns = rounds."""
    grid = {r: v for r, v in grid.items() if v}
    if not grid:
        return missing_fig(title, sub)
    W = WW if wide else globals()["W"]  # noqa: N806
    rounds = sorted(grid)
    lens = sorted({L for v in grid.values() for L in v}, key=lambda s: float(s))
    vals = [v for g in grid.values() for v in g.values() if v is not None]
    vmax = v_max if v_max is not None else (max(vals) if vals else 1.0) or 1.0
    vmin = min(vals) if log and vals else 0.0
    pw, ph = W - ML - MR, H - MT - MB
    cw, chh = pw / len(rounds), ph / len(lens)
    cells = []
    for i, r in enumerate(rounds):
        for j, L in enumerate(lens):
            v = grid[r].get(L)
            x, y = ML + i * cw, MT + ph - (j + 1) * chh
            if v is None:
                cls = "na"
            else:
                f = (v - vmin) / ((vmax - vmin) or 1)
                cls = f"q{min(8, max(0, int(f * 8.999)))}"
            tip = json.dumps([[f"round {r}, {y_label} {L}", "", None], ["", fmt(v, v_fmt), None]])
            cells.append(
                f'<rect class="{cls}" x="{x:.2f}" y="{y:.2f}" width="{max(cw - 1, 0.6):.2f}" '
                f'height="{chh - 2:.2f}" data-tip="{esc(tip)}"/>'
            )
    g = [
        f'<text class="tick" x="{ML - 6}" y="{MT + ph - (j + 0.5) * chh + 4:.1f}" '
        'text-anchor="end">'
        f"{esc(L)}</text>"
        for j, L in enumerate(lens)
    ]
    for t in nice_ticks(rounds[0], rounds[-1], 12 if wide else 7, integer=True):
        if t in rounds:
            i = rounds.index(int(t))
            g.append(
                f'<text class="tick" x="{ML + (i + 0.5) * cw:.1f}" y="{H - MB + 16}" '
                'text-anchor="middle">'
                f"{int(t)}</text>"
            )
    g.append(
        f'<text class="alabel" x="{ML + pw / 2}" y="{H - 4}" text-anchor="middle">round</text>'
    )
    g.append(
        f'<text class="alabel" x="12" y="{MT + ph / 2}" transform="rotate(-90 12 {MT + ph / 2})" '
        f'text-anchor="middle">{esc(y_label)}</text>'
    )
    scale = (
        f'<div class="scale"><span>{esc(fmt(vmin, v_fmt))}</span>'
        + "".join(f'<i class="q{k}"></i>' for k in range(9))
        + f"<span>{esc(fmt(vmax, v_fmt))}</span>"
        + '<span class="na-key"><i class="na"></i> missing</span></div>'
    )
    svg = (
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}">'
        f"{''.join(g)}{''.join(cells)}</svg>"
    )
    table = _table(
        ["round"] + [f"{y_label} {L}" for L in lens],
        [[str(r)] + [fmt(grid[r].get(L), v_fmt) for L in lens] for r in rounds],
    )
    return figure(title, sub, f'{scale}<div class="plot">{svg}</div>{table}', note, wide)


def columns(
    title: str,
    sub: str,
    xs: list[str],
    stacks: list[Series],
    *,
    y_fmt: str = "g",
    x_label: str = "round",
    note: str = "",
    wide: bool = False,
) -> str:
    """Stacked columns: each Series holds one value per x (pts[i] = (i, value))."""
    stacks = [s for s in stacks if any(v for _, v in s.pts)]
    if not stacks or not xs:
        return missing_fig(title, sub)
    W = WW if wide else globals()["W"]  # noqa: N806
    vals = [{int(i): v for i, v in s.pts} for s in stacks]
    tot = [sum(v.get(i, 0.0) for v in vals) for i in range(len(xs))]
    hi = max(tot) or 1.0
    yt = nice_ticks(0, hi)
    hi = max(yt[-1], hi)
    pw, ph = W - ML - MR, H - MT - MB
    slot = pw / len(xs)
    bw = min(24.0, max(slot - 2, 1.0))

    def py(y: float) -> float:
        return MT + ph - y / hi * ph

    g = []
    ystep = yt[1] - yt[0] if len(yt) > 1 else 1.0
    for t in yt:
        g.append(f'<line class="grid" x1="{ML}" x2="{W - MR}" y1="{py(t):.1f}" y2="{py(t):.1f}"/>')
        g.append(
            f'<text class="tick" x="{ML - 6}" y="{py(t) + 4:.1f}" text-anchor="end">'
            f"{esc(tick_fmt(t, ystep, y_fmt))}</text>"
        )
    step = max(1, len(xs) // 8)
    for i in range(0, len(xs), step):
        g.append(
            f'<text class="tick" x="{ML + (i + 0.5) * slot:.1f}" y="{H - MB + 16}" '
            'text-anchor="middle">'
            f"{esc(xs[i])}</text>"
        )
    g.append(
        f'<text class="alabel" x="{ML + pw / 2}" y="{H - 4}" text-anchor="middle">'
        f"{esc(x_label)}</text>"
    )
    marks = []
    for i in range(len(xs)):
        base = 0.0
        x = ML + (i + 0.5) * slot - bw / 2
        tip = [[f"{x_label} {xs[i]}", "", None]] + [
            [s.name, fmt(v.get(i), y_fmt), s.color] for s, v in zip(stacks, vals, strict=True)
        ]
        top_k = max((k for k, v in enumerate(vals) if v.get(i)), default=-1)
        for k, (s, v) in enumerate(zip(stacks, vals, strict=True)):
            h = v.get(i, 0.0)
            if not h:
                continue
            y0, y1 = py(base), py(base + h)
            base += h
            if k == top_k and y0 - y1 > 4:
                r = min(4.0, bw / 2)
                d = (
                    f"M{x:.1f},{y0:.1f} L{x:.1f},{y1 + r:.1f} "
                    f"Q{x:.1f},{y1:.1f} {x + r:.1f},{y1:.1f} "
                    f"L{x + bw - r:.1f},{y1:.1f} Q{x + bw:.1f},{y1:.1f} {x + bw:.1f},{y1 + r:.1f} "
                    f"L{x + bw:.1f},{y0:.1f} Z"
                )
                marks.append(
                    f'<path class="seg" d="{d}" style="fill:{s.color}" '
                    f'data-tip="{esc(json.dumps(tip))}"/>'
                )
            else:
                marks.append(
                    f'<rect class="seg" x="{x:.1f}" y="{y1:.1f}" width="{bw:.1f}" '
                    f'height="{max(y0 - y1, 0.5):.1f}" '
                    f'style="fill:{s.color}" data-tip="{esc(json.dumps(tip))}"/>'
                )
    g.append(f'<line class="axis" x1="{ML}" x2="{W - MR}" y1="{MT + ph}" y2="{MT + ph}"/>')
    svg = (
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}">'
        f"{''.join(g)}{''.join(marks)}</svg>"
    )
    table = _table(
        [x_label] + [s.name for s in stacks],
        [[xs[i]] + [fmt(v.get(i), y_fmt) for v in vals] for i in range(len(xs))],
    )
    return figure(
        title, sub, f'{_legend(stacks, "bar")}<div class="plot">{svg}</div>{table}', note, wide
    )


def table_fig(
    title: str,
    sub: str,
    headers: list[str],
    rows: list[list[str]],
    note: str = "",
    wide: bool = True,
) -> str:
    th = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in rows)
    return figure(
        title,
        sub,
        f'<div class="tw"><table class="txt"><tr>{th}</tr>{body}</table></div>',
        note,
        wide,
    )


# ---------------------------------------------------------------- page

CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;
--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);--s1:#2a78d6;
--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;--ctl:#898781;--q0:#cde2fb;
--q1:#b7d3f6;--q2:#9ec5f4;--q3:#86b6ef;--q4:#5598e7;--q5:#2a78d6;--q6:#1c5cab;--q7:#104281;
--q8:#0d366b;--o1:#86b6ef;--o2:#3987e5;--o3:#256abf;--o4:#184f95;--o5:#0d366b}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;
--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;
--s5:#d55181;--s6:#008300;--ctl:#898781;--q0:#0d366b;--q1:#104281;--q2:#184f95;--q3:#1c5cab;
--q4:#256abf;--q5:#3987e5;--q6:#6da7ec;--q7:#9ec5f4;--q8:#cde2fb;--o1:#9ec5f4;--o2:#6da7ec;
--o3:#3987e5;--o4:#256abf;--o5:#184f95}}:root[data-theme="dark"]{color-scheme:dark;
--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;
--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;
--s5:#d55181;--s6:#008300;--ctl:#898781;--q0:#0d366b;--q1:#104281;--q2:#184f95;--q3:#1c5cab;
--q4:#256abf;--q5:#3987e5;--q6:#6da7ec;--q7:#9ec5f4;--q8:#cde2fb;--o1:#9ec5f4;--o2:#6da7ec;
--o3:#3987e5;--o4:#256abf;--o5:#184f95}*{box-sizing:border-box}body{margin:0;
background:var(--page);color:var(--ink);
font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}header{padding:20px 16px 8px;
max-width:1360px;margin:0 auto}h1{font-size:20px;margin:0 0 4px}header .sub{color:var(--ink2);
margin:0 0 10px;word-break:break-all}nav{display:flex;flex-wrap:wrap;gap:6px}
nav a{color:var(--ink2);text-decoration:none;padding:4px 10px;border-radius:6px;
border:1px solid var(--ring)}nav a.on{color:var(--ink);font-weight:600;
background:var(--surface)}main{max-width:1360px;margin:0 auto;padding:8px 16px 40px;
display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,560px),1fr));gap:14px}
.card{margin:0;background:var(--surface);border:1px solid var(--ring);border-radius:10px;
padding:12px 14px;min-width:0}.card.wide{grid-column:1/-1}figcaption h3{font-size:15px;margin:0}
figcaption .sub,.note{color:var(--ink2);margin:2px 0 6px;font-size:12.5px}.note{margin-top:8px}
svg{width:100%;height:auto;display:block}.grid{stroke:var(--grid);stroke-width:1}
.axis{stroke:var(--axis);stroke-width:1}.tick{fill:var(--muted);font-size:11px;
font-variant-numeric:tabular-nums}.alabel{fill:var(--ink2);font-size:11.5px}.ln{fill:none;
stroke-width:2;stroke-linejoin:round;stroke-linecap:round}.dot{stroke:var(--surface);
stroke-width:2}.cross{stroke:var(--axis);stroke-width:1}.hit{fill:transparent;cursor:crosshair}
.seg{stroke:var(--surface);stroke-width:1}.seg:hover,rect[data-tip]:hover{opacity:.8}
.legend{display:flex;flex-wrap:wrap;gap:4px 14px;margin:2px 0 4px;color:var(--ink2);
font-size:12.5px}.li{display:inline-flex;align-items:center;gap:6px}.lkey{width:14px;height:3px;
border-radius:2px;display:inline-block}.bkey{width:10px;height:10px;border-radius:2px;
display:inline-block}.q0{fill:var(--q0);background:var(--q0)}.q1{fill:var(--q1);
background:var(--q1)}.q2{fill:var(--q2);background:var(--q2)}.q3{fill:var(--q3);
background:var(--q3)}.q4{fill:var(--q4);background:var(--q4)}.q5{fill:var(--q5);
background:var(--q5)}.q6{fill:var(--q6);background:var(--q6)}.q7{fill:var(--q7);
background:var(--q7)}.q8{fill:var(--q8);background:var(--q8)}.na{fill:transparent;
stroke:var(--grid);stroke-width:1}i.na{border:1px solid var(--grid)}.scale{display:flex;
align-items:center;gap:2px;font-size:11.5px;color:var(--ink2);margin:0 0 4px}
.scale i{display:inline-block;width:16px;height:10px}.scale span{margin:0 6px}
.na-key{display:inline-flex;gap:4px;align-items:center}.missing{color:var(--muted);
border:1px dashed var(--grid);border-radius:8px;padding:28px;text-align:center}
details{margin-top:6px;font-size:12.5px}summary{cursor:pointer;color:var(--ink2)}
.tw{overflow-x:auto;max-height:420px}table{border-collapse:collapse;
font-variant-numeric:tabular-nums;font-size:12.5px}td,th{border-bottom:1px solid var(--grid);
padding:3px 8px;text-align:right;white-space:nowrap}th{position:sticky;top:0;
background:var(--surface);color:var(--ink2);font-weight:600}
td:first-child,th:first-child{text-align:left}table.txt td,table.txt th{text-align:left;
white-space:normal}.ex td{white-space:normal;text-align:left;font-family:ui-monospace,monospace;
font-size:11.5px;max-width:900px}.tip{position:fixed;display:none;pointer-events:none;
background:var(--surface);color:var(--ink);border:1px solid var(--ring);border-radius:8px;
padding:6px 9px;font-size:12px;box-shadow:0 2px 10px rgba(0,0,0,.15);z-index:9}.tr{display:flex;
align-items:center;gap:6px;white-space:nowrap}.tr b{font-variant-numeric:tabular-nums}
.tr .k{width:12px;height:3px;border-radius:2px;display:inline-block}
"""

JS = """
(function(){const tip=document.createElement('div');tip.className='tip';
document.body.appendChild(tip);function show(e,rows){tip.replaceChildren();
for(const [n,v,c] of rows){const r=document.createElement('div');r.className='tr';if(c){
const k=document.createElement('span');k.className='k';k.style.background=c;r.appendChild(k);}
if(v){const b=document.createElement('b');b.textContent=v;r.appendChild(b);}
const t=document.createElement('span');t.textContent=n;r.appendChild(t);tip.appendChild(r);}
tip.style.display='block';const w=tip.offsetWidth,h=tip.offsetHeight;
let x=e.clientX+14,y=e.clientY+14;if(x+w>innerWidth-8)x=e.clientX-w-14;
if(y+h>innerHeight-8)y=e.clientY-h-14;tip.style.left=x+'px';tip.style.top=y+'px';}
function hide(){tip.style.display='none';}
document.querySelectorAll('.plot[data-chart="line"]').forEach(p=>{
const d=JSON.parse(p.querySelector('script').textContent);
const svg=p.querySelector('svg'),hit=svg.querySelector('.hit'),cr=svg.querySelector('.cross');
hit.addEventListener('pointermove',e=>{const pt=svg.createSVGPoint();pt.x=e.clientX;
pt.y=e.clientY;const q=pt.matrixTransform(svg.getScreenCTM().inverse());let b=0,bd=1e9;
d.px.forEach((x,i)=>{const dd=Math.abs(x-q.x);if(dd<bd){bd=dd;b=i;}});
cr.setAttribute('x1',d.px[b]);cr.setAttribute('x2',d.px[b]);
cr.setAttribute('visibility','visible');
show(e,[[d.xl+' '+d.xs[b],'',null]].concat(d.s.map(s=>[s.n,s.v[b],s.c])));});
hit.addEventListener('pointerleave',()=>{hide();cr.setAttribute('visibility','hidden');});});
document.querySelectorAll('[data-tip]').forEach(el=>{
el.addEventListener('pointermove',e=>show(e,JSON.parse(el.dataset.tip)));
el.addEventListener('pointerleave',hide);});})();
"""


def page(
    title: str, sub: str, active: str, figs: list[str], prefix: str = "", comparison: str = ""
) -> str:
    links = []
    for key, name in DASHBOARDS:
        if key == "5_comparison":
            href = comparison or f"{prefix}5_comparison.html"
        else:
            href = f"{prefix}{key}.html"
        on = ' class="on"' if key == active else ""
        links.append(f'<a{on} href="{esc(href)}">{esc(name)}</a>')
    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{esc(title)}</title><style>{CSS}</style></head><body>"
        f'<header><h1>{esc(title)}</h1><p class="sub">{esc(sub)}</p>'
        f"<nav>{''.join(links)}</nav></header><main>{''.join(figs)}</main>"
        f"<script>{JS}</script></body></html>"
    )


# ---------------------------------------------------------------- dashboards

RUN_C, CTL_C = "var(--s1)", "var(--ctl)"


def _run_ctl(run: Run, ctl: Run | None, key: str, name: str | None = None) -> list[Series]:
    out = [Series(name or run.label, series_of(run, key), RUN_C)]
    if ctl is not None:
        out.append(Series(f"frozen control ({ctl.label})", series_of(ctl, key), CTL_C, muted=True))
    return out


def _last(pts: list[tuple[float, float]], n: int = 10) -> list[float]:
    return [y for _, y in pts[-n:]]


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _slope(pts: list[tuple[float, float]]) -> float | None:
    if len(pts) < 3:
        return None
    mx = sum(x for x, _ in pts) / len(pts)
    my = sum(y for _, y in pts) / len(pts)
    den = sum((x - mx) ** 2 for x, _ in pts)
    return sum((x - mx) * (y - my) for x, y in pts) / den if den else None


def coverage_fig(run: Run) -> str:
    rows = []
    for f, desc in list(FIELDS.items()) + list(BY_LEN_FIELDS.items()):
        how = run.source.get(f)
        rows.append([f, how or "missing", desc])
    note = (
        "Missing fields are shown as missing, never as zero. Derived fields use an exact identity "
        "with older fields."
    )
    return table_fig(
        "Field coverage",
        f"{run.label}: which dashboard fields this run logged",
        ["field", "source", "meaning"],
        rows,
        note,
    )


def limit_signals(run: Run, ctl: Run | None) -> str:
    sr = series_of(run, "solve_rate")
    rows = []
    sf = sum(v for _, v in series_of(run, "sample_flops"))
    tf = sum(v for _, v in series_of(run, "ft_flops"))
    df = sum(v for _, v in series_of(run, "dash_flops")) + sum(
        v for _, v in series_of(run, "eval_flops")
    )
    ss = sum(v for _, v in series_of(run, "sample_secs")) + sum(
        v for _, v in series_of(run, "judge_secs")
    )
    ts = (
        ss
        + sum(v for _, v in series_of(run, "ft_secs"))
        + sum(v for _, v in series_of(run, "other_secs"))
    )
    rows.append(
        [
            "sampling-bound",
            "share of compute (time) spent sampling and verifying",
            f"{fmt(sf / (sf + tf + df), 'pct') if sf + tf + df else 'missing'} "
            f"({fmt(ss / ts, 'pct') if ts else 'missing'} of time)",
            "",
        ]
    )
    top = None
    for r in reversed(run.rounds):
        d = run.rows[r].get("cum_by_len")
        if d:
            top = max(d, key=float)
            late_top = [
                v
                for x in run.rounds[-10:]
                if (v := run.rows[x].get("solve_rate_by_len", {}).get(top)) is not None
            ]
            rows.append(
                [
                    "exploration-bound",
                    f"share of L_true {top} targets ever solved; "
                    f"window solve rate at L_true {top} (last 10)",
                    f"{fmt(d[top], 'pct')}; {fmt(_mean(late_top), 'pct')}",
                    "",
                ]
            )
            break
    if top is None:
        rows.append(
            ["exploration-bound", "share of the longest targets ever solved", "missing", ""]
        )
    if ctl is not None:
        cs = dict(series_of(ctl, "solve_rate"))
        same = [
            x
            for x in run.rounds
            if x in ctl.rows
            and x > 0
            and run.sample.get(x, {}).get("window_by_len")
            == ctl.sample.get(x, {}).get("window_by_len")
            and _first_pass(run, x)
        ]
        common = [(x, y, cs[x]) for x, y in sr if x in cs and x in same]
        ratio = (
            _mean([y for _, y, _ in common]) / (_mean([c for _, _, c in common]) or math.nan)
            if common
            else None
        )
        last = common[-1] if common else None
        rows.append(
            [
                "exploitation-bound",
                "run solve rate / frozen control, first pass (same windows): "
                "mean; last matched round",
                f"{fmt(ratio, 'f3')}; {fmt(last[1] / last[2] if last and last[2] else None, 'f3')}",
                f"rounds {int(common[0][0])}-{int(common[-1][0])}"
                if common
                else "no matched window",
            ]
        )
    else:
        rows.append(
            ["exploitation-bound", "run solve rate / frozen control", "no control given", ""]
        )
    gn = series_of(run, "grad_norm_max")
    lo = series_of(run, "loss")
    med = sorted(y for _, y in gn)[len(gn) // 2] if gn else None
    spikes = [int(x) for x, y in gn if med and y > 3 * med]
    ho = series_of(run, "heldout")
    rows.append(
        [
            "learning-bound",
            "rounds with max grad norm > 3x its median; held-out first -> last",
            f"{spikes[:8] if spikes else 'none'}; "
            f"{fmt(ho[0][1], 'pct') + ' -> ' + fmt(ho[-1][1], 'pct') if ho else 'missing'}",
            f"loss last round {fmt(lo[-1][1], 'f3') if lo else 'missing'}",
        ]
    )
    cum = series_of(run, "solved_cum_frac")
    rows.append(
        [
            "capacity-bound",
            "slope over the last 10 rounds: solve rate; share solved so far (per round)",
            f"{fmt(_slope(sr[-10:]), 'sci')}; {fmt(_slope(cum[-10:]), 'sci')}",
            "",
        ]
    )
    note = (
        "The five limits of the monitoring survey (Table 3), checked in order. "
        "These are measurements, not verdicts: no alert thresholds are set."
    )
    return table_fig(
        "Limit signals",
        "what each of the five candidate limits predicts, measured on this run",
        ["limit", "signal", "this run", "note"],
        rows,
        note,
    )


def _first_pass(run: Run, rnd: int) -> bool:
    """True while round `rnd` is inside the first pass over the subset (no refill yet)."""
    n_t = _num((run.summary.get("rl_stats") or {}).get("n_targets")) or 6000
    win = _num(run.sample.get(rnd, {}).get("n_prompts")) or 400
    return rnd * win <= n_t


def dash_limiting(run: Run, ctl: Run | None) -> list[str]:
    figs = [limit_signals(run, ctl)]
    idx = []
    for key, name, c in (
        ("solve_rate", "solve rate", "var(--s1)"),
        ("new_proofs", "new proofs", "var(--s2)"),
        ("frac_mixed", "mixed share", "var(--s3)"),
        ("loss", "loss", "var(--s4)"),
        ("grad_norm", "grad norm", "var(--s5)"),
    ):
        pts = series_of(run, key)
        base = next((y for _, y in pts if y), None)
        if base:
            idx.append(Series(name, [(x, y / base) for x, y in pts], c))
    figs.append(
        line_chart(
            "Key signals indexed to round 1",
            "each curve divided by its first non-zero value",
            idx,
            y_fmt="f3",
            wide=True,
            note="One axis for measures of different scale: which curve went flat first?",
        )
    )
    figs.append(
        line_chart(
            "Solve rate per round",
            "window targets with at least one verified sample",
            _run_ctl(run, ctl, "solve_rate"),
            y_fmt="pct",
            note="The first pass over the subset (6,000 targets / 400 per round = 15 rounds) "
            "draws the same windows in the run and the control. After it, refills are "
            "hardest-first (unsolved targets first), so each window depends on what that "
            "run has solved, and the frozen control's windows are almost all unsolved "
            "targets.",
        )
    )
    gf = [
        Series(n, series_of(run, k), c)
        for k, n, c in (
            ("frac_all_fail", "all k failed", "var(--s1)"),
            ("frac_mixed", "mixed", "var(--s2)"),
            ("frac_all_pass", "all k passed", "var(--s3)"),
        )
    ]
    if ctl is not None:
        gf.append(
            Series("all failed, frozen control", series_of(ctl, "frac_all_fail"), CTL_C, muted=True)
        )
    figs.append(
        line_chart(
            "Group fractions",
            "share of window targets by how many of their k samples verified",
            gf,
            y_fmt="pct",
            y_min=0,
            y_max=1,
            note="Only mixed groups carry a signal a group-relative method could use; for "
            "expert iteration the all-fail share is the unexplored frontier.",
        )
    )
    figs.append(
        line_chart(
            "New distinct proofs per round",
            "normalised proofs not seen before",
            _run_ctl(run, ctl, "new_proofs"),
            y_fmt="int",
        )
    )
    rs = run.rounds
    xs = [str(r) for r in rs if r > 0]
    rr = [r for r in rs if r > 0]
    figs.append(
        columns(
            "Compute per round by phase",
            "metered FLOPs (the budget is FLOPs)",
            xs,
            [
                Series(n, [(i, _num(run.rows[r].get(k)) or 0.0) for i, r in enumerate(rr)], c)
                for k, n, c in (
                    ("sample_flops", "sampling", "var(--s1)"),
                    ("ft_flops", "fine-tune", "var(--s2)"),
                    ("dash_flops", "dashboard checks", "var(--s3)"),
                    ("eval_flops", "in-run eval", "var(--s4)"),
                )
            ],
            y_fmt="sci",
        )
    )
    figs.append(
        columns(
            "Wall time per round by phase",
            "seconds on this run's card",
            xs,
            [
                Series(n, [(i, _num(run.rows[r].get(k)) or 0.0) for i, r in enumerate(rr)], c)
                for k, n, c in (
                    ("sample_secs", "sampling", "var(--s1)"),
                    ("judge_secs", "verification", "var(--s3)"),
                    ("ft_secs", "fine-tune", "var(--s2)"),
                    ("other_secs", "other", "var(--s4)"),
                )
            ],
            y_fmt="g",
        )
    )
    figs.append(
        line_chart(
            "Fine-tune loss",
            "mean per proof token over the round's steps",
            [Series(run.label, series_of(run, "loss"), RUN_C)],
            y_fmt="f3",
        )
    )
    figs.append(
        line_chart(
            "Gradient norm",
            "before clipping (clip 1.0)",
            [
                Series("mean", series_of(run, "grad_norm"), "var(--s1)"),
                Series("max", series_of(run, "grad_norm_max"), "var(--s2)"),
            ],
            y_fmt="f3",
        )
    )
    return figs


def dash_diversity(run: Run, ctl: Run | None) -> list[str]:
    figs = [
        line_chart(
            "pass@k on the sampling window",
            "unbiased estimates from each target's successes "
            "(training temperature, this round's window)",
            [
                Series(n, series_of(run, k), c)
                for k, n, c in (
                    ("pass1", "pass@1", "var(--s1)"),
                    ("pass4", "pass@4", "var(--s2)"),
                    ("pass32", "pass@32", "var(--s3)"),
                )
            ],
            y_fmt="pct",
            wide=True,
            note="Diversity loss shows as pass@1 rising while pass@32 does not.",
        )
    ]
    figs.append(
        line_chart(
            "pass@1 against the frozen control",
            "window pass@1",
            _run_ctl(run, ctl, "pass1"),
            y_fmt="pct",
        )
    )
    figs.append(
        heatmap(
            "pass@1 by L_true",
            "window pass@1 per round and target length",
            {r: run.rows[r].get("pass1_by_len", {}) for r in run.rounds},
            v_fmt="pct",
        )
    )
    figs.append(
        line_chart(
            "Distinct proofs per solved target",
            "this round's distinct verified proofs / targets solved this round",
            _run_ctl(run, ctl, "distinct_per_solved"),
            y_fmt="f3",
        )
    )
    figs.append(
        line_chart(
            "Token entropy on sampled proofs",
            "nats per token at the sampling temperature, "
            "on a fixed-size subsample of the round's samples",
            _run_ctl(run, ctl, "entropy"),
            y_fmt="f3",
        )
    )
    return figs


def _reason_mix(run: Run) -> list[Series]:
    tot = Counter()
    for r in run.rounds:
        for name, c in run.rows[r].get("fail_reasons", []) or []:
            tot[name] += c
    top = [n for n, _ in tot.most_common(5)]
    out = []
    colors = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)", "var(--s5)", "var(--s6)"]
    for i, name in enumerate(top):
        pts = []
        for r in run.rounds:
            fr = run.rows[r].get("fail_reasons")
            s = run.sample.get(r, {})
            n_rej = (s.get("samples") or 0) * (1 - (s.get("accept_rate") or 0))
            if fr and n_rej:
                pts.append((float(r), dict((a, b) for a, b in fr).get(name, 0) / n_rej))
        out.append(Series(name, pts, colors[i]))
    return out


def dash_integrity(run: Run, ctl: Run | None) -> list[str]:
    figs = [
        line_chart(
            "Written against pruned length",
            "mean lines of verified samples",
            [
                Series("written", series_of(run, "len_written"), "var(--s1)"),
                Series("pruned", series_of(run, "len_pruned"), "var(--s2)"),
            ],
            y_fmt="f3",
            note="Pruned keeps only the lines the conclusion transitively cites, so the gap is "
            "padding plus premises the proof never uses.",
        )
    ]
    figs.append(
        line_chart(
            "Failure reasons",
            "share of rejected samples by nd_verify reason (top 5 overall; "
            "only each round's top 5 were logged)",
            _reason_mix(run),
            y_fmt="pct",
            wide=True,
        )
    )
    figs.append(
        line_chart(
            "Accepted against rejected length",
            "mean written lines",
            [
                Series("accepted", series_of(run, "len_accepted"), "var(--s1)"),
                Series("rejected", series_of(run, "len_rejected"), "var(--s2)"),
            ],
            y_fmt="f3",
        )
    )
    figs.append(
        line_chart(
            "Truncation",
            "samples that reached max_new without QED",
            _run_ctl(run, ctl, "trunc_rate"),
            y_fmt="pct",
        )
    )
    # label check from found proofs: fewer written lines than L_true means the label is too high
    by_round: dict[int, list[Rec]] = defaultdict(list)
    for x in run.found:
        if "round" in x:
            by_round[int(x["round"])].append(x)
    if by_round:
        pts_short = [
            (float(r), sum(1 for x in v if x.get("written", 99) < x.get("L_true", 0)) / len(v))
            for r, v in sorted(by_round.items())
        ]
        pts_long = [
            (float(r), sum(1 for x in v if x.get("written", 0) > x.get("L_true", 99)) / len(v))
            for r, v in sorted(by_round.items())
        ]
        figs.append(
            line_chart(
                "New proofs against L_true",
                "written lines of each new distinct proof "
                "(found_targets.jsonl), by round of discovery",
                [
                    Series("written < L_true", pts_short, "var(--s1)"),
                    Series("written > L_true", pts_long, "var(--s2)"),
                ],
                y_fmt="pct",
                note="A verified proof with fewer written lines than L_true means the label "
                "is too high (minlen searches a restricted formula space) or a verifier "
                "bug. Longer than L_true = not minimal (padding or a detour).",
            )
        )
    else:
        figs.append(missing_fig("New proofs against L_true", "needs found_targets.jsonl"))
    if run.examples:
        rs = sorted(run.examples)
        pick = sorted({rs[0], rs[len(rs) // 2], rs[-1]})
        parts = []
        for r in pick:
            ex = run.examples[r]
            rows = "".join(
                f"<tr><td>{esc(kind)}</td><td>{esc(e.get('name'))}</td><td>{esc(e.get('L_true'))}</td>"
                f"<td>{esc(e.get('written'))}</td><td>{esc(e.get('reason') or '')}</td>"
                f"<td>{esc(e.get('prompt'))}<br>{esc(e.get('proof'))}</td></tr>"
                for kind in ("accepted", "rejected")
                for e in ex.get(kind, [])
            )
            parts.append(
                f"<details{' open' if r == pick[-1] else ''}><summary>round {r}</summary>"
                '<div class="tw"><table class="ex"><tr><th>verdict</th><th>target</th>'
                "<th>L_true</th><th>written</th><th>reason</th><th>prompt / proof</th></tr>"
                f"{rows}</table></div></details>"
            )
        figs.append(
            figure(
                "Example proofs",
                "8 accepted and 8 rejected samples per round, drawn with the "
                "instrumentation's own RNG (first, middle and last round; all rounds in "
                "examples.jsonl)",
                "".join(parts),
                wide=True,
            )
        )
    else:
        figs.append(missing_fig("Example proofs", "needs examples.jsonl (seed_dash recipe)"))
    return figs


def dash_retention(run: Run, ctl: Run | None) -> list[str]:
    ho = [
        Series("held-out greedy (200 records, every round)", series_of(run, "heldout"), "var(--s1)")
    ]
    fin = run.final_eval or {}
    if fin.get("heldout_greedy") is not None and run.rounds:
        ho.append(
            Series(
                "final check (harness, 1,000 records)",
                [(float(run.rounds[-1]), fin["heldout_greedy"])],
                "var(--s2)",
            )
        )
    figs = [
        line_chart(
            "Held-out cap-6 greedy accuracy",
            "round 0 = the start model",
            ho,
            y_fmt="pct",
            note="The harness's final check is the only held-out number for runs that did not "
            "log per-round checks.",
        )
    ]
    grid = {r: run.rows[r].get("heldout_by_len", {}) for r in run.rounds}
    lens = sorted({L for v in grid.values() for L in v}, key=int)
    oc = ["var(--o1)", "var(--o2)", "var(--o3)", "var(--o4)", "var(--o5)"]
    figs.append(
        line_chart(
            "Held-out accuracy by proof length",
            "greedy, 40 records per length",
            [
                Series(
                    f"{L} lines",
                    [(float(r), grid[r][L]) for r in run.rounds if L in grid[r]],
                    oc[i % 5],
                )
                for i, L in enumerate(lens)
            ],
            y_fmt="pct",
            note="Forgetting shows first at the longest retained lengths. 40 records per length: "
            "one standard error is about 5 points at 90% accuracy.",
        )
    )
    figs.append(
        line_chart(
            "KL to the start model",
            "nats per token",
            [
                Series(
                    "fixed held-out set (reference proofs)",
                    series_of(run, "kl_to_init"),
                    "var(--s1)",
                ),
                Series("this round's samples", series_of(run, "kl_to_init_sampled"), "var(--s2)"),
            ],
            y_fmt="sci",
        )
    )
    share = [
        (x, n / (n + m))
        for (x, n), (_, m) in zip(
            series_of(run, "n_rl_records"), series_of(run, "n_retained"), strict=False
        )
        if n + m
    ]
    figs.append(
        line_chart(
            "RL records in the fine-tune mix",
            "found-proof records / (found-proof + retained)",
            [Series(run.label, share, RUN_C)],
            y_fmt="pct",
        )
    )
    return figs


def _hazard(
    hist: dict[str, int], risk: dict[str, int], min_risk: int = 100
) -> list[tuple[float, float]]:
    return [
        (float(p), hist.get(p, 0) / risk[p]) for p in sorted(risk, key=int) if risk[p] >= min_risk
    ]


def dash_length_wall(run: Run, ctl: Run | None) -> list[str]:
    figs = [
        heatmap(
            "Attempts long enough",
            "share of samples with at least L_true written lines",
            {r: run.rows[r].get("long_enough_by_len", {}) for r in run.rounds},
            v_fmt="pct",
            v_max=1.0,
        )
    ]
    figs.append(
        line_chart(
            "Attempts long enough, all lengths",
            "share of samples with >= L_true lines",
            _run_ctl(run, ctl, "long_enough"),
            y_fmt="pct",
        )
    )
    rs = [r for r in run.rounds if run.rows[r].get("first_error_line_hist")]
    if rs:
        pick = sorted({rs[0], rs[len(rs) // 2], rs[-1]})
        hz = [
            Series(
                f"round {r}",
                _hazard(run.rows[r]["first_error_line_hist"], run.rows[r]["lines_at_risk_hist"]),
                c,
            )
            for r, c in zip(pick, ("var(--s1)", "var(--s2)", "var(--s3)"), strict=False)
        ]
        if ctl is not None and ctl.rounds:
            cr = [r for r in ctl.rounds if ctl.rows[r].get("first_error_line_hist")]
            if cr:
                agg_h, agg_r = Counter(), Counter()
                for r in cr:
                    agg_h.update(ctl.rows[r]["first_error_line_hist"])
                    agg_r.update(ctl.rows[r]["lines_at_risk_hist"])
                hz.append(
                    Series(
                        "frozen control, all rounds",
                        _hazard(dict(agg_h), dict(agg_r)),
                        CTL_C,
                        muted=True,
                    )
                )
        figs.append(
            line_chart(
                "Per-line failure hazard",
                "P(first error at line p | lines 1..p-1 valid), "
                "training samples; positions with >= 100 at risk",
                hz,
                x_label="line",
                y_fmt="pct",
                note="A spike where training data ends points to a length-specific "
                "skill; a flat hazard means errors are spread evenly.",
            )
        )
    else:
        figs.append(
            missing_fig("Per-line failure hazard", "needs first_error_line_hist (seed_dash recipe)")
        )
    figs.append(
        heatmap(
            "Attempts ending on the conclusion",
            "last written line states the theorem's conclusion at depth 0 (valid or not)",
            {r: run.rows[r].get("ends_on_conclusion_by_len", {}) for r in run.rounds},
            v_fmt="pct",
            v_max=1.0,
        )
    )
    agg = Counter()
    for r in run.rounds:
        agg.update(run.rows[r].get("cite_dist_hist") or {})
    src = "verified samples, all rounds"
    if not agg and run.found:
        for x in run.found:
            lines = _parse_lines(x.get("proof", ""))
            for idx, rule, refs in lines:
                if idx is not None and rule not in ("PR", "AS"):
                    agg.update(str(idx - q) for q in refs)
        src = "distinct found proofs (found_targets.jsonl; derived)"
    if agg:
        ds = sorted(agg, key=int)
        total = sum(agg.values())
        cover = [(float(d), sum(v for k, v in agg.items() if int(k) >= int(d)) / total) for d in ds]
        figs.append(
            line_chart(
                "Citation distance coverage",
                f"share of citations at distance >= d ({src})",
                [Series(run.label, cover, RUN_C)],
                x_label="citation distance d (lines back)",
                y_fmt="pct",
                note=f"max distance {max(int(d) for d in ds)}; {total:,} citations",
            )
        )
    else:
        figs.append(
            missing_fig("Citation distance coverage", "needs cite_dist_hist or found_targets.jsonl")
        )
    figs.append(
        line_chart(
            "Largest citation distance per round",
            "verified samples",
            _run_ctl(run, ctl, "cite_dist_max"),
            y_fmt="int",
        )
    )
    grid = {r: run.rows[r].get("ref_logp_by_len", {}) for r in run.rounds}
    lens = sorted({L for v in grid.values() for L in v}, key=int)
    oc = ["var(--o2)", "var(--o4)", "var(--o5)"]
    figs.append(
        line_chart(
            "Ignition set: log-prob of known proofs",
            "mean per token, teacher-forced; 300 "
            "non-sealed targets at L_true 9-10 whose proofs are never trained on",
            [
                Series(
                    f"L_true {L}",
                    [(float(r), grid[r][L]) for r in run.rounds if L in grid[r]],
                    oc[i % 3],
                )
                for i, L in enumerate(lens)
            ],
            y_fmt="f3",
            note="Should rise before the model starts finding proofs at these lengths.",
        )
    )
    if run.eval_dir is not None:
        ev = load_eval(run.eval_dir)
        figs.append(
            line_chart(
                "Test-time pass@1 by length",
                f"std sets, k = 32, T = 0.8 ({run.eval_dir.name})",
                [
                    Series(k, [(float(L), v) for L, v in sorted(ev[k].items())], c)
                    for k, c in (
                        ("pass1", "var(--s1)"),
                        ("pass4", "var(--s2)"),
                        ("pass32", "var(--s3)"),
                    )
                ],
                x_label="test length L_true",
                y_fmt="pct",
            )
        )
        figs.append(
            line_chart(
                "Test-time attempts long enough",
                "share of samples with >= L_true written lines",
                [
                    Series(
                        run.label,
                        [(float(L), v) for L, v in sorted(ev["long_enough"].items())],
                        RUN_C,
                    )
                ],
                x_label="test length L_true",
                y_fmt="pct",
            )
        )
        groups = (
            ("L6-11", range(6, 12), "var(--s1)"),
            ("L12-14", range(12, 15), "var(--s2)"),
            ("L15-18", range(15, 19), "var(--s3)"),
        )
        hz = []
        for name, rng, c in groups:
            vals: dict[int, list[float]] = defaultdict(list)
            for L, h in ev["hazard"].items():
                if L in rng:
                    for p, v in h.items():
                        vals[p].append(v)
            hz.append(
                Series(name, [(float(p), sum(v) / len(v)) for p, v in sorted(vals.items())], c)
            )
        figs.append(
            line_chart(
                "Test-time per-line hazard",
                "mean over the lengths in each group; positions with >= 100 samples at risk",
                hz,
                x_label="line",
                y_fmt="pct",
            )
        )
    else:
        figs.append(missing_fig("Test-time length wall", "no evaluation directory for this run"))
    return figs


def _parse_lines(proof: str) -> list[tuple[int | None, str | None, list[int]]]:
    toks = proof.split()
    if "QED" in toks:
        toks = toks[: toks.index("QED")]
    out, cur = [], []
    for t in toks:
        cur.append(t)
        if t == ";":
            idx = int(cur[0][1:]) if cur[0][:1] == "N" and cur[0][1:].isdigit() else None
            j = len(cur) - 1 - cur[::-1].index(":") if ":" in cur else None
            rule = cur[j + 1] if j is not None and j + 1 < len(cur) - 1 else None
            refs = [int(x[1:]) for x in cur[(j or 0) + 2 : -1] if x[:1] == "N" and x[1:].isdigit()]
            out.append((idx, rule, refs if j is not None else []))
            cur = []
    return out


def group_colors(runs: list[Run]) -> dict[str, str]:
    out: dict[str, str] = {}
    for r in runs:
        if r.group not in out:
            k = len(out)
            out[r.group] = f"var(--s{k + 1})" if k < N_SLOTS else CTL_C
    return out


def dash_comparison(runs: list[Run], ctl: Run | None) -> list[str]:
    col = group_colors(runs)
    over = len(col) > N_SLOTS
    rows = []
    for r in runs:
        s = r.summary
        comp = s.get("compute") or {}
        ev = load_eval(r.eval_dir) if r.eval_dir else None
        p1 = ev["pass1"] if ev else {}
        score = None
        if s.get("pass1") is not None and s.get("frontier_pass1") is not None:
            score = (s["pass1"] + s["pass4"]) * (s.get("score") or {}).get("n", 1200) + 4 * (
                s["frontier_pass1"] + s["frontier_pass4"]
            ) * (s.get("frontier_score") or {}).get("n", 1000)
        rows.append(
            [
                r.label,
                str(len([x for x in r.rounds if x > 0])),
                fmt(comp.get("used_flops"), "sci"),
                fmt(s.get("heldout_greedy"), "pct"),
                fmt(s.get("pass1"), "f3"),
                fmt(s.get("pass4"), "f3"),
                fmt(s.get("frontier_pass1"), "f3"),
                fmt(s.get("frontier_pass4"), "f3"),
                fmt(score, "g"),
            ]
            + [fmt(p1.get(L), "pct") for L in (8, 11, 12, 14, 16, 18)]
        )
    note = (
        "Scores: dev and frontier from the score daemon (summary.json); "
        "score = (dev pass@1 + pass@4) x n + 4 x (frontier pass@1 + pass@4) x n. "
        "Eval pass@1: the run's evaluation directory, std sets, k = 32, T = 0.8."
    ) + (" More groups than colour slots: the rest are grey." if over else "")
    figs = [
        table_fig(
            "Runs and final scores",
            "one row per run",
            [
                "run",
                "rounds",
                "compute used",
                "held-out",
                "dev pass@1",
                "dev pass@4",
                "frontier pass@1",
                "frontier pass@4",
                "score",
                "eval L8",
                "eval L11",
                "eval L12",
                "eval L14",
                "eval L16",
                "eval L18",
            ],
            rows,
            note,
        )
    ]

    def over_runs(key: str) -> list[Series]:
        out = [Series(r.label, series_of(r, key), col[r.group]) for r in runs]
        if ctl is not None:
            out.append(
                Series(f"frozen control ({ctl.label})", series_of(ctl, key), CTL_C, muted=True)
            )
        return out

    legend_note = (
        "Colour = run group (seeds of one arm share a colour); the tooltip names each run."
    )
    figs.append(
        line_chart(
            "Solve rate per round",
            "window targets solved this round",
            over_runs("solve_rate"),
            y_fmt="pct",
            note=legend_note,
        )
    )
    figs.append(
        line_chart(
            "Share of the run's targets solved so far",
            "cumulative",
            over_runs("solved_cum_frac"),
            y_fmt="pct",
        )
    )
    figs.append(
        line_chart(
            "All-fail share",
            "window targets with no verified sample",
            over_runs("frac_all_fail"),
            y_fmt="pct",
        )
    )
    figs.append(
        line_chart(
            "Held-out greedy per round",
            "dash checks (missing for runs without them)",
            over_runs("heldout"),
            y_fmt="pct",
        )
    )
    evs = [(r, load_eval(r.eval_dir)) for r in runs if r.eval_dir]
    if evs:
        figs.append(
            line_chart(
                "Test-time pass@1 by length",
                "std sets, k = 32, T = 0.8",
                [
                    Series(
                        r.label,
                        [(float(L), v) for L, v in sorted(e["pass1"].items())],
                        col[r.group],
                    )
                    for r, e in evs
                ],
                x_label="test length L_true",
                y_fmt="pct",
                wide=True,
            )
        )
        figs.append(
            line_chart(
                "Test-time attempts long enough",
                "share of samples with >= L_true written lines",
                [
                    Series(
                        r.label,
                        [(float(L), v) for L, v in sorted(e["long_enough"].items())],
                        col[r.group],
                    )
                    for r, e in evs
                ],
                x_label="test length L_true",
                y_fmt="pct",
            )
        )
    return figs


# ---------------------------------------------------------------- render


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def render(runs: list[Run], ctl: Run | None, out: Path, title: str = "") -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    builders = {
        "1_limiting": dash_limiting,
        "2_diversity": dash_diversity,
        "3_integrity": dash_integrity,
        "4_retention": dash_retention,
        "6_length_wall": dash_length_wall,
    }
    for run in runs + ([ctl] if ctl is not None and ctl not in runs else []):
        d = out / slug(run.label)
        d.mkdir(exist_ok=True)
        c = None if run is ctl else ctl
        sub = (
            f"{run.path} | {len([r for r in run.rounds if r > 0])} rounds | control: "
            f"{c.label if c else 'none'} | eval: {run.eval_dir or 'none'}"
        )
        for key, name in DASHBOARDS:
            if key == "5_comparison":
                continue
            figs = builders[key](run, c) + [coverage_fig(run)]
            p = d / f"{key}.html"
            p.write_text(
                page(f"{run.label}: {name}", sub, key, figs, comparison="../5_comparison.html")
            )
            written.append(p)
    p = out / "5_comparison.html"
    first = slug(runs[0].label) + "/" if runs else ""
    p.write_text(
        page(
            title or "Comparison",
            f"{len(runs)} runs; control: {ctl.label if ctl else 'none'}",
            "5_comparison",
            dash_comparison(runs, ctl),
            prefix=first,
            comparison="5_comparison.html",
        )
    )
    written.append(p)
    idx = ["<ul>"] + [
        f'<li><a href="{esc(slug(r.label))}/1_limiting.html">{esc(r.label)}</a> '
        f'<span class="sub">{esc(r.path)}</span></li>'
        for r in runs
    ]
    if ctl is not None:
        idx.append(
            f'<li><a href="{esc(slug(ctl.label))}/1_limiting.html">{esc(ctl.label)}</a> '
            "(frozen control)</li>"
        )
    idx.append('</ul><p><a href="5_comparison.html">Comparison across runs</a></p>')
    body = figure("Runs", "open a run's dashboards", "".join(idx), wide=True)
    ip = out / "index.html"
    ip.write_text(
        page(
            title or "RL run dashboards",
            str(out),
            "",
            [body],
            prefix=first,
            comparison="5_comparison.html",
        )
    )
    written.append(ip)
    return written


# ---------------------------------------------------------------- W&B mirror (dry-run by default)


def flatten_record(rec: Rec) -> Rec:
    """One metrics.jsonl record -> flat W&B keys `<phase>/<field>`.

    Nested dicts join with '.', fail_reasons becomes fail/<reason>, lists and strings are dropped;
    the record's step stays the x-axis (logged as rl_step)."""
    ph = rec.get("phase", "misc")
    out: Rec = {}

    def walk(prefix: str, v: Any) -> None:
        if isinstance(v, dict):
            for k, x in v.items():
                walk(f"{prefix}.{k}", x)
        elif isinstance(v, bool):
            out[prefix] = int(v)
        elif isinstance(v, (int, float)):
            out[prefix] = v

    for k, v in rec.items():
        if k in ("t", "phase", "step"):
            continue
        if k == "fail_reasons" and isinstance(v, list):
            for item in v:
                if isinstance(item, list) and len(item) == 2:
                    out[f"{ph}/fail/{item[0]}"] = item[1]
            continue
        walk(f"{ph}/{k}", v)
    return out


def wandb_sync(
    runs: list[Run], project: str, entity: str | None, mode: str, wdir: str = "wandb_local"
) -> None:
    if mode == "dry-run":
        for run in runs:
            keys: set[str] = set()
            for rec in run.records:
                keys |= set(flatten_record(rec))
            print(
                f"{run.label}: {len(run.records)} records, {len(keys)} keys, "
                f"e.g. {sorted(keys)[:6]}"
            )
        print("dry-run: nothing uploaded")
        return
    import wandb  # noqa: PLC0415  (only needed for offline / online)

    for run in runs:
        cfg = {
            k: run.summary.get(k)
            for k in ("seed", "flops_budget", "gpu", "started_utc", "run_name", "arm")
        }
        wr = wandb.init(
            project=project,
            entity=entity,
            name=run.label,
            group=run.group,
            config=cfg,
            mode=mode,
            reinit=True,
            dir=wdir,
        )
        wandb.define_metric("rl_step")
        wandb.define_metric("*", step_metric="rl_step")
        for rec in run.records:
            wandb.log({**flatten_record(rec), "rl_step": rec.get("step", 0)})
        for k in ("metric", "pass1", "pass4", "frontier_pass1", "frontier_pass4", "heldout_greedy"):
            if run.summary.get(k) is not None:
                wr.summary[k] = run.summary[k]
        wr.finish()


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("render", help="write the HTML dashboards")
    r.add_argument("runs", nargs="+")
    r.add_argument("--out", required=True)
    r.add_argument(
        "--control", default=None, help="frozen-control run directory (overlaid as the baseline)"
    )
    r.add_argument(
        "--eval", action="append", default=[], help="LABEL=DIR evaluation directory override"
    )
    r.add_argument("--title", default="")
    f = sp.add_parser("fields", help="print which dashboard fields each run has")
    f.add_argument("runs", nargs="+")
    w = sp.add_parser("wandb", help="mirror runs to W&B (dry-run unless --mode offline/online)")
    w.add_argument("runs", nargs="+")
    w.add_argument("--project", default="nd-rl-hillclimb")
    w.add_argument("--entity", default=None)
    w.add_argument("--mode", default="dry-run", choices=("dry-run", "offline", "online"))
    w.add_argument("--confirm-upload", action="store_true", help="required with --mode online")
    w.add_argument("--dir", default="wandb_local", help="local W&B directory (offline files)")
    a = ap.parse_args(argv)
    evs = {}
    for e in getattr(a, "eval", []) or []:
        k, _, v = e.partition("=")
        evs[k] = Path(v)
    runs = [load_run(x, evs) for x in a.runs]
    if a.cmd == "render":
        ctl = load_run(a.control) if a.control else None
        for p in render(runs, ctl, Path(a.out), a.title):
            print(p)
    elif a.cmd == "fields":
        for run in runs:
            have = {k: v for k, v in run.source.items() if k in FIELDS or k in BY_LEN_FIELDS}
            miss = [k for k in list(FIELDS) + list(BY_LEN_FIELDS) if k not in have]
            logged = sorted(k for k, v in have.items() if v == "logged")
            print(f"{run.label}: {len(run.rounds)} rounds; logged {logged}")
            print(f"  derived {sorted(k for k, v in have.items() if v != 'logged')}")
            print(f"  missing {miss}")
    else:
        if a.mode == "online" and not a.confirm_upload:
            raise SystemExit(
                "rl_dash: --mode online uploads to W&B; pass --confirm-upload once approved"
            )
        Path(a.dir).mkdir(parents=True, exist_ok=True)
        wandb_sync(runs, a.project, a.entity, a.mode, a.dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
