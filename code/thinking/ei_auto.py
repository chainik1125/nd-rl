"""Expert iteration over the effort choice: the model learns to pick its own think level.

    python -m code.thinking.ei_auto --init DIR/model.pt --name A_s0 --out EIDIR \
        --targets pool.jsonl --train S1g3.pkl --levels 0,64,256 --choice_k 11,11,10 \
        [--choice_lambda 0] [--choice_keep 0] [--rounds 1]

Per round r, with the current checkpoint (a `lean_thinke` model; a `lean_thinkn` init gets the
effort tokens added, as `ei --effort` does):
1. **Sample at every level.**  Each target is attempted k_l times at each fixed level l of
   `--levels` (`--choice_k`, default 11 / 11 / 10 = 32 attempts per target, the attempts of one
   standard EI round), plain arm, Lean-judged, exactly as `ei.sample_round` samples.
2. **Label.**  Per target, l* = argmax over the levels with at least one success of
   pass rate - lambda * (mean sampled tokens per attempt at that level) / 1000
   (`--choice_lambda`, default 0), ties to the cheaper (lower) level (`choice_label`).  A target
   with no success at any level gets no label and contributes nothing.  The label is the model's own
   outcome, not a property of the search that wrote the Stage-1 traces.
3. **Keep.**  The target's distinct successful proofs at l* (distinct = the applied action sequence,
   as `ei.entry_key`), shortest first (sampled tokens), at most `--choice_keep` (0: `--max_per_thm`,
   the data volume a standard EI round keeps per target).
4. **Fine-tune** exactly as an EI round (`ei.build_mix`'s mix: kept proofs x `--rl_weight` plus
   `--retain` random Stage-1 records; `train.finetune` with the ladder's settings), except that
   the kept proofs are chosen-effort pairs (`ei.entry_pairs(choice=True)`):
   `state | <e{l*}> <think> ...`, the effort token a target at every step.  Retained Stage-1
   records keep their layout (`state <eB> | <think> ...`), so the model keeps following a
   prefilled level.

At eval time the model then picks its level per step (`eval --budgets auto`,
`sample.choose_levels`).  round_<r>.json logs the label distribution, per-level pass rates and
tokens per attempt, the labelled pass rate against the best fixed level's (in-sample: the label is
chosen on the same attempts, so this is the training-pool oracle, an upper bound), and the mix;
found_<r>.jsonl holds the kept proofs with their label.  Files otherwise as `ei`:
ckpts/<name>_r<r>.pt, args.json, registry/, lean_gate.jsonl.
`--transfer` / `--heldout` are not supported; `--budgets`, `--k` and `--select` are ignored.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from . import ei
from .ckpt import load_ckpt, run_env, save_ckpt
from .dan import bare, load
from .data import Pair, load_pairs, read_jsonl
from .effort import LEVELS, has_effort, relabel_records
from .train import finetune

D = load()


def choice_label(stats: Sequence[tuple[int, int, int, float]], lam: float = 0.0) -> int | None:
    """(level, successes, attempts, mean sampled tokens per attempt) per level -> the label: the
    level with the highest pass rate - lam * tokens / 1000 among the levels with a success; ties go
    to the lower level.  None if no level has a success."""
    best: tuple[float, int] | None = None
    for lv, c, n, tpa in sorted(stats):
        if c <= 0 or n <= 0:
            continue
        score = c / n - lam * tpa / 1000.0
        if best is None or score > best[0] + 1e-12:
            best = (score, lv)
    return None if best is None else best[1]


def level_stats(atts: Sequence[dict[str, Any]]) -> tuple[int, int, float]:
    """One target's attempts at one level -> (successes, attempts, mean sampled tokens)."""
    n = len(atts)
    return (
        sum(bool(x["ok"]) for x in atts),
        n,
        sum(x["tokens"] for x in atts) / max(1, n),
    )


def kept_entries(
    atts: Sequence[dict[str, Any]],
    tok: Any,
    target: dict[str, Any],
    r: int,
    keep: int,
    forced: str,
) -> list[dict[str, Any]]:
    """The distinct successful proofs among `atts` (one level's attempts), shortest first (sampled
    tokens; ties keep the earlier attempt), at most `keep`, usable under the --forced policy."""
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    ok = sorted((x for x in atts if x["ok"]), key=lambda x: x["tokens"])
    for x in ok:
        e = ei.found_entry(x, tok, target, r, True)
        k = ei.entry_key(e)
        if k in seen or not ei.usable(e, forced):
            continue
        seen.add(k)
        out.append(e)
        if len(out) >= keep:
            break
    return out


def label_round(
    targets: Sequence[dict[str, Any]],
    per_level: dict[int, list[list[dict[str, Any]]]],
    lam: float,
    keep: int,
    tok: Any,
    r: int,
    forced: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Labels and kept proofs for every target -> (name -> kept entries, the label stats)."""
    levels = sorted(per_level)
    kept: dict[str, list[dict[str, Any]]] = {}
    labels: collections.Counter[int] = collections.Counter()
    p_lab: list[float] = []
    p_fixed: dict[int, list[float]] = {lv: [] for lv in levels}
    tok_lab: list[float] = []
    for ti, t in enumerate(targets):
        st = {lv: level_stats(per_level[lv][ti]) for lv in levels}
        for lv, (c, n, _) in st.items():
            p_fixed[lv].append(c / max(1, n))
        lab = choice_label([(lv, *st[lv]) for lv in levels], lam)
        if lab is None:
            p_lab.append(0.0)
            continue
        c, n, tpa = st[lab]
        p_lab.append(c / max(1, n))
        tok_lab.append(tpa)
        labels[lab] += 1
        es = kept_entries(per_level[lab][ti], tok, t, r, keep, forced)
        for e in es:
            e["label"] = lab
            e["level_stats"] = {str(lv): list(st[lv]) for lv in levels}
        if es:
            kept[t["name"]] = es
    n_t = max(1, len(targets))
    mean_fixed = {lv: 100.0 * sum(v) / n_t for lv, v in p_fixed.items()}
    best_lv = max(levels, key=lambda lv: (mean_fixed[lv], -lv))
    n_lab = sum(labels.values())
    return kept, {
        "labelled": n_lab,
        "unlabelled": len(targets) - n_lab,
        "labels": {str(lv): labels.get(lv, 0) for lv in levels},
        "label_share_pct": {str(lv): 100.0 * labels.get(lv, 0) / max(1, n_lab) for lv in levels},
        "pass_rate_pct_by_level": {str(lv): v for lv, v in mean_fixed.items()},
        "pass_rate_pct_at_label": 100.0 * sum(p_lab) / n_t,
        "best_fixed_level": best_lv,
        "label_minus_best_fixed_pp": 100.0 * sum(p_lab) / n_t - mean_fixed[best_lv],
        "tokens_per_attempt_at_label": sum(tok_lab) / max(1, len(tok_lab)),
        "kept_entries": sum(len(v) for v in kept.values()),
        "kept_targets": len(kept),
    }


def choice_mix(
    targets: Sequence[dict[str, Any]],
    kept: dict[str, list[dict[str, Any]]],
    train_records: list[list[Pair]],
    tok: Any,
    rng: random.Random,
    rl_weight: int,
    retain: int,
    validate: str,
) -> tuple[list[list[Pair]], dict[str, int]]:
    """`ei.build_mix` with chosen-effort pairs for the kept proofs; retained records unchanged."""
    mix: list[list[Pair]] = []
    n_rl = n_steps_dropped = n_targets = 0
    for t in targets:
        used = 0
        for e in kept.get(t["name"], []):
            pairs = ei.entry_pairs(e, tok, validate, choice=True)
            n_steps_dropped += len(e["steps"]) - len(pairs)
            if not pairs:
                continue
            used += 1
            for _ in range(rl_weight):
                mix.append(pairs)
                n_rl += 1
        n_targets += used > 0
    for rec in rng.sample(train_records, min(retain, len(train_records))):
        mix.append(rec)
    return mix, {
        "mix_rl_records": n_rl,
        "mix_retained": len(mix) - n_rl,
        "mix_targets": n_targets,
        "steps_dropped": n_steps_dropped,
    }


def parser() -> argparse.ArgumentParser:
    ap = ei.parser()
    ap.description = __doc__
    ap.add_argument("--levels", default="0,64,256", help="the effort levels the model picks from")
    ap.add_argument("--choice_k", default="11,11,10", help="attempts per target at each level")
    ap.add_argument(
        "--choice_lambda", type=float, default=0.0, help="token cost per 1,000 in the label"
    )
    ap.add_argument(
        "--choice_keep", type=int, default=0, help="proofs kept per target (0: --max_per_thm)"
    )
    return ap


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def run(a: argparse.Namespace) -> list[dict[str, Any]]:
    levels, ks = _ints(a.levels), _ints(a.choice_k)
    if len(levels) != len(ks) or not levels or any(k <= 0 for k in ks):
        raise SystemExit(f"error: --levels {a.levels} and --choice_k {a.choice_k} must pair up")
    if sorted(set(levels)) != sorted(levels) or any(lv not in LEVELS for lv in levels):
        raise SystemExit(f"error: --levels {a.levels}: distinct effort levels of {LEVELS}")
    if a.transfer or a.heldout:
        raise SystemExit("error: ei_auto does not support --transfer / --heldout")
    a.effort = True
    a.forced = ei.forced_policy(a.forced, True)
    keep = a.choice_keep or a.max_per_thm
    out = Path(a.out)
    run_env(out)
    with bare():
        D.record.save_config(vars(a), str(out) + "/", arm=a.name, role="rl")
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    targets = read_jsonl(a.targets)
    for i, t in enumerate(targets):
        t.setdefault("name", f"target_{i}")
    train_data = None if a.no_train else load_pairs(a.train)
    if train_data is not None and train_data["mode"] != "lean_thinke":
        train_data = relabel_records(train_data)
        print(f"effort levels of the retained records: {train_data['report']['effort_levels']}")
    rng = random.Random(a.seed * 7919)
    if a.resume:
        ckpt = str(out / "ckpts" / f"{a.name}_r{a.start_round - 1}.pt")
    else:
        ckpt = str(ei.effort_init(a.init, out / "ckpts" / f"{a.name}_r0.pt", seed=a.seed))
    history = []
    for r in range(a.start_round, a.start_round + a.rounds):
        t0 = time.time()
        D.record.phase("sample", round=r)
        model, tok, extra = load_ckpt(ckpt, dev)
        assert has_effort(tok), "ei_auto needs a lean_thinke checkpoint"
        assert train_data is None or train_data["itos"] == list(tok.itos), "--train vocabulary"
        seed = a.seed * 1000 + r
        per_level: dict[int, list[list[dict[str, Any]]]] = {}
        stats: dict[str, Any] = {
            "round": r,
            "ckpt": ckpt,
            "seed": seed,
            "levels": levels,
            "choice_k": ks,
            "choice_lambda": a.choice_lambda,
            "choice_keep": keep,
            "per_level": {},
        }
        for li, (lv, k) in enumerate(zip(levels, ks, strict=True)):
            a.budget_list = [lv]
            per, st = ei.sample_round(model, tok, targets, a, seed * 10 + li, True, True, k=k)
            per_level[lv] = per
            stats["per_level"][str(lv)] = {
                **ei.round_summary(per, k),
                "sampler": {
                    k2: (dict(v) if isinstance(v, collections.Counter) else v)
                    for k2, v in st.items()
                },
            }
        kept, lab = label_round(targets, per_level, a.choice_lambda, keep, tok, r, a.forced)
        stats["labels"] = lab
        if a.validate_think != "off":
            for es in kept.values():
                for e in es:
                    e["think_valid"] = ei.validate_entry(e)
        with open(out / f"found_{r}.jsonl", "w") as f:
            for t in targets:
                for e in kept.get(t["name"], []):
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
        D.record.phase("finetune", round=r)
        if not a.no_train:
            assert train_data is not None
            mix, ms = choice_mix(
                targets,
                kept,
                train_data["records"],
                tok,
                rng,
                a.rl_weight,
                a.retain,
                a.validate_think,
            )
            stats.update(ms)
            if ms["mix_rl_records"]:
                model, last = finetune(
                    model,
                    tok,
                    mix,
                    a.ft_steps,
                    a.ft_lr,
                    a.ft_lr / 10,
                    50,
                    0.1,
                    a.ft_recs,
                    seed,
                    dev,
                    a.think_weight,
                )
                ckpt = str(out / "ckpts" / f"{a.name}_r{r}.pt")
                save_ckpt(
                    ckpt,
                    model,
                    tok.mode,
                    extra={
                        **extra,
                        "ei_round": r,
                        "ei_init": a.init,
                        "ft_last_loss": last,
                        "args": vars(a),
                        "auto_effort": {"levels": levels, "choice_lambda": a.choice_lambda},
                    },
                )
            else:
                print("no labelled targets: skipping training this round", flush=True)
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()
        stats["secs"] = time.time() - t0
        (out / f"round_{r}.json").write_text(json.dumps(stats, indent=1, default=str))
        share = " ".join(f"e{lv} {v:.1f}%" for lv, v in lab["label_share_pct"].items())
        rates = " ".join(f"e{lv} {v:.1f}%" for lv, v in lab["pass_rate_pct_by_level"].items())
        print(
            f"=== round {r}: labelled {lab['labelled']}/{len(targets)} (labels {share}); "
            f"pass rate by level {rates}; at label {lab['pass_rate_pct_at_label']:.1f}% "
            f"({lab['label_minus_best_fixed_pp']:+.1f} pp vs best fixed "
            f"e{lab['best_fixed_level']}); "
            f"kept {lab['kept_entries']}, {stats['secs']:.0f}s",
            flush=True,
        )
        history.append(stats)
    D.record.phase(None)
    return history


def main(argv: list[str] | None = None) -> int:
    run(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
