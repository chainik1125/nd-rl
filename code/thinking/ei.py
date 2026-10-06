"""Expert iteration with thinking: the fork's `state_ladder_ei.py` T1 rung (uniform allocation) in
this pipeline.

    python -m code.thinking.ei --init DIR/model.pt --name T2_s0 --out EIDIR \
        --targets rl_targets.jsonl \
        --train pairs.pkl --budgets 16,64,256 [--validate_think drop]

Per round r (as the ladder, state_ladder_ei.py:231-387@4dcd0e4f): sample every target k times at
temperature T, judge with Lean, accumulate distinct accepted attempts per target; optionally sample
a transfer set and run held-out greedy; then fine-tune the current checkpoint on <= --max_per_thm
accepted attempts per target x --rl_weight plus --retain random Stage-1 training records (the fork's
plain AdamW loop: 600 steps, lr 3e-4 -> 3e-5, warmup 50, 128 records per step; `train.finetune`).

What changes: the ladder keeps only the ND proof string and re-derives (state, action) pairs from
it, so a sampled think block would vanish.  Here an accepted attempt keeps its **sampled sequence**:
per step the state, the think content the model wrote and the action the environment applied,
renamed from the attempt's name base back to base 0 (the ladder's replay pairs start at base 0 too).
Those are the fine-tune pairs.  Options:
  --budgets 16,64,256      per-attempt think budget drawn uniformly (one value: a fixed budget)
  --validate_think drop    keep only steps whose think block passes trace_dsl.check_step (strict);
  `empty`
                           keeps the step's action with an empty think block instead
  --effort                 effort conditioning (`effort.py`): the init (lean_thinkn) is expanded to
                           lean_thinke, the retained records are relabelled, and every pair's state
                           ends in the attempt's <eB>; budgets must be effort levels (0,16,64,256)
  --select shortest        per target, spread the kept attempts over the budgets they were found at
                           and keep the shortest (sampled tokens) within each: a shortness pressure
                           on the think blocks; `random` (default) is the ladder's choice.  Also
                           picks which think variant of a proof is kept (below)
  --forced drop|keep       attempts with any forced `</think>` (a step's `forced` flag: the sampler
                           closed the block at the budget) still count as solved, but with `drop`
                           they are not fine-tuned on.  Default `drop`, or `keep` with --effort
                           (there a forced close teaches budget compliance; with a hidden budget it
                           teaches stopping at an arbitrary point)

Distinct = a different **action sequence** (the actions the environment applied, renamed to base
0).  Think variants of one proof are one entry, so a thinking model does not get more fine-tune
records per target than the no-think control for the same proofs (LIT_CRITIQUE C6: before this,
79% of T1's solved targets had >= 4 entries against 45% for T0, with the same share of >= 4
distinct proofs).  Per action sequence the entry kept is the first found (`random`) or the one with
the fewest think tokens (`shortest`; ties keep the earlier), and an entry --forced drop would
discard is always replaced by a usable variant of the same proof.  Found records written before
this change carry no `forced` count; it is inferred as a block of >= budget think tokens (exact for
the hard cut, the only forcing those runs used), and a resumed found file is deduplicated by action
sequence.  round_<r>.json reports `entries_per_target` (cumulative kept entries per solved target,
all and usable).  Files in EIDIR: found_<r>.jsonl (cumulative, token strings), round_<r>.json,
ckpts/<name>_r<r>.pt, args.json, registry/, lean_gate.jsonl.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch

from .ckpt import load_ckpt, run_env, save_ckpt
from .dan import bare, load, trace_dsl
from .data import Pair, encode_pair, load_pairs, read_jsonl
from .effort import LEVELS, expand_to_effort, has_effort, relabel_records
from .eval import pass_at_k
from .sample import judge, think_generate
from .tok import END_THINK, THINK, has_think, rebase_names
from .train import finetune

D = load()


def _s(tok: Any, ids: bytes | list[int]) -> str:
    return " ".join(tok.itos[i] for i in ids)


def found_entry(
    x: dict[str, Any], tok: Any, target: dict[str, Any], r: int, think: bool
) -> dict[str, Any]:
    """An accepted attempt (a `think_generate` result with its trace) -> a found record."""
    steps = [s for s in x["trace"] if not s.get("rejected")]
    reb = [
        (
            rebase_names(s["state"], tok, x["base"]),
            rebase_names(s["think"], tok, x["base"]),
            rebase_names(s["applied"], tok, x["base"]),
        )
        for s in steps
    ]
    rebased = all(a is not None and b is not None and c is not None for a, b, c in reb)
    src: list[tuple[Any, Any, Any]] = (
        list(reb) if rebased else [(s["state"], s["think"], s["applied"]) for s in steps]
    )
    return {
        "name": target.get("name"),
        "prompt": target["prompt"],
        "round": r,
        "base": 0 if rebased else x["base"],
        "sample_base": x["base"],
        "budget": x["budget"],
        "think_format": "think" if think else "none",
        "steps": [
            {"state": _s(tok, a), "think": _s(tok, b), "action": _s(tok, c)} for a, b, c in src
        ],
        "think_tokens": sum(len(b) for _, b, _ in src),
        "forced": sum(bool(s.get("forced")) for s in steps),
        "tokens": x["tokens"],
    }


def entry_key(e: dict[str, Any]) -> tuple[str, ...]:
    """Distinctness: the applied action sequence (renamed), whatever the think blocks were."""
    return tuple(s["action"] for s in e["steps"])


def n_forced(e: dict[str, Any]) -> int:
    """Forced `</think>` closes among an entry's steps.  Records from before the `forced` field: a
    block of >= budget think tokens (the hard cut forced exactly those)."""
    if "forced" in e:
        return int(e["forced"])
    b = int(e.get("budget", 0))
    if b <= 0 or e.get("think_format") == "none":
        return 0
    return sum(len(s["think"].split()) >= b for s in e["steps"])


def usable(e: dict[str, Any], forced: str) -> bool:
    """May this entry be fine-tuned on under the --forced policy?"""
    return forced == "keep" or n_forced(e) == 0


def add_found(
    have: dict[tuple[str, ...], int],
    fs: list[dict[str, Any]],
    e: dict[str, Any],
    select: str = "random",
    forced: str = "drop",
) -> str:
    """Add an accepted attempt to a target's entries, one per action sequence.  `have` maps a key
    to its index in `fs`.  -> 'new', 'dup' (an existing entry stays) or 'replaced' (the existing
    variant was unusable under --forced and this one is usable, or --select shortest and this one
    has fewer think tokens)."""
    k = entry_key(e)
    i = have.get(k)
    if i is None:
        have[k] = len(fs)
        fs.append(e)
        return "new"
    old = fs[i]
    uo, un = usable(old, forced), usable(e, forced)
    if (un and not uo) or (
        un == uo and select == "shortest" and e["think_tokens"] < old["think_tokens"]
    ):
        fs[i] = e
        return "replaced"
    return "dup"


def entries_report(found: dict[str, list[dict[str, Any]]], forced: str = "drop") -> dict[str, Any]:
    """Kept entries (distinct proofs) per solved target: all of them, and those usable under the
    --forced policy.  Histograms are keyed 0, 1, 2, 3, '>=4' (max_per_thm is 4 by default)."""

    def hist(ns: list[int]) -> dict[str, Any]:
        h = collections.Counter(str(n) if n < 4 else ">=4" for n in ns)
        return {
            "targets": len(ns),
            "mean": round(sum(ns) / max(1, len(ns)), 3),
            "hist": {k: h.get(k, 0) for k in ("0", "1", "2", "3", ">=4")},
            "ge4_pct": round(100.0 * sum(n >= 4 for n in ns) / max(1, len(ns)), 2),
        }

    solved = [v for v in found.values() if v]
    return {
        "all": hist([len(v) for v in solved]),
        "usable": hist([sum(usable(e, forced) for e in v) for v in solved]),
        "entries": sum(len(v) for v in solved),
        "entries_forced": sum(n_forced(e) > 0 for v in solved for e in v),
    }


def validate_entry(e: dict[str, Any]) -> list[bool]:
    """Per step: does the think block pass trace_dsl.check_step (strict: soft findings fail)?"""
    T = trace_dsl()
    out = []
    with bare():
        env = D.state_env.Env(e["prompt"], canon=True, base=e["base"], assign=False)
        for s in e["steps"]:
            think = [THINK, *s["think"].split(), END_THINK]
            act = s["action"].split()
            try:
                soft = T.check_step(env, think, act, {})
                out.append(not soft)
            except (T.TraceError, D.lean_tok.ParseFail, ValueError, IndexError, KeyError):
                out.append(False)
            ok, _ = env.apply(act)
            if not ok:
                out += [False] * (len(e["steps"]) - len(out))
                break
    return out


def entry_pairs(e: dict[str, Any], tok: Any, validate: str = "off") -> list[Pair]:
    """A found record -> training pairs (state, think block + action + <eos>)."""
    valid = e.get("think_valid") or [True] * len(e["steps"])
    out = []
    for s, ok in zip(e["steps"], valid, strict=True):
        content = s["think"].split()
        if validate != "off" and not ok:
            if validate == "drop":
                continue
            content = []
        block = [THINK, *content, END_THINK] if e["think_format"] == "think" else []
        state = s["state"].split()
        if has_effort(tok):
            state.append(f"<e{e['budget']}>")
        out.append(encode_pair(tok, state, block, s["action"].split()))
    return out


def pick(fs: list[dict[str, Any]], n: int, how: str, rng: random.Random) -> list[dict[str, Any]]:
    """<= n of a target's found attempts: `random` (the ladder), or `shortest`: round-robin over the
    budgets they were found at (ascending), shortest sampled-token count first within a budget."""
    fs = list(fs)
    rng.shuffle(fs)
    if how == "random":
        return fs[:n]
    by: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for e in fs:
        by[int(e.get("budget", 0))].append(e)
    queues = [sorted(by[b], key=lambda e: e["tokens"]) for b in sorted(by)]
    out: list[dict[str, Any]] = []
    while len(out) < n and any(queues):
        for q in queues:
            if q and len(out) < n:
                out.append(q.pop(0))
    return out


def build_mix(
    targets: list[dict[str, Any]],
    found: dict[str, list[dict[str, Any]]],
    train_records: list[list[Pair]],
    tok: Any,
    rng: random.Random,
    max_per_thm: int,
    rl_weight: int,
    retain: int,
    validate: str,
    select: str = "random",
    forced: str = "keep",
) -> tuple[list[list[Pair]], dict[str, int]]:
    """The ladder's mix (state_ladder_ei.py:351-364@4dcd0e4f): <= max_per_thm distinct attempts per
    target x rl_weight, plus `retain` random Stage-1 training records.  With forced='drop' only
    entries without a forced `</think>` are candidates."""
    mix: list[list[Pair]] = []
    n_rl = n_steps_dropped = n_forced_skipped = n_targets = 0
    for t in targets:
        fs = found.get(t["name"], [])
        ok = [e for e in fs if usable(e, forced)]
        n_forced_skipped += len(fs) - len(ok)
        used = 0
        for e in pick(ok, max_per_thm, select, rng):
            pairs = entry_pairs(e, tok, validate)
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
        "entries_forced_skipped": n_forced_skipped,
        "steps_dropped": n_steps_dropped,
    }


def sample_round(
    model: Any,
    tok: Any,
    recs: list[dict[str, Any]],
    a: argparse.Namespace,
    seed: int,
    think: bool,
    keep_trace: bool,
    greedy: bool = False,
    k: int = 0,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    k = k or a.k
    prompts = [t["prompt"] for t in recs for _ in range(k)]
    st: dict[str, Any] = {}
    res = think_generate(
        model,
        tok,
        prompts,
        a.arm,
        a.budget_list if think else 0,
        a.min_think,
        think,
        a.temperature,
        greedy,
        a.max_action,
        a.max_steps,
        a.max_rej,
        a.batch,
        seed,
        st,
        keep_trace=keep_trace,
    )
    res = judge(tok, prompts, res, a.lean)
    return [res[i * k : (i + 1) * k] for i in range(len(recs))], st


def round_summary(per: list[list[dict[str, Any]]], k: int) -> dict[str, Any]:
    n = len(per)
    ok = [sum(bool(x["ok"]) for x in att) for att in per]
    acc = [x for att in per for x in att if x["ok"]]
    return {
        "n": n,
        "solved": sum(c > 0 for c in ok),
        "solved_pct": 100.0 * sum(c > 0 for c in ok) / max(1, n),
        "pass1_pct": 100.0 * sum(pass_at_k(k, c, 1) for c in ok) / max(1, n),
        "sample_acc": sum(ok) / max(1, n * k),
        "tokens_per_attempt": sum(x["tokens"] for att in per for x in att) / max(1, n * k),
        "think_per_attempt": sum(x["think"] for att in per for x in att) / max(1, n * k),
        "accepted_budgets": dict(collections.Counter(x["budget"] for x in acc)),
        "status": dict(collections.Counter(x["status"] for att in per for x in att).most_common()),
    }


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--init", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument(
        "--train", required=True, help="Stage-1 pairs (code.thinking.data) for the retained mix"
    )
    ap.add_argument("--transfer", default=None)
    ap.add_argument(
        "--heldout", default=None, help="greedy pass@1 each round (first --heldout_limit records)"
    )
    ap.add_argument("--heldout_limit", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--start_round", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--arm", default="plain", choices=["plain", "structural", "logical"])
    ap.add_argument(
        "--budgets", default="64", help="one value, or a list drawn per attempt (16,64,256)"
    )
    ap.add_argument("--min_think", type=int, default=0)
    ap.add_argument("--think", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--max_action", type=int, default=256, help="action tokens (the ladder's)")
    ap.add_argument("--max_steps", type=int, default=48)
    ap.add_argument("--max_rej", type=int, default=10)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_train", action="store_true")
    ap.add_argument("--ft_steps", type=int, default=600)
    ap.add_argument("--ft_lr", type=float, default=3e-4)
    ap.add_argument("--ft_recs", type=int, default=128)
    ap.add_argument("--think_weight", type=float, default=1.0)
    ap.add_argument("--retain", type=int, default=20000)
    ap.add_argument("--max_per_thm", type=int, default=4)
    ap.add_argument("--rl_weight", type=int, default=4)
    ap.add_argument("--validate_think", choices=["off", "drop", "empty"], default="off")
    ap.add_argument("--effort", action="store_true", help="effort conditioning (see the docstring)")
    ap.add_argument("--select", choices=["random", "shortest"], default="random")
    ap.add_argument(
        "--forced",
        choices=["keep", "drop"],
        default=None,
        help="fine-tune on attempts with a forced </think>? (default: drop; keep with --effort)",
    )
    ap.add_argument("--lean", default=None)
    ap.add_argument("--device", default=None)
    return ap


def forced_policy(forced: str | None, effort: bool) -> str:
    """--forced, defaulted: keep forced closes only when the effort token states the budget."""
    return forced or ("keep" if effort else "drop")


def run(a: argparse.Namespace) -> list[dict[str, Any]]:
    a.forced = forced_policy(a.forced, a.effort)
    out = Path(a.out)
    run_env(out)
    with bare():
        D.record.save_config(vars(a), str(out) + "/", arm=a.name, role="rl")
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    a.budget_list = [int(b) for b in a.budgets.split(",")]
    targets = read_jsonl(a.targets)
    for i, t in enumerate(targets):
        t.setdefault("name", f"target_{i}")
    transfer = read_jsonl(a.transfer) if a.transfer else []
    heldout = read_jsonl(a.heldout) if a.heldout else []
    if a.heldout_limit:
        heldout = heldout[: a.heldout_limit]
    train_data = None if a.no_train else load_pairs(a.train)
    rng = random.Random(a.seed * 7919)
    ckpt = a.init
    if a.effort:
        bad = [b for b in a.budget_list if b not in LEVELS]
        assert not bad, f"--effort: budgets {bad} are not effort levels {LEVELS}"
        if not a.resume:
            ckpt = str(expand_to_effort(a.init, out / "ckpts" / f"{a.name}_r0.pt", seed=a.seed))
        if train_data is not None and train_data["mode"] != "lean_thinke":
            train_data = relabel_records(train_data)
            print(f"effort levels of the retained records: {train_data['report']['effort_levels']}")
    found: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    keys: dict[str, dict[tuple[str, ...], int]] = collections.defaultdict(dict)
    if a.resume:
        r0 = a.start_round - 1
        for e in read_jsonl(out / f"found_{r0}.jsonl"):
            add_found(keys[e["name"]], found[e["name"]], e, a.select, a.forced)
        if not a.no_train:
            ckpt = str(out / "ckpts" / f"{a.name}_r{r0}.pt")
    history = []
    for r in range(a.start_round, a.start_round + a.rounds):
        t0 = time.time()
        D.record.phase("sample", round=r)
        model, tok, extra = load_ckpt(ckpt, dev)
        assert train_data is None or train_data["itos"] == list(tok.itos), "--train vocabulary"
        think = (
            has_think(tok) and extra.get("think_format", "think") != "none"
            if a.think == "auto"
            else a.think == "on"
        )
        seed = a.seed * 1000 + r
        per, st = sample_round(model, tok, targets, a, seed, think, keep_trace=True)
        stats: dict[str, Any] = {
            "round": r,
            "ckpt": ckpt,
            "k": a.k,
            "seed": seed,
            "budgets": a.budget_list,
        }
        stats["targets_round"] = round_summary(per, a.k)
        added: collections.Counter[str] = collections.Counter()
        n_forced_acc = 0
        for t, att in zip(targets, per, strict=True):
            for x in att:
                if not x["ok"]:
                    continue
                e = found_entry(x, tok, t, r, think)
                n_forced_acc += e["forced"] > 0
                how = add_found(keys[t["name"]], found[t["name"]], e, a.select, a.forced)
                added[how] += 1
                if how != "dup" and a.validate_think != "off" and think:
                    e["think_valid"] = validate_entry(e)
        new = added["new"]
        stats["new_attempts"] = new
        stats["accepted_attempts"] = {
            "new_proofs": new,
            "same_proof_kept_old": added["dup"],
            "same_proof_replaced": added["replaced"],
            "with_forced_close": n_forced_acc,
        }
        stats["forced_policy"] = a.forced
        stats["targets_cum"] = {
            "solved": sum(1 for t in targets if found.get(t["name"])),
            "n": len(targets),
            "distinct_attempts": sum(len(v) for v in found.values()),
        }
        stats["entries_per_target"] = entries_report(found, a.forced)
        if a.validate_think != "off":
            flags = [v for vs in found.values() for e in vs for v in e.get("think_valid") or []]
            stats["think_valid_frac"] = sum(flags) / max(1, len(flags))
        stats["sampler"] = {
            k2: (dict(v) if isinstance(v, collections.Counter) else v) for k2, v in st.items()
        }
        D.record.phase("eval", round=r)
        if transfer:
            pt, _ = sample_round(model, tok, transfer, a, seed + 500, think, keep_trace=False)
            stats["transfer_round"] = round_summary(pt, a.k)
        if heldout:
            ph, _ = sample_round(
                model, tok, heldout, a, seed + 700, think, keep_trace=False, greedy=True, k=1
            )
            stats["heldout_greedy"] = round_summary(ph, 1)
        with open(out / f"found_{r}.jsonl", "w") as f:
            for t in targets:
                for e in found.get(t["name"], []):
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
        D.record.phase("finetune", round=r)
        if not a.no_train:
            assert train_data is not None
            mix, ms = build_mix(
                targets,
                found,
                train_data["records"],
                tok,
                rng,
                a.max_per_thm,
                a.rl_weight,
                a.retain,
                a.validate_think,
                a.select,
                a.forced,
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
                    },
                )
            else:
                print("no accepted attempts: skipping training this round", flush=True)
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()
        stats["secs"] = time.time() - t0
        (out / f"round_{r}.json").write_text(json.dumps(stats, indent=1, default=str))
        print(
            f"=== round {r}: solved {stats['targets_round']['solved']}/{len(targets)} this round, "
            f"cumulative {stats['targets_cum']['solved']}, new {new}, {stats['secs']:.0f}s",
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
