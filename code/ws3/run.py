"""ws3 experiment driver: `uv run python ws3/run.py --config <yaml> [--stages a,b,c]`.

Stages, in order (plan sections 3, 4 and 6):

    baseline   the frozen checkpoint: T1 gate (next-token accuracy and loss on heldout), greedy
               pass rate by length on a heldout stride, textbook 72 at k samples, long-pool
               (rl_targets) pass@k. This is the frozen-checkpoint control for every later claim.
    train      TRL GRPOTrainer (or RLOOTrainer) with the binary reward; metrics.jsonl carries the
               plan section 4 set, one JSON line per step; completions.jsonl every judged rollout.
    evaluate   the trained model through the same evaluations as `baseline` (same k, temperature,
               seeds), plus a control-vs-RL comparison.
    report     graphs (pass rate by length before/after; reward, KL, entropy curves), Markdown
               tables, and a check that metrics.jsonl carries every required metric.

Paths in the YAML are relative to the YAML's own directory, so an experiment folder
`experiments/ws3/<name>/config.yaml` with `run.out_dir: run` writes to `experiments/ws3/<name>/run/`.
Model directories under it (`hf_base/`, `final/`, `trainer/`) are gitignored; JSON summaries,
metrics.jsonl and the proof dumps are committed for the reviewer's recount.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import ndpath
import torch
import yaml

# isort: split
from data import build_prompt_set, load_pool, load_textbook72, stride
from eval import evaluate_pool, evaluate_textbook, next_token_accuracy, textbook_table
from eval.evaluator import utc_now
from eval.reward import ProofReward
from hf_wrapper import build_tokenizer, load_pretrained_wrapper, save_base
from metrics_callback import REQUIRED_METRICS, MetricsJsonlCallback, PeriodicEvalCallback

STAGES = ("baseline", "train", "evaluate", "report")

# Conservative T4 start (plan section 3): group 8, lr 1e-5, KL beta 0.01, clip 0.2, temp 1.0.
TRAINER_DEFAULTS = dict(
    num_generations=8,
    per_device_train_batch_size=64,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    beta=0.01,
    epsilon=0.2,
    temperature=1.0,
    max_completion_length=400,
    generation_kwargs={"use_cache": True},  # NDCache through HF generate (hf_wrapper.py)
    num_iterations=1,
    max_steps=100,
    warmup_steps=0,
    lr_scheduler_type="constant",
    max_grad_norm=1.0,
    gradient_checkpointing=False,  # the wrapper does not support it
    report_to="none",
    save_strategy="no",
    logging_steps=1,
    remove_unused_columns=False,
    disable_dropout=True,
    log_completions=False,
)
GRPO_ONLY = dict(loss_type="grpo", scale_rewards="group", mask_truncated_completions=False)
RLOO_ONLY = dict(normalize_advantages=False)


# ---------------------------------------------------------------------------- helpers


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stages", default=",".join(STAGES), help="comma-separated subset, in order")
    ap.add_argument("--max-steps", type=int, default=None, help="override train.trainer.max_steps")
    return ap.parse_args(argv)


def load_config(path: str, max_steps=None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = os.path.dirname(os.path.abspath(path))
    run = cfg.setdefault("run", {})
    run["config_path"] = os.path.abspath(path)
    run["out_dir"] = os.path.normpath(os.path.join(base, run.get("out_dir", "run")))
    run.setdefault("seed", 0)
    run.setdefault("device", "cuda" if torch.cuda.is_available() else "cpu")
    run.setdefault("algo", "grpo")
    cfg.setdefault("eval", {})
    tr = cfg.setdefault("train", {})
    tr.setdefault("trainer", {})
    if max_steps is not None:
        tr["trainer"]["max_steps"] = max_steps
    return cfg


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ndpath.CODE, text=True
        ).strip()
    except Exception:
        return None


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def environment_record(cfg, extra=None):
    import transformers
    import trl

    ck = ndpath.FORK_CKPT
    return {
        "utc": utc_now(),
        "config": cfg,
        "git_commit": git_commit(),
        "checkpoint": ck,
        "checkpoint_sha256": _sha256(ck),
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "trl": trl.__version__,
        },
        "platform": platform.platform(),
        "cpu_threads": torch.get_num_threads(),
        "argv": sys.argv,
        **(extra or {}),
    }


def load_final_model(out_dir, device):
    from transformers import AutoModelForCausalLM

    import hf_wrapper  # noqa: F401  (registers the architecture with the Auto classes)

    final = os.path.join(out_dir, "final")
    model = AutoModelForCausalLM.from_pretrained(final).to(device).eval()
    tok = build_tokenizer(getattr(model.config, "tok_mode", "abs"))
    return model, tok


# ---------------------------------------------------------------------------- evaluations


def common_evals(model, tok, cfg, d, seed):
    """Parts (b), (c), (d): greedy heldout by length, textbook 72 at k, long pool pass@k.

    Identical for the frozen checkpoint (`baseline`) and the trained model (`evaluate`), so the
    control is at equal attempts (same k, temperature, seed). Returns the per-part results.
    """
    e = cfg["eval"]
    dumps = os.path.join(d, "dumps")
    os.makedirs(dumps, exist_ok=True)
    out = {}
    g = e.get("greedy", {})
    if g:
        recs = stride(load_pool("heldout"), int(g.get("n_heldout", 1000)))
        out["greedy_heldout"] = evaluate_pool(
            model,
            tok,
            recs,
            k=1,
            temperature=0.0,
            max_new=int(g.get("max_new", 400)),
            batch_size=int(g.get("batch_size", 100)),
            seed=seed,
            dump=os.path.join(dumps, "greedy_heldout.jsonl"),
            label="greedy_heldout",
        )
        write_json(os.path.join(d, "greedy_heldout.json"), out["greedy_heldout"])
        print("greedy heldout by length:", _rates(out["greedy_heldout"]), flush=True)
    t = e.get("textbook", {})
    if t:
        out["textbook"] = evaluate_textbook(
            model,
            tok,
            load_textbook72(),
            k=int(t.get("k", 64)),
            temperature=float(t.get("temperature", 1.0)),
            max_new=int(t.get("max_new", 400)),
            batch_size=int(t.get("batch_size", 256)),
            seed=seed,
            dump=os.path.join(dumps, "textbook72.jsonl"),
        )
        write_json(os.path.join(d, "textbook.json"), out["textbook"])
        print("textbook by slice:", out["textbook"]["by_slice"], flush=True)
    lp = e.get("long", {})
    if lp:
        recs = stride(load_pool("rl_targets"), int(lp.get("n_prompts", 200)))
        out["long_pool"] = evaluate_pool(
            model,
            tok,
            recs,
            k=int(lp.get("k", 32)),
            temperature=float(lp.get("temperature", 1.0)),
            max_new=int(lp.get("max_new", 400)),
            batch_size=int(lp.get("batch_size", 256)),
            seed=seed,
            dump=os.path.join(dumps, "long_pool.jsonl"),
            label="long_pool_rl_targets",
        )
        write_json(os.path.join(d, "long_pool.json"), out["long_pool"])
        print("long pool pass@k by length:", _rates(out["long_pool"]), flush=True)
    return out


def _rates(res):
    return {L: f"{v['solved']}/{v['n']}" for L, v in res["by_length"].items()}


def stage_baseline(cfg):
    run, e = cfg["run"], cfg["eval"]
    d = os.path.join(run["out_dir"], "baseline")
    os.makedirs(d, exist_ok=True)
    write_json(os.path.join(d, "environment.json"), environment_record(cfg))
    t0 = time.time()
    model, tok = load_pretrained_wrapper(device=run["device"])
    summary = {"utc_start": utc_now(), "model": "frozen checkpoint " + ndpath.FORK_CKPT}
    nta = e.get("nta", {})
    if nta:
        recs = stride(load_pool("heldout"), int(nta.get("n_heldout", 5000)))
        res = next_token_accuracy(model, tok, recs, batch_size=int(nta.get("batch_size", 250)))
        gate = float(nta.get("gate", 0.95))
        res["gate_threshold"] = gate
        res["gate_pass"] = res["next_token_accuracy"] >= gate
        write_json(os.path.join(d, "nta.json"), res)
        summary["nta"] = res
        print(f"T1 next-token accuracy {res['next_token_accuracy']:.4f} loss {res['loss']:.4f} "
              f"(gate {gate}: {'PASS' if res['gate_pass'] else 'FAIL'})", flush=True)
    summary.update(common_evals(model, tok, cfg, d, run["seed"]))
    summary["secs"] = time.time() - t0
    summary["utc_end"] = utc_now()
    write_json(os.path.join(d, "summary.json"), summary)
    return summary


# ---------------------------------------------------------------------------- training


def build_trainer_config(cfg, out_dir):
    run, tr = cfg["run"], cfg["train"]
    from trl import GRPOConfig, RLOOConfig

    kw = dict(TRAINER_DEFAULTS)
    kw.update(GRPO_ONLY if run["algo"] == "grpo" else RLOO_ONLY)
    kw.update(tr.get("trainer", {}))
    kw["output_dir"] = os.path.join(out_dir, "trainer")
    kw.setdefault("seed", run["seed"])
    kw.setdefault("use_cpu", run["device"] == "cpu")
    kw.setdefault("bf16", run["device"] == "cuda")
    cls = GRPOConfig if run["algo"] == "grpo" else RLOOConfig
    return cls(**kw)


def stage_train(cfg):
    from datasets import Dataset
    from trl import GRPOTrainer, RLOOTrainer

    run, tr = cfg["run"], cfg["train"]
    out_dir = run["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(run["seed"])
    t0 = time.time()

    model, tok = load_pretrained_wrapper(device=run["device"])
    base_dir = save_base(model, tok, os.path.join(out_dir, "hf_base"))

    p = tr["prompt"]
    prompts = build_prompt_set(p["length_weights"], int(p["n_prompts"]), seed=run["seed"])
    with open(os.path.join(out_dir, "prompts.jsonl"), "w") as f:
        for r in prompts:
            f.write(json.dumps(r) + "\n")
    dataset = Dataset.from_list(
        [{"prompt": r["prompt"], "length": r["length"], "source": r["source"]} for r in prompts]
    )
    reward_fn = ProofReward(
        tok_mode=tok.tok_mode, dump_path=os.path.join(out_dir, "completions.jsonl")
    )
    tcfg = build_trainer_config(cfg, out_dir)
    completions_per_step = int(tcfg.per_device_train_batch_size) * int(
        tcfg.gradient_accumulation_steps
    )
    metrics_path = os.path.join(out_dir, "metrics.jsonl")
    if os.path.exists(metrics_path):
        os.remove(metrics_path)
    callbacks = [MetricsJsonlCallback(metrics_path, completions_per_step, reward_fn)]
    ev = tr.get("periodic_eval", {})
    if ev.get("every"):
        held = stride(load_pool("heldout"), int(ev.get("n_heldout", 100)))
        longp = stride(load_pool("rl_targets"), int(ev.get("n_long", 50)))
        callbacks.append(
            PeriodicEvalCallback(
                metrics_path,
                tok,
                held + longp,
                every=int(ev["every"]),
                max_new=int(ev.get("max_new", 400)),
                batch_size=int(ev.get("batch_size", 128)),
                dump=os.path.join(out_dir, "eval_periodic.jsonl"),
            )
        )
    manifest = environment_record(
        cfg,
        {
            "algo": run["algo"],
            "hf_base": base_dir,
            "n_prompts": len(prompts),
            "prompt_lengths": sorted({r["length"] for r in prompts}),
            "completions_per_step": completions_per_step,
            "trainer_config": {k: v for k, v in tcfg.to_dict().items() if not k.startswith("_")},
        },
    )
    write_json(os.path.join(out_dir, "run_manifest.json"), manifest)
    cls = GRPOTrainer if run["algo"] == "grpo" else RLOOTrainer
    trainer = cls(
        model=model,
        reward_funcs=reward_fn,
        args=tcfg,
        train_dataset=dataset,
        processing_class=tok,
        callbacks=callbacks,
    )
    trainer.train()
    final_dir = os.path.join(out_dir, "final")
    trainer.save_model(final_dir)
    tok.save_pretrained(final_dir)
    rollout_logs = [l for l in trainer.state.log_history if "reward" in l]
    summary = {
        "utc_end": utc_now(),
        "secs": time.time() - t0,
        "steps": trainer.state.global_step,
        "completions_per_step": completions_per_step,
        "first_log": rollout_logs[0] if rollout_logs else None,
        "last_log": rollout_logs[-1] if rollout_logs else None,
        "mean_reward_first5": _mean([l["reward"] for l in rollout_logs[:5]]),
        "mean_reward_last5": _mean([l["reward"] for l in rollout_logs[-5:]]),
        "max_kl": max((l.get("kl", 0.0) for l in rollout_logs), default=None),
    }
    write_json(os.path.join(out_dir, "train_summary.json"), summary)
    print("train summary:", json.dumps({k: v for k, v in summary.items() if "log" not in k}))
    return summary


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


# ---------------------------------------------------------------------------- evaluate


def stage_evaluate(cfg):
    run = cfg["run"]
    out_dir = run["out_dir"]
    d = os.path.join(out_dir, "evaluate")
    os.makedirs(d, exist_ok=True)
    t0 = time.time()
    model, tok = load_final_model(out_dir, run["device"])
    summary = {"utc_start": utc_now(), "model": os.path.join(out_dir, "final")}
    summary.update(common_evals(model, tok, cfg, d, run["seed"]))
    summary["secs"] = time.time() - t0
    summary["utc_end"] = utc_now()
    write_json(os.path.join(d, "summary.json"), summary)
    bdir = os.path.join(out_dir, "baseline")
    if os.path.exists(os.path.join(bdir, "summary.json")):
        write_json(os.path.join(d, "comparison.json"), comparison(bdir, d))
    return summary


def comparison(bdir, edir):
    """Control vs RL, per evaluation, per length / slice. Same k, temperature and seed."""
    out = {"utc": utc_now()}
    for part in ("greedy_heldout", "long_pool", "textbook"):
        bp, ep = os.path.join(bdir, part + ".json"), os.path.join(edir, part + ".json")
        if not (os.path.exists(bp) and os.path.exists(ep)):
            continue
        b, e = read_json(bp), read_json(ep)
        rows = {}
        for L in sorted(set(b["by_length"]) | set(e["by_length"]), key=int):
            rows[L] = {"control": b["by_length"].get(L), "rl": e["by_length"].get(L)}
        entry = {
            "k": b["k"],
            "temperature": b["temperature"],
            "by_length": rows,
            "overall": {"control": b["overall"], "rl": e["overall"]},
            "max_length_solved": {"control": b["max_length_solved"], "rl": e["max_length_solved"]},
        }
        if "by_slice" in b:
            entry["by_slice"] = {"control": b["by_slice"], "rl": e["by_slice"]}
            entry["ref_len_ge7"] = {"control": b["ref_len_ge7"], "rl": e["ref_len_ge7"]}
        out[part] = entry
    return out


# ---------------------------------------------------------------------------- report


BLUE, ORANGE = "#2a78d6", "#eb6834"  # control, RL (fixed categorical order)


def _style(ax):
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


def read_metrics(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def metric_coverage(rows):
    keys = set()
    for r in rows:
        keys |= set(r)
    cov = {}
    for name, key in REQUIRED_METRICS.items():
        if key.endswith("*"):
            cov[name] = any(k.startswith(key[:-1]) for k in keys)
        else:
            cov[name] = key in keys
    return cov


def plot_curves(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    roll = [r for r in rows if "reward" in r]
    panels = [
        ("reward", "mean reward"),
        ("reward_std", "reward std within groups"),
        ("kl", "KL to reference"),
        ("entropy", "policy entropy"),
        ("completions/mean_length", "completion length (mean tokens)"),
        ("grad_norm", "gradient norm"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
    for ax, (key, title) in zip(axes.flat, panels, strict=True):
        xs = [r["step"] for r in roll if key in r]
        ys = [r[key] for r in roll if key in r]
        ax.plot(xs, ys, color=ORANGE, linewidth=2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("optimizer step")
        _style(ax)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_periodic(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ev = [r for r in rows if "eval/pass_rate" in r]
    if not ev:
        return False
    lengths = sorted(
        {int(k[len("eval/pass_rate_len"):]) for r in ev for k in r if k.startswith("eval/pass_rate_len")}
    )
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    cmap = plt.get_cmap("Blues")
    for i, L in enumerate(lengths):
        key = f"eval/pass_rate_len{L}"
        xs = [r["step"] for r in ev if key in r]
        ys = [r[key] for r in ev if key in r]
        ax.plot(xs, ys, marker="o", markersize=4, linewidth=2,
                color=cmap(0.35 + 0.6 * i / max(len(lengths) - 1, 1)), label=f"length {L}")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("greedy pass rate")
    ax.set_ylim(0, 1.02)
    ax.set_title("periodic greedy eval by proof length (small fixed set)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _style(ax)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def plot_pass_by_length(comp, part, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = comp[part]["by_length"]
    Ls = [L for L in rows if int(L) >= 0]
    x = list(range(len(Ls)))
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    w = 0.38
    for off, who, color in ((-w / 2, "control", BLUE), (w / 2, "rl", ORANGE)):
        ys, lo, hi = [], [], []
        for L in Ls:
            v = rows[L][who] or {"rate": 0, "ci": [0, 0]}
            ys.append(v["rate"])
            lo.append(v["rate"] - v["ci"][0])
            hi.append(v["ci"][1] - v["rate"])
        ax.bar([xi + off for xi in x], ys, width=w - 0.04, color=color,
               label="frozen checkpoint" if who == "control" else "after RL")
        ax.errorbar([xi + off for xi in x], ys, yerr=[lo, hi], fmt="none", ecolor="#444",
                    elinewidth=1, capsize=2)
    ax.set_xticks(x, [str(L) for L in Ls])
    ax.set_xlabel("reference proof length")
    ax.set_ylabel(f"pass@{comp[part]['k']} (Wilson 95%)")
    ax.set_ylim(0, 1.05)
    if "6" in Ls and int(Ls[-1]) > 6:  # the pre-training length cutoff (train.jsonl is lengths 2-6)
        ax.axvline(x=Ls.index("6") + 0.5, color="#888", linestyle="--", linewidth=1)
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def length_table(comp, part):
    rows = comp[part]["by_length"]
    lines = [
        f"| length | control solved/n | control rate [95% CI] | RL solved/n | RL rate [95% CI] |",
        "| --- | --- | --- | --- | --- |",
    ]
    for L, r in rows.items():
        cells = [str(L)]
        for who in ("control", "rl"):
            v = r[who]
            if v is None:
                cells += ["-", "-"]
            else:
                cells += [f"{v['solved']}/{v['n']}", f"{v['rate']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]"]
        lines.append("| " + " | ".join(cells) + " |")
    o = comp[part]["overall"]
    lines.append(
        f"| all | {o['control']['solved']}/{o['control']['n']} | {o['control']['rate']:.3f} | "
        f"{o['rl']['solved']}/{o['rl']['n']} | {o['rl']['rate']:.3f} |"
    )
    return "\n".join(lines)


def stage_report(cfg):
    out_dir = cfg["run"]["out_dir"]
    d = os.path.join(out_dir, "report")
    os.makedirs(d, exist_ok=True)
    md = [f"# Report for {os.path.basename(os.path.dirname(out_dir))}", "", f"Generated {utc_now()}.", ""]
    metrics_path = os.path.join(out_dir, "metrics.jsonl")
    if os.path.exists(metrics_path):
        rows = read_metrics(metrics_path)
        cov = metric_coverage(rows)
        md += ["## Gate B: metrics.jsonl coverage (plan section 4)", ""]
        md += [f"- {'ok' if v else 'MISSING'}: {k} (`{REQUIRED_METRICS[k]}`)" for k, v in cov.items()]
        md += ["", f"{len([r for r in rows if 'reward' in r])} rollout log lines, "
               f"{len([r for r in rows if 'eval/pass_rate' in r])} periodic eval lines.", ""]
        plot_curves(rows, os.path.join(d, "curves.png"))
        md += ["![training curves](curves.png)", ""]
        if plot_periodic(rows, os.path.join(d, "periodic_eval.png")):
            md += ["![periodic eval](periodic_eval.png)", ""]
        write_json(os.path.join(d, "metric_coverage.json"), cov)
    comp_path = os.path.join(out_dir, "evaluate", "comparison.json")
    bdir = os.path.join(out_dir, "baseline")
    if os.path.exists(comp_path):
        comp = read_json(comp_path)
        for part, title in (
            ("greedy_heldout", "greedy pass rate by length, heldout stride"),
            ("long_pool", "pass@k by true length, rl_targets stride"),
        ):
            if part in comp:
                plot_pass_by_length(comp, part, os.path.join(d, f"{part}.png"), title)
                md += [f"## {title}", "", f"![{part}]({part}.png)", "", length_table(comp, part), ""]
        if "textbook" in comp:
            b, e = read_json(os.path.join(bdir, "textbook.json")), read_json(
                os.path.join(out_dir, "evaluate", "textbook.json")
            )
            md += [f"## textbook 72 at k={b['k']}, temperature {b['temperature']} (control vs RL)", "",
                   textbook_table({"frozen checkpoint": b, "after RL": e}), ""]
    elif os.path.exists(os.path.join(bdir, "summary.json")):
        b = read_json(os.path.join(bdir, "textbook.json")) if os.path.exists(
            os.path.join(bdir, "textbook.json")) else None
        if b:
            md += ["## textbook 72 (frozen checkpoint only)", "", textbook_table({"frozen checkpoint": b}), ""]
    with open(os.path.join(d, "report.md"), "w") as f:
        f.write("\n".join(md))
    print("\n".join(md))
    return d


# ---------------------------------------------------------------------------- main


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config, args.max_steps)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    for s in stages:
        if s not in STAGES:
            raise SystemExit(f"unknown stage {s}; choose from {STAGES}")
    os.makedirs(cfg["run"]["out_dir"], exist_ok=True)
    status = os.path.join(cfg["run"]["out_dir"], "stages.jsonl")
    for s in stages:
        t0 = time.time()
        print(f"=== stage {s} start {utc_now()}", flush=True)
        globals()[f"stage_{s}"](cfg)
        with open(status, "a") as f:
            f.write(json.dumps({"stage": s, "utc": utc_now(), "secs": time.time() - t0}) + "\n")
        print(f"=== stage {s} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
