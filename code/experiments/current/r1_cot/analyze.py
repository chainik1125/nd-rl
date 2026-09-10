"""Strict, analysis-only replay of released complete evaluation samples.

CLI requires a release manifest BEFORE opening labels or predictions. Example:
  python analyze.py --root TAKE_HOME --predictions RUN/attempts.jsonl \
      --labels LABELS.jsonl --release release.json --out REPORT

Release fields: status='complete', role, labels_sha256, evaluation_k,
prediction_sha256={absolute_path: sha256}, plus rl_run_completed=true for
textbook_dev or final_selection_frozen=true for textbook_final. Optional
model_names maps absolute prediction paths to plot labels. Never call this on
textbook data before the root releases the appropriate completed-run endpoint.
Textbook releases also require model_modes={absolute_path: 'direct' or 'cot'}
to reconstruct final bodies from the saved raw response token IDs.
For repair models declare model_tokenizer_versions={absolute_path:
'relative-unary-r1-repair-v1'}. Omitted entries retain the legacy 43-token parser;
raw token IDs never select a tokenizer automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def check_release(release, labels_path, prediction_paths):
    """Validate permission metadata first; only then hash the released inputs."""
    if release.get("status") != "complete":
        raise ValueError("Only complete evaluations may produce a report")
    role = release.get("role")
    if role not in ("synthetic", "textbook_dev", "textbook_final"):
        raise ValueError("Explicit evaluation role required")
    if role == "textbook_dev" and release.get("rl_run_completed") is not True:
        raise ValueError("Textbook development requires completed RL runs")
    if role == "textbook_final" and release.get("final_selection_frozen") is not True:
        raise ValueError("Final-only labels require frozen final selection")
    if type(release.get("evaluation_k")) is not int or release["evaluation_k"] < 1:
        raise ValueError("Positive evaluation_k required")
    expected = release.get("prediction_sha256", {})
    for path in prediction_paths:
        if str(Path(path).resolve()) not in expected:
            raise ValueError(f"Prediction file not explicitly released: {path}")
        if role.startswith("textbook_") and release.get("model_modes", {}).get(
            str(Path(path).resolve())
        ) not in ("direct", "cot"):
            raise ValueError("Textbook replay requires released direct/cot model_modes")
    if not release.get("labels_sha256"):
        raise ValueError("Labels hash required")
    if sha(labels_path) != release["labels_sha256"]:
        raise ValueError("Labels hash differs from release")
    for path in prediction_paths:
        if sha(path) != expected[str(Path(path).resolve())]:
            raise ValueError(f"Predictions differ from release: {path}")


def load_labels(records):
    result = {}
    for row in records:
        name = str(row["name"])
        if name in result:
            raise ValueError(f"Duplicate label target: {name}")
        length = row.get("reference_lines", row.get("n_lines", row.get("proof_length")))
        if type(length) is not int or length < 1:
            raise ValueError(f"Positive reference witness length required: {name}")
        result[name] = {
            "prompt": " ".join(row["prompt"].split()),
            "reference_lines": length,
            "role": row.get("role"),
        }
    if not result:
        raise ValueError("Empty label population")
    return result


def count_tokens(row):
    generated = row.get("generated_tokens")
    if generated is None and "response_ids" in row:
        generated = len(row["response_ids"])
    scratch, final = row.get("scratch_tokens"), row.get("final_tokens")
    if any(type(x) is not int or x < 0 for x in (generated, scratch, final)):
        raise ValueError("Exact nonnegative generated/scratch/final token counts required")
    if "response_ids" in row and generated != len(row["response_ids"]):
        raise ValueError("Generated token count disagrees with response_ids")
    if scratch + final > generated:
        raise ValueError("Scratch plus final tokens exceed generation")
    return generated, scratch, final


def select_tokenizer(runner, version=None):
    version = version or runner.ScratchTokenizer.VERSION
    classes = [runner.ProofTokenizer, runner.ScratchTokenizer]
    if hasattr(runner, "ScratchRepairTokenizer"):
        classes.append(runner.ScratchRepairTokenizer)
    for tokenizer_class in classes:
        if version == tokenizer_class.VERSION:
            return tokenizer_class()
    raise ValueError(f"Unknown release-declared tokenizer version: {version}")


def replay(records, labels, verify, compress, evaluation_k, reconstruct=None):
    """Replay every saved literal final proof, including all reported failures."""
    grouped = defaultdict(list)
    for row in records:
        name = str(row["name"])
        if name not in labels:
            raise ValueError(f"Unknown prediction target: {name}")
        label = labels[name]
        prompt = " ".join(row["prompt"].split())
        if prompt != label["prompt"]:
            raise ValueError(f"Original prompt mismatch: {name}")
        if "reference_lines" in row and row["reference_lines"] != label["reference_lines"]:
            raise ValueError("Prediction reference length disagrees with labels")
        generated, scratch, final = count_tokens(row)
        body = row["body"]
        if not isinstance(body, str):
            raise ValueError("Literal decoded final body must be saved")
        if reconstruct is not None:
            rebuilt = reconstruct(prompt, row["response_ids"])
            for field in ("body", "reward", "scratch_tokens", "final_tokens"):
                if rebuilt[field] != row[field]:
                    raise ValueError(
                        f"Raw response reconstruction disagrees with saved {field}: {name}"
                    )
        # Runner writes a body only after its response-format check succeeds.
        # New-format records additionally expose format_valid for auditing.
        format_valid = row.get("format_valid", bool(body))
        checked, reason, raw_lines = verify(prompt, body)
        ok = bool(format_valid and checked)
        if not format_valid:
            reason = "response-format-invalid"
        compressed_lines = None
        if ok:
            diagnostic = compress(prompt, body)
            if not diagnostic.get("raw_valid"):
                raise ValueError("Compressor rejected strict-valid raw proof")
            accepted, diagnostic_reason, compressed_lines = verify(
                prompt, diagnostic["compressed_proof"]
            )
            if not accepted:
                raise ValueError(f"Compressed proof fails STRICT replay: {diagnostic_reason}")
            if compressed_lines != diagnostic["compressed_length"] or compressed_lines > raw_lines:
                raise ValueError("Invalid compressed-length accounting")
        if "reward" in row and bool(row["reward"]) != ok:
            raise ValueError(f"Saved reward disagrees with strict replay: {name}/{row['attempt']}")
        attempt = row["attempt"]
        if type(attempt) is not int or attempt < 0:
            raise ValueError("Attempt must be a zero-based nonnegative integer")
        grouped[name].append(
            {
                "attempt": attempt,
                "solved": ok,
                "reason": reason,
                "raw_lines": raw_lines if ok else None,
                "compressed_lines": compressed_lines,
                "generated_tokens": generated,
                "scratch_tokens": scratch,
                "final_tokens": final,
                "delimiter_or_other_tokens": generated - scratch - final,
            }
        )
    if set(grouped) != set(labels):
        raise ValueError("Incomplete evaluation: every label needs all saved attempts")
    for name, samples in grouped.items():
        samples.sort(key=lambda x: x["attempt"])
        if [x["attempt"] for x in samples] != list(range(evaluation_k)):
            raise ValueError(f"Incomplete/duplicate/out-of-order attempt coverage for {name}")
    return dict(grouped)


def wilson(successes, n):
    if not n:
        return [None, None]
    z = 1.959963984540054
    p = successes / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0, center - radius), min(1, center + radius)]


def outcome(samples, k=None, budget=None):
    """Prefix policy: never skip an expensive attempt to take a later one."""
    used, scratch, final, attempts, success = 0, 0, 0, 0, False
    raw, compressed = [], []
    for sample in samples[:k]:
        cost = sample["generated_tokens"]
        if budget is not None and used + cost > budget:
            break
        used += cost
        scratch += sample["scratch_tokens"]
        final += sample["final_tokens"]
        attempts += 1
        if sample["solved"]:
            success = True
            raw.append(sample["raw_lines"])
            compressed.append(sample["compressed_lines"])
    return {
        "solved": success,
        "attempts": attempts,
        "generated_tokens": used,
        "scratch_tokens": scratch,
        "final_tokens": final,
        "shortest_raw_lines": min(raw, default=None),
        "shortest_compressed_lines": min(compressed, default=None),
    }


def curve_point(grouped, k=None, budget=None):
    outcomes = {name: outcome(samples, k=k, budget=budget) for name, samples in grouped.items()}
    solved = sum(r["solved"] for r in outcomes.values())
    return {
        "k": k,
        "per_target_generated_token_budget": budget,
        "solved": solved,
        "total": len(outcomes),
        "rate": solved / len(outcomes),
        "wilson95": wilson(solved, len(outcomes)),
        "summed_consumed_generated_tokens": sum(r["generated_tokens"] for r in outcomes.values()),
        "summed_consumed_scratch_tokens": sum(r["scratch_tokens"] for r in outcomes.values()),
        "summed_consumed_final_tokens": sum(r["final_tokens"] for r in outcomes.values()),
        "summed_completed_attempts": sum(r["attempts"] for r in outcomes.values()),
        "minimum_completed_attempts_per_target": min(r["attempts"] for r in outcomes.values()),
    }


def summarize(grouped, labels, ks, budgets, initial_cap):
    per_target = {
        name: dict(reference_lines=labels[name]["reference_lines"], **outcome(samples))
        for name, samples in grouped.items()
    }
    bins = []
    for length in sorted({x["reference_lines"] for x in labels.values()}):
        targets = [r for r in per_target.values() if r["reference_lines"] == length]
        solved = sum(r["solved"] for r in targets)
        bins.append(
            {
                "reference_lines": length,
                "solved": solved,
                "total": len(targets),
                "rate": solved / len(targets),
                "wilson95": wilson(solved, len(targets)),
            }
        )
    histograms = {}
    for field in ("shortest_raw_lines", "shortest_compressed_lines"):
        histogram = Counter(r[field] for r in per_target.values() if r["solved"])
        histograms[field] = {
            "histogram": dict(sorted(histogram.items())),
            "longest_shortest_found_upper_bound": max(histogram, default=None),
            "five_target_length_frontier": max(
                (n for n, c in histogram.items() if c >= 5), default=None
            ),
        }
    over = [r for r in per_target.values() if r["reference_lines"] > initial_cap]
    all_samples = list(itertools.chain.from_iterable(grouped.values()))
    return {
        "all_attempts": curve_point(grouped),
        "reference_length_bins": bins,
        "beyond_initial_cap": {
            "initial_cap": initial_cap,
            "solved": sum(r["solved"] for r in over),
            "total": len(over),
        },
        "found_length_upper_bounds": histograms,
        "pass_at_k_prefix": [curve_point(grouped, k=k) for k in ks],
        "token_budget_prefix": [curve_point(grouped, budget=b) for b in budgets],
        "failed_reason_counts": dict(Counter(r["reason"] for r in all_samples if not r["solved"])),
        "per_target": per_target,
    }


def paired_bootstrap(a, b, seed=20260909, resamples=5000):
    """Paired target resampling for model B minus model A, not seed variance."""
    if set(a) != set(b) or not a:
        raise ValueError("Paired comparison needs the same nonempty target population")
    differences = [int(b[k]) - int(a[k]) for k in sorted(a)]
    rng = random.Random(seed)
    n = len(differences)
    draws = sorted(
        sum(differences[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)
    )
    return {
        "difference_b_minus_a": sum(differences) / n,
        "percentile95": [draws[int(0.025 * (resamples - 1))], draws[int(0.975 * (resamples - 1))]],
        "a_only": differences.count(-1),
        "b_only": differences.count(1),
        "n_targets": n,
        "seed": seed,
        "resamples": resamples,
    }


def compare_models(models, ks, budgets, seed, resamples):
    result = []
    for a, b in itertools.combinations(models, 2):
        for kind, values in [("k", ks), ("budget", budgets)]:
            for val in values:
                av = {
                    name: outcome(samples, **{kind: val})["solved"]
                    for name, samples in models[a].items()
                }
                bv = {
                    name: outcome(samples, **{kind: val})["solved"]
                    for name, samples in models[b].items()
                }
                result.append(
                    {"a": a, "b": b, kind: val, **paired_bootstrap(av, bv, seed, resamples)}
                )
    return result


def plot_figures(summaries, out, initial_cap, role):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for name, summary in summaries.items():
        bins = summary["reference_length_bins"]
        axes[0].plot(
            [r["reference_lines"] for r in bins], [r["rate"] for r in bins], marker="o", label=name
        )
        curve = summary["token_budget_prefix"]
        axes[1].plot(
            [r["per_target_generated_token_budget"] for r in curve],
            [r["rate"] for r in curve],
            marker="o",
            drawstyle="steps-post",
            label=name,
        )
    axes[0].axvline(
        initial_cap, linestyle="--", color="0.45", label=f"Initial label cap {initial_cap}"
    )
    axes[0].set(
        xlabel="Reference witness length (lines; upper bound)",
        ylabel="Fraction of targets solved",
        title=f"{role}: all saved attempts",
    )
    axes[1].set(
        xlabel="Generated-token allowance per target",
        ylabel="Fraction of targets solved",
        title="Complete-attempt prefixes; actual costs in JSON",
    )
    for ax in axes:
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.savefig(out / "heldout_length_and_budget.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    names = list(summaries)
    maxima = [
        summaries[n]["found_length_upper_bounds"]["shortest_compressed_lines"][
            "longest_shortest_found_upper_bound"
        ]
        for n in names
    ]
    bars = axes[0].bar(names, [x or 0 for x in maxima])
    for bar, value in zip(bars, maxima, strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(value) if value else "No success",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[0].set(
        ylabel="Lines (upper bound, not minimum complexity)",
        title="Longest shortest-found compressed witness",
    )
    axes[0].tick_params(axis="x", labelrotation=25)
    for name, summary in summaries.items():
        curve = summary["pass_at_k_prefix"]
        axes[1].plot([r["k"] for r in curve], [r["rate"] for r in curve], marker="o", label=name)
    axes[1].set(
        xlabel="First k saved attempts",
        ylabel="Fraction solved",
        ylim=(-0.03, 1.03),
        title="Empirical verifier-selected coverage",
    )
    axes[1].set_xticks([r["k"] for r in next(iter(summaries.values()))["pass_at_k_prefix"]])
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    fig.savefig(out / "found_length_and_passk.png", dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", nargs="+", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument(
        "--root",
        type=Path,
        required=True,
        help="take-home checkout containing strict verifier/compressor",
    )
    ap.add_argument("--initial-cap", type=int, default=6)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument(
        "--budgets", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    )
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--bootstrap-resamples", type=int, default=5000)
    args = ap.parse_args()
    release = json.loads(args.release.read_text())
    check_release(release, args.labels, args.predictions)
    labels = load_labels(rows(args.labels))
    if any(x["role"] and x["role"] != release["role"] for x in labels.values()):
        raise ValueError("Label role differs from release")
    if args.initial_cap < 1 or args.bootstrap_resamples < 100 or min(args.ks + args.budgets) < 1:
        raise ValueError("Invalid analysis limits")
    k = release["evaluation_k"]
    ks = sorted(set(x for x in args.ks if x <= k) | {k})
    budgets = sorted(set(args.budgets))
    root = args.root.resolve()
    sys.path[:0] = [str(root / "transfer_sprint_20260907/code"), str(root)]
    import proof_diagnostics
    import strict_verify

    models, summaries = {}, {}
    for path in args.predictions:
        name = release.get("model_names", {}).get(str(path.resolve()), path.parent.name)
        if name in models:
            raise ValueError(f"Duplicate model label: {name}")
        reconstruct = None
        mode = release.get("model_modes", {}).get(str(path.resolve()))
        if mode is not None:
            if mode not in ("direct", "cot"):
                raise ValueError("Unknown response mode")
            spec = importlib.util.spec_from_file_location(
                "r1_cot_run_analysis", Path(__file__).with_name("run.py")
            )
            runner = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = runner
            spec.loader.exec_module(runner)
            declared_version = release.get("model_tokenizer_versions", {}).get(str(path.resolve()))
            tok = select_tokenizer(runner, declared_version)

            def reconstruct(prompt, ids, mode=mode, runner=runner, tok=tok):
                if any(type(token) is not int or not 0 <= token < len(tok) for token in ids):
                    raise ValueError("Raw response IDs exceed the release-declared tokenizer")
                rebuilt = runner.judge_response(prompt, ids, tok, mode)
                final_id = tok.stoi["<FINAL>"]
                rebuilt["final_tokens"] = (
                    len(ids)
                    if mode == "direct"
                    else len(ids) - ids.index(final_id) - 1
                    if final_id in ids
                    else 0
                )
                return rebuilt

        models[name] = replay(
            rows(path),
            labels,
            strict_verify.safe_verify,
            proof_diagnostics.compress_proof,
            k,
            reconstruct,
        )
        summaries[name] = summarize(models[name], labels, ks, budgets, args.initial_cap)
    report = {
        "analysis_only": True,
        "status": "complete",
        "role": release["role"],
        "written_utc": datetime.now(UTC).isoformat(),
        "release_sha256": sha(args.release),
        "labels_sha256": sha(args.labels),
        "prediction_sha256": {str(p.resolve()): sha(p) for p in args.predictions},
        "source_sha256": {
            "analysis": sha(__file__),
            "strict_verify": sha(strict_verify.__file__),
            "compression": sha(proof_diagnostics.__file__),
            "official_verify": sha(root / "nd_verify/verify.py"),
            "official_safe_verify": sha(root / "nd_metrics.py"),
        },
        "models": summaries,
        "paired": compare_models(models, ks, budgets, args.seed, args.bootstrap_resamples),
        "limitations": [
            "All reference and found lengths are upper bounds, not proven minimum lengths.",
            "Compression is analysis-only and reverified with the strict checker.",
            "Pass@k is first-k empirical coverage, not the unbiased combinatorial estimator.",
            "Token curves admit complete attempts in recorded order; "
            "no skipped expensive attempts.",
            "Recorded token costs exclude prompt prefill, padding and batching overhead; "
            "equal generated tokens are not equal FLOPs.",
            "Curves are censored by the saved number of attempts and their response caps.",
            "Paired intervals resample targets, "
            "not independent training seeds or textbook sources.",
            "A sprint-final-only partition does not erase historical benchmark exposure.",
        ],
    }
    if release.get("model_modes"):
        report["source_sha256"]["response_parser"] = sha(runner.__file__)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    plot_figures(summaries, args.out, args.initial_cap, release["role"])
    (args.out / "completed.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "summary_sha256": sha(args.out / "summary.json"),
                "figure_sha256": {p.name: sha(p) for p in args.out.glob("*.png")},
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "out": str(args.out),
                "role": release["role"],
                "models": {n: s["all_attempts"]["solved"] for n, s in summaries.items()},
                "n_targets": len(labels),
            }
        )
    )


if __name__ == "__main__":
    main()
