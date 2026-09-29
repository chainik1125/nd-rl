"""Strict replay, matched-budget metrics, paired uncertainty, and export plots."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import random

from evaluate import attempt_key, planned_batches, read_rows, sha, stable_hash, token_accounting, validate_resume
from protocol import judge, protocol_identity, render_prompt, safe_verify
from proof_diagnostics import compress_proof
from nd_metrics import wilson


def paired_bootstrap(left: list[float], right: list[float], draws=10000, seed=202609092015) -> dict:
    if len(left) != len(right) or not left:
        raise ValueError("Paired bootstrap requires equal nonempty theorem vectors")
    differences = [r - l for l, r in zip(left, right, strict=True)]
    rng = random.Random(seed)
    n = len(differences)
    samples = sorted(sum(differences[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws))
    lo = samples[max(0, math.floor(0.025 * (draws - 1)))]
    hi = samples[min(draws - 1, math.ceil(0.975 * (draws - 1)))]
    return {"difference_right_minus_left": sum(differences) / n, "ci95_percentile": [lo, hi], "paired_theorems": n, "bootstrap_draws": draws, "seed": seed, "unit": "theorem_with_paired_outcomes"}


def cohort_metric(rows: list[dict]) -> dict:
    solved = sum(row["solved"] for row in rows)
    n = len(rows)
    return {
        "solved": solved, "n": n, "rate": solved / n if n else None,
        "wilson_ci95": list(wilson(solved, n)) if n else None,
        "successful_samples": sum(row["successful_attempts"] for row in rows),
        "sample_attempts": sum(row["attempts"] for row in rows),
        "mean_sample_success": sum(row["successful_attempts"] / row["attempts"] for row in rows) / n if n else None,
    }


def replay_compute_ledger(path: Path, plan: dict, planned: dict, tokenizer, prompt_ids: dict) -> tuple[dict, dict]:
    """Charge all completed compute calls, including regenerated partial batches."""
    from modeling import decode_completion, trim_completion
    batches = {batch["batch_id"]: batch for batch in plan["batches"]}
    starts, completed, outputs, totals = {}, set(), {}, Counter()
    for event in read_rows(path):
        call = event["call_id"]
        batch = event["batch"]
        if event["plan_sha256"] != stable_hash(plan) or batches.get(batch["batch_id"]) != batch:
            raise ValueError("Compute ledger call differs from the frozen batch plan")
        if event["phase"] == "started":
            if call in starts or not set(event["already_committed_attempts"]).issubset(batch["attempt_keys"]):
                raise ValueError("Compute ledger has duplicated or malformed start events")
            starts[call] = event
            continue
        if event["phase"] != "completed" or call not in starts or call in completed:
            raise ValueError("Compute ledger completion has no unique preceding start")
        if starts[call]["batch"] != batch or [value["attempt_key"] for value in event["native_outputs"]] != batch["attempt_keys"]:
            raise ValueError("Compute ledger changed batch membership or output order")
        completed.add(call)
        totals["completed_batches"] += 1
        totals["model_wall_seconds"] += event["wall_seconds"]
        batch_decode_width = max(len(value["response_ids"]) for value in event["native_outputs"])
        batch_prompt_width = max(value["prompt_tokens"] for value in event["native_outputs"])
        for native in event["native_outputs"]:
            key = native["attempt_key"]
            target = planned[key][0]
            ids = native["response_ids"]
            trimmed, reason = trim_completion(tokenizer, ids)
            if trimmed != ids or reason != native["finish_reason"] or len(ids) > plan["settings"]["max_new_tokens"]:
                raise ValueError("Compute ledger contains invalid native stopping/budget metadata")
            if reason == "length" and len(ids) != plan["settings"]["max_new_tokens"]:
                raise ValueError("Compute ledger length stop does not use its full cap")
            if native["prompt_tokens"] != len(prompt_ids[target["canonical_hash"]]):
                raise ValueError("Compute ledger prefill differs from the complete frozen prompt")
            if native["batch_generation_steps"] != batch_decode_width or native["post_stop_padding_positions"] != batch_decode_width - len(ids):
                raise ValueError("Compute ledger cannot reconcile logical output and batch padding")
            if native["prompt_padding_positions"] != batch_prompt_width - native["prompt_tokens"]:
                raise ValueError("Compute ledger cannot reconcile logical prefill and batch padding")
            generation = dict(native, text=decode_completion(tokenizer, ids))
            accounting = token_accounting(tokenizer, generation, judge(target["prompt"], generation["text"]))
            for field, value in accounting.items():
                if isinstance(value, int):
                    totals[field] += value
            totals["generated_attempt_positions"] += 1
            outputs[(call, key)] = native
    unfinished = set(starts) - completed
    return {
        **dict(totals), "unclosed_compute_calls": len(unfinished),
        "unclosed_calls_max_additional_native_tokens": sum(len(starts[call]["batch"]["attempt_keys"]) * plan["settings"]["max_new_tokens"] for call in unfinished),
        "is_lower_bound_due_to_unclosed_calls": bool(unfinished),
        "ledger_sha256": sha(path),
    }, outputs


def replay_run(directory: Path, allow_partial=False, tokenizer=None) -> tuple[dict, list[dict], list[dict]]:
    plan = json.loads((directory / "evaluation_plan.json").read_text())
    if plan["protocol"] != protocol_identity():
        raise ValueError("Current literal-extraction/verifier protocol differs from the frozen run")
    from modeling import decode_completion, load_tokenizer, trim_completion
    if tokenizer is None:
        model_path = Path(plan["model_path"])
        if sha(model_path / "config.json") != plan["model_config_sha256"]:
            raise ValueError("Native tokenizer model configuration changed")
        if sha(model_path / "download_completed.json") != plan["download_manifest_sha256"]:
            raise ValueError("Pinned model download identity changed")
        if not plan.get("tokenizer_files") or any(sha(Path(path)) != expected for path, expected in plan["tokenizer_files"].items()):
            raise ValueError("Pinned native tokenizer files changed")
        tokenizer = load_tokenizer(model_path)
        identity = {"class": type(tokenizer).__name__, "vocab_size": len(tokenizer), "eos_token_id": tokenizer.eos_token_id}
        if identity != plan["tokenizer_identity"]:
            raise ValueError("Native tokenizer identity changed")
    demonstrations = []
    if plan["demonstrations_per_query"]:
        bank_paths = [Path(path) for path in plan["input_files"] if Path(path).name == "shared_cap6.jsonl"]
        if len(bank_paths) != 1 or sha(bank_paths[0]) != plan["input_files"][str(bank_paths[0])]:
            raise ValueError("Frozen ICL demonstration bank identity changed")
        demonstrations = read_rows(bank_paths[0])
        if len(demonstrations) != plan["demonstrations_per_query"]:
            raise ValueError("Frozen ICL demonstration count changed")
    targets = read_rows(directory / "targets.jsonl")
    if stable_hash(targets) != plan["target_identity"] or len(targets) != plan["target_count"]:
        raise ValueError("Saved evaluation target manifest changed")
    planned = {}
    for target in targets:
        for mode in plan["settings"]["modes"]:
            for draw in range(1 if mode == "greedy" else 4):
                planned[attempt_key(target, mode, draw)] = (target, mode, draw)
    records = read_rows(directory / "attempts.jsonl")
    seen = validate_resume(records, planned, stable_hash(plan))
    complete = len(seen) == len(planned)
    if not complete and not allow_partial:
        raise ValueError(f"Run is incomplete ({len(seen)}/{len(planned)}); use --allow-partial for explicitly partial diagnostics")
    grouped, certificates = defaultdict(list), []
    totals, failure_reasons = Counter(), Counter()
    prompt_ids = {target["canonical_hash"]: tokenizer.encode(render_prompt(target["prompt"], demonstrations), add_special_tokens=False) for target in targets}
    batch_for_attempt = {}
    if plan.get("version", 1) >= 2:
        reconstructed, batches = planned_batches(targets, plan["settings"]["modes"], plan["settings"]["batch_size"], plan["settings"]["seed"])
        if reconstructed != planned or batches != plan["batches"]:
            raise ValueError("Frozen batch membership or RNG seeds cannot be reconstructed")
        batch_for_attempt = {key: (batch, index) for batch in batches for index, key in enumerate(batch["attempt_keys"])}
    for record in records:
        target, mode, draw = planned[record["attempt_key"]]
        generation = record["generation"]
        if batch_for_attempt:
            batch, index = batch_for_attempt[record["attempt_key"]]
            if record["batch_id"] != batch["batch_id"] or record["batch_index"] != index or record["sampling_batch_seed"] != batch["seed"] or generation["seed"] != batch["seed"]:
                raise ValueError("Saved attempt RNG metadata differs from its frozen batch")
            expected_sampling = {"temperature": 0.0 if mode == "greedy" else 1.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0}
            if generation["sampling"] != expected_sampling:
                raise ValueError("Saved attempt has a different sampling distribution")
        if generation["prompt_ids"] != prompt_ids[target["canonical_hash"]]:
            raise ValueError("Saved native prompt IDs differ from the complete frozen prompt")
        if generation["prompt_tokens"] != len(generation["prompt_ids"]) or generation["completion_tokens"] != len(generation["response_ids"]):
            raise ValueError("Saved native token counts differ from emitted IDs")
        if generation["text"] != decode_completion(tokenizer, generation["response_ids"]):
            raise ValueError("Saved completion text differs from literal native-token reconstruction")
        trimmed, finish_reason = trim_completion(tokenizer, generation["response_ids"])
        if trimmed != generation["response_ids"] or finish_reason != generation["finish_reason"]:
            raise ValueError("Saved response includes post-stop padding or has an incorrect stop reason")
        if len(generation["response_ids"]) > plan["settings"]["max_new_tokens"]:
            raise ValueError("An attempt exceeds its frozen native-token budget")
        if finish_reason == "length" and len(generation["response_ids"]) != plan["settings"]["max_new_tokens"]:
            raise ValueError("A length-stopped attempt does not consume its declared cap")
        verdict = judge(target["prompt"], generation["text"])
        if verdict != record["judge"]:
            raise ValueError("Saved judgment differs from independent literal strict replay")
        if token_accounting(tokenizer, generation, verdict) != record["token_accounting"]:
            raise ValueError("Saved token accounting differs from native reconstruction")
        compressed = None
        if verdict["accepted"]:
            compressed = compress_proof(target["prompt"], verdict["proof"])
            ok, reason, count = safe_verify(target["prompt"], compressed["compressed_proof"])
            if not ok or count != compressed["compressed_length"] or count > verdict["n_lines"]:
                raise ValueError("Analysis compression failed strict replay: " + reason)
            certificates.append({
                "arm": plan["arm"], "attempt_key": record["attempt_key"], "canonical_hash": target["canonical_hash"],
                "mode": mode, "draw": draw, "raw_proof": verdict["proof"], "raw_lines": verdict["n_lines"],
                "compressed_proof": compressed["compressed_proof"], "compressed_lines": count,
                "analysis_only_never_training": True,
            })
        else:
            failure_reasons[verdict["reason"]] += 1
        grouped[(target["cohort"], target["canonical_hash"], mode)].append((record, verdict, compressed))
        for field, value in record["token_accounting"].items():
            if isinstance(value, int):
                totals[field] += value
        totals["attempts"] += 1
        totals["accepted_attempts"] += verdict["accepted"]
        totals["length_stopped_attempts"] += generation["finish_reason"] == "length"
        totals["eos_stopped_attempts"] += generation["finish_reason"] == "eos"
        totals["final_marker_stopped_attempts"] += generation["finish_reason"] == "final_marker"
    per_target, pending = [], []
    for target in targets:
        for mode in plan["settings"]["modes"]:
            values = grouped[(target["cohort"], target["canonical_hash"], mode)]
            expected = 1 if mode == "greedy" else 4
            if len(values) != expected:
                pending.append({"cohort": target["cohort"], "canonical_hash": target["canonical_hash"], "mode": mode, "attempts": len(values), "expected": expected})
                continue
            accepted = [verdict for _, verdict, _ in values if verdict["accepted"]]
            compressed = [result for _, verdict, result in values if verdict["accepted"]]
            per_target.append({
                "arm": plan["arm"], "cohort": target["cohort"], "canonical_hash": target["canonical_hash"],
                "family": target.get("family", "textbook"), "reference_lines": target["reference_lines"],
                "mode": mode, "metric": "greedy_pass1" if mode == "greedy" else "observed_pass4",
                "attempts": expected, "successful_attempts": len(accepted), "solved": bool(accepted),
                "shortest_raw_lines": min((value["n_lines"] for value in accepted), default=None),
                "shortest_compressed_lines": min((value["compressed_length"] for value in compressed), default=None),
                "prefill_tokens": sum(record["token_accounting"]["prefill_native_tokens"] for record, _, _ in values),
                "generated_tokens": sum(record["token_accounting"]["generated_native_tokens"] for record, _, _ in values),
            })
    grouped_metrics = {}
    for cohort in plan["cohorts"]:
        grouped_metrics[cohort] = {}
        for mode in plan["settings"]["modes"]:
            rows = [row for row in per_target if row["cohort"] == cohort and row["mode"] == mode]
            intended_n = sum(target["cohort"] == cohort for target in targets)
            lengths = sorted({target["reference_lines"] for target in targets if target["cohort"] == cohort})
            grouped_metrics[cohort][mode] = {
                "overall": dict(cohort_metric(rows), intended_n=intended_n),
                "by_reference_length": {str(length): cohort_metric([row for row in rows if row["reference_lines"] == length]) for length in lengths},
                "by_family": {family: cohort_metric([row for row in rows if row["family"] == family]) for family in sorted({row["family"] for row in rows})},
            }
    compute_totals = None
    if batch_for_attempt:
        compute_totals, native_outputs = replay_compute_ledger(directory / "compute_batches.jsonl", plan, planned, tokenizer, prompt_ids)
        for record in records:
            native = native_outputs.get((record["compute_call_id"], record["attempt_key"]))
            if native is None or any(native[field] != record["generation"].get(field) for field in ("response_ids", "prompt_tokens", "finish_reason", "post_stop_padding_positions", "prompt_padding_positions", "batch_generation_steps")):
                raise ValueError("Saved attempt lacks matching native evidence in the compute ledger")
        extra = compute_totals.get("generated_native_tokens", 0) - totals["generated_native_tokens"]
        if extra < 0:
            raise ValueError("Compute ledger omits generated tokens from committed attempts")
        compute_totals["extra_native_tokens_beyond_unique_scored_attempts"] = extra
    result = {
        "arm": plan["arm"], "run_directory": str(directory), "status": "complete" if complete else "partial_diagnostic",
        "evaluation_plan_sha256": sha(directory / "evaluation_plan.json"), "attempts_sha256": sha(directory / "attempts.jsonl"),
        "planned_attempts": len(planned), "observed_attempts": len(records), "pending_theorem_mode_groups": len(pending),
        "groups": grouped_metrics, "token_totals": dict(totals), "failure_reasons": dict(failure_reasons),
        "actual_compute_totals_including_regeneration": compute_totals,
        "model_id": plan["model_id"], "model_revision": plan["model_revision"], "settings": plan["settings"],
        "protocol": plan["protocol"], "demonstrations_per_query": plan["demonstrations_per_query"],
        "modeling_sha256": plan.get("modeling_sha256"),
        "physical_batch_size": plan["settings"].get("batch_size", 1),
        "selection_warning": "Completed theorem groups only in partial diagnostics; do not compare incomplete arms as headline results." if not complete else None,
    }
    return result, per_target, certificates


def compare_runs(per_target: list[dict], draws: int, partial: bool) -> list[dict]:
    groups = defaultdict(dict)
    for row in per_target:
        groups[(row["arm"], row["cohort"], row["mode"])][row["canonical_hash"]] = row
    comparisons = []
    for cohort, mode in sorted({(cohort, mode) for _, cohort, mode in groups}):
        arms = sorted(arm for arm, candidate_cohort, candidate_mode in groups if candidate_cohort == cohort and candidate_mode == mode)
        for left, right in itertools.combinations(arms, 2):
            a, b = groups[(left, cohort, mode)], groups[(right, cohort, mode)]
            if set(a) != set(b) and not partial:
                raise ValueError("Paired arms do not have identical theorem membership")
            keys = sorted(set(a) & set(b))
            for length in [None, *sorted({a[key]["reference_lines"] for key in keys})]:
                selected = [key for key in keys if length is None or a[key]["reference_lines"] == length]
                if not selected:
                    continue
                seed = int(stable_hash([cohort, mode, left, right, length])[:8], 16)
                estimate = paired_bootstrap([float(a[key]["solved"]) for key in selected], [float(b[key]["solved"]) for key in selected], draws, seed)
                comparisons.append(dict(estimate, left_arm=left, right_arm=right, cohort=cohort, mode=mode, reference_lines=length, partial_diagnostic=partial))
    return comparisons


def plots(results: list[dict], out: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = []
    cohorts = sorted({cohort for result in results for cohort in result["groups"]})
    for mode in ("greedy", "sampled"):
        active = [cohort for cohort in cohorts if any(mode in result["groups"].get(cohort, {}) for result in results)]
        if not active:
            continue
        fig, axes = plt.subplots(len(active), 1, figsize=(10, 4 * len(active)), squeeze=False, constrained_layout=True)
        for axis, cohort in zip(axes[:, 0], active, strict=True):
            for index, result in enumerate(results):
                metrics = result["groups"].get(cohort, {}).get(mode)
                if metrics is None:
                    continue
                bins = metrics["by_reference_length"]
                x = [int(length) for length, value in bins.items() if value["n"]]
                values = [bins[str(length)] for length in x]
                y = [value["rate"] for value in values]
                lower = [value["rate"] - value["wilson_ci95"][0] for value in values]
                upper = [value["wilson_ci95"][1] - value["rate"] for value in values]
                offset = (index - (len(results) - 1) / 2) * .035
                shifted = [length + offset for length in x]
                axis.errorbar(shifted, y, yerr=[lower, upper], marker="o", linewidth=1.3, capsize=2, label=result["arm"])
                for px, py, value in zip(shifted, y, values, strict=True):
                    axis.annotate(f"{value['solved']}/{value['n']}", (px, py), xytext=(0, 5 + 9 * index), textcoords="offset points", ha="center", fontsize=7)
            axis.set_title(f"{cohort}: {'greedy pass@1' if mode == 'greedy' else 'observed pass@4'}")
            axis.set_xlabel("Reference witness lines (upper bound; not minimum proof length)")
            axis.set_ylabel("Strictly valid proofs / theorems")
            axis.set_ylim(-.04, 1.17)
            axis.set_yticks([0, .25, .5, .75, 1])
            axis.grid(alpha=.2)
            axis.legend(loc="best", fontsize=8)
        fig.suptitle("Matched OLMo proof baseline — equal 2,048-token decode cap; bars: 95% Wilson intervals", fontsize=11)
        for extension in ("png", "pdf", "svg"):
            name = f"reasoning_length_{mode}.{extension}"
            fig.savefig(out / name, dpi=180, bbox_inches="tight")
            names.append(name)
        plt.close(fig)
    return names


def run(args) -> dict:
    if args.out.exists():
        raise ValueError("Analysis output must be fresh; preserve previous reports")
    results, per_target, certificates = [], [], []
    arm_cohorts = set()
    for directory in args.runs:
        result, targets, proofs = replay_run(directory.resolve(), args.allow_partial)
        for cohort, modes in result["groups"].items():
            for mode in modes:
                key = (result["arm"], cohort, mode)
                if key in arm_cohorts:
                    raise ValueError("Duplicate arm/cohort/mode run would double count targets")
                arm_cohorts.add(key)
        results.append(result)
        per_target.extend(targets)
        certificates.extend(proofs)
    identities = {stable_hash({
        **{field: result[field] for field in ("model_id", "model_revision", "protocol", "modeling_sha256")},
        "settings": {field: value for field, value in result["settings"].items() if field != "batch_size"},
    }) for result in results}
    # A greedy-only run and a sampled-only run may be analyzed separately;
    # combined primary reports should use a single common settings schedule.
    if len(identities) != 1:
        raise ValueError("Arms differ in model, prompt protocol, or generation settings")
    partial = any(result["status"] != "complete" for result in results)
    comparisons = compare_runs(per_target, args.bootstrap_draws, partial)
    args.out.mkdir(parents=True)
    for name, rows in (("per_theorem.jsonl", per_target), ("certificates_analysis_only.jsonl", certificates)):
        with (args.out / name).open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
    figures = [] if args.no_plots else plots(results, args.out)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "partial_diagnostic" if partial else "complete",
        "runs": results, "paired_comparisons": comparisons, "figures": figures,
        "analyzer_sha256": sha(Path(__file__)),
        "compression_source_sha256": sha(Path(__import__("proof_diagnostics").__file__)),
        "raw_success_is_primary": True, "compressed_proofs_analysis_only": True,
        "length_caveat": "x+P is a reference-witness-length stratum, not a certified minimum-complexity stratum. Any valid shorter proof is accepted.",
        "uncertainty_caveat": "Wilson intervals describe theorem success within an arm. Paired bootstrap resamples the same theorem identifiers jointly across arms. One training seed does not estimate training-seed variability; bootstrap is descriptive and uncorrected for multiple comparisons.",
        "cost_caveat": "Native generated tokens include failed outputs, scratch, final proof, and EOS. Prefill is separate. Scratch/final region counts are decoded substring retokenization estimates; they are not asserted to partition native generated tokens exactly.",
        "batching_caveat": "Physical generation batch sizes may differ by arm according to memory/throughput capacity. Each arm freezes its batch membership and records actual batch RNG seeds; per-theorem sampling distributions and decode budgets match, but different physical batches do not promise identical random streams. Paired bootstrap pairs theorem identities, not RNG draws. Actual compute includes regenerated partial batches and separately counts padding positions.",
    }
    (args.out / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = ["# Matched OLMo reasoning baseline", "", "Status: " + report["status"], "", "| Arm | Cohort | Decode | Solved / total | Rate |", "|---|---|---|---:|---:|"]
    for result in results:
        for cohort, modes in result["groups"].items():
            for mode, metrics in modes.items():
                value = metrics["overall"]
                rate = f"{value['rate']:.1%}" if value["rate"] is not None else "pending"
                lines.append(f"| {result['arm']} | {cohort} | {mode} | {value['solved']}/{value['n']} | {rate} |")
    lines += ["", report["length_caveat"], "", report["uncertainty_caveat"], "", report["cost_caveat"], "", report["batching_caveat"], "", "Every accepted raw proof was strictly replayed. Dependency/reiteration compression was separately strictly replayed and never turns an invalid attempt into a success. Per-theorem shortest raw and compressed lengths are in `per_theorem.jsonl`; compressed certificates are explicitly analysis-only.", ""]
    (args.out / "report.md").write_text("\n".join(lines))
    return {"status": report["status"], "out": str(args.out), "runs": len(results), "theorem_mode_rows": len(per_target), "strict_accepted_certificates": len(certificates), "figures": figures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_draws < 100:
        parser.error("Use at least 100 bootstrap resamples")
    print(json.dumps(run(args), sort_keys=True))


if __name__ == "__main__":
    main()
