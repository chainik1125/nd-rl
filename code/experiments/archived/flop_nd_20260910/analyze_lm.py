"""Auditable paired analysis of frozen LM evaluations and training ledgers.

No model loading, checkpoint selection, training, or reference-conditioned
generation occurs here. Bootstrap intervals describe held-out context sampling
conditional on the supplied training seeds; they do not estimate seed variance.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
import common


def jsonl(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def read_evaluation(directory, audit, audit_sha256):
    directory = Path(directory)
    result = json.loads((directory / "result.json").read_text())
    if result.get("status") != "completed":
        raise ValueError(f"Incomplete evaluation: {directory}")
    attempts_path = directory / "attempts.jsonl"
    if common.sha(attempts_path) != result["attempts_sha256"]:
        raise ValueError(f"Attempt hash mismatch: {directory}")
    config = json.loads((directory / "config.json").read_text())
    declared_inputs = config["input_sha256"]
    if result.get("input_sha256") != declared_inputs:
        raise ValueError("Evaluation config/result input identities disagree")
    audit_input = str(Path(config["args"]["audit"]).resolve())
    if declared_inputs.get(audit_input) != audit_sha256:
        raise ValueError("Supplied analysis audit identity differs from evaluation audit")
    for path, digest in declared_inputs.items():
        if common.sha(path) != digest:
            raise ValueError(f"An evaluation input changed: {path}")
    n_attempts = int(config["args"]["attempts"])
    if (result.get("sampled_attempts") != n_attempts or
            result.get("temperature") != config["args"]["temperature"] or
            result.get("per_theorem_token_cap") != n_attempts * config["args"]["max_new_tokens"]):
        raise ValueError("Evaluation config/result inference profiles disagree")
    grouped = defaultdict(list)
    for row in jsonl(attempts_path):
        sid = row["source_id"]
        if sid not in audit or row["prompt"] != audit[sid]["prompt"]:
            raise ValueError("Evaluation contains an unknown/mismatched theorem")
        if row.get("reset_index") != 0 or row["reward"] not in (0, 1):
            raise ValueError("Evaluation must be unaided with binary raw reward")
        if row["reward"]:
            comp = row.get("compression", {})
            if not row.get("independent_ok") or not comp.get("raw_valid"):
                raise ValueError("Success lacks independent verification/compression")
            if not 0 < comp["compressed_length"] <= comp["dependency_pruned_length"] <= row["n_lines"]:
                raise ValueError("Invalid successful proof lengths")
        grouped[sid].append(row)
    if set(grouped) != set(audit):
        raise ValueError("Every evaluation must cover the complete declared population")
    rows = {}
    for sid, group in grouped.items():
        group.sort(key=lambda row: row["attempt"])
        if [r["attempt"] for r in group] != list(range(n_attempts)):
            raise ValueError("Missing or duplicate per-theorem attempts")
        solved = [r for r in group if r["reward"]]
        written = [r["n_lines"] for r in solved]
        pruned = [r["compression"]["dependency_pruned_length"] for r in solved]
        compressed = [r["compression"]["compressed_length"] for r in solved]
        rows[sid] = {
            "first": int(group[0]["reward"]), "any": int(bool(solved)),
            "attempt_successes": len(solved), "attempts": len(group),
            "tokens": sum(r["generated_tokens"] for r in group),
            "any_pruned_over6": int(any(n > 6 for n in pruned)),
            "shortest_found_pruned_over6": int(bool(pruned) and min(pruned) > 6),
            "any_compressed_over6": int(any(n > 6 for n in compressed)),
            "shortest_found_compressed_over6": int(bool(compressed) and min(compressed) > 6),
            "shortest_written": min(written, default=None),
            "shortest_pruned": min(pruned, default=None),
            "shortest_compressed": min(compressed, default=None),
            "max_pruned": max(pruned, default=None),
            "successes": [{"attempt": r["attempt"], "written": r["n_lines"],
                           "pruned": r["compression"]["dependency_pruned_length"],
                           "compressed": r["compression"]["compressed_length"]} for r in solved],
        }
    return {"directory": str(directory.resolve()), "rows": rows,
            "attempts_sha256": common.sha(attempts_path),
            "result_sha256": common.sha(directory / "result.json"),
            "checkpoint_sha256": config["input_sha256"][config["args"]["checkpoint"]],
            "profile": {k: config["args"][k] for k in
                        ("temperature", "attempts", "max_new_tokens", "batch_size", "seed", "precision")}}


def strata(audit):
    result = {"all": list(audit),
              "reference_over6": [sid for sid, row in audit.items() if row["n_lines"] > 6]}
    for key in ("length_bin", "family"):
        for value in sorted({row[key] for row in audit.values()}):
            result[f"{key}:{value}"] = [sid for sid, row in audit.items() if row[key] == value]
    return {k: sorted(v) for k, v in result.items() if v}


def paired_context_bootstrap(values, contexts, replicates=4000, seed=202609101936):
    """Resample contexts with replacement, preserving within-context pairings.

    values is a theorem vector (which may already average training seeds).
    The estimand is theorem-weighted, not an unweighted average of contexts.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) != len(contexts) or not len(values):
        raise ValueError("Expected one finite value per nonempty theorem/context")
    if not np.isfinite(values).all() or replicates < 1:
        raise ValueError("Nonfinite values or invalid replicate count")
    unique, inverse = np.unique(contexts, return_inverse=True)
    sums = np.bincount(inverse, weights=values, minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique))
    rng = np.random.default_rng(seed)
    estimates = []
    for offset in range(0, replicates, 256):
        indices = rng.integers(0, len(unique), size=(min(256, replicates-offset), len(unique)))
        estimates.extend((sums[indices].sum(axis=1) / counts[indices].sum(axis=1)).tolist())
    interval = np.quantile(estimates, [.025, .975]).tolist()
    return {"estimate": float(values.mean()), "ci95_context_bootstrap": interval,
            "theorems": len(values), "contexts": len(unique), "replicates": replicates,
            "seed": seed, "uncertainty_scope": "Context sampling conditional on these trained policies; not training-seed uncertainty."}


def summary_rows(rows, ids):
    selected = [rows[sid] for sid in ids]
    sums = {key: sum(r[key] for r in selected) for key in
            ("first", "any", "attempt_successes", "attempts", "tokens",
             "any_pruned_over6", "shortest_found_pruned_over6",
             "any_compressed_over6", "shortest_found_compressed_over6")}
    return {"theorems": len(ids), **sums,
            "first_rate": sums["first"] / len(ids), "any_rate": sums["any"] / len(ids),
            "shortest_pruned_histogram": dict(sorted(Counter(r["shortest_pruned"] for r in selected if r["any"]).items())),
            "shortest_compressed_histogram": dict(sorted(Counter(r["shortest_compressed"] for r in selected if r["any"]).items())),
            "maximum_pruned": max((r["max_pruned"] for r in selected if r["any"]), default=0)}


def evaluation_analysis(config):
    audit_path = Path(config["audit"])
    audit_list = list(jsonl(audit_path))
    audit = {r["source_id"]: r for r in audit_list}
    if len(audit) != len(audit_list):
        raise ValueError("Duplicate audit identities")
    runs = {name: read_evaluation(path, audit, common.sha(audit_path)) for name, path in config["evaluations"].items()}
    profiles = {common.digest(run["profile"]) for run in runs.values()}
    if len(profiles) != 1:
        raise ValueError("Paired conditions must have identical inference profiles")
    populations = strata(audit)
    summaries = {name: {s: summary_rows(run["rows"], ids) for s, ids in populations.items()}
                 for name, run in runs.items()}
    contrasts = {}
    for contrast_name, groups in config.get("contrasts", {}).items():
        # Each mapping is a within-seed contrast. The mean pools the same
        # theorem's contrasts across seeds before resampling contexts.
        by_seed = {}
        for label, terms in groups.items():
            if not terms or abs(sum(terms.values())) > 1e-12:
                raise ValueError("Contrasts must contain coefficients summing to zero")
            if any(name not in runs for name in terms):
                raise ValueError("Contrast references an absent run")
            by_seed[label] = terms
        contrast_result = {}
        for stratum, ids in populations.items():
            metrics = {}
            for metric in ("first", "any", "any_pruned_over6", "shortest_found_pruned_over6"):
                vectors = {label: np.array([sum(weight * runs[name]["rows"][sid][metric]
                                               for name, weight in terms.items()) for sid in ids])
                           for label, terms in by_seed.items()}
                if len(vectors) > 1:
                    vectors["mean_across_seeds"] = np.mean(list(vectors.values()), axis=0)
                metrics[metric] = {label: paired_context_bootstrap(
                    values, [audit[sid]["context_id"] for sid in ids],
                    config.get("bootstrap_replicates", 4000), config.get("bootstrap_seed", 202609101936))
                    for label, values in vectors.items()}
            contrast_result[stratum] = metrics
        contrasts[contrast_name] = contrast_result
    return {"status": "completed", "population_sha256": common.sha(audit_path),
            "analysis_code_sha256": common.sha(__file__), "profile": next(iter(runs.values()))["profile"],
            "run_provenance": {name: {k:v for k,v in run.items() if k != "rows"} for name,run in runs.items()},
            "summaries": summaries, "contrasts": contrasts,
            "per_theorem": {name: run["rows"] for name,run in runs.items()},
            "notes": ["Reference length is an upper bound on proof length, not a difficulty label.",
                      "Pruning retains all mandated premise lines and removes unused dependencies; compression also removes verified reiterations.",
                      "A generated proof above six lines is not evidence that every proof requires more than six lines.",
                      "Shortest-found refers only to the fixed sampled attempts, not a minimum over all possible proofs.",
                      "Any-of-k is an observed fixed-k success rate, not an unbiased pass@k extrapolation.",
                      "Intervals are paired by theorem and clustered by canonical context; two training seeds remain descriptive."]}


def training_analysis(directory, window=500):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    result_path = directory / "result.json"
    if not result_path.exists():
        raise ValueError("Training analysis requires a completed endpoint")
    result = json.loads(result_path.read_text())
    config = manifest["config"]
    updates = list(jsonl(directory / "updates.jsonl"))
    if [u["step"] for u in updates] != list(range(1,len(updates)+1)):
        raise ValueError("Training update log has a gap or duplicate")
    if not updates or result["step"] != len(updates):
        raise ValueError("Training endpoint and update ledger disagree")
    for endpoint_key, update_key in (("generated_tokens","generated_tokens_total"),
            ("optimizer_steps","optimizer_steps_total"), ("cumulative_flop_proxy","cumulative_flop_proxy"),
            ("schedule_digest","schedule_digest"),("retention_schedule_digest","retention_schedule_digest")):
        if result[endpoint_key] != updates[-1][update_key]:
            raise ValueError(f"Training endpoint differs from last committed update: {endpoint_key}")
    for key, value in (("generated_tokens",sum(u["advantages"]["action_tokens"] for u in updates)),
            ("optimizer_steps",sum(u["optimizer_steps"] for u in updates)),
            ("cumulative_flop_proxy",sum(u["step_flop_proxy"] for u in updates)),
            ("retention_examples",sum(u["retention_examples"] for u in updates)),
            ("retention_response_tokens",sum(u["retention_response_tokens"] for u in updates))):
        if result[key] != value:raise ValueError(f"Training total differs from update sum: {key}")
    for kind in ("source","target"):
        for count in ("attempts","successes"):
            if result[count][kind] != sum(u["outcome"].get(kind,{}).get(count,0) for u in updates):
                raise ValueError(f"Training outcome total differs: {kind}/{count}")
    buckets = defaultdict(lambda: {"updates": 0, "generated_tokens": 0, "flop_proxy": 0,
        "source_attempts": 0, "source_successes": 0, "target_attempts": 0, "target_successes": 0,
        "active_advantage_tokens": 0, "action_tokens": 0, "zero_advantage_updates": 0,
        "groups": Counter(), "group_successes": Counter(), "mixed_groups": Counter(),
        "curriculum_cuts": Counter(), "source_ids": set()})
    for u in updates:
        b = buckets[(u["step"]-1)//window]
        b["updates"] += 1
        b["generated_tokens"] += u["advantages"]["action_tokens"]
        b["flop_proxy"] += u["step_flop_proxy"]
        b["active_advantage_tokens"] += u["advantages"]["active_advantage_tokens"]
        b["action_tokens"] += u["advantages"]["action_tokens"]
        b["zero_advantage_updates"] += u["advantages"]["active_advantage_tokens"] == 0
        for kind, outcome in u["outcome"].items():
            b[kind+"_attempts"] += outcome["attempts"]
            b[kind+"_successes"] += outcome["successes"]
    cut_history = defaultdict(list)
    committed_groups = defaultdict(dict)
    curriculum_events=[]
    first_zero={}
    for row in jsonl(directory / "groups.jsonl"):
        if not 1 <= row["step"] <= len(updates):
            raise ValueError("Uncommitted group beyond final update")
        if row["group"] in committed_groups[row["step"]]:
            raise ValueError("Duplicate training group")
        committed_groups[row["step"]][row["group"]] = row
        if "curriculum_window" in row:
            event={"step":row["step"],"group":row["group"],"source_id":row["source_id"],
                   **row["curriculum_window"]}
            curriculum_events.append(event)
            if event["new"]==0 and event["old"]!=0:
                first_zero.setdefault(row["source_id"],row["step"])
        b = buckets[(row["step"]-1)//window]
        kind = row["training_kind"] + ("_assisted" if row["reset_index"] else "_empty")
        b["groups"][kind] += 1
        b["group_successes"][kind] += row["successes"]
        b["mixed_groups"][kind] += row["mixed"]
        if row["training_kind"] == "source":
            b["curriculum_cuts"][row["reset_index"]] += 1
            b["source_ids"].add(row["source_id"])
            if row["reset_index"]:
                cut_history[row["source_id"]].append((row["step"],row["reset_index"]))
    for update in updates:
        groups=committed_groups[update["step"]]
        if set(groups) != set(range(config["prompts_per_step"])):
            raise ValueError("Missing or extra training groups")
        if any(g["group_size"] != config["group_size"] for g in groups.values()):
            raise ValueError("Training group size differs from protocol")
        if sum(g["generated_tokens"] for g in groups.values()) != update["advantages"]["action_tokens"]:
            raise ValueError("Training group/update token totals disagree")
        for kind,outcome in update["outcome"].items():
            subset=[g for g in groups.values() if g["training_kind"]==kind]
            calculated={"attempts":sum(g["group_size"] for g in subset),
                        "successes":sum(g["successes"] for g in subset),
                        "mixed_groups":sum(g["mixed"] for g in subset)}
            if calculated != outcome:raise ValueError("Training group/update outcomes disagree")
    discovery_path = directory / "discoveries.jsonl"
    discoveries = list(jsonl(discovery_path)) if discovery_path.exists() else []
    discoveries_by_target = defaultdict(list)
    for row in discoveries:
        if row["step"] > len(updates):
            raise ValueError("Discovery beyond endpoint")
        discoveries_by_target[row["source_id"]].append(row)
    curves = []
    for index, values in sorted(buckets.items()):
        values = dict(values)
        values["unique_sources"] = len(values.pop("source_ids"))
        values.update(start_step=index*window+1,end_step=min((index+1)*window,len(updates)))
        curves.append(values)
    epochs = [e for u in updates for e in u["epochs"]]
    health = {key: {"min": min(e[key] for e in epochs), "max": max(e[key] for e in epochs),
                    "median": float(np.median([e[key] for e in epochs]))}
              for key in ("grad_norm","approx_kl_per_action","clip_fraction","ratio_min","ratio_max","value_loss")}
    curriculum = json.loads((directory/"curriculum.json").read_text())
    return {"status": "completed", "analysis_code_sha256": common.sha(__file__),
            "directory": str(directory.resolve()), "config": config,
            "result": result, "updates": len(updates), "last_update": updates[-1],
            "health": health, "behavior_max_abs_error": max(u["behavior_audit"]["max_abs_error"] for u in updates),
            "curves": curves, "curriculum_final_cut_histogram": dict(Counter(s["cut"] for s in curriculum.values())),
            "curriculum_is_used": config["start"]=="reset",
            "curriculum_decisions": len(curriculum_events),
            "curriculum_backward_moves": sum(e["new"]<e["old"] for e in curriculum_events),
            "curriculum_forward_moves": sum(e["new"]>e["old"] for e in curriculum_events),
            "curriculum_first_reached_zero": first_zero, "curriculum_events": curriculum_events,
            "source_ids_observed_at_multiple_positive_cuts": sum(len({c for _,c in h})>1 for h in cut_history.values()),
            "unique_discovered_targets": len(discoveries_by_target), "unique_witnesses": len(discoveries),
            "discovery_shortest_written_histogram": dict(Counter(min(r["n_lines"] for r in rows) for rows in discoveries_by_target.values())),
            "discovery_first_steps": {sid:min(r["step"] for r in rows) for sid,rows in discoveries_by_target.items()},
            "provenance": {name:common.sha(directory/name) for name in
                           ("manifest.json","result.json","updates.jsonl","groups.jsonl","curriculum.json")},
            "notes": "Discovery lengths here are raw written lengths; certificate rechecking and compression are separate audits."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode",required=True)
    e = sub.add_parser("evaluation")
    e.add_argument("--config",type=Path,required=True)
    e.add_argument("--output",type=Path,required=True)
    t = sub.add_parser("training")
    t.add_argument("--directory",type=Path,required=True)
    t.add_argument("--output",type=Path,required=True)
    t.add_argument("--window",type=int,default=500)
    args=p.parse_args()
    if args.mode=="evaluation":
        config=json.loads(args.config.read_text())
        report=evaluation_analysis(config)
        report["config_sha256"]=common.sha(args.config)
    else:
        report=training_analysis(args.directory,args.window)
    common.write_json(args.output,report)
    print(json.dumps({"status":"completed","output":str(args.output),"sha256":common.sha(args.output)}))


if __name__ == "__main__":
    main()
