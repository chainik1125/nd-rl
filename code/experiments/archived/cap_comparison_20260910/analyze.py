"""Strict completed-arm replay and prespecified comparisons; no final-test access."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "snapshots/main_v1"), str(ROOT / "snapshots/main_v1/vendor")]
import pipeline
import run as trainer
from proof_diagnostics import compress_proof
from strict_verify import safe_verify


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, obj):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(obj, indent=2) + "\n")
    temporary.replace(path)


def prefix_outcomes(attempts, count, *, k=None, token_budget=None):
    """Whole attempts in order; partial last attempt consumes cost, no outcome."""
    solved = [0] * count
    spent = [0] * count
    by_target = [[] for _ in range(count)]
    for row in attempts:
        by_target[row["index"]].append(row)
    for i, rows in enumerate(by_target):
        for row in sorted(rows, key=lambda x: x["attempt"]):
            if k is not None and row["attempt"] >= k:
                break
            length = row["generated_tokens"]
            if token_budget is not None and spent[i] + length > token_budget:
                spent[i] = token_budget
                break
            spent[i] += length
            solved[i] |= int(row["reward"])
    return {"outcomes": solved, "solved": sum(solved), "problems": count,
            "accuracy": sum(solved) / count, "charged_tokens": sum(spent)}


def paired(before, after, protocol):
    differences = np.asarray(after, dtype=float) - np.asarray(before, dtype=float)
    rng = np.random.default_rng(protocol["seed"])
    means = differences[rng.integers(0, len(differences), size=(protocol["resamples"], len(differences)))].mean(axis=1)
    low, high = np.quantile(means, [.025, .975])
    return {"delta_solved": int(differences.sum()), "delta_accuracy": float(differences.mean()),
            "paired_target_bootstrap95": [float(low), float(high)],
            "newly_solved": int((differences > 0).sum()), "lost": int((differences < 0).sum()),
            "scope": protocol["scope"]}


def validate_arm(root, name, protocol_sha):
    base = root / "runs" / name
    complete = read(base / "pipeline_completed.json")
    spec_path = root / "plans" / f"{name}.json"
    spec = read(spec_path)
    protocol = read(root / "plans/protocol.json")
    if (complete["status"] != "complete" or complete["protocol_sha256"] != protocol_sha
            or complete["config_sha256"] != sha(spec_path)
            or protocol["input_sha256"][str(spec_path.relative_to(root))] != sha(spec_path)
            or complete["progress_sha256"] != sha(base / "progress.json")):
        raise ValueError("Completed pipeline identity differs")
    for stage in spec["training_stages"]:
        if pipeline.validate_training(stage, root) != complete["training"][stage["command_name"]]:
            raise ValueError("Training endpoint differs from complete pipeline")
    release_path = base / "development_release.json"
    if sha(release_path) != complete["development_release_sha256"]:
        raise ValueError("Development release differs")
    release = read(release_path)
    if (release["protocol_sha256"] != protocol_sha or release["config_sha256"] != sha(spec_path)
            or release["full_rl_budget_reached"] is not True):
        raise ValueError("Development release is not bound to this full-budget experiment")
    for stage in complete["completed_stages"]:
        if stage["name"] in ("sft", "rl"):
            continue
        directory = base / "evaluations" / stage["name"]
        for field, filename in (("completed_sha256", "completed.json"), ("summary_sha256", "summary.json"),
                                ("attempts_sha256", "attempts.jsonl")):
            if sha(directory / filename) != stage[field]:
                raise ValueError("Recorded evaluation changed")
    return spec, complete, release


def replay(root, spec, phase, procedure, data, release):
    directory = root / spec["run_base"] / "evaluations" / f"{phase}_dev_{procedure}"
    manifest = read(directory / "manifest.json")
    summary = read(directory / "summary.json")
    completion = read(directory / "completed.json")
    settings = manifest["args"]
    expected_checkpoint = sha(root / spec["run_base"] / phase / "final.pt")
    expected_k, expected_temp = (1, 0) if procedure == "greedy" else (16, .8)
    if (settings["mode"] != spec["mode"] or settings["samples"] != expected_k
            or settings["temperature"] != expected_temp or settings["max_new_tokens"] != 1536
            or settings["max_proof_lines"] != spec["cap"] or settings["seed"] != spec["evaluation"]["seed"]
            or manifest["source_sha256"] != expected_checkpoint
            or expected_checkpoint not in release["checkpoints"]
            or manifest["data_sha256"] != spec["development_data"]["sha256"]
            or settings["precision"] != "fp32" or settings["decode_batch"] != spec["evaluation"]["decode_batch"]
            or settings["max_seq_len"] != spec["evaluation"]["max_seq_len"]
            or completion["stage"] != "eval"
            or completion["checkpoint_sha256"] != expected_checkpoint
            or manifest["evaluation_release_sha256"] != sha(root / spec["development_release"])):
        raise ValueError("Evaluation protocol/checkpoint mismatch")
    tok = trainer.ScratchTokenizer()
    rows, seen, raw_shortest, compressed_shortest = [], set(), {}, {}
    truncated = 0
    with (directory / "attempts.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            index, attempt = row["index"], row["attempt"]
            if not 0 <= index < len(data) or not 0 <= attempt < expected_k or (index, attempt) in seen:
                raise ValueError("Duplicate or invalid attempt identity")
            seen.add((index, attempt))
            target = data[index]
            prompt = trainer.target_prompt(target)
            if row["prompt"] != prompt or row["prompt_ids"] != tok.encode_prompt(prompt):
                raise ValueError("Saved attempt theorem differs from immutable input")
            judged = trainer.judge_response(prompt, row["response_ids"], tok, spec["mode"])
            for key in ("reward", "reason", "body", "found_len"):
                if judged[key] != row[key]:
                    raise ValueError(f"Strict saved-response replay disagrees: {key}")
            if row["generated_tokens"] != len(row["response_ids"]):
                raise ValueError("Generated token denominator mismatch")
            truncated += int(row.get("finish_reason") == "token_limit")
            if row["reward"]:
                compressed = compress_proof(prompt, judged["body"])
                valid, reason, length = safe_verify(prompt, compressed["compressed_proof"])
                if not valid or length != compressed["compressed_length"]:
                    raise ValueError("Compressed certificate fails strict reverification: " + reason)
                raw_shortest[index] = min(raw_shortest.get(index, 10**9), judged["found_len"])
                compressed_shortest[index] = min(compressed_shortest.get(index, 10**9), length)
            rows.append({"index": index, "attempt": attempt, "reward": row["reward"],
                         "generated_tokens": row["generated_tokens"]})
    if len(seen) != len(data) * expected_k:
        raise ValueError("Evaluation has missing attempts")
    coverage = prefix_outcomes(rows, len(data), k=expected_k)
    if (coverage["solved"] != summary["solved"] or coverage["charged_tokens"] != summary["actual_response_tokens"]
            or summary["problems"] != len(data) or summary["samples_per_problem"] != expected_k
            or summary["data_sha256"] != manifest["data_sha256"]
            or summary["truncated_attempts"] != truncated
            or summary["checkpoint_sha256"] != expected_checkpoint):
        raise ValueError("Strict replay does not reproduce source summary")
    lengths = [trainer.reference_length(t) for t in data]
    if any(type(n) is not int or n < 1 for n in lengths):
        raise ValueError("Missing reference witness lengths")
    cohorts = {}
    for threshold in sorted({6, 10, 12, spec["cap"]}):
        selected = [i for i, length in enumerate(lengths) if length > threshold]
        solved = sum(coverage["outcomes"][i] for i in selected)
        cohorts[str(threshold)] = {"solved": solved, "problems": len(selected),
                                   "accuracy": solved / len(selected) if selected else None}
    details = {**coverage, "cap": spec["cap"], "mode": spec["mode"], "phase": phase,
               "procedure": procedure, "checkpoint_sha256": expected_checkpoint,
               "attempts_sha256": sha(directory / "attempts.jsonl"),
               "strict_replayed_attempts": len(rows), "cohorts_above_reference": cohorts,
               "truncated_attempts": truncated, "truncated_fraction": truncated / len(rows),
               "max_solved_reference": max((lengths[i] for i in raw_shortest), default=None),
               "raw_shortest_by_target": raw_shortest, "compressed_shortest_by_target": compressed_shortest,
               "max_shortest_raw_witness": max(raw_shortest.values(), default=None),
               "max_shortest_compressed_witness": max(compressed_shortest.values(), default=None),
               "raw_length_histogram": dict(Counter(raw_shortest.values())),
               "compressed_length_histogram": dict(Counter(compressed_shortest.values()))}
    return details, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    plan = read(root / "reports/analysis_protocol.json")
    protocol_sha = sha(root / "plans/protocol.json")
    if plan["training_protocol_sha256"] != protocol_sha or not plan["forbid_final_test"]:
        raise ValueError("Analysis/training protocol mismatch")
    inventory = {name: (root / "runs" / name / "pipeline_completed.json").exists() for name in plan["arms"]}
    if not args.partial and not all(inventory.values()):
        raise ValueError("Full comparison requires all six completed arms")
    output = {"training_protocol_sha256": protocol_sha,
              "analysis_protocol_sha256": sha(root / "reports/analysis_protocol.json"),
              "analysis_source_sha256": sha(Path(__file__).resolve()),
              "complete": all(inventory.values()), "inventory": inventory, "arms": {}}
    for name, completed in inventory.items():
        if not completed:
            continue
        spec, completion, release = validate_arm(root, name, protocol_sha)
        # The only textbook payload opened, after actual full-budget completion.
        path = root / "evaluation/textbook_dev.jsonl"
        if sha(path) != spec["development_data"]["sha256"] or str(path) != release["data_path"]:
            raise ValueError("Released development payload identity differs")
        data = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert len(data) == 58
        result = {"label": spec["label"], "cap": spec["cap"], "mode": spec["mode"],
                  "training": completion["training"], "endpoints": {}, "contrasts": {}}
        for procedure in ("greedy", "sampled"):
            endpoints = {}
            for phase in ("sft", "rl"):
                detail, rows = replay(root, spec, phase, procedure, data, release)
                detail["prefix_k"] = {str(k): prefix_outcomes(rows, len(data), k=k)
                                      for k in ([1] if procedure == "greedy" else plan["samples"])}
                detail["prefix_tokens"] = {str(b): prefix_outcomes(rows, len(data), token_budget=b)
                                           for b in plan["token_budgets"]}
                result["endpoints"][f"{phase}_{procedure}"] = detail
                endpoints[phase] = detail
            comparisons = {"full": paired(endpoints["sft"]["outcomes"], endpoints["rl"]["outcomes"], plan["bootstrap"])}
            for kind in ("prefix_k", "prefix_tokens"):
                comparisons[kind] = {key: paired(endpoints["sft"][kind][key]["outcomes"],
                                                endpoints["rl"][kind][key]["outcomes"], plan["bootstrap"])
                                     for key in endpoints["sft"][kind]}
            result["contrasts"][procedure] = comparisons
        output["arms"][name] = result
        print(json.dumps({"arm": name, "replayed": True,
                          "scores": {k: v["solved"] for k,v in result["endpoints"].items()}}), flush=True)
    write(root / "reports/analysis_results.json", output)


if __name__ == "__main__":
    main()
