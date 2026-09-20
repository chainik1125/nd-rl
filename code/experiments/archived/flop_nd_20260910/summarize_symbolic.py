"""Paired analysis of completely replayed symbolic endpoint evaluations."""
import argparse
import json
from pathlib import Path
import numpy as np
import common
from analyze_lm import paired_context_bootstrap
from analyze_symbolic import strata


def analyze(study, population, split, profile):
    prefix, seeds = ("symbolic", (4301, 4302)) if study == "general" else ("minimal", (4401, 4402))
    labels = [f"{mode}_s{seed}" for mode in ("initial", "full", "reset", "bc") for seed in seeds] + ["uniform"]
    reports, paths = {}, {}
    for label in labels:
        path = common.ROOT / "reports" / f"audit_{prefix}_{label}_{population}_{split}_{profile}_v1.json"
        report = json.loads(path.read_text())
        if report.get("status") != "complete":
            raise ValueError("Symbolic result has not passed complete replay audit")
        for artifact, digest in report["input_sha256"].items():
            if common.sha(artifact) != digest:
                raise ValueError(f"A replay-audited artifact changed: {artifact}")
        reports[label], paths[label] = report, path
    profiles = [{k: v for k, v in r["profile"].items() if k != "policy"} for r in reports.values()]
    if len({common.digest(p) for p in profiles}) != 1:
        raise ValueError("Compared policies have different inference profiles")
    audit_paths = {r["audit"] for r in reports.values()}
    if len(audit_paths) != 1:
        raise ValueError("Compared policies use different populations")
    audit_path = Path(next(iter(audit_paths)))
    audit = {r["source_id"]: r for r in common.read_jsonl(audit_path)}
    if any(set(r["rows"]) != set(audit) for r in reports.values()):
        raise ValueError("Different theorem identities in paired comparison")
    differences = {}
    for label, positive, negative in (("reset_vs_full", "reset", "full"), ("reset_vs_bc", "reset", "bc"),
                                       ("full_vs_bc", "full", "bc"), ("reset_vs_initial", "reset", "initial")):
        differences[label] = {}
        for stratum, ids in strata(audit).items():
            values = {}
            for metric in ("first", "any", "raw_eligible_512_any", "raw_eligible_1024_any", "any_pruned_over6"):
                by_seed = {f"seed{seed}": np.array([reports[f"{positive}_s{seed}"]["rows"][sid][metric] -
                    reports[f"{negative}_s{seed}"]["rows"][sid][metric] for sid in ids], dtype=float) for seed in seeds}
                by_seed["mean_across_seeds"] = np.mean(list(by_seed.values()), axis=0)
                values[metric] = {key: paired_context_bootstrap(vector, [audit[sid]["context_id"] for sid in ids],
                                                              4000, 202609101936) for key, vector in by_seed.items()}
            differences[label][stratum] = values
    return {"status": "complete", "study": study, "population": population, "split": split, "profile": profile,
        "inference_profile": profiles[0], "seeds": seeds,
        "audit_path": str(audit_path), "audit_sha256": common.sha(audit_path),
        "summaries": {k: r["summaries"] for k, r in reports.items()}, "contrasts": differences,
        "checkpoint_sha256": {k: r["checkpoint_sha256"] for k, r in reports.items()},
        "input_sha256": {str(path): common.sha(path) for path in [*paths.values(), Path(__file__), Path(__file__).with_name("analyze_lm.py"), Path(__file__).with_name("analyze_symbolic.py")]},
        "notes": ["PPO full/reset are matched by sampled inference actions, not candidate-scoring compute or wall time.",
                  "Behavior cloning is a separately budgeted demonstration-fitting control, not compute matched; its source training accuracy is reported separately.",
                  "Uniform greedy-filename output is one random action-policy attempt.",
                  "All theorem coverage comes from original theorem inputs, within a symbolic legal-action interface.",
                  "Intervals condition on these two trained seeds; they do not estimate training-seed uncertainty.",
                  "Exact raw response-token eligibility is a serialization constraint, not a matched-compute comparison to language-model decoding."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("general", "minimal"), required=True)
    parser.add_argument("--population", choices=("primary", "structured"), required=True)
    parser.add_argument("--split", choices=("development", "final"), required=True)
    parser.add_argument("--profile", choices=("greedy", "sample4"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.study, args.population, args.split, args.profile)
    common.write_json(args.output, result)
    print(json.dumps({"status": "complete", "output": str(args.output), "sha256": common.sha(args.output)}))
