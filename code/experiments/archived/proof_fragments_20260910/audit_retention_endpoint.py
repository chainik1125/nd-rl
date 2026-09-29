"""Independent CPU reconstruction of the completed retention-only trajectory."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time

import torch

import common
import learning as L
from nd_environment import FORMAT
from release_populations import check_record


def run(directory, output):
    started = time.time()
    directory = Path(directory).resolve()
    result = json.loads((directory / "result.json").read_text())
    done = json.loads((directory / "completed.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    args = manifest["args"]
    if not done["ok"] or done["result_sha256"] != common.sha(directory / "result.json"):
        raise ValueError("Retention producer completion does not bind result")
    if done["final_sha256"] != result["final_sha256"] or common.sha(directory / "final.pt") != result["final_sha256"]:
        raise ValueError("Retention final checkpoint changed")
    expected = {"batch_size": 16, "seed": 20260910, "retention_weight": .1, "lr": 5e-5, "precision": "fp32"}
    if any(args[key] != value for key, value in expected.items()):
        raise ValueError("Retention run differs from the fixed main removal ablation")
    for key in ("checkpoint", "retention", "main_manifest"):
        if common.sha(args[key]) != manifest["input_sha256"][key]:
            raise ValueError("Retention input identity changed")
    for path, digest in {**manifest["code_sha256"], **manifest["dependency_sha256"]}.items():
        if common.sha(path) != digest:
            raise ValueError("Retention code/dependency identity changed")
    main = json.loads(Path(args["main_manifest"]).read_text())
    if any(main["args"][key] != args[other] for key, other in
           (("seed", "seed"), ("lr", "lr"), ("retention_batch", "batch_size"),
            ("retention_weight", "retention_weight"), ("precision", "precision"))):
        raise ValueError("Retention optimizer/schedule differs from bound main settings")
    records = common.read_jsonl(args["retention"])
    if len(records) != 512 or len({r["source_id"] for r in records}) != 512:
        raise ValueError("Expected all 512 distinct retention certificates")
    for row in records:
        check_record(row, "retention", 6)
    torch.set_num_threads(1)
    model, tok, _ = L.load_model(args["checkpoint"], "cpu")
    rows = L.encode_population(records, tok, "empty")
    parameters = sum(p.numel() for p in model.parameters())
    config = model.config
    rng = random.Random(args["seed"] + 1000003)
    schedule = hashlib.sha256()
    exposures = Counter()
    counts = Counter()
    total = 0
    saved = {}
    requested = set(args["save_at_steps"])
    largest = max(len(r["prompt_ids"]) + len(r["response_ids"]) - 1 for r in rows)
    bound = 6 * parameters * 16 * largest + 12 * config.n_layers * 16 * largest * largest * config.d_model
    updates = common.read_jsonl(directory / "updates.jsonl")
    for step, update in enumerate(updates, 1):
        retained = [rows[rng.randrange(len(rows))] for _ in range(16)]
        ids = [r["source_id"] for r in retained]
        if update["step"] != step or update["retention_source_ids"] != ids:
            raise ValueError("Ordered retention RNG schedule differs")
        schedule.update(json.dumps(ids).encode())
        exposures.update(ids)
        lengths = [len(r["prompt_ids"]) + len(r["response_ids"]) for r in retained]
        actions = sum(len(r["response_ids"]) for r in retained)
        width = max(lengths) - 1
        cost = 6 * parameters * 16 * width + 12 * config.n_layers * 16 * width * width * config.d_model
        previous = total
        total += cost
        if (not 0 < cost <= bound or update["step_flop_proxy"] != cost or update["cumulative_flop_proxy"] != total
                or update["retention_examples"] != 16 or update["retention_response_tokens"] != actions
                or update["policy_loss"] != 0 or update["weighted_retention_loss"] != .1 * update["retention_loss"]):
            raise ValueError("Retention step cost, token count or loss metadata differs")
        counts.update(calls=1, padded_token_slots=16 * width, actual_input_tokens=sum(n - 1 for n in lengths),
                      action_tokens=actions, supplied_tokens=sum(lengths) - actions)
        if step in requested:
            checkpoint_path = directory / f"step_{step:06d}.pt"
            receipt_path = directory / f"step_{step:06d}.json"
            receipt = json.loads(receipt_path.read_text())
            expected_endpoint = {"step": step, "retention_examples": 16 * step,
                                 "retention_response_tokens": counts["action_tokens"],
                                 "retention_schedule_sha256": schedule.hexdigest(), "actual_flop_proxy": total,
                                 "endpoint_kind": "retention_exposure_matched_only", "compute_matched_to_main": False}
            if common.sha(checkpoint_path) != receipt["checkpoint_sha256"]:
                raise ValueError("Exposure checkpoint hash changed")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            metadata = checkpoint["metadata"]
            if any(receipt[key] != value or metadata[key] != value for key, value in expected_endpoint.items()):
                raise ValueError("Exposure checkpoint step/schedule/compute differs")
            if metadata.get("format") != FORMAT or metadata.get("source_checkpoint_sha256") != manifest["input_sha256"]["checkpoint"]:
                raise ValueError("Exposure checkpoint ancestry or format differs")
            saved[str(step)] = {**expected_endpoint, "checkpoint_sha256": receipt["checkpoint_sha256"]}
            del checkpoint
    if set(map(int, saved)) != requested or len(updates) != result["steps"]:
        raise ValueError("Missing declared exposure checkpoint or update")
    budget = int(float(args["max_flop_proxy"]))
    if not previous < budget <= total or result["stop_reason"] != "flop_proxy_ceiling":
        raise ValueError("Retention endpoint is not the first completed compute crossing")
    if (total != result["total_flop_proxy"] or total != L.total_flop_proxy(result["ledger"])
            or result["flop_proxy_overshoot"] != total - budget or result["worst_case_one_step_flop_proxy"] != bound
            or result["retention_schedule_sha256"] != schedule.hexdigest()
            or result["retention_examples"] != len(updates) * 16 or result["retention_response_tokens"] != counts["action_tokens"]
            or result["source_exposure_counts"] != dict(exposures) or result["policy_rollouts"] != 0
            or result["policy_updates_from_rollouts"] != 0 or not result["compute_ceiling_matched_to_main"]):
        raise ValueError("Final retention compute/exposure accounting differs")
    if any(result["ledger"]["rl_retention"][key] != value for key, value in counts.items()):
        raise ValueError("Final retention dense accounting counters differ")
    final = torch.load(directory / "final.pt", map_location="cpu", weights_only=True)["metadata"]
    if (final["step"] != len(updates) or final["endpoint_kind"] != "declared_total_compute_budget_endpoint"
            or final["actual_flop_proxy"] != total or final["retention_schedule_sha256"] != schedule.hexdigest()
            or final["retention_examples"] != result["retention_examples"] or not final["compute_ceiling_matched_to_main"]):
        raise ValueError("Final retention checkpoint metadata differs")
    receipt = {"status": "passed", "directory": str(directory), "steps": len(updates),
               "retention_examples": result["retention_examples"], "retention_response_tokens": counts["action_tokens"],
               "unique_sources": len(exposures), "total_flop_proxy": total, "overshoot": total - budget,
               "worst_case_one_step_flop_proxy": bound, "retention_schedule_sha256": schedule.hexdigest(),
               "exposure_checkpoints": saved, "final_checkpoint_sha256": result["final_sha256"],
               "manifest_sha256": common.sha(directory / "manifest.json"), "result_sha256": done["result_sha256"],
               "updates_sha256": common.sha(directory / "updates.jsonl"), "audit_code_sha256": common.sha(__file__),
               "elapsed_seconds": time.time() - started, "gpu_work_launched": False, "model_forward_or_backward_executed": False,
               "final_population_accessed": False,
               "limitation": "Audit checks all schedules, costs and saved metadata, not every parameter/Adam update; zero-policy equivalence was tested separately"}
    if Path(output).exists():
        raise ValueError("Refusing to overwrite retention audit receipt")
    common.write_json(output, receipt)
    print(json.dumps({key: receipt[key] for key in ("status", "steps", "total_flop_proxy", "overshoot", "retention_examples")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.run, args.output)
