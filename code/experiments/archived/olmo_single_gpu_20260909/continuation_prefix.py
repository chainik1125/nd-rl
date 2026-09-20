"""Read-only, fail-closed validation of the successful v3 prefix and failed RL.

This does not authorize evaluation, assert v3 pipeline success, create files,
load a model, or inspect protected textbook data. Synthetic evaluation inputs
are hashed only as part of the original source freeze; never parsed here.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path

SPRINT = Path(__file__).resolve().parents[1]
PINNED = {
    "plans/primary_source_freeze_v3.json": "a568d0c534b08d7062f28d64746b553c1aebc94e2dcafbea7914886dcf4db774",
    "runs/sft/final/adapter_model.safetensors": "2700b000bbd02562406469dfe6d91a9c54d6fdaa8ebb3692ecdeb9781e599c2d",
    "runs/dpo/final/adapter_model.safetensors": "e9f53db46eb1ec8b54db8a513bf0cae4b3c6386397d90a2ec8a72e2b30352b8a",
    "runs/pairs/pairs.jsonl": "223968eb8168a4d92005cf9e0e8bf797feb5f3355e3c64b2217b7b1b17f0b12c",
    "runs/pairs/pair_samples.jsonl": "2691fe5b36900c279373d099955f5a63edfa790d842f15ee0ff041184d088263",
    "runs/grpo/events.jsonl": "eba0858c96091a4dcbf972be21228e314171187c7d8f6b5edd95b1a24d41d26e",
    "runs/grpo/rollouts.jsonl": "994af2490d5da007a64a9a00c0b19e4c58c55bb1d7e8502eb9b5e7ef7bc83e9e",
}
AMENDABLE = frozenset({"code/modeling.py", "code/train.py", "code/launch.py"})
POLICY = {"repetition_penalty": 1.0, "temperature": 1.0, "top_k": 0, "top_p": 1.0}
MODEL_ID = "allenai/Olmo-3-1125-32B"
MODEL_REVISION = "c2b61dae89a1ad10e4ad5653d0e46b590902607b"


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Receipts:
    """Hashes parsed metadata from the same bytes, without retaining weights."""
    def __init__(self):
        self.sha256 = {}

    def remember(self, path, digest):
        name = str(Path(path).resolve())
        require(name not in self.sha256 or self.sha256[name] == digest, f"Input changed during validation: {name}")
        self.sha256[name] = digest
        return digest

    def digest(self, path):
        with Path(path).open("rb") as stream:
            return self.remember(path, hashlib.file_digest(stream, "sha256").hexdigest())

    def json(self, path):
        raw = Path(path).read_bytes()
        self.remember(path, hashlib.sha256(raw).hexdigest())
        return json.loads(raw)

    def rows(self, path):
        raw = Path(path).read_bytes()
        self.remember(path, hashlib.sha256(raw).hexdigest())
        require(raw.endswith(b"\n"), f"Uncommitted JSONL tail: {path}")
        return [json.loads(line) for line in raw.splitlines() if line.strip()]

    def verify_stable(self):
        for name in list(self.sha256):
            self.digest(name)


def _path(sprint, value):
    value = Path(value)
    return (sprint / value).resolve() if not value.is_absolute() else value.resolve()


def _one(events, kind):
    found = [row for row in events if row.get("kind") == kind]
    require(len(found) == 1, f"Expected exactly one {kind} receipt")
    return found[0]


def _same_event(summary, events, kind):
    event = _one(events, kind)
    require({k: v for k, v in event.items() if k not in {"kind", "utc"}} == summary,
            f"Summary does not match original {kind} event")


def _load_tools(sprint):
    """Import only validated local CPU helpers; never load model weights."""
    import sys
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    before = list(sys.path)
    try:
        protocol = load("_prefix_protocol", sprint / "code/protocol.py")
        modeling = load("_prefix_v3_modeling", sprint / "code/archive_v3/modeling.py")
        tokenizer = modeling.load_tokenizer(sprint / "models/olmo32b")
    finally:
        sys.path[:] = before
    return protocol, modeling, tokenizer


def _check_tensor_header(path, audit):
    # Header-only dtype/shape validation: no tensors are copied and no GPU runs.
    from safetensors import safe_open
    count = parameters = 0
    with safe_open(str(path), framework="np") as checkpoint:
        for key in checkpoint.keys():
            tensor = checkpoint.get_slice(key)
            require(tensor.get_dtype() == "F32", "Checkpoint contains a non-FP32 tensor")
            count += 1
            parameters += math.prod(tensor.get_shape())
    require((count, parameters) == (audit["tensors"], audit["parameters"]), "Saved tensor header/receipt mismatch")


def _check_audit(audit, weight, digest, name="default"):
    require(audit.get("all_tensors_exact") is True and audit.get("dtype") == "float32"
            and audit.get("adapter_name") == name and Path(audit.get("checkpoint", "")).resolve() == weight
            and audit.get("sha256") == digest and audit.get("tensors") == 896
            and audit.get("parameters") == 134217728, "Saved/loaded checkpoint identity mismatch")


def _check_samples(sprint, receipts, runs, events, summary, expected_sft):
    protocol, modeling, tokenizer = _load_tools(sprint)
    vocabulary_size = len(tokenizer)
    gold = receipts.rows(sprint / "data/shared_cap6.jsonl")
    pool = {row["canonical_hash"]: row for row in gold}
    require(len(pool) == len(gold) == 32, "Training bank identity count changed")
    for row in gold:
        verdict = protocol.judge(row["prompt"], protocol.render_completion(row["proof"]))
        require(verdict["accepted"] and verdict["n_lines"] == row["n_lines"] <= 6, "Invalid shared short gold proof")
    pairs = receipts.rows(sprint / "runs/pairs/pairs.jsonl")
    samples = receipts.rows(sprint / "runs/pairs/pair_samples.jsonl")
    pair_map = {row["canonical_hash"]: row for row in pairs}
    require(len(pair_map) == len(pairs) == 5 and len(samples) == 120, "Pinned preference coverage changed")
    sampling_events = [r for r in events if r.get("kind") == "preference_sample"]
    require(len(sampling_events) == len(samples), "Missing preference sample event")
    first_invalid, seen, reasons = {}, set(), Counter()
    for sample, event in zip(samples, sampling_events, strict=True):
        key, attempt, output = sample["canonical_hash"], sample["attempt"], sample["sample"]
        require(key in pool and type(attempt) is int and 0 <= attempt < 4
                and (key, attempt) not in seen and key not in first_invalid, "Invalid/repeated/late preference sample")
        require(all((key, earlier) in seen for earlier in range(attempt)), "Missing earlier preference attempt")
        seen.add((key, attempt))
        source = pool[key]
        prompt_ids = tokenizer.encode(protocol.render_prompt(source["prompt"]), add_special_tokens=False)
        ids = output["response_ids"]
        require(ids and all(type(token) is int and 0 <= token < vocabulary_size for token in ids), "Invalid sampled token IDs")
        text = modeling.decode_completion(tokenizer, ids)
        trimmed, stop = modeling.trim_completion(tokenizer, ids)
        require(text == output["text"] and trimmed == ids and stop == output["finish_reason"], "Preference native replay mismatch")
        require(output["prompt_ids"] == prompt_ids and output["prompt_tokens"] == len(prompt_ids)
                and output["completion_tokens"] == len(ids) <= 2048 and len(prompt_ids) + len(ids) <= 4096
                and output["sampling"] == POLICY, "Preference prompt, budget, or sampling mismatch")
        verdict = protocol.judge(source["prompt"], text)
        require(verdict == sample["verdict"], "Preference verifier replay differs")
        require(event["canonical_hash"] == key and event["attempt"] == attempt
                and event["verifier_valid"] == verdict["accepted"]
                and event["completion_tokens"] == len(ids) and event["finish_reason"] == stop,
                "Preference sample/event identity mismatch")
        reasons[verdict["reason"]] += 1
        if not verdict["accepted"]:
            first_invalid[key] = sample
    require(set(first_invalid) == set(pair_map), "Pairs do not equal first actual invalid samples")
    for key, pair in pair_map.items():
        output = first_invalid[key]["sample"]
        require(pair["prompt"] == protocol.render_prompt(pool[key]["prompt"])
                and pair["chosen"] == protocol.render_completion(pool[key]["proof"])
                and pair["rejected"] == output["text"] and pair["sampled_response_ids"] == output["response_ids"]
                and pair["sample_seed"] == output["seed"] and pair["sampling"] == output["sampling"]
                and pair["verdict"] == first_invalid[key]["verdict"]
                and pair["generation_policy"] == "sft_adapter"
                and _path(sprint, pair["source_adapter"]) == sprint / "runs/sft/final"
                and pair["source_adapter_sha256"] == expected_sft, "DPO pair lacks exact SFT sample provenance")
        for completion in (pair["chosen"], pair["rejected"]):
            require(len(tokenizer.encode(pair["prompt"], add_special_tokens=False))
                    + len(tokenizer.encode(completion, add_special_tokens=False)) + 1 <= 4096,
                    "DPO pair would require truncation")
    require(summary["pairs"] == len(pairs) and summary["shared_pool"] == 32
            and summary["covered_theorems"] == sorted(pair_map) and summary["reasons"] == dict(reasons)
            and summary["status"] == "partial", "Preference summary/coverage mismatch")
    # Every unpaired theorem exhausted its four allowed attempts.
    require(all(key in pair_map or all((key, n) in seen for n in range(4)) for key in pool), "Unexplained missing preference coverage")
    batches = [r for r in events if r.get("kind") == "preference_generation_batch"]
    flattened = []
    for batch in batches:
        keys = batch["canonical_hashes"]
        require(keys and len(keys) <= 16 and len(set(keys)) == len(keys), "Invalid preference generation batch")
        seed_key = f"{runs['pairs']['args']['seed']}:{batch['attempt']}:" + ":".join(keys)
        seed = int(hashlib.sha256(seed_key.encode()).hexdigest()[:15], 16)
        require(batch["seed"] == seed, "Preference batch seed differs from frozen schedule")
        for key in keys:
            flattened.append((key, batch["attempt"], seed, len(keys)))
    require(flattened == [(r["canonical_hash"], r["attempt"], r["sample"]["seed"], r["sample"]["sampling_batch_size"]) for r in samples],
            "Preference samples do not follow native batch receipts")
    return {"pairs": len(pairs), "sampled_responses": len(samples), "shared_pool": 32,
            "covered_theorems": sorted(pair_map), "coverage_status": "partial", "strict_native_replay": True}


def validate_prefix(sprint=SPRINT, *, approved_source_changes=None):
    """Return receipt hashes, or fail closed; never mutate any sprint artifact.

    ``approved_source_changes`` maps ONLY code/modeling.py, code/train.py or
    code/launch.py to explicitly approved current SHA256 values. Their exact v3
    counterparts must remain in code/archive_v3. Unlisted drift always fails.
    The parameterizable sprint root exists for isolated fixture tests.
    """
    sprint = Path(sprint).resolve()
    approvals = dict(approved_source_changes or {})
    require(set(approvals) <= AMENDABLE, "Unapproved source-change path")
    require(all(isinstance(v, str) and len(v) == 64 and set(v) <= set("0123456789abcdef") for v in approvals.values()), "Invalid approved source hash")
    receipts = Receipts()
    for relative, expected in PINNED.items():
        require(receipts.digest(sprint / relative) == expected, f"Pinned v3 artifact changed: {relative}")
    freeze_path = sprint / "plans/primary_source_freeze_v3.json"
    freeze = receipts.json(freeze_path)
    require(freeze.get("version") == 3 and freeze.get("sha256"), "Expected original v3 freeze")
    old_sources, changes = freeze["sha256"], {}
    for name, expected in old_sources.items():
        path = Path(name).resolve()
        relative = str(path.relative_to(sprint)) if path.is_relative_to(sprint) else None
        current = receipts.digest(path)
        if relative in AMENDABLE:
            archive = sprint / "code/archive_v3" / path.name
            require(receipts.digest(archive) == expected, f"Original source archive changed: {relative}")
        if current != expected:
            require(relative in approvals and approvals[relative] == current, f"Unapproved frozen-source drift: {name}")
            changes[relative] = {"original_sha256": expected, "current_sha256": current,
                                 "original_path": str(sprint / "code/archive_v3" / path.name)}
    require(set(approvals) == set(changes), "Approval does not exactly describe current source changes")
    for forbidden in ("runs/training_pipeline_v3/completed.json", "runs/grpo/summary.json", "runs/grpo/final", "runs/grpo/trainer_state.json"):
        require(not (sprint / forbidden).exists(), f"Failed v3 endpoint unexpectedly became complete: {forbidden}")
    for pattern in ("checkpoint-*", "final*", "summary_resume_*", "run_resume_*"):
        require(not list((sprint / "runs/grpo").glob(pattern)), "Failed GRPO directory was reused for continuation")
    plan_path = sprint / "plans/primary_training_v3.json"
    plan = receipts.json(plan_path)
    require(receipts.digest(sprint / "runs/training_pipeline_v3/plan.json") == receipts.digest(plan_path), "Pipeline loaded a different plan")
    require(_path(sprint, plan["source_freeze"]) == freeze_path and _path(sprint, plan["out"]) == sprint / "runs/training_pipeline_v3", "Pipeline plan identity mismatch")
    steps = plan["steps"]
    require([row["name"] for row in steps] == ["sft", "pairs", "dpo", "grpo"], "Pipeline stage ordering changed")
    timeline = receipts.rows(sprint / "runs/training_pipeline_v3/events.jsonl")
    require([(r.get("event"), r.get("name")) for r in timeline]
            == [(kind, name) for name in ("sft", "pairs", "dpo", "grpo") for kind in ("stage_start", "stage_pid", "stage_exit")], "Pipeline prefix events missing, repeated or reordered")
    for i, step in enumerate(steps):
        start, pid, end = timeline[3*i:3*i+3]
        require(step["entrypoint"] == "train.py" and start["command"] == [str(sprint / ".venv/bin/python"), "-u", str(sprint / "code/train.py"), *step["args"]], "Pipeline command differs from frozen plan")
        require(type(pid["pid"]) is int and pid["pid"] > 0 and type(end["exit_code"]) is int
                and end["exit_code"] == (1 if step["name"] == "grpo" else 0), "Pipeline prefix did not succeed then fail at GRPO")
    job = receipts.json(sprint / "jobs/primary_training_v3/completed.json")
    require(type(job.get("exit_code")) is int and job["exit_code"] == 1, "Owning failed v3 job receipt missing")
    require(receipts.json(sprint / "jobs/primary_training_v3/spec.json") == receipts.json(sprint / "plans/primary_training_job_v3.json"), "Owning job spec identity changed")
    runs, events, summaries = {}, {}, {}
    for stage in ("sft", "pairs", "dpo", "grpo"):
        run = runs[stage] = receipts.json(sprint / "runs" / stage / "run.json")
        events[stage] = receipts.rows(sprint / "runs" / stage / "events.jsonl")
        args = run["args"]
        require(run.get("mechanical_fixture") is False and args["stage"] == stage and args.get("resume") is None
                and args.get("resume_pairs") is False and _path(sprint, args["out"]) == sprint / "runs" / stage, "Wrong/mechanical/resumed stage provenance")
        require(run["source_sha256"] == {name: old_sources[str(sprint / "code" / name)] for name in ("modeling.py", "train.py")}, "Stage source is not original v3")
        require(run["data"]["rows"] == 32 and run["data"]["proof_cap"] == 6, "Stage training pool/cap changed")
        require(set(run["data"]["input_sha256"]) == {"shared_cap6.jsonl", "rl_short_prompts.jsonl", "training_exclusions.json"}, "Stage training input set changed")
        require(run["data"]["manifest_sha256"] == receipts.digest(sprint / "data/manifest.json"), "Training manifest changed")
        for name, expected in run["data"]["input_sha256"].items():
            path = sprint / "data" / name
            require(str(path) in old_sources and receipts.digest(path) == expected == old_sources[str(path)], "Stage data identity mismatch")
        for relative, expected in run["data"]["protocol"]["files"].items():
            path = sprint.parent / relative
            require(str(path) in old_sources and receipts.digest(path) == expected == old_sources[str(path)], "Stage verifier/protocol identity mismatch")
        require(args["rank"] == 16 and args["seed"] == 202609092014 and args["max_length"] == 4096
                and args["max_completion_length"] == 2048, "Stage common schedule changed")
        if stage != "sft":
            prior = "dpo" if stage == "grpo" else "sft"
            require(_path(sprint, args["adapter"]) == sprint / "runs" / prior / "final", "Stage incoming checkpoint path changed")
            identity = _one(events[stage], "model_loaded")["identity"]
            require(_path(sprint, identity["adapter_path"]) == sprint / "runs" / prior / "final", "Loaded adapter path differs from requested prefix")
            for filename in ("adapter_model.safetensors", "adapter_config.json"):
                require(identity["adapter_sha256"][filename] == receipts.digest(sprint / "runs" / prior / "final" / filename), "Stage incoming checkpoint/config hash changed")
            require(identity["adapter_sha256"]["adapter_model.safetensors"] == PINNED[f"runs/{prior}/final/adapter_model.safetensors"], "Stage incoming checkpoint hash changed")
        if stage in ("sft", "dpo"):
            count = 80 if stage == "sft" else 40
            require(args["steps"] == count and args["accumulation"] == 4
                    and args["learning_rate"] == (2e-5 if stage == "sft" else 5e-6), "Prefix optimizer schedule changed")
            summary = summaries[stage] = receipts.json(sprint / "runs" / stage / "summary.json")
            _same_event(summary, events[stage], "stage_complete")
            require(summary["stage"] == stage and summary["status"] == "complete" and summary["mechanical_fixture"] is False
                    and summary["optimizer_steps"] == summary["requested_steps"] == count and summary["deadline_reached"] is False
                    and summary["nonzero_gradient_steps"] > 0 and summary["observed_parameter_changes"] > 0, "Incomplete/nonlearning prefix endpoint")
            require(summary["protocol"] == run["data"]["protocol"], "Endpoint protocol differs from its run")
            adapter = sprint / "runs" / stage / "final"
            require(_path(sprint, summary["adapter"]) == adapter, "Endpoint adapter path changed")
            weight = adapter / "adapter_model.safetensors"
            _check_audit(summary["saved_tensor_audit"], weight, receipts.digest(weight))
            _check_tensor_header(weight, summary["saved_tensor_audit"])
            config = receipts.json(adapter / "adapter_config.json")
            require(config["r"] == 16 and config["lora_alpha"] == 32 and config["lora_dropout"] == 0.0
                    and config["peft_type"] == "LORA" and config["bias"] == "none", "Endpoint adapter config changed")
            require(any(r.get("kind") == "parameter_update" and r.get("max_abs_change", 0) > 0 for r in events[stage]), "Missing genuine prefix parameter update")
    require(all(run["data"] == runs["sft"]["data"] for run in runs.values()), "Training/protocol inputs differ between stages")
    require(runs["pairs"]["args"]["max_pair_attempts"] == 4 and runs["pairs"]["args"]["pair_batch_size"] == 16
            and _path(sprint, runs["dpo"]["args"]["pairs"]) == sprint / "runs/pairs/pairs.jsonl", "Preference schedule/input changed")
    reference = _one(events["dpo"], "reference_frozen")
    require(reference["before"] == reference["after"] and reference["before"]["parameters"] == 134217728, "DPO reference changed")
    _check_audit(_one(events["dpo"], "exact_reference_load"), sprint / "runs/sft/final/adapter_model.safetensors",
                 PINNED["runs/sft/final/adapter_model.safetensors"], name="reference")
    pair_summary = receipts.json(sprint / "runs/pairs/summary.json")
    _same_event(pair_summary, events["pairs"], "pairs_complete")
    require(pair_summary["pairs_sha256"] == receipts.digest(sprint / "runs/pairs/pairs.jsonl"), "Pair endpoint hash changed")
    # Pin tokenizer/config bytes used for native replay, not merely their paths.
    model_path = sprint / "models/olmo32b"
    download_path = model_path / "download_completed.json"
    download = receipts.json(download_path)
    require(download.get("repo") == MODEL_ID and download.get("revision") == MODEL_REVISION
            and not download.get("metadata_only"), "Pinned base download identity changed")
    metadata_names = set()
    for item in download["verified"]:
        path = (model_path / item["name"]).resolve()
        require(path.is_relative_to(model_path), "Base metadata path escapes model directory")
        if path.suffix != ".safetensors":
            require(receipts.digest(path) == item["sha256"], "Pinned tokenizer/base metadata changed")
            metadata_names.add(item["name"])
    require({"config.json", "tokenizer.json", "tokenizer_config.json"} <= metadata_names, "Tokenizer metadata is not fully pinned")
    for stage in runs:
        identity = _one(events[stage], "model_loaded")["identity"]
        require(identity.get("model_id") == MODEL_ID and identity.get("revision") == MODEL_REVISION
                and identity.get("config_sha256") == receipts.digest(model_path / "config.json")
                and identity.get("download_manifest_sha256") == receipts.digest(download_path), "Stage base/tokenizer identity mismatch")
    preferences = _check_samples(sprint, receipts, runs, events["pairs"], pair_summary, PINNED["runs/sft/final/adapter_model.safetensors"])
    failure = _one(events["grpo"], "actual_grpo_likelihood_audit")
    require(failure["step"] == 0 and failure["tolerance"] == 0.1 and math.isfinite(failure["max_abs_error"])
            and failure["max_abs_error"] > 0.1 and failure["tokens"] == 1805
            and failure["denominator"] == "actual_streamed_sampling_logprobs", "Missing original pre-update likelihood failure")
    require(not any(r.get("kind") in {"gradient", "parameter_update", "stage_complete"} for r in events["grpo"]), "Failed GRPO unexpectedly optimized")
    preserved = _one(events["grpo"], "trainer_preserved_adapter_values")
    require(preserved["before"] == preserved["after"] == summaries["dpo"]["adapter_digest"], "Failed GRPO did not start at exact DPO endpoint")
    receipts.verify_stable()
    # Return metadata only: never include proof text, token IDs or sampled answers.
    return {"status": "validated", "kind": "validated_v3_training_prefix", "completed_stages": ["sft", "pairs", "dpo"],
            "v3_pipeline_status": "failed", "v3_pipeline_success": False,
            "stage_optimizer_steps": {"sft": 80, "dpo": 40, "failed_grpo": 0},
            "checkpoint_files": {str(sprint / "runs" / stage / "final" / name): receipts.sha256[str(sprint / "runs" / stage / "final" / name)]
                                 for stage in ("sft", "dpo") for name in ("adapter_model.safetensors", "adapter_config.json")},
            "endpoints": {stage: {"optimizer_steps": summaries[stage]["optimizer_steps"],
                                  "adapter": summaries[stage]["adapter"], "saved_tensor_audit": summaries[stage]["saved_tensor_audit"]}
                          for stage in ("sft", "dpo")},
            "preferences": preferences, "failed_grpo": {"optimizer_steps": 0, "likelihood_audit": failure},
            "approved_source_changes": changes, "sha256": dict(sorted(receipts.sha256.items())),
            "parsed_heldout_inputs": False, "read_heldout_outcomes": False, "read_protected_textbook_payloads": False}


if __name__ == "__main__":
    print(json.dumps(validate_prefix(), indent=2, sort_keys=True, allow_nan=False))
