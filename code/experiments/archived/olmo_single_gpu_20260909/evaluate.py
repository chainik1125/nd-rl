"""Resumable native-token evaluation with identical budgets across OLMo arms."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

from protocol import ROOT, SPRINT, judge, protocol_identity, render_prompt


MODEL_ID = "allenai/Olmo-3-1125-32B"
MODEL_REVISION = "c2b61dae89a1ad10e4ad5653d0e46b590902607b"
FINAL_NOT_BEFORE = datetime(2026, 9, 10, 5, 14, tzinfo=timezone.utc)
ARMS = ("base_zero", "base_icl32", "sft", "dpo", "rl", "sft_icl32")
SYNTHETIC_FILES = {
    "short_panel": "eval_short_panel.jsonl", "long_panel": "eval_long_panel.jsonl",
    "short_full": "eval_short_prompts.jsonl", "long_full": "eval_long_prompts.jsonl",
}


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def stable_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_rows(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_resume_rows(path: Path) -> list[dict]:
    """Recover only a trailing interrupted append, preserving its exact bytes."""
    rows = []
    offset = 0
    with path.open("rb") as stream:
        lines = stream.readlines()
    for index, line in enumerate(lines):
        try:
            row = json.loads(line) if line.strip() else None
        except (json.JSONDecodeError, UnicodeDecodeError):
            if index != len(lines) - 1 or line.endswith(b"\n"):
                raise ValueError("Malformed committed attempt record; automatic recovery is forbidden") from None
            identity = hashlib.sha256(line).hexdigest()
            backup = path.with_name(f"{path.name}.uncommitted_tail_{identity}.bin")
            if not backup.exists():
                with backup.open("xb") as stream:
                    stream.write(line)
            elif backup.read_bytes() != line:
                raise ValueError("Interrupted-tail backup identity mismatch")
            with path.open("r+b") as stream:
                stream.truncate(offset)
            print(json.dumps({"recovered_interrupted_append": str(path), "preserved_tail": str(backup), "bytes": len(line)}), flush=True)
            break
        if row is not None:
            rows.append(row)
        if index == len(lines) - 1 and not line.endswith(b"\n"):
            with path.open("ab") as stream:
                stream.write(b"\n")
            print(json.dumps({"completed_interrupted_line_separator": str(path)}), flush=True)
        offset += len(line)
    return rows


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def check_textbook_release(release: dict, arm: str, cohort: str, adapter_path: Path | None, now=None) -> Path:
    """Validate opaque release metadata before the first textbook payload read."""
    if cohort not in ("textbook_dev", "textbook_final"):
        raise ValueError("Unexpected protected cohort")
    final = cohort == "textbook_final"
    kind = "textbook_final_release" if final else "textbook_development_release"
    if release.get("kind") != kind or arm not in release.get("allowed_arms", []):
        raise ValueError("Release does not authorize this cohort and arm")
    if release.get("settings") != {"max_new_tokens": 2048, "sampled_k": 4, "temperature": 1.0, "top_p": 1.0, "top_k": 0}:
        raise ValueError("Release has different evaluation settings")
    if final:
        declared = datetime.fromisoformat(release["not_before_utc"].replace("Z", "+00:00"))
        if declared.tzinfo is None or declared < FINAL_NOT_BEFORE:
            raise ValueError("Final release violates the sprint final-access floor")
        if (now or datetime.now(timezone.utc)) < declared:
            raise ValueError("Final evaluation window has not opened")
        if release.get("all_checkpoints_and_evaluation_settings_frozen") is not True:
            raise ValueError("Final freeze declaration is absent")
    checkpoint_files = release.get("checkpoint_files", {})
    completed = release.get("completed_training", [])
    if not checkpoint_files or not completed:
        raise ValueError("Release must pin completed training and frozen checkpoints")
    for name, expected in checkpoint_files.items():
        path = Path(name).resolve()
        if not path.is_file() or sha(path) != expected:
            raise ValueError("A released checkpoint file changed or is missing")
    if adapter_path is not None:
        adapter_files = [path for path in adapter_path.resolve().glob("adapter*") if path.is_file()]
        if not adapter_files or any(str(path) not in checkpoint_files for path in adapter_files):
            raise ValueError("This adapter is not fully pinned in the textbook release")
    for record in completed:
        path = Path(record["path"]).resolve()
        if sha(path) != record["sha256"]:
            raise ValueError("A completed-training record changed")
        metadata = json.loads(path.read_text())
        if metadata.get("status") not in ("complete", "completed"):
            raise ValueError("Training is not completed")
    expected_path = ROOT / "r1-cot-20260909/data" / ("sealed_final/textbook_final.jsonl" if final else "textbook_dev.jsonl")
    path = Path(release["data_path"]).resolve()
    if path != expected_path.resolve() or not isinstance(release.get("data_sha256"), str):
        raise ValueError("Release points outside the original frozen textbook split")
    return path


def load_targets(args) -> tuple[list[dict], dict]:
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    targets, inputs = [], {str(args.data_dir / "manifest.json"): sha(args.data_dir / "manifest.json")}
    for cohort in args.cohorts:
        if cohort in SYNTHETIC_FILES:
            name = SYNTHETIC_FILES[cohort]
            path = args.data_dir / name
            identity = sha(path)
            if identity != manifest["output_sha256"][name]:
                raise ValueError("Frozen synthetic evaluation file changed")
        else:
            if args.release is None:
                raise ValueError("Textbook evaluation requires a completed-training release")
            release = json.loads(args.release.read_text())
            path = check_textbook_release(release, args.arm, cohort, args.adapter_path)
            # This is the first payload access, after every release check passed.
            identity = sha(path)
            if identity != release["data_sha256"]:
                raise ValueError("Released textbook payload identity changed")
            inputs[str(args.release)] = sha(args.release)
        rows = read_rows(path)
        if cohort.startswith("textbook_") and len(rows) != (58 if cohort == "textbook_dev" else 73):
            raise ValueError("Frozen textbook denominator changed")
        for row in rows:
            if any(field in row for field in ("proof", "text", "reference_proof", "answer")):
                raise ValueError("Evaluation input unexpectedly contains proof labels")
            targets.append(dict(row, cohort=cohort))
        inputs[str(path)] = identity
    keys = [(row["cohort"], row["canonical_hash"]) for row in targets]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate evaluation theorem within a cohort")
    return targets, inputs


def attempt_key(target: dict, mode: str, draw: int) -> str:
    return f"{target['cohort']}:{target['canonical_hash']}:{mode}:{draw}"


def planned_batches(targets: list[dict], modes: list[str], batch_size: int, seed: int) -> tuple[dict, list[dict]]:
    """Freeze batch membership and RNG streams independently of arm/resume."""
    planned, batches = {}, []
    for mode in modes:
        keys = []
        for target in targets:
            for draw in range(1 if mode == "greedy" else 4):
                key = attempt_key(target, mode, draw)
                planned[key] = (target, mode, draw)
                keys.append(key)
        for offset in range(0, len(keys), batch_size):
            members = keys[offset:offset + batch_size]
            batches.append({"batch_id": stable_hash(members), "mode": mode, "attempt_keys": members,
                "seed": int(stable_hash([seed, members])[:8], 16) & 0x7FFFFFFF})
    return planned, batches


def validate_resume(records: list[dict], planned: dict, plan_hash: str) -> set[str]:
    seen = set()
    for record in records:
        key = record["attempt_key"]
        if key not in planned or key in seen or record.get("plan_sha256") != plan_hash:
            raise ValueError("Unplanned, duplicated, or differently configured resumed attempt")
        target, mode, draw = planned[key]
        if record["prompt"] != target["prompt"] or record["mode"] != mode or record["draw"] != draw:
            raise ValueError("Resumed attempt target changed")
        seen.add(key)
    return seen


def token_accounting(tokenizer, generation: dict, verdict: dict) -> dict:
    """Exact native budget totals; decoded-region counts are labeled estimates."""
    response = generation["response_ids"]
    eos = tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
    text = generation["text"]
    opening = text.find("<FINAL>")
    closing = text.find("</FINAL>")
    scratch = text[:opening] if opening >= 0 else text
    final = text[opening:closing + len("</FINAL>")] if opening >= 0 and closing > opening else ""
    suffix = text[closing + len("</FINAL>"):] if final else ""
    count = lambda value: len(tokenizer.encode(value, add_special_tokens=False))
    return {
        "prefill_native_tokens": int(generation["prompt_tokens"]),
        "generated_native_tokens": len(response),
        "eos_native_tokens": sum(token in eos_ids for token in response),
        "failed_attempt_native_tokens": 0 if verdict["accepted"] else len(response),
        "prompt_padding_positions": int(generation.get("prompt_padding_positions", 0)),
        "post_stop_padding_positions": int(generation.get("post_stop_padding_positions", 0)),
        "padded_prefill_positions": int(generation["prompt_tokens"]) + int(generation.get("prompt_padding_positions", 0)),
        "padded_generation_positions": len(response) + int(generation.get("post_stop_padding_positions", 0)),
        "scratch_retokenized_tokens": count(scratch),
        "final_retokenized_tokens": count(final),
        "suffix_retokenized_tokens": count(suffix),
        "region_count_semantics": "independently_retokenized_decoded_substrings_not_native_token_partition",
    }


def run(args) -> dict:
    args.data_dir = args.data_dir.resolve()
    args.out = args.out.resolve()
    args.model_path = args.model_path.resolve()
    if args.adapter_path is not None:
        args.adapter_path = args.adapter_path.resolve()
    if args.arm in ("base_zero", "base_icl32") and args.adapter_path is not None:
        raise ValueError("Base arms cannot load an adapter")
    if args.arm not in ("base_zero", "base_icl32") and args.adapter_path is None:
        raise ValueError("A trained arm requires its frozen adapter")
    targets, inputs = load_targets(args)
    shared_path = args.data_dir / "shared_cap6.jsonl"
    shared = read_rows(shared_path)
    data_manifest = json.loads((args.data_dir / "manifest.json").read_text())
    if len(shared) != 32 or sha(shared_path) != data_manifest["output_sha256"][shared_path.name]:
        raise ValueError("The frozen matched 32-example bank changed")
    demonstrations = shared if args.arm in ("base_icl32", "sft_icl32") else []
    inputs[str(shared_path)] = sha(shared_path)
    from modeling import load_model, load_tokenizer, memory_snapshot, sample_completions
    tokenizer = load_tokenizer(str(args.model_path))
    rendered = {row["canonical_hash"]: render_prompt(row["prompt"], demonstrations) for row in targets}
    max_context = int(getattr(tokenizer, "model_max_length", 65536))
    model_config = json.loads((args.model_path / "config.json").read_text())
    max_context = min(max_context, int(model_config.get("max_position_embeddings", 65536)))
    token_lengths = {key: len(tokenizer.encode(value, add_special_tokens=False)) for key, value in rendered.items()}
    if any(length + args.max_new_tokens > max_context for length in token_lengths.values()):
        raise ValueError("The complete ICL bank and decode budget do not fit; truncation is forbidden")
    adapter_files = {} if args.adapter_path is None else {str(path): sha(path) for path in sorted(args.adapter_path.glob("adapter*")) if path.is_file()}
    planned, batches = planned_batches(targets, args.modes, args.batch_size, args.seed)
    plan = {
        "version": 2, "arm": args.arm, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "model_path": str(args.model_path), "model_config_sha256": sha(args.model_path / "config.json"),
        "download_manifest_sha256": sha(args.model_path / "download_completed.json"),
        "adapter_files": adapter_files, "input_files": inputs, "protocol": protocol_identity(),
        "tokenizer_identity": {"class": type(tokenizer).__name__, "vocab_size": len(tokenizer), "eos_token_id": tokenizer.eos_token_id},
        "tokenizer_files": {str(path): sha(path) for path in sorted(args.model_path.iterdir()) if path.is_file() and (path.name.startswith("tokenizer") or path.name == "special_tokens_map.json")},
        "settings": {"max_new_tokens": args.max_new_tokens, "sampled_k": 4, "sampled_temperature": 1.0, "top_p": 1.0, "top_k": 0, "stop_on_actually_generated_final_marker": True, "native_eos_fallback": True, "inserted_eos": False, "seed": args.seed, "modes": args.modes, "batch_size": args.batch_size, "rng_unit": "frozen_ordered_batch_not_independent_per_target"},
        "batches": batches,
        "cohorts": args.cohorts, "target_count": len(targets), "target_identity": stable_hash(targets),
        "demonstrations_per_query": len(demonstrations), "prompt_tokens_min": min(token_lengths.values()),
        "prompt_tokens_max": max(token_lengths.values()), "checkpoint_selection": "fixed_completed_endpoint_no_evaluation_driven_selection",
        "modeling_sha256": sha(Path(__file__).with_name("modeling.py")), "evaluator_sha256": sha(Path(__file__)),
    }
    plan_hash = stable_hash(plan)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "evaluation_plan.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != plan:
            raise ValueError("Cannot resume after changing evaluation plan, code, data, or checkpoint")
    else:
        if any(args.out.iterdir()):
            raise ValueError("New evaluation output directory must be empty")
        manifest_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        with (args.out / "targets.jsonl").open("x") as stream:
            for row in targets:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
    attempts_path = args.out / "attempts.jsonl"
    old = read_resume_rows(attempts_path) if attempts_path.exists() else []
    seen = validate_resume(old, planned, plan_hash)
    old_by_key = {record["attempt_key"]: record for record in old}
    ledger_path = args.out / "compute_batches.jsonl"
    if ledger_path.exists():
        read_resume_rows(ledger_path)
    if len(seen) == len(planned):
        return {"status": "complete", "attempts": len(seen), "out": str(args.out)}
    model = load_model(str(args.model_path), adapter_path=str(args.adapter_path) if args.adapter_path else None, trainable=False)
    start = time.monotonic()
    generated_this_invocation = 0
    with attempts_path.open("a", buffering=1) as stream, ledger_path.open("a", buffering=1) as ledger:
        for batch in batches:
            keys = batch["attempt_keys"]
            if all(key in seen for key in keys):
                continue
            if args.max_attempts is not None and generated_this_invocation + len(keys) > args.max_attempts:
                break
            mode, seed = batch["mode"], batch["seed"]
            call_id = uuid.uuid4().hex
            ledger.write(json.dumps({"phase": "started", "call_id": call_id, "plan_sha256": plan_hash, "batch": batch,
                "already_committed_attempts": [key for key in keys if key in seen], "started_utc": utcnow()}, sort_keys=True) + "\n")
            before = time.monotonic()
            generations = sample_completions(model, tokenizer, [rendered[planned[key][0]["canonical_hash"]] for key in keys], max_new_tokens=args.max_new_tokens, temperature=0.0 if mode == "greedy" else 1.0, batch_size=args.batch_size, seed=seed)
            elapsed = time.monotonic() - before
            if len(generations) != len(keys):
                raise ValueError("Batched sampler changed the number of planned attempts")
            # Charge and preserve every actually generated output before any
            # individual attempt append. A restarted partial batch is fully
            # regenerated at the same seed; only missing attempts are appended.
            outputs = [{"attempt_key": key, "response_ids": generation["response_ids"],
                "prompt_tokens": generation["prompt_tokens"], "finish_reason": generation["finish_reason"],
                "post_stop_padding_positions": generation.get("post_stop_padding_positions", 0),
                "prompt_padding_positions": generation.get("prompt_padding_positions", 0),
                "batch_generation_steps": generation.get("batch_generation_steps", len(generation["response_ids"]))}
                for key, generation in zip(keys, generations, strict=True)]
            ledger.write(json.dumps({"phase": "completed", "call_id": call_id, "plan_sha256": plan_hash,
                "batch": batch, "native_outputs": outputs, "wall_seconds": elapsed, "completed_utc": utcnow()}, sort_keys=True) + "\n")
            generated_this_invocation += len(keys)
            for index, (key, generation) in enumerate(zip(keys, generations, strict=True)):
                target, _, draw = planned[key]
                if key in seen:
                    previous = old_by_key[key]["generation"]
                    if generation["response_ids"] != previous["response_ids"] or generation["prompt_ids"] != previous["prompt_ids"]:
                        raise ValueError("Regenerated partial batch differs from its committed seeded outputs")
                    continue
                verdict = judge(target["prompt"], generation["text"])
                record = {
                    "attempt_key": key, "plan_sha256": plan_hash, "arm": args.arm, "cohort": target["cohort"],
                    "name": target["name"], "canonical_hash": target["canonical_hash"], "prompt": target["prompt"],
                    "family": target.get("family", "textbook"), "reference_lines": target["reference_lines"],
                    "mode": mode, "draw": draw, "sampling_batch_seed": seed, "batch_id": batch["batch_id"],
                    "batch_index": index, "compute_call_id": call_id, "generation": generation, "judge": verdict,
                    "token_accounting": token_accounting(tokenizer, generation, verdict),
                    "allocated_batch_wall_seconds": elapsed / len(keys), "completed_utc": utcnow(),
                }
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                old_by_key[key] = record
                seen.add(key)
            print(json.dumps({"arm": args.arm, "mode": mode, "completed": len(seen), "planned": len(planned), "batch_wall_seconds": round(elapsed, 2), "elapsed_seconds": round(time.monotonic() - start, 2)}), flush=True)
    result = {"status": "complete" if len(seen) == len(planned) else "partial", "attempts": len(seen), "planned_attempts": len(planned), "out": str(args.out), "model_identity": getattr(model, "_sprint_identity", {}), "memory": memory_snapshot(), "updated_utc": utcnow()}
    (args.out / "evaluation_status.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--model-path", type=Path, default=SPRINT / "models/olmo32b")
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--data-dir", type=Path, default=SPRINT / "data")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cohorts", nargs="+", choices=[*SYNTHETIC_FILES, "textbook_dev", "textbook_final"], default=["short_panel", "long_panel"])
    parser.add_argument("--modes", nargs="+", choices=["greedy", "sampled"], default=["greedy", "sampled"])
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=202609092015)
    parser.add_argument("--max-attempts", type=int, help="Budget actual generated positions in whole frozen batches, including partial-resume regeneration")
    parser.add_argument("--release", type=Path)
    args = parser.parse_args()
    if args.max_new_tokens != 2048:
        parser.error("The matched primary protocol fixes max_new_tokens=2048")
    if args.batch_size < 1 or (args.max_attempts is not None and (args.max_attempts < args.batch_size or args.max_attempts % args.batch_size)):
        parser.error("Use a positive batch size; --max-attempts must be a positive multiple of it")
    if len(set(args.cohorts)) != len(args.cohorts) or len(set(args.modes)) != len(args.modes):
        parser.error("Duplicate cohorts or modes")
    print(json.dumps(run(args), sort_keys=True))


if __name__ == "__main__":
    main()
