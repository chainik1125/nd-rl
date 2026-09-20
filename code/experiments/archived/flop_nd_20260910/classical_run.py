#!/usr/bin/env python3
"""Frozen pure-prompt classical comparisons with hard isolated-worker limits.

Prepare: --data THEOREM_ONLY.jsonl --protocol PROTOCOL.json --prepare-protocol
Run: same data/protocol plus --output NEW_DIRECTORY. Final inputs additionally
require a root release binding the protocol and all comparator endpoints.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import multiprocessing as mp
from multiprocessing.connection import wait
from pathlib import Path
import time

import common
import classical_solver as C


def load_inputs(path):
    rows = common.read_jsonl(path)
    if not rows:
        raise ValueError("Empty theorem-only population")
    seen = set()
    for row in rows:
        if set(row) - {"name", "source_id", "prompt"} or not {"source_id", "prompt"} <= set(row):
            raise ValueError("Classical solver accepts theorem-only rows; labels/audit metadata are forbidden")
        if row["source_id"] in seen:
            raise ValueError("Duplicate theorem identity")
        seen.add(row["source_id"])
    return rows


def is_final(path, rows=None):
    named = "final" in Path(path).stem.lower().replace("-", "_").split("_")
    return named or any(str(row.get("name", "")).startswith("final_") for row in rows or [])


def make_protocol(data, *, timeout_seconds=15.0, methods=None):
    if not 0 < timeout_seconds <= 15:
        raise ValueError("Hard per-attempt timeout must be positive and at most15 seconds")
    selected = [m for m in C.METHODS if methods is None or m["name"] in methods]
    if not selected:
        raise ValueError("No selected comparator methods")
    return {"status": "frozen", "version": "classical-pure-prompt-v1", "data_path": str(Path(data).resolve()),
            "data_sha256": common.sha(data), "methods": selected, "hard_timeout_seconds": timeout_seconds,
            "max_certificate_lines": C.MAX_LINES, "max_certificate_response_tokens": C.MAX_RESPONSE_TOKENS,
            "code_sha256": C.code_identity(), "no_model_or_reference_inputs": True,
            "operation_units_are_not_neural_tokens": True,
            "scope": "Nonlearned bounded ND search plus separate bounded classical truth-assignment constructor; no minimum-length claim"}


def verify_protocol(protocol_path, data, release=None, rows=None):
    protocol = json.loads(Path(protocol_path).read_text())
    expected = make_protocol(data, timeout_seconds=protocol["hard_timeout_seconds"],
                             methods=[m["name"] for m in protocol["methods"]])
    if protocol != expected:
        raise ValueError("Frozen comparator protocol differs from data, code, method, or limit identities")
    if is_final(data, rows):
        if release is None:
            raise ValueError("Final comparator execution requires root's endpoint release")
        record = json.loads(Path(release).read_text())
        if record.get("status") != "released" or record.get("all_comparator_endpoints_frozen") is not True:
            raise ValueError("Final comparator release must freeze all comparator endpoints")
        if record.get("protocol_sha256") != common.sha(protocol_path):
            raise ValueError("Final comparator protocol differs from root release")
    return protocol


def _worker(connection, prompt, method):
    try:
        connection.send(C.solve(prompt, method))
    except BaseException as error:
        connection.send({"method": method["name"], "method_config": method, "status": "worker_exception",
                         "reward": 0, "exception_type": type(error).__name__, "reason": str(error)[:2000]})
    finally:
        connection.close()


def bounded_attempts(tasks, *, workers=4, timeout_seconds=15.0, worker=_worker):
    """Fork only from this single-threaded parent; forcibly terminate timed-out children."""
    if not 1 <= workers <= 4 or not 0 < timeout_seconds <= 15:
        raise ValueError("At most4 CPU workers and15 seconds per attempt are allowed")
    context = mp.get_context("fork")
    pending = iter(tasks)
    active = {}
    exhausted = False
    try:
        while active or not exhausted:
            while len(active) < workers and not exhausted:
                try:
                    task = next(pending)
                except StopIteration:
                    exhausted = True
                    break
                receiver, sender = context.Pipe(duplex=False)
                process = context.Process(target=worker, args=(sender, task["prompt"], task["method"]))
                start = time.monotonic()
                process.start()
                sender.close()
                active[receiver] = (process, task, start)
            if not active:
                break
            for receiver in wait(list(active), timeout=.02):
                process, task, start = active.pop(receiver)
                try:
                    result = receiver.recv()
                except EOFError:
                    result = {"method": task["method"]["name"], "status": "worker_exited_without_result", "reward": 0}
                receiver.close()
                process.join(timeout=.1)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=.2)
                if process.is_alive():
                    process.kill()
                    process.join()
                if time.monotonic() - start > timeout_seconds:
                    result = {"method": task["method"]["name"], "method_config": task["method"],
                              "status": "hard_timeout", "reward": 0, "operations": None,
                              "operation_budget": task["method"]["budget"],
                              "note": "Result arrived after hard wall deadline and was censored"}
                yield task, {**result, "worker_wall_seconds": time.monotonic() - start,
                             "hard_timeout_seconds": timeout_seconds, "hard_timeout": result["status"] == "hard_timeout"}
            now = time.monotonic()
            for receiver, (process, task, start) in list(active.items()):
                if now - start < timeout_seconds:
                    continue
                process.terminate()
                process.join(timeout=.1)
                if process.is_alive():
                    process.kill()
                    process.join()
                receiver.close()
                del active[receiver]
                yield task, {"method": task["method"]["name"], "method_config": task["method"],
                             "status": "hard_timeout", "reward": 0, "hard_timeout": True,
                             "hard_timeout_seconds": timeout_seconds, "worker_wall_seconds": time.monotonic() - start,
                             "operation_budget": task["method"]["budget"], "operations": None,
                             "note": "Timed-out worker was terminated; no partial certificate counted"}
    finally:
        for receiver, (process, _, _) in active.items():
            if process.is_alive():
                process.terminate()
                process.join(timeout=.1)
            if process.is_alive():
                process.kill()
                process.join()
            receiver.close()


def aggregate(attempts, methods):
    result = {}
    for method in methods:
        rows = [r for r in attempts if r["method"] == method["name"]]
        successes = [r for r in rows if r["reward"]]
        result[method["name"]] = {"attempts": len(rows), "verified_successes": len(successes),
            "status_counts": dict(Counter(r["status"] for r in rows)),
            "raw_line_histogram": dict(Counter(r["raw_lines"] for r in successes)),
            "pruned_line_histogram": dict(Counter(r["pruned_lines"] for r in successes)),
            "raw_budget_eligible_successes": {str(b): sum(r["raw_budget_eligibility"][str(b)] for r in successes) for b in C.BUDGETS},
            "pruned_budget_eligible_successes": {str(b): sum(r["pruned_budget_eligibility"][str(b)] for r in successes) for b in C.BUDGETS},
            "total_reported_operations": sum(r.get("operations") or 0 for r in rows),
            "unknown_operations_due_to_worker_limits": sum(r.get("operations") is None for r in rows),
            "total_worker_wall_seconds": sum(r["worker_wall_seconds"] for r in rows),
            "max_worker_wall_seconds": max((r["worker_wall_seconds"] for r in rows), default=0)}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--prepare-protocol", action="store_true")
    p.add_argument("--methods", nargs="+", choices=[m["name"] for m in C.METHODS])
    p.add_argument("--timeout-seconds", type=float, default=15.0)
    p.add_argument("--output", type=Path)
    p.add_argument("--release", type=Path)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if args.prepare_protocol:
        common.write_json(args.protocol, make_protocol(args.data, timeout_seconds=args.timeout_seconds, methods=args.methods))
        print(json.dumps({"protocol": str(args.protocol), "sha256": common.sha(args.protocol)}))
        return
    if args.output is None:
        p.error("Execution requires --output")
    if not 1 <= args.workers <= 4:
        p.error("--workers must lie in1..4")
    protocol = verify_protocol(args.protocol, args.data, args.release)
    rows = load_inputs(args.data)
    protocol = verify_protocol(args.protocol, args.data, args.release, rows)
    args.output.mkdir(parents=True, exist_ok=False)
    common.write_json(args.output / "config.json", {"protocol": protocol, "protocol_sha256": common.sha(args.protocol),
                      "workers": args.workers, "limits_include_serialization_verification_compression": True,
                      "operation_counts_exclude_parser_serialization_verification_compression": True})
    tasks = [{**row, "row_index": i, "method": method} for i, row in enumerate(rows) for method in protocol["methods"]]
    started = time.monotonic()
    attempts = []
    with (args.output / "attempts.jsonl").open("w", buffering=1) as log, (args.output / "certificates_NEVER_TRAIN.jsonl").open("w", buffering=1) as certificates:
        for task, result in bounded_attempts(tasks, workers=args.workers, timeout_seconds=protocol["hard_timeout_seconds"]):
            record = {"source_id": task["source_id"], "prompt": task["prompt"], "name": task.get("name", task["source_id"]),
                      "row_index": task["row_index"], **result}
            if record["reward"]:
                certificates.write(json.dumps({key: record[key] for key in ("source_id", "prompt", "name", "row_index", "method",
                    "raw_certificate", "pruned_certificate", "raw_certificate_sha256", "pruned_certificate_sha256")}) + "\n")
                record.pop("raw_certificate")
                record.pop("pruned_certificate")
            log.write(json.dumps(record) + "\n")
            attempts.append(record)
            if len(attempts) == 1 or len(attempts) % 32 == 0 or len(attempts) == len(tasks):
                progress = {"attempts_done": len(attempts), "attempts_total": len(tasks),
                            "verified_successes": sum(r["reward"] for r in attempts), "elapsed_seconds": time.monotonic() - started}
                common.write_json(args.output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    result = {"status": "complete", "theorems": len(rows), "attempts": len(attempts),
              "metrics": aggregate(attempts, protocol["methods"]), "elapsed_seconds": time.monotonic() - started,
              "protocol_sha256": common.sha(args.protocol), "data_sha256": common.sha(args.data),
              "attempts_sha256": common.sha(args.output / "attempts.jsonl"),
              "certificates_sha256": common.sha(args.output / "certificates_NEVER_TRAIN.jsonl"),
              "no_model_or_reference_inputs": True, "owned_worker_processes_remaining": 0,
              "interpretation": "Nonlearned classical ND construction; restricted structured-fragment bounds and unrestricted minimum lengths are not inferred."}
    common.write_json(args.output / "result.json", result)
    print(json.dumps({"complete": True, "theorems": len(rows), "attempts": len(attempts), "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
