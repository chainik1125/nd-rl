"""Registered continuation interventions for a frozen live policy.

Generate every public transition before checking any proof. The candidate set
is retained exactly. Index choices are integrated, not sampled. This estimates
one extra attempt followed by acceptance or forced continuation then the policy.
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

os.environ["ND_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[5]
ALLOCATION = ROOT / "code/experiments/current/learned_compute_allocation_20261006/allocation"
sys.path.insert(0, str(ROOT))
from code.shared_code.evaluation_freeze import validate_frozen_evaluation  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def intervention_signature(args):
    return {
        "method": "matched_live_continuation_intervention",
        "base_sha256": digest(args.base),
        "checkpoint_sha256": digest(args.checkpoint) if args.checkpoint else None,
        "public_sha256": digest(args.public),
        "protocol": {
            key: getattr(args, key)
            for key in [
                "limit",
                "replicates",
                "continuation",
                "forced_extra",
                "history_removal_limit",
                "counts",
                "seed",
                "context",
                "attempt_bound",
                "state_reserve",
                "max_attempts",
                "max_action",
                "max_steps",
                "temperature",
            ]
        },
        "precision": "FP32 generation and controller",
        "implementation_sha256": digest(__file__),
        "analysis_sha256": digest(Path(__file__).with_name("analyze_causal.py")),
        "runner_hashes": {
            name: digest(ALLOCATION / name)
            for name in ["live.py", "controller.py", "train_controller.py"]
        },
        "source_hashes": {
            name: digest(Path(args.source) / name)
            for name in [
                "model.py",
                "best_model.py",
                "sample.py",
                "state_env.py",
                "state_sample.py",
                "lean_tok.py",
                "tokenizer.py",
                "lean_gate.py",
                "nd2lean.py",
            ]
        },
    }


def stream_record(stream, record):
    stream.write(json.dumps(record) + "\n")
    stream.flush()


def keyed_seed(seed, *parts):
    text = json.dumps([seed, *parts])
    return int(hashlib.sha256(text.encode()).hexdigest()[:15], 16)


def future_stop(*, hard_limit, continuation, forced_extra, extra, accept_probability, stop_uniform):
    """Hard limits take precedence over the requested continuation intervention."""
    if continuation not in {"one_extra", "policy"} or min(forced_extra, extra) < 1:
        raise ValueError("Invalid continuation protocol")
    if not 0 <= accept_probability <= 1 or not 0 <= stop_uniform < 1:
        raise ValueError("Invalid Bernoulli probability or uniform draw")
    forced_continuation = extra < forced_extra and not hard_limit
    accepted = hard_limit or (
        not forced_continuation
        and (continuation == "one_extra" or stop_uniform < accept_probability)
    )
    return accepted, forced_continuation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--public", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--continuation", choices=["one_extra", "policy"], default="one_extra")
    parser.add_argument("--forced-extra", type=int, default=1)
    parser.add_argument("--history-removal-limit", type=int, default=16)
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--seed", type=int, default=71006)
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--attempt-bound", type=int, default=1024)
    parser.add_argument("--state-reserve", type=int, default=1024)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--max-action", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--frozen-test-protocol")
    parser.add_argument("--signature-only", action="store_true")
    args = parser.parse_args()
    signature = intervention_signature(args)
    if args.signature_only:
        print(json.dumps(signature, sort_keys=True))
        return
    rows = [json.loads(line) for line in Path(args.public).read_text().splitlines()][: args.limit]
    if not rows or any(row["split"] != args.split for row in rows):
        raise ValueError("Intervention panel split differs from the requested split")
    if args.split == "test":
        if not args.frozen_test_protocol or not args.checkpoint:
            raise ValueError(
                "Test interventions require a registered frozen checkpoint and protocol"
            )
        validate_frozen_evaluation(
            json.loads(Path(args.frozen_test_protocol).read_text()), args.public, signature
        )
        if len(rows) != args.limit:
            raise ValueError("Registered test intervention panel is incomplete")
    if args.replicates < 1 or min(args.counts) < 1 or max(args.counts) >= args.max_attempts:
        raise ValueError("Need positive branches and states before the attempt limit")
    if args.forced_extra < 1 or (args.continuation == "one_extra" and args.forced_extra != 1):
        raise ValueError("One-extra mode requires exactly one forced attempt")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    snapshot = out / "runner_snapshot"
    snapshot.mkdir()
    for name in ["live.py", "controller.py", "train_controller.py"]:
        shutil.copy2(ALLOCATION / name, snapshot / name)
    shutil.copy2(__file__, snapshot / "causal_probe.py")
    # Import the actual copied sources, making provenance insensitive to later edits.
    sys.path.insert(0, str(Path(args.source).resolve()))
    sys.path.insert(0, str(snapshot.resolve()))
    from controller import Controller, encode_history
    from lean_gate import lean_check
    from live import PROTOCOL_VERSION, generate_attempt
    from model import load_ckpt
    from train_controller import batch_states

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    model, tok, _ = load_ckpt(args.base, "cuda")
    controller = Controller(model, args.max_attempts).cuda()
    training_config = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cuda")
        training_config = checkpoint["config"]
        if training_config["base_sha256"] != digest(args.base):
            raise ValueError("Checkpoint initialization differs from the supplied base")
        if training_config["cost"] != 0:
            raise ValueError("Primary continuation probe requires a zero-cost checkpoint")
        for key in [
            "context",
            "attempt_bound",
            "state_reserve",
            "max_attempts",
            "max_action",
            "max_steps",
            "temperature",
        ]:
            if getattr(args, key) != training_config[key]:
                raise ValueError(f"Probe differs from checkpoint protocol: {key}")
        if training_config.get("live_protocol_version") != PROTOCOL_VERSION:
            raise ValueError("Probe uses a different live generation protocol")
        for name in ["live.py", "controller.py", "train_controller.py"]:
            if training_config["runner_hashes"][name] != digest(snapshot / name):
                raise ValueError(f"Probe helper differs from training: {name}")
        for name, recorded_hash in training_config["source_hashes"].items():
            if recorded_hash != digest(Path(args.source) / name):
                raise ValueError(f"Probe upstream source differs from training: {name}")
        controller.load_state_dict(checkpoint["state"], strict=True)
    controller.eval()
    config = vars(args) | {
        "started_utc": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        "base_sha256": digest(args.base),
        "public_sha256": digest(args.public),
        "checkpoint_sha256": digest(args.checkpoint) if args.checkpoint else None,
        "runner_hashes": {p.name: digest(p) for p in snapshot.glob("*.py")},
        "upstream_hashes": {
            name: digest(Path(args.source) / name)
            for name in [
                "model.py",
                "best_model.py",
                "sample.py",
                "state_env.py",
                "state_sample.py",
                "lean_tok.py",
                "tokenizer.py",
                "lean_gate.py",
                "nd2lean.py",
            ]
        },
        "theorem_ids": [row["theorem_id"] for row in rows],
        "training_config": training_config,
        "registered_signature": signature,
        "frozen_test_protocol_sha256": digest(args.frozen_test_protocol)
        if args.frozen_test_protocol
        else None,
        "precision": "FP32 generation and controller; inherited live sampler",
        "estimand": (
            "force one more attempt then force ACCEPT, integrate index probabilities"
            if args.continuation == "one_extra"
            else (
                f"force {args.forced_extra} extra attempts then follow stop policy; "
                "integrate final index"
            )
        ),
        "state_panel": (
            "force collection through maximum requested count, even if policy would stop"
        ),
        "history_intervention": (
            "memoryless generator at every future step: remove all prior generator history; "
            "preserve the selectable pool and original theorem for the chooser"
        ),
        "history_removal_theorem_ids": [
            r["theorem_id"] for r in rows[: args.history_removal_limit]
        ],
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    @torch.no_grad()
    def decide(row, candidates):
        ids = encode_history(
            tok,
            tok.encode_prompt(row["prompt"]),
            [c["transition_tokens"] for c in candidates],
            args.context,
        )
        accept, index = batch_states(controller, tok, [(ids, len(candidates))], "cuda")
        forced = (
            len(candidates) == args.max_attempts
            or len(ids) + args.attempt_bound + args.state_reserve + 3 > args.context
        )
        return {
            "accept_probability": float(accept.probs.item()),
            "selection_probabilities": index.probs[0, : len(candidates)].tolist(),
            "controller_tokens": len(ids),
            "forced": forced,
        }

    def generate(row, previous, seed):
        candidate, private, actions = generate_attempt(
            controller,
            tok,
            row["prompt"],
            previous,
            random.Random(seed),
            context_limit=args.context,
            attempt_bound=args.attempt_bound,
            max_action=args.max_action,
            max_steps=args.max_steps,
            temperature=args.temperature,
        )
        candidate["generation_action_seeds"] = [action["seed"] for action in actions]
        return candidate, private

    private_records = []
    states = []
    branches = []
    with (
        (out / "attempts_public.jsonl").open("w") as attempts_stream,
        (out / "states_public.jsonl").open("w") as states_stream,
        (out / "branches_public.jsonl").open("w") as branches_stream,
        (out / "attempts_private_unjudged.jsonl").open("w") as private_stream,
    ):
        for row_index, row in enumerate(rows):
            tid = row["theorem_id"]
            candidates, incumbent_ids = [], []
            for n in range(1, max(args.counts) + 1):
                candidate, private = generate(
                    row, candidates, keyed_seed(args.seed, tid, "incumbent", n)
                )
                uid = f"{tid}/incumbent/{n}"
                candidates.append(candidate)
                incumbent_ids.append(uid)
                stream_record(
                    attempts_stream, {"attempt_uid": uid, "theorem_id": tid, "candidate": candidate}
                )
                record = {"attempt_uid": uid, "theorem_id": tid, "prompt": row["prompt"], **private}
                private_records.append(record)
                stream_record(private_stream, record)
                decision = decide(row, candidates)
                if n not in args.counts:
                    if decision["forced"]:
                        break
                    continue
                state_id = f"{tid}/state/{n}"
                state = {
                    "state_id": state_id,
                    "theorem_id": tid,
                    "family": row.get("family", "unknown"),
                    "count": n,
                    "incumbent_ids": list(incumbent_ids),
                    **decision,
                }
                states.append(state)
                stream_record(states_stream, state)
                if decision["forced"]:
                    break
                modes = ["actual_history"]
                if row_index < args.history_removal_limit:
                    modes.append("history_removed")
                for repeat in range(args.replicates):
                    seed = keyed_seed(args.seed, tid, "continuation", n, repeat)
                    for mode in modes:
                        retained = list(candidates)
                        futures, future_ids, decision_path = [], [], []
                        stop_rng = random.Random(keyed_seed(seed, "stop_decisions"))
                        for extra in range(1, args.max_attempts - n + 1):
                            previous = retained if mode == "actual_history" else []
                            generation_seed = (
                                seed if extra == 1 else keyed_seed(seed, "extra", extra)
                            )
                            future, private = generate(row, previous, generation_seed)
                            # Generator interventions cannot change selectable identities.
                            future["candidate_id"] = len(retained)
                            uid = f"{state_id}/{mode}/{repeat}/extra{extra}"
                            retained.append(future)
                            futures.append(future)
                            future_ids.append(uid)
                            stream_record(
                                attempts_stream,
                                {"attempt_uid": uid, "theorem_id": tid, "candidate": future},
                            )
                            record = {
                                "attempt_uid": uid,
                                "theorem_id": tid,
                                "prompt": row["prompt"],
                                **private,
                            }
                            private_records.append(record)
                            stream_record(private_stream, record)
                            after = decide(row, retained)
                            stop_uniform = stop_rng.random()
                            accepted, forced_continuation = future_stop(
                                hard_limit=after["forced"],
                                continuation=args.continuation,
                                forced_extra=args.forced_extra,
                                extra=extra,
                                accept_probability=after["accept_probability"],
                                stop_uniform=stop_uniform,
                            )
                            decision_path.append(
                                {
                                    "extra_attempts": extra,
                                    "generation_seed": generation_seed,
                                    "stop_uniform": stop_uniform,
                                    "accepted": accepted,
                                    "forced_continuation": forced_continuation,
                                    **after,
                                }
                            )
                            if accepted:
                                break
                        branch = {
                            "state_id": state_id,
                            "theorem_id": tid,
                            "count": n,
                            "mode": mode,
                            "repeat": repeat,
                            "seed": seed,
                            "future_id": future_ids[0],
                            "future_ids": future_ids,
                            "continuation": args.continuation,
                            "forced_extra_requested": args.forced_extra,
                            "decision_path": decision_path,
                            "after": after,
                            "generated_tokens": sum(f["generated_tokens"] for f in futures),
                            "generator_forward_tokens": sum(f["forward_tokens"] for f in futures),
                        }
                        branches.append(branch)
                        stream_record(branches_stream, branch)
            print(
                json.dumps(
                    {
                        "theorems_complete": row_index + 1,
                        "theorems": len(rows),
                        "attempts_saved": len(private_records),
                    }
                ),
                flush=True,
            )

    # Verification is a distinct final phase. No verdict exists during generation.
    complete = [r for r in private_records if r["parse_complete"]]
    verdicts, lean_wall, lean_process = lean_check(
        [(tok.statement(r["prompt"]), r["lean_text"]) for r in complete]
    )
    validity = {r["attempt_uid"]: False for r in private_records}
    validity.update({r["attempt_uid"]: bool(v) for r, v in zip(complete, verdicts, strict=True)})
    state_map = {state["state_id"]: state for state in states}
    for state in states:
        labels = [validity[uid] for uid in state["incumbent_ids"]]
        state["accept_value"] = sum(
            p * y for p, y in zip(state["selection_probabilities"], labels, strict=True)
        )
        state["oracle_available"] = any(labels)
    with (out / "branches_scored.jsonl").open("w") as stream:
        for branch in branches:
            state = state_map[branch["state_id"]]
            labels = [validity[uid] for uid in state["incumbent_ids"]]
            probabilities = branch["after"]["selection_probabilities"]
            old_mass = sum(p * y for p, y in zip(probabilities[: len(labels)], labels, strict=True))
            future_labels = [validity[uid] for uid in branch["future_ids"]]
            new_mass = sum(
                p * y for p, y in zip(probabilities[len(labels) :], future_labels, strict=True)
            )
            branch.update(
                accept_value=state["accept_value"],
                continue_value=old_mass + new_mass,
                delta_continuation=old_mass + new_mass - state["accept_value"],
                old_correct_mass=old_mass,
                new_correct_mass=new_mass,
                new_attempt_valid=future_labels[0],
                any_new_valid=any(future_labels),
                new_attempt_count=len(future_labels),
                oracle_before=state["oracle_available"],
                oracle_after=state["oracle_available"] or any(future_labels),
                accept_probability=state["accept_probability"],
                family=state["family"],
            )
            if args.continuation == "one_extra":
                branch["delta_one"] = branch["delta_continuation"]
            stream_record(stream, branch)
    with (out / "states_scored.jsonl").open("w") as stream:
        for state in states:
            stream_record(stream, state)
    (out / "validity.json").write_text(json.dumps(validity, indent=2) + "\n")
    (out / "completion.json").write_text(
        json.dumps(
            {
                "states": len(states),
                "forced_states": sum(s["forced"] for s in states),
                "branches": len(branches),
                "attempts": len(private_records),
                "valid_attempts": sum(validity.values()),
                "lean_wall_seconds": lean_wall,
                "lean_process_seconds": lean_process,
                "completed_utc": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
