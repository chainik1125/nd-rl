"""Exact policy-relative continuation value on a fixed frozen candidate ordering.

This diagnostic integrates selection and stop probabilities exactly. Future proofs
are fixed independent samples, so it measures selection interference only and does
not estimate contextual generation harm or the full proposal's continuation value.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ["ND_OFFLINE"] = "1"
import torch
from controller import Controller, encode_history
from train_controller import batch_states, read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--public", required=True)
    parser.add_argument("--private", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--attempt-bound", type=int, default=512)
    args = parser.parse_args()
    sys.path.insert(0, args.source)
    from model import load_ckpt

    torch.manual_seed(args.seed)
    model, tok, _ = load_ckpt(args.base, "cuda")
    controller = Controller(model).cuda()
    controller.load_state_dict(torch.load(args.checkpoint, map_location="cuda")["state"])
    controller.eval()
    rows = read_jsonl(args.public)
    private = {v["theorem_id"]: v["labels"] for v in read_jsonl(args.private)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    rng = random.Random(args.seed)
    with (out / "states.jsonl").open("w") as log, torch.no_grad():
        for row in rows:
            validity = {v["candidate_id"]: v["valid"] for v in private[row["theorem_id"]]}
            for repeat in range(args.repeats):
                candidates = list(row["candidates"])
                rng.shuffle(candidates)
                states = []
                for n in range(1, min(8, len(candidates)) + 1):
                    raw = [c["transition_tokens"] for c in candidates[:n]]
                    ids = encode_history(tok, tok.encode_prompt(row["prompt"]), raw, args.context)
                    accept, index = batch_states(controller, tok, [(ids, n)], "cuda")
                    selected_success = sum(
                        float(index.probs[0, j]) * int(validity[c["candidate_id"]])
                        for j, c in enumerate(candidates[:n])
                    )
                    forced = (
                        n == min(8, len(candidates))
                        or len(ids) + args.attempt_bound + 3 > args.context
                    )
                    states.append(
                        {
                            "theorem_id": row["theorem_id"],
                            "repeat": repeat,
                            "attempts": n,
                            "occupied_tokens": len(ids),
                            "accept_probability": float(accept.probs[0]),
                            "accept_success_probability": selected_success,
                            "oracle_available": any(
                                validity[c["candidate_id"]] for c in candidates[:n]
                            ),
                            "forced": forced,
                        }
                    )
                    if forced:
                        break
                value = states[-1]["accept_success_probability"]
                for state in reversed(states):
                    if state["forced"]:
                        state["follow_policy_success_probability"] = value
                        state["continue_minus_accept"] = None
                    else:
                        state["forced_continue_success_probability"] = value
                        state["continue_minus_accept"] = value - state["accept_success_probability"]
                        p = state["accept_probability"]
                        value = p * state["accept_success_probability"] + (1 - p) * value
                        state["follow_policy_success_probability"] = value
                    log.write(json.dumps(state) + "\n")


if __name__ == "__main__":
    main()
