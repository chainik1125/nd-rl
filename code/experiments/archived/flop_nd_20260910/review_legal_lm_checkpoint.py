"""Compare cached constrained decoding with full causal scoring at real weights."""
import argparse
import json
from pathlib import Path
import time
import torch
import common
import learning as L
from legal_lm import generate


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint_hash = common.sha(args.checkpoint)
    model, tok, _ = L.load_model(args.checkpoint, "cpu")
    populations = [common.ROOT / "data/v1/development.jsonl", common.ROOT / "data/structured_lm_v1/development.jsonl"]
    rows = [row for path in populations for row in common.read_jsonl(path)[:4]]
    started = time.monotonic()
    together = generate(model, tok, rows, max_new_tokens=1024, batch_size=8, max_lines=24, temperature=0)
    branch_positions, total_positions = 0, 0
    for row, cached in zip(rows, together):
        singleton = generate(model, tok, [row], max_new_tokens=1024, batch_size=1, max_lines=24, temperature=0)[0]
        if singleton["response_ids"] != cached["response_ids"] or singleton["action_trace"] != cached["action_trace"]:
            raise AssertionError("Real-checkpoint cached batch differs from singleton decoding")
        prompt = tok.encode_prompt(row["prompt"]) + [tok.prf_id]
        tokens = prompt + cached["response_ids"]
        logits = model(torch.tensor([tokens[:-1]], dtype=torch.long))[0]
        for offset, (chosen, allowed) in enumerate(zip(cached["response_ids"], cached["allowed_token_ids"])):
            actual = allowed[int(logits[len(prompt) - 1 + offset, allowed].argmax())]
            if actual != chosen:
                raise AssertionError("Cached constrained argmax differs from full causal argmax")
            branch_positions += len(allowed) > 1
            total_positions += 1
    if common.sha(args.checkpoint) != checkpoint_hash or any(p.grad is not None for p in model.parameters()):
        raise AssertionError("Read-only checkpoint check failed")
    result = {"status": "passed", "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint_hash,
              "theorems": len(rows), "source_ids": [r["source_id"] for r in rows], "response_positions_checked": total_positions,
              "branch_positions_checked": branch_positions, "elapsed_seconds": time.monotonic() - started,
              "input_sha256": {str(p.resolve()): common.sha(p) for p in [*populations, Path(__file__), Path(__file__).with_name("legal_lm.py"), Path(L.__file__), *common.VENDOR.rglob("*.py")]},
              "checks": ["Actual checkpoint, eight prospectively indexed development prompts", "Cached batch exactly matches singleton responses/actions", "Every constrained token matches independent full causal scoring", "Weights unchanged; no gradients"]}
    common.write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("input_sha256", "source_ids")}))


if __name__ == "__main__":
    main()
