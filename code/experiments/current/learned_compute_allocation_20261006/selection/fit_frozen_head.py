"""One prespecified convex binary head fit; no model fine-tuning or DEV selection."""

import argparse
import datetime
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

COMMON = Path(__file__).with_name("frozen_head_common.py")
spec = importlib.util.spec_from_file_location("frozen_head_fit_common", COMMON)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def fit(features, labels):
    weight = torch.zeros(384, dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weight, bias],
        lr=1,
        max_iter=500,
        max_eval=625,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=100,
        line_search_fn="strong_wolfe",
    )
    calls = []

    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(features @ weight + bias, labels)
        objective = loss + 0.5e-3 * weight.square().sum()
        assert torch.isfinite(objective)
        objective.backward()
        calls.append(float(objective.detach()))
        return objective

    optimizer.step(closure)
    final = closure()
    gradient = max(float(weight.grad.abs().max()), float(bias.grad.abs()))
    state = optimizer.state[weight]
    return (
        weight.detach(),
        bias.detach(),
        {
            "objective": float(final.detach()),
            "closure_evaluations": len(calls),
            "optimizer_iterations": int(state.get("n_iter", 0)),
            "optimizer_function_evaluations": int(state.get("func_evals", 0)),
            "max_absolute_gradient": gradient,
            "gradient_tolerance_met": gradient <= 1e-7,
            "objective_history": calls,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--train-private", required=True)
    parser.add_argument("--dev-private", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    started = datetime.datetime.now(datetime.UTC).isoformat()
    timer = time.monotonic()
    torch.set_num_threads(4)
    cache, metadata = common.read_cache(args.features, "train")
    extraction = json.loads((Path(args.features) / "config.json").read_text())
    assert extraction["original_parameters_unchanged"]
    assert extraction["trainable_original_parameters"] == 0
    labels = common.aligned_labels(cache, args.train_private)
    weight, bias, convergence = fit(
        torch.from_numpy(cache["features"]).double(), torch.from_numpy(labels).double()
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__)
    config = {
        "args": vars(args),
        "argv": sys.argv,
        "pid": os.getpid(),
        "started_utc": started,
        "fit_wall_seconds_monotonic": time.monotonic() - timer,
        "precision": "CPU FP64 scalar fit/readout",
        "standardization": None,
        "head_parameters": weight.numel() + bias.numel(),
        "training_selection": "one terminal optimizer result; DEV never selects checkpoint",
        "optimizer": {
            "name": "full-batch L-BFGS",
            "lr": 1,
            "max_iter": 500,
            "max_eval": 625,
            "history_size": 100,
            "tolerance_grad": 1e-7,
            "tolerance_change": 1e-9,
            "line_search_fn": "strong_wolfe",
        },
        "objective": "mean unweighted BCE +0.5e-3*sum(weight**2); intercept unpenalized",
        "original_checkpoint_sha256": extraction["checkpoint_sha256"],
        "frozen_original_parameter_sha256": extraction["parameter_sha256_after"],
        "source_hashes": {
            str(p): common.digest(p)
            for p in [source, COMMON, source.with_name("frozen_head_method_freeze.md")]
        },
        "input_hashes": {
            str(p): common.digest(p)
            for p in [
                Path(args.train_private),
                Path(args.features) / "train_features.npz",
                Path(args.features) / "config.json",
            ]
        },
        "convergence": convergence,
    }
    for path in [source, COMMON, source.with_name("frozen_head_method_freeze.md")]:
        (out / path.name).write_bytes(path.read_bytes())
    head = out / "head.pt"
    torch.save({"weight": weight, "bias": bias, "config": config}, head)
    config["head_sha256"] = common.digest(head)
    # Development labels are read only after the one prespecified head has been saved.
    dev, dev_metadata = common.read_cache(args.features, "dev")
    assert set(cache["theorem_ids"]).isdisjoint(set(dev["theorem_ids"]))
    dev_labels = common.aligned_labels(dev, args.dev_private)
    predictions, summary = common.predict(dev, dev_labels, weight, bias)
    config["input_hashes"].update(
        {
            str(p): common.digest(p)
            for p in [Path(args.dev_private), Path(args.features) / "dev_features.npz"]
        }
    )
    config["process_wall_seconds_monotonic"] = time.monotonic() - timer
    (out / "dev_predictions.json").write_text(json.dumps(predictions) + "\n")
    (out / "dev_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({"convergence": convergence, "dev": summary}))


if __name__ == "__main__":
    main()
