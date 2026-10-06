"""CPU-only public DEV feature decisions; never opens verifier labels."""

import argparse
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLDER))
import cached_confidence_diagnostic as primary  # noqa: E402
from tokenizer import make_tokenizer  # noqa: E402

HELPER = FOLDER.parent / "analysis/analyze_candidates.py"
spec = importlib.util.spec_from_file_location("frozen_public_feature_helper", HELPER)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    source = Path(args.diagnostic)
    config = json.loads((source / "config.json").read_text())
    panel = helper.load_rows(source / "public_panel.jsonl")
    assert {r["split"] for r in panel} == {"dev"}
    assert config["seed"] == 124006
    actions = helper.load_rows(source / "action_records.jsonl")
    discrepancies = helper.load_rows(source / "candidate_discrepancies.jsonl")
    totals = {(r["theorem_id"], r["candidate_id"]): r for r in discrepancies}
    model_path = (
        primary.ROOT
        / "artifacts/learned_compute_allocation_20261006/analysis"
        / "features1600_main_dev/feature_model.json"
    )
    model = json.loads(model_path.read_text())
    assert model["features"] == helper.FEATURE_NAMES
    tok = make_tokenizer("lean_staten")
    candidate_results = []
    pool_results = []
    for row_index, row in enumerate(panel):
        replay_features = []
        cached_features = []
        for k in range(8):
            base = random.Random(config["seed"] * 1000003 + row_index * 8 + k).randint(0, 32)
            env = primary.collect.Env(row["prompt"], canon=True, base=base, assign=True)
            records = sorted(
                [
                    r
                    for r in actions
                    if r["theorem_id"] == row["theorem_id"] and r["candidate_id"] == k
                ],
                key=lambda r: r["wave"],
            )
            effective = []
            closed = False
            for record in records:
                assert not closed
                assert record["prefix"] == primary.collect.prompt_ids(tok, env)
                action, ended = tok.decode_action(record["raw_action"])
                if ended:
                    ok, _ = env.apply(action)
                    effective.append(tok.encode_toks(env._eff) + [tok.eos])
                    closed = not ok or env.done or env.steps >= 48
                else:
                    effective.append(record["raw_action"])
                    closed = True
            assert closed
            total = totals[(row["theorem_id"], k)]
            sample, base_lp, entropy = total["replay_mean"]
            candidate = {
                "canonical_name_base": base,
                "effective_actions": effective,
                "rendered_text": tok.text([tok.itos[token] for a in effective for token in a]),
                "generated_tokens": total["tokens"],
                "prompt_tokens": sum(len(r["prefix"]) for r in records),
                "mean_logprob": sample,
                "mean_base_logprob": base_lp,
                "mean_entropy": entropy,
            }
            ordinary = helper.public_features(row, candidate, tok, primary.collect.Env)
            sample, base_lp, entropy = total["cached_mean"]
            candidate.update(mean_logprob=sample, mean_base_logprob=base_lp, mean_entropy=entropy)
            alternate = helper.public_features(row, candidate, tok, primary.collect.Env)
            assert ordinary[3:] == alternate[3:]
            replay_features.append(ordinary)
            cached_features.append(alternate)
        replay_logits = helper.feature_scores(replay_features, model)
        cached_logits = helper.feature_scores(cached_features, model)
        replay_probs = [1 / (1 + math.exp(-v)) for v in replay_logits]
        cached_probs = [1 / (1 + math.exp(-v)) for v in cached_logits]
        original_ties = [k for k, v in enumerate(replay_logits) if v == max(replay_logits)]
        alternate_ties = [k for k, v in enumerate(cached_logits) if v == max(cached_logits)]
        pool_results.append(
            {
                "theorem_id": row["theorem_id"],
                "replay_ties": original_ties,
                "cached_ties": alternate_ties,
                "same_argmax_tie_set": original_ties == alternate_ties,
            }
        )
        for k in range(8):
            candidate_results.append(
                {
                    "theorem_id": row["theorem_id"],
                    "candidate_id": k,
                    "replay_logit": replay_logits[k],
                    "cached_logit": cached_logits[k],
                    "replay_probability": replay_probs[k],
                    "cached_probability": cached_probs[k],
                    "same_threshold_crossing_0.85": (replay_probs[k] >= 0.85)
                    == (cached_probs[k] >= 0.85),
                }
            )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "candidate_scores.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in candidate_results)
    )
    (out / "pool_ties.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pool_results))
    summary = {
        "scope": (
            "Same public DEV records; only three cached confidence aggregates substituted; "
            "no refit or labels"
        ),
        "candidates": len(candidate_results),
        "pools": len(pool_results),
        "argmax_tie_set_agreement": sum(r["same_argmax_tie_set"] for r in pool_results),
        "threshold_0.85_crossing_agreement": sum(
            r["same_threshold_crossing_0.85"] for r in candidate_results
        ),
        "max_absolute_logit_difference": max(
            abs(r["cached_logit"] - r["replay_logit"]) for r in candidate_results
        ),
        "max_absolute_probability_difference": max(
            abs(r["cached_probability"] - r["replay_probability"]) for r in candidate_results
        ),
        "input_hashes": {
            str(p): primary.digest(p)
            for p in [
                model_path,
                HELPER,
                Path(__file__),
                source / "action_records.jsonl",
                source / "candidate_discrepancies.jsonl",
                source / "public_panel.jsonl",
            ]
        },
        "interpretation": (
            "Decision agreement on a small DEV panel does not establish general substitution, "
            "correctness or speedup"
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
