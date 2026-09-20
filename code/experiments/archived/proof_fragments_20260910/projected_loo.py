"""Fixed-policy reward/score projection from two previously acquired draw sets.

No optimizer, rollout sampler, Fisher estimate, or final-population reader is
used. Positive family benefit means lower target loss under infinitesimal raw
reward-gradient ascent, not under the realized Adam training procedure.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

import common
import learning as L
import nd_environment as E
from all_cut_pilot import contract, metadata_compatibility
from initial_cuts import state_hash
from release_populations import check_record
from score_rl_selection import selection_metadata, validate_pipeline

GROUP_SIZE = 8
TOKEN_DENOMINATOR = 512
SAMPLING_TEMPERATURE = .8


def project_group(model, tok, rows, query_gradient, *, ledger, precision="fp32"):
    """Project the unnormalized whole-response LOO reward-ascent gradient."""
    if len(rows) != GROUP_SIZE or any(r["reward"] not in (0, 1) for r in rows):
        raise ValueError("Expected eight binary-reward attempts from one source state")
    if len({r["source_id"] for r in rows}) != 1:
        raise ValueError("A projected group must contain exactly one source")
    if any(r.get("reset_index", 0) != 0 or r.get("prefix_ids", []) for r in rows):
        raise ValueError("Projection is restricted to full-start states")
    rewards = [r["reward"] for r in rows]
    k = sum(rewards)
    model.eval()
    model.zero_grad(set_to_none=True)
    if k in (0, GROUP_SIZE):
        block = ledger.setdefault("algebraic_zero_groups", {"groups": 0, "flop_proxy": 0})
        block["groups"] += 1
        return {"D": 0., "D_div_512": 0., "successes": k, "mixed": False,
                "backward_executed": False, "group_size": GROUP_SIZE}
    batch = L.collate(rows, tok.pad_id, next(model.parameters()).device)
    scores = L.logprobs(model, batch, SAMPLING_TEMPERATURE, precision)
    advantages = torch.tensor([(GROUP_SIZE * r - k) / (GROUP_SIZE - 1) for r in rewards],
                              device=scores.device, dtype=scores.dtype)
    # The 1/G is essential: A_i already contains G/(G-1), not 1/(G-1).
    # Keep entire response log probability, including sampled termination.
    objective = ((scores * batch["mask"]).sum(-1) * advantages).sum() / GROUP_SIZE
    objective.backward()
    L.account(ledger, model, batch, "raw_loo_teacher_backward", True)
    projection = 0.
    parameters = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise FloatingPointError("Missing or nonfinite group score gradient")
        projection += float((query_gradient[name].float() * parameter.grad.float()).sum())
        parameters += parameter.numel()
    if not math.isfinite(projection):
        raise FloatingPointError("Nonfinite projected reward gradient")
    block = ledger.setdefault("group_projection_arithmetic", {"groups": 0, "parameters": 0, "flop_proxy": 0})
    block["groups"] += 1
    block["parameters"] += parameters
    block["flop_proxy"] += 2 * parameters
    model.zero_grad(set_to_none=True)
    return {"D": projection, "D_div_512": projection / TOKEN_DENOMINATOR,
            "successes": k, "mixed": True, "backward_executed": True, "group_size": GROUP_SIZE}


def check_source_files(manifest):
    for path, digest in {**manifest["code_sha256"], **manifest["dependency_sha256"]}.items():
        if common.sha(path) != digest:
            raise ValueError("Producer source code or dependency identity changed")


def bind_draw_directory(path, kind, current, candidate_hash):
    """Validate producer metadata before opening its attempt payload."""
    path = Path(path).resolve()
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    args = manifest["args"]
    if manifest["input_sha256"].get("data") != candidate_hash:
        raise ValueError("Draw producer does not use the exact released candidate population")
    if common.sha(args["data"]) != candidate_hash or common.sha(args["checkpoint"]) != current["checkpoint_sha256"]:
        raise ValueError("Draw producer data/checkpoint input changed")
    check_source_files(manifest)
    result_path = path / "result.json"
    completion = json.loads((path / "completed.json").read_text())
    if not completion.get("ok") or completion["result_sha256"] != common.sha(result_path):
        raise ValueError("Draw producer is incomplete or its result changed")
    result = json.loads(result_path.read_text())
    binding = {"directory": str(path), "manifest_sha256": common.sha(manifest_path),
               "result_sha256": common.sha(result_path), "completed_sha256": common.sha(path / "completed.json")}
    if kind == "original":
        problem = metadata_compatibility(manifest, current)
        if problem or args.get("command") != "evaluate" or args.get("start") != "full":
            raise ValueError("Original full-start draw producer is incompatible: " + str(problem))
        if result.get("samples") != 8 or result.get("start") != "full" or result.get("n_theorems") != 768:
            raise ValueError("Original producer must cover all 768 states with eight draws")
        attempts_path = path / "attempts.jsonl"
        seed = args["seed"]
        reused_cost = result["total_flop_proxy"]
        if reused_cost != L.total_flop_proxy(result["ledger"]):
            raise ValueError("Original producer compute does not reconcile")
        binding["reused_cost_scope"] = "Entire original evaluation, including its reference scoring and behavior audit"
        binding["expected_covered_sources"] = result["strict_solved"]
    elif kind == "fresh_audit":
        if manifest.get("command") != "all_cut_pilot_audit" or manifest.get("contract") != current:
            raise ValueError("Fresh draw producer is not the compatible all-cut audit")
        if args.get("command") != "audit" or result.get("independent_of_fit_draws") is not True or result.get("policy_updates") != 0:
            raise ValueError("Fresh draws must be the independently seeded, unchanged-policy audit")
        for key in ("temperature", "max_new_tokens", "precision"):
            if args.get(key) != current[key]:
                raise ValueError("Audit sampler arguments differ from its contract")
        if manifest.get("dependency_sha256") != common.dependency_hashes():
            raise ValueError("Audit dependencies differ from the frozen scoring environment")
        for filename, key in (("learning.py", "sampler_sha256"), ("nd_environment.py", "environment_sha256")):
            observed = [digest for path, digest in manifest["code_sha256"].items() if Path(path).name == filename]
            if observed != [current[key]]:
                raise ValueError("Audit implementation differs from the declared sampling contract")
        full = result["conditions"]["full"]
        if full["n_sources"] != 768 or full["attempts"] != 768 * GROUP_SIZE:
            raise ValueError("Fresh full-start component must include every candidate state")
        seed = args["seed"] + 1000003
        if full["seed"] != seed:
            raise ValueError("Audit full-start seed is not its declared independent component")
        map_binding_path = path / "map_binding.json"
        map_binding = json.loads(map_binding_path.read_text())
        if (map_binding["audit_seeds"] != [args["seed"] + i * 1000003 for i in range(3)]
                or common.sha(map_binding["path"]) != map_binding["sha256"]
                or result["initial_cuts_sha256"] != map_binding["sha256"]):
            raise ValueError("Fresh audit map or independent seed binding changed")
        binding["map_binding_sha256"] = common.sha(map_binding_path)
        attempts_path = path / "full_attempts.jsonl"
        reused_cost = full["total_flop_proxy"]
        if reused_cost != L.total_flop_proxy(result["ledger"]["full"]):
            raise ValueError("Fresh full-start compute does not reconcile")
        binding["reused_cost_scope"] = "Full-start audit component only; fit/chosen/latest costs remain in their original ledgers"
        binding["expected_successes"] = full["successes"]
        binding["expected_mixed_groups"] = full["mixed_groups"]
    else:
        raise ValueError("Unknown reused draw set")
    binding.update(kind=kind, sampling_seed=seed, attempts_path=str(attempts_path),
                   attempts_sha256=common.sha(attempts_path), reused_previously_charged_flop_proxy=reused_cost)
    return binding


def validate_attempt(row, expected, tok, *, kind, seed):
    if type(row.get("attempt_index")) is not int or not 0 <= row["attempt_index"] < GROUP_SIZE:
        raise ValueError("Attempt index must identify one of eight distinct draws")
    for key in ("source_id", "prompt", "prompt_ids", "original_prompt_ids", "prefix_ids",
                "reset_index", "source_lines", "format", "family"):
        if row.get(key) != expected[key]:
            raise ValueError("Reused attempt differs from the exact released full-start state")
    if row["reset_index"] != 0 or row["prefix_ids"] or row.get("effective_max_new_tokens") != TOKEN_DENOMINATOR:
        raise ValueError("Only complete-budget, empty-prefix attempts are eligible")
    if kind == "fresh_audit" and (row.get("condition") != "full" or row.get("sampling_seed") != seed):
        raise ValueError("Fresh group is not the unconditional full-start audit component")
    response = row["response_ids"]
    logp = row["rollout_logprobs"]
    if not 1 <= len(response) <= TOKEN_DENOMINATOR or len(logp) != len(response):
        raise ValueError("Every sampled action must have one saved behavior log probability")
    if any(type(i) is not int or not 0 <= i < len(tok.TOKENS) for i in response):
        raise ValueError("Invalid sampled action token")
    if any(not math.isfinite(p) or p > 1e-6 for p in logp):
        raise ValueError("Invalid saved behavior log probability")
    if tok.eos_id in response:
        if response[-1] != tok.eos_id or response.count(tok.eos_id) != 1 or row.get("finish_reason") != "qed":
            raise ValueError("Sampled stopping token was truncated, duplicated or followed by more actions")
    elif len(response) != TOKEN_DENOMINATOR or row.get("finish_reason") != "token_limit":
        raise ValueError("Unterminated sample must retain the complete fixed action budget")
    verdict = E.judge(expected, response, tok)
    if verdict["reward"] != row.get("reward"):
        raise ValueError("Saved reward disagrees with strict independent replay")
    return {**expected, "response_ids": response, "rollout_logprobs": logp,
            "reward": verdict["reward"], "attempt_index": row["attempt_index"],
            "finish_reason": row["finish_reason"], "effective_max_new_tokens": TOKEN_DENOMINATOR}


def load_groups(binding, expected, tok):
    raw = common.read_jsonl(binding["attempts_path"])
    if len(raw) != len(expected) * GROUP_SIZE:
        raise ValueError("Reused draw set must have exactly eight attempts per candidate")
    groups = defaultdict(list)
    for index, row in enumerate(raw):
        sid = row.get("source_id")
        if sid not in expected:
            raise ValueError("Reused source is outside the frozen candidate population")
        checked = validate_attempt(row, expected[sid], tok, kind=binding["kind"], seed=binding["sampling_seed"])
        checked["origin_row_index"] = index
        groups[sid].append(checked)
    if set(groups) != set(expected):
        raise ValueError("Reused draw set does not cover every source")
    for rows in groups.values():
        if len(rows) != GROUP_SIZE or sorted(r["attempt_index"] for r in rows) != list(range(GROUP_SIZE)):
            raise ValueError("A source must have each attempt index exactly once")
        rows.sort(key=lambda row: row["attempt_index"])
    successes = [sum(r["reward"] for r in rows) for rows in groups.values()]
    observed = {"expected_covered_sources": sum(k > 0 for k in successes),
                "expected_successes": sum(successes),
                "expected_mixed_groups": sum(0 < k < GROUP_SIZE for k in successes)}
    if any(binding[key] != actual for key, actual in observed.items() if key in binding):
        raise ValueError("Replayed rewards differ from the completed producer summary")
    return dict(groups)


def load_banks(path, candidates, candidate_hash):
    manifest = json.loads(Path(path).read_text())
    if (manifest.get("status") != "frozen_rl_calibration_banks" or manifest.get("candidate_sha256") != candidate_hash
            or manifest.get("bank_count") != 24 or manifest.get("full_source_cap") != 6):
        raise ValueError("Expected all 24 original matched cap6 calibration banks")
    banks = {}
    hashes = {str(Path(path).resolve()): common.sha(path)}
    for name, bank in manifest["banks"].items():
        family, arm = name.rsplit("_", 1)
        if arm not in ("family", "control"):
            raise ValueError("Unknown calibration bank arm")
        for key, hash_key in (("bank", "bank_sha256"), ("original_matching", "matching_sha256")):
            if common.sha(bank[key]) != bank[hash_key]:
                raise ValueError("Frozen bank or original matching receipt changed")
            hashes[bank[key]] = bank[hash_key]
        rows = common.read_jsonl(bank["bank"])
        ids = [r["source_id"] for r in rows]
        matching = common.read_jsonl(bank["original_matching"])
        expected_ids = [r["family_source" if arm == "family" else "ordinary_source"] for r in matching]
        if len(ids) != 64 or len(set(ids)) != 64 or ids != bank["source_ids_in_order"] or ids != expected_ids:
            raise ValueError("Bank does not preserve its original ordered 64-source matching receipt")
        if any(r != candidates.get(r["source_id"]) for r in rows):
            raise ValueError("Bank source content differs from released candidate provenance")
        if arm == "family" and any(r["family"] != family for r in rows):
            raise ValueError("Family bank label differs from actual sources")
        banks[name] = ids
    families = {name.rsplit("_", 1)[0] for name in banks}
    if len(banks) != 24 or len(families) != 12 or any(f"{family}_{arm}" not in banks for family in families for arm in ("family", "control")):
        raise ValueError("Every one of 12 families must retain both original banks")
    return banks, hashes


def bank_contrasts(source_values, banks):
    result = {}
    for family in sorted({name.rsplit("_", 1)[0] for name in banks}):
        means = {arm: sum(source_values[sid] for sid in banks[f"{family}_{arm}"]) / 64
                 for arm in ("family", "control")}
        result[family] = {"family_mean_D": means["family"], "control_mean_D": means["control"],
                          "predicted_target_bits_benefit_per_unit_raw_ascent_step": (means["control"] - means["family"]) / TOKEN_DENOMINATOR}
    return result


def draw_noise_summary(first, second, banks, *, repeats=2000, seed=20260916):
    if set(first) != set(second):
        raise ValueError("Both independent estimates must cover identical source states")
    if repeats < 0:
        raise ValueError("Negative descriptive resampling count")
    sources = sorted(first)
    families = sorted({name.rsplit("_", 1)[0] for name in banks})
    offsets = {sid: index for index, sid in enumerate(sources)}
    weights = np.zeros((len(sources), len(families)))
    for j, family in enumerate(families):
        for arm, sign in (("control", 1), ("family", -1)):
            for sid in banks[f"{family}_{arm}"]:
                weights[offsets[sid], j] += sign / (64 * TOKEN_DENOMINATOR)
    a = np.array([first[sid] for sid in sources])
    b = np.array([second[sid] for sid in sources])
    point = ((a + b) / 2) @ weights
    # The same resampled source estimate enters every family/control membership.
    rng = np.random.default_rng(seed)
    draws = np.empty((repeats, len(families)))
    for index in range(repeats):
        fraction_second = rng.integers(0, 2, size=(len(sources), 2)).mean(axis=1)
        draws[index] = (a + fraction_second * (b - a)) @ weights
    interval = np.quantile(draws, [.025, .975], axis=0) if repeats else None
    return {"scope": "Descriptive conditional resampling sensitivity, not calibrated Monte Carlo confidence intervals",
            "conditioning": "Frozen query gradient, fixed sources and fixed banks",
            "method": "Resample two entire group estimates within each source; synchronize its value across all bank memberships",
            "limitations": "Only two independent groups per source; rare successes can be missed and zero/zero does not establish zero variance. Percentile tails are poorly estimated. No context/population resampling and no pooling of LOO groups.",
            "variance_assumptions": "Paired-difference variance estimate assumes two independent identically distributed group estimates per fixed source and independent draws across distinct sources, conditional on the fixed query. Shared bank sources use combined coefficients. Two replicates cannot validate these variance estimates or rare-event tails.",
            "seed": seed, "replicates": repeats,
            "families": {family: {"mean_two_draw_benefit": float(point[j]),
                                    "descriptive_resampling_percentiles_2_5_and_97_5": interval[:, j].tolist() if repeats else None,
                                    "two_draw_variance_estimate": float(np.sum(weights[:, j] ** 2 * (a - b) ** 2 / 4))}
                         for j, family in enumerate(families)}}


def run(args):
    started = time.time()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Diagnostic output must be empty")
    if not 0 < args.max_logprob_error <= .001:
        raise ValueError("Behavior tolerance cannot weaken the frozen 0.001 bound")
    pipeline = json.loads(Path(args.pipeline).read_text())
    design = json.loads(Path(args.design).read_text())
    validate_pipeline(pipeline, design)
    if Path(args.checkpoint).resolve() != Path(design["checkpoint"]).resolve():
        raise ValueError("Projection requires the design's common adapted checkpoint")
    if Path(args.banks_manifest).resolve() not in {Path(p).resolve() for p in pipeline["inputs"]}:
        raise ValueError("Calibration banks are not declared frozen pipeline inputs")
    candidate_path = Path(args.candidates).resolve()
    candidate_hash = common.sha(candidate_path)
    population_path = candidate_path.parent / "manifest.json"
    population = json.loads(population_path.read_text())
    if population.get("status") != "released_complete_primary" or population["population_sha256"].get("candidate") != candidate_hash:
        raise ValueError("Diagnostic requires the complete released candidate population")
    selection_path = Path(args.selection).resolve()
    selection_hash = common.sha(selection_path)
    selection_metadata(selection_path, selection_hash, pipeline["inputs"])
    current = contract(argparse.Namespace(temperature=SAMPLING_TEMPERATURE, max_new_tokens=TOKEN_DENOMINATOR,
                                         precision=args.precision), common.sha(args.checkpoint))
    original = bind_draw_directory(args.original, "original", current, candidate_hash)
    fresh = bind_draw_directory(args.audit, "fresh_audit", current, candidate_hash)
    if original["sampling_seed"] == fresh["sampling_seed"]:
        raise ValueError("The two independent draw sets cannot share a sampling seed")
    records = common.read_jsonl(candidate_path)
    if len(records) != 768 or len({r["source_id"] for r in records}) != 768:
        raise ValueError("All 768 distinct released candidate states are required")
    for row in records:
        check_record(row, "candidate", 6)
    by_source = {r["source_id"]: r for r in records}
    if sorted(Counter(r["family"] for r in records).values()) != [64] * 12:
        raise ValueError("Candidate population must retain 64 sources in each of 12 families")
    banks, bank_hashes = load_banks(args.banks_manifest, by_source, candidate_hash)
    selection = common.read_jsonl(selection_path)
    for row in selection:
        check_record(row, "selection", 12)
    if len(selection) != 384 or Counter(r["length_bin"] for r in selection) != {"2-6": 128, "7-9": 128, "10-12": 128}:
        raise ValueError("Expected the full three-stratum frozen selection population")
    torch.set_num_threads(args.cpu_threads)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model, tok, metadata = L.load_model(args.checkpoint, args.device)
    if metadata.get("format") != E.FORMAT:
        raise ValueError("Common adapted input format is required")
    expected = {r["source_id"]: E.episode(r, 0, tok) for r in records}
    query = L.encode_population(selection, tok, "empty")
    if (any(len(r["prompt_ids"]) + TOKEN_DENOMINATOR > model.config.max_seq_len for r in expected.values())
            or any(len(r["prompt_ids"]) + len(r["response_ids"]) > model.config.max_seq_len for r in query)):
        raise ValueError("Complete trajectory/reference budget must fit without truncation")
    groups = {binding["kind"]: load_groups(binding, expected, tok) for binding in (original, fresh)}
    inputs = {str(Path(path).resolve()): common.sha(path) for path in
              (args.pipeline, args.design, args.checkpoint, candidate_path, population_path, selection_path)}
    manifest = {"command": "projected_raw_loo", "args": vars(args), "contract": current,
                "contract_sha256": common.digest(current),
                "query": "Equal reference-length-stratum whole-proof selection bits at T=1",
                "preconditioner": "identity", "group_size": GROUP_SIZE, "fixed_token_denominator": TOKEN_DENOMINATOR,
                "source_cap": 6, "selection_oracle_cap": 12, "input_sha256": {**inputs, **bank_hashes},
                "draw_bindings": [original, fresh], "source_bank_ids": banks,
                "tokenizer_sha256": common.digest(tok.state_dict()), "model_config": vars(model.config),
                "state_sha256": {sid: state_hash(row) for sid, row in expected.items()},
                "code_sha256": {str(p): common.sha(p) for p in Path(__file__).parent.glob("*.py")},
                "dependency_sha256": common.dependency_hashes(), "started_unix": started,
                "policy_updates": 0, "new_rollouts": 0, "final_data_accessed": False, "selector_changes": False}
    output.mkdir(parents=True, exist_ok=True)
    common.write_json(output / "manifest.json", manifest)
    baseline = {name: p.detach().clone() for name, p in model.named_parameters()}
    ledger = {}
    query_ledger = {}
    gradient = L.query_gradient(model, tok, query, args.gradient_batch_size, query_ledger, args.precision)
    ledger["single_frozen_query_gradient"] = query_ledger
    gradient_path = output / "selection_gradient.pt"
    torch.save({"gradient": {name: value.cpu() for name, value in gradient.items()},
                "checkpoint_sha256": current["checkpoint_sha256"], "selection_sha256": selection_hash,
                "score_temperature": 1., "weighting": manifest["query"], "preconditioner": "identity",
                "ledger": query_ledger, "used_for_updates": False}, gradient_path)
    estimates = {}
    draw_results = {}
    for binding in (original, fresh):
        kind = binding["kind"]
        local = {}
        ordered = [row for record in records for row in groups[kind][record["source_id"]]]
        behavior = L.audit_behavior(model, tok, ordered, temperature=SAMPLING_TEMPERATURE,
                                   batch_size=args.microbatch, precision=args.precision,
                                   tolerance=args.max_logprob_error, ledger=local)
        rows = []
        for index, record in enumerate(records):
            sid = record["source_id"]
            attempts = groups[kind][sid]
            projection = project_group(model, tok, attempts, gradient, ledger=local, precision=args.precision)
            rows.append({**projection, "source_id": sid, "theorem_id": record["theorem_id"],
                         "context_id": record["context_id"], "family": record["family"], "source_lines": record["n_lines"],
                         "draw_set": kind, "sampling_seed": binding["sampling_seed"], "state_sha256": state_hash(expected[sid]),
                         "action_group_sha256": common.digest([r["response_ids"] for r in attempts]),
                         "behavior_group_sha256": common.digest([r["rollout_logprobs"] for r in attempts]),
                         "origin_row_indices": [r["origin_row_index"] for r in attempts],
                         "action_tokens": sum(len(r["response_ids"]) for r in attempts)})
            if (index + 1) % 64 == 0:
                print(json.dumps({"draw_set": kind, "completed_sources": index + 1, "total_sources": len(records)}), flush=True)
        common.write_jsonl(output / f"{kind}_source_estimates.jsonl", rows)
        estimates[kind] = {r["source_id"]: r["D"] for r in rows}
        local["verification"] = {"strict_reward_replays": len(ordered), "exact_full_start_states": len(records)}
        draw_results[kind] = {"n_sources": len(rows), "mixed_groups": sum(r["mixed"] for r in rows),
                              "likelihood_audit": behavior, "family_contrasts": bank_contrasts(estimates[kind], banks),
                              "source_estimates_sha256": common.sha(output / f"{kind}_source_estimates.jsonl"),
                              "ledger": local, "new_derivative_and_audit_flop_proxy": L.total_flop_proxy(local)}
        ledger[kind] = local
    if any(not torch.equal(p.detach(), baseline[name]) or p.grad is not None for name, p in model.named_parameters()):
        raise AssertionError("Diagnostic changed model weights or left an attached gradient")
    noise = draw_noise_summary(estimates["original"], estimates["fresh_audit"], banks,
                               repeats=args.bootstrap_repeats, seed=args.bootstrap_seed)
    result = {"draw_sets": draw_results, "draw_noise": noise, "selection_gradient_sha256": common.sha(gradient_path),
              "ledger": ledger, "new_total_flop_proxy": L.total_flop_proxy(ledger),
              "reused_previously_charged_flop_proxy": sum(b["reused_previously_charged_flop_proxy"] for b in (original, fresh)),
              "reused_cost_note": "Acquisition cost receipt only; do not add it twice to the sprint total",
              "elapsed_seconds": time.time() - started, "weights_unchanged_exactly": True,
              "policy_updates": 0, "new_rollouts": 0, "ranking_changed": False, "final_data_accessed": False,
              "interpretation": "Fixed-query raw reward-gradient mechanism; no Adam, finite-step, Fisher or long-horizon prediction claim"}
    common.write_json(output / "result.json", result)
    common.write_json(output / "completed.json", {"ok": True, "result_sha256": common.sha(output / "result.json")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for flag in ("pipeline", "design", "checkpoint", "candidates", "selection", "banks-manifest", "original", "audit", "output"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--gradient-batch-size", type=int, default=16)
    parser.add_argument("--microbatch", type=int, default=16)
    parser.add_argument("--max-logprob-error", type=float, default=.001)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260916)
    run(parser.parse_args())
