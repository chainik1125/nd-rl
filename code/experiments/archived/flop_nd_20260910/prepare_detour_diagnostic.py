"""Freeze a development-only test of FLoP's clean-versus-distracted seed claim."""
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil

import common
from nd_environment import strict_complete
from prepare_data import identities
from structured_generate import make_task
from symbolic_general_env import Environment, Action


def main():
    root = common.ROOT
    output = root / "data/detour_v1"
    output.mkdir(exist_ok=False)
    excluded, paths = set(), []
    for population in ("v1", "structured_v1"):
        for split in ("targets", "development", "final"):
            path = root / "data" / population / (split + ".jsonl")
            if not path.exists():
                continue
            paths.append(path)
            for row in common.read_jsonl(path):
                excluded.add(identities(row["prompt"])[1])
    rng = random.Random(202609102230)
    clean, noisy = [], []
    for attempt in range(10000):
        task = make_task(rng, 4, "projection_atomic")
        _, context, guards = identities(task["prompt"])
        if task["target_atom"] != "P" or context in excluded:
            continue
        excluded.add(context)
        base = {"prompt": task["prompt"], "body": task["body"], "source_id": common.digest(task["prompt"]),
                "context_id": context, "guard_ids": guards, "family": "projection_atomic", "n_lines": 4,
                "target_atom": "P", "directions": task["directions"]}
        clean.append(base)
        env = Environment(base["prompt"], max_lines=6, max_scope=4)
        desired = env.source_actions(base)
        renumber = {1: 1}
        for index, action in enumerate(desired):
            if index < 2:
                ref = renumber[action.refs[0]]
                reward, done = env.step(Action("R", env.lines[ref - 1].formula, (ref,), 0))
                assert not reward and not done
            reward, done = env.step(Action(action.rule, action.formula, tuple(renumber[r] for r in action.refs), 0))
            renumber[index + 2] = len(env.lines)
        assert reward and done
        noisy_row = {**base, "body": env.body(), "n_lines": 6, "detours": "Two uncited R steps before the first and second projection"}
        assert env.pruned_body() == base["body"]
        for row in (base, noisy_row):
            verdict = strict_complete(row["prompt"], row["body"])
            assert verdict["ok"] and verdict["n_lines"] == row["n_lines"]
            actions = Environment(row["prompt"], max_lines=6).source_actions(row)
            for cut in range(len(actions)):
                reset = Environment(row["prompt"], max_lines=6)
                reset.replay_source(row, cut)
                assert not reset.success()
        noisy.append(noisy_row)
        if len(clean) == 2:
            break
    assert len(clean) == len(noisy) == 2
    common.write_jsonl(output / "clean.jsonl", clean)
    common.write_jsonl(output / "noisy.jsonl", noisy)
    (output / "no_targets.jsonl").write_text("")
    protocol = {"status": "frozen_before_training", "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Secondary development-only diagnostic; added after observing the existing clean-seed development results and rereading paper Experiment 6. No final outcomes or model selection.",
        "hypothesis": "The decrease in long-proof coverage caused by harmless seed detours is smaller for backward-curriculum PPO than for behavioral cloning.",
        "falsifier": "Cloning is equally robust, or curriculum loses as much or more coverage; report all conditions regardless of outcome.",
        "treatment": "Identical two theorem prompts; clean references have four lines, noisy references add two uncited reiterations and have six. Noisy dependency-pruned certificate exactly equals clean.",
        "algorithm": "Frozen minimal-general legal ND policy, separate critic, 500x2 tanh, gamma=lambda=.95 per inference, centered advantage scaling, entropy .01, PPO clip .2, Adam 1e-4, EMA .6 old + .4 reward, threshold >.9, episode fallback >20000; no online admission.",
        "training": {"seeds": [4501, 4502], "ppo_actions_per_arm": 131072, "ppo_batch": 512, "ppo_minibatch": 64,
            "ppo_epochs": 4, "ppo_conditions": ["clean_reset", "noisy_reset"], "bc_full_batch_updates": 1000,
            "bc_presentations": {"clean": 6000, "noisy": 10000}, "initialization": "random, matched within seed",
            "source_fraction": 1, "max_total_lines_all_conditions": 6, "max_scope": 4, "external_certificates_max_lines": 6,
            "no_long_training_theorems": True, "bc_compute_matched_to_ppo": False},
        "evaluation": {"population": str(root / "data/structured_v1/development.jsonl"), "theorems": 320,
            "profiles": {"greedy": {"attempts": 1, "temperature": 0}, "sample4": {"attempts": 4, "temperature": 1}},
            "max_total_lines": 32, "max_scope": 4, "seed": 202609101935, "prefix_assistance": False,
            "all_endpoints": True, "statistics": "Paired context bootstrap of noise effect within each method and difference of noise effects, conditional on two trained seeds.",
            "primary_metric": "Greedy theorem success across all 320 statements; report family and reference-length strata and four-sample any success secondarily."},
        "limits": ["Development population was already inspected in other experiments", "Two seeds and two seed theorems",
            "Cloning and PPO budgets are not matched", "Clean and noisy cloning have different numbers of labeled action states",
            "Proof compression is known; the test measures learned robustness, not a new mathematical strategy"],
        "source_sha256": {str(p): common.sha(p) for p in output.glob('*.jsonl')},
        "exclusion_sha256": {str(p): common.sha(p) for p in paths}, "generator_sha256": common.sha(__file__)}
    protocol_path = root / "configs/detour_protocol_v1.json"
    common.write_json(protocol_path, protocol)
    snapshot = root / "snapshots/detour_v1"
    shutil.copytree(root / "snapshots/minimal_general_v1", snapshot, ignore=shutil.ignore_patterns('__pycache__', 'source_manifest.json'))
    shutil.copy2(__file__, snapshot / Path(__file__).name)
    hashes = {str(p): common.sha(p) for p in snapshot.glob('*.py')}
    hashes.update(common.dependency_hashes())
    hashes.update(protocol["source_sha256"])
    hashes[str(protocol_path)] = common.sha(protocol_path)
    python = "/data/users/dmitry/simplex-research/.venv/bin/python"
    stages, endpoints = [], []
    for seed, variants in ((4501, ("clean", "noisy")), (4502, ("noisy", "clean"))):
        for mode in ("reset", "bc"):
            for variant in variants:
                name = f"detour_{variant}_{mode}_s{seed}_v1"
                directory = root / "results" / name
                command = [python, str(snapshot / ("symbolic_general_bc.py" if mode == "bc" else "symbolic_general_train.py")),
                    "--sources", str(output / (variant + '.jsonl')), "--out", str(directory), "--seed", str(seed),
                    "--width", "500", "--layers", "2", "--threads", "4", "--lr", "0.0001", "--max-seconds", "1800"]
                if mode == "bc":
                    command += ["--full-batch-updates", "1000"]
                else:
                    command += ["--targets", str(output / 'no_targets.jsonl'), "--mode", "reset", "--steps", "131072",
                        "--batch", "512", "--minibatch", "64", "--epochs", "4", "--clip", "0.2", "--entropy", "0.01",
                        "--gamma", "0.95", "--gae-lambda", "0.95", "--source-portion", "1", "--max-lines", "6", "--max-scope", "4"]
                stages.append({"name": name, "command": command, "timeout_seconds": 1860, "input_sha256": hashes})
                endpoints.append({"name": name, "variant": variant, "method": mode, "seed": seed,
                    "checkpoint": str(directory / "symbolic_general_checkpoint.pt")})
    for endpoint in endpoints:
        for profile in ("greedy", "sample4"):
            name = endpoint["name"].removesuffix('_v1') + '_structured_development_' + profile + '_v1'
            command = [python, str(snapshot / 'symbolic_general_eval.py'), '--population', protocol['evaluation']['population'],
                '--out', str(root / 'results' / name), '--policy', 'model', '--checkpoint', endpoint['checkpoint'],
                '--attempts', '1' if profile == 'greedy' else '4', '--seed', '202609101935', '--max-lines', '32',
                '--max-scope', '4', '--threads', '4', '--max-seconds', '600']
            if profile == 'sample4':
                command.append('--sample')
            stages.append({'name': name, 'command': command, 'timeout_seconds': 660, 'input_sha256': {
                **hashes, protocol['evaluation']['population']: common.sha(protocol['evaluation']['population'])}})
    common.write_json(root / 'configs/detour_endpoints_v1.json', {'status': 'prospective_all_endpoints', 'endpoints': endpoints,
        'protocol_sha256': common.sha(protocol_path)})
    common.write_json(root / 'configs/detour_pipeline_v1.json', {'output': str(root / 'jobs/detour_v1'),
        'deadline_utc': '2026-09-11T01:30:00Z', 'stages': stages})
    common.write_json(root / 'reports/detour_sources_verified.json', {'status': 'passed', 'clean': clean, 'noisy': noisy,
        'all_legal_prefixes_replayed': True, 'noisy_prunes_exactly_to_clean': True, 'eval_context_overlap': 0,
        'protocol_sha256': common.sha(protocol_path), 'pipeline_sha256': common.sha(root / 'configs/detour_pipeline_v1.json')})
    print(json.dumps({'status': 'ready', 'training_runs': len(endpoints), 'development_evaluations': len(stages) - len(endpoints)}))


if __name__ == '__main__':
    main()
