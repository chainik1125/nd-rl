"""Declare main learning-curve development evaluation, without opening final data.

This creates a prospective recipe, analysis plan and pipeline plan. Run
prepare_pipeline.py only after every declared checkpoint exists.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/data/users/dmitry/simplex-research/.venv/bin/python'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_new(path, content):
    with path.open('x') as stream:
        json.dump(content, stream, indent=2)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed-index', type=int, choices=[1, 2], required=True)
    parser.add_argument('--deadline-utc', required=True)
    args = parser.parse_args()
    seed_index = args.seed_index
    name = f'main_development_seed{seed_index}_v1'
    recipe_path = ROOT / 'configs' / f'recipe_{name}.json'
    pipeline_path = ROOT / 'configs' / f'pipeline_{name}.json'
    analysis_path = ROOT / 'configs' / f'analysis_{name}.json'
    release_path = ROOT / 'configs' / f'release_{name}.json'
    data = ROOT / 'data/released_v1/confirmation.jsonl'
    source = ROOT / 'snapshots/confirmation_v1'
    release_source = ROOT / 'snapshots/confirmation_release_recipe_v1'
    declared_inputs = [
        ROOT / f'configs/pipeline_factorial_seed{seed_index}_v1.json',
        ROOT / 'configs/frozen_selectors_v1.json',
        ROOT / 'configs/calibration_gate_and_rl_budget_v1.json',
        ROOT / 'data/released_v1/manifest.json',
        ROOT / 'data/bank_selected_v1/manifest.json',
        ROOT / 'data/bank_uniform_v1/manifest.json',
        ROOT / 'reports/analysis_protocol_v1.md',
        ROOT / 'theory/final_prefix_exposure_analysis_rule_v1.json',
        ROOT / 'tools/analyze_evaluations.py',
        Path(__file__).resolve(),
    ]
    models, checkpoint_paths, evaluation_commands = {}, [], []
    arms = ['selected_full', 'selected_reset', 'uniform_reset', 'uniform_full']
    checkpoints = {'f05': 'flops_500000000000000.pt',
                   'f10': 'flops_1000000000000000.pt',
                   'f20': 'flops_2000000000000000.pt', 'f40': 'final.pt'}
    for arm in arms:
        for milestone, filename in checkpoints.items():
            model_name = f'{arm}_s{seed_index}_{milestone}'
            checkpoint = ROOT / 'results' / f'rl_{arm}_seed{seed_index}_v1' / filename
            output = ROOT / 'results' / f'confirmation_{model_name}_v1'
            checkpoint_paths.append(str(checkpoint))
            models[model_name] = {'nll': str(output / 'whole_proof_nll.jsonl'),
                                  'attempts': str(output / 'attempts.jsonl')}
            evaluation_commands.append([
                PYTHON, str(source / 'fragment_experiment.py'), 'evaluate',
                '--checkpoint', str(checkpoint), '--data', str(data),
                '--release', str(release_path), '--output', str(output),
                '--start', 'full', '--samples', '8', '--temperature', '0.8',
                '--max-new-tokens', '512', '--batch-size', '64', '--microbatch', '16',
                '--max-logprob-error', '0.001', '--audit-likelihood',
                '--device', 'cuda', '--precision', 'fp32', '--seed', '20260910',
                '--cpu-threads', '4'])
    comparisons, contrasts = [], {}
    for milestone in checkpoints:
        sf, sr, ur, uf = [f'{a}_s{seed_index}_{milestone}' for a in arms]
        comparisons += [[sr, sf], [ur, uf], [sr, ur], [sf, uf]]
        contrasts[f'interaction_{milestone}'] = {sr: 1, sf: -1, ur: -1, uf: 1}
        contrasts[f'selection_main_effect_{milestone}'] = {sr: .5, sf: .5, ur: -.5, uf: -.5}
        contrasts[f'reset_main_effect_{milestone}'] = {sr: .5, ur: .5, sf: -.5, uf: -.5}
    for model_name in models:
        comparisons.append([model_name, 'adapt'])
    models['adapt'] = {
        'nll': str(ROOT / 'results/confirmation_adapt_v1/whole_proof_nll.jsonl'),
        'attempts': str(ROOT / 'results/confirmation_adapt_v1/attempts.jsonl')}
    analysis = {'data': str(data), 'samples': 8, 'bootstrap_draws': 5000,
                'bootstrap_seed': 20260911, 'models': models,
                'comparisons': comparisons, 'contrasts': contrasts,
                'output': str(ROOT / 'reports' / f'{name}.json'),
                'status': 'prospectively_declared_development_analysis',
                'note': 'All four milestones retained. Final endpoint is the 4e15 crossing; its duplicate milestone weights are not evaluated twice. NLL at T=1 is descriptive; verified sampling uses T=0.8. Interactions and main effects combine paired source outcomes before context resampling. No checkpoint selection by development outcomes is planned.'}
    write_new(analysis_path, analysis)
    declared_inputs.append(analysis_path)
    recipe = {'frozen': True, 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'role': 'confirmation', 'no_further_training': False,
              'data': str(data), 'population_manifest': str(ROOT / 'data/released_v1/manifest.json'),
              'checkpoint_paths': checkpoint_paths, 'output': str(release_path),
              'decoding': {'start': 'full', 'samples': 8, 'temperature': .8,
                           'max_new_tokens': 512, 'seed': 20260910},
              'prospective_input_sha256': {str(p): sha(p) for p in declared_inputs},
              'recipe_note': 'Main architecture, banks, budgets, seeds, all curve checkpoints, contrasts and decoding fixed before main heldout predictions. Confirmation is development data already exposed by calibration. Final confirmation remains unopened.'}
    write_new(recipe_path, recipe)
    pipeline = {'name': name,
                'question': 'Measure the full predeclared learning curves for the 2x2 selection/reset experiment on development tasks, including paired main effects and interaction.',
                'expected_seconds': 600, 'timeout_seconds': 1800,
                'deadline_utc': args.deadline_utc,
                'source_snapshots': [str(source), str(release_source)],
                'inputs': [str(p) for p in declared_inputs] + [str(recipe_path), str(data)] + checkpoint_paths,
                'commands': [[PYTHON, str(release_source / 'release_checkpoints.py'), str(recipe_path)]] + evaluation_commands,
                'status': 'declared_waiting_for_all_checkpoints',
                'final_confirmation_opened': False}
    write_new(pipeline_path, pipeline)
    print(json.dumps({'pipeline': str(pipeline_path), 'recipe': str(recipe_path),
                      'analysis': str(analysis_path), 'new_evaluations': len(evaluation_commands)}))


if __name__ == '__main__':
    main()
