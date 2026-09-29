"""Bind retention-only controls to completed main-arm exposure schedules."""
import argparse
import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/data/users/dmitry/simplex-research/.venv/bin/python'
MILESTONES = {'f05': 500000000000000, 'f10': 1000000000000000,
              'f20': 2000000000000000, 'f40': 4000000000000000}


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed-index', type=int, choices=[1, 2], required=True)
    parser.add_argument('--deadline-utc', required=True)
    args = parser.parse_args()
    name = f'retention_control_seed{args.seed_index}_v1'
    source = ROOT / 'snapshots/retention_mining_v1'
    inputs = [ROOT / 'results/adapt_v1/final.pt', ROOT / 'data/released_v1/retention.jsonl',
              ROOT / 'data/released_v1/manifest.json', ROOT / 'theory/retention_control_review_v1.md',
              Path(__file__).resolve()]
    requested_steps, mapping = set(), {}
    seed = 20260909 + args.seed_index
    for arm in ['selected_full', 'selected_reset', 'uniform_reset', 'uniform_full']:
        run_name = f'rl_{arm}_seed{args.seed_index}_v1'
        directory = ROOT / 'results' / run_name
        result = json.loads((directory / 'result.json').read_text())
        completion = json.loads((directory / 'completed.json').read_text())
        manifest = json.loads((directory / 'manifest.json').read_text())
        if not completion.get('ok') or result['stop_reason'] != 'flop_proxy_ceiling':
            raise ValueError('A main arm has not completed its compute ceiling')
        if manifest['args']['seed'] != seed or result['max_flop_proxy'] != MILESTONES['f40']:
            raise ValueError('Main seed/budget differs from the declared control')
        updates = [json.loads(line) for line in (directory / 'updates.jsonl').read_text().splitlines()]
        if len(updates) != result['steps'] or [r['step'] for r in updates] != list(range(1, result['steps'] + 1)):
            raise ValueError('Main update log is incomplete')
        for label, threshold in MILESTONES.items():
            event = next(r for r in updates if r['cumulative_flop_proxy'] >= threshold)
            step = event['step']
            requested_steps.add(step)
            mapping[f'{arm}_s{args.seed_index}_{label}'] = {
                'main_run': run_name, 'main_step': step,
                'main_actual_flop_proxy': event['cumulative_flop_proxy'],
                'retention_examples': 16 * step,
                'control_checkpoint': str(ROOT / 'results' / name / f'step_{step:06d}.pt'),
                'matching': 'Same retention data, ordered draws, optimizer initialization and number of retained examples; total compute is lower and Adam history differs after removing the policy-loss term.'}
        inputs += [directory / filename for filename in ['manifest.json', 'completed.json', 'result.json', 'updates.jsonl']]
    comparison_path = ROOT / 'configs' / f'comparisons_{name}.json'
    comparison = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'seed_index': args.seed_index, 'seed': seed, 'exposure_controls': mapping,
                  'total_compute_control': str(ROOT / 'results' / name / 'final.pt'),
                  'total_compute_ceiling': MILESTONES['f40'],
                  'note': 'One retention-only trajectory supplies all declared exposure checkpoints and a separate equal-total-proxy endpoint. This removes the policy-loss term; it is not an additive causal decomposition of the joint Adam trajectory. Loss scale 0.1 can largely cancel in standalone Adam apart from epsilon and clipping.'}
    write_new(comparison_path, comparison)
    inputs.append(comparison_path)
    main_manifest = ROOT / 'results' / f'rl_selected_full_seed{args.seed_index}_v1/manifest.json'
    command = [PYTHON, str(source / 'retention_only.py'),
               '--checkpoint', str(ROOT / 'results/adapt_v1/final.pt'),
               '--retention', str(ROOT / 'data/released_v1/retention.jsonl'),
               '--main-manifest', str(main_manifest), '--seed', str(seed),
               '--save-at-steps', *[str(s) for s in sorted(requested_steps)],
               '--max-flop-proxy', '4e15', '--batch-size', '16',
               '--lr', '0.00005', '--retention-weight', '0.1',
               '--output', str(ROOT / 'results' / name),
               '--device', 'cuda', '--precision', 'fp32', '--cpu-threads', '4']
    plan = {'name': name,
            'question': 'How does unaided proof performance change when policy loss is removed, at matched retention exposures and separately at the same total training-compute proxy?',
            'expected_seconds': 600, 'timeout_seconds': 1800,
            'deadline_utc': args.deadline_utc, 'source_snapshots': [str(source)],
            'inputs': [str(p) for p in inputs], 'commands': [command],
            'status': 'frozen_ready', 'exposure_checkpoint_count': len(requested_steps),
            'training_seed_role': 'Same online RNG seed as the corresponding main trajectory; common adapted policy.'}
    pipeline_path = ROOT / 'configs' / f'pipeline_{name}.json'
    write_new(pipeline_path, plan)
    print(json.dumps({'pipeline': str(pipeline_path), 'comparison_map': str(comparison_path),
                      'exposure_steps': sorted(requested_steps)}))


if __name__ == '__main__':
    main()
