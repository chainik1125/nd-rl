"""Predeclared paired analysis of the development-only seed-detour diagnostic."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import common
from analyze_lm import paired_context_bootstrap
from analyze_symbolic import strata
import plot_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = common.ROOT
    protocol_path = root / 'configs/detour_protocol_v1.json'
    if common.sha(protocol_path) != config['protocol_sha256']:
        raise ValueError('Detour protocol changed')
    endpoint_path = root / 'configs/detour_endpoints_v1.json'
    endpoints = json.loads(endpoint_path.read_text())['endpoints']
    inputs = {str(p.resolve()): common.sha(p) for p in (args.config, protocol_path, endpoint_path, Path(__file__))}
    training, initial_by_seed = {}, {}
    for endpoint in endpoints:
        directory = Path(endpoint['checkpoint']).parent
        summary_path = directory / 'symbolic_general_train_summary.json'
        summary = json.loads(summary_path.read_text())
        if summary['status'] != 'complete':
            raise ValueError('Incomplete training endpoint')
        initial_path = directory / 'symbolic_general_initial.pt'
        initial = torch.load(initial_path, map_location='cpu', weights_only=False)['state_dict']
        prior = initial_by_seed.setdefault(endpoint['seed'], initial)
        if prior.keys() != initial.keys() or not all(torch.equal(prior[k], initial[k]) for k in prior):
            raise ValueError('Within-seed initial policies or critics are not identical')
        for path in (summary_path, initial_path, Path(endpoint['checkpoint'])):
            inputs[str(path)] = common.sha(path)
        training[endpoint['name']] = {k: v for k, v in summary.items() if k != 'config'}
    audit_path = root / 'data/structured_v1/audit/development.jsonl'
    audit = {r['source_id']: r for r in common.read_jsonl(audit_path)}
    all_strata = strata(audit)
    for length in sorted({r['n_lines'] for r in audit.values()}):
        all_strata[f'reference_lines:{length}'] = sorted(k for k, r in audit.items() if r['n_lines'] == length)
    profiles = {}
    for profile in config['profiles']:
        reports = {}
        for condition in config['conditions']:
            for seed in config['seed_indices']:
                label = f'{condition}_s{seed}'
                path = root / 'reports' / f'audit_detour_{label}_structured_development_{profile}_v1.json'
                report = json.loads(path.read_text())
                if report['status'] != 'complete' or set(report['rows']) != set(audit):
                    raise ValueError('Incomplete replay-audited detour evaluation')
                for artifact, digest in report['input_sha256'].items():
                    if common.sha(artifact) != digest:
                        raise ValueError('Replay-audited artifact changed')
                expected_checkpoint = root / 'results' / f'detour_{label}_v1' / 'symbolic_general_checkpoint.pt'
                if report['checkpoint_sha256'] != common.sha(expected_checkpoint):
                    raise ValueError('Evaluation is assigned to the wrong training endpoint')
                inputs[str(path)] = common.sha(path)
                reports[label] = report
        inference_profiles = [r['profile'] for r in reports.values()]
        if len({common.digest(p) for p in inference_profiles}) != 1:
            raise ValueError('Compared inference profiles differ')
        effects = {}
        for name, coefficients in config['contrasts'].items():
            effects[name] = {}
            for stratum, ids in all_strata.items():
                metrics = {}
                for metric in ('first', 'any', 'any_pruned_over6'):
                    values = {f'seed{seed}': np.array([sum(weight * reports[f'{condition}_s{seed}']['rows'][sid][metric]
                        for condition, weight in coefficients.items()) for sid in ids], dtype=float)
                        for seed in config['seed_indices']}
                    values['mean_across_seeds'] = np.mean(list(values.values()), axis=0)
                    metrics[metric] = {key: paired_context_bootstrap(vector, [audit[sid]['context_id'] for sid in ids],
                        config['bootstrap_replicates'], config['bootstrap_seed']) for key, vector in values.items()}
                effects[name][stratum] = metrics
        profiles[profile] = {'summaries': {k: r['summaries'] for k, r in reports.items()}, 'contrasts': effects,
                             'inference_profile': inference_profiles[0]}
    report = {'status': 'complete', 'profiles': profiles, 'training': training, 'initial_weights_match_within_seed': True,
        'config': config, 'input_sha256': inputs,
        'scope': 'Post hoc motivated, prospectively configured development diagnostic. No sealed final evaluation was performed for these models. PPO and cloning are separately budgeted.'}
    common.write_json(args.output, report)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plot_results.style()
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    for ax, (profile, title) in zip(axes, [('greedy', 'One greedy attempt'), ('sample4', 'Any of four sampled attempts')]):
        for method, color, label in [('reset', '#009E73', 'PPO backward starts'), ('bc', '#0072B2', 'Behavior cloning')]:
            rows = []
            for index, seed in enumerate(config['seed_indices']):
                values = [profiles[profile]['summaries'][f'{variant}_{method}_s{seed}']['all']['any_rate'] for variant in ('clean', 'noisy')]
                rows.append(values)
                ax.plot([0, 1], values, color=color, marker='o' if index == 0 else 's', alpha=.55, lw=.8)
            ax.plot([0, 1], np.mean(rows, axis=0), color=color, lw=2, label=label)
        ax.set_xticks([0, 1], ['Clean: 4 lines', 'Two unused steps: 6 lines'])
        ax.set_ylim(-.01, 1.01)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title(title)
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Verified coverage of 320 development theorems')
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle('Same two seed theorems, with or without unnecessary proof steps', fontsize=11)
    fig.tight_layout()
    plot_results.save(fig, root / 'figures/detour_development_v1', [args.output, Path(__file__)],
        'The same two theorem prompts have clean four-line or distracted six-line demonstrations. Lines connect conditions within the two training seeds; thick lines show their mean. Each model is tested on the same 320 structured development theorems, with zero supplied proof. PPO uses 131072 sampled actions; cloning uses 1000 full-batch updates, or 6000 clean and 10000 distracted action presentations. This is a development diagnostic, not a final test or a compute-matched PPO-versus-cloning comparison.')
    print(json.dumps({'status': 'complete', 'output': str(args.output), 'counts': {p: {k: v['all']['any'] for k, v in x['summaries'].items()} for p, x in profiles.items()}}))


if __name__ == '__main__':
    main()
