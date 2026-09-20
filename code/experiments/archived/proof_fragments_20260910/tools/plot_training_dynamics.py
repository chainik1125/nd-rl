"""Plot accounted main training dynamics from completed artifacts only."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'runtime/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed-index', type=int, choices=[1, 2], required=True)
    args = parser.parse_args()
    curves, hashes, receipts = {}, {str(Path(__file__).resolve()): sha(Path(__file__))}, {}
    for arm in ['selected_full', 'selected_reset', 'uniform_full', 'uniform_reset']:
        directory = ROOT / 'results' / f'rl_{arm}_seed{args.seed_index}_v1'
        result = json.loads((directory / 'result.json').read_text())
        completion = json.loads((directory / 'completed.json').read_text())
        if not completion['ok'] or completion['result_sha256'] != sha(directory / 'result.json'):
            raise ValueError('Training plot requires unchanged completed results')
        updates = [json.loads(line) for line in (directory / 'updates.jsonl').read_text().splitlines()]
        if len(updates) != result['steps']:
            raise ValueError('Incomplete update log')
        bins = defaultdict(lambda: defaultdict(float))
        for event in updates:
            index = min(39, int(event['cumulative_flop_proxy'] / 1e14))
            bins[index]['updates'] += 1
            bins[index]['zero_advantage_updates'] += event['active_advantage_tokens'] == 0
            bins[index]['retention_loss_sum'] += event['retention_loss']
        with (directory / 'groups.jsonl').open() as stream:
            for line in stream:
                group = json.loads(line)
                update = updates[group['step'] - 1]
                index = min(39, int(update['cumulative_flop_proxy'] / 1e14))
                bins[index]['groups'] += 1
                bins[index]['attempts'] += group['group_size']
                bins[index]['successes'] += group['successes']
                bins[index]['mixed_groups'] += group['mixed']
                bins[index]['supplied_prefix_groups'] += group['reset_index'] > 0
        curves[arm] = [{'proxy_bin_center_1e15': (index + .5) / 10,
                        **dict(values),
                        'training_success': values['successes'] / values['attempts'],
                        'mixed_fraction': values['mixed_groups'] / values['groups'],
                        'supplied_prefix_fraction': values['supplied_prefix_groups'] / values['groups'],
                        'zero_current_advantage_fraction': values['zero_advantage_updates'] / values['updates']}
                       for index, values in sorted(bins.items())]
        receipts[arm] = {k: result[k] for k in ['steps', 'total_flop_proxy', 'elapsed_seconds',
                                               'mixed_groups', 'n_groups', 'retention_total_examples']}
        for name in ['result.json', 'completed.json', 'updates.jsonl', 'groups.jsonl']:
            hashes[str(directory / name)] = sha(directory / name)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.7), sharex=True)
    names = {'selected_full': 'Selected / full', 'selected_reset': 'Selected / resets',
             'uniform_full': 'Matched uniform / full', 'uniform_reset': 'Matched uniform / resets'}
    colors = {'selected_full': '#007A55', 'selected_reset': '#52B788',
              'uniform_full': '#005EA8', 'uniform_reset': '#56B4E9'}
    panels = [('training_success', 'Training-state attempt success (%)'),
              ('mixed_fraction', 'Groups with both success and failure (%)'),
              ('supplied_prefix_fraction', 'Groups receiving a nonempty proof prefix (%)'),
              ('zero_current_advantage_fraction', 'Updates with zero current policy advantages (%)')]
    for axis, (metric, label) in zip(axes.flat, panels):
        for arm, rows in curves.items():
            axis.plot([row['proxy_bin_center_1e15'] for row in rows],
                      [100 * row[metric] for row in rows], color=colors[arm],
                      ls='--' if arm.endswith('reset') else '-', label=names[arm], lw=1.8)
        axis.set_ylabel(label)
        axis.set_ylim(-2, 102)
        axis.grid(color='#eeeeee', lw=.7)
    for axis in axes[-1]:
        axis.set_xlabel('Accounted training compute proxy (10¹⁵ units)')
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2, loc='lower right')
    fig.suptitle(f'Main training dynamics, online seed {20260909 + args.seed_index}', fontsize=14)
    fig.text(.06, .015, 'Bins span 10¹⁴ proxy units. Training reward includes assisted reset episodes; held-out proof success is evaluated separately.\n'
             'Zero current advantages do not remove historical Adam momentum or the retained-proof loss. Proxy units are not measured hardware FLOPs.', fontsize=9)
    fig.tight_layout(rect=[0, .065, 1, .96])
    stem = f'training_dynamics_seed{args.seed_index}_v1'
    target = ROOT / 'figures'
    outputs = []
    for extension in ['png', 'svg', 'pdf']:
        path = target / f'{stem}.{extension}'
        fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
        outputs.append(path)
    plt.close(fig)
    receipt_path = target / f'{stem}.json'
    receipt_path.write_text(json.dumps({'curves': curves, 'run_receipts': receipts,
                                        'input_sha256': hashes,
                                        'figure_sha256': {str(p): sha(p) for p in outputs}}, indent=2) + '\n')
    print(json.dumps({'receipt': str(receipt_path), 'figures': len(outputs)}))


if __name__ == '__main__':
    main()
