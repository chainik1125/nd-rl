"""Plot completed final retention-exposure and online-admission comparisons."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import common


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, default=common.ROOT / 'reports/analysis_primary_final_greedy_v1.json')
    parser.add_argument('--output-prefix', type=Path, default=common.ROOT / 'figures/lm_followup_controls_final_v2')
    args = parser.parse_args()
    report = json.loads(args.analysis.read_text())
    if report.get('status') != 'completed':
        raise ValueError('Only complete final analyses may enter this figure')
    summaries = report['summaries']
    mapping_path = common.ROOT / 'configs/retention_endpoint_mapping_v1.json'
    mapping = json.loads(mapping_path.read_text())['mapping'] if mapping_path.exists() else {}
    denominator = summaries['baseline']['all']['theorems']

    def coverage(name):
        row = summaries[name]['all']
        if row['theorems'] != denominator:
            raise ValueError('Different theorem populations')
        return row['first'] / denominator

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax in axes:
        ax.set_ylim(0, .35)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis='y', alpha=.18)
        ax.axhline(coverage('baseline'), color='#999999', linestyle=':', linewidth=1)
    conditions = ('relative_full', 'relative_reset', 'ppo_full', 'ppo_reset')
    pairs = []
    for index, condition in enumerate(conditions):
        for seed, offset, marker in ((1, -.09, 'o'), (2, .09, 's')):
            endpoint = f'main_{condition}_seed{seed}_v1'
            control = mapping.get(endpoint)
            if control not in summaries:
                continue
            x = index + offset
            before, after = coverage(control), coverage(endpoint)
            axes[0].plot([x, x], [before, after], color='#999999', linewidth=1.2)
            axes[0].plot(x, before, marker=marker, markerfacecolor='white', markeredgecolor='#666666', markersize=7)
            axes[0].plot(x, after, marker=marker, color='#0072B2', markersize=6)
            pairs.append({'seed': seed, 'condition': condition, 'rl_endpoint': endpoint, 'retention_endpoint': control,
                          'rl_coverage': after, 'retention_coverage': before})
    axes[0].set_title('RL versus matched short-proof fitting exposure')
    axes[0].set_xticks(range(4), ('Relative\nfull', 'Relative\nbackward', 'PPO\nfull', 'PPO\nbackward'))
    axes[0].set_ylabel(f'Unaided greedy coverage ({denominator} final statements)')
    if pairs:
        axes[0].legend(handles=[Line2D([], [], marker='o', color='none', markerfacecolor='white', markeredgecolor='#666666', label='Short-proof fitting only'),
                               Line2D([], [], marker='o', color='#0072B2', linestyle='none', label='RL plus short-proof fitting')],
                       loc='upper right', frameon=False, fontsize=9)
    else:
        axes[0].text(.5, .65, 'Retention controls were not released.\nSee the recorded resource/failure decision.', ha='center', transform=axes[0].transAxes)
    online_rows = []
    styles = [('parent', -.16, '#777777', 'D', 'Primary PPO/backward endpoint'),
              ('off', 0, '#D55E00', 'o', 'Continuation: admission off'),
              ('on', .16, '#009E73', 'o', 'Continuation: admission on')]
    for seed in (1, 2):
        coordinates = []
        for mode, offset, color, marker, label in styles:
            name = f'main_ppo_reset_seed{seed}_v1' if mode == 'parent' else f'online_{mode}_seed{seed}_v1'
            if name not in summaries:
                continue
            y = coverage(name)
            axes[1].plot(seed + offset, y, marker=marker, color=color, markersize=7, linestyle='none', label=label if seed == 1 else None)
            online_rows.append({'seed': seed, 'mode': mode, 'endpoint': name, 'coverage': y})
            if mode != 'parent':
                coordinates.append((seed + offset, y))
        if len(coordinates) == 2:
            axes[1].plot(*zip(*coordinates), color='#999999', linewidth=1, zorder=0)
    axes[1].set_title('Curricula from self-discovered target proofs')
    axes[1].set_xticks((1, 2), ('Training seed 1', 'Training seed 2'))
    axes[1].set_xlim(.55, 2.45)
    axes[1].legend(frameon=False, fontsize=9, loc='upper right')
    if not any(r['mode'] == 'on' for r in online_rows):
        axes[1].text(.5, .65, 'Online continuations were not released.\nSee the recorded resource/failure decision.', ha='center', transform=axes[1].transAxes)
    fig.text(.5, -.01, 'Dotted line: initial LM. Left: circle = seed 1; square = seed 2. On/off continuations share the same restart and training settings.',
             ha='center', fontsize=9)
    if len(pairs) != 8 or len(online_rows) != 6:
        raise ValueError('The final report requires every declared paired control and continuation')
    fig.tight_layout()
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(prefix.with_suffix('.' + ext), bbox_inches='tight', dpi=180)
    plt.close(fig)
    inputs = [args.analysis, *([mapping_path] if mapping_path.exists() else [])]
    common.write_json(prefix.with_suffix('.json'), {'status': 'complete', 'theorems': denominator, 'retention_pairs': pairs,
        'online_endpoints': online_rows, 'input_sha256': {str(p.resolve()): common.sha(p) for p in inputs},
        'code_sha256': common.sha(__file__), 'outputs_sha256': {ext: common.sha(prefix.with_suffix('.' + ext)) for ext in ('png', 'pdf', 'svg')},
        'description': 'Final unaided greedy LM evaluation. Retention controls match exact supervised examples and optimizer exposures but not total compute. Online on/off share the same continuation package; parent comparisons additionally change GAE, EMA, entropy, normalization, optimizer and compute. All declared available endpoints are retained. No intervals over two training seeds are inferred from their spread.'})
    print(json.dumps({'output': str(prefix), 'retention_pairs': len(pairs), 'online_endpoints': len(online_rows)}))


if __name__ == '__main__':
    main()
