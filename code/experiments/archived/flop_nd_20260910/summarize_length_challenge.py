"""Summarize every prespecified frozen-policy long-projection evaluation."""
from collections import Counter
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import common
from nd_environment import strict_complete
from proof_diagnostics import compress_proof


def main():
    root = common.ROOT
    config_path = root / 'configs/length_challenge_v1.json'
    config = json.loads(config_path.read_text())
    job = json.loads((root / 'jobs/length_challenge_v1/result.json').read_text())
    audit_job = json.loads((root / 'jobs/length_challenge_audits_v2/result.json').read_text())
    if job['status'] not in ('completed', 'completed_with_failures') or audit_job['status'] not in ('completed', 'completed_with_failures'):
        raise ValueError('Length challenge has not finished')
    records, missing, inputs, witnesses = [], [], {str(config_path): common.sha(config_path)}, []
    for task in config['tasks']:
        name = task['name']
        path = root / 'reports' / ('audit_' + name + '.json')
        result_path = Path(task['result_path'])
        outcome = json.loads(result_path.read_text()) if result_path.exists() else {}
        if not path.exists():
            missing.append({'name': name, 'job_status': job['tasks'].get(name, {}).get('status'),
                            'result_status': outcome.get('status'), 'theorems_attempted': outcome.get('theorems_attempted'),
                            'unaudited_success_count': outcome.get('theorems_with_any_success')})
            continue
        audit = json.loads(path.read_text())
        if audit['status'] != 'complete':
            raise ValueError('Invalid audit status')
        for source, digest in audit['input_sha256'].items():
            if common.sha(source) != digest:
                raise ValueError('Audited artifact changed: ' + source)
        assert audit['summaries']['all']['theorems'] == 16
        length = int(name.split('_length')[1].split('_')[0])
        policy = name.split('_length')[0].removeprefix('minimal_')
        records.append({'name': name, 'policy': policy, 'reference_lines': length, 'max_lines': length + 8,
                        'elapsed_seconds': outcome['elapsed_seconds'], 'summaries': audit['summaries'],
                        'checkpoint_sha256': audit['checkpoint_sha256'], 'audit': str(path)})
        witnesses.extend({**w, 'name': name, 'policy': policy} for w in audit['witnesses'])
        inputs[str(path)] = common.sha(path)
    result = {'status': 'complete' if not missing else 'partial', 'config_sha256': common.sha(config_path),
              'scope': config['interpretation'], 'records': records, 'missing_complete_audits': missing,
              'input_sha256': inputs, 'summary_code_sha256': common.sha(__file__),
              'notes': ['Only 8 theorems per family/length cell; both training seeds are shown separately.',
                        'No checkpoint or policy changed during this additional challenge.',
                        'Every complete evaluation has all actions and certificates replayed; missing coverage is not converted to failure.',
                        'The six trained minimal endpoints use the same two six-line seed problems; cloning and PPO budgets differ.',
                        'Uniform is one random attempt; all other policies use greedy action choice.']}
    if witnesses:
        # An illustrative maximum-length certificate, explicitly post-hoc.
        witness = sorted(witnesses, key=lambda w: (-w['pruned_lines'], w['name'], w['source_id']))[0]
        assert strict_complete(witness['prompt'], witness['pruned_body'])['ok']
        compressed = compress_proof(witness['prompt'], witness['body'])
        assert compressed['compressed_length'] == witness['compressed_lines']
        common.write_json(root / 'reports/witness_length_challenge_v1.json', witness)
        (root / 'reports/witness_length_challenge_v1.txt').write_text(witness['prompt'] + '\n\n' + witness['pruned_body'].replace(' ; ', ' ;\n') + '\n')
        result['illustrative_maximum_witness'] = {k: witness[k] for k in ('name', 'source_id', 'family', 'target_atom', 'raw_lines', 'pruned_lines', 'compressed_lines', 'raw_tokens', 'pruned_tokens')}
    common.write_json(root / 'reports/analysis_length_challenge_v1.json', result)
    text = ['# Additional proof-length challenge', '', config['interpretation'], '',
            'All counts are successful original theorem prompts out of 16 at each reference length (eight atomic and eight compound distractor cases). Two examples of each target atom P/Q/R/S occur in every family/length cell. The training demonstrations remain two six-line proofs.', '',
            '| Frozen policy | 33-line references | 65-line references | 129-line references |', '|---|---:|---:|---:|']
    by_key = {(r['policy'], r['reference_lines']): r for r in records}
    policies = [f'{mode}_s{seed}' for mode in ('initial', 'full', 'reset', 'bc') for seed in (4401, 4402)] + ['uniform']
    for policy in policies:
        cells = [str(by_key[policy, length]['summaries']['all']['any']) if (policy, length) in by_key else 'Incomplete; see JSON' for length in (33, 65, 129)]
        text.append('| ' + policy + ' | ' + ' | '.join(cells) + ' |')
    text += ['', 'Reference certificates and found certificate lengths are checked separately. These data test depth extrapolation of a projection procedure; they do not establish a new mathematical strategy or an unrestricted proof lower bound. The challenge was selected after earlier development results and uses a small population. PPO and cloning training budgets differ.', '',
             'The [full JSON](analysis_length_challenge_v1.json) records all families, target atoms, inference caps, candidate-scoring counts, timing, checkpoints and missing audits. The [illustrative certificate](witness_length_challenge_v1.txt), when present, is selected after evaluation by maximum pruned length and then deterministic name/identity order.']
    (root / 'reports/length_challenge.md').write_text('\n'.join(text) + '\n')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    colors = {'reset': '#0072B2', 'bc': '#D55E00', 'full': '#555555', 'initial': '#999999', 'uniform': '#009E73'}
    for ax, family in zip(axes, ('projection_atomic', 'projection_compound')):
        for mode in ('reset', 'bc', 'full', 'initial', 'uniform'):
            seeds = (None,) if mode == 'uniform' else (4401, 4402)
            for seed in seeds:
                policy = mode if seed is None else f'{mode}_s{seed}'
                values = [(length, by_key[policy, length]['summaries']['family:' + family]['any'] / 8) for length in (33, 65, 129) if (policy, length) in by_key]
                if not values:
                    continue
                xs, ys = zip(*values)
                label = {'reset': 'Curriculum PPO', 'bc': 'Cloning', 'full': 'Full-start PPO', 'initial': 'Untrained', 'uniform': 'Uniform'}[mode]
                ax.plot(xs, ys, marker='o' if seed != 4402 else 's', color=colors[mode], linestyle='-' if seed != 4402 else '--',
                        linewidth=1.2, markersize=4, alpha=.8, label=label if seed in (None, 4401) else None)
        ax.set_title('Atomic distractors' if family == 'projection_atomic' else 'Compound distractors')
        ax.set_xticks((33, 65, 129))
        ax.set_xlabel('Reference certificate lines')
        ax.set_ylim(-.025, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Greedy verified coverage (8 theorems per point)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, -.08), ncol=3, frameon=False)
    fig.suptitle('Frozen policies after two six-line demonstrations')
    fig.tight_layout()
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(root / 'figures' / ('length_challenge_v1.' + extension), bbox_inches='tight', dpi=180)
    plt.close(fig)
    common.write_json(root / 'figures/length_challenge_v1.json', {'description': 'Additional post-development projection challenge; solid circles and dashed squares show separate training seeds. All trained and untrained controls retained; overlapping zero traces are shown in the numeric table. Cloning and PPO training budgets differ. Each family/length cell has only eight theorems.',
                       'input_sha256': {str(root / 'reports/analysis_length_challenge_v1.json'): common.sha(root / 'reports/analysis_length_challenge_v1.json')}, 'code_sha256': common.sha(__file__)})
    print(json.dumps({'status': result['status'], 'complete_evaluations': len(records), 'missing': missing, 'witness': result.get('illustrative_maximum_witness')}))


if __name__ == '__main__':
    main()
