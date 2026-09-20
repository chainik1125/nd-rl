"""Render measured endpoint tables from the complete replay-audited analyses.

This is report assembly, not a new statistical analysis or endpoint selector.
Missing final analyses remain explicit; --require-final rejects any missing one.
"""
import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import common


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--require-final', action='store_true')
    p.add_argument('--output-prefix', type=Path, default=common.ROOT / 'reports/endpoint_tables')
    args = p.parse_args()
    expected = [(study, population, split, profile) for study in ('lm', 'general', 'minimal')
                for population in ('primary', 'structured') for split in ('development', 'final')
                for profile in ('greedy', 'sample4')]
    documents, missing, rows, differences, inputs = [], [], [], [], {}
    for study, population, split, profile in expected:
        if study == 'lm' and split == 'development':
            continue  # The main preregistered LM paired analysis uses final endpoints.
        filename = f'analysis_{"" if study == "lm" else study + "_"}{population}_{split}_{profile}_v1.json'
        path = common.ROOT / 'reports' / filename
        label = dict(study=study, population=population, split=split, profile=profile)
        if not path.exists():
            missing.append({**label, 'path': str(path)})
            continue
        report = json.loads(path.read_text())
        if report.get('status') not in ('complete', 'completed'):
            raise ValueError('Incomplete analysis: ' + str(path))
        inputs[str(path)] = common.sha(path)
        documents.append(f'## {study.upper() if study == "lm" else study.capitalize()} · {population} · {split} · {profile}\n\n'
                         f'Source: [{filename}]({filename}). Counts refer to the same theorem population within this table.\n\n'
                         '| Frozen endpoint | Theorems | First succeeds | Any succeeds | Pruned >6 succeeds | Longest pruned certificate |\n'
                         '|---|---:|---:|---:|---:|---:|')
        for endpoint, strata in report['summaries'].items():
            for stratum, metrics in strata.items():
                rows.append({**label, 'endpoint': endpoint, 'stratum': stratum, **metrics, 'source': str(path)})
            m = strata['all']
            documents.append(f'| {endpoint} | {m["theorems"]} | {m["first"]} | {m["any"]} | {m["any_pruned_over6"]} | {m["maximum_pruned"]} |')
        documents.append('')
        for name, strata in report['contrasts'].items():
            for stratum, metrics in strata.items():
                for metric, seeds in metrics.items():
                    for seed, estimate in seeds.items():
                        differences.append({**label, 'contrast': name, 'stratum': stratum, 'metric': metric,
                                            'seed': seed, **estimate, 'source': str(path)})
        documents.append('Paired differences in any-success coverage, percentage points; intervals resample canonical contexts conditional on the trained seeds.\n\n'
                         '| Comparison | Mean difference | Context-bootstrap 95% interval |\n|---|---:|---:|')
        for name, strata in report['contrasts'].items():
            estimate = strata['all']['any'].get('mean_across_seeds')
            if estimate is None:
                continue
            low, high = estimate['ci95_context_bootstrap']
            documents.append(f'| {name} | {100 * estimate["estimate"]:.2f} | [{100 * low:.2f}, {100 * high:.2f}] |')
        documents.append('')
    missing_final = [m for m in missing if m['split'] == 'final']
    if args.require_final and missing_final:
        raise ValueError('Missing final analyses: ' + ', '.join(m['path'] for m in missing_final))
    result = {'status': 'complete' if not missing else 'partial', 'assembled_utc': datetime.now(timezone.utc).isoformat(),
              'final_analyses_complete': not missing_final, 'missing': missing, 'evaluations': rows, 'contrasts': differences,
              'input_sha256': inputs, 'assembly_code_sha256': common.sha(__file__),
              'interpretation': 'Exact extraction of all declared endpoints from complete audited analyses. Symbolic legal-action and token-generation interfaces have different budgets. Any success is the observed fixed number of attempts, not a pass@k extrapolation. Certificate length is not minimum theorem difficulty.'}
    prefix = args.output_prefix
    common.write_json(prefix.with_suffix('.json'), result)
    fields = ['study', 'population', 'split', 'profile', 'endpoint', 'stratum', 'theorems', 'first', 'any', 'any_pruned_over6', 'maximum_pruned', 'source']
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    prefix.with_suffix('.csv').write_text(stream.getvalue())
    intro = '# Endpoint tables\n\nThese tables are assembled directly from replay-audited result files. Both trained seeds and all frozen endpoints are retained. '
    intro += 'Greedy uses one attempt; sample4 uses four sampled attempts. Uniform is always a random action policy. '
    intro += 'The general symbolic structured population and the LM structured population share the 7/9/13-line subset; minimal symbolic evaluation uses the full 7/9/13/17/25-line population. '
    intro += 'Cloning, symbolic PPO and token-LM training budgets differ. See the method guide before comparing interfaces.\n\n'
    if missing:
        intro += '**Assembly is partial. Missing analyses:** ' + ', '.join(f'{m["study"]}/{m["population"]}/{m["split"]}/{m["profile"]}' for m in missing) + '.\n\n'
    prefix.with_suffix('.md').write_text(intro + '\n'.join(documents) + '\n')
    print(json.dumps({'status': result['status'], 'evaluations': len(rows), 'contrasts': len(differences), 'missing_final': len(missing_final), 'output': str(prefix)}))


if __name__ == '__main__':
    main()
