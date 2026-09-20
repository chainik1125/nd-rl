"""Independent JSON-only evaluation aggregation with paired context resampling.

Usage: python analyze_evaluations.py configs/analysis_NAME.json
The plan binds data and evaluation directories. This never loads model weights.
"""
from pathlib import Path
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import numpy as np


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def ordered_scores(path, ids):
    rows = read_rows(path)
    by_id = {r['source_id']: r for r in rows}
    if len(by_id) != len(rows) or set(by_id) != set(ids):
        raise ValueError(f'Duplicate or unequal source populations: {path}')
    return np.array([by_id[i]['bits'] for i in ids], dtype=float), np.array([by_id[i]['tokens'] for i in ids], dtype=float)


def interval(values):
    return [float(v) for v in np.quantile(values, [.025, .975])]


def cluster_weights(rows, draws, seed):
    contexts = sorted({r['context_id'] for r in rows})
    indices = {c: i for i, c in enumerate(contexts)}
    groups = np.array([indices[r['context_id']] for r in rows])
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(contexts), np.full(len(contexts), 1 / len(contexts)), size=draws)
    return counts[:, groups].astype(float)


def strata(rows):
    return {label: np.array([r['length_bin'] == label for r in rows]) for label in sorted({r['length_bin'] for r in rows})}


def summarize(values, rows, weights):
    parts = strata(rows)
    estimates, boots, result = [], [], {}
    for label, mask in parts.items():
        local_weights = weights[:, mask]
        den = local_weights.sum(axis=1)
        if (den == 0).any():
            raise ValueError('Empty bootstrap stratum; increase population support or change declared resampling')
        estimate = float(values[mask].mean())
        boot = (local_weights @ values[mask]) / den
        result[label] = {'mean': estimate, 'context_bootstrap95': interval(boot), 'n_sources': int(mask.sum())}
        estimates.append(estimate)
        boots.append(boot)
    balanced = np.mean(boots, axis=0)
    result['balanced'] = {'mean': float(np.mean(estimates)), 'context_bootstrap95': interval(balanced), 'n_sources': len(rows)}
    return result, balanced


def attempts_metrics(path, ids, expected_samples):
    grouped = defaultdict(list)
    raw_lengths, pruned_lengths = [], []
    for row in read_rows(path):
        if row['reset_index'] != 0:
            raise ValueError('Headline comparisons require unaided full starts')
        # Decoder-boundary rejection occurs before either proof checker is called.
        # These explicit failures must stay in the denominator.
        if 'ok' not in row:
            if row['reward'] != 0 or row.get('reason') not in (
                    'Exactly one sampled FINAL required',
                    'Malformed sampled scratch boundary',
                    'Missing unique terminal QED'):
                raise ValueError('Unknown pre-verifier outcome schema')
            row['ok'] = False
            row['independent_ok'] = False
        if bool(row['ok']) != bool(row['independent_ok']) or int(row['reward']) != int(row['ok']):
            raise ValueError('Verifier/reward disagreement')
        if row['ok']:
            raw_lengths.append(row['n_lines'])
            pruned_lengths.append(row.get('compression', {}).get('compressed_length', row['n_lines']))
        grouped[row['source_id']].append(row)
    if set(grouped) != set(ids):
        raise ValueError('Attempt populations differ')
    rates, covered, mixed, successes = [], [], [], []
    for source in ids:
        rows = grouped[source]
        if len(rows) != expected_samples or sorted(r['attempt_index'] for r in rows) != list(range(expected_samples)):
            raise ValueError('Duplicate or missing sampling attempts')
        k = sum(r['ok'] for r in rows)
        successes.append(k)
        rates.append(k / expected_samples)
        covered.append(float(k > 0))
        mixed.append(float(0 < k < expected_samples))
    return {'attempt_success': np.array(rates), 'coverage_at_k': np.array(covered), 'mixed_group_fraction': np.array(mixed)}, {
        'samples_per_source': expected_samples,
        'successful_attempts': int(sum(successes)), 'all_attempts': len(ids) * expected_samples,
        'covered_sources': int(sum(covered)), 'mixed_groups': int(sum(mixed)),
        'valid_written_length_histogram': dict(Counter(raw_lengths)),
        'valid_compressed_length_histogram': dict(Counter(pruned_lengths))}


def linear_contrast(raw, coefficients):
    """Combine paired per-source observations before context resampling."""
    if not coefficients or any(name not in raw for name in coefficients):
        raise ValueError('A contrast needs nonempty, known model names')
    if any(isinstance(c, bool) or not np.isfinite(c) for c in coefficients.values()):
        raise ValueError('Contrast coefficients must be finite numbers')
    active = {name: float(c) for name, c in coefficients.items() if c != 0}
    if not active:
        raise ValueError('A contrast must have a nonzero coefficient')
    common = set.intersection(*(set(raw[name]) for name in active))
    if not common:
        raise ValueError('Contrast models have no common metrics')
    return {metric: sum(c * raw[name][metric] for name, c in active.items())
            for metric in sorted(common)}


def run(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    rows = read_rows(plan['data'])
    ids = [r['source_id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate population source')
    weights = cluster_weights(rows, plan.get('bootstrap_draws', 5000), plan.get('bootstrap_seed', 20260911))
    raw, summaries, receipts = {}, {}, {}
    hashes = {str(Path(plan_path).resolve()): sha(plan_path), str(Path(__file__).resolve()): sha(__file__), plan['data']: sha(plan['data'])}
    for name, spec in plan['models'].items():
        bits, tokens = ordered_scores(spec['nll'], ids)
        metrics = {'whole_proof_bits': bits, 'per_proof_bits_per_token': bits / tokens}
        hashes[spec['nll']] = sha(spec['nll'])
        receipt = {'aggregate_bits_per_token': float(bits.sum() / tokens.sum())}
        if spec.get('attempts'):
            generated, audit = attempts_metrics(spec['attempts'], ids, plan['samples'])
            metrics.update(generated)
            receipt.update(audit)
            hashes[spec['attempts']] = sha(spec['attempts'])
        raw[name] = metrics
        summaries[name] = {metric: summarize(values, rows, weights)[0] for metric, values in metrics.items()}
        receipts[name] = receipt
    comparisons = {}
    for left, right in plan.get('comparisons', []):
        common = set(raw[left]) & set(raw[right])
        comparisons[left + '_minus_' + right] = {metric: summarize(raw[left][metric] - raw[right][metric], rows, weights)[0] for metric in sorted(common)}
    contrasts = {}
    for name, coefficients in plan.get('contrasts', {}).items():
        values = linear_contrast(raw, coefficients)
        contrasts[name] = {'coefficients': coefficients,
                           'metrics': {metric: summarize(v, rows, weights)[0]
                                       for metric, v in values.items()}}
    output = {'data_sha256': sha(plan['data']), 'population_sources': len(rows),
              'population_contexts': len({r['context_id'] for r in rows}),
              'method': 'Paired canonical-context cluster percentile bootstrap; sample contexts with replacement; recompute within-length means and equally average populated length strata.',
              'uncertainty_scope': 'Measured task/context population at fixed training and decoding seeds; does not estimate training-seed uncertainty. Multiple comparisons are descriptive, without multiplicity correction.',
              'bootstrap_draws': len(weights), 'bootstrap_seed': plan.get('bootstrap_seed', 20260911),
              'models': summaries, 'receipts': receipts, 'comparisons': comparisons,
              'contrasts': contrasts, 'input_sha256': hashes}
    target = Path(plan['output'])
    if target.exists():
        raise ValueError('Refusing to replace prior analysis artifact')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'output': str(target), 'models': len(summaries), 'comparisons': len(comparisons),
                      'contrasts': len(contrasts), 'sources': len(rows)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('plan')
    run(parser.parse_args().plan)
