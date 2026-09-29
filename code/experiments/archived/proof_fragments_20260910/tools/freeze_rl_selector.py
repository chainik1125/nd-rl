"""Freeze the prespecified SHORT-RL selector from selection-only NLL outputs."""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / 'configs/rl_calibration_design_v1.json').is_file()
            and (parent / 'data/released_v1/manifest.json').is_file())
BINS = ['2-6', '7-9', '10-12']


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_scores(path, expected_ids=None):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    ids = [row['source_id'] for row in rows]
    if len(rows) != 384 or len(set(ids)) != 384 or (expected_ids is not None and set(ids) != expected_ids):
        raise ValueError('Selection scores must cover all 384 distinct frozen sources')
    if any(row['role'] != 'selection' or row['score_temperature'] != 1. or row['start'] != 'full'
           or not math.isfinite(row['bits']) or row['tokens'] <= 0 for row in rows):
        raise ValueError('Scores differ from the declared whole-reference selection objective')
    grouped = {label: [row['bits'] for row in rows if row['length_bin'] == label] for label in BINS}
    if any(len(values) != 128 for values in grouped.values()):
        raise ValueError('Selection length-bin support differs')
    by_bin = {label: fmean(values) for label, values in grouped.items()}
    return rows, by_bin, fmean(by_bin.values())


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    directory = Path(args.scores).resolve()
    completion = json.loads((directory / 'completed.json').read_text())
    result_path = directory / 'result.json'
    result = json.loads(result_path.read_text())
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if not completion.get('ok') or completion['result_sha256'] != sha(result_path):
        raise ValueError('Selection scoring is incomplete or changed')
    if result['role'] != 'selection' or result['score_temperature'] != 1. or result['n_endpoints'] != 48:
        raise ValueError('Expected all 48 selection-scored endpoints')
    design_path = ROOT / 'configs/rl_calibration_design_v1.json'
    pipeline_path = ROOT / 'configs/pipeline_rl_calibration_v1.json'
    design = json.loads(design_path.read_text())
    if manifest['input_sha256'].get(str(design_path)) != sha(design_path):
        raise ValueError('Selection scorer did not use the frozen design')
    if manifest['input_sha256'].get(str(pipeline_path)) != sha(pipeline_path):
        raise ValueError('Selection scorer did not use the frozen training pipeline')
    if design['ranking']['horizon'] != 'short only' or design['ranking']['top_k'] != 3:
        raise ValueError('The prespecified short-horizon rule differs')
    base_path = directory / 'base_selection_nll.jsonl'
    base_rows, base_bins, base_bits = read_scores(base_path)
    ids = {row['source_id'] for row in base_rows}
    hashes = {str(path): sha(path) for path in [result_path, manifest_path, directory / 'completed.json',
                                               design_path, pipeline_path, base_path, Path(__file__).resolve()]}
    endpoints = {}
    short_training_cost, short_scoring_cost = 0, 0
    for endpoint in result['checkpoints']:
        key = (endpoint['family'], endpoint['arm'], endpoint['horizon'])
        if key in endpoints:
            raise ValueError('Duplicate family/arm/horizon endpoint')
        path = directory / endpoint['scores_directory'] / 'selection_nll.jsonl'
        rows, by_bin, bits = read_scores(path, ids)
        if not math.isclose(bits, endpoint['length_balanced_mean_bits'], rel_tol=1e-10, abs_tol=1e-8):
            raise ValueError('Independent score mean differs from evaluator receipt')
        endpoints[key] = {**endpoint, 'independently_aggregated_bits': bits, 'independent_bits_by_length': by_bin}
        hashes[str(path)] = sha(path)
        if endpoint['horizon'] == 'short':
            short_training_cost += endpoint['actual_training_flop_proxy']
            short_scoring_cost += endpoint['ledger']['teacher_scoring']['flop_proxy']
    families = sorted({key[0] for key in endpoints})
    if len(families) != 12 or set(endpoints) != {(family, arm, horizon)
            for family in families for arm in ['control', 'family'] for horizon in ['short', 'long']}:
        raise ValueError('Incomplete paired horizon matrix')
    utilities = {}
    for horizon in ['short', 'long']:
        records = []
        for family in families:
            control, treated = [endpoints[(family, arm, horizon)] for arm in ['control', 'family']]
            record = {'family': family,
                      'utility_bits': control['independently_aggregated_bits'] - treated['independently_aggregated_bits'],
                      'control_bits': control['independently_aggregated_bits'],
                      'family_bits': treated['independently_aggregated_bits'],
                      'absolute_family_gain_bits': base_bits - treated['independently_aggregated_bits'],
                      'absolute_control_gain_bits': base_bits - control['independently_aggregated_bits'],
                      'utility_by_length': {label: control['independent_bits_by_length'][label] - treated['independent_bits_by_length'][label]
                                            for label in BINS},
                      'control_actual_training_proxy': control['actual_training_flop_proxy'],
                      'family_actual_training_proxy': treated['actual_training_flop_proxy'],
                      'control_steps': control['training_steps'], 'family_steps': treated['training_steps']}
            if 'selection_gradient_dot_actual_displacement' in control:
                record['actual_displacement_prediction_bits'] = control['selection_gradient_dot_actual_displacement'] - treated['selection_gradient_dot_actual_displacement']
            records.append(record)
        utilities[horizon] = sorted(records, key=lambda row: (-row['utility_bits'], row['family']))
    top3 = [row['family'] for row in utilities['short'][:3]]
    frozen = {'frozen': True, 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'selector': 'short_full_start_rl_matched_intervention', 'top3': top3,
              'horizon': 'short', 'training_proxy_ceiling_per_trial': design['short_proxy_ceiling'],
              'tie_break': 'ascending family name', 'ranking': utilities['short'],
              'ranking_objective': 'Equal length-stratum whole-reference bits: matched control minus family, T=1.',
              'selection_population_sha256': manifest['input_sha256'][str(ROOT / 'data/released_v1/selection.jsonl')],
              'full_source_training_cap': 6, 'selection_oracle_reference_cap': 12,
              'online_training_seed': 20260910, 'source_count_per_family': 64,
              'required_acquisition_components': {'short_trial_actual_training_proxy': short_training_cost,
                                                  'short_endpoint_query_scoring_proxy': short_scoring_cost,
                                                  'total': short_training_cost + short_scoring_cost},
              'experimental_cost_note': 'All long continuations, base scoring, actual-displacement gradients and other validation remain charged in total sprint compute. They do not choose the frozen short-horizon ranking.',
              'confirmation_used_to_rank': False, 'final_confirmation_opened': False,
              'input_sha256': hashes}
    write_new(Path(args.output), frozen)
    report = {'utc': frozen['utc'], 'base_selection_bits': base_bits, 'base_selection_by_length': base_bins,
              'rankings': utilities, 'frozen_short_top3': top3,
              'short_horizon_actual_training_proxy': short_training_cost,
              'scoring_diagnostic_total_proxy': result['total_flop_proxy'],
              'scope': 'Selection-only development analysis; no independent task-generalization claim here.',
              'frozen_selector_sha256': sha(Path(args.output)), 'input_sha256': hashes}
    write_new(Path(args.report), report)
    print(json.dumps({'frozen_selector': args.output, 'short_top3': top3,
                      'short_acquisition_proxy': short_training_cost + short_scoring_cost}))


if __name__ == '__main__':
    main()
