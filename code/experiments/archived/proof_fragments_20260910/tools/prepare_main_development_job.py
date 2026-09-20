"""Supplement frozen evaluation recipes with completed-endpoint provenance.

The original recipes remain unchanged. A v2 pipeline adds hash-bound receipts
for the reused baseline and each actual compute crossing before GPU launch.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = [500000000000000, 1000000000000000, 2000000000000000, 4000000000000000]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_new(path, content):
    with path.open('x') as stream:
        json.dump(content, stream, indent=2)
        stream.write('\n')


def inspect_arm(directory):
    result_path, completion_path = directory / 'result.json', directory / 'completed.json'
    result = json.loads(result_path.read_text())
    completion = json.loads(completion_path.read_text())
    if completion.get('ok') is not True or completion['result_sha256'] != sha(result_path):
        raise ValueError('Main completion receipt is missing or changed')
    final_path = directory / 'final.pt'
    if completion['final_sha256'] != sha(final_path) or result['final_sha256'] != sha(final_path):
        raise ValueError('Final checkpoint differs from its completion/result receipts')
    updates = [json.loads(line) for line in (directory / 'updates.jsonl').read_text().splitlines()]
    if len(updates) != result['steps'] or [u['step'] for u in updates] != list(range(1, len(updates) + 1)):
        raise ValueError('Incomplete update sequence')
    if result['stop_reason'] != 'flop_proxy_ceiling' or result['max_flop_proxy'] != THRESHOLDS[-1]:
        raise ValueError('Endpoint did not finish the main declared budget')
    cumulative = 0
    for update in updates:
        step_cost = update['step_flop_proxy']
        if not 0 < step_cost <= result['worst_case_one_step_flop_proxy']:
            raise ValueError('Update compute is nonpositive or exceeds its bound')
        cumulative += step_cost
        if update['cumulative_flop_proxy'] != cumulative:
            raise ValueError('Intermediate cumulative compute differs from the running sum')
    if cumulative != result['total_flop_proxy']:
        raise ValueError('Update compute does not reconcile with final result')
    if updates[-2]['cumulative_flop_proxy'] >= THRESHOLDS[-1]:
        raise ValueError('Endpoint continued after the first crossing')
    if result['flop_proxy_overshoot'] > result['worst_case_one_step_flop_proxy']:
        raise ValueError('Endpoint overshoot exceeds its bound')
    inputs = [directory / x for x in ['manifest.json', 'result.json', 'completed.json', 'updates.jsonl', 'final.pt']]
    receipts = {}
    final = torch.load(final_path, map_location='cpu', weights_only=True)
    for threshold in THRESHOLDS:
        event = next(u for u in updates if u['cumulative_flop_proxy'] >= threshold)
        path = directory / f'flops_{threshold}.pt'
        payload = torch.load(path, map_location='cpu', weights_only=True)
        metadata = payload['metadata']
        expected = {'step': event['step'], 'requested_flop_proxy': threshold,
                    'actual_flop_proxy': event['cumulative_flop_proxy']}
        if any(metadata.get(k) != v for k, v in expected.items()):
            raise ValueError('Milestone metadata does not match its first actual crossing')
        if threshold == THRESHOLDS[-1]:
            if payload['config'] != final['config'] or payload['tokenizer'] != final['tokenizer']:
                raise ValueError('Final and last milestone model format differs')
            if set(payload['model']) != set(final['model']) or any(
                    not torch.equal(tensor, final['model'][name]) for name, tensor in payload['model'].items()):
                raise ValueError('Final weights differ from last milestone')
        receipts[str(threshold)] = {**expected, 'checkpoint_sha256': sha(path),
                                    'overshoot': event['cumulative_flop_proxy'] - threshold}
        inputs.append(path)
    return {'steps': result['steps'], 'total_flop_proxy': result['total_flop_proxy'],
            'milestones': receipts, 'final_matches_last_milestone_weights': True}, inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed-index', type=int, choices=[1, 2], required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    old_name = f'main_development_seed{args.seed_index}_v1'
    new_name = f'main_development_seed{args.seed_index}_v2'
    old_path = ROOT / 'configs' / f'pipeline_{old_name}.json'
    plan = json.loads(old_path.read_text())
    recipe_path = ROOT / 'configs' / f'recipe_{old_name}.json'
    recipe = json.loads(recipe_path.read_text())
    for path, digest in recipe['prospective_input_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Prospective recipe input changed: ' + path)
    baseline = ROOT / 'results/confirmation_adapt_v1'
    manifest = json.loads((baseline / 'manifest.json').read_text())
    expected = {**recipe['decoding'], 'checkpoint': str(ROOT / 'results/adapt_v1/final.pt'),
                'data': recipe['data'], 'precision': 'fp32', 'audit_likelihood': True,
                'batch_size': 64, 'microbatch': 16}
    if any(manifest['args'].get(k) != v for k, v in expected.items()):
        raise ValueError('Reused baseline decoding/model/data recipe differs')
    for label in ['checkpoint', 'data', 'release']:
        if sha(manifest['args'][label]) != manifest['input_sha256'][label]:
            raise ValueError('Reused baseline input changed')
    completion = json.loads((baseline / 'completed.json').read_text())
    if completion.get('ok') is not True or completion['result_sha256'] != sha(baseline / 'result.json'):
        raise ValueError('Reused baseline completion changed')
    inputs = [baseline / filename for filename in ['manifest.json', 'completed.json', 'result.json',
                                                   'whole_proof_nll.jsonl', 'attempts.jsonl']]
    inputs += [old_path, recipe_path, Path(__file__).resolve()]
    arms = {}
    for arm in ['selected_full', 'selected_reset', 'uniform_reset', 'uniform_full']:
        arm_name = f'rl_{arm}_seed{args.seed_index}_v1'
        arms[arm_name], local_inputs = inspect_arm(ROOT / 'results' / arm_name)
        inputs.extend(local_inputs)
    receipt_path = ROOT / 'reports' / f'main_development_seed{args.seed_index}_provenance_v1.json'
    receipt = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'status': 'passed', 'original_recipe_unchanged': True,
               'reused_baseline_settings_match': expected, 'arms': arms,
               'input_sha256': {str(p): sha(p) for p in sorted(set(inputs))}}
    write_new(receipt_path, receipt)
    plan.update(name=new_name, status='frozen_ready_after_endpoint_provenance_review',
                supersedes_pipeline=str(old_path), provenance_receipt=str(receipt_path),
                inputs=sorted(set(plan['inputs'] + [str(p) for p in inputs] + [str(receipt_path)])))
    new_path = ROOT / 'configs' / f'pipeline_{new_name}.json'
    write_new(new_path, plan)
    print(json.dumps({'pipeline': str(new_path), 'provenance_receipt': str(receipt_path),
                      'verified_arms': len(arms), 'reused_baseline_bound': True}))


if __name__ == '__main__':
    main()
