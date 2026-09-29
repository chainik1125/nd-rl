"""Freeze RL calibration banks from the original SFT matching receipts.

This changes neither source identities nor controls after inspecting outcomes.
Only generated candidate payloads and already public matching metadata are read.
"""
import datetime
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'snapshots/release_v1'
sys.path.insert(0, str(SOURCE))
import common
from make_bank import validated_pool, describe

PYTHON = '/data/users/dmitry/simplex-research/.venv/bin/python'
SHORT = 106300035072000
LONG = 4 * SHORT


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    candidates = ROOT / 'data/released_v1/candidate.jsonl'
    metadata_path = candidates.parent / 'manifest.json'
    metadata = json.loads(metadata_path.read_text())
    if metadata['population_sha256']['candidate'] != sha(candidates):
        raise ValueError('Candidate population identity differs')
    records = validated_pool(common.read_jsonl(candidates))
    by_id = {r['source_id']: r for r in records}
    families = sorted({r['family'] for r in records})
    if len(families) != 12 or len(records) != 768:
        raise ValueError('Expected the frozen twelve-family population')
    order_rng = random.Random(20260914)
    order_rng.shuffle(families)
    directory = ROOT / 'data/rl_calibration_banks_v1'
    directory.mkdir(exist_ok=False)
    inputs = [candidates, metadata_path, Path(__file__).resolve(),
              ROOT / 'results/adapt_v1/final.pt', ROOT / 'data/released_v1/retention.jsonl',
              ROOT / 'configs/pipeline_factorial_seed1_v1.json',
              ROOT / 'configs/rl_calibration_design_v1.json',
              ROOT / 'theory/rl_specific_calibration_design.md']
    template = json.loads((ROOT / 'configs/pipeline_factorial_seed1_v1.json').read_text())['commands'][0]
    commands, bank_receipts, checkpoints, arm_order = [], {}, [], []
    for family in families:
        matches_path = ROOT / 'results/calibration_v1' / f'{family}_matching.jsonl'
        matches = common.read_jsonl(matches_path)
        inputs.append(matches_path)
        if len(matches) != 64:
            raise ValueError('Original pair lacks 64 matches')
        pair_order = [('control', 'ordinary_source'), ('family', 'family_source')]
        order_rng.shuffle(pair_order)
        for kind, field in pair_order:
            arm_order.append([family, kind])
            ids = [m[field] for m in matches]
            if len(set(ids)) != 64:
                raise ValueError('Original matching repeats a source')
            rows = [by_id[i] for i in ids]
            if kind == 'family' and any(r['family'] != family for r in rows):
                raise ValueError('Family receipt does not match its source family')
            if kind == 'control' and any(r['family'] == family for r in rows):
                raise ValueError('Control includes its compared family')
            bank_path = directory / f'{family}_{kind}.jsonl'
            common.write_jsonl(bank_path, rows)
            inputs.append(bank_path)
            bank_receipts[f'{family}_{kind}'] = {
                'bank': str(bank_path), 'bank_sha256': sha(bank_path),
                'original_matching': str(matches_path), 'matching_sha256': sha(matches_path),
                'statistics': describe(rows), 'source_ids_in_order': ids,
                'matching_note': 'Exact source identities and paired order inherited from SFT calibration. Matching was chosen using reference/suffix features; full-start RL uses the original whole theorem.'}
            output = ROOT / 'results/rl_calibration_v1' / f'{family}_{kind}'
            command = list(template)
            for flag, value in [('data', str(bank_path)), ('output', str(output)),
                                ('start', 'full'), ('max-flop-proxy', str(LONG)),
                                ('steps', '20000')]:
                command[command.index('--' + flag) + 1] = value
            first = command.index('--flop-milestones') + 1
            last = command.index('--curriculum-rollouts')
            command[first:last] = [str(SHORT), str(LONG)]
            commands.append(command)
            checkpoints += [str(output / f'flops_{SHORT}.pt'), str(output / 'final.pt')]
    manifest = {'status': 'frozen_rl_calibration_banks',
                'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'full_source_cap': 6, 'candidate_sha256': sha(candidates),
                'bank_count': len(bank_receipts), 'banks': bank_receipts,
                'new_matching_performed': False, 'new_model_outcomes_used_by_bank_builder': False}
    write_new(directory / 'manifest.json', manifest)
    inputs.append(directory / 'manifest.json')
    pipeline = {'name': 'rl_calibration_v1',
                'question': 'Does family utility transport from SFT to full-start RL at equal short training proxy, and does short-RL utility predict a four-times-longer RL intervention?',
                'expected_seconds': 4500, 'timeout_seconds': 6300,
                'deadline_utc': '2026-09-11T01:15:00+00:00',
                'source_snapshots': [str(ROOT / 'snapshots/training_v1')],
                'inputs': [str(p) for p in inputs], 'commands': commands,
                'status': 'frozen_ready', 'family_order': families,
                'arm_order': arm_order, 'job_order_seed': 20260914,
                'short_training_proxy_ceiling': SHORT, 'long_training_proxy_ceiling': LONG,
                'aggregate_training_proxy_ceiling': len(commands) * LONG,
                'checkpoints_for_scoring': checkpoints,
                'training_seed': 20260910, 'final_confirmation_opened': False}
    plan_path = ROOT / 'configs/pipeline_rl_calibration_v1.json'
    write_new(plan_path, pipeline)
    print(json.dumps({'pipeline': str(plan_path), 'banks': len(bank_receipts),
                      'scoring_checkpoints': len(checkpoints), 'total_training_proxy_ceiling': len(commands) * LONG}))


if __name__ == '__main__':
    main()
