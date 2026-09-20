"""Replay saved raw tokens against authoritative population theorem identities."""
from pathlib import Path
import argparse
from collections import Counter
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'snapshots/training_v1'))
import common
from nd_environment import Tokenizer, judge
from strict_verify import safe_verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('plan')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    start = time.time()
    plan = json.loads(Path(args.plan).read_text())
    records = {r['source_id']: r for r in common.read_jsonl(plan['data'])}
    tokenizer = Tokenizer()
    summaries = {}
    for name, model in plan['models'].items():
        if not model.get('attempts'):
            continue
        counts = Counter()
        for row in common.read_jsonl(model['attempts']):
            source = records[row['source_id']]
            if row['prompt'] != source['prompt'] or row['context_id'] != source['context_id']:
                raise AssertionError('Saved attempt theorem/context differs from population')
            if row['source_lines'] != source['n_lines'] or row['length_bin'] != source['length_bin']:
                raise AssertionError('Saved attempt length identity differs')
            if row['reset_index'] != 0 or row['prefix_ids']:
                raise AssertionError('Supplied proof prefix in unaided evaluation')
            expected_ids = tokenizer.encode_prompt(source['prompt'])
            if row['original_prompt_ids'] != expected_ids or row['prompt_ids'] != expected_ids + [tokenizer.prf_id]:
                raise AssertionError('Encoded theorem prompt differs from source')
            replay = judge({'prompt': source['prompt'], 'prefix_ids': []}, row['response_ids'], tokenizer)
            if replay['reward'] != row['reward'] or replay['body'] != row['body']:
                raise AssertionError('Reward or body does not reproduce from saved sampled tokens')
            counts['attempts'] += 1
            counts['strict_successes'] += replay['reward']
            if replay['body']:
                official, reason, length = safe_verify(source['prompt'], replay['body'])
                counts['official_successes'] += int(official)
                if replay['reward'] and not official:
                    raise AssertionError('Credited proof fails unchanged strict checker')
                if official and not replay['reward']:
                    counts['official_only_rejected_by_incremental_checker'] += 1
            else:
                counts['boundary_failures_counted'] += 1
            if row['reward'] and row.get('compression'):
                comp = row['compression']
                ok, reason, length = safe_verify(source['prompt'], comp['compressed_proof'])
                if not ok or length != comp['compressed_length']:
                    raise AssertionError('Compressed proof failed independent replay')
                counts['compressed_reverified'] += 1
        summaries[name] = dict(counts)
    output = {'ok': True, 'models': summaries, 'data_sha256': common.sha(plan['data']),
              'plan_sha256': common.sha(args.plan), 'analysis_sha256': common.sha(__file__),
              'verifier_code_sha256': common.dependency_hashes(),
              'boundary_decoder_sha256': common.sha(ROOT / 'snapshots/training_v1/nd_environment.py'),
              'method': 'Rebuild model input from population identity, re-decode saved action tokens, check reward/body equality, verify every credited and compressed proof with unchanged strict verifier.',
              'elapsed_seconds': time.time() - start}
    with Path(args.output).open('x') as stream:
        json.dump(output, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'ok': True, 'models': len(summaries), 'attempts': sum(r['attempts'] for r in summaries.values()), 'output': args.output}))


if __name__ == '__main__':
    main()
