"""Replay saved constrained LM tokens, legal masks, actions, and verdicts.

This checks the archived decoder execution and independently rechecks complete
proofs. It does not recompute every neural argmax; the separate real-checkpoint
cached/batched/full-forward test covers that implementation property.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import common
from legal_lm import ConstrainedState
from nd_environment import Tokenizer, strict_complete
from analyze_symbolic import strata


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(directory, audit_path):
    started = time.monotonic()
    directory, audit_path = Path(directory).resolve(), Path(audit_path).resolve()
    config_path, result_path = directory / 'config.json', directory / 'result.json'
    manifest = json.loads(config_path.read_text())
    config = manifest['args']
    result = json.loads(result_path.read_text())
    require(result['status'] == 'completed', 'Constrained evaluation is incomplete')
    require(result['input_sha256'] == manifest['input_sha256'], 'Constrained result input identities differ')
    for path, digest in manifest['input_sha256'].items():
        require(common.sha(path) == digest, f'Constrained inference input changed: {path}')
    for name in ('legal_lm.py', 'symbolic_general_env.py', 'nd_environment.py'):
        versions = {h for p, h in manifest['input_sha256'].items() if Path(p).name == name}
        require(versions == {common.sha(Path(__file__).with_name(name))}, f'Replay uses different {name}')
    profile = {k: config[k] for k in ('max_new_tokens', 'max_lines', 'max_scope', 'batch_size', 'attempts', 'temperature', 'seed')}
    require(profile == result['profile'], 'Reported inference profile differs')
    population = common.read_jsonl(config['data'])
    require(not any(set(r) - {'name', 'source_id', 'prompt'} for r in population), 'Prompt input contains reference fields')
    labels = {r['source_id']: r for r in common.read_jsonl(audit_path)}
    require({r['source_id']: r['prompt'] for r in population} == {k: r['prompt'] for k, r in labels.items()},
            'Constrained prompt/audit population differs')
    archive = directory / 'attempts.jsonl'
    require(common.sha(archive) == result['attempts_sha256'], 'Constrained attempts archive changed')
    tok = Tokenizer()
    counts, seen, outcomes, hist = Counter(), set(), defaultdict(list), Counter()
    rows = common.read_jsonl(archive)
    for row in rows:
        sid, attempt = row['source_id'], row['attempt']
        require(sid in labels and row['prompt'] == labels[sid]['prompt'], 'Archived constrained theorem mismatch')
        require(0 <= attempt < config['attempts'] and (sid, attempt) not in seen, 'Invalid/duplicate constrained attempt')
        seen.add((sid, attempt))
        require(len(row['response_ids']) <= config['max_new_tokens'], 'Response exceeds declared token cap')
        state = ConstrainedState({'source_id': sid, 'prompt': row['prompt'], 'attempt': attempt}, tok,
                                 config['max_lines'], config['max_scope'])
        for token in row['response_ids']:
            allowed = state.allowed()
            require(not state.done and token in allowed, 'Archived token was outside the legal mask or after termination')
            state.consume(token, allowed)
        if not state.done and len(row['response_ids']) < config['max_new_tokens']:
            state.allowed()  # The real batch may have performed an ignored padding step after hitting the line cap.
        replayed = state.finish()
        require(common.digest(replayed) == common.digest(row), 'Replayed masks/actions/verdict differ from archived result')
        counts['attempts'] += 1
        counts['successes'] += int(row['reward'])
        for key in ('generated_tokens', 'branch_tokens', 'forced_tokens', 'generated_inferences', 'candidate_actions_enumerated'):
            counts[key] += row[key]
        if row['reward']:
            verdict = strict_complete(row['prompt'], row['body'])
            compressed = strict_complete(row['prompt'], row['compression']['compressed_proof'])
            require(verdict['ok'] and compressed['ok'], 'Constrained certificate failed independent re-verification')
            hist[str(compressed['n_lines'])] += 1
        outcomes[sid].append(int(row['reward']))
    require(len(seen) == len(labels) * config['attempts'] == result['attempts'], 'Incomplete constrained attempt coverage')
    for key in ('successes', 'generated_tokens', 'branch_tokens', 'forced_tokens'):
        require(counts[key] == result[key], f'Constrained summary count differs: {key}')
    require(counts['generated_tokens'] == counts['branch_tokens'] + counts['forced_tokens'], 'Token decision accounting differs')
    require(sum(any(v) for v in outcomes.values()) == result['any_successes'], 'Constrained theorem success count differs')
    summaries = {name: {'theorems': len(ids), 'any_successes': sum(any(outcomes[sid]) for sid in ids),
                         'successful_attempts': sum(sum(outcomes[sid]) for sid in ids)} for name, ids in strata(labels).items()}
    paths = [config_path, result_path, archive, audit_path, Path(__file__), Path(__file__).with_name('legal_lm.py')]
    return {'status': 'passed', 'run': str(directory), 'counts': dict(counts), 'summaries': summaries,
            'successful_compressed_length_histogram': dict(hist), 'profile': profile,
            'checkpoint_sha256': common.sha(config['checkpoint']), 'input_sha256': {str(p): common.sha(p) for p in paths},
            'elapsed_seconds': time.monotonic() - started,
            'scope': 'Every saved token, legal mask, full inference, force/branch count, and verifier result was reconstructed. No neural argmax recomputation in this audit. Premises and legal syntax remain supplied by the symbolic interface.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.run, args.audit)
    common.write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('input_sha256', 'summaries')}))
