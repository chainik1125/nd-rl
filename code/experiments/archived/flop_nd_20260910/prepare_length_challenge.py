"""Freeze a fresh extrapolation challenge for already trained minimal policies."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil
import common
from structured_generate import make_task
from prepare_data import identities, OLDER
from nd_environment import strict_complete, Tokenizer
from proof_diagnostics import compress_proof


def main():
    root = common.ROOT
    out, snap = root / 'data/length_challenge_v1', root / 'snapshots/length_challenge_v1'
    out.mkdir(exist_ok=False)
    snap.mkdir(exist_ok=False)
    for name in ('common.py', 'run_bounded_cpu.py', 'prepare_length_challenge.py', 'structured_generate.py', 'prepare_data.py', 'nd_environment.py', 'analyze_symbolic.py', 'symbolic_general_env.py'):
        shutil.copy2(root / 'code' / name, snap / name)
    seed = 202609102327
    rng, tok = random.Random(seed), Tokenizer()
    excluded_theorems, excluded_contexts = set(), set()
    exclusion_paths = []
    for directory in ('v1', 'structured_v1', 'detour_v1'):
        for path in sorted((root / 'data' / directory).glob('*.jsonl')):
            for row in common.read_jsonl(path):
                if 'prompt' not in row:
                    continue
                t, c, _ = identities(row['prompt']) if 'theorem_id' not in row or 'context_id' not in row else (row['theorem_id'], row['context_id'], None)
                excluded_theorems.add(t)
                excluded_contexts.add(c)
            exclusion_paths.append(path)
    for path in (root / 'data').glob('symbolic_*.jsonl'):
        for row in common.read_jsonl(path):
            t, c, _ = identities(row['prompt'])
            excluded_theorems.add(t)
            excluded_contexts.add(c)
        exclusion_paths.append(path)
    guard_path = OLDER / 'data/all_textbook_guards.json'
    guard_source = json.loads(guard_path.read_text())
    guards = [set(guard_source[key]) for key in ('canonical_theorem_sha256', 'nonempty_premise_context_sha256', 'closed_formula_skeleton_sha256')]
    all_rows, counts, rejected = [], Counter(), Counter()
    for length in (33, 65, 129):
        for family in ('projection_atomic', 'projection_compound'):
            attempts = 0
            while sum(counts[length, family, atom] for atom in 'PQRS') < 8:
                attempts += 1
                if attempts > 10000:
                    raise ValueError('Quota not filled')
                row = make_task(rng, length, family)
                if counts[length, family, row['target_atom']] >= 2:
                    continue
                theorem, context, guard = identities(row['prompt'])
                if theorem in excluded_theorems or context in excluded_contexts or any(k is not None and k in g for k, g in zip(guard, guards)):
                    rejected['identity_or_context_guard'] += 1
                    continue
                verdict = strict_complete(row['prompt'], row['body'])
                compressed = compress_proof(row['prompt'], row['body'])
                assert verdict['ok'] and verdict['independent_ok']
                assert compressed['compressed_length'] == compressed['dependency_pruned_length'] == length
                assert strict_complete(row['prompt'], compressed['compressed_proof'])['ok']
                row.update(source_id=common.digest([row['prompt'], 'length-challenge-v1']), theorem_id=theorem, context_id=context,
                           name=f'length_challenge_{theorem[:16]}', role='prospective_post_development_challenge', length_bin=str(length),
                           reference_tokens=1 + len(tok.encode_record(row['prompt'] + ' ' + row['body'])) - len(tok.encode_prompt(row['prompt'])),
                           prompt_tokens=len(tok.encode_prompt(row['prompt'])) + 1, dependency_pruned_lines=length,
                           independent_ok=True, external_training_label=False)
                all_rows.append(row)
                counts[length, family, row['target_atom']] += 1
                excluded_theorems.add(theorem)
                excluded_contexts.add(context)
        rows = [r for r in all_rows if r['n_lines'] == length]
        rng.shuffle(rows)
        common.write_jsonl(out / f'audit/lines_{length}.jsonl', rows)
        common.write_jsonl(out / f'lines_{length}.jsonl', [{k: r[k] for k in ('prompt', 'source_id', 'name')} for r in rows])
        print(json.dumps({'length': length, 'theorems': len(rows), 'references_double_checked': True}), flush=True)
    existing = json.loads((root / 'configs/final_comparators_v1.json').read_text())
    templates = [t for t in existing['tasks'] if t['name'].startswith('minimal_') and '_structured_final_greedy_' in t['name']]
    assert len(templates) == 9
    tasks, audit_tasks = [], []
    for length in (33, 65, 129):
        for template in templates:
            name = template['name'].replace('_structured_final_greedy_v1', f'_length{length}_challenge_greedy_v1')
            command = template['command'].copy()
            replacements = {'--population': str(out / f'lines_{length}.jsonl'), '--out': str(root / 'results' / name),
                            '--max-lines': str(length + 8), '--max-seconds': '600', '--seed': '202609102328'}
            for flag, value in replacements.items():
                command[command.index(flag) + 1] = value
            result_path = root / 'results' / name / 'symbolic_general_eval_summary.json'
            tasks.append({'name': name, 'command': command, 'timeout_seconds': 660, 'result_path': str(result_path), 'endpoint_paths': template['endpoint_paths']})
            report_path = root / 'reports' / ('audit_' + name + '.json')
            audit_tasks.append({'name': 'audit_' + name, 'ready_paths': [str(result_path)], 'input_sha256': {str(p): common.sha(p) for p in [*snap.glob('*.py'), out / f'audit/lines_{length}.jsonl']}, 'command': [command[0], str(snap / 'analyze_symbolic.py'), '--run', str(result_path.parent), '--audit', str(out / f'audit/lines_{length}.jsonl'), '--evaluator-directory', str(Path(command[1]).parent), '--output', str(report_path)], 'timeout_seconds': 600})
    inputs = sorted(set([*snap.glob('*.py'), *out.rglob('*.jsonl'), guard_path, root / 'configs/final_comparators_v1.json', *exclusion_paths,
                         *[Path(p) for t in tasks for p in t['endpoint_paths']], *(root / 'snapshots/minimal_general_v1').glob('*.py'), *common.VENDOR.rglob('*.py')]))
    protocol = {'status': 'frozen_before_new_challenge_outcomes', 'frozen_utc': datetime.now(timezone.utc).isoformat(), 'generator_seed': seed,
                'selection': 'All nine frozen minimal policies, all 48 prospectively generated theorems, every prespecified length and family; no training or checkpoint selection.',
                'interpretation': 'Additional post-development repeated-procedure extrapolation challenge, separate from sealed final populations. Greedy models and one fixed random-control attempt; line caps equal reference length plus eight. Not an LM compute comparison or unrestricted proof lower bound.',
                'excluded_theorems_count': len(excluded_theorems) - 48, 'excluded_contexts_count': len(excluded_contexts) - 48, 'rejected': dict(rejected),
                'deadline_utc': '2026-09-11T01:20:00Z', 'output': str(root / 'jobs/length_challenge_v1'), 'parallel_tasks': 3,
                'input_sha256': {str(p): common.sha(p) for p in inputs}, 'tasks': tasks}
    common.write_json(root / 'configs/length_challenge_v1.json', protocol)
    common.write_json(root / 'configs/length_challenge_audits_v1.json', {'deadline_utc': '2026-09-11T01:35:00Z', 'output': str(root / 'jobs/length_challenge_audits_v1'), 'tasks': audit_tasks})
    common.write_json(out / 'manifest.json', {'status': 'complete', 'rows': 48, 'counts': {str(k): v for k, v in counts.items()}, 'config_sha256': common.sha(root / 'configs/length_challenge_v1.json')})


if __name__ == '__main__':
    main()
