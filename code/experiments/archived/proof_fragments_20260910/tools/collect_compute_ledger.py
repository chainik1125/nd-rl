"""Collect actual launched-job costs without counting reused artifacts twice."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def new_result_cost(directory, result):
    # Difficulty and projected-gradient diagnostics report acquisition receipts
    # that include prior rollouts; only newly executed work enters this ledger.
    if 'new_total_flop_proxy' in result:
        return result['new_total_flop_proxy'], 'new_total_flop_proxy'
    if 'fresh_audit_flop_proxy' in result:
        return result['fresh_audit_flop_proxy'], 'fresh_audit_flop_proxy'
    if 'total_pilot_flop_proxy' in result and (directory / 'initial_cuts.json').exists():
        cut_map = json.loads((directory / 'initial_cuts.json').read_text())
        return cut_map['new_pilot_flop_proxy'], 'initial_cuts.new_pilot_flop_proxy'
    if 'total_flop_proxy' in result:
        return result['total_flop_proxy'], 'total_flop_proxy'
    raise ValueError('Unknown completed cost schema: ' + str(directory))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    now = datetime.datetime.now(datetime.timezone.utc)
    jobs, outputs, hashes = [], {}, {}
    completed_gpu_seconds = active_gpu_seconds = 0.
    for directory in sorted((ROOT / 'jobs').iterdir()):
        events_path = directory / 'events.jsonl'
        spec_path = directory / 'spec.json'
        if not events_path.exists() or not spec_path.exists():
            continue
        events = rows(events_path)
        acquired = next((e for e in events if e['kind'] == 'acquired'), None)
        if acquired is None:
            continue
        spec = json.loads(spec_path.read_text())
        released = next((e for e in events if e['kind'] == 'released'), None)
        hashes[str(events_path)] = sha(events_path)
        hashes[str(spec_path)] = sha(spec_path)
        if released:
            seconds, status = released['elapsed_seconds'], released['status']
            completed_gpu_seconds += seconds
        elif directory.name == 'base_sft_v1':
            path = ROOT / 'reports/base_sft_v1_receipt.json'
            receipt = json.loads(path.read_text())
            seconds, status = receipt['elapsed_gpu_seconds'], 'worker_completed_supervisor_final_log_error'
            completed_gpu_seconds += seconds
            hashes[str(path)] = sha(path)
        else:
            seconds = (now - datetime.datetime.fromisoformat(acquired['utc'])).total_seconds()
            status = 'active_or_awaiting_release_receipt'
            active_gpu_seconds += seconds
        gpu = acquired['gpu']['index']
        if gpu in [0, 2]:
            raise ValueError('Excluded GPU appeared in a sprint lease')
        jobs.append({'name': directory.name, 'status': status, 'gpu': gpu,
                     'lease_seconds': seconds, 'cloud_spend_usd': (released or {}).get('cloud_spend_usd', 0)})
        if not spec.get('pipeline_plan'):
            continue
        plan_path = Path(spec['pipeline_plan'])
        hashes[str(plan_path)] = sha(plan_path)
        plan = json.loads(plan_path.read_text())
        for command in plan['commands']:
            if '--output' not in command:
                continue
            output = Path(command[command.index('--output') + 1]).resolve()
            if not output.is_relative_to(ROOT / 'results'):
                continue
            key = str(output)
            if key in outputs:
                outputs[key]['referencing_jobs'].append(directory.name)
                continue
            record = {'referencing_jobs': [directory.name], 'new_accounted_flop_proxy': 0,
                      'status': 'declared_stage_not_completed', 'command': command[:3]}
            if (output / 'result.json').exists() and (output / 'completed.json').exists():
                result_path, completed_path = output / 'result.json', output / 'completed.json'
                result, completion = json.loads(result_path.read_text()), json.loads(completed_path.read_text())
                if completion.get('ok') is not True or completion['result_sha256'] != sha(result_path):
                    raise ValueError('Output completion integrity differs: ' + key)
                cost, field = new_result_cost(output, result)
                record.update(status='completed', new_accounted_flop_proxy=cost, cost_field=field,
                              process_elapsed_seconds=result.get('elapsed_seconds'))
                hashes[str(result_path)] = sha(result_path)
                hashes[str(completed_path)] = sha(completed_path)
                if field.startswith('initial_cuts.'):
                    hashes[str(output / 'initial_cuts.json')] = sha(output / 'initial_cuts.json')
            elif (output / 'progress.json').exists():
                progress = json.loads((output / 'progress.json').read_text())
                observed = next(((progress[k], k) for k in ['cumulative_flop_proxy', 'training_flop_proxy'] if k in progress), None)
                record['status'] = 'incomplete_observed_lower_bound'
                if observed:
                    record.update(new_accounted_flop_proxy=observed[0], cost_field='progress.' + observed[1])
                else:
                    record['cost_field'] = 'partial_score_cost_not_yet_reconciled'
            outputs[key] = record
    base_events = ROOT / 'results/base_sft_v1/events.jsonl'
    base_last = rows(base_events)[-1]
    hashes[str(base_events)] = sha(base_events)
    hashes[str(Path(__file__).resolve())] = sha(Path(__file__))
    report = {'utc': now.isoformat(), 'jobs': jobs, 'model_operations': outputs,
              'completed_gpu_lease_seconds': completed_gpu_seconds,
              'active_gpu_lease_seconds_so_far': active_gpu_seconds,
              'total_gpu_lease_hours_so_far': (completed_gpu_seconds + active_gpu_seconds) / 3600,
              'cloud_spend_usd': sum(j['cloud_spend_usd'] for j in jobs),
              'post_baseline_new_accounted_flop_proxy': sum(v['new_accounted_flop_proxy'] for v in outputs.values()),
              'shared_base_training': {'steps': base_last['step'],
                                       'actual_response_tokens': base_last['actual_response_tokens'],
                                       'training_elapsed_seconds': base_last['elapsed_seconds'],
                                       'dense_flop_proxy': None,
                                       'note': 'Base trainer did not instrument the same dense proxy. Its measured tokens and time are reported separately; GPU lease total includes it.'},
              'accounting_notes': ['Only acquired GPU jobs are included; prepared jobs do not count as executions.',
                                   'Each physical operation output is counted once, even if referenced by multiple plans.',
                                   'Aggregate calibration results already include their nested arm costs.',
                                   'Reused pilot/acquisition receipts are excluded from new compute and remain separately recorded in diagnostic artifacts.',
                                   'Incomplete operations contribute an observed lower bound; a closed report requires all launched jobs reconciled.',
                                   'Proxy approximates model arithmetic, not measured hardware FLOPs. CPU data collection and interactive analysis are not included in model proxy.'],
              'input_sha256': hashes}
    target = Path(args.output)
    with target.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: report[k] for k in ['total_gpu_lease_hours_so_far', 'cloud_spend_usd', 'post_baseline_new_accounted_flop_proxy']}))


if __name__ == '__main__':
    main()
