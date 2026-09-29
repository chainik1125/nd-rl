"""Run a frozen, finite CPU task list and preserve failures and partial coverage."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.config.read_text())
    digest = common.sha(args.config)
    out = Path(spec['output'])
    out.mkdir(parents=True, exist_ok=False)
    deadline = datetime.fromisoformat(spec['deadline_utc'].replace('Z', '+00:00')).timestamp()
    stopped, lock, children = threading.Event(), threading.Lock(), {}
    status = {'status': 'active', 'pid': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat(),
              'protocol_sha256': digest, 'tasks': {}}

    def verify():
        if common.sha(args.config) != digest:
            raise ValueError('Task list changed')
        for path, expected in spec['input_sha256'].items():
            if common.sha(path) != expected:
                raise ValueError('Frozen input changed: ' + path)

    def stop(signum, frame):
        stopped.set()
        with lock:
            for child in children.values():
                if child.poll() is None:
                    child.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    verify()
    common.write_json(out / 'progress.json', status)

    def run(task):
        record = {'status': 'starting', 'command': task['command']}
        child, started = None, time.monotonic()
        try:
            verify()
            if stopped.is_set() or time.time() >= deadline:
                record['status'] = 'deadline_or_interruption'
                return record
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
            with (out / (task['name'] + '.log')).open('w') as log:
                child = subprocess.Popen(task['command'], cwd=common.ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
                with lock:
                    children[task['name']] = child
                    record.update(status='active', pid=child.pid)
                    status['tasks'][task['name']] = record.copy()
                    common.write_json(out / 'progress.json', status)
                try:
                    code = child.wait(timeout=max(.1, min(task['timeout_seconds'], deadline - time.time())))
                except subprocess.TimeoutExpired:
                    record['status'] = 'timeout'
                else:
                    record.update(status='completed' if code == 0 else 'failed', exit_code=code)
                    verify()
                    if code == 0:
                        result = json.loads(Path(task['result_path']).read_text())
                        record['result_status'] = result.get('status')
                        record['result_sha256'] = common.sha(task['result_path'])
                        if result.get('status') not in ('complete', 'completed'):
                            record['status'] = 'partial'
        except Exception as error:
            record.update(status='failed', error=type(error).__name__, message=str(error))
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
            with lock:
                children.pop(task['name'], None)
            record['elapsed_seconds'] = time.monotonic() - started
        return record

    with ThreadPoolExecutor(max_workers=spec['parallel_tasks']) as pool:
        futures = {pool.submit(run, task): task['name'] for task in spec['tasks']}
        for future in as_completed(futures):
            name, record = futures[future], future.result()
            with lock:
                status['tasks'][name] = record
                common.write_json(out / 'progress.json', status)
            print(json.dumps({'name': name, **{k: v for k, v in record.items() if k != 'command'}}), flush=True)
    status.update(status='completed' if all(r['status'] == 'completed' for r in status['tasks'].values()) else 'completed_with_failures',
                  finished_utc=datetime.now(timezone.utc).isoformat())
    common.write_json(out / 'result.json', status)
    common.write_json(out / 'progress.json', status)


if __name__ == '__main__':
    main()
