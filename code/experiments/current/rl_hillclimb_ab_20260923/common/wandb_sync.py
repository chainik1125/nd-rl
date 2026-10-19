#!/usr/bin/env python3
"""HOST-SIDE Weights & Biases mirror for the RL hill-climb A/B (run by the monitor; no W&B key enters a container).

  python wandb_sync.py --root /scratch/cowifornia/nd-rl/ab --project nd-rl-hillclimb [--poll 20] [--once]

Finds every <root>/<arm>/**/runs/<name>/metrics.jsonl for arm in --arms (default gear,orch) and mirrors
new complete JSONL lines to W&B: one W&B run per metrics file, name <arm>/<workspace>/<name>, group <arm>,
job_type <workspace> (the path between the arm dir and runs/, '/' -> '-'). Config comes from the sibling
summary.json (shas, seed, budget, gpu, started_utc) when it appears.

Each record is logged with keys `<phase>/<field>` (nested dicts flattened with '.', e.g.
sample/solved_by_len.7), plus `elapsed` and `rl_step`. The W&B step is the record's `step` while it is
monotone non-decreasing, else last step + 1. `fail_reasons` becomes a wandb.Table (reason, count).
When summary.json appears with a status, dev_score (= metric), by_len, heldout_greedy and status go into
the run summary and the run is finished. Per-file byte offsets, run ids and last steps live in
<root>/logs/wandb_sync_state.json, so restarts resume the same W&B runs (resume="allow").
"""
import argparse, glob, json, os, time, traceback, uuid

import wandb

CONFIG_KEYS = ('shas', 'seed', 'budget', 'gpu', 'started_utc')
BACKOFF = 300      # seconds before retrying a file whose W&B call failed
IDLE_CLOSE = 1800   # close (not finish-as-done) a run with no new lines for this long; it resumes if lines come


def log(msg):
    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}', flush=True)


def flatten(d, prefix=''):
    out = {}
    for k, v in d.items():
        key = f'{prefix}{k}'
        if isinstance(v, dict):
            out.update(flatten(v, key + '.'))
        else:
            out[key] = v
    return out


def load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def save_state(path, state):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, path)


class Syncer:
    def __init__(self, a):
        self.a = a
        self.state_path = a.state or os.path.join(a.root, 'logs', 'wandb_sync_state.json')
        self.state = load_json(self.state_path, {'files': {}})
        self.runs = {}        # metrics path -> live wandb Run
        self.last_io = {}     # metrics path -> last time new data was logged
        self.wandb_dir = os.path.join(a.root, 'logs', 'wandb_local')
        os.makedirs(self.wandb_dir, exist_ok=True)
        self.backoff = {}     # metrics path -> time before which a failed wandb.init is not retried

    def discover(self):
        found = []
        for arm in self.a.arms.split(','):
            base = os.path.join(self.a.root, arm)
            for p in glob.glob(os.path.join(base, '**', 'runs', '*', 'metrics.jsonl'), recursive=True):
                rel = os.path.relpath(p, base).split(os.sep)     # [ws..., 'runs', name, 'metrics.jsonl']
                if 'frozen' in rel[:-3]:
                    continue
                ws = '-'.join(rel[:-3]) or 'root'
                found.append((arm, ws, rel[-2], p))
        return found

    def run_for(self, path, st):
        r = self.runs.get(path)
        if r is not None:
            return r
        cfg = {'arm': st['arm'], 'workspace': st['ws'], 'metrics_path': path}
        summ = load_json(os.path.join(os.path.dirname(path), 'summary.json'), None)
        if isinstance(summ, dict):
            cfg.update({k: summ[k] for k in CONFIG_KEYS if k in summ})
        r = wandb.init(project=self.a.project, entity=self.a.entity, id=st['run_id'], resume='allow',
                       name=st['name'], group=st['arm'], job_type=st['ws'], config=cfg,
                       reinit='create_new', dir=self.wandb_dir,
                       settings=wandb.Settings(silent=True, console='off'))
        r.define_metric('elapsed')
        self.runs[path] = r
        log(f'wandb run {st["name"]} ({st["run_id"]}) opened: {r.url}')
        return r

    def log_record(self, run, st, rec):
        step = rec.get('step')
        last = st.get('last_step', -1)
        if isinstance(step, int) and not isinstance(step, bool) and step >= last:
            wstep = step
        else:
            wstep = last + 1
        phase = rec.get('phase') or 'misc'
        data = {}
        for k, v in rec.items():
            if k in ('phase', 't'):
                continue
            if k == 'step':
                data['rl_step'] = v
            elif k == 'elapsed':
                data['elapsed'] = v
            elif k == 'fail_reasons' and isinstance(v, list):
                rows = [[str(x[0]), x[1]] for x in v if isinstance(x, (list, tuple)) and len(x) == 2]
                data[f'{phase}/fail_reasons'] = wandb.Table(columns=['reason', 'count'], data=rows)
            elif isinstance(v, dict):
                data.update({f'{phase}/{kk}': vv for kk, vv in flatten(v, f'{k}.').items()
                             if isinstance(vv, (int, float, bool)) or vv is None})
            elif isinstance(v, (int, float, bool, str)) or v is None:
                data[f'{phase}/{k}'] = v
            else:
                data[f'{phase}/{k}'] = json.dumps(v)[:1000]
        if phase == 'eval' and 'heldout_greedy' in rec:
            st['heldout_greedy'] = rec['heldout_greedy']
        run.log(data, step=wstep)
        st['last_step'] = wstep

    def sync_file(self, arm, ws, name, path):
        files = self.state['files']
        st = files.get(path)
        size = os.path.getsize(path)
        if st is None or size < st.get('offset', 0):
            if st is not None:
                log(f'{path} shrank ({size} < {st["offset"]}); starting a new W&B run')
            st = files[path] = {'arm': arm, 'ws': ws, 'name': f'{arm}/{ws}/{name}', 'offset': 0,
                                'run_id': uuid.uuid4().hex[:12], 'last_step': -1, 'finished': False}
            self.close(path)
        summ_path = os.path.join(os.path.dirname(path), 'summary.json')
        summ = load_json(summ_path, None) if os.path.exists(summ_path) else None
        has_new = size > st['offset']
        done_now = isinstance(summ, dict) and 'status' in summ and not st['finished']
        if not has_new and not done_now:
            return False
        run = self.run_for(path, st)
        n = 0
        if has_new:
            with open(path, 'rb') as f:
                f.seek(st['offset'])
                chunk = f.read(size - st['offset'])
            end = chunk.rfind(b'\n')
            if end >= 0:
                for line in chunk[:end + 1].splitlines():
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                        if isinstance(rec, dict):
                            self.log_record(run, st, rec)
                            n += 1
                    except ValueError:
                        log(f'{path}: skipping malformed line at offset ~{st["offset"]}')
                st['offset'] += end + 1
                self.last_io[path] = time.time()
            if n:
                log(f'{st["name"]}: +{n} records (step {st["last_step"]})')
        if isinstance(summ, dict) and 'status' in summ and not st['finished']:
            run.config.update({k: summ[k] for k in CONFIG_KEYS if k in summ}, allow_val_change=True)
            rs = summ.get('rl_stats') or {}
            hg = summ.get('heldout_greedy', rs.get('heldout_greedy', st.get('heldout_greedy')))
            final = {'status': summ.get('status'), 'dev_score': summ.get('metric'),
                     'heldout_greedy': hg, 'rl_secs': summ.get('rl_secs'), 'score_secs': summ.get('score_secs')}
            for L, v in (summ.get('by_len') or {}).items():
                final[f'dev_by_len.{L}'] = v[0] if isinstance(v, (list, tuple)) else v
            for k, v in flatten(rs, 'rl_stats.').items():
                if isinstance(v, (int, float, bool, str)):
                    final[k] = v
            if summ.get('metric') is not None:
                run.config.update({'dev_score': summ.get('metric')}, allow_val_change=True)
            for k, v in final.items():
                if v is not None:
                    run.summary[k] = v
            st['finished'] = True
            log(f'{st["name"]}: finished status={final["status"]} dev_score={final["dev_score"]} '
                f'heldout_greedy={hg}')
            self.close(path)
        return True

    def close(self, path):
        r = self.runs.pop(path, None)
        if r is not None:
            try:
                r.finish()
            except Exception:
                log(f'finish failed for {path}\n{traceback.format_exc()}')
        self.last_io.pop(path, None)

    def poll(self):
        changed = False
        for arm, ws, name, path in self.discover():
            if time.time() < self.backoff.get(path, 0):
                continue
            try:
                changed |= self.sync_file(arm, ws, name, path)
                self.backoff.pop(path, None)
            except Exception as e:
                changed = True   # a new state entry may have been created
                log(f'sync error {path}: {type(e).__name__}: {str(e)[:300]} (retry in {BACKOFF}s)')
                self.backoff[path] = time.time() + BACKOFF
                self.close(path)
        now = time.time()
        for path, t in list(self.last_io.items()):
            if now - t > IDLE_CLOSE:
                log(f'{path}: idle {IDLE_CLOSE}s, closing W&B run (will resume on new data)')
                self.close(path)
        if changed:
            save_state(self.state_path, self.state)

    def shutdown(self):
        save_state(self.state_path, self.state)
        for p in list(self.runs):
            self.close(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/scratch/cowifornia/nd-rl/ab')
    ap.add_argument('--project', default='nd-rl-hillclimb')
    ap.add_argument('--entity', default=None)
    ap.add_argument('--arms', default='gear,orch')
    ap.add_argument('--state', default=None, help='state file (default <root>/logs/wandb_sync_state.json)')
    ap.add_argument('--poll', type=float, default=20)
    ap.add_argument('--once', action='store_true', help='one pass, close all runs, exit (for tests)')
    a = ap.parse_args()
    if os.environ.get('WANDB_MODE') not in ('offline', 'disabled'):
        if not wandb.login(anonymous='never', verify=True):
            raise SystemExit('wandb login failed')
        a.entity = a.entity or os.environ.get('WANDB_ENTITY') or wandb.Api().default_entity
    log(f'entity {a.entity!r}')
    s = Syncer(a)
    log(f'mirroring {a.arms} under {a.root} to project {a.project}; state {s.state_path}')
    try:
        while True:
            s.poll()
            if a.once:
                break
            time.sleep(a.poll)
    except KeyboardInterrupt:
        log('interrupted')
    finally:
        s.shutdown()


if __name__ == '__main__':
    main()
