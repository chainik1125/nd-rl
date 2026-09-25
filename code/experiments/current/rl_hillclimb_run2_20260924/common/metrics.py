"""FROZEN per-step metrics logger (JSONL only; W&B mirroring is done host-side by wandb_sync.py).

    m = Metrics(out_dir, run_name, arm)
    m.log(phase='sample', round=1, samples=48000, accept_rate=0.01, ...)
    m.set_summary(rounds=4, targets_solved_cum=812)

Every record carries t (unix seconds), elapsed (seconds since the harness started, or since this
object was built), step (monotone int; an internal counter if the caller does not pass one) and
phase in {sample, train, eval, summary}. See SPEC.md "Metrics schema" for the per-phase fields.
"""
import json, math, os, time

PHASES = ('sample', 'train', 'eval', 'summary')


def _clean(x):
    """Make x JSON-serialisable (tensors / numpy scalars -> python numbers; NaN/inf -> None)."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if hasattr(x, 'item') and callable(x.item):
        try:
            x = x.item()
        except Exception:
            return str(x)
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


class Metrics:
    def __init__(self, out_dir, run_name, arm, wandb_project='nd-rl-hillclimb', t0=None):
        self.out_dir, self.run_name, self.arm, self.wandb_project = out_dir, run_name, arm, wandb_project
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, 'metrics.jsonl')
        self.t0 = time.time() if t0 is None else t0
        self.step = 0
        self.summary = {}

    def log(self, **fields):
        phase = fields.get('phase')
        if phase not in PHASES:
            raise ValueError(f'metrics.log needs phase in {PHASES}, got {phase!r}')
        if fields.get('step') is None:
            self.step += 1
            fields['step'] = self.step
        else:
            fields['step'] = int(fields['step'])
            self.step = max(self.step, fields['step'])
        now = time.time()
        rec = {'t': now, 'elapsed': now - self.t0, **_clean(fields)}
        with open(self.path, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        return rec

    def set_summary(self, **kv):
        self.summary.update(_clean(kv))
        return self.summary


def read_metrics(path, phase=None):
    """path: a metrics.jsonl file or a run directory. Returns the list of records (optionally one phase)."""
    if os.path.isdir(path):
        path = os.path.join(path, 'metrics.jsonl')
    out = []
    with open(path) as f:
        for l in f:
            if l.strip():
                r = json.loads(l)
                if phase is None or r.get('phase') == phase:
                    out.append(r)
    return out
