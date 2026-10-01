#!/usr/bin/env python3
"""Results registry (proposal 15 §5): one JSON row per headline number, and checkpoints to the bucket on save.

  import record
  record.set_config(vars(a), arm='ei', role='rl')          # once, after argparse: the script's full config
  record.record('solved_frac', 0.912, n=5000, split='heldout', decode='greedy', ckpt=a.ckpt, data=a.inp,
                source=a.summary)

Every row carries run id (ND_RUN_ID), arm, seed, role, git SHA, the script's config, the dataset file + md5, the
checkpoint + md5 + bucket URI, metric, value, n, the source file the number can be re-derived from, host and UTC.
Rows are appended to  artifacts/<run-id>/registry/<utc>_<host>_<pid>.jsonl  (one file per process, so jobs on two
pods, or two jobs on one pod, never overwrite each other's rows when pulled) and synced to
  hf://buckets/dan-pandori/nd-rl/registry/<run-id>/<same file name>
at most every ND_REGISTRY_SYNC_S seconds (default 120) and at exit.  `registry_merge.py` builds one table.

`publish_ckpt(path)` (called by model.save_ckpt) uploads a checkpoint to
  hf://buckets/dan-pandori/nd-rl/<run-id>/ckpts/<path below ckpts/>
checks the bucket's size, writes <path>.upload.json {md5, bytes, uri, utc} and a `ckpt_saved` row, and raises if
ND_RUN_ID is unset or the upload fails.  Environment:
  ND_RUN_ID            run id (required for uploads)
  ND_OFFLINE=1         explicit opt-out: no uploads, no syncs (tests, local scratch); rows are still written
  ND_REGISTRY=0        write no rows at all;  ND_REGISTRY_SYNC=0: write rows, never sync them (tests)
  ND_ARM, ND_SEED      default arm / seed labels (a label passed to set_config/record wins)
  ND_GIT_SHA           git SHA when the code has no .git (pods: rsync excludes it); else .git_sha, else git
  ND_REGISTRY_DIR      override the row directory; ND_BUCKET overrides the bucket root
See REGISTRY.md.
"""
import os, sys, json, time, socket, hashlib, subprocess, shutil, atexit

ROOT = os.path.dirname(os.path.abspath(__file__))
BUCKET = os.environ.get('ND_BUCKET', 'hf://buckets/dan-pandori/nd-rl').rstrip('/')
SCHEMA = 1
TOP = ('arm', 'seed', 'role', 'split', 'ckpt', 'data', 'source')   # labels promoted to columns; the rest go in `labels`

_ctx = {'config': None, 'labels': {}, 'script': os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else None}
_state = {'file': None, 'last_sync': 0.0, 'dirty': False, 'warned': False, 'git': None}
_md5 = {}


def offline():
    return os.environ.get('ND_OFFLINE', '') not in ('', '0')


def run_id(required=False):
    r = os.environ.get('ND_RUN_ID', '').strip()
    if not r and required:
        raise RuntimeError('ND_RUN_ID is not set: checkpoints are uploaded to hf://…/<ND_RUN_ID>/ckpts/ on save. '
                           'Export ND_RUN_ID=<run id>, or ND_OFFLINE=1 for a test that must not upload.')
    return r or None


def md5_file(path):
    """md5 of a file, cached on (realpath, size, mtime_ns)."""
    st = os.stat(path)
    key = (os.path.realpath(path), st.st_size, st.st_mtime_ns)
    if key not in _md5:
        h = hashlib.md5()
        with open(path, 'rb') as f:
            for b in iter(lambda: f.read(1 << 20), b''):
                h.update(b)
        _md5[key] = h.hexdigest()
    return _md5[key]


def git_sha():
    if _state['git'] is None:
        sha, dirty = os.environ.get('ND_GIT_SHA'), None
        if not sha and os.path.exists(os.path.join(ROOT, '.git_sha')):
            sha = open(os.path.join(ROOT, '.git_sha')).read().strip()
        if not sha:
            try:
                sha = subprocess.run(['git', '-C', ROOT, 'rev-parse', 'HEAD'], capture_output=True, text=True,
                                     timeout=10).stdout.strip() or None
                if sha:
                    dirty = bool(subprocess.run(['git', '-C', ROOT, 'status', '--porcelain', '-uno', '--', '*.py'],
                                                capture_output=True, text=True, timeout=20).stdout.strip())
            except Exception:
                sha = None
        _state['git'] = (sha, dirty)
    return _state['git']


def hf_bin():
    b = os.environ.get('ND_HF_BIN') or shutil.which('hf') or os.path.expanduser('~/.local/bin/hf')
    if not os.path.exists(b):
        raise RuntimeError('the `hf` CLI is not installed (pods: pip install --break-system-packages -U huggingface_hub)')
    return b


def bucket_cp(local, uri, tries=3, verify=True):
    """Upload one file; raise unless the bucket then holds a file of the same size."""
    err = ''
    for i in range(tries):
        p = subprocess.run([hf_bin(), 'buckets', 'cp', local, uri], capture_output=True, text=True, timeout=1800)
        if p.returncode == 0:
            if not verify or remote_size(uri) == os.path.getsize(local):
                return
            err = f'size mismatch after upload: remote {remote_size(uri)} != local {os.path.getsize(local)}'
        else:
            err = (p.stderr or p.stdout).strip()[-400:]
        time.sleep(2 * (i + 1))
    raise RuntimeError(f'upload FAILED {local} -> {uri}: {err}')


def remote_size(uri):
    p = subprocess.run([hf_bin(), 'buckets', 'list', uri, '--json'], capture_output=True, text=True, timeout=300)
    try:
        es = json.loads(p.stdout)
    except Exception:
        return None
    path = uri.split('/', 5)[-1]    # hf://buckets/<ns>/<bucket>/<path>
    for e in es if isinstance(es, list) else []:
        if e.get('type') == 'file' and (e.get('path') == path or path.endswith('/' + e.get('path', '\0'))):
            return e.get('size')
    return None


def ckpt_uri(path, rid):
    rel = os.path.relpath(os.path.abspath(path), ROOT)
    if rel.startswith('..'):
        rel = 'ckpts/_ext/' + os.path.basename(path)
    elif not rel.startswith('ckpts/'):
        rel = 'ckpts/' + rel
    return f'{BUCKET}/{rid}/{rel}'


def sidecar(path):
    return path + '.upload.json'


def ckpt_info(path):
    """(md5, uri) of a checkpoint: from its upload sidecar when it is current, else the md5 alone."""
    if not path or not os.path.isfile(path):
        return None, None
    sc = sidecar(path)
    if os.path.exists(sc):
        try:
            s = json.load(open(sc))
            if s.get('bytes') == os.path.getsize(path) and os.path.getmtime(sc) >= os.path.getmtime(path):
                return s['md5'], s.get('uri')
        except Exception:
            pass
    return md5_file(path), None


def preflight():
    """Call at the start of any script that will save checkpoints: fail now, not after hours of training, if the
    upload save_ckpt performs cannot happen (no ND_RUN_ID, no `hf` CLI).  No-op with ND_OFFLINE=1."""
    if not offline():
        run_id(required=True)
        hf_bin()


def publish_ckpt(path, step=None):
    """Upload a just-written checkpoint and record it. Raises on failure unless ND_OFFLINE=1."""
    md5, nbytes = md5_file(path), os.path.getsize(path)
    uri = None
    t0 = time.time()
    if not offline():
        rid = run_id(required=True)
        uri = ckpt_uri(path, rid)
        bucket_cp(path, uri)
    info = {'md5': md5, 'bytes': nbytes, 'uri': uri, 'utc': time.strftime('%FT%TZ', time.gmtime()),
            'upload_s': round(time.time() - t0, 2), 'offline': offline()}
    with open(sidecar(path), 'w') as f:
        json.dump(info, f)
    record('ckpt_saved', nbytes, ckpt=path, source=path, step=step, upload_s=info['upload_s'])
    if uri:
        print(f'uploaded {path} -> {uri} (md5 {md5}, {info["upload_s"]:.1f}s)', flush=True)
    return info


def set_config(cfg, **labels):
    """Register the script's full config (e.g. vars(args)) and default labels for every later row."""
    _ctx['config'] = {k: v for k, v in dict(cfg or {}).items()}
    _ctx['labels'].update({k: v for k, v in labels.items() if v is not None})
    if labels.get('arm') is not None and not os.environ.get('ND_ARM'):
        os.environ['ND_ARM'] = str(labels['arm'])    # child processes (the EI drivers' train.py calls) inherit the arm


def save_config(cfg, out, **labels):
    """set_config, and write the resolved config next to the outputs (run repo-hygiene): `<out>/args.json` if `out` is an
    existing directory or ends in '/', else `<out>.args.json` (an output file or prefix).  The file is `vars(args)` as before (readers index it
    directly) plus one key `_meta` {script, argv, run_id, git_sha, git_dirty, host, utc}.  Every later row names the file (label `config_file`).
    Returns the path written, or None when `out` is None (output to stdout)."""
    set_config(cfg, **labels)
    job_compute()    # the process's compute rows (gpu_seconds, counters), written at exit
    if not out:
        return None
    out = str(out)
    path = os.path.join(out, 'args.json') if (os.path.isdir(out) or out.endswith('/')) else out + '.args.json'
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sha, dirty = git_sha()
    with open(path, 'w') as f:
        json.dump({**_ctx['config'], '_meta': {'script': _ctx['script'], 'argv': sys.argv, 'run_id': run_id(),
                   'git_sha': sha, 'git_dirty': dirty, 'host': socket.gethostname(),
                   'utc': time.strftime('%FT%TZ', time.gmtime())}}, f, indent=1, default=str)
    _ctx['labels']['config_file'] = _rel(path)
    return path


# ---- compute (AGENT_POLICY 2026-09-29: record the compute behind every arm) ------------------------------------
# Rows: gpu_seconds (always) and each counter that is non-zero: gen_tokens, attempts, actions, train_steps,
# train_tokens, lean_checks (label lean_s: summed Lean process seconds).  Every row of one block carries the labels
# phase, round (labels.round), device ('cuda'/'cpu'), gpu (device name), n_gpu, wall_s, compute_id.
# Time is EXCLUSIVE: a block's clock stops while a nested block (or a child process, `child()`) runs, so summing
# gpu_seconds over all rows of a run gives its total with nothing counted twice.  save_config() opens a process-level
# block (phase 'job', or ND_PHASE) closed at exit, so every script that registers its config records its compute; its
# clock starts at the process's creation (/proc), since the process holds its GPU from then.
# gpu_seconds = exclusive wall-clock x n_gpu when the process initialised CUDA (n_gpu = 1, or ND_N_GPU); on CPU it is
# the wall-clock with device='cpu' (one CPU process), and a CPU block with every counter zero writes no rows.
# ND_COMPUTE=0 turns it off.  See REGISTRY.md.
COUNTERS = ('gen_tokens', 'attempts', 'actions', 'train_steps', 'train_tokens', 'lean_checks', 'lean_s')
_blocks = []                                   # open compute blocks, innermost last


class Compute:
    def __init__(self, labels):
        self.labels = labels
        self.n = dict.fromkeys(COUNTERS, 0)
        self.active_s, self._t, self.closed, self.job = 0.0, None, False, False
        self.id = f'{socket.gethostname()}-{os.getpid()}-{time.time_ns() % 10 ** 12}'

    def __getattr__(self, k):                  # c.gen_tokens += n
        if k in COUNTERS:
            return self.n[k]
        raise AttributeError(k)

    def __setattr__(self, k, v):
        if k in COUNTERS:
            self.n[k] = v
        else:
            object.__setattr__(self, k, v)

    def _run(self):
        if self._t is None:
            self._t = time.perf_counter()

    def _stop(self):
        if self._t is not None:
            _cuda_sync()
            self.active_s += time.perf_counter() - self._t
            self._t = None

    def start(self):
        if _blocks:
            self.labels.setdefault('round', _blocks[-1].labels.get('round'))    # a phase inside a round block
            if self.labels['round'] is None:
                del self.labels['round']
            _blocks[-1]._stop()
        _blocks.append(self)
        _cuda_sync()
        self._run()
        return self

    def close(self, status='ok'):
        if self.closed:
            return
        self.closed = True
        self._stop()
        top = bool(_blocks) and _blocks[-1] is self
        if self in _blocks:
            _blocks.remove(self)
        if top and _blocks:
            _blocks[-1]._run()
        _write(self, status)

    def suspend(self):
        """Stop the clock and leave the stack (the block stays open; start() resumes it)."""
        if self in _blocks:
            top = _blocks[-1] is self
            self._stop()
            _blocks.remove(self)
            if top and _blocks:
                _blocks[-1]._run()

    def __enter__(self):
        return self.start()

    def __exit__(self, et, ev, tb):
        self.close('ok' if et is None else 'error')


def _cuda_sync():
    t = sys.modules.get('torch')
    try:
        if t is not None and t.cuda.is_available() and t.cuda.is_initialized():
            t.cuda.synchronize()
    except Exception:       # a sticky CUDA error must not cost the rows (the job's own traceback reports it)
        pass


def _device():
    """('cuda', device name, n_gpu) if this process initialised CUDA, else ('cpu', None, 0)."""
    t = sys.modules.get('torch')
    if t is not None and t.cuda.is_available() and t.cuda.is_initialized():
        try:
            name = t.cuda.get_device_name(t.cuda.current_device())
        except Exception:
            name = None
        return 'cuda', name, int(os.environ.get('ND_N_GPU', 1))
    return 'cpu', None, 0


def _write(c, status):
    dev, name, ngpu = _device()
    if dev == 'cpu' and not any(c.n.values()):
        return
    lab = {**c.labels, 'device': dev, 'gpu': name, 'n_gpu': ngpu, 'wall_s': round(c.active_s, 3),
           'compute_id': c.id, 'status': status}
    record('gpu_seconds', round(c.active_s * (ngpu or 1), 3), **lab)
    for k in COUNTERS[:-1]:
        if c.n[k]:
            record(k, c.n[k], **({**lab, 'lean_s': round(c.n['lean_s'], 3)} if k == 'lean_checks' else lab))
    if c.job:
        sync(raise_on_fail=False)     # the process-level block closes after _final_sync has run; other blocks' rows
                                      # follow record()'s ND_REGISTRY_SYNC_S rule and this final sync


def enabled():
    return os.environ.get('ND_COMPUTE', '1') != '0' and os.environ.get('ND_REGISTRY', '1') != '0'


def compute(phase=None, round=None, **labels):
    """A compute block: `with record.compute(phase='sample', round=r) as c: ... c.gen_tokens += n`.  arm / seed come
    from set_config / ND_ARM / ND_SEED unless given; round defaults to ND_ROUND (set for child processes by child())."""
    if round is None and os.environ.get('ND_ROUND', '').lstrip('-').isdigit():
        round = int(os.environ['ND_ROUND'])
    sd = os.environ.get('ND_COMPUTE_SEED', '')    # a child process's rows carry its driver's seed (child())
    lab = {'phase': phase or os.environ.get('ND_PHASE') or 'job', 'round': round,
           'seed': int(sd) if sd.lstrip('-').isdigit() else None, **labels}
    return Compute({k: v for k, v in lab.items() if v is not None})


def count(**kw):
    """Add to the innermost open block's counters (library code: sampler, Lean gate).  No-op when none is open."""
    if _blocks:
        n = _blocks[-1].n
        for k, v in kw.items():
            n[k] += v


def _proc_age():
    """Seconds since this process started (Linux /proc), else 0."""
    try:
        ticks = int(open('/proc/self/stat').read().rsplit(')', 1)[1].split()[19])
        return max(0.0, float(open('/proc/uptime').read().split()[0]) - ticks / os.sysconf('SC_CLK_TCK'))
    except Exception:
        return 0.0


def job_compute():
    """The process-level block (opened by save_config), closed at exit."""
    if not enabled() or any(b.job for b in _blocks):
        return None
    c = compute()
    c.job = True
    c.start()
    c._t -= _proc_age()    # the process held its GPU from exec, not from save_config (imports: ~2-5 s)
    atexit.register(c.close, 'exit')
    return c


_phases = {}                                   # (phase, round) -> open phase block


def phase(name, round=None, **labels):
    """Driver loops, without re-indenting: `record.phase('sample', round=r)` suspends the current phase block and
    starts (or resumes) the one for (name, round); blocks of other rounds are closed and written, so a loop that
    alternates phases every step still writes one set of rows per (phase, round).  phase(None) closes them all
    (also done at exit)."""
    if not _state.get('phase_atexit'):
        _state['phase_atexit'] = True
        atexit.register(phase, None)          # runs before the job block's closer (atexit is LIFO)
    key = (name, round)
    for k, b in list(_phases.items()):
        if k == key:
            continue
        if name is None or k[1] != round:
            b.close(); del _phases[k]
        else:
            b.suspend()
    if name is None or not enabled():
        return None
    if key not in _phases:
        _phases[key] = compute(name, round, **labels)
    b = _phases[key]
    if b not in _blocks:
        b.start()
    return b


class child:
    """`with record.child('finetune', round=r): subprocess.run(...)`: the child process records its own compute
    (its save_config block, labelled with this phase and round via ND_PHASE / ND_ROUND); this process's clock stops."""
    def __init__(self, phase, round=None):
        sd = _default('seed', ('seed',))
        self.env = {'ND_PHASE': phase, 'ND_ROUND': None if round is None else str(round),
                    'ND_COMPUTE_SEED': None if sd is None else str(sd)}

    def __enter__(self):
        self.old = {k: os.environ.get(k) for k in self.env}
        for k, v in self.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if _blocks:
            _blocks[-1]._stop()
        return self

    def __exit__(self, *e):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if _blocks:
            _blocks[-1]._run()


def _default(name, cfg_keys):
    if name in _ctx['labels']:
        return _ctx['labels'][name]
    env = os.environ.get('ND_' + name.upper())
    if env:
        return int(env) if name == 'seed' and env.lstrip('-').isdigit() else env
    cfg = _ctx['config'] or {}
    for k in cfg_keys:
        if cfg.get(k) is not None:
            return cfg[k]
    return None


def _rel(p):
    if not p:
        return p
    a = os.path.abspath(p)
    return os.path.relpath(a, ROOT) if a.startswith(ROOT + os.sep) else p


def _file():
    if _state['file'] is None:
        rid = run_id() or '_norun'
        d = os.environ.get('ND_REGISTRY_DIR') or os.path.join(ROOT, 'artifacts', rid, 'registry')
        os.makedirs(d, exist_ok=True)
        _state['file'] = os.path.join(d, f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{socket.gethostname()}_{os.getpid()}.jsonl")
        atexit.register(_final_sync)
    return _state['file']


def record(metric, value, n=None, **labels):
    """Append one row. `labels` may include arm, seed, role, split, ckpt, data, source and anything else."""
    if os.environ.get('ND_REGISTRY', '1') == '0':
        return None
    lab = dict(_ctx['labels'])
    lab.update({k: v for k, v in labels.items() if v is not None})
    row = {'schema': SCHEMA, 'run_id': run_id(), 'metric': metric, 'value': value, 'n': n}
    row['arm'] = lab.pop('arm', None) or _default('arm', ('arm',))
    row['seed'] = lab.pop('seed', None) if 'seed' in lab else _default('seed', ('seed',))
    row['role'] = lab.pop('role', None) or _default('role', ())
    row['split'] = lab.pop('split', None)
    ck = lab.pop('ckpt', None) or _default('ckpt', ('ckpt', 'init'))
    md5, uri = ckpt_info(ck) if ck else (None, None)
    row.update({'ckpt': _rel(ck), 'ckpt_md5': md5, 'ckpt_uri': uri})
    data = lab.pop('data', None) or _default('data', ('data', 'inp', 'targets', 'in'))
    row.update({'data': _rel(data), 'data_md5': md5_file(data) if data and os.path.isfile(data) else None})
    row['source'] = _rel(lab.pop('source', None))
    sha, dirty = git_sha()
    row.update({'script': _ctx['script'], 'git_sha': sha, 'git_dirty': dirty, 'config': _ctx['config'],
                'labels': lab, 'host': socket.gethostname(), 'utc': time.strftime('%FT%TZ', time.gmtime()),
                'backfilled': False})
    with open(_file(), 'a') as f:
        f.write(json.dumps(row, default=str) + '\n')
    _state['dirty'] = True
    if time.time() - _state['last_sync'] > float(os.environ.get('ND_REGISTRY_SYNC_S', 120)):
        sync(raise_on_fail=False)
    return row


def sync(raise_on_fail=True):
    """Upload this process's row file to hf://…/registry/<run-id>/. No-op offline or without a run id."""
    f, rid = _state['file'], run_id()
    _state['last_sync'] = time.time()
    if not f or not _state['dirty'] or offline() or not rid or os.environ.get('ND_REGISTRY_SYNC', '1') == '0':
        return
    try:
        bucket_cp(f, f'{BUCKET}/registry/{rid}/{os.path.basename(f)}', tries=2, verify=False)
        _state['dirty'] = False
    except Exception as e:
        msg = f'REGISTRY SYNC FAILED ({f}): {e}'
        if raise_on_fail:
            raise RuntimeError(msg)
        print(msg, file=sys.stderr, flush=True)


def _final_sync():
    try:
        sync(raise_on_fail=True)
    except Exception as e:
        print(f'!!! {e} — the rows are still in {_state["file"]}; pull them and run registry_merge.py --upload',
              file=sys.stderr, flush=True)


# ---- helpers the instrumented scripts call ----------------------------------------------------------------------
# Metric vocabulary (REGISTRY.md): <split>_greedy_acc, <split>_pass@<k>, <split>_solved_cum / _union, <split>_lstar_*,
# <split>_sample_acc; train_loss, val2k_loss; coverage_pass@<B>, coverage_sample_rate; ckpt_saved.
ROUND_KEYS = {'heldout_greedy': ('heldout', 'greedy'), 'transfer_greedy': ('transfer', 'greedy'),
              'targets_round': ('targets', 'pass@k'), 'transfer_round': ('transfer', 'pass@k'),
              'targets_cum': ('targets', 'cum'), 'transfer_cum': ('transfer', 'cum'), 'targets_union': ('targets', 'union')}


def split_of(path):
    """The evaluation file's stem: data/p2/heldout.jsonl -> 'heldout', data/ca/heldout_B_d3.jsonl -> 'heldout_B_d3'
    (a subset is never pooled with the full set; `data` holds the exact file)."""
    return os.path.basename(path or '').split('.')[0] or None


def ckpt_role(extra):
    """'stage1' for a from-scratch train.py checkpoint, 'rl' for a GRPO / fine-tuned one, None if unknown."""
    if not isinstance(extra, dict):
        return None
    if 'grpo_round' in extra:
        return 'rl'
    args = extra.get('args') or {}
    if 'init' in args:
        return 'finetune' if args.get('init') else 'stage1'
    return None


def model_seed(extra):
    """The training seed stored in a checkpoint's extra (train.py args), else None."""
    return ((extra or {}).get('args') or {}).get('seed') if isinstance(extra, dict) else None


def summary_rows(summ, split, decode, k=None, **labels):
    """eval_set.summarize() output -> one overall row and one row per length bin."""
    metric = f'{split}_greedy_acc' if decode == 'greedy' else f'{split}_pass@{k}'
    record(metric, summ['rate'], n=summ['n'], solved=summ['solved'], ci=summ.get('ci'), split=split, decode=decode,
           k=k, **labels)
    for L, b in summ.get('by_len', {}).items():
        record(metric, b['rate'], n=b['n'], solved=b['solved'], L=L, split=split, decode=decode, k=k, **labels)


def sdeval_rows(d, source, **labels):
    """sd_eval.py per-checkpoint summary -> per-slice <split>_greedy_acc and <split>_mean_term_size rows, labelled with
    the model (n_params, format, training seed, init) the summary itself records."""
    m = d.get('model') or {}
    ta = m.get('train_args') or {}
    lab = dict(ckpt=d.get('ckpt'), data=d.get('heldout'), source=source, n_params=m.get('n_params'),
               format=m.get('mode'), step=m.get('step'), train_data=ta.get('data'),
               hit_max_new=d.get('hit_max_new'), max_new=(d.get('sampler') or {}).get('max_new'))
    lab['role'] = labels.pop('role', None) or ('stage1' if ta and not ta.get('init') else ('finetune' if ta.get('init') else None))
    sd = labels.pop('seed', None)
    lab['seed'] = sd if sd is not None else ta.get('seed')
    lab.update(labels)
    split = split_of(d.get('heldout'))
    greedy = (d.get('sampler') or {}).get('greedy', True)
    for name, s in (d.get('slices') or {}).items():
        sl = {'L': name[3:]} if name.startswith('len') and name[3:].isdigit() else ({} if name == 'all' else {'slice': name})
        metric = f'{split}_greedy_acc' if greedy else f'{split}_sample_acc'
        record(metric, s['rate'], n=s['n'], solved=s['solved'], ci=s.get('ci'), split=split, decode='greedy' if greedy else 'sample', **sl, **lab)
        if s.get('mean_term_size') is not None:
            record(f'{split}_mean_term_size', s['mean_term_size'], n=s['solved'], split=split, **sl, **lab)
            record(f'{split}_mean_written_lines', s['mean_written_lines'], n=s['solved'], split=split, **sl, **lab)


def round_stats(stats, source, init=None, frozen=False, **labels):
    """One EI / GRPO round's stats dict -> overall rows (per-length detail stays in `source`).  The greedy numbers are
    measured on stats['ckpt'], the checkpoint the round sampled from (round 1: the initial checkpoint)."""
    r, k, ck = stats.get('round'), stats.get('k'), stats.get('ckpt')
    role = 'frozen' if frozen else ('init' if ck and init and os.path.abspath(ck) == os.path.abspath(init) else 'rl')
    labels = {'role': role, **labels}
    cfg = _ctx['config'] or {}
    for key, (split, dec) in ROUND_KEYS.items():
        s = stats.get(key)
        if not isinstance(s, dict) or 'solved' not in s:
            continue
        n = s.get('n')
        v = s.get('rate', s['solved'] / n if n else None)
        data = labels.get('data') or cfg.get(split)    # the arm's --heldout / --transfer / --targets file
        lab = {**labels, 'data': data}
        if split == 'heldout' and data:
            split = split_of(data)    # a held-out subset (e.g. heldout_B_d3) is never pooled with the full set
        metric = {'greedy': f'{split}_greedy_acc', 'pass@k': f'{split}_pass@{k}'}.get(dec, f'{split}_solved_{dec}')
        record(metric, v, n=n, solved=s['solved'], split=split, decode=dec, round=r, k=k, ckpt=ck, source=source, **lab)
        if s.get('lstar') is not None:
            record(f'{split}_lstar_{dec}', s['lstar'], n=n, split=split, decode=dec, round=r, k=k, ckpt=ck, source=source, **lab)
    for key in ('target_sample_acc', 'transfer_sample_acc'):
        if stats.get(key) is not None:
            split = 'targets' if key.startswith('target_') else 'transfer'
            record(f'{split}_sample_acc', stats[key], split=split, round=r, k=k, ckpt=ck, source=source,
                   **{**labels, 'data': labels.get('data') or cfg.get(split)})


def train_rows(a, last, extra):
    """train.py / fast_train.py / state_train.py: the final logged step's losses for checkpoint a.out."""
    lab = dict(ckpt=a.out, data=a.data, source=getattr(a, 'metrics', None) or a.out, step=(extra or {}).get('step'),
               secs=(extra or {}).get('secs'), n_params=(extra or {}).get('n_params'))
    if last.get('loss') is not None:
        record('train_loss', last['loss'], **lab)
    if last.get('val2k') is not None:
        record('val2k_loss', last['val2k'], n=2000, **lab)


if __name__ == '__main__':
    # python3 record.py publish <ckpt> ...   re-upload checkpoints whose upload failed (save_ckpt raised after writing)
    if len(sys.argv) > 2 and sys.argv[1] == 'publish':
        for p in sys.argv[2:]:
            print(publish_ckpt(p))
    else:
        print(__doc__)
