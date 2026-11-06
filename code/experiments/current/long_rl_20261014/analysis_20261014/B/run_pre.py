"""Agent B: sample failed attempts of looped-ei s0 at n_loops 1 and 8 on a stratified target set.
Usage: ND_ARCH=looped LEAN_GATE_WORKERS=2 python run_samples.py select|sample <loops> <k>
"""
import sys, os, json, random, hashlib, time
sys.path.insert(0, '/private/tmp/claude-501/-Users-robbie-repos-nd-rl/a1d02fb4-ba43-48c4-838b-e227293b6c2a/scratchpad/probe')
B = os.path.dirname(os.path.abspath(__file__))
E = '/Users/robbie/repos/nd-rl/code/experiments/current/long_rl_20261014'
A = '/Users/robbie/repos/nd-rl/artifacts/long_rl_20261014/v2/long-rl-20261014-v2_looped-ei_s0'
FORK = '/private/tmp/claude-501/-Users-robbie-repos-nd-rl/a1d02fb4-ba43-48c4-838b-e227293b6c2a/scratchpad/fork51604b3'


def read(fn):
    return [json.loads(l) for l in open(fn) if l.strip()]


def select():
    rng = random.Random(7)
    dev = read(f'{E}/data/dev.jsonl')
    ev = {r['name']: r for r in read(f'{A}/eval_dev.jsonl')}
    old = [r for r in read(f'{FORK}/data/ladder/transfer.jsonl')
           if int(hashlib.sha1(r['key'].encode()).hexdigest(), 16) % 2 == 0]
    evo = read(f'{A}/eval_old_dev.jsonl')
    assert len(old) == len(evo) == 1108
    for o, e in zip(old, evo):
        assert e['name'] == o['name'], (e['name'], o['name'])
        o['body'] = o['n_lines'] - o['n_prem']
        o['solved'] = e['solved']; o['n_ok'] = e['n_ok']; o['proof'] = None
    rl = read(f'{E}/data/rl.jsonl')
    found = {r['name'] for r in read(f'{A}/found_targets.jsonl')}
    for r in rl:
        r['solved'] = r['name'] in found
    for r in dev:
        r['solved'] = ev[r['name']]['solved']; r['n_ok'] = ev[r['name']]['n_ok']

    def pick(pool, n, cond, tag):
        c = [r for r in pool if cond(r)]
        rng.shuffle(c)
        out = []
        for r in c[:n]:
            out.append({'pool': tag, 'name': r['name'], 'prompt': r['prompt'], 'body': r['body'], 'n_prem': r['n_prem'],
                        'proof': r.get('proof'), 'solved_eval': r['solved'], 'src': r.get('src') or r.get('schema'),
                        'key': r['key']})
        print(f'{tag}: {len(c)} candidates, took {len(out)}', file=sys.stderr)
        return out

    sel = []
    sel += pick(old, 14, lambda r: not r['solved'] and r['body'] <= 9, 'old_unsolved_short')
    sel += pick(old, 6, lambda r: not r['solved'] and r['body'] >= 10, 'old_unsolved_mid')
    sel += pick(dev, 10, lambda r: not r['solved'] and r['body'] <= 14, 'dev_unsolved_short')
    sel += pick(dev, 6, lambda r: not r['solved'] and 15 <= r['body'] <= 20, 'dev_unsolved_mid')
    fam = lambda r: r['src'].split(':')[0].replace('_f20.jsonl', '').replace('.jsonl', '')
    sel += pick(rl, 4, lambda r: not r['solved'] and r['body'] <= 9 and fam(r) == 'ladder', 'rl_unsolved_short_ladder')
    sel += pick(rl, 2, lambda r: not r['solved'] and 10 <= r['body'] <= 12 and fam(r) == 'ladder', 'rl_unsolved_short_ladder')
    sel += pick(rl, 3, lambda r: not r['solved'] and r['body'] <= 12 and fam(r) == 'd', 'rl_unsolved_short_d')
    sel += pick(rl, 3, lambda r: not r['solved'] and r['body'] <= 12 and fam(r) in ('s', 's2'), 'rl_unsolved_short_s')
    sel += pick(rl, 2, lambda r: not r['solved'] and 13 <= r['body'] <= 20 and fam(r) == 'c', 'rl_unsolved_mid_c')
    sel += pick(rl, 2, lambda r: not r['solved'] and 13 <= r['body'] <= 20 and fam(r) in ('s', 's2'), 'rl_unsolved_mid_s')
    sel += pick(rl, 2, lambda r: not r['solved'] and 13 <= r['body'] <= 20 and fam(r) == 'd', 'rl_unsolved_mid_d')
    sel += pick(rl, 1, lambda r: not r['solved'] and 13 <= r['body'] <= 20 and fam(r).startswith('b'), 'rl_unsolved_mid_b')
    sel += pick(old, 3, lambda r: r['solved'] and r['body'] >= 6 and r['n_ok'] <= 8, 'old_solved_ctl')
    sel += pick(dev, 4, lambda r: r['solved'] and 10 <= r['body'] <= 16 and r['n_ok'] <= 20, 'dev_solved_ctl')
    sel += pick(rl, 3, lambda r: r['solved'] and 10 <= r['body'] <= 16, 'rl_solved_ctl')
    with open(f'{B}/targets.jsonl', 'w') as f:
        for r in sel:
            f.write(json.dumps(r) + '\n')
    print(len(sel), 'targets', file=sys.stderr)


def sample(loops, k, seed=0):
    import torch
    torch.set_num_threads(2)
    import ndmodel as M
    model, tok = M.load(f'{E}/resume/looped-ei_s0/pretrain.pt')
    model.n_loops = loops
    recs = read(f'{B}/targets.jsonl')
    out = f'{B}/samples_PRE_L{loops}_k{k}_s{seed}.jsonl'
    done = {r['name'] for r in read(out)} if os.path.exists(out) else set()
    todo = [r for r in recs if r['name'] not in done]
    # shorter first so partial results are usable
    todo.sort(key=lambda r: r['body'])
    with open(out, 'a') as f:
        for i in range(0, len(todo), 8):
            chunk = todo[i:i + 8]
            t0 = time.time()
            res = M.sample(model, tok, chunk, k=k, seed=seed)
            for r, samples in zip(chunk, res):
                f.write(json.dumps({'name': r['name'], 'pool': r['pool'], 'body': r['body'], 'loops': loops,
                                    'samples': [{'nd': nd, 'ok': ok, 'raw': raw} for nd, ok, raw in samples]}) + '\n')
            f.flush()
            print(f'[{time.strftime("%H:%M:%S")}] loops {loops}: {i + len(chunk)}/{len(todo)} targets, '
                  f'{time.time() - t0:.0f}s for {len(chunk) * k} samples, ok {sum(ok for s in res for _, ok, _ in s)}',
                  flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'select':
        select()
    else:
        sample(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 0)
