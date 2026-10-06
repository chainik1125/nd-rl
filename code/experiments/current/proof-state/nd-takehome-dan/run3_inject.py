#!/usr/bin/env python3
"""Run 3: inject a small set of OUTSIDE records into one expert-iteration training step from a non-ignited arm's round-4
state, then continue expert iteration.  Generalises ignition_transfer_mix.py (which injected a sibling's whole found_4).

  python run3_inject.py --own artifacts/p2/<arm> --init ckpts/p2/<arm>_r4.pt --train data/p2/train_X.jsonl --seed S \
      --inject FILE.jsonl --n 4 --pattern depth3 --out ckpts/r3/<arm>_<cond>_r4t.pt --tag <cond>
--inject records: {prompt, proof, n_lines[, ...]}; for a sibling found file the pattern-containing proofs of --n distinct
theorems are drawn (seeded); for other files the first --n records are used. Each injected record is repeated --rl_weight
times (like an RL proof), mixed with the arm's own found_4 (<= 4 proofs per theorem, x4) and --retain random Stage-1
records; one train.py step identical to a round's fine-tune (600 steps, lr 3e-4, --cap 0: NO verification, so the
invalid-string control (d) is accepted by train.py — see run3.md). The mix and a manifest are written next to --own.
"""
import argparse, json, os, random, subprocess, collections, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patterns import classify
from nd_verify import verify_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--own', required=True); ap.add_argument('--init', required=True); ap.add_argument('--train', required=True)
    ap.add_argument('--seed', type=int, required=True); ap.add_argument('--inject', required=True); ap.add_argument('--n', type=int, required=True)
    ap.add_argument('--pattern', required=True); ap.add_argument('--out', required=True); ap.add_argument('--tag', required=True)
    ap.add_argument('--sibling_mode', action='store_true', help='--inject is a found_<r>.jsonl: draw --n pattern proofs of distinct theorems')
    ap.add_argument('--round', type=int, default=4); ap.add_argument('--retain', type=int, default=20000); ap.add_argument('--max_per_thm', type=int, default=4); ap.add_argument('--rl_weight', type=int, default=4)
    a = ap.parse_args()
    rng = random.Random(a.seed * 1000 + a.round + hash(a.tag) % 997)
    own = collections.defaultdict(list)
    fn_own = f'{a.own}/found_{a.round}.jsonl'
    if os.path.exists(fn_own):
        for l in open(fn_own):
            x = json.loads(l); own[x['name']].append(x)
    inj = [json.loads(l) for l in open(a.inject) if l.strip()]
    if a.sibling_mode:
        by = collections.defaultdict(list)
        for x in inj:
            if classify(x['proof'])[a.pattern]:
                by[x['name']].append(x)
        names = sorted(by); rng.shuffle(names)
        inj = [rng.choice(by[n]) for n in names[:a.n]]
    else:
        inj = inj[:a.n]
    assert len(inj) == a.n, (len(inj), a.n)
    valid = [verify_text(x['prompt'] + ' ' + x['proof'])[0] for x in inj]
    haspat = [bool(classify(x['proof']) and classify(x['proof'])[a.pattern]) for x in inj]
    train = [json.loads(l) for l in open(a.train) if l.strip()]
    mix = f'{a.own}_{a.tag}_mix.jsonl'
    n_own = 0
    with open(mix, 'w') as f:
        for name, fs in own.items():
            fs = list(fs); rng.shuffle(fs)
            for x in fs[:a.max_per_thm]:
                for _ in range(a.rl_weight):
                    f.write(json.dumps({'prompt': x['prompt'], 'proof': x['proof'], 'n_lines': x['written'], 'src': 'own'}) + '\n')
                n_own += 1
        for x in inj:
            for _ in range(a.rl_weight):
                f.write(json.dumps({'prompt': x['prompt'], 'proof': x['proof'], 'n_lines': x.get('written', x.get('n_lines', 0)), 'src': a.tag}) + '\n')
        for x in rng.sample(train, min(a.retain, len(train))):
            f.write(json.dumps({'prompt': x['prompt'], 'proof': x['proof'], 'n_lines': x['n_lines']}) + '\n')
    man = {'own': a.own, 'own_proofs': n_own, 'own_theorems': len(own), 'inject_file': a.inject, 'tag': a.tag, 'n_injected': len(inj), 'injected_valid': valid,
           'injected_has_pattern': haspat, 'injected': [{'prompt': x['prompt'], 'proof': x['proof']} for x in inj], 'init': a.init, 'out': a.out, 'retain': min(a.retain, len(train))}
    json.dump(man, open(f'{a.own}_{a.tag}_manifest.json', 'w'), indent=1)
    print(f'mix {mix}: own {n_own} proofs / {len(own)} theorems; injected {len(inj)} ({a.tag}; valid {sum(valid)}, pattern {sum(haspat)}); retained {man["retain"]}', flush=True)
    cmd = ['python3', 'train.py', '--data', mix, '--init', a.init, '--steps', '600', '--lr', '3e-4', '--min_lr', '3e-5', '--warmup', '50', '--cap', '0', '--out', a.out, '--seed', str(a.seed * 1000 + a.round), '--log_every', '200']
    print(' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
