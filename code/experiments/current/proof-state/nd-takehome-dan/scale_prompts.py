#!/usr/bin/env python3
"""Run 1, step 3: novelty by model scale. For each class of RL-found proof from the campaign, the theorems (>= 30 where they
exist), rendered in the Lean form with the same 20 worked examples (draw 0 of lean_prompts.py), to be sampled k = 16 at
T = 0.7 by each Qwen3 instruct size; the smallest size that proves a theorem is its "scale of first success".
Classes: depth3_f0 (targets of `targets_depth3` solved with a depth-3 proof by the campaign's f = 0 arm a1 s0),
reductio_f0 (run 5's required reductio targets solved with the strict shape by f = 0 s0), transfer9 (transfer theorems whose
RL proof was written in 9 lines in the take-home run; base log-probabilities from Phase 1 attached when available).
  python scale_prompts.py --out data/r1/scale_prompts.jsonl
"""
import argparse, json, os, sys, random, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patterns import classify
from lean_prompts import build_messages


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='data/r1/scale_prompts.jsonl'); ap.add_argument('--per_class', type=int, default=40)
    a = ap.parse_args()
    rng = random.Random(0)
    P = [json.loads(l) for l in open('data/r1/prompts.jsonl')]
    ex_prompts = next(p for p in P if p['draw'] == 0 and p['form'] == 'lean')['examples']
    # rebuild the example (prompt, proof) pairs of draw 0 from the sources used by lean_prompts.build
    short = {r['prompt']: r['proof'] for r in (json.loads(l) for l in open('data/heldout.jsonl'))}
    longp = collections.defaultdict(list)
    for l in open('artifacts/ei_abs_s0_cont/found_transfer_16.jsonl'):
        x = json.loads(l); longp[x['prompt']].append(x['proof'])
    ex = []
    for ep in ex_prompts:
        ex.append((ep, short[ep] if ep in short else longp[ep][0]))
    classes = {}
    # depth3_f0: campaign f=0 arm a1 s0, targets solved with a depth-3 proof (written form)
    T3 = {json.loads(l)['name']: json.loads(l) for l in open('data/p2/targets_depth3.jsonl')}
    d3 = collections.defaultdict(list)
    for l in open('artifacts/p2/ei_depth3_f0_a1_s0/found_8.jsonl'):
        x = json.loads(l)
        c = classify(x['proof'])
        if c and c['depth3']: d3[x['name']].append(x)
    classes['depth3_f0'] = [{'name': n, 'prompt': T3[n]['prompt'], 'thm': T3[n]['thm'], 'rl_proof': v[0]['proof'], 'rl_round': min(y['round'] for y in v), 'gen_lines': T3[n]['n_lines']} for n, v in d3.items()]
    # reductio_f0: run 5 required pool, f=0 s0 solved (all strict)
    TR = {json.loads(l)['name']: json.loads(l) for l in open('data/p2/targets_reductio_req.jsonl')}
    rd = collections.defaultdict(list)
    for l in open('artifacts/r5/ei_reductio_f0_s0_req/found_8.jsonl'):
        x = json.loads(l); rd[x['name']].append(x)
    classes['reductio_f0'] = [{'name': n, 'prompt': TR[n]['prompt'], 'thm': TR[n]['thm'], 'rl_proof': v[0]['proof'], 'rl_round': min(y['round'] for y in v), 'gen_lines': TR[n]['n_lines'], 'schema': TR[n]['schema']} for n, v in rd.items()]
    # transfer9: take-home transfer theorems with a 9-line RL proof; base log-prob from Phase 1 when present
    lp = {}
    if os.path.exists('artifacts/novelty_phase1_theorems.jsonl'):
        for l in open('artifacts/novelty_phase1_theorems.jsonl'):
            x = json.loads(l); lp[x.get('name')] = x
    t9 = collections.defaultdict(list)
    for l in open('artifacts/ei_abs_s0_cont/found_transfer_16.jsonl'):
        x = json.loads(l)
        if x['written'] == 9: t9[x['name']].append(x)
    TT = {json.loads(l)['name']: json.loads(l) for l in open('data/transfer.jsonl')}
    classes['transfer9'] = [{'name': n, 'prompt': TT[n]['prompt'], 'thm': TT[n]['thm'], 'rl_proof': v[0]['proof'], 'rl_round': min(y['round'] for y in v), 'gen_lines': TT[n]['n_lines'], 'phase1': {k: v for k, v in lp[n].items() if k.startswith('base_logp') or k in ('n_proofs', 'min_written', 'first_round', 'solved_base_1e4', 'base_n_ok')} if n in lp else None} for n, v in t9.items()]
    out = []
    for cls, items in classes.items():
        rng.shuffle(items); items = items[:a.per_class]
        for it in items:
            out.append({'id': f'scale_{cls}_{it["name"]}', 'class': cls, 'theorem_name': it['name'], 'prompt': it['prompt'], 'thm': it['thm'], 'form': 'lean', 'draw': 0, 'bin': cls, 'src': cls,
                        'gen_lines': it['gen_lines'], 'rl_proof': it['rl_proof'], 'rl_round': it['rl_round'], 'phase1': it.get('phase1'), 'messages': build_messages('lean', ex, it['prompt']), 'examples': ex_prompts})
        print(cls, len(items), 'theorems (available', len(classes[cls]), ')')
    with open(a.out, 'w') as f:
        for x in out: f.write(json.dumps(x) + '\n')
    print(len(out), 'prompts ->', a.out)


if __name__ == '__main__':
    main()
