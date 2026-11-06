"""Where does the model first disagree with the gold proof? Per-token logp of the gold proof under the looped model at
n_loops 1 and 8; per gold line (have) the summed -logp, the first token where argmax != gold, and the rule there.
Writes goldlogp.jsonl. Usage: ND_ARCH=looped python goldlogp.py
"""
import sys, os, json, re, hashlib, random, collections
sys.path.insert(0, '/private/tmp/claude-501/-Users-robbie-repos-nd-rl/a1d02fb4-ba43-48c4-838b-e227293b6c2a/scratchpad/probe')
B = os.path.dirname(os.path.abspath(__file__))
E = '/Users/robbie/repos/nd-rl/code/experiments/current/long_rl_20261014'
A = '/Users/robbie/repos/nd-rl/artifacts/long_rl_20261014/v2/long-rl-20261014-v2_looped-ei_s0'
import torch
torch.set_num_threads(2)
import ndmodel as M


def read(fn):
    return [json.loads(l) for l in open(fn) if l.strip()]


def lines_of(tokens):
    """proof tokens -> per token (line index among haves, -1 for final exact; depth)"""
    li, depth, out, pend = -1, 0, [], 0
    for i, t in enumerate(tokens):
        if t == 'have':
            li += 1
        out.append((li, depth))
        if t == 'by':
            depth += 1
        if t == ')' and i >= 2 and tokens[i - 2] == 'exact' or (t == ')' and i >= 5 and tokens[i - 5] == 'exact' and tokens[i - 1] == ')'):
            depth = max(0, depth - 1)
    return out


def rule_of(tokens, li):
    """rule head of the li-th have"""
    idx = [i for i, t in enumerate(tokens) if t == 'have']
    if li < 0 or li >= len(idx): return 'exact'
    j = idx[li]
    k = j + 1
    while tokens[k] != ':=': k += 1
    t = tokens[k + 1]
    if t.startswith('h') and t[1:].isdigit(): return 'PR'
    if t == '⟨': return 'ANDI'
    if t in ('Or.inl', 'Or.inr'): return 'ORI'
    if t == 'Classical.byContradiction': return 'DN'
    if t == 'Or.elim': return 'ORE'
    if t == '(': return 'IMPI' if tokens[k + 1:k + 2] and 'False' not in tokens[k + 2:k + 20] else 'IMPI/NEGI'
    if t == 'False': return 'BOTE'
    if tokens[k + 2] in ('.1', '.2'): return 'ANDE'
    if tokens[k + 2] == ';': return 'R'
    return 'IMPE/NEGE'


def main():
    dev = read(f'{E}/data/dev.jsonl')
    ev = {r['name']: r for r in read(f'{A}/eval_dev.jsonl')}
    rng = random.Random(3)
    uns = [r for r in dev if not ev[r['name']]['solved'] and r['body'] <= 24]
    sol = [r for r in dev if ev[r['name']]['solved'] and r['body'] >= 10]
    rng.shuffle(uns); rng.shuffle(sol)
    recs = [dict(r, solved=False) for r in uns[:120]] + [dict(r, solved=True) for r in sol[:60]]
    tg = [r for r in read(f'{B}/targets.jsonl') if r.get('proof')]
    names = {r['name'] for r in recs}
    recs += [dict(r, solved=r['solved_eval']) for r in tg if r['name'] not in names]
    model, tok = M.load(f'{E}/resume/looped-ei_s0/final.pt')
    with open(f'{B}/goldlogp.jsonl', 'w') as f:
        for r in recs:
            row = {'name': r['name'], 'body': r['body'], 'solved': r['solved'], 'src': r.get('src')}
            for L in (1, 8):
                model.n_loops = L
                rows = M.token_logp(model, tok, r['prompt'], r['proof'])
                toks = [t for t, _, _ in rows]
                lmap = lines_of(toks)
                per_line = collections.defaultdict(float)
                first_any = first_struct = None
                for i, (t, lp, am) in enumerate(rows):
                    per_line[lmap[i][0]] += -lp
                    if am != t:
                        if first_any is None: first_any = i
                        if first_struct is None and not (t[0] == 'n' and t[1:].isdigit() and am[0] == 'n'):
                            first_struct = i
                n_have = toks.count('have')
                d = {'sum_nlp': -sum(lp for _, lp, _ in rows), 'n_tok': len(rows), 'n_have': n_have,
                     'worst_line': max(per_line, key=per_line.get), 'worst_line_nlp': max(per_line.values()),
                     'worst_line_rule': rule_of(toks, max(per_line, key=per_line.get)),
                     'nlp_by_rule': {}}
                byrule = collections.defaultdict(float); cnt = collections.Counter()
                for li, v in per_line.items():
                    ru = rule_of(toks, li); byrule[ru] += v; cnt[ru] += 1
                d['nlp_by_rule'] = {k: [round(v, 2), cnt[k]] for k, v in byrule.items()}
                for key, i in (('first_any', first_any), ('first_struct', first_struct)):
                    if i is not None:
                        li, dp = lmap[i]
                        d[key] = {'tok_i': i, 'line': li, 'frac': (li + 1) / max(1, n_have), 'depth': dp,
                                  'gold_tok': toks[i], 'model_tok': rows[i][2], 'logp_gold': round(rows[i][1], 2),
                                  'rule': rule_of(toks, li), 'ctx': ' '.join(toks[max(0, i - 12):i]) + ' [[' + toks[i] + ' vs ' + rows[i][2] + ']]'}
                    else:
                        d[key] = None
                row[f'L{L}'] = d
            f.write(json.dumps(row) + '\n'); f.flush()
    print('done', len(recs))


if __name__ == '__main__':
    main()
