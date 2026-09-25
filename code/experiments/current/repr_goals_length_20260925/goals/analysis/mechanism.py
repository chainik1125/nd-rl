#!/usr/bin/env python3
"""Mechanism diagnostic (CPU): on the test theorems whose conclusion is a conjunction A & B (55-95% of the std theorems at
L15-18, 38% at L12), what the attempts do, for the goals runs and the repr abs baseline (same seeds, same samples per theorem).

    python analysis/mechanism.py [--lengths 12-18] [--procs 32]      -> analysis/mechanism.json, analysis/mechanism.md

Per attempt (stripped proof, lengthexp/ndcheck.classify for the first rejected line):
  fake goal       the first rejected line writes the conclusion A & B (an invalid step to the goal)
  fake conjunct   the first rejected line writes A or B
  early QED       all lines valid, QED before the proof is complete
  got first       some valid line at depth 0 writes A or B (one conjunct validly derived before the first error)
  got both        both conjuncts validly derived at depth 0 before the first error (or in a correct proof)
goals only (its own annotation groups, raw text):
  names a part    some group names A or B, or a conjunct inside one of them (the model planned the split; the
                  canonical annotation names the deepest owed conjunct)
  names both      groups name (components of) both A and B
  owes second     after its first valid depth-0 line writing A or B, a group names (a component of) the other part
                  (the record of what is still owed, written at the moment it matters)
"""
import argparse, collections, gzip, json, multiprocessing as mp, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
G = os.path.dirname(HERE)
AB3 = '/scratch/cowifornia/nd-rl/ab3'
sys.path.insert(0, f'{AB3}/lengthexp')
sys.path.insert(0, os.path.join(G, 'lib'))
import goals as GL  # noqa: E402
from ndcheck import classify, sequent  # noqa: E402

GROUPS = {'goals': os.path.join(G, 'eval', 'goals_s{}'), 'abs': f'{AB3}/repr/eval/abs_s{{}}'}
SEEDS = (0, 1, 2)


def parse_lengths(s):
    a, b = s.split('-')
    return list(range(int(a), int(b) + 1))


def line_formula(ln):
    """Formula string of a line token list (N<i> [|]* formula : RULE ...)."""
    i = 1
    while i < len(ln) and ln[i] == '|':
        i += 1
    if ':' not in ln[i:]:
        return None, i - 1
    j = ln.index(':', i)
    return ' '.join(ln[i:j]), i - 1


def components(f):
    """f and, recursively, the conjuncts of f (the canonical annotation names the deepest owed conjunct)."""
    out = {GL.fstr(f)}
    if f[0] == 'and':
        out |= components(f[1]) | components(f[2])
    return out


def one(args):
    prompt, raw, body = args
    prem, concl = sequent(prompt)
    if concl[0] != 'and':
        # not a conjunction: only the all-theorem fake-conclusion statistic (lengthexp fake_finish.md's measure)
        c = classify(prompt, body)
        lines, _ = GL.body_lines(body)
        fake = False
        if c['kind'] == 'bad_line' and c['line'] - 1 < len(lines):
            fake = line_formula(lines[c['line'] - 1])[0] == GL.fstr(concl)
        return {'nonconj': True, 'failed': not c['ok'], 'fake_goal': fake}
    A, B, C = GL.fstr(concl[1]), GL.fstr(concl[2]), GL.fstr(concl)
    c = classify(prompt, body)
    lines, _ = GL.body_lines(body)
    n_valid = len(lines) if c['kind'] in ('ok', 'early_qed', 'no_qed') else c['line'] - 1
    out = {'ok': c['ok'], 'kind': c['kind'], 'n_lines': len(lines)}
    got = []
    first_pos = None
    for p, ln in enumerate(lines[:n_valid]):
        f, d = line_formula(ln)
        if d == 0 and f in (A, B) and f not in got:
            got.append(f)
            if first_pos is None:
                first_pos = p
    out['got_first'] = bool(got)
    out['got_both'] = len(got) == 2 or (A == B and bool(got))
    if c['kind'] == 'bad_line' and c['line'] - 1 < len(lines):
        f, _ = line_formula(lines[c['line'] - 1])
        out['fake_goal'] = f == C
        out['fake_conjunct'] = f in (A, B)
    else:
        out['fake_goal'] = out['fake_conjunct'] = False
    if raw is not None:
        _, ann, _ = GL.read_annotations(raw)
        named = {g for gs in ann.values() for g in gs}
        cA, cB = components(concl[1]), components(concl[2])
        out['names_part'] = bool(named & (cA | cB))
        out['names_both'] = bool(named & cA) and bool(named & cB)
        owes = False
        nxt = None
        if first_pos is not None:
            other = cB if A == line_formula(lines[first_pos])[0] else cA
            owes = any(g in other for k, gs in ann.items() if k > first_pos for g in gs)
            # the first note written after the first derived part (before the first rejected line / the end)
            later = [(k, g) for k, gs in sorted(ann.items()) if first_pos < k <= n_valid for g in gs]
            if not later:
                nxt = 'none'
            else:
                g = later[0][1]
                nxt = 'other_part' if g in other else ('conclusion' if g == C else ('first_part' if g in (cA | cB) - other else 'other'))
        out['owes_second'] = owes
        out['next_note'] = nxt
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lengths', default='12-18')
    ap.add_argument('--sets', default='std')
    ap.add_argument('--procs', type=int, default=32)
    a = ap.parse_args()
    pool = mp.Pool(a.procs)
    res = {}
    prompts = {}
    for s in a.sets.split(','):
        for r in (json.loads(l) for l in open(os.path.join(G, 'testsets', f'{s}.jsonl'))):
            prompts[(s, r['name'])] = r['prompt']
    for g, pat in GROUPS.items():
        for L in parse_lengths(a.lengths):
            for st in a.sets.split(','):
                tot = collections.Counter()
                for seed in SEEDS:
                    fn = os.path.join(pat.format(seed), 'samples', f'{st}_L{L}.jsonl.gz')
                    if not os.path.isfile(fn):
                        continue
                    xs = [json.loads(l) for l in gzip.open(fn, 'rt')]
                    args = [(prompts[(st, x['name'])], x.get('raw'), x['body']) for x in xs]
                    for o in pool.imap_unordered(one, args, chunksize=256):
                        failed = o['failed'] if 'nonconj' in o else not o['ok']
                        tot['all_failed'] += failed
                        tot['all_failed_fake_goal'] += failed and o['fake_goal']
                        if 'nonconj' in o:
                            tot['not_conjunction'] += 1
                            continue
                        tot['n'] += 1
                        tot['seeds_' + str(seed)] = 1
                        for k, v in o.items():
                            if isinstance(v, bool):
                                tot[k] += v
                        tot['kind_' + o['kind']] += 1
                        if o.get('next_note'):
                            tot['next_note_' + o['next_note']] += 1
                        if 'names_both' in o:
                            tot['ok_and_names_both'] += o['ok'] and o['names_both']
                            tot['ok_and_not_names_both'] += o['ok'] and not o['names_both']
                            tot['got_first_and_owes'] += o['got_first'] and o['owes_second']
                            tot['got_both_given_owes'] += o['got_both'] and o['owes_second']
                            tot['got_both_given_not_owes'] += o['got_both'] and o['got_first'] and not o['owes_second']
                if tot['n'] or tot['all_failed']:
                    res[f'{g}|{st}|{L}'] = dict(tot)
                    print(g, st, L, dict(tot), flush=True)
    json.dump(res, open(os.path.join(HERE, 'mechanism.json'), 'w'), indent=1)
    # markdown
    md = ['Attempts on test theorems whose conclusion is a conjunction A & B (std set), pooled over the seeds of each group.',
          'Shares are of all such attempts unless stated.', '',
          '| group | L | attempts | pass | fake goal | fake conjunct | early QED | first conjunct derived | both derived | '
          'names a part (goals) | names both | writes GOAL <other part> after deriving one | pass if names both / not |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for key, t in sorted(res.items(), key=lambda kv: (int(kv[0].split('|')[2]), kv[0])):
        g, st, L = key.split('|')
        n = t.get('n', 0)
        if not n:
            continue
        f = lambda k: f"{100 * t.get(k, 0) / n:.1f}%"
        nb = t.get('names_both', 0)
        cond = (f"{100 * t.get('ok_and_names_both', 0) / nb:.1f}% / {100 * t.get('ok_and_not_names_both', 0) / max(1, n - nb):.1f}%"
                if g == 'goals' and nb else '-')
        gf = t.get('got_first', 0)
        owes = f"{100 * t.get('got_first_and_owes', 0) / gf:.1f}% of those with one" if g == 'goals' and gf else '-'
        md.append(f"| {g} | {L} | {n} | {f('ok')} | {f('fake_goal')} | {f('fake_conjunct')} | {f('kind_early_qed')} | {f('got_first')} | "
                  f"{f('got_both')} | {f('names_part') if g == 'goals' else '-'} | {f('names_both') if g == 'goals' else '-'} | {owes} | {cond} |")
    md += ['', 'All theorems of each length (conjunctions or not): share of failed attempts whose first rejected line '
           'writes the conclusion (the measure of lengthexp/report/fake_finish.md):', '',
           '| group | ' + ' | '.join(f'L{k.split("|")[2]}' for k in sorted((k for k in res if k.startswith('goals|')),
                                                                     key=lambda k: int(k.split('|')[2]))) + ' |',
           '|---|' + '---|' * sum(1 for k in res if k.startswith('goals|'))]
    for g in GROUPS:
        cells = []
        for k in sorted((k for k in res if k.startswith('goals|')), key=lambda k: int(k.split('|')[2])):
            t = res.get(k.replace('goals|', g + '|', 1), {})
            cells.append(f"{100 * t.get('all_failed_fake_goal', 0) / t['all_failed']:.1f}%" if t.get('all_failed') else '-')
        md.append(f'| {g} | ' + ' | '.join(cells) + ' |')
    md += ['', 'goals only: the first note written after the first validly derived part (attempts that derived one part):', '',
           '| L | attempts with one part derived | names the part still owed | names the whole conclusion | names the part just done | other formula | no note before the error |',
           '|---|---|---|---|---|---|---|']
    for key, t in sorted(res.items(), key=lambda kv: int(kv[0].split('|')[2])):
        g, st, L = key.split('|')
        if g != 'goals' or not t.get('got_first'):
            continue
        n1 = t['got_first']
        f1 = lambda k: f"{100 * t.get('next_note_' + k, 0) / n1:.1f}%"
        md.append(f"| {L} | {n1} | {f1('other_part')} | {f1('conclusion')} | {f1('first_part')} | {f1('other')} | {f1('none')} |")
    open(os.path.join(HERE, 'mechanism.md'), 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
