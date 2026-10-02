#!/usr/bin/env python3
"""Composition diagnosis, step 1-2 theorem sets (CPU only).

    python composition/build.py

Selects 100 ANDI theorems per joined length (L6-14 from the matched test draw testsets/matched.jsonl, L15-18
from the standard draw testsets/std.jsonl, source ab3_andi; fixed seed), recovers each theorem's two parts from
its constructed proof (premises, part-1 body, part-2 body, one ANDI line; ab3/data_build/tools/build_cands.py)
and the original part theorems (ab2 reserve, by orig_name), and builds the prompt controls:

  part_ren   Γi ⊢ φi as in the joined prompt (renamed atoms, premises in joined-prompt order)
  part_orig  the original un-renamed part theorem (ab2 reserve prompt, verbatim)
  ctrl_a     premises grouped by part (left conjunct's part first, each part's premises in its original order);
             conclusion unchanged
  ctrl_b     conjunct order fixed to the order in which the parts' premises first appear in the prompt;
             premises unchanged
  ctrl_c     original atom letters for both parts (only when the two parts' original atom sets are disjoint);
             premise and conjunct order unchanged
A control identical to the joined prompt is not re-evaluated (the joined result is used). Every part and control
gets a proof (the parts' minimal proofs, renamed and renumbered, plus the ANDI line) checked by nd_verify, so each is
a valid theorem of the stated length (renaming and reordering do not change the minimal length).

Writes composition/joined.jsonl (one row per joined theorem, with the prompt of each role) and
composition/prompts.jsonl (unique prompts to evaluate, with L_true).
"""
import collections, itertools, json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
LE = os.path.dirname(HERE)
sys.path.insert(0, LE)
sys.dont_write_bytecode = True
from common import AB2, AB3, TESTSETS, read_jsonl, write_jsonl, sha256_file  # noqa: E402
from nd_verify import verify_text  # noqa: E402  (frozen/lib via common)
from nd_verify.verify import parse_formula  # noqa: E402

PER_LEN = 100
SEED = 20260925
SEALED = {'matched': f'{AB3}/sealed/matched_L6-14.jsonl', 'std15': f'{AB3}/sealed/sealed_L15-16.jsonl',
          'std17': f'{AB3}/sealed/sealed_L17-18.jsonl'}
RESERVE = f'{AB2}/data/reserve.jsonl'


def fstr(f):
    t = f[0]
    if t == 'atom':
        return f[1]
    if t == 'bot':
        return 'F'
    if t == 'not':
        return f'( ~ {fstr(f[1])} )'
    return f'( {fstr(f[1])} {dict(and_="&", or_="v", imp=">").get(t, None) or {"and": "&", "or": "v", "imp": ">"}[t]} {fstr(f[2])} )'


def pf(s):
    return parse_formula(s.split(), 0)[0]


def parse_prompt(prompt):
    t = prompt.split()
    body = ' '.join(t[1:-1])
    lhs, rhs = body.split('SEQ')
    prem = [pf(p) for p in lhs.split(' , ')] if lhs.strip() else []
    return prem, pf(rhs)


def parse_proof(text):
    out = []
    for seg in text.replace('QED', '').split(';'):
        t = seg.split()
        if not t:
            continue
        i, d = 1, 0
        while t[i] == '|':
            d += 1; i += 1
        j = t.index(':', i)
        out.append((d, pf(' '.join(t[i:j])), t[j + 1], [int(x[1:]) for x in t[j + 2:]]))
    return out


def proof_text(lines):
    toks = []
    for i, (d, f, rule, refs) in enumerate(lines):
        toks += [f'N{i + 1}'] + ['|'] * d + fstr(f).split() + [':', rule] + [f'N{r}' for r in refs] + [';']
    return ' '.join(toks + ['QED'])


def prompt_of(prem, concl):
    return f'THM {" , ".join(fstr(p) for p in prem)} SEQ {fstr(concl)} PRF' if prem else f'THM SEQ {fstr(concl)} PRF'


def atoms(f, acc=None):
    acc = set() if acc is None else acc
    if f[0] == 'atom':
        acc.add(f[1])
    elif f[0] == 'not':
        atoms(f[1], acc)
    elif f[0] in ('and', 'or', 'imp'):
        atoms(f[1], acc); atoms(f[2], acc)
    return acc


def rename(f, m):
    if f[0] == 'atom':
        return ('atom', m[f[1]])
    if f[0] == 'not':
        return ('not', rename(f[1], m))
    if f[0] in ('and', 'or', 'imp'):
        return (f[0], rename(f[1], m), rename(f[2], m))
    return f


def with_premises(part_lines, n_prem_old, old_prem, new_prem):
    """A part proof (premise lines first) rewritten with PR lines in `new_prem` order (a superset of the part's
    premises is not allowed here: exactly the same set)."""
    assert sorted(map(str, old_prem)) == sorted(map(str, new_prem))
    pidx = {str(p): i + 1 for i, p in enumerate(new_prem)}
    lines = [(0, p, 'PR', []) for p in new_prem]
    m = {i + 1: pidx[str(p)] for i, p in enumerate(old_prem)}
    for k, (d, f, r, refs) in enumerate(part_lines[n_prem_old:]):
        lines.append((d, f, r, [m[x] for x in refs]))
        m[n_prem_old + k + 1] = len(lines)
    return lines


def joined_proof(prem, concl, parts):
    """parts: list of (prem_i, concl_i, lines_i) (lines_i with its own premise lines first). Premise lines in
    `prem` order, bodies in the given order, ANDI citing the conjuncts in conclusion order."""
    pidx = {str(p): i + 1 for i, p in enumerate(prem)}
    lines = [(0, p, 'PR', []) for p in prem]
    end = {}
    for pp, cc, ll in parts:
        m = {i + 1: pidx[str(p)] for i, p in enumerate(pp)}
        for k, (d, f, r, refs) in enumerate(ll[len(pp):]):
            lines.append((d, f, r, [m[x] for x in refs]))
            m[len(pp) + k + 1] = len(lines)
        end[str(cc)] = len(lines)
    lines.append((0, concl, 'ANDI', [end[str(concl[1])], end[str(concl[2])]]))
    return lines


def check(prem, concl, lines, L):
    pr = prompt_of(prem, concl)
    ok, why, nl = verify_text(pr + ' ' + proof_text(lines))
    return ok and nl == L, why, pr


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=HERE, help='output directory (default composition/)')
    ap.add_argument('--lmax', type=int, default=18)
    ap.add_argument('--roles', default='part_ren,part_orig,ctrl_a,ctrl_b,ctrl_c', help='prompt kinds written to prompts.jsonl')
    a = ap.parse_args()
    roles = set(a.roles.split(','))
    os.makedirs(a.dir, exist_ok=True)
    tests = {r['name']: r for f in ('matched.jsonl', 'std.jsonl') for r in read_jsonl(os.path.join(TESTSETS, f))}
    full = {}
    for f in SEALED.values():
        for r in read_jsonl(f):
            full[r['name']] = r
    sel = []
    for L in range(6, a.lmax + 1):
        s = 'matched' if L <= 14 else 'std'
        cand = sorted(n for n, r in tests.items() if r['set'] == s and r['L_true'] == L and r['source'] == 'ab3_andi')
        pick = random.Random(f'{SEED}-{L}').sample(cand, min(PER_LEN, len(cand)))
        sel += [(s, L, n) for n in sorted(pick)]
    need = {p for _, _, n in sel for p in full[n]['parts']}
    orig = {}
    with open(RESERVE) as f:
        for l in f:
            if any(x in l for x in ('"orig_name": "ab2g_',)):
                r = json.loads(l)
                if r['orig_name'] in need:
                    orig[r['orig_name']] = r
    missing = need - set(orig)
    print(f'selected {len(sel)} joined theorems; {len(need)} distinct parts, {len(missing)} not found in the reserve')
    rows, prompts = [], {}
    stats = collections.Counter()

    def add_prompt(pr, L, kind):
        if kind not in roles:
            return pr
        if pr not in prompts:
            prompts[pr] = {'prompt': pr, 'L_true': L, 'kinds': []}
        if kind not in prompts[pr]['kinds']:
            prompts[pr]['kinds'].append(kind)
        return pr

    for s, L, name in sel:
        r = full[name]
        prem, concl = parse_prompt(r['prompt'])
        assert concl[0] == 'and', name
        cl = parse_proof(r['constructed_proof'])
        n = len(prem)
        andi = cl[-1]
        assert andi[2] == 'ANDI' and andi[1] == concl and len(cl) == L, name
        e_first, e_second = sorted(andi[3])          # body end of the part written first / second
        bodies = [cl[n:e_first], cl[e_first:e_second]]
        concl_body = [cl[e_first - 1][1], cl[e_second - 1][1]]   # conclusion of body 1 / body 2
        info = {'name': name, 'set': s, 'L': L, 'label_kind': r.get('label_kind'), 'n_prem': n,
                'n_atoms': len(set().union(*[atoms(p) for p in prem], atoms(concl))), 'parts': [], 'ok': True}
        rec_parts = []
        origs = [orig.get(p) for p in r['parts']]
        if any(o is None for o in origs):
            stats['part_missing'] += 1
            continue
        parsed = [parse_prompt(o['prompt']) for o in origs]
        oas = [sorted(set().union(*[atoms(p) for p in op], atoms(oc))) for op, oc in parsed]
        cands = []
        for i in range(2):
            ms = []
            for tgt in itertools.permutations('PQRS', len(oas[i])):
                m = dict(zip(oas[i], tgt))
                if rename(parsed[i][1], m) == concl_body[i] and all(rename(p, m) in prem for p in parsed[i][0]):
                    ms.append(m)
            cands.append(ms)
        joint = []
        for m0, m1 in itertools.product(*cands):
            if set(m0.values()) & set(m1.values()):
                continue
            ps = [rename(p, m0) for p in parsed[0][0]] + [rename(p, m1) for p in parsed[1][0]]
            if len(ps) == len(prem) and sorted(map(str, ps)) == sorted(map(str, prem)):
                joint.append((m0, m1))
        if not joint:
            stats['no_map'] += 1
            continue
        if len({str(sorted([str(rename(p, j[0])) for p in parsed[0][0]])) for j in joint}) > 1:
            stats['ambiguous_partition'] += 1
        for i, pname in enumerate(r['parts']):
            o = origs[i]
            op, oc = parsed[i]
            oa = oas[i]
            target = concl_body[i]
            found = joint[0][i]
            rp = [rename(p, found) for p in op]
            rp_joined_order = [p for p in prem if p in rp]
            olines = parse_proof(o['minlen_proof'])
            rlines = [(d, rename(f, found), ru, refs) for d, f, ru, refs in olines]
            assert len(olines) == o['L_true'] == r['part_L'][i], (name, pname)
            # renamed part, premises in joined-prompt order
            pl = with_premises(rlines, len(op), rp, rp_joined_order)
            ok, why, pr_ren = check(rp_joined_order, target, pl, o['L_true'])
            assert ok, (name, pname, why)
            ok2, why2, _ = check(op, oc, olines, o['L_true'])
            assert ok2 and o['prompt'].split() == prompt_of(op, oc).split(), (name, pname, why2)
            rec_parts.append({'orig_name': pname, 'L': o['L_true'], 'map': found, 'orig_atoms': oa,
                              'ren_atoms': sorted(found[a] for a in oa), 'n_prem': len(op),
                              'prem_ren': rp, 'prem_orig_order_ren': rp, 'concl_ren': target, 'lines_ren': rlines,
                              'prompt_ren': add_prompt(pr_ren, o['L_true'], 'part_ren'),
                              'prompt_orig': add_prompt(o['prompt'], o['L_true'], 'part_orig')})
        # which part is the left conjunct?
        left = 0 if rec_parts[0]['concl_ren'] == concl[1] else 1
        assert rec_parts[left]['concl_ren'] == concl[1] and rec_parts[1 - left]['concl_ren'] == concl[2]
        first_prem = [min(prem.index(p) for p in rp['prem_ren']) for rp in rec_parts]
        info['left_part'] = left
        info['conj_matches_prem_order'] = (first_prem[left] < first_prem[1 - left])
        info['prem_grouped'] = all(
            max(prem.index(p) for p in rec_parts[a]['prem_ren']) < min(prem.index(p) for p in rec_parts[b]['prem_ren'])
            for a, b in ((0, 1),)) or all(
            max(prem.index(p) for p in rec_parts[1]['prem_ren']) < min(prem.index(p) for p in rec_parts[0]['prem_ren']) for _ in (0,))
        pp = [(rp['prem_ren'], rp['concl_ren'], rp['lines_ren']) for rp in rec_parts]
        ok, why, pr_j = check(prem, concl, joined_proof(prem, concl, pp), L)
        assert ok and pr_j.split() == r['prompt'].split(), (name, why)
        info['prompt'] = r['prompt']
        # (a) grouped premises, left conjunct's part first, each part in its original premise order
        pa = rec_parts[left]['prem_ren'] + rec_parts[1 - left]['prem_ren']
        ok, why, pr_a = check(pa, concl, joined_proof(pa, concl, pp), L)
        assert ok, (name, 'a', why)
        # (b) conjunct order = order of first premise appearance
        first = 0 if first_prem[0] < first_prem[1] else 1
        cb = ('and', rec_parts[first]['concl_ren'], rec_parts[1 - first]['concl_ren'])
        ok, why, pr_b = check(prem, cb, joined_proof(prem, cb, pp), L)
        assert ok, (name, 'b', why)
        # (c) original atom letters, if the two parts' original atom sets are disjoint
        pr_c = None
        if not set(rec_parts[0]['orig_atoms']) & set(rec_parts[1]['orig_atoms']):
            inv = {}
            for rp in rec_parts:
                inv.update({v: k for k, v in rp['map'].items()})
            pc = [rename(p, inv) for p in prem]
            cc = rename(concl, inv)
            ppc = [([rename(p, inv) for p in a], rename(c, inv), [(d, rename(f, inv), ru, refs) for d, f, ru, refs in ll])
                   for a, c, ll in pp]
            ok, why, pr_c = check(pc, cc, joined_proof(pc, cc, ppc), L)
            assert ok, (name, 'c', why)
        else:
            stats['c_clash'] += 1
        info['ctrl'] = {}
        for k, pr in (('a', pr_a), ('b', pr_b), ('c', pr_c)):
            if pr is None:
                info['ctrl'][k] = None
            elif pr.split() == r['prompt'].split():
                info['ctrl'][k] = 'same'; stats[f'{k}_same_as_joined'] += 1
            else:
                info['ctrl'][k] = add_prompt(pr, L, f'ctrl_{k}')
        info['parts'] = [{k: v for k, v in rp.items() if k in ('orig_name', 'L', 'map', 'orig_atoms', 'ren_atoms', 'n_prem',
                                                                'prompt_ren', 'prompt_orig')} for rp in rec_parts]
        info['parts_prompt_order'] = first
        rows.append(info)
    write_jsonl(os.path.join(a.dir, 'joined.jsonl'), rows)
    pl = sorted(prompts.values(), key=lambda x: (x['L_true'], x['prompt']))
    write_jsonl(os.path.join(a.dir, 'prompts.jsonl'), pl)
    kinds = collections.Counter(k for p in pl for k in p['kinds'])
    print(f'joined kept {len(rows)}; by length', dict(collections.Counter(r["L"] for r in rows)))
    print('unique prompts to evaluate', len(pl), dict(kinds), 'stats', dict(stats))
    print('reserve sha256', sha256_file(RESERVE)[:12])


if __name__ == '__main__':
    main()
