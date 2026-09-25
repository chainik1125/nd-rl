#!/usr/bin/env python3
"""Design statistics (CPU only): box / goal structure of verified proofs and the token overhead of each annotation
policy. Sources: the cap-6 pretraining data, the cap-6 held-out file and the proofs found by RL runs (the
lengthexp arm runs and the base039 runs; training-side data only, no test-set proofs).

    python analysis/structure_stats.py [--limit N] [--out analysis/structure_stats.json]
"""
import argparse, collections, glob, json, multiprocessing as mp, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import goals as G  # noqa: E402

AB3 = '/scratch/cowifornia/nd-rl/ab3'
SOURCES = {
    'train_cap6': [f'{AB3}/frozen/data/train_cap6.jsonl'],
    'heldout_cap6': [f'{AB3}/frozen/data/heldout_cap6.jsonl'],
    'found_rl': sorted(glob.glob(f'{AB3}/lengthexp/runs/*/found_targets.jsonl')) + sorted(glob.glob(f'{AB3}/base039/runs/*/found_targets.jsonl')),
    # token-fit check only (never used for training or design choices): the generator-constructed proofs of the
    # long test sets and the matched control, the only proofs of 12-18 lines available
    'fitcheck_constructed': [f'{AB3}/sealed/matched_L6-14.jsonl', f'{AB3}/sealed/sealed_L15-16.jsonl', f'{AB3}/sealed/sealed_L17-18.jsonl'],
}


def records(src, limit):
    n = 0
    for fn in SOURCES[src]:
        for l in open(fn):
            if not l.strip():
                continue
            r = json.loads(l)
            prompt = r.get('prompt') or ' '.join(('THM ' + r['thm'].replace('|-', 'SEQ') + ' PRF').split())
            yield prompt, r.get('proof') or r['constructed_proof']
            n += 1
            if limit and n >= limit:
                return


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def work(chunk):
    c = collections.Counter()
    disc = collections.Counter()
    tok_plain, tok_extra = collections.defaultdict(list), {p: collections.defaultdict(list) for p in G.POLICIES}
    tok_prompt = collections.defaultdict(list)
    per_line_ann = {p: [] for p in G.POLICIES}
    max_tok_line = []
    if True:
        for prompt, body in chunk:
            ok, reason, nl = G.verify_text(prompt + ' ' + body)
            if not ok:
                c['not_verified'] += 1
                continue
            c['proofs'] += 1
            P = G.parse_body(prompt, body)
            # boxes and how they are discharged
            cites = collections.defaultdict(list)
            for p, ln in enumerate(P.lines):
                if ln['rule'] in ('IMPI', 'NEGI'):
                    cites[P.box_of_as.get(P.pos[p][0])].append((p, ln['rule']))
                elif ln['rule'] == 'ORE':
                    cites[P.box_of_as.get(P.pos[p][1])].append((p, 'ORE'))
                    cites[P.box_of_as.get(P.pos[p][3])].append((p, 'ORE'))
            obs = G.obligations(P, splits=True)
            c['has_box'] += bool(P.boxes)
            for b, B in P.boxes.items():
                c['boxes'] += 1
                cs = cites.get(b, [])
                if not cs:
                    disc['undischarged'] += 1
                    continue
                if len(cs) > 1:
                    c['box_cited_twice'] += 1
                p, rule = cs[0]
                disc[rule] += 1
                # goal-directed: the discharging line writes the obligation pending (displayed) when the box opened
                pend = [o for o in obs if o.push < B['s'] and o.ach >= B['s']]
                top = min(pend, key=lambda o: (o.ach, -o.push)) if pend else None
                if top is not None and P.lines[p]['formula'] == top.f:
                    disc[rule + ':goal_directed'] += 1
                elif rule == 'NEGI':
                    disc['NEGI:lemma'] += 1
                else:
                    disc[rule + ':lemma'] += 1
            subs = [o for o in obs if o.kind == 'sub']
            c['sub_goals'] += len(subs)
            c['has_split'] += bool(subs)
            c['root_split'] += any(o.parent is not None and o.parent.kind == 'root' for o in subs)
            c['andi_final'] += P.lines[-1]['rule'] == 'ANDI'
            n_plain = len(body.split())
            n_prompt = len(prompt.split())
            L = P.n
            tok_plain[L].append(n_plain)
            tok_prompt[L].append(n_prompt)
            max_tok_line.append(max(len(x) for x in G.body_lines(body)[0]))
            for pol in G.POLICIES:
                ann = G.render(body, G.displays(P, pol))
                back = G.strip(ann)
                if back != ' '.join(body.split()):
                    c[f'strip_mismatch_{pol}'] += 1
                ok2 = G.verify_text(prompt + ' ' + back)[0]
                if ok2 != ok:
                    c[f'verdict_mismatch_{pol}'] += 1
                extra = len(ann.split()) - n_plain
                tok_extra[pol][L].append(extra)
                per_line_ann[pol].append(len(G.displays(P, pol)) / max(1, L))
    return c, disc, tok_plain, tok_extra, tok_prompt, per_line_ann, max_tok_line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--procs', type=int, default=24)
    ap.add_argument('--out', default=os.path.join(HERE, 'structure_stats.json'))
    a = ap.parse_args()
    res = {}
    pool = mp.Pool(a.procs)
    for src in SOURCES:
        recs = list(dict.fromkeys(records(src, a.limit)))
        chunks = [recs[i::a.procs * 4] for i in range(a.procs * 4)]
        c, disc = collections.Counter(), collections.Counter()
        tok_plain, tok_extra = collections.defaultdict(list), {p: collections.defaultdict(list) for p in G.POLICIES}
        tok_prompt = collections.defaultdict(list)
        per_line_ann = {p: [] for p in G.POLICIES}
        max_tok_line = []
        for c1, d1, tp, te, tpr, pla, mtl in pool.imap_unordered(work, chunks):
            c.update(c1); disc.update(d1); max_tok_line += mtl
            for L, v in tp.items():
                tok_plain[L] += v
            for L, v in tpr.items():
                tok_prompt[L] += v
            for pol in G.POLICIES:
                per_line_ann[pol] += pla[pol]
                for L, v in te[pol].items():
                    tok_extra[pol][L] += v
        out = {'counts': dict(c), 'discharge': dict(disc), 'max_tokens_per_line_p99': pct(max_tok_line, 0.99),
               'max_tokens_per_line_max': max(max_tok_line) if max_tok_line else None, 'by_len': {}}
        tot_plain = sum(sum(v) for v in tok_plain.values())
        out['overhead'] = {pol: sum(sum(v) for v in tok_extra[pol].values()) / max(tot_plain, 1) for pol in G.POLICIES}
        out['groups_per_line'] = {pol: sum(v) / max(len(v), 1) for pol, v in per_line_ann.items()}
        for L in sorted(tok_plain):
            row = {'n': len(tok_plain[L]), 'plain_mean': sum(tok_plain[L]) / len(tok_plain[L]), 'plain_max': max(tok_plain[L])}
            for pol in G.POLICIES:
                ex = tok_extra[pol][L]
                tot = [a + b for a, b in zip(tok_plain[L], ex)]
                seq = [a + b + c for a, b, c in zip(tok_plain[L], ex, tok_prompt[L])]
                row[pol] = {'extra_mean': sum(ex) / len(ex), 'total_mean': sum(tot) / len(tot), 'total_max': max(tot),
                            'total_p99': pct(tot, 0.99), 'with_prompt_max': max(seq), 'with_prompt_p99': pct(seq, 0.99),
                            'over_1024': sum(1 for x in seq if x > 1024)}
            row['prompt_max'] = max(tok_prompt[L])
            row['plain_with_prompt_max'] = max(a + b for a, b in zip(tok_plain[L], tok_prompt[L]))
            out['by_len'][L] = row
        res[src] = out
        print(src, json.dumps({k: v for k, v in out.items() if k != 'by_len'}, indent=1), flush=True)
        for L, row in out['by_len'].items():
            print(f"  L{L}: n {row['n']} plain {row['plain_mean']:.1f} (max {row['plain_max']}) | " + ' | '.join(
                f"{pol} +{row[pol]['extra_mean']:.1f} tot {row[pol]['total_mean']:.1f} max+prompt {row[pol]['with_prompt_max']}" for pol in G.POLICIES))
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
