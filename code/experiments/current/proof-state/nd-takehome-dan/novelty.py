#!/usr/bin/env python3
"""Teacher-forced log-probabilities of found proofs, marginalised over start indices; surprisal loci.

  python novelty.py --ckpts base=ckpts/stage1_abs.pt final=ckpts/final.pt \
      --src final_transfer=artifacts/ei_abs_s0_cont/found_transfer_16.jsonl \
            final_val36=artifacts/ei_abs_s0_cont_r16_val36_k32.jsonl ... \
      --out artifacts/novelty_phase1

Each source may be a found_*.jsonl (one proof per record), an eval_set / coverage jsonl (proofs list per
record).  Every proof is START-INDEX-NORMALISED (normalize.norm) and deduplicated per (source, theorem).

For every proof and every checkpoint:
  logp_T1  = log sum_s p(proof shifted by s | prompt)      (s over all shifts the tokenizer allows, T=1)
  logp_T08 = same with logits/0.8 (the sampling distribution used everywhere in this project)
  best_shift, logp at the best shift, and (for the first checkpoint) the per-token surprisal (T=1) at the
  best shift, the per-line surprisal (start-index token excluded and reported separately), the line with
  maximum surprisal and its rule.
Outputs:
  <out>_proofs.jsonl    one record per (source, theorem, normalised proof)
  <out>_theorems.jsonl  one record per (source, theorem): log of the summed probability over its proofs
"""
import argparse, json, os, sys, math, collections, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn.functional as F
from model import load_ckpt
from tokenizer import MAXN
from normalize import norm
from prune import pruned_length
from nd_verify import verify_text


def load_source(name, fn):
    """-> list of dicts {src, name, thm, prompt, proof(norm), written, pruned, gen_lines, round}"""
    out, seen = [], set()
    for l in open(fn):
        if not l.strip():
            continue
        x = json.loads(l)
        if 'proof' in x and isinstance(x['proof'], str):
            items = [(x['proof'], x.get('written'), x.get('pruned'), x.get('round'))]
        else:
            ps = x.get('proofs', [])
            if ps and isinstance(ps[0], dict):
                items = [(p['proof'], p.get('written'), p.get('pruned'), None) for p in ps]
            else:
                items = [(p, w, q, None) for p, w, q in zip(ps, x.get('written_lens', [None] * len(ps)), x.get('pruned_lens', [None] * len(ps)))]
        tname = x.get('name', x.get('thm'))
        for p, w, q, rnd in items:
            pn = norm(p)
            k = (tname, pn)
            if k in seen:
                continue
            seen.add(k)
            if w is None:
                w = sum(1 for t in pn.split() if t == ';')
            if q is None:
                q = pruned_length(x['prompt'], pn)
            out.append({'src': name, 'name': tname, 'thm': x.get('thm'), 'prompt': x['prompt'], 'proof': pn,
                        'written': w, 'pruned': q, 'gen_lines': x.get('gen_lines', x.get('n_lines')), 'round': rnd})
    return out


def lines_of(proof):
    toks = proof.split()
    lines, cur = [], []
    for t in toks:
        if t == 'QED':
            break
        cur.append(t)
        if t == ';':
            lines.append(cur); cur = []
    return lines


def rule_of(line):
    try:
        return line[line.index(':') + 1]
    except (ValueError, IndexError):
        return '?'


@torch.no_grad()
def score(model, tok, recs, temps=(1.0, 0.8), batch=768, want_tokens=False, dev='cuda'):
    """Fills recs[i]['res'] = {logp_T1, logp_T08, best_shift, logp_best_T1, n_shifts, (tokens...)}"""
    # build all (rec, shift) sequences
    jobs = []
    for i, r in enumerate(recs):
        pid = tok.encode_prompt(r['prompt'])
        qid = tok.encode_proof(r['proof'])
        mx = max((x - tok.ref0 + 1 for x in qid if x >= tok.ref0), default=0)
        for s in range(0, MAXN - mx + 1):
            ids = pid + [x + s if x >= tok.ref0 else x for x in qid]
            jobs.append((i, s, len(pid), ids))
    jobs.sort(key=lambda j: len(j[3]))
    per = collections.defaultdict(dict)   # i -> {shift: (lp1, lp08)}
    t0 = time.time()
    for b0 in range(0, len(jobs), batch):
        chunk = jobs[b0:b0 + batch]
        T = max(len(j[3]) for j in chunk)
        x = torch.full((len(chunk), T), tok.pad, dtype=torch.long)
        m = torch.zeros((len(chunk), T), dtype=torch.bool)
        for k, (i, s, lp, ids) in enumerate(chunk):
            x[k, :len(ids)] = torch.tensor(ids)
            m[k, lp:len(ids)] = True          # proof positions (targets)
        x, m = x.to(dev), m.to(dev)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')):
            logits = model(x[:, :-1]).float()
        tgt = x[:, 1:]
        mk = m[:, 1:]
        outs = []
        for temp in temps:
            lp = F.log_softmax(logits / temp, -1).gather(-1, tgt[..., None]).squeeze(-1)
            outs.append((lp * mk).sum(1).tolist())
        for k, (i, s, lp, ids) in enumerate(chunk):
            per[i][s] = tuple(o[k] for o in outs)
        if (b0 // batch) % 200 == 0:
            print(f'  scored {b0 + len(chunk)}/{len(jobs)} sequences ({time.time()-t0:.0f}s)', flush=True)
    for i, r in enumerate(recs):
        d = per[i]
        l1 = torch.tensor([v[0] for v in d.values()]); l08 = torch.tensor([v[1] for v in d.values()])
        best = max(d, key=lambda s: d[s][0])
        r['res'] = {'logp_T1': torch.logsumexp(l1, 0).item(), 'logp_T08': torch.logsumexp(l08, 0).item(),
                    'n_shifts': len(d), 'best_shift': best, 'logp_best_T1': d[best][0], 'logp_best_T08': d[best][1]}
    if not want_tokens:
        return
    # second pass: per-token surprisal at the best shift (T=1)
    jobs = []
    for i, r in enumerate(recs):
        pid = tok.encode_prompt(r['prompt']); qid = tok.encode_proof(r['proof']); s = r['res']['best_shift']
        jobs.append((i, s, len(pid), pid + [x + s if x >= tok.ref0 else x for x in qid]))
    jobs.sort(key=lambda j: len(j[3]))
    for b0 in range(0, len(jobs), batch):
        chunk = jobs[b0:b0 + batch]
        T = max(len(j[3]) for j in chunk)
        x = torch.full((len(chunk), T), tok.pad, dtype=torch.long)
        for k, (i, s, lp, ids) in enumerate(chunk):
            x[k, :len(ids)] = torch.tensor(ids)
        x = x.to(dev)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')):
            logits = model(x[:, :-1]).float()
        lp = F.log_softmax(logits, -1).gather(-1, x[:, 1:, None]).squeeze(-1)
        for k, (i, s, lpn, ids) in enumerate(chunk):
            sur = (-lp[k, lpn - 1:len(ids) - 1]).tolist()      # surprisal of each proof token incl. start index, QED
            r = recs[i]
            toks = r['proof'].split()
            assert len(toks) == len(sur), (len(toks), len(sur))
            lines = lines_of(r['proof'])
            # per-line sums; token 0 (start index) excluded from line 1 and reported separately
            per_line, pos = [], 0
            for li, ln in enumerate(lines):
                seg = sur[pos:pos + len(ln)]
                if li == 0:
                    seg = seg[1:]
                per_line.append(sum(seg)); pos += len(ln)
            body = sur[1:]                                       # exclude the start-index token
            jmax = max(range(len(body)), key=lambda j: body[j]) if body else 0
            lmax = max(range(len(per_line)), key=lambda j: per_line[j]) if per_line else 0
            r['res'].update({'start_surprisal': sur[0], 'tok_surprisal': [round(v, 3) for v in sur],
                             'line_surprisal': [round(v, 3) for v in per_line],
                             'max_line': lmax + 1, 'max_line_rule': rule_of(lines[lmax]) if lines else '?',
                             'max_line_text': ' '.join(lines[lmax]) if lines else '',
                             'max_line_share': per_line[lmax] / max(sum(per_line), 1e-9) if per_line else 0,
                             'max_tok': toks[jmax + 1], 'max_tok_surprisal': body[jmax] if body else 0,
                             'body_surprisal': sum(body)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', nargs='+', required=True, help='label=path ... (first one gets the surprisal profile)')
    ap.add_argument('--src', nargs='+', required=True, help='label=path ...')
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch', type=int, default=768)
    ap.add_argument('--verify', action='store_true', help='re-verify every normalised proof (sanity)')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    recs = []
    for s in a.src:
        label, fn = s.split('=', 1)
        rs = load_source(label, fn)
        print(f'{label}: {len(rs)} normalised-distinct proofs, {len({r["name"] for r in rs})} theorems', flush=True)
        recs += rs
    if a.verify:
        bad = 0
        for r in recs:
            ok, reason, nl = verify_text(r['prompt'] + ' ' + r['proof'])
            bad += not ok
        print('verify failures', bad, flush=True)
    allres = {}
    for ci, c in enumerate(a.ckpts):
        label, path = c.split('=', 1)
        model, tok, _ = load_ckpt(path, dev)
        assert tok.mode == 'abs'
        print(f'scoring under {label} ({path})', flush=True)
        score(model, tok, recs, batch=a.batch, want_tokens=(ci == 0), dev=dev)
        allres[label] = [r.pop('res') for r in recs]
        del model; torch.cuda.empty_cache()
    labels = [c.split('=', 1)[0] for c in a.ckpts]
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out + '_proofs.jsonl', 'w') as f:
        for i, r in enumerate(recs):
            rec = dict(r)
            for lb in labels:
                res = allres[lb][i]
                for k, v in res.items():
                    rec[f'{lb}_{k}'] = v
            f.write(json.dumps(rec) + '\n')
    # per theorem
    by = collections.defaultdict(list)
    for i, r in enumerate(recs):
        by[(r['src'], r['name'])].append(i)
    with open(a.out + '_theorems.jsonl', 'w') as f:
        for (src, name), idx in by.items():
            r0 = recs[idx[0]]
            rec = {'src': src, 'name': name, 'thm': r0['thm'], 'prompt': r0['prompt'], 'gen_lines': r0['gen_lines'],
                   'n_proofs': len(idx), 'min_written': min(recs[i]['written'] for i in idx), 'max_written': max(recs[i]['written'] for i in idx),
                   'min_pruned': min(recs[i]['pruned'] for i in idx), 'first_round': min((recs[i]['round'] for i in idx if recs[i]['round'] is not None), default=None)}
            for lb in labels:
                for key in ('logp_T1', 'logp_T08'):
                    vals = torch.tensor([allres[lb][i][key] for i in idx])
                    rec[f'{lb}_{key}_any'] = torch.logsumexp(vals, 0).item()
                    rec[f'{lb}_{key}_max'] = vals.max().item()
            f.write(json.dumps(rec) + '\n')
    print('wrote', a.out + '_proofs.jsonl', a.out + '_theorems.jsonl', flush=True)


if __name__ == '__main__':
    main()
