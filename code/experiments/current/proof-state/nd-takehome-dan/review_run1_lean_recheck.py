#!/usr/bin/env python3
"""Reviewer's independent Lean re-check of every model-written Lean output in round2-run1 (step 2: Qwen3-Coder-30B, 3
forms; step 3: six Qwen3 sizes).  Own extraction from the raw generations, own batching (40 theorems per file, one
`example : True := trivial` sentinel after every theorem so a parse error inside theorem k is reported before theorem
k+1 starts), own error attribution, and an individual one-theorem-per-file re-check of every record whose verdict
differs from the executor's scored file.  Also scans every accepted body for `sorry`/`admit`/`exact?` and for tactics
outside the have/exact fragment.  Output: artifacts/review_r1/lean_recheck.json
"""
import json, os, re, sys, subprocess, tempfile, collections, time
from concurrent.futures import ThreadPoolExecutor
WS = os.path.expanduser('~/review/round2-run1')
OUT = os.path.expanduser('~/nd-takehome/artifacts/review_r1')
LEAN = os.path.expanduser('~/.elan/bin/lean')
ATOMS = ('P', 'Q', 'R', 'S')


def parse_formula(toks, i):
    t = toks[i]
    if t in ATOMS or t == 'F': return t, i + 1
    assert t == '('
    if toks[i + 1] == '~':
        sub, j = parse_formula(toks, i + 2); assert toks[j] == ')'; return ('~', sub), j + 1
    a, j = parse_formula(toks, i + 1); op = toks[j]; b, k = parse_formula(toks, j + 1); assert toks[k] == ')'
    return (op, a, b), k + 1


def lean_f(f):
    if isinstance(f, str): return 'False' if f == 'F' else f
    if f[0] == '~': return f'(¬{lean_f(f[1])})'
    op = {'&': '∧', 'v': '∨', '>': '→'}[f[0]]
    return f'({lean_f(f[1])} {op} {lean_f(f[2])})'


def statement(prompt, name):
    """my own rendering of the theorem header from the token prompt"""
    toks = prompt.split(); assert toks[0] == 'THM' and toks[-1] == 'PRF'
    i = 1; prem = []
    if toks[i] != 'SEQ':
        while True:
            f, i = parse_formula(toks, i); prem.append(f)
            if toks[i] == ',': i += 1; continue
            break
    assert toks[i] == 'SEQ'; concl, i = parse_formula(toks, i + 1); assert i == len(toks) - 1
    hyps = ' '.join(f'(h{j + 1} : {lean_f(p)})' for j, p in enumerate(prem))
    return f'theorem {name} (P Q R S : Prop) {hyps} : {lean_f(concl)} := by'


def extract(text):
    t = text.strip()
    t = re.sub(r'^```[a-zA-Z0-9]*\n', '', t); t = re.sub(r'\n```\s*$', '', t).strip()
    if ':= by' in t: t = t.split(':= by', 1)[1]
    return '\n'.join('  ' + l.strip() for l in t.splitlines() if l.strip())


def run_lean(path):
    p = subprocess.run([LEAN, path], capture_output=True, text=True)
    return p.stdout + p.stderr, p.returncode


def check_chunk(items):
    """items: list of (key, header, body). returns {key: (ok, msg)}"""
    text = 'set_option maxRecDepth 4000\n'; starts = []
    for k, (key, header, body) in enumerate(items):
        starts.append(text.count('\n') + 1)
        text += header.replace('theorem t ', f'theorem t{k} ', 1) + '\n' + body + '\nexample : True := trivial\n'
    fd, fn = tempfile.mkstemp(suffix='.lean', prefix='rv_'); os.write(fd, text.encode()); os.close(fd)
    out, rc = run_lean(fn); os.unlink(fn)
    errs = collections.defaultdict(list)
    for m in re.finditer(r'^[^\n]*?:(\d+):(\d+): error(?:\([^)]*\))?: ([^\n]*)', out, re.M):
        line = int(m.group(1)); k = max(j for j, s in enumerate(starts) if s <= line); errs[k].append(m.group(3)[:100])
    n_err = len(re.findall(r': error(?:\([^)]*\))?: ', out))
    if n_err >= 90 and len(items) > 1:       # Lean's per-file error cap: split
        res = {}
        h = len(items) // 2
        res.update(check_chunk(items[:h])); res.update(check_chunk(items[h:])); return res
    if rc != 0 and not errs:
        return {key: (False, 'lean rc ' + out[:100]) for key, _, _ in items}
    if 'sorry' in out or "declaration uses 'sorry'" in out:
        for k in range(len(items)): errs[k].append('SORRY-WARNING-IN-FILE')
    return {items[k][0]: (k not in errs, '; '.join(errs.get(k, []))) for k in range(len(items))}


def check_single(header, body):
    return check_chunk([('x', header, body)])['x']


FRAG_OK = re.compile(r'^(have n\d+ : .* := |exact n\d+\)*$|exact \(n\d+ : False\)\)*$)')
AUTOM = ['sorry', 'admit', 'exact?', 'apply?', 'simp', 'decide', 'tauto', 'omega', 'aesop', 'trivial', 'contradiction', 'assumption', 'rfl', '#']


def classify(body):
    """'fragment' if every line is a have/exact in the ND fragment and every term uses only the allowed constructs;
    'automation' if a tactic or search that proves goals by itself appears; else 'other-lean'."""
    if any(a in body for a in AUTOM): return 'automation'
    for l in body.splitlines():
        s = l.strip()
        if not s: continue
        if not FRAG_OK.match(s): return 'other-lean'
    allowed = re.sub(r'(Or\.elim|Or\.inl|Or\.inr|Classical\.byContradiction|\.elim|\.1|\.2|fun \(n\d+ : [^)]*\)|fun \(n\d+ : \(.*?\)\)|fun hh|=> by|have n\d+ :|exact|False|[PQRS]|¬|∧|∨|→|[(),⟨⟩:= \n]|n\d+|hh|h\d+)', '', body)
    if re.sub(r'\s', '', allowed): return 'other-lean'
    return 'fragment'


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = []      # (label, prompts file, gens file, scored file)
    jobs.append(('qwen30b', f'{WS}/data/r1/prompts.jsonl', f'{WS}/artifacts/r1/gens_qwen30b.jsonl', f'{WS}/artifacts/r1/scored_qwen30b.jsonl'))
    for m in ['Qwen3-0.6B', 'Qwen3-1.7B', 'Qwen3-4B', 'Qwen3-8B', 'Qwen3-14B', 'Qwen3-32B']:
        jobs.append((m, f'{WS}/data/r1/scale_prompts.jsonl', f'{WS}/artifacts/r1/gens_scale_{m}.jsonl', f'{WS}/artifacts/r1/scored_scale_{m}.jsonl'))
    report = {}
    for label, pf, gf, sf in jobs:
        t0 = time.time()
        P = {}
        for l in open(pf):
            r = json.loads(l)
            if r['form'] == 'lean': P[r['id']] = r
        exec_ok = {}
        for l in open(sf):
            r = json.loads(l)
            if r['form'] == 'lean': exec_ok[(r['id'], r['sample'])] = r['ok']
        items = []; hdr_mismatch = 0
        for l in open(gf):
            g = json.loads(l)
            if g['id'] not in P: continue
            p = P[g['id']]
            header = statement(p['prompt'], 't')
            exec_header = p['messages'][1]['content'].rsplit('### Problem\nTheorem:\n', 1)[1].split('\nProof:')[0].strip()
            if exec_header != header: hdr_mismatch += 1
            for j, text in enumerate(g['outputs']):
                items.append(((g['id'], j), header, extract(text)))
        with ThreadPoolExecutor(2) as ex:
            chunks = [items[b:b + 40] for b in range(0, len(items), 40)]
            res = {}
            for r in ex.map(check_chunk, chunks): res.update(r)
        dis = [k for k in res if res[k][0] != exec_ok.get(k)]
        single = {}
        for k in dis:
            _, header, body = next(it for it in items if it[0] == k)
            single[k] = check_single(header, body)
        still = [k for k in dis if single[k][0] != exec_ok.get(k)]
        cls = collections.Counter(); cls_acc = collections.Counter()
        for key, header, body in items:
            c = classify(body); cls[c] += 1
            if res[key][0]: cls_acc[c] += 1
        acc_mine = sum(1 for k in res if res[k][0]); acc_exec = sum(1 for v in exec_ok.values() if v)
        report[label] = {'n': len(items), 'executor_accepted': acc_exec, 'my_accepted_chunked': acc_mine, 'my_accepted_after_single_recheck': acc_mine - sum(1 for k in dis if res[k][0]) + sum(1 for k in dis if single[k][0]),
                         'disagree_chunked': len(dis), 'disagree_after_single': len(still), 'header_mismatch_vs_prompt': hdr_mismatch,
                         'still_disagree_examples': [{'id': k[0], 'sample': k[1], 'exec': exec_ok.get(k), 'mine': single[k]} for k in still[:10]],
                         'class_all': dict(cls), 'class_accepted': dict(cls_acc), 'seconds': round(time.time() - t0)}
        json.dump({'my_ok': {f'{k[0]}|{k[1]}': (single.get(k, res[k]))[0] for k in res}}, open(f'{OUT}/lean_recheck_{label}_verdicts.json', 'w'))
        print(label, report[label], flush=True)
        json.dump(report, open(f'{OUT}/lean_recheck.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
