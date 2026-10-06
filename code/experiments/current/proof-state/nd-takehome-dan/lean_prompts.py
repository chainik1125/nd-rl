#!/usr/bin/env python3
"""Run 1, step 2: build the in-context evaluation sets in three surface forms and score model outputs.

Surface forms of one ND proof (spec.md) — the SAME theorems and the SAME worked examples in each:
  tokens  : the take-home format verbatim ("THM ... PRF" / "N1 ... ; ... QED"), checked with nd_verify
  lean    : Lean 4 (nd2lean.translate): statement "theorem t (P Q R S : Prop) (h1 : ..) : C := by" + tactic body, checked with Lean
  english : the same lines with English rule names and connectives (deterministic, invertible; parsed back to tokens and
            checked with nd_verify): "line 3 (depth 1): (not P) by assumption", "line 4 (depth 1): False by negation-elimination from 3, 2"
Test theorems: validation-36 + 200 transfer theorems stratified by generating length (7-16). Worked examples: 20 verified
proofs per draw (12 from the held-out generator set, lengths 2-6; 8 from the take-home RL run's verified transfer proofs of
7-9 written lines, class-disjoint from every test theorem), 5 draws (seeds 0-4).
  python lean_prompts.py build --out data/r1/prompts.jsonl         # records {id, draw, form, theorem_name, bin, prompt(messages), answer_prefix}
  python lean_prompts.py score --prompts data/r1/prompts.jsonl --gens artifacts/r1/gens.jsonl --out artifacts/r1/scored.jsonl
"""
import argparse, json, os, sys, random, re, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nd_verify import verify_text
from nd_verify.verify import parse_proof_tokens, parse_formula, ParseError
from gen import canon_key
from nd2lean import translate, lean_check, lf, parse_prompt, TranslationError

RULE_EN = {'PR': 'premise', 'AS': 'assumption', 'R': 'reiteration', 'ANDI': 'and-introduction', 'ANDE1': 'and-elimination-left', 'ANDE2': 'and-elimination-right',
           'IMPE': 'implication-elimination', 'IMPI': 'implication-introduction', 'ORI1': 'or-introduction-left', 'ORI2': 'or-introduction-right',
           'ORE': 'or-elimination', 'NEGE': 'negation-elimination', 'NEGI': 'negation-introduction', 'BOTE': 'falsum-elimination', 'DN': 'double-negation-elimination'}
EN_RULE = {v: k for k, v in RULE_EN.items()}


def fen(f):
    t = f[0]
    if t == 'atom': return f[1]
    if t == 'bot': return 'False'
    if t == 'not': return f'(not {fen(f[1])})'
    op = {'and': 'and', 'or': 'or', 'imp': 'implies'}[t]
    return f'({fen(f[1])} {op} {fen(f[2])})'


def ftok(f):
    t = f[0]
    if t == 'atom': return f[1]
    if t == 'bot': return 'F'
    if t == 'not': return f'( ~ {ftok(f[1])} )'
    op = {'and': '&', 'or': 'v', 'imp': '>'}[t]
    return f'( {ftok(f[1])} {op} {ftok(f[2])} )'


def en_parse_formula(s):
    """English infix (fully parenthesised) -> formula tuple"""
    toks = s.replace('(', ' ( ').replace(')', ' ) ').split()
    def p(i):
        t = toks[i]
        if t in ('P', 'Q', 'R', 'S'): return ('atom', t), i + 1
        if t == 'False': return ('bot',), i + 1
        if t == '(':
            if toks[i + 1] == 'not':
                sub, j = p(i + 2); assert toks[j] == ')'; return ('not', sub), j + 1
            l, j = p(i + 1); op = toks[j]; r, k = p(j + 1); assert toks[k] == ')'
            return ({'and': 'and', 'or': 'or', 'implies': 'imp'}[op], l, r), k + 1
        raise ValueError(t)
    f, i = p(0); assert i == len(toks); return f


def thm_english(prompt):
    prem, concl = parse_prompt(prompt)
    return ('Premises: ' + '; '.join(fen(p) for p in prem) if prem else 'Premises: none') + '. Conclusion: ' + fen(concl) + '.'


def proof_english(proof):
    lines = parse_proof_tokens(proof.split())
    out = []
    for ln in lines:
        refs = ', '.join(str(r) for r in ln['refs'])
        out.append(f"line {ln['idx']} (depth {ln['depth']}): {fen(ln['formula'])} by {RULE_EN[ln['rule']]}" + (f' from {refs}' if refs else ''))
    return '\n'.join(out)


def english_to_tokens(text):
    """parse English lines back to the token format; unparseable -> None"""
    toks = []
    pat = re.compile(r'line\s+(\d+)\s*\(depth\s+(\d+)\)\s*:\s*(.+?)\s+by\s+([a-z\-]+)(?:\s+from\s+([\d,\s]+))?\s*$')
    for raw in text.strip().splitlines():
        raw = raw.strip()
        if not raw or raw.lower().startswith('qed'): continue
        m = pat.match(raw)
        if not m: return None
        idx, depth, formula, rule, refs = m.groups()
        if rule not in EN_RULE: return None
        try: f = en_parse_formula(formula)
        except Exception: return None
        toks += [f'N{idx}'] + ['|'] * int(depth) + [ftok(f), ':', EN_RULE[rule]] + [f'N{r.strip()}' for r in refs.split(',') if r.strip()] if refs else [f'N{idx}'] + ['|'] * int(depth) + [ftok(f), ':', EN_RULE[rule]]
        toks.append(';')
    toks.append('QED')
    return ' '.join(toks)


def lean_statement(prompt):
    prem, concl = parse_prompt(prompt)
    hyps = ' '.join(f'(h{j+1} : {lf(p)})' for j, p in enumerate(prem))
    return f'theorem t (P Q R S : Prop) {hyps} : {lf(concl)} := by'


def lean_body(prompt, proof):
    src = translate(prompt, proof)
    return src.split(':= by\n', 1)[1].rstrip('\n')


SYSTEM = {
    'tokens': "You prove theorems of propositional logic in a Fitch-style natural deduction calculus written in a token format. Atoms are P Q R S; F is falsum; connectives ~ & v > (fully parenthesised). A proof is a sequence of lines 'N<i> [| per box depth] <formula> : <RULE> [N<refs>] ;' ending with QED. Rules: PR (premise, in order), AS (assumption opens a box), R (reiterate), ANDI Na Nb, ANDE1 Na, ANDE2 Na, IMPE Nimp Nant, IMPI Nstart Nend (box from A ending in B gives (A > B)), ORI1 Na, ORI2 Na, ORE Ndisj Ns1 Ne1 Ns2 Ne2, NEGE Na Nnot (gives F), NEGI Nstart Nend (box from A ending in F gives (~ A)), BOTE Nf, DN Nnotnot. The last line must be the conclusion at depth 0. Answer with the proof only, starting with N1 and ending with QED.",
    'lean': "You prove theorems of propositional logic in Lean 4 (core library only, no Mathlib). Given a theorem statement ending in ':= by', answer with the tactic block only: a sequence of 'have nK : <Prop> := <term>' lines and a final 'exact nK'. Use And.intro as ⟨a, b⟩, .1/.2 for And elimination, application for modus ponens, 'fun (h : A) => by ... exact ...' for implications and negations (¬A is A → False), Or.inl / Or.inr, Or.elim h (fun (a : A) => by ...) (fun (b : B) => by ...), h.elim for False elimination, and Classical.byContradiction (fun hh => hnn hh) for double-negation elimination. Answer with the tactic lines only.",
    'english': "You prove theorems of propositional logic in a Fitch-style natural deduction calculus written in plain English. Formulas use P Q R S, False, and the connectives not/and/or/implies (fully parenthesised). Each proof line is 'line <i> (depth <d>): <formula> by <rule> from <cited line numbers>'; depth counts the open assumption boxes. Rules: premise, assumption (opens a box), reiteration, and-introduction, and-elimination-left, and-elimination-right, implication-elimination (cite the implication then the antecedent), implication-introduction (cite the box's first and last line), or-introduction-left, or-introduction-right, or-elimination (cite the disjunction, then the first box's first and last line, then the second box's), negation-elimination (cite the formula then its negation; gives False), negation-introduction (cite the box's first and last line; the box must end in False), falsum-elimination, double-negation-elimination. The last line must be the conclusion at depth 0. Answer with the proof lines only.",
}


def render(form, prompt, proof=None):
    """(question text, answer text or None)"""
    if form == 'tokens':
        return prompt, (proof if proof else None)
    if form == 'lean':
        return lean_statement(prompt), (lean_body(prompt, proof) if proof else None)
    if form == 'english':
        return thm_english(prompt), (proof_english(proof) if proof else None)


def build_messages(form, examples, prompt):
    msgs = [{'role': 'system', 'content': SYSTEM[form]}]
    parts = []
    for k, (ep, epr) in enumerate(examples):
        q, a = render(form, ep, epr)
        parts.append(f'### Example {k + 1}\nTheorem:\n{q}\nProof:\n{a}')
    q, _ = render(form, prompt)
    parts.append(f'### Problem\nTheorem:\n{q}\nProof:')
    msgs.append({'role': 'user', 'content': '\n\n'.join(parts)})
    return msgs


def cmd_build(a):
    rng = random.Random(0)
    val = [json.loads(l) for l in open('targets/validation_36.jsonl')]
    tests = [{'name': v['name'], 'prompt': v['prompt'], 'thm': v['thm'], 'bin': 'val36_' + v['bin'], 'gen_lines': None, 'src': 'val36'} for v in val]
    T = [json.loads(l) for l in open('data/transfer.jsonl')]
    by = collections.defaultdict(list)
    for t in T: by[t['n_lines']].append(t)
    lens = sorted(by); per = a.n_transfer // len(lens)
    for L in lens:
        rs = by[L]; rng.shuffle(rs)
        for t in rs[:per + (1 if L <= lens[a.n_transfer - per * len(lens) - 1] else 0)][:per + 1]:
            if len([x for x in tests if x['src'] == 'transfer']) >= a.n_transfer: break
            tests.append({'name': t['name'], 'prompt': t['prompt'], 'thm': t['thm'], 'bin': f'transfer_{L}', 'gen_lines': L, 'src': 'transfer'})
    test_keys = {canon_key(t['thm'].strip()) for t in tests}
    # example pools
    short = [json.loads(l) for l in open('data/heldout.jsonl')]
    short = [r for r in short if r['key'] not in test_keys and verify_text(r['prompt'] + ' ' + r['proof'])[0]]
    longp = collections.defaultdict(list)
    for l in open('artifacts/ei_abs_s0_cont/found_transfer_16.jsonl'):
        x = json.loads(l)
        if 7 <= x['written'] <= 9 and canon_key(x['thm'].strip()) not in test_keys:
            longp[x['name']].append(x)
    long_names = sorted(longp)
    out = []
    for draw in range(a.draws):
        r = random.Random(100 + draw)
        ex = []
        for L in (2, 3, 4, 5, 6):
            pool = [x for x in short if x['n_lines'] == L]; ex += [(x['prompt'], x['proof']) for x in r.sample(pool, 3 if L >= 5 else 2)]
        for nm in r.sample(long_names, 8):
            x = r.choice(longp[nm]); ex.append((x['prompt'], x['proof']))
        r.shuffle(ex)
        assert len(ex) == 20
        for form in ('tokens', 'lean', 'english'):
            for t in tests:
                out.append({'id': f'd{draw}_{form}_{t["name"]}', 'draw': draw, 'form': form, 'theorem_name': t['name'], 'bin': t['bin'], 'src': t['src'], 'gen_lines': t['gen_lines'],
                            'prompt': t['prompt'], 'thm': t['thm'], 'messages': build_messages(form, ex, t['prompt']), 'examples': [e[0] for e in ex]})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w') as f:
        for x in out: f.write(json.dumps(x) + '\n')
    print(len(out), 'prompts;', len(tests), 'theorems;', collections.Counter(t['bin'] for t in tests))
    # self-check: every example renders and its answer verifies in its own form
    ok = 0
    for form in ('tokens', 'lean', 'english'):
        for ep, epr in ex:
            q, ans = render(form, ep, epr)
            if form == 'tokens': ok += verify_text(ep + ' ' + ans)[0]
            elif form == 'english': ok += verify_text(ep + ' ' + english_to_tokens(ans))[0]
            else: ok += lean_check([lean_statement(ep) + '\n' + '\n'.join('  ' + l.lstrip() if not l.startswith('  ') else l for l in ans.splitlines()) + '\n'])[0][0]
    print('example self-check (3 forms x 20):', ok)


def extract(form, text):
    t = text.strip()
    t = re.sub(r'^```[a-zA-Z0-9]*\n|\n```$', '', t).strip()
    if form == 'tokens':
        m = re.search(r'(N\d+ .*?QED)', t, re.S)
        return m.group(1) if m else t
    if form == 'lean':
        if ':= by' in t: t = t.split(':= by', 1)[1]
        return t.strip('\n')
    return t


def score_one(form, prompt, text):
    body = extract(form, text)
    if form == 'tokens':
        ok, reason, nl = verify_text(prompt + ' ' + body); return ok, reason, body
    if form == 'english':
        tk = english_to_tokens(body)
        if tk is None: return False, 'english parse', body
        ok, reason, nl = verify_text(prompt + ' ' + tk); return ok, reason, tk
    lines = [('  ' + l.strip()) for l in body.splitlines() if l.strip()]
    src = lean_statement(prompt) + '\n' + '\n'.join(lines) + '\n'
    return None, src, body      # Lean checks are batched by the caller


def cmd_score(a):
    P = {json.loads(l)['id']: json.loads(l) for l in open(a.prompts)}
    gens = [json.loads(l) for l in open(a.gens)]
    rows = []; lean_srcs = []; lean_idx = []
    for g in gens:
        p = P[g['id']]
        for j, text in enumerate(g['outputs']):
            ok, info, body = score_one(p['form'], p['prompt'], text)
            row = {'id': g['id'], 'draw': p['draw'], 'form': p['form'], 'theorem_name': p['theorem_name'], 'bin': p['bin'], 'src': p['src'], 'sample': j, 'greedy': j == 0, 'ok': ok, 'info': info if ok is not None else '', 'body': body[:2000]}
            if ok is None:
                lean_srcs.append(info); lean_idx.append(len(rows))
            rows.append(row)
    if lean_srcs:
        res = lean_check(lean_srcs, 40)
        for k, (ok, msg) in zip(lean_idx, res):
            rows[k]['ok'] = bool(ok); rows[k]['info'] = msg
    with open(a.out, 'w') as f:
        for r in rows: f.write(json.dumps(r) + '\n')
    print(len(rows), 'scored')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build'); b.add_argument('--out', default='data/r1/prompts.jsonl'); b.add_argument('--n_transfer', type=int, default=200); b.add_argument('--draws', type=int, default=5)
    s = sub.add_parser('score'); s.add_argument('--prompts', required=True); s.add_argument('--gens', required=True); s.add_argument('--out', required=True)
    a = ap.parse_args()
    {'build': cmd_build, 'score': cmd_score}[a.cmd](a)
