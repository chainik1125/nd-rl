"""Classify sampled attempts: parse failures (with how far the strict decoder got), Lean rejections (first Lean error,
mapped to the `have` it is on and the rule used there), and structural features vs the gold proof.
Usage: ND_ARCH=looped python analyze.py samples_*.jsonl  -> writes <name>.classified.jsonl and prints a table.
"""
import sys, os, json, re, subprocess, tempfile, collections
sys.path.insert(0, '/private/tmp/claude-501/-Users-robbie-repos-nd-rl/a1d02fb4-ba43-48c4-838b-e227293b6c2a/scratchpad/probe')
os.environ.setdefault("ND_FORK", "/private/tmp/claude-501/-Users-robbie-repos-nd-rl/a1d02fb4-ba43-48c4-838b-e227293b6c2a/scratchpad/fork51604b3")
sys.path.insert(0, "/Users/robbie/repos/nd-rl/code/experiments/current/long_rl_20261014")
import harness as H  # noqa
import pretrain as P  # noqa
from nd_verify.verify import parse_proof_tokens  # noqa
lt = sys.modules['lean_tok']
B = os.path.dirname(os.path.abspath(__file__))
tok = P.make_tokenizer(P.TOK_MODE)
LEAN = os.path.expanduser('~/.elan/bin/lean')
ERR = re.compile(r'^[^\n]*?:(\d+):(\d+): error: (.*?)(?=^\S+\.lean:\d+:\d+: |\Z)', re.M | re.S)


# ---------------------------------------------------------------- partial strict decode (copy of lean_tok.inverse that reports progress)
def partial_inverse(toks):
    """-> (ND lines emitted so far, fail reason or None, token position of failure, depth at failure)"""
    pos = [0]; out = []; forms = {}; n_pr = [0]; seen_non_pr = [False]; cur_depth = [0]
    PF = lt.ParseFail
    binds = {}; path = []; nbox = [0]; cur_path = [()]; opened = {}; box_start = [0]
    def peek(): return toks[pos[0]] if pos[0] < len(toks) else None
    def eat(x=None):
        t = peek()
        if t is None or (x is not None and t != x): raise PF(f'expected {x}')
        pos[0] += 1; return t
    def name_tok():
        t = eat()
        if not (t[0] == 'n' and t[1:].isdigit()): raise PF('name')
        return t
    def formula():
        m = {'(': '(', ')': ')', '¬': '~', '∧': '&', '∨': 'v', '→': '>', 'False': 'F', **{a: a for a in lt.ATOMS}}
        j = pos[0]; nd = []
        while j < len(toks) and toks[j] in m: nd.append(m[toks[j]]); j += 1
        try: f, used = lt.parse_formula(nd + ['$'], 0)
        except Exception: raise PF('formula')
        pos[0] += used; return f
    def ref(scope):
        t = name_tok()
        if t not in scope: raise PF('unbound')
        return scope[t]
    def emit(depth, f, rule, refs):
        i = len(out) + 1
        out.append(f'N{i} ' + '| ' * depth + f'{lt.fnd(f)} : {rule}' + ''.join(f' N{r}' for r in refs) + ' ;')
        forms[i] = f; return i
    def box(scope, depth):
        eat('('); eat('fun'); eat('('); nm = name_tok(); eat(':'); hf = formula(); eat(')'); eat('=>'); eat('by')
        box_start.append(len(out))
        s = emit(depth + 1, hf, 'AS', [])
        sc = dict(scope); sc[nm] = s
        nbox[0] += 1; path.append(nbox[0]); cur_path[0] = tuple(path)
        binds.setdefault(nm, []).append((s, tuple(path)))
        cur_depth[0] = depth + 1
        lastl = stmts(sc, depth + 1, s)
        eat('exact')
        if peek() == '(':
            eat('('); e = ref(sc); eat(':'); eat('False'); eat(')'); neg = True
        else:
            e = ref(sc); neg = False
        if e != lastl: raise PF('exact not last')
        eat(')')
        path.pop(); cur_path[0] = tuple(path); box_start.pop()
        cur_depth[0] = depth
        return s, e, neg
    def stmts(scope, depth, lastl):
        while peek() == 'have':
            eat('have'); nm = name_tok(); eat(':'); f = formula(); eat(':=')
            opened[nm] = (tuple(path), f)
            t = peek()
            if t is None: raise PF('eof')
            if t[0] == 'h' and t[1:].isdigit():
                eat(); k = int(t[1:])
                if depth != 0 or seen_non_pr[0] or k != n_pr[0] + 1: raise PF('PR position')
                n_pr[0] += 1; i = emit(0, f, 'PR', [])
            else:
                seen_non_pr[0] = True
                if t == '⟨':
                    eat(); a = ref(scope); eat(','); b = ref(scope); eat('⟩'); rule, refs = 'ANDI', [a, b]
                elif t in ('Or.inl', 'Or.inr'):
                    eat(); a = ref(scope); rule, refs = ('ORI1' if t == 'Or.inl' else 'ORI2'), [a]
                elif t == 'Classical.byContradiction':
                    eat(); eat('('); eat('fun'); eat('hh'); eat('=>'); a = ref(scope); eat('hh'); eat(')'); rule, refs = 'DN', [a]
                elif t == '(':
                    s, e, neg = box(scope, depth); rule, refs = ('NEGI' if neg else 'IMPI'), [s, e]
                elif t == 'Or.elim':
                    eat(); j = ref(scope); s1, e1, g1 = box(scope, depth); s2, e2, g2 = box(scope, depth)
                    if g1 or g2: raise PF('ORE box')
                    rule, refs = 'ORE', [j, s1, e1, s2, e2]
                elif t == 'False':
                    eat(); eat('.elim'); a = ref(scope); rule, refs = 'BOTE', [a]
                else:
                    a = ref(scope); t2 = peek()
                    if t2 == '.1': eat(); rule, refs = 'ANDE1', [a]
                    elif t2 == '.2': eat(); rule, refs = 'ANDE2', [a]
                    elif t2 == ';': rule, refs = 'R', [a]
                    else:
                        b = ref(scope)
                        if forms[a][0] == 'not': rule, refs = 'NEGE', [b, a]
                        else: rule, refs = 'IMPE', [a, b]
                i = emit(depth, f, rule, refs)
            eat(';'); scope[nm] = i; lastl = i
            opened.pop(nm, None)
            binds.setdefault(nm, []).append((i, tuple(path)))
        return lastl
    try:
        top = {}
        lastl = stmts(top, 0, None)
        eat('exact'); e = ref(top)
        if pos[0] != len(toks): raise PF('trailing tokens')
        if lastl is None or e != lastl: raise PF('exact not last')
        return out, None, pos[0], 0, binds, (), opened, box_start[-1]
    except PF as e:
        return out, str(e), pos[0], cur_depth[0], binds, cur_path[0], opened, box_start[-1]


# ---------------------------------------------------------------- Lean errors
def lean_errors(items):
    """items: [(statement, tactic text)] -> per item list of (col, message)"""
    if not items: return []
    lines = [f'{s.replace("theorem t ", f"theorem t{k} ", 1)} {b}' for k, (s, b) in enumerate(items)]
    wd = tempfile.mkdtemp(prefix='bprobe_')
    fn = os.path.join(wd, 'c.lean')
    open(fn, 'w').write('set_option linter.unusedVariables false\n' + '\n'.join(lines) + '\n')
    p = subprocess.run([LEAN, '-DmaxErrors=100000000', fn], capture_output=True, text=True, timeout=600)
    o = p.stdout + p.stderr
    res = [[] for _ in items]
    for m in ERR.finditer(o):
        ln, col, msg = int(m.group(1)) - 2, int(m.group(2)), m.group(3).strip()
        if 0 <= ln < len(items):
            res[ln].append((col - len(items[ln][0]) - 1, msg))
    os.remove(fn); os.rmdir(wd)
    return res


def err_kind(msg):
    h = msg.split('\n')[0]
    if h.startswith('Application type mismatch'): return 'app-mismatch'
    if h.startswith('Invalid projection'): return 'proj-non-and'
    if h.startswith('Function expected'): return 'apply-non-imp'
    if h.startswith('Type mismatch'): return 'type-mismatch'
    if 'unknown' in h: return 'unknown-id'
    if 'unsolved goals' in h: return 'unsolved'
    return h[:40]


def have_at(text, col):
    """index (0-based, among `have`s in text order) of the have whose span contains col; -1 if in the final exact"""
    starts = [m.start() for m in re.finditer(r'\bhave\b', text)]
    k = sum(1 for s in starts if s <= col) - 1
    fin = text.rfind(' exact ')
    if k < 0 or (col >= fin and fin > starts[-1] if starts else True):
        return -1
    return k


def rule_of_have(text, k):
    """the term head of the k-th have"""
    parts = re.split(r'\bhave\b', text)[1:]
    if k < 0 or k >= len(parts): return 'exact'
    term = parts[k].split(':=', 1)[1].strip() if ':=' in parts[k] else ''
    t = term.split()[0] if term else ''
    if t.startswith('h') and t[1:].isdigit(): return 'PR'
    if t == '⟨': return 'ANDI'
    if t in ('Or.inl', 'Or.inr'): return 'ORI'
    if t == 'Classical.byContradiction': return 'DN'
    if t == 'Or.elim': return 'ORE'
    if t == '(': return 'IMPI/NEGI'
    if t == 'False.elim': return 'BOTE'
    if t.endswith('.1') or t.endswith('.2'): return 'ANDE'
    toks = term.split()
    if len(toks) >= 2 and toks[1] != ';': return 'IMPE/NEGE'
    return 'R'


# ---------------------------------------------------------------- structural features
def skeleton(nd):
    """gold/sample ND proof -> (closing rule of last line, top-level box rules in order, max depth, n body lines,
    set of (rule, formula) body lines)"""
    lines = parse_proof_tokens(nd.split())
    body = [l for l in lines if l['rule'] != 'PR']
    top_boxes = [l['rule'] for l in body if l['depth'] == 0 and l['rule'] in ('IMPI', 'NEGI', 'ORE')]
    return {'close': body[-1]['rule'] if body else None, 'top_boxes': top_boxes,
            'depth': max((l['depth'] for l in lines), default=0), 'n_body': len(body),
            'facts': {(l['rule'], lt.fnd(l['formula'])) for l in body},
            'rules': collections.Counter(l['rule'] for l in body)}


def classify_file(fn):
    tg = {json.loads(l)['name']: json.loads(l) for l in open(f'{B}/targets.jsonl')}
    rows = [json.loads(l) for l in open(fn)]
    rej_items, rej_idx = [], []
    out = []
    for r in rows:
        t = tg[r['name']]
        goal = lt.fnd(lt.parse_prompt(t['prompt'])[1])
        gold = skeleton(t['proof']) if t.get('proof') else None
        for j, s in enumerate(r['samples']):
            raw = s['raw'].split()
            ended = raw and raw[-1] == '<eos>'
            toks = raw[:-1] if ended else raw
            text = tok.text(toks)
            nd_lines, reason, pos, depth, binds, cur_path, opened, bstart = partial_inverse(toks)
            n_have = text.count('have ')
            haves = re.findall(r'have (n\d+) : (.*?) :=', text)
            stmts = [h[1] for h in haves]
            dup = n_have - len(set(stmts))
            top_stmts = []  # formulas of top-level haves (depth 0): approximate via partial decode
            c = {'name': r['name'], 'pool': r['pool'], 'body': r['body'], 'loops': r['loops'], 'j': j, 'ok': s['ok'],
                 'n_tok': len(toks), 'ended': bool(ended), 'n_have': n_have, 'dup_stmts': dup, 'goal': goal,
                 'parse_fail': reason, 'fail_pos': pos, 'fail_depth': depth, 'decoded_lines': len(nd_lines),
                 'text': text}
            if reason and reason != 'no-eos':
                # which have was being parsed when it failed
                col = len(tok.text(toks[:pos]))
                c['fail_have'] = have_at(text, col)
                c['fail_rule'] = rule_of_have(text, c['fail_have'])
                c['tok_at_fail'] = toks[pos] if pos < len(toks) else None
            if reason == 'unbound' or (reason and c.get('tok_at_fail') is None and toks and pos > 0 and toks[pos - 1].startswith('n')):
                nm = toks[pos - 1]
                before = ' '.join(toks[:pos - 1]); after = ' '.join(toks[pos:])
                if nm in opened:
                    c['unbound_kind'] = 'cites-enclosing-have'; c['unbound_name'] = nm
                    bp, f = opened[nm]
                    c['enclosing_levels_up'] = len(cur_path) - len(bp)
                    # what the current box must end with, if the enclosing have is the box's own: consequent (IMPI), False (NEGI) or f itself (ORE)
                    box_lines = nd_lines[bstart:]
                    hyp = box_lines[0].split(' ', 1)[1].lstrip('| ').rsplit(' : ', 1)[0] if box_lines else None
                    fs = lt.fnd(f)
                    need = None
                    if f[0] == 'imp' and lt.fnd(f[1]) == hyp: need = lt.fnd(f[2])
                    elif f[0] == 'not' and lt.fnd(f[1]) == hyp: need = 'F'
                    else: need = fs  # Or.elim branch (or an IMPI box whose hypothesis is not the antecedent)
                    inbox = [l.split(' ', 1)[1].lstrip('| ').rsplit(' : ', 1)[0] for l in box_lines]
                    c['box_need'] = need; c['box_hyp'] = hyp; c['box_need_derived'] = need in inbox[1:]; c['box_n_lines'] = len(box_lines) - 1; c['box_kind'] = 'IMPI' if (f[0] == 'imp' and lt.fnd(f[1]) == hyp) else 'NEGI' if (f[0] == 'not' and lt.fnd(f[1]) == hyp) else 'ORE/other'
                elif nm in binds:
                    idx, bp = binds[nm][-1]
                    c['unbound_name'] = nm
                    # relation of the defining box path to the current one
                    k = 0
                    while k < min(len(bp), len(cur_path)) and bp[k] == cur_path[k]: k += 1
                    if len(bp) > k and len(cur_path) == k: c['unbound_kind'] = 'leak-after-box'   # defined in a box that closed, used at the enclosing level
                    elif len(bp) > k and len(cur_path) > k: c['unbound_kind'] = 'leak-sibling-box'  # defined in a closed box, used in a sibling box (other Or.elim branch etc.)
                    else: c['unbound_kind'] = 'other-defined'
                    c['unbound_def_depth'] = len(bp); c['unbound_use_depth'] = len(cur_path)
                    c['unbound_def_is_hyp'] = nd_lines[idx - 1].endswith(': AS ;') if idx <= len(nd_lines) else None
                elif re.search(r'(have|fun \() ' + nm + r' :', after):
                    c['unbound_kind'] = 'forward'
                else:
                    c['unbound_kind'] = 'never'
                    # self-reference: have nX := ... nX
                    if re.search(r'have ' + nm + r' : [^;]*:= [^;]*\b' + nm + r'\b', ' '.join(toks[:pos + 1])): c['unbound_kind'] = 'self-ref'
            if s['ok']:
                c['status'] = 'OK'
            elif not ended:
                c['status'] = 'PARSE no-eos'
            elif reason:
                c['status'] = 'PARSE ' + reason
            else:
                c['status'] = 'LEANREJ'
                rej_items.append((tok.statement(t['prompt']), text)); rej_idx.append(len(out))
            # last top-level formula vs goal (when parse got to the end or at least some lines)
            if nd_lines:
                last = nd_lines[-1]
                c['last_line'] = last
                # does the last top-level (depth 0) decoded line state the goal?
                top0 = [l for l in nd_lines if not l.split(' ', 1)[1].startswith('|')]
                if top0:
                    f = top0[-1].split(' ', 1)[1].rsplit(' : ', 1)[0]
                    c['last_top_is_goal'] = (f == goal)
                    c['goal_stated_anywhere'] = any(l.split(' ', 1)[1].lstrip('| ').rsplit(' : ', 1)[0] == goal for l in nd_lines)
                if gold:
                    try:
                        sk = skeleton(' '.join(nd_lines) + ' QED') if not reason else None
                    except Exception:
                        sk = None
                    if sk:
                        c['close'] = sk['close']; c['top_boxes'] = sk['top_boxes']; c['depth'] = sk['depth']
                        c['gold_close'] = gold['close']; c['gold_top_boxes'] = gold['top_boxes']; c['gold_depth'] = gold['depth']
                        c['gold_fact_overlap'] = len(sk['facts'] & gold['facts']) / max(1, len(gold['facts']))
            out.append(c)
    errs = lean_errors(rej_items)
    for i, el in zip(rej_idx, errs):
        c = out[i]
        if el:
            col, msg = el[0]
            c['lean_err'] = err_kind(msg); c['lean_msg'] = msg[:300]; c['err_have'] = have_at(c['text'], col)
            c['err_rule'] = rule_of_have(c['text'], c['err_have'])
            c['err_frac'] = (c['err_have'] + 1) / max(1, c['n_have']) if c['err_have'] >= 0 else 1.0
            c['n_errs'] = len(el)
        else:
            c['lean_err'] = 'none?'
    ofn = fn.replace('.jsonl', '.classified.jsonl')
    with open(ofn, 'w') as f:
        for c in out: f.write(json.dumps(c) + '\n')
    cnt = collections.Counter(c['status'] if c['status'] != 'LEANREJ' else 'LEANREJ ' + c.get('lean_err', '?') for c in out)
    print(fn, len(out), 'samples'); [print(f'  {k:40s} {v}') for k, v in cnt.most_common()]
    return out


if __name__ == '__main__':
    for fn in sys.argv[1:]:
        classify_file(fn)
