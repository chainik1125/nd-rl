"""For Lean-rejected samples: did the fact Lean expected at the failing step exist anywhere earlier in the sample's own
proof (pointer/naming error) or was it never derived (bluff)?"""
import json, re, sys, collections
sys.path.insert(0, '/private/tmp/claude-501/-Users-robbie-repos-nd-rl/a1d02fb4-ba43-48c4-838b-e227293b6c2a/scratchpad/fork51604b3')
from nd_verify.verify import parse_formula, parse_proof_tokens
PREC = {'imp': 25, 'or': 30, 'and': 35, 'not': 100, 'atom': 200, 'bot': 200}
def lean(f, parent=0, right=False):
    t = f[0]
    if t == 'atom': return f[1]
    if t == 'bot': return 'False'
    if t == 'not':
        s = '¬' + lean(f[1], 100)
        return s
    op = {'and': ' ∧ ', 'or': ' ∨ ', 'imp': ' → '}[t]
    p = PREC[t]
    s = lean(f[1], p, False) + op + lean(f[2], p, True)
    need = p < parent or (p == parent and not right)
    return '(' + s + ')' if need else s
def norm(s): return s.replace(' ', '')
def run(fn):
    rs = [json.loads(l) for l in open(fn)]
    c = collections.Counter(); out = []
    for r in rs:
        if r['status'] != 'LEANREJ' or 'lean_msg' not in r: continue
        m = r['lean_msg'].replace('\n', ' ')
        g = re.search(r'but is expected to have type\s+(.*?)\s+in the application', m) or re.search(r'but is expected to have type\s+(.*)$', m)
        if not g: continue
        exp = norm(g.group(1))
        if r['err_rule'] == 'ORE':
            c[('ORE', 'skip')] += 1; continue
        # decoded ND lines: re-decode from text via the strict inverse is costly; use the have statements in order
        haves = re.findall(r'have n\d+ : (.*?) :=', r['text'])
        hyps = re.findall(r'fun \( n\d+ : (.*?) \) =>', r['text'])
        def tolean(s):
            m2 = {'(': '(', ')': ')', '¬': '~', '∧': '&', '∨': 'v', '→': '>', 'False': 'F'}
            nd = [m2.get(x, x) for x in s.split()]
            try: f, _ = parse_formula(nd + ['$'], 0)
            except Exception: return None
            return norm(lean(f))
        k = r['err_have']
        before = [tolean(h) for h in haves[:k]] + [tolean(h) for h in hyps]  # hyps: any box hypothesis (scope ignored)
        n_pr = sum(1 for h in re.findall(r':= (h\d+)', r['text']))
        prem = [tolean(h) for h in haves[:n_pr]]
        if exp in prem: kind = 'is-a-premise'
        elif exp in before: kind = 'derived-earlier'
        elif exp in [tolean(h) for h in haves]: kind = 'derived-later'
        else: kind = 'never-derived'
        c[(r['err_rule'], kind)] += 1; c[('ALL', kind)] += 1
        out.append((r['name'], r['err_rule'], kind, exp))
    for k, v in sorted(c.items(), key=lambda kv: (kv[0][0] != 'ALL', -kv[1])): print(k, v)
    return out
if __name__ == '__main__':
    for fn in sys.argv[1:]: print(fn); run(fn)
