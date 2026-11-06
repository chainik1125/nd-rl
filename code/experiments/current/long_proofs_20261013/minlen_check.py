"""Shortest-proof upper bound (Dan's fork minlen.py) for a sample of long generated proofs.

    python minlen_check.py IN.jsonl MIN_BODY N BOUND SECONDS OUT.jsonl
Skips records with a premise and its negation both present (contra: trivially provable via F).
"""
import json, random, sys
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, '/private/tmp/claude-501/-Users-robbie-repos-nd-rl/a1d02fb4-ba43-48c4-838b-e227293b6c2a/scratchpad/fork')
import nd_verify
V = sys.modules['nd_verify.verify']
V.ATOM_NAMES.update(['P', 'Q', 'R', 'S', 'T', 'U', 'W', 'X', 'Y', 'Z'])
from minlen import minlen, parse_thm

src, min_body, n, bound, secs, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), sys.argv[6]


def contra(prompt):
    prem, _ = parse_thm(prompt)
    return any(('not', p) in prem for p in prem)


def work(r):
    prompt = r['text'].split(' PRF ', 1)[0] + ' PRF'
    m = minlen(prompt, bound=bound, time_limit=secs)
    return {**{k: r[k] for k in ('n_lines', 'n_body', 'n_prem', 'n_atoms', 'max_depth')},
            'min_lines_ub': m['min_lines_ub'], 'timeout': m['timeout'], 'prompt': prompt}


if __name__ == '__main__':
    rs = [json.loads(l) for l in open(src)]
    long = [r for r in rs if r['n_body'] >= min_body]
    n_contra = sum(contra(r['text'].split(' PRF ', 1)[0] + ' PRF') for r in long)
    ok = [r for r in long if not contra(r['text'].split(' PRF ', 1)[0] + ' PRF')]
    print(f'{len(long)} with body >= {min_body}; {n_contra} contra ({n_contra / max(len(long), 1):.0%}); {len(ok)} non-contra')
    sample = random.Random(0).sample(ok, min(n, len(ok)))
    with ProcessPoolExecutor(8) as ex, open(out, 'w') as f:
        res = list(ex.map(work, sample))
        for x in res:
            f.write(json.dumps(x) + '\n')
    short = [x for x in res if x['min_lines_ub'] is not None]
    print(f'sampled {len(res)}: shorter proof found (<= {bound} lines) for {len(short)}; timeout {sum(x["timeout"] for x in res)}; none-in-bound {sum(x["min_lines_ub"] is None and not x["timeout"] for x in res)}')
    if short:
        print('  found lengths', sorted(x['min_lines_ub'] for x in short), ' vs generated', sorted(x['n_lines'] for x in short)[:10], '...')
