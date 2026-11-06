"""Check certify_fast against minlen's labels in approach_d_theorem_first/*_cert*.jsonl.

    python validate.py OUT.jsonl [--secs 60] [--procs 2] [--mem 1024] [--files r2_cert20 ...]

Per record: label min_body -> we must find exactly that body (and a verified proof); label cert_ge -> we must
exhaust; label timeout -> whatever we get is new information (reported separately). Same floor as the label.
"""
import argparse
import collections
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from certify_fast import run_fast  # noqa: E402

FILES = ['r2_cert20', 'r1_cert20', 't1_cert20', 't1_rest_cert16', 'hunt_cert18']
D = os.path.join(os.path.dirname(HERE), 'approach_d_theorem_first')


def work(args):
    rec, secs, mem = args
    r = run_fast(rec['prompt'], rec['floor'], secs, mem)
    if rec.get('min_body') is not None:
        lab = ('found', rec['min_body'])
    elif rec.get('cert_ge'):
        lab = ('exhausted', None)
    else:
        lab = ('timeout', None)
    got = (r['status'], r.get('body'))
    if lab[0] == 'timeout':
        verdict = 'new' if got[0] in ('found', 'exhausted') else 'both_timeout' if got[0] == 'timeout' else 'ERROR'
    elif got[0] == 'timeout':
        verdict = 'we_timeout'
    else:
        verdict = 'ok' if got == lab else 'MISMATCH'
    return {'prompt': rec['prompt'], 'floor': rec['floor'], 'label': lab, 'got': got, 'verdict': verdict,
            'minlen_secs': rec.get('cert_secs'), 'fast_secs': round(r['secs'], 3), 'stats': r.get('stats'),
            'error': r.get('error')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--secs', type=float, default=60)
    ap.add_argument('--procs', type=int, default=2)
    ap.add_argument('--mem', type=float, default=1024)
    ap.add_argument('--files', nargs='*', default=FILES)
    a = ap.parse_args()
    recs = []
    for f in a.files:
        for line in open(os.path.join(D, f + '.jsonl')):
            r = json.loads(line)
            r['_file'] = f
            recs.append(r)
    cnt = collections.Counter()
    with ProcessPoolExecutor(a.procs) as ex, open(a.out, 'w') as fo:
        for rec, r in zip(recs, ex.map(work, [(r, a.secs, a.mem) for r in recs])):
            r['file'] = rec['_file']
            fo.write(json.dumps(r) + '\n')
            fo.flush()
            cnt[r['verdict']] += 1
            if r['verdict'] in ('MISMATCH', 'ERROR', 'new'):
                print(r['verdict'], r['file'], r['label'], r['got'], r['fast_secs'], r['prompt'], r.get('error'), flush=True)
    print(dict(cnt))


if __name__ == '__main__':
    main()
