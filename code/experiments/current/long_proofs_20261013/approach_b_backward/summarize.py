"""Summarise certify.py outputs: certify-time distribution, timeout rate, written vs minimal lengths.
    python summarize.py cert_*.jsonl"""
import collections, json, statistics, sys
for f in sys.argv[1:]:
    rs = [json.loads(l) for l in open(f)]
    if not rs:
        print(f, 'empty'); continue
    secs = sorted(r['cert_secs'] for r in rs)
    q = lambda p: secs[min(len(secs) - 1, int(p * len(secs)))]
    to = [r for r in rs if r['timeout']]
    cert = [r for r in rs if r['cert_ge']]
    exact = [r for r in rs if r['min_body'] is not None]
    print(f'{f}: {len(rs)} records, floor {rs[0]["floor"]}; timeouts {len(to)} ({len(to)/len(rs):.0%}); '
          f'cert_ge {len(cert)}; exact-min found {len(exact)}; cert secs median {q(.5)} p25 {q(.25)} p75 {q(.75)} sum {sum(secs):.0f}')
    print('  written body:', dict(sorted(collections.Counter(r['n_body'] for r in rs).items())))
    print('  premises:', dict(sorted(collections.Counter(r['n_prem'] for r in rs).items())), ' atoms:', dict(sorted(collections.Counter(r['n_atoms'] for r in rs).items())))
    if cert:
        print('  certified: body', dict(sorted(collections.Counter(r['n_body'] for r in cert).items())), 'prem', dict(sorted(collections.Counter(r['n_prem'] for r in cert).items())),
              'secs', [r['cert_secs'] for r in cert])
    if exact:
        print('  exact: (written, minimal, secs):', [(r['n_body'], r['min_body'], r['cert_secs']) for r in exact])
        print('  written == minimal for', sum(r['n_body'] == r['min_body'] for r in exact), 'of', len(exact))
    if to:
        print('  timeouts by written body:', dict(sorted(collections.Counter(r['n_body'] for r in to).items())))
