"""Certify a jsonl of {"text": "THM ... QED"} records with the signature bound; write cert_ge_sig per record.

    python run_e.py IN.jsonl OUT.jsonl
Output record: the input's n_body/n_prem/n_atoms/S plus prompt, cert_ge_sig (= lb), n_sig, n_hyp.
cert_ge_sig = L means: every proof in the nu-system (covers nd_verify's rules with boxes cited after they close,
minlen's whole space, and every proof the nd2lean + Lean gate accepts) has >= L body lines.
"""
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sigbound as SB  # noqa: E402


def main():
    src, dst = sys.argv[1:3]
    rows = []
    with open(dst, 'w') as out:
        for line in open(src):
            r = json.loads(line)
            prompt = r['text'].split(' PRF ', 1)[0] + ' PRF'
            b = SB.sig_bound_prompt(prompt)
            row = {k: r.get(k) for k in ('n_body', 'n_prem', 'n_atoms', 'S')}
            row.update(prompt=prompt, cert_ge_sig=b['lb'], n_sig=b['n_sig'], n_hyp=b['n_hyp'])
            out.write(json.dumps(row) + '\n')
            rows.append(row)
    print(f'{len(rows)} records')
    for t in (20, 30, 40, 50, 60, 70):
        print(f'  cert_ge_sig >= {t}: {sum(r["cert_ge_sig"] >= t for r in rows)}')
    by = collections.defaultdict(list)
    for r in rows:
        lo = 35 if r['n_body'] < 40 else min(r['n_body'] // 10 * 10, 70)
        by[lo].append(r)
    print('bucket (written)  n   >=20  >=30  >=40  bound min/median/max   bound/written mean (min)')
    for lo in sorted(by):
        rs = by[lo]
        b = sorted(r['cert_ge_sig'] for r in rs)
        rat = [r['cert_ge_sig'] / r['n_body'] for r in rs]
        name = '35-39' if lo == 35 else ('70-80' if lo >= 70 else f'{lo}-{lo + 9}')
        print(f'  {name:14s} {len(rs):4d} {sum(x >= 20 for x in b):5d} {sum(x >= 30 for x in b):5d} '
              f'{sum(x >= 40 for x in b):5d}   {b[0]:3d} / {b[len(b) // 2]:3d} / {b[-1]:3d}         '
              f'{sum(rat) / len(rat):.3f} ({min(rat):.3f})')


if __name__ == '__main__':
    main()
