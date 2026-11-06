"""Score an estimate.py output against exact labels.   python score_est.py EST.jsonl [--labels logs/exact_labels.jsonl]
Truth per record: min_body (exact) or cert_ge; records with neither are 'unknown'."""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import certify as C  # noqa: E402
from nd_verify import verify_text  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('path')
ap.add_argument('--labels', nargs='*', default=[os.path.join(HERE, 'logs', 'exact_labels.jsonl')])
a = ap.parse_args()
labels = {}
for p in a.labels:
    if os.path.exists(p):
        for line in open(p):
            d = json.loads(line)
            if d.get('min_body') is not None:
                labels[d['key']] = d['min_body']
rs = [json.loads(line) for line in open(a.path)]
floor = rs[0]['floor']
tp = fp = fn = tn = unk = 0
bad_proofs = 0
pf_claims = []
stage_found = collections.Counter()
upper_vs_true = collections.Counter()
reach = collections.defaultdict(list)
for r in rs:
    tm = r.get('min_body')
    if tm is None:
        tm = labels.get(r['key'])
    if r.get('proof'):
        ok, reason, nl = verify_text(r['prompt'] + ' ' + r['proof'])
        if not ok or nl - r['n_prem'] != r['upper']:
            bad_proofs += 1
        if tm is not None and r['upper'] < tm:
            print('ANOMALY: proof shorter than exact label', r['key'])
    if r['found_by']:
        stage_found[r['found_by']] += 1
        if tm is not None:
            upper_vs_true[r['upper'] - tm] += 1
    for name, stop, exh, secs, memo in r['stages']:
        reach[name].append(exh)
    claim = r['claim_ge'] is not None
    if tm is None and not (r.get('cert_ge') and r['cert_ge'] >= floor):
        unk += 1
        continue
    long = tm is None or tm >= floor
    if claim:
        pf_claims.append(r['p_false'])
    if claim and long: tp += 1
    elif claim and not long: fp += 1; print('FALSE CERTIFICATE:', tm, r['key'], r.get('reached'), r['p_false'])
    elif not claim and long: fn += 1
    else: tn += 1
secs = [r['secs'] for r in rs]
print(f'{a.path}: floor {floor}, {len(rs)} records, {unk} truth-unknown')
print(f'  TP {tp}  FP {fp}  FN {fn}  TN {tn}   false-cert rate FP/(FP+TN) = {fp / max(1, fp + tn):.4f}   recall TP/(TP+FN) = {tp / max(1, tp + fn):.3f}')
print(f'  claims: {tp + fp}; p_false among claims: {sorted(collections.Counter(pf_claims).items())}')
print(f'  proofs returned: {sum(1 for r in rs if r.get("proof"))}, failing nd_verify or length mismatch: {bad_proofs}')
print(f'  found by stage: {dict(stage_found)};  upper - exact_min histogram: {dict(sorted(upper_vs_true.items()))}')
print(f'  covered_ge among non-claims without a proof: {sorted(collections.Counter(r.get("covered_ge") for r in rs if r["found_by"] is None and r["claim_ge"] is None).items(), key=str)}')
print(f'  secs: mean {sum(secs) / len(secs):.2f}, max {max(secs):.1f}, total {sum(secs):.0f}  ->  {3600 / (sum(secs) / len(secs)):.0f} theorems per core-hour')
long_secs = [r['secs'] for r in rs if r['found_by'] is None]
if long_secs:
    print(f'  secs on records with no proof found (the long ones): mean {sum(long_secs) / len(long_secs):.1f} (n={len(long_secs)})')
