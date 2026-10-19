"""Overlap audit: do any of our theorems (by certify.theorem_key) appear in the group's other theorem files?

Scans every `THM ... PRF` prompt in: nd-takehome's working tree (targets/, data/), all data/**.jsonl on the take-home
fork's dan/dan branch, and every .jsonl on nd-rl's origin/charles/main. Sealed sets that are not in any repo
(e.g. textbook72's sealed half) cannot be checked here.

    python audit_overlap.py production/v2/*_f20.jsonl
"""
import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import certify as C  # noqa: E402

THM = re.compile(r'THM .*? PRF')
TAKEHOME = os.path.expanduser('~/repos/nd-takehome')
NDRL = os.path.expanduser('~/repos/nd-rl')


def sources():
    for p in glob.glob(f'{TAKEHOME}/targets/**/*.jsonl', recursive=True) + glob.glob(f'{TAKEHOME}/data/**/*.jsonl', recursive=True):
        yield p, open(p).read()
    for repo, ref, prefix in ((TAKEHOME, 'dan/dan', 'data/'), (NDRL, 'origin/charles/main', '')):
        names = subprocess.run(['git', '-C', repo, 'ls-tree', '-r', '--name-only', ref], capture_output=True, text=True).stdout.split()
        for n in names:
            if n.endswith('.jsonl') and n.startswith(prefix):
                yield f'{ref}:{n}', subprocess.run(['git', '-C', repo, 'show', f'{ref}:{n}'], capture_output=True, text=True).stdout


ours = {}
for p in sys.argv[1:]:
    for line in open(p):
        r = json.loads(line)
        ours.setdefault(r['key'], (p, r.get('cert_ge'), r.get('min_body')))
hits, n_thm = {}, 0
for src, text in sources():
    for m in set(THM.findall(text)):
        try:
            prem, concl = C.parse_thm(m)
        except Exception:
            continue
        n_thm += 1
        k = C.theorem_key(prem, concl)
        if k in ours:
            hits.setdefault(k, []).append(src)
print(f'{len(ours)} of our theorems vs {n_thm} theorem prompts scanned; {len(hits)} overlap')
for k, srcs in sorted(hits.items()):
    print(ours[k], k, sorted(set(srcs))[:3])
