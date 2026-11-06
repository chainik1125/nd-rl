"""Turn hunt jsonl rows with min_body >= MIN into {"text"} records, using the proof found by the semantic-pruned
minlen (verifier-checked, exactly minimal within minlen's space) as the proof text.
    python hunt_to_records.py HUNT.jsonl OUT.jsonl MIN"""
import json, sys
from minlen_sem import minlen_sem
from minlen import parse_thm
src, dst, mn = sys.argv[1], sys.argv[2], int(sys.argv[3])
n = 0
with open(dst, 'w') as f:
    for l in open(src):
        r = json.loads(l)
        if r['min_body'] is None or r['min_body'] < mn:
            continue
        prem, _ = parse_thm(r['prompt'])
        m = minlen_sem(r['prompt'], bound=len(prem) + r['min_body'], time_limit=120)
        if m['proof'] is None:
            print('no proof', r['prompt']); continue
        f.write(json.dumps({'text': r['prompt'] + ' ' + m['proof'], 'hunt_min_body': r['min_body'], 'S': r['S']}) + '\n'); n += 1
print(n, 'records')
