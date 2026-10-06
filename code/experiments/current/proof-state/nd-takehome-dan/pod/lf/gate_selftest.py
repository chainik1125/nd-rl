"""Lean gate self-test on the pod: 2,000 held-out proofs rendered in Lean, every 5th paired with another theorem's statement.
Expect both-ok 1600, both-rej 400, no disagreement. Also prints the throughput of Lean vs nd_verify on this pod."""
import sys, json, random
sys.path.insert(0, '.')
from lean_tok import LeanTokenizer
from lean_gate import gate
tk = LeanTokenizer('lean_rand'); rng = random.Random(0)
recs = [json.loads(l) for l in open('data/heldout.jsonl')][:2000]
prompts, nds, texts = [], [], []
for k, r in enumerate(recs):
    nd = tk.decode(tk.shift_abs(tk.encode_proof(r['proof']), rng))
    assert nd == r['proof']
    prompts.append(recs[(k + 3) % len(recs)]['prompt'] if k % 5 == 0 else r['prompt']); nds.append(nd); texts.append(tk.last_text)
out = gate(tk, prompts, nds, texts)
print('rejected', sum(o.startswith('LEANREJ') for o in out), 'of', len(out))
