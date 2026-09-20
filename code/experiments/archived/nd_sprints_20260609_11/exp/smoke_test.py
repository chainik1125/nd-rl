"""End-to-end smoke test (run on GPU pod before real experiments).

Generates a tiny dataset in-process, trains a tiny model briefly, samples,
verifies the pipeline produces verifier-checkable text.
"""
import json
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from nd.core import sample_tree, linearize, proof_to_tokens, theorem_id, rule_profile, DISCHARGING
from nd.verify import verify_text
from nd.tokenizer import STOI, ITOS, PAD
from train.model import GPT
from train import train as T
from train.evalgen import run_eval

rng = random.Random(0)
recs = []
seen = set()
while len(recs) < 3000:
    t = sample_tree(rng, n_steps=rng.randint(2, 10))
    if t is None:
        continue
    try:
        prem, concl, lines = linearize(t)
    except ValueError:
        continue
    if concl in prem or not (2 <= len(lines) <= 10):
        continue
    toks = proof_to_tokens(prem, concl, lines)
    text = ' '.join(toks)
    tid = theorem_id(prem, concl)
    if tid in seen:
        continue
    seen.add(tid)
    ok, reason, _ = verify_text(text)
    assert ok, reason
    rules = sorted(rule_profile(t))
    recs.append({'thm': tid, 'text': text, 'n_lines': len(lines),
                 'rules': rules, 'discharge': bool(set(rules) & DISCHARGING)})
print(f'generated {len(recs)} examples')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data = T.encode_examples(recs)
model = GPT(len(STOI), block_size=512, n_layer=2, n_head=4, d_model=128).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
r = random.Random(0)
model.train()
for it in range(300):
    idxs = [r.randrange(len(data)) for _ in range(64)]
    x, y, m, pos = T.make_batch(data, idxs, 512, device, r)
    _, loss = model(x, pos_ids=pos, targets=y, loss_mask=m)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if it % 100 == 0:
        print(it, float(loss))

results = run_eval(model, recs[:64], 4, 0.8, 200, 64, device)
solved = sum(rr['solved'] for rr in results)
print(f'smoke: solved {solved}/64 with tiny model (any number fine; pipeline works)')
ex = next((rr for rr in results if rr['solved']), None)
if ex:
    print('example verified proof:', ex['best_text'][:300])
print('SMOKE TEST PASSED')
