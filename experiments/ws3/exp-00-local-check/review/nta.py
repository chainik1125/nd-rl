"""T1 gate recount: prompt-masked next-token accuracy and teacher-forced loss of the pinned
checkpoint on all 5,000 heldout records, with the fork's own GPT and Tokenizer('abs'), my code.
Right-padded batches; causal attention makes right padding harmless for earlier positions."""
import os, sys, json, torch, torch.nn.functional as Fn
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "..", "..", "..", "code", "ws3")); import ndpath
from load_pretrain import load_pretrain
model, tok, meta = load_pretrain()
recs = [json.loads(l) for l in open(os.path.join(ndpath.FORK_DATA, "heldout.jsonl"))]
seqs = []
for r in recs:
    p = tok.encode_prompt(r["prompt"]); b = tok.encode_proof(r["proof"])
    assert tok.decode(b) == r["proof"]
    seqs.append((p + b, len(p)))
correct = total = 0; loss_sum = 0.0
B = 250
with torch.no_grad():
    for i in range(0, len(seqs), B):
        chunk = seqs[i:i + B]; T = max(len(s) for s, _ in chunk)
        x = torch.zeros(len(chunk), T, dtype=torch.long)
        for j, (s, _) in enumerate(chunk): x[j, :len(s)] = torch.tensor(s)
        logits = model(x)  # (B, T, V)
        for j, (s, lp) in enumerate(chunk):
            tgt = torch.tensor(s[lp:]); lg = logits[j, lp - 1:len(s) - 1]
            correct += int((lg.argmax(-1) == tgt).sum()); total += len(tgt)
            loss_sum += Fn.cross_entropy(lg, tgt, reduction="sum").item()
out = {"n_records": len(recs), "n_target_tokens": total, "next_token_accuracy": correct / total, "loss": loss_sum / total,
       "ckpt_meta": {k: v for k, v in meta.items() if k in ("tok_mode", "stage", "seed", "n_params")}}
print(json.dumps(out, indent=1)); json.dump(out, open(os.path.join(HERE, "nta.json"), "w"), indent=1)
