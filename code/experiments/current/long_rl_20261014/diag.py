"""Pod-side diagnostic: greedy samples from a checkpoint on old dev and the 7-9 band, with Lean's error text.

    ND_FORK=... ND_ARCH=... python diag.py OUT_DIR/pretrain.pt
"""
import json
import os
import sys
import tempfile
import subprocess

import harness as H
import pretrain as P

ck = H.torch.load(sys.argv[1], map_location=H.DEV, weights_only=True)
model, tok, _ = P.pretrain(H.read(f"{H.DATA}/pretrain.jsonl.gz")[:1100], 1e-9, H.DEV, 0, H.log)  # builds the model, ~no steps
model.load_state_dict(ck["state"])
model.eval()
old = [r for r in H.read(f"{H.FORK}/data/ladder/transfer.jsonl")][:40]
old = [{**r, "body": r["n_lines"] - r["n_prem"]} for r in old]
new = [r for r in H.read(f"{H.DATA}/rl.jsonl") if r["band"] == "7-9"][:40]
train = H.read(f"{H.DATA}/pretrain.jsonl.gz")[2000:2040]
for tag, recs in (("train", train), ("old", old), ("new", new)):
    prompts = [r["prompt"] for r in recs]
    ids = [tok.encode_prompt(p) for p in prompts]
    with H.torch.autocast("cuda", dtype=H.torch.bfloat16):
        outs = H.generate_ids(model, tok, ids, True, 1.0, 512, None)
    texts, nds = [], []
    for o in outs:
        nds.append(tok.decode(o))
        texts.append(tok.last_text)
    ok_items = [(tok.statement(p), t) for p, t, nd in zip(prompts, texts, nds) if t is not None and not nd.startswith("LEANPARSE")]
    res, _, _ = H._overlay.__globals__["sys"].modules["lean_gate"].lean_check(ok_items) if ok_items else ([], 0, 0)
    print(f"== {tag}: {len(recs)} greedy, parse-fail {sum(nd.startswith('LEANPARSE') for nd in nds)}, lean ok {sum(res)}/{len(res)}")
    shown = 0
    for p, t, nd in zip(prompts, texts, nds):
        if shown >= 3:
            break
        shown += 1
        print("PROMPT", p)
        print("  ND  ", nd[:400])
        if t is not None and not nd.startswith("LEANPARSE"):
            src = "set_option linter.unusedVariables false\n" + tok.statement(p) + " " + t + "\n"
            with tempfile.NamedTemporaryFile("w", suffix=".lean", delete=False) as f:
                f.write(src)
            out = subprocess.run([os.path.expanduser("~/.elan/bin/lean"), f.name], capture_output=True, text=True)
            print("  LEAN", (out.stdout + out.stderr)[:600].replace("\n", " | "))
