"""Round trip of the overlay tokenizer on real records: ND proof -> ids -> decode == the same ND proof, and Lean
accepts the rendered text. Also reports token lengths and name/premise ranges.

    ND_FORK=<fork at 51604b3> python selftest_tok.py FILE.jsonl [N]
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "overlay"), os.environ["ND_FORK"]]
from lean_gate import lean_check  # noqa: E402  (overlay)
from lean_tok import LeanTokenizer  # noqa: E402  (overlay)

tok = LeanTokenizer("lean_seq")
recs = [json.loads(line) for line in open(sys.argv[1])]
random.Random(0).shuffle(recs)
recs = recs[: int(sys.argv[2]) if len(sys.argv) > 2 else 200]
items, bad, lens, names = [], 0, [], []
for r in recs:
    head, body = r["text"].split(" PRF ", 1) if "text" in r else (None, None)
    prompt, proof = r.get("prompt") or head + " PRF", r.get("proof") or body
    ids = tok.encode_proof(proof)
    back = tok.decode(ids)
    if back != proof:
        bad += 1
        if bad <= 3:
            print("ROUNDTRIP MISMATCH\n ", proof, "\n ", back)
    lens.append(len(tok.encode_prompt(prompt)) + len(ids))
    names.append(max(x - tok.ref0 + 1 for x in ids if x >= tok.ref0))
    items.append(tok.lean_source(prompt, proof))
ok, wall, _ = lean_check(items)
lens.sort()
print(f"{len(recs)} records: round-trip mismatches {bad}, Lean accepts {sum(ok)}/{len(ok)} ({wall:.0f}s); "
      f"tokens p50 {lens[len(lens) // 2]} p90 {lens[int(0.9 * len(lens))]} max {lens[-1]}; max names {max(names)}; vocab {tok.vocab_size}")
