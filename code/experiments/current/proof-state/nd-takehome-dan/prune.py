"""Dependency-pruned length of a verified proof: number of lines the conclusion transitively
cites (box citations pull in the AS line and the box's last line). Detects padding
(reiterations, dead branches). Returns the written length if the proof does not parse."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nd_verify.verify import parse_proof_tokens, ParseError


def pruned_length(prompt, body):
    toks = body.split()
    try:
        lines = parse_proof_tokens(toks)
    except ParseError:
        return sum(1 for t in toks if t == ';')
    if not lines:
        return 0
    byidx = {ln['idx']: ln for ln in lines}
    keep = set()
    stack = [lines[-1]['idx']]
    while stack:
        i = stack.pop()
        if i in keep or i not in byidx:
            continue
        keep.add(i)
        stack.extend(byidx[i]['refs'])
    return len(keep)
