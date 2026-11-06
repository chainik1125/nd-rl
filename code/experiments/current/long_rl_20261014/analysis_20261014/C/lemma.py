import annot
def ore_kinds(proof):
    """ORE line idx -> 'GOAL' if its conclusion is the goal of the innermost enclosing scope, else 'LEMMA'"""
    lines = annot.parse_proof_tokens(proof.split())
    spans = []  # (as idx, last idx)
    for l in lines:
        if l['rule'] in ('IMPI', 'NEGI'): spans.append((l['refs'][0], l['refs'][1]))
        if l['rule'] == 'ORE': spans += [(l['refs'][1], l['refs'][2]), (l['refs'][3], l['refs'][4])]
    by = {l['idx']: l for l in lines}
    out = {}
    for l in lines:
        if l['rule'] != 'ORE': continue
        enc = [s for s in spans if s[0] < l['idx'] <= s[1]]
        goal_idx = max(enc)[1] if enc else lines[-1]['idx']
        out[l['idx']] = 'GOAL' if by[goal_idx]['formula'] == l['formula'] else 'LEMMA'
    return out
