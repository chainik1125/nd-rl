"""Small helpers shared by the reviewer scripts (same definitions as in recount.py)."""
import gzip
import json
import os

ATOMS = ("P", "Q", "R", "S")


def read_jsonl(path):
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        path = path + ".gz"
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return [json.loads(l) for l in f if l.strip()]


def split_prompt(prompt):
    t = prompt.split()
    assert t[0] == "THM" and t[-1] == "PRF", prompt
    i = t.index("SEQ")
    prems, cur = [], []
    for tok in t[1:i]:
        if tok == ",":
            prems.append(" ".join(cur))
            cur = []
        else:
            cur.append(tok)
    if cur:
        prems.append(" ".join(cur))
    return prems, " ".join(t[i + 1 : -1])


def thm_of_prompt(prompt):
    prems, con = split_prompt(prompt)
    return (" , ".join(prems) + " |- " + con) if prems else ("|- " + con)


def canon_key(prompt):
    """The fork's gen.canon_key: relabel atoms by first appearance on 'prem |- concl'."""
    m, out = {}, []
    for tok in thm_of_prompt(prompt).split():
        if tok in ATOMS:
            if tok not in m:
                m[tok] = ATOMS[len(m)]
            out.append(m[tok])
        else:
            out.append(tok)
    return " ".join(out)


def is_trivial(prompt):
    prems, con = split_prompt(prompt)
    return con in prems or "F" in prems


def renumber(completion):
    """Proof text with line labels N<k> renumbered from 1 (for de-duplication)."""
    toks = completion.split()
    m = {}
    out = []
    for t in toks:
        if len(t) > 1 and t[0] == "N" and t[1:].isdigit():
            if t not in m:
                m[t] = f"N{len(m) + 1}"
            out.append(m[t])
        else:
            out.append(t)
    return " ".join(out)
