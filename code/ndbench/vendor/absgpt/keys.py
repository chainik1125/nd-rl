"""Atom-renaming class key (from the take-home generator).""" 
ATOMS = ['P', 'Q', 'R', 'S']


def canon_key(thm):
    """Atom-renaming class: relabel atoms by first appearance in the theorem string."""
    m = {}
    out = []
    for t in thm.split():
        if t in ATOMS:
            if t not in m:
                m[t] = ATOMS[len(m)]
            out.append(m[t])
        else:
            out.append(t)
    return ' '.join(out)


def prompt_key(prompt):
    """Class key of a 'THM ... SEQ ... PRF' prompt."""
    t = prompt.split()
    body = ' '.join(t[1:-1]).replace(' SEQ ', ' |- ').replace('SEQ ', '|- ')
    return canon_key(body)
