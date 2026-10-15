# ruff: noqa: E501  (research scripts: long result lines, bare open(), short loop names)
"""close-run role labelling shared by the interpretation scripts (from interpret_top.py)."""


def roles(s, n_p):
    """per position: (role, D_start, r) inside close runs; role in exact / name / close / semi / pop"""
    out = {}
    T = len(s)
    depth = 0
    stack = []
    dep = [0] * T
    for i in range(T):
        dep[i] = depth
        if i >= n_p and s[i] == "(":
            box = i + 2 < T and s[i + 1] == "fun" and s[i + 2] == "("
            stack.append(box)
            depth += box
        elif i >= n_p and s[i] == ")" and stack and stack.pop():
            depth -= 1
    i = n_p
    while i < T - 2:
        if s[i] == "exact" and s[i + 2] == ")" and dep[i] >= 1:
            # start of a close run: walk forward over `exact n ) ;` groups that close boxes
            D0 = dep[i]
            r = 0
            j = i
            while j + 2 < T and s[j] == "exact" and s[j + 2] == ")" and dep[j] >= 1:
                r += 1
                out[j] = ("exact", D0, r)
                out[j + 1] = ("name", D0, r)
                out[j + 2] = ("close", D0, r)
                if j + 4 < T and s[j + 3] == ";" and s[j + 4] == "exact":
                    out[j + 3] = ("semi", D0, r)
                    out[j + 4] = ("pop", D0, r)
                    j += 4
                    if not (j + 2 < T and s[j + 2] == ")"):
                        break
                else:
                    break
            i = j + 1
        else:
            i += 1
    return out
