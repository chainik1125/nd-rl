"""Pretty-print a proof from its token text. Usage:
   python -m exp.show_proof "<full text THM ... QED>"
or pipe jsonl records with --jsonl path --name <target name>.
"""
import argparse
import json
import sys


def render(text):
    toks = text.split()
    i = toks.index('PRF') + 1
    head = ' '.join(toks[:i])
    out = [head]
    line = []
    for t in toks[i:]:
        if t == ';':
            # line = [N<i>, |*, formula..., :, rule, refs...]
            idx = line[0]
            d = 0
            j = 1
            while j < len(line) and line[j] == '|':
                d += 1
                j += 1
            k = line.index(':')
            formula = ' '.join(line[j:k])
            rule = line[k + 1]
            refs = ' '.join(line[k + 2:])
            out.append(f'{idx:>4} {"  " * d}{formula:<44} {rule} {refs}')
            line = []
        elif t == 'QED':
            break
        else:
            line.append(t)
    return '\n'.join(out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('text', nargs='?')
    ap.add_argument('--jsonl')
    ap.add_argument('--name')
    ap.add_argument('--thm')
    args = ap.parse_args()
    if args.text:
        print(render(args.text))
    else:
        for line in open(args.jsonl):
            if not line.strip():
                continue
            r = json.loads(line)
            if args.name and r.get('name') != args.name:
                continue
            if args.thm and r.get('thm') != args.thm:
                continue
            if r.get('best_text'):
                print(f"== {r.get('name') or r.get('thm')} (len {r.get('best_len')}) ==")
                print(render(r['best_text']))
                print()
