# ruff: noqa
# Scratch analysis script written by a research agent on 2026-11-18; not reviewed.
import pickle, sys
from ndlib import *

o = pickle.load(open("tab.pkl", "rb"))


def short(proof):
    p = Proof(proof)
    return " ; ".join(p.L[i].txt for i in p.order if p.L[i].live)


for m in sys.argv[1:]:
    print("=====", m)
    for corp in ("pre", "rl", "ctl"):
        for body, name, proof in o["ex"].get(m, {}).get(corp, [])[:2]:
            print(f"  [{corp} {name} body={body}] {short(proof)}")
