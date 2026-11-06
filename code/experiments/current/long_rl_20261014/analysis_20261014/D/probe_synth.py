"""Probe 2: synthetic theorem families that vary one feature. Each template is instantiated with 10 random atom
relabelings (+ premise shuffles, + optional distractor premises), its gold proof is Lean-checked through the harness
gate, then the model is sampled at k=8.

Template DSL: premises/goal are formula strings over placeholder atoms A..H; body lines are (depth, formula, rule, refs)
with refs 'p<i>' = i-th template premise (1-based) and 'b<j>' = j-th body line (1-based)."""
import os, sys, json, random, re, time
os.environ["ND_ARCH"] = os.environ.get("ND_ARCH", "untied"); os.environ["LEAN_GATE_WORKERS"] = "2"
os.environ["LEAN_GATE_LOG"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lean_gate.jsonl")
sys.path.insert(0, "/private/tmp/claude-501/-Users-robbie-repos-nd-rl/a1d02fb4-ba43-48c4-838b-e227293b6c2a/scratchpad/probe")
import ndmodel as M, torch
torch.set_num_threads(3)
import harness as H
from common import *

CKPT = sys.argv[1] if len(sys.argv) > 1 else f"{E}/resume/untied-ei_s1/final.pt"
OUT = sys.argv[2] if len(sys.argv) > 2 else "probe_synth_results.json"
ATOMS = list("PQRSTUWXYZ")

T = {}
# --- T1: an IMPI box inside the FIRST vs SECOND branch of the top-level case split (10 body lines, 5 premises)
T["impi_first"] = dict(
    prem=["( A v B )", "( A > ( C > E ) )", "( E > D )", "( ( C > D ) > G )", "( B > G )"], goal="G",
    body=[(1, "A", "AS", []), (2, "C", "AS", []), (2, "( C > E )", "IMPE", ["p2", "b1"]), (2, "E", "IMPE", ["b3", "b2"]),
          (2, "D", "IMPE", ["p3", "b4"]), (1, "( C > D )", "IMPI", ["b2", "b5"]), (1, "G", "IMPE", ["p4", "b6"]),
          (1, "B", "AS", []), (1, "G", "IMPE", ["p5", "b8"]), (0, "G", "ORE", ["p1", "b1", "b7", "b8", "b9"])])
T["impi_second"] = dict(
    prem=["( B v A )", "( A > ( C > E ) )", "( E > D )", "( ( C > D ) > G )", "( B > G )"], goal="G",
    body=[(1, "B", "AS", []), (1, "G", "IMPE", ["p5", "b1"]),
          (1, "A", "AS", []), (2, "C", "AS", []), (2, "( C > E )", "IMPE", ["p2", "b3"]), (2, "E", "IMPE", ["b5", "b4"]),
          (2, "D", "IMPE", ["p3", "b6"]), (1, "( C > D )", "IMPI", ["b4", "b7"]), (1, "G", "IMPE", ["p4", "b8"]),
          (0, "G", "ORE", ["p1", "b1", "b2", "b3", "b9"])])
# --- T2: NEGI box in the second branch; its conclusion IS the goal (direct) vs an intermediate lemma (inter). 9 lines, 4 premises
T["negi_direct"] = dict(
    prem=["( A v B )", "( A > D )", "( D > ( ~ C ) )", "( ~ ( B & C ) )"], goal="( ~ C )",
    body=[(1, "A", "AS", []), (1, "D", "IMPE", ["p2", "b1"]), (1, "( ~ C )", "IMPE", ["p3", "b2"]),
          (1, "B", "AS", []), (2, "C", "AS", []), (2, "( B & C )", "ANDI", ["b4", "b5"]), (2, "F", "NEGE", ["b6", "p4"]),
          (1, "( ~ C )", "NEGI", ["b5", "b7"]), (0, "( ~ C )", "ORE", ["p1", "b1", "b3", "b4", "b8"])])
T["negi_inter"] = dict(
    prem=["( A v B )", "( A > ( ~ G ) )", "( ~ ( B & C ) )", "( ( ~ C ) > ( ~ G ) )"], goal="( ~ G )",
    body=[(1, "A", "AS", []), (1, "( ~ G )", "IMPE", ["p2", "b1"]),
          (1, "B", "AS", []), (2, "C", "AS", []), (2, "( B & C )", "ANDI", ["b3", "b4"]), (2, "F", "NEGE", ["b5", "p3"]),
          (1, "( ~ C )", "NEGI", ["b4", "b6"]), (1, "( ~ G )", "IMPE", ["p4", "b7"]), (0, "( ~ G )", "ORE", ["p1", "b1", "b2", "b3", "b8"])])
# --- T3: classical schemas from old dev: excluded middle (9 lines, nested NEGI) and Peirce (9 lines), vs a constructive
#     control with a nested NEGI of the same depth: ~(A v B), (C > A) |- ~C ... (5 lines) padded by a chain to 9 lines
T["excluded_middle"] = dict(
    prem=[], goal="( A v ( ~ A ) )",
    body=[(1, "( ~ ( A v ( ~ A ) ) )", "AS", []), (2, "A", "AS", []), (2, "( A v ( ~ A ) )", "ORI1", ["b2"]), (2, "F", "NEGE", ["b3", "b1"]),
          (1, "( ~ A )", "NEGI", ["b2", "b4"]), (1, "( A v ( ~ A ) )", "ORI2", ["b5"]), (1, "F", "NEGE", ["b6", "b1"]),
          (0, "( ~ ( ~ ( A v ( ~ A ) ) ) )", "NEGI", ["b1", "b7"]), (0, "( A v ( ~ A ) )", "DN", ["b8"])])
T["peirce"] = dict(
    prem=["( ( A > B ) > A )"], goal="A",
    body=[(1, "( ~ A )", "AS", []), (2, "A", "AS", []), (2, "F", "NEGE", ["b2", "b1"]), (2, "B", "BOTE", ["b3"]),
          (1, "( A > B )", "IMPI", ["b2", "b4"]), (1, "A", "IMPE", ["p1", "b5"]), (1, "F", "NEGE", ["b6", "b1"]),
          (0, "( ~ ( ~ A ) )", "NEGI", ["b1", "b7"]), (0, "A", "DN", ["b8"])])
T["nested_negi_constructive"] = dict(  # same nesting (NEGI inside NEGI) and length, no reductio on the goal
    prem=["( ~ ( A v B ) )", "( C > ( D > A ) )", "( E > D )", "( C > E )"], goal="( ~ C )",
    body=[(1, "C", "AS", []), (1, "( D > A )", "IMPE", ["p2", "b1"]), (1, "E", "IMPE", ["p4", "b1"]), (1, "D", "IMPE", ["p3", "b3"]),
          (1, "A", "IMPE", ["b2", "b4"]), (1, "( A v B )", "ORI1", ["b5"]), (1, "F", "NEGE", ["b6", "p1"]),
          (0, "( ~ C )", "NEGI", ["b1", "b7"])])
# --- T4: distractor sensitivity: impi_second with 4 distractor premises
T["impi_second_distract4"] = dict(T["impi_second"], distract=4)
T["negi_direct_distract4"] = dict(T["negi_direct"], distract=4)
T["excluded_middle_distract3"] = dict(T["excluded_middle"], distract=3)


def instantiate(t, rng):
    used = sorted(set(re.findall(r"\b[A-EGH]\b", " ".join(t["prem"] + [t["goal"]] + [b[1] for b in t["body"]]))))
    nd = t.get("distract", 0)
    atoms = rng.sample(ATOMS, len(used) + nd)
    sub = dict(zip(used, atoms[:len(used)]))
    extra = atoms[len(used):]
    ren = lambda s: " ".join(sub.get(x, x) for x in s.split())
    prem = [ren(p) for p in t["prem"]]
    dis = [f"( {extra[i]} > {extra[(i+1) % nd]} )" for i in range(nd)]  # a cycle over fresh atoms: no shortcut to the goal
    allp = [("p", i + 1, p) for i, p in enumerate(prem)] + [("d", i, d) for i, d in enumerate(dis)]
    rng.shuffle(allp)
    pos = {f"p{i}": k + 1 for k, (kind, i, _) in enumerate(allp) if kind == "p"}
    n_prem = len(allp)
    lines = [f"N{k+1} {f} : PR" for k, (_, _, f) in enumerate(allp)]
    for j, (depth, f, rule, refs) in enumerate(t["body"]):
        idx = n_prem + j + 1
        rr = [pos[x] if x.startswith("p") else n_prem + int(x[1:]) for x in refs]
        lines.append(f"N{idx} " + "| " * depth + f"{ren(f)} : {rule}" + (" " + " ".join(f"N{x}" for x in rr) if rr else ""))
    prompt = "THM " + (" , ".join(f for _, _, f in allp) + " " if allp else "") + "SEQ " + ren(t["goal"]) + " PRF"
    return prompt, " ; ".join(lines) + " ; QED", len(t["body"])


def lean_ok(tok, prompt, proof):
    ids = tok.encode_proof(proof); nd = tok.decode(ids)
    return not H.gate(tok, [prompt], [nd], [tok.last_text])[0].startswith("LEAN")


def main():
    model, tok = M.load(CKPT)
    if hasattr(model, "n_loops") and os.environ.get("ND_LOOPS"):
        model.n_loops = int(os.environ["ND_LOOPS"])
    rng = random.Random(7)
    recs, meta = [], {}
    for name, t in T.items():
        for i in range(10):
            prompt, proof, body = instantiate(t, rng)
            ok = lean_ok(tok, prompt, proof)
            if not ok:
                print("GOLD REJECTED", name, prompt, proof); continue
            recs.append(dict(name=f"{name}/{i}", prompt=prompt, body=body)); meta[f"{name}/{i}"] = dict(proof=proof)
    print("sampling", len(recs), "records x 8", flush=True)
    results = {}; t0 = time.time()
    for i in range(0, len(recs), 30):
        chunk = recs[i:i + 30]
        for rec, samples in zip(chunk, M.sample(model, tok, chunk, k=8, seed=1)):
            results[rec["name"]] = dict(prompt=rec["prompt"], gold=meta[rec["name"]]["proof"], n_ok=sum(ok for _, ok, _ in samples),
                                        tags=[nd if nd.startswith("LEAN") else "OK" for nd, ok, _ in samples], raw=[raw for _, _, raw in samples][:3])
        print(f"{i+len(chunk)} done {time.time()-t0:.0f}s", flush=True)
        json.dump(results, open(OUT, "w"), indent=1)
    print("\n### summary (untied s1 unless stated): template -> samples ok /80, instances solved /10")
    for name in T:
        rs = [results[k] for k in results if k.startswith(name + "/")]
        print(f"{name:28s} ok {sum(r['n_ok'] for r in rs):3d}/{8*len(rs)}  solved {sum(r['n_ok']>0 for r in rs)}/{len(rs)}")


if __name__ == "__main__":
    main()
