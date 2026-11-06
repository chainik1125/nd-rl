"""Probe 1: swap the disjunct order of the top-level case split so the hard (nested-box) branch moves from the first
to the second Or.elim branch (or vice versa). Everything else in the theorem is unchanged. Gold proofs for the swapped
theorems are produced by reordering the two boxes and Lean-checked through the harness gate. Then untied s1 final.pt
is sampled at k=8 on both versions."""
import os, sys, json, random, time
os.environ["ND_ARCH"] = os.environ.get("ND_ARCH", "untied"); os.environ["LEAN_GATE_WORKERS"] = "2"
os.environ["LEAN_GATE_LOG"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lean_gate.jsonl")
sys.path.insert(0, "/private/tmp/claude-501/-Users-robbie-repos-nd-rl/a1d02fb4-ba43-48c4-838b-e227293b6c2a/scratchpad/probe")
import ndmodel as M, torch
torch.set_num_threads(3)
import harness as H
from common import *

CKPT = sys.argv[1] if len(sys.argv) > 1 else f"{E}/resume/untied-ei_s1/final.pt"
OUT = sys.argv[2] if len(sys.argv) > 2 else "probe_swap_results.json"
N_S, N_D = 10, 6


def render(lines):
    out = []
    for ln in lines:
        s = f"N{ln['idx']} " + "| " * ln["depth"] + f"{fmt_formula(ln['formula'])} : {ln['rule']}"
        if ln["refs"]:
            s += " " + " ".join(f"N{r}" for r in ln["refs"])
        out.append(s)
    return " ; ".join(out) + " ; QED"


def top_ore(lines):
    ores = [ln for ln in lines if ln["rule"] == "ORE" and ln["depth"] == 0]
    return ores[-1] if ores else None


def branch_cls(r):
    lines = parse_proof_tokens(r["proof"].split())
    o = top_ore(lines)
    if o is None:
        return None
    j, s1, e1, s2, e2 = o["refs"]
    h = lambda a, b: sum(1 for ln in lines if a <= ln["idx"] <= b and ln["rule"] in ("ORE", "NEGI"))
    h1, h2 = h(s1, e1), h(s2, e2)
    return "none" if not h1 and not h2 else "first" if not h2 else "second" if not h1 else "both"


def swap(r):
    """Return (prompt', proof') with the top-level case split's disjunction reversed, or None."""
    lines = parse_proof_tokens(r["proof"].split())
    by = {ln["idx"]: ln for ln in lines}
    o = top_ore(lines)
    j, s1, e1, s2, e2 = o["refs"]
    dis = by[j]
    assert dis["formula"][0] == "or"
    a_, b_ = dis["formula"][1], dis["formula"][2]
    new_dis = ("or", b_, a_)
    # which premise carries the disjunction?
    if dis["rule"] == "PR":
        old_t, new_t = fmt_formula(dis["formula"]), fmt_formula(new_dis)
        src_line = dis
    elif dis["rule"] == "IMPE" and by[dis["refs"][0]]["rule"] == "PR" and by[dis["refs"][0]]["formula"][2] == dis["formula"]:
        src_line = by[dis["refs"][0]]
        old_t = fmt_formula(src_line["formula"])
        new_t = fmt_formula((src_line["formula"][0], src_line["formula"][1], new_dis))
    else:
        return None
    head, goal = r["prompt"].split(" SEQ ")
    if head.count(old_t) != 1:
        return None
    prompt2 = head.replace(old_t, new_t) + " SEQ " + goal
    # reorder boxes
    pre = [ln for ln in lines if ln["idx"] < s1]
    box1 = [ln for ln in lines if s1 <= ln["idx"] <= e1]
    mid = [ln for ln in lines if e1 < ln["idx"] < s2]
    box2 = [ln for ln in lines if s2 <= ln["idx"] <= e2]
    post = [ln for ln in lines if ln["idx"] > e2]
    if mid:
        return None
    order = pre + box2 + box1 + post
    remap = {ln["idx"]: i + 1 for i, ln in enumerate(order)}
    new = []
    for ln in order:
        d = dict(ln)
        d["idx"] = remap[ln["idx"]]
        d["refs"] = [remap[x] for x in ln["refs"]]
        if ln is o:
            d["refs"] = [remap[j], remap[s2], remap[e2], remap[s1], remap[e1]]
        if ln is src_line:
            d["formula"] = new_dis if ln is dis else (ln["formula"][0], ln["formula"][1], new_dis)
        if ln is dis and ln is not src_line:
            d["formula"] = new_dis
        new.append(d)
    return prompt2, render(new)


def lean_ok(tok, prompt, proof):
    ids = tok.encode_proof(proof)
    nd = tok.decode(ids)
    g = H.gate(tok, [prompt], [nd], [tok.last_text])
    return not g[0].startswith("LEAN")


def main():
    rl = load_rl(); S = solved_sets()
    gold = [r for r in rl if "proof" in r]
    nsolv = lambda n: sum(1 for a in ARMS if n in S[a])
    rng = random.Random(0)
    sel = []
    for fam, lo, hi, n in (("s+s2", 11, 13, N_S), ("d", 9, 10, N_D)):
        recs = [r for r in gold if src_family(r["src"]) in fam.split("+") and lo <= r["body"] <= hi]
        first0 = [r for r in recs if branch_cls(r) == "first" and nsolv(r["name"]) == 0]
        second4 = [r for r in recs if branch_cls(r) == "second" and nsolv(r["name"]) == 4]
        rng.shuffle(first0); rng.shuffle(second4)
        sel += [("first->second", r) for r in first0[:n]] + [("second->first", r) for r in second4[:n]]
    model, tok = M.load(CKPT)
    if hasattr(model, "n_loops") and os.environ.get("ND_LOOPS"):
        model.n_loops = int(os.environ["ND_LOOPS"])
    probes = []
    for tag, r in sel:
        sw = swap(r)
        if sw is None:
            print("skip", r["name"]); continue
        p2, pr2 = sw
        assert branch_cls({"proof": pr2}) == tag.split("->")[1], (r["name"], branch_cls({"proof": pr2}))
        ok = lean_ok(tok, p2, pr2)
        print(f"{r['name']} {tag} body={r['body']} swapped gold lean_ok={ok}")
        if not ok:
            continue
        probes.append(dict(name=r["name"], tag=tag, fam=src_family(r["src"]), body=r["body"], orig_prompt=r["prompt"], orig_proof=r["proof"], swap_prompt=p2, swap_proof=pr2))
    recs = []
    for p in probes:
        recs.append(dict(name=p["name"] + "/orig", prompt=p["orig_prompt"], body=p["body"]))
        recs.append(dict(name=p["name"] + "/swap", prompt=p["swap_prompt"], body=p["body"]))
    print("sampling", len(recs), "records x 8")
    results = {}
    t0 = time.time()
    for i in range(0, len(recs), 20):
        chunk = recs[i:i + 20]
        out = M.sample(model, tok, chunk, k=8, seed=1)
        for rec, samples in zip(chunk, out):
            results[rec["name"]] = dict(n_ok=sum(1 for _, ok, _ in samples if ok), tags=[nd if nd.startswith("LEAN") else "OK" for nd, ok, _ in samples],
                                        raw=[raw for _, _, raw in samples][:2])
            print(rec["name"], results[rec["name"]]["n_ok"], f"{time.time()-t0:.0f}s", flush=True)
        json.dump(dict(probes=probes, results=results), open(OUT, "w"), indent=1)
    print("\n### summary")
    for tag in ("first->second", "second->first"):
        for fam in ("s", "s2", "d"):
            ps = [p for p in probes if p["tag"] == tag and p["fam"] == fam]
            if not ps: continue
            o = sum(results[p["name"] + "/orig"]["n_ok"] for p in ps); s = sum(results[p["name"] + "/swap"]["n_ok"] for p in ps)
            os_ = sum(results[p["name"] + "/orig"]["n_ok"] > 0 for p in ps); ss = sum(results[p["name"] + "/swap"]["n_ok"] > 0 for p in ps)
            print(f"{tag} fam={fam} n={len(ps)}: orig {o}/{8*len(ps)} samples ok, {os_}/{len(ps)} targets solved | swapped {s}/{8*len(ps)} ok, {ss}/{len(ps)} solved")


if __name__ == "__main__":
    main()
