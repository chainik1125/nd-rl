"""Build data/textbook_problem/release_2026-10-02/: every problem set with a verified reference proof,
re-audited under the curry-aware overlap rule (Charles, 2026-09-30), plus Lean renderings and the
aggregate model baselines of 2026-10-02.

    cd code && uv run python textbook_problem/release_1002.py

Sets (one file each; a problem appears in exactly one file):
  candidate_v0, candidate_v1  copies of the frozen sets, re-cut under the curry rule
  hand_proved_14              the formerly unsolved 14 with hand-written reference proofs
  batch3                      Pelletier, forall x R3, Oxford: the 17 search-proved ones (the 4 Pelletier
                              hand-proved problems live in hand_proved_14)
  derived                     the 21-nuance derived set (232 rows); rows keep their base's class by design,
                              so within-release curried twins are recorded, not removed; sealed twins are removed
Removed rows go to removed.jsonl with the reason, except sealed matches, which are counted only.
The sealed textbook 72 are never copied here.
"""

import collections
import datetime as dt
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "../ws3")]
import ndpath  # noqa: E402,F401
import lean  # noqa: E402
import prove  # noqa: E402
import transcribe as T  # noqa: E402
from data import canon_key, load_pool, load_textbook72, thm_of_prompt  # noqa: E402
from derive import KEPT  # noqa: E402
from nd_data import canonical_theorem  # noqa: E402
from strict_verify import safe_verify  # noqa: E402

L = prove.L


class BR:
    """Curry-aware overlap helpers, the same definitions as build_release.py (the v2 bundle)."""

    @staticmethod
    def curry_normal(prompt):
        prem, goal = L.parse_prompt(prompt)
        prem = list(prem)
        while goal[0] == "imp":
            prem.append(goal[1])
            goal = goal[2]
        p2, g2, _ = T.rename(prem, goal)
        return T.make_prompt(p2, g2)

    @staticmethod
    def keys(prompt):
        out = set()
        for p in (prompt, BR.curry_normal(prompt)):
            out |= {("ct", canonical_theorem(p)), ("ck", canon_key(thm_of_prompt(p)))}
        return out

    @staticmethod
    def index(prompts):
        s = set()
        for p in prompts:
            s |= BR.keys(p)
        return s

    @staticmethod
    def tag(r, proof, method, derived):
        """Fill the proof-dependent fields the way prove.py does."""
        prem, goal = L.parse_prompt(r["prompt"])
        ok, reason, n = safe_verify(r["prompt"], proof)
        r["reference_proof"] = proof
        r["reference_lines"] = r["min_lines"] = n
        r["min_lines_method"] = method
        r["strict_verify"] = {"ok": bool(ok), "reason": reason}
        r["requires_derived_ore"] = derived
        r["proof_features"] = prove.proof_features(proof)
        r["length_bin"] = prove.length_bin(r["reference_lines"])
        r["family"], r["tags"] = prove.families({**r, "requires_derived_ore": bool(derived)}, goal)
        laws = prove.named_laws(prem, goal)
        r["schema"] = laws
        r["schema_in_rl_pool"] = any(x.split(" (")[0] in prove.SCHEMATA for x in laws)
        return ok
DATA = os.path.join(HERE, "../../data/textbook_problem")
OUT = os.path.join(DATA, "release_2026-10-02")
EX = os.path.join(HERE, "../../experiments/textbook-problem")
TODAY = "2026-10-02"


def jl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def wjl(p, rows):
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    os.makedirs(os.path.join(OUT, "lean"), exist_ok=True)
    # ---------------- sources
    v0 = jl(f"{DATA}/candidate_v0/problems.jsonl")
    v1 = jl(f"{DATA}/candidate_v1/problems.jsonl")
    deferred = {r["name"]: r for r in jl(f"{DATA}/candidate_v1/deferred_no_proof.jsonl")}
    hand = []
    for h in jl(f"{DATA}/candidates_v3/hand_proofs.jsonl"):
        r = dict(deferred[h["name"]])
        r.pop("deferred_reason", None)
        feats = prove.proof_features(h["reference_proof"])
        BR.tag(r, h["reference_proof"], "hand proof (not minimal; a search bound of 40 lines found none)", "ORE" in feats["rule_set"])
        r["role"] = "textbook_candidate"
        hand.append(r)
    hand_names = {r["name"] for r in hand}
    batch3 = [r for r in jl(f"{DATA}/candidates_v3/batch3_proved.jsonl") if r["name"] not in hand_names and r.get("reference_proof")]
    for r in batch3:
        r["role"] = "textbook_candidate"
    derived = []
    for nu in KEPT:
        for r in jl(f"{DATA}/candidates_derived/{nu}.jsonl"):
            derived.append(r)
    # ---------------- audit (curry-aware), priority order
    sealed = BR.index([r["prompt"] for r in load_textbook72()] + [r["prompt"] for r in load_pool("test_short_prompts") + load_pool("test_long_prompts")])
    named = {n: BR.index([r["prompt"] for r in load_pool(n)]) for n in ("validation_36", "heldout", "transfer", "rl_targets", "train")}
    kept_index = {}  # key -> (set, name)
    removed, sealed_counts = [], collections.Counter()
    out = {}

    def audit(setname, rows, keep_twins_within=False):
        kept = []
        for r in rows:
            ks = BR.keys(r["prompt"])
            if ks & sealed:
                sealed_counts[setname] += 1
                continue
            hits = [n for n, idx in named.items() if ks & idx]
            if hits:
                removed.append(dict(set=setname, name=r["name"], source=r.get("source", {}).get("id") or r.get("base", {}).get("source"),
                                    original=r.get("original"), prompt=r["prompt"], reason=f"curried twin of an item in {', '.join(hits)}"))
                continue
            twin = next((kept_index[k] for k in ks if k in kept_index), None)
            if twin:
                if keep_twins_within:
                    r["curried_twin_of"] = f"{twin[0]} {twin[1]}"
                else:
                    removed.append(dict(set=setname, name=r["name"], source=r.get("source", {}).get("id"), original=r.get("original"),
                                        prompt=r["prompt"], reason=f"curried twin of {twin[0]} {twin[1]} (kept)"))
                    continue
            for k in ks:
                kept_index.setdefault(k, (setname, r["name"]))
            kept.append(r)
        out[setname] = kept
        return kept

    audit("candidate_v0", v0)
    audit("candidate_v1", v1)
    audit("hand_proved_14", hand)
    audit("batch3", batch3)
    audit("derived", derived, keep_twins_within=True)
    # ---------------- verify every reference proof once more with the pinned verifier
    bad = 0
    for s, rows in out.items():
        for r in rows:
            ok = prove.safe_verify(r["prompt"], r["reference_proof"])[0]
            bad += not ok
    assert bad == 0, f"{bad} reference proofs failed strict_verify"
    # ---------------- write files + Lean
    manifest = dict(built=TODAY, builder="code/textbook_problem/release_1002.py",
                    audit="curry-aware renaming class (canonical_theorem / canon_key of the prompt and of its curried normal form) "
                          "against the sealed 72, the test files, validation_36, heldout, transfer, rl_targets, train and the earlier files of this release",
                    sealed_matches_removed_counts=dict(sealed_counts), files={})
    lean_ok_total = 0
    for s, rows in out.items():
        fn = os.path.join(OUT, f"{s}.jsonl")
        wjl(fn, rows)
        texts, idx = [], []
        for i, r in enumerate(rows):
            try:
                texts.append(lean.lean_theorem(r["prompt"], r["reference_proof"]))
                idx.append(i)
            except ValueError:  # lean_seq names n1..n64: proofs longer than 64 lines cannot be rendered
                pass
        oks_r = lean.check([(rows[i]["prompt"], rows[i]["reference_proof"]) for i in idx]) if idx else []
        oks = [None] * len(rows)
        for i, ok in zip(idx, oks_r):
            oks[i] = ok
        lean_ok_total += sum(1 for o in oks if o)
        with open(os.path.join(OUT, "lean", f"{s}.lean"), "w") as f:
            for r, ok in zip(rows, oks):
                if ok is None:
                    f.write(f"-- {r['name']}  (reference proof longer than 64 lines: lean_seq cannot render it; statement only)\n{lean.statement(r['prompt'])} sorry\n\n")
                else:
                    f.write(f"-- {r['name']}{'' if ok else '  (NOT accepted by Lean)'}\n{lean.lean_theorem(r['prompt'], r['reference_proof'])}\n\n")
        manifest["files"][f"{s}.jsonl"] = dict(rows=len(rows), sha256=sha(fn), lean_accepted=sum(1 for o in oks if o), lean_not_rendered=sum(1 for o in oks if o is None))
        manifest["files"][f"lean/{s}.lean"] = dict(sha256=sha(os.path.join(OUT, "lean", f"{s}.lean")))
        print(f"{s:16s} {len(rows):4d} rows, Lean {sum(1 for o in oks if o)}/{len(rows)} (not rendered {sum(1 for o in oks if o is None)})")
    wjl(os.path.join(OUT, "removed.jsonl"), removed)
    manifest["files"]["removed.jsonl"] = dict(rows=len(removed), sha256=sha(os.path.join(OUT, "removed.jsonl")))
    print("removed:", collections.Counter(r["set"] for r in removed), "| sealed (counted only):", dict(sealed_counts))
    print("derived rows with a within-release curried twin:", sum(1 for r in out["derived"] if r.get("curried_twin_of")))
    json.dump(manifest, open(os.path.join(OUT, "MANIFEST.json"), "w"), indent=1)
    # ---------------- baselines
    def metrics(path):
        return {m["set"]: m for m in jl(path)}
    pt = {k: metrics(f"{EX}/eval_scale_pretrain/run_{k}_s0/metrics.jsonl") for k in ("d384", "d768")}
    rl = collections.defaultdict(dict)
    for m in jl(f"{EX}/eval_dan_best_state/run_best12_r8/metrics.jsonl"):
        rl[m["set"]][m["seed"]] = m
    names = {"v0": "candidate_v0", "v1": "candidate_v1", "hand14": "hand_proved_14", "batch3": "batch3 (21 incl. the 4 hand-proved)", "lean_v2": "Lean-sourced (not in this release)",
             "derived": "derived (18 feedable nuances, pretrain)", "derived_plain": "derived (13 plain nuances, RL)", "textbook72": "textbook 72 (sealed, aggregate)"}
    lines = ["# Baselines (aggregate), 2026-10-02\n",
             "Scored on the sets as they stood before the curry-aware re-cut (candidate_v0 104, candidate_v1 80); k = 32 samples per problem, temperature 1.0; every proof judged by the pinned nd_verify through strict_verify. "
             "avg = mean over problems of the fraction of samples accepted; solved = problems with at least one accepted sample. One seed per pretrain, two seeds for the RL model. "
             "Details, per-problem rows and the leak audit: `experiments/textbook-problem/eval_scale_pretrain/` and `eval_dan_best_state/`.\n",
             "| set | n | pretrain d384 avg / solved | pretrain d768 avg / solved | RL best-state r8 s0 avg / solved | RL s1 avg / solved |", "|---|---|---|---|---|---|"]
    for key in ["v0", "v1", "hand14", "batch3", "lean_v2", "derived", "derived_plain", "textbook72"]:
        a, b = pt["d384"].get(key), pt["d768"].get(key)
        r0, r1 = rl.get(key, {}).get(0), rl.get(key, {}).get(1)
        if key == "derived_plain":
            a = b = None
        if key == "derived":
            r0 = r1 = None
        cell = lambda m: f"{m['avg_solve_rate']:.3f} / {m['any_of_k']}" if m else "—"
        n = (a or b or r0 or r1)["n"]
        lines.append(f"| {names[key]} | {n} | {cell(a)} | {cell(b)} | {cell(r0)} | {cell(r1)} |")
    ck = {}
    for k in ("d384", "d768"):
        p = os.path.expanduser(f"~/.cache/huggingface/hub/models--robbiethompson2018--nd-rl-checkpoints/snapshots/b55351ac598af600e90a76ef8310e77971125afb/runs/scale-{k}-p1200_s0/pretrain.pt")
        ck[f"scale-{k}-p1200_s0/pretrain.pt"] = sha(p) if os.path.exists(p) else "not cached"
    for l in open(f"{EX}/eval_dan_best_state/run_best12_r8/ckpt_sha256.txt"):
        h, p = l.split()
        ck["best-state/ckpts/bs/ladder/" + os.path.basename(p)] = h
    lines += ["", "Checkpoints (sha256):", ""] + [f"- `{k}`: {v}" for k, v in ck.items()] + [
        "", "Pretrains: Robbie's `scale-*-p1200` runs of 2026-09-30 (HF `robbiethompson2018/nd-rl-checkpoints`), lean_seq tokens, 400-token cap. "
        "RL: Dan's best-state cap-12 T1 ladder round 8 (HF bucket `dan-pandori/nd-rl`, done 2026-10-01), proof-state environment with Lean in the loop, 48 steps; "
        "the derived prefix nuances and the three marker-word nuances could not be fed to it. Overlap between every set and the RL model's training pools is at most 3 problems per set (`eval_dan_best_state/run_best12_r8/leak_audit.md`)."]
    open(os.path.join(OUT, "baselines.md"), "w").write("\n".join(lines) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
