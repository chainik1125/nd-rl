"""Build data/pretrain_data/release_2026-10-12_textbook72_distract2/: the distracted-72 experiment as a folder others
can read and reproduce without the sealed problems ever entering git.

    cd code && uv run python pretrain_data/textbook72_distract_release.py

Writes, next to the hand-written README.md and apply_distractors.py:
  distractors.jsonl   per sealed problem: the two added premises and their positions (no original content)
  examples.jsonl      the same construction on four NON-sealed textbook exercises (candidate_v0), for the README
  metrics/            copies of the four models' metrics.jsonl plus summary.json (the headline table, recomputed)
  MANIFEST.json       sha256 of the inputs, the canonical distracted rows, the checkpoints; counts; protocol
Reads the gitignored docs/textbook-problem/textbook72/ (distracted set, Robbie's sample dumps) and the sealed 72, and
prints aggregates only.
"""
from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
import os
import random
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "textbook_problem")]
import pool_common as PC  # noqa: E402
import textbook72_distract as TD  # noqa: E402  (distract_one, load_textbook72, SP, fstr)
from strict_verify import safe_verify  # noqa: E402

REL = os.path.join(PC.DATA, "release_2026-10-12_textbook72_distract2")
T72 = os.path.join(PC.ROOT, "docs", "textbook-problem", "textbook72")
EXP = os.path.join(PC.ROOT, "experiments", "pretrain-data", "textbook72_distract2")
CAND = os.path.join(PC.ROOT, "data", "textbook_problem", "release_2026-09-30_v2", "candidate_v0.jsonl")
EXAMPLE_NAMES = ["textbook_093d53538c2072386890", "textbook_0557545ab024a85bd5ed", "textbook_17cf6bcfdfe010a1d9b2",
                 "textbook_26c5501ec4eb97272dc1"]  # 0 / 1 / 2 / 3 premises, forall x: Calgary
ROBBIE = {"robbie_d384": "scale-d384-p1200_s0", "robbie_d768": "scale-d768-p1200_s0"}
DAN = ["stage1_best12_s0_b1200", "la_T1_best12_s0_r8"]

spec = importlib.util.spec_from_file_location("apply_distractors", os.path.join(REL, "apply_distractors.py"))
AD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AD)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def distractor_rows(dist, sealed):
    """(name, slice, distractors, positions) for every distracted problem; checks the pure-string rebuild."""
    out = []
    for r in dist:
        base_prem, goal = AD.split_prompt(r["base_prompt"])
        prem, goal2 = AD.split_prompt(r["prompt"])
        assert goal == goal2 and sealed[r["name"]]["prompt"] == r["base_prompt"]
        positions = [i for i, p in enumerate(prem) if p not in base_prem]
        ds = [prem[i] for i in positions]
        assert len(ds) == 2 == r["n_distract"] and [p for p in prem if p in base_prem] == base_prem
        assert AD.apply(r["base_prompt"], ds, positions) == r["prompt"]
        out.append(dict(name=r["name"], slice=r["slice"], distractors=ds, positions=positions))
    return out


def examples():
    cand = {r["name"]: r for r in jl(CAND)}
    out = []
    for i, name in enumerate(EXAMPLE_NAMES):
        r = cand[name]
        e = TD.distract_one(random.Random(i), r["prompt"], r.get("reference_lines"))
        assert e is not None, name
        ref_ok = safe_verify(e["prompt"], r["reference_proof"])[0]
        out.append(dict(name=name, source=dict(book=r["source"]["book"], where=r["source"]["where"], licence=r["source"]["licence"]),
                        original_prompt=r["prompt"], distractors=e["distractors"], positions=e["positions"], distracted_prompt=e["prompt"],
                        n_prem_orig=e["n_prem_orig"], n_prem=e["n_prem"], reference_lines=r["reference_lines"], min_lines=e["min_lines"],
                        min_status=e["min_status"], shortcut_check=e["shortcut_check"], draws=e["draws"],
                        original_reference_proof_verifies_on_distracted=ref_ok, reference_proof=r["reference_proof"]))
    return out


def robbie_bins(label, run, dist):
    """Robbie's by-bin counts from the gitignored sample dumps (solved = solve_rate > 0 at k 32)."""
    o = {r["name"]: r["solve_rate"] for r in jl(os.path.join(T72, f"samples_{run}_pretrain_k32.jsonl"))}
    d = {r["name"]: r["solve_rate"] for r in jl(os.path.join(T72, f"samples_{run}_pretrain_k32_textbook72_distract2.jsonl"))}
    bins = collections.defaultdict(lambda: dict(n=0, orig_solved=0, dist_solved=0, both=0))
    cross = dict(n=0, orig_solved=0, dist_solved=0)
    stay = dict(n=0, orig_solved=0, dist_solved=0)
    for r in dist:
        ho, hd = o[r["name"]] > 0, d[r["name"]] > 0
        b = bins[min(3, r["n_prem_orig"])]
        b["n"] += 1; b["orig_solved"] += ho; b["dist_solved"] += hd; b["both"] += ho and hd
        t = cross if r["n_prem"] >= 4 else stay
        t["n"] += 1; t["orig_solved"] += ho; t["dist_solved"] += hd
    m = {x["set"]: x for x in jl(os.path.join(EXP, "run", label, "metrics.jsonl"))}
    return dict(model=label, ckpt=run, k=32, seed=0, n=len(dist),
                original=dict(n=m["textbook72"]["n"], avg_solve_rate=m["textbook72"]["avg_solve_rate"], any_of_k=m["textbook72"]["any_of_k"],
                              parse_fail_frac=m["textbook72"]["parse_fail_frac"]),
                distract2=dict(n=m["textbook72_distract2"]["n"], avg_solve_rate=m["textbook72_distract2"]["avg_solve_rate"],
                               any_of_k=m["textbook72_distract2"]["any_of_k"], parse_fail_frac=m["textbook72_distract2"]["parse_fail_frac"]),
                original_on_the_71=dict(any_of_k=sum(1 for r in dist if o[r["name"]] > 0),
                                        avg_solve_rate=round(sum(o[r["name"]] for r in dist) / len(dist), 4)),
                by_orig_premises={str(k): v for k, v in sorted(bins.items())}, crossing_to_4plus=cross, staying_le3=stay)


def main():
    os.makedirs(os.path.join(REL, "metrics"), exist_ok=True)
    sealed = {r["name"]: r for r in TD.load_textbook72()}
    dist = jl(os.path.join(T72, "textbook72_distract2.jsonl"))
    order = {n: i for i, n in enumerate(sealed)}
    dist.sort(key=lambda r: order[r["name"]])
    # 1. distractors.jsonl and the canonical hash
    drows = distractor_rows(dist, sealed)
    PC_write = lambda p, rows: open(p, "w").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))  # noqa: E731
    PC_write(os.path.join(REL, "distractors.jsonl"), drows)
    canon = AD.canon_sha256([dict(r, n_distract=2) for r in dist])
    # 2. examples on non-sealed problems
    ex = examples()
    PC_write(os.path.join(REL, "examples.jsonl"), ex)
    # 3. metrics
    summary = []
    for label, run in ROBBIE.items():
        shutil.copy(os.path.join(EXP, "run", label, "metrics.jsonl"), os.path.join(REL, "metrics", f"{label}.metrics.jsonl"))
        summary.append(robbie_bins(label, run, dist))
    shutil.copy(os.path.join(EXP, "run", "metrics.jsonl"), os.path.join(REL, "metrics", "dan.metrics.jsonl"))
    for m in jl(os.path.join(EXP, "run", "metrics.jsonl")):
        # 95 % interval over problems of the average solve rate, from the gitignored per-problem hit counts
        hits = json.load(open(os.path.join(T72, f"distract2_hits_{m['ckpt']}_k32.json")))
        for key in ("original", "distract2"):
            rates = [h / m["k"] for h in hits[key]]
            mu = sum(rates) / len(rates)
            sd = (sum((x - mu) ** 2 for x in rates) / (len(rates) - 1)) ** 0.5
            m[key]["ci95"] = round(1.96 * sd / len(rates) ** 0.5, 4)
        summary.append(dict(model="dan_stage1" if m["ckpt"].startswith("stage1") else "dan_r8_s0", **m))
    json.dump(summary, open(os.path.join(REL, "metrics", "summary.json"), "w"), indent=1)
    # 4. manifest
    import eval_ckpt as EC
    robbie_ckpts = {}
    for label, run in ROBBIE.items():
        d = EC.find_run(run)
        log = open(os.path.join(EXP, "run", label, "log.txt")).readline()
        meta = json.loads(log.split("loaded")[1].split(":", 1)[1])
        robbie_ckpts[run] = dict(repo="robbiethompson2018/nd-rl-checkpoints", snapshot=os.path.basename(os.path.dirname(os.path.dirname(d))),
                                 file=f"runs/{run}/pretrain.pt", sha256=sha256(os.path.join(d, "pretrain.pt")), n_params=meta["n_params"],
                                 arch=meta["arch"], tok_mode=meta["tok_mode"], harness_sha=meta.get("harness_sha"), pretrain_sha=meta.get("pretrain_sha"))
    dan_ckpts = {c: dict(bucket="hf://buckets/dan-pandori/nd-rl",
                         file=f"best-state/ckpts/bs/{'ladder/' if c.startswith('la_') else ''}{c}.pt",
                         sha256=sha256(os.path.expanduser(f"~/.cache/nd-rl/dan_ckpts/{c}.pt")), n_params=9560832,
                         format="lean_staten proof-state format, Dan's nd-takehome at 0c492902") for c in DAN}
    ws3 = json.load(open(os.path.join(PC.ROOT, "code", "ws3", "textbook72", "MANIFEST.json")))
    manifest = dict(
        release="2026-10-12", experiment="experiments/pretrain-data/textbook72_distract2", built_by="code/pretrain_data/textbook72_distract_release.py",
        sealed_inputs=dict(source=ws3["source_branch"] + ":" + ws3["source_dir"], files=ws3["files"], restore="bash code/ws3/textbook72/fetch.sh"),
        distracted_set=dict(n=len(dist), of=len(sealed), builder="code/pretrain_data/textbook72_distract.py", seed=0, canonical_keys=list(AD.CANON_KEYS),
                            canonical_sha256=canon, rebuild="python apply_distractors.py",
                            premises_before=dict(sorted(collections.Counter(r["n_prem_orig"] for r in dist).items())),
                            premises_after=dict(sorted(collections.Counter(r["n_prem"] for r in dist).items())),
                            crossing_to_4plus=sum(r["n_prem"] >= 4 for r in dist), staying_le3=sum(r["n_prem"] <= 3 for r in dist),
                            shortcut_check=dict(collections.Counter(r["shortcut_check"] for r in dist)),
                            minimal_length_status=dict(collections.Counter(r["min_status"] for r in dist)),
                            mean_prompt_tokens=dict(original=round(sum(len(r["base_prompt"].split()) for r in dist) / len(dist), 1),
                                                    distracted=round(sum(len(r["prompt"].split()) for r in dist) / len(dist), 1))),
        files={f: sha256(os.path.join(REL, f)) for f in ("distractors.jsonl", "examples.jsonl", "apply_distractors.py")},
        models=dict(robbie=robbie_ckpts, dan=dan_ckpts),
        protocol=dict(k=32, temperature=1.0, seed=0, verifier="pinned nd_verify via code/textbook_problem/strict_verify.py",
                      robbie="code/textbook_problem/eval_ckpt.py --sets textbook72,textbook72_distract2 --k 32 --temp 1.0 --max-new 400 (lean_seq, free sampling)",
                      dan="code/pretrain_data/textbook72_distract_eval_dan.py --ckpt <ckpt> (state_sample.env_generate, 48 steps, 256 tokens per action)"),
    )
    json.dump(manifest, open(os.path.join(REL, "MANIFEST.json"), "w"), indent=1)
    print("distractors", len(drows), "canonical sha256", canon)
    print("examples", [(e["name"][:18], e["n_prem_orig"], e["n_prem"], e["shortcut_check"], e["min_status"], e["draws"], e["original_reference_proof_verifies_on_distracted"]) for e in ex])
    for s in summary:
        print(s["model"], "orig", s["original"]["any_of_k"], "dist", s["distract2"]["any_of_k"], "cross", s["crossing_to_4plus"], "stay", s["staying_le3"],
              "bins", {k: (v["orig_solved"], v["dist_solved"]) for k, v in s["by_orig_premises"].items()})


if __name__ == "__main__":
    main()
