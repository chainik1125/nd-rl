# ruff: noqa: E501, SIM115
"""Build the Stage 2 (T, H) and Stage 0c (rep) inputs from Dan's own files -> /scratch/cowifornia/nd-rl/olmo_elicit/stage2/.

T (transplant, his olmo-transplant / olmo-answeronly reach arms): for each checkpoint c, T_<c>.jsonl.gz =
  - his olmo_transplant/requests/s2300_reach.jsonl.gz rows with cond T2, cut full (115 traces of the final model's
    genuine thinking on the 41 reach-41 prompts, 8 answers each; no instruction), cond relabelled "T";
  - his olmo_answeronly/requests/s2300_reach.jsonl.gz rows with cond A1 (the same 115 thinking prefixes, user turn +
    "\\n\\nAfter thinking, reply with only your final answer, exactly as drafted, with no introduction or explanation.");
  with `writer` set to c (his sample_prefix.py asserts writer == --name). Nothing else changes: prefix ids, user_suffix,
  n, rid, group are his bytes. Checked: every T prefix equals his dpo-writer T1 prefix and his s2300-writer T2 prefix
  of the same trace, every A1 prefix equals the T prefix of its trace, the A1 suffix is his registered wording.
H (60-word hint on the 48 reach prompts): problems_H.jsonl = his olmo_iffu/problems_hint.jsonl and
  problems_hint2c.jsonl lines, byte for byte, in pids_reach.txt order (prompt_sample = prompt + "\\n\\nAnswer in at most
  60 words."; every other field equals olmo_ifomega/problems.jsonl's row, checked); pids_H_n8.txt / pids_H_n4.txt.
rep (Stage 0c): pids_rep.txt = the 218 arm-A prompts x 2 draws, on olmo_ifomega/problems.jsonl (no new file).
seeds.csv: our engine seeds, new (Dan's were < 60,000): T 610000 + step, H 620000 + step, rep 630000 + step.
manifest.json: sha256 of every input read and every file written.
Usage: python make_stage2.py [--ckpts dpo,s0050,...]   (default: all nine)
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path

DAN = Path("/scratch/cowifornia/nd-rl/olmo_elicit/ext/nd-takehome")
OUT = Path("/scratch/cowifornia/nd-rl/olmo_elicit/stage2")
CKPTS = ["dpo", "s0050", "s0100", "s0200", "s0500", "s0750", "s1100", "s1600", "s2300"]
STEP = {c: (0 if c == "dpo" else int(c[1:])) for c in CKPTS}
A1 = "After thinking, reply with only your final answer, exactly as drafted, with no introduction or explanation."
HINT = "Answer in at most 60 words."
SRC = {
    "tp_s2300": DAN / "artifacts/olmo_transplant/requests/s2300_reach.jsonl.gz",
    "tp_dpo": DAN / "artifacts/olmo_transplant/requests/dpo_reach.jsonl.gz",
    "ao_s2300": DAN / "artifacts/olmo_answeronly/requests/s2300_reach.jsonl.gz",
    "hint16": DAN / "olmo_iffu/problems_hint.jsonl",
    "hint32": DAN / "olmo_iffu/problems_hint2c.jsonl",
    "problems": DAN / "olmo_ifomega/problems.jsonl",
    "reach": DAN / "olmo_ifcurve/pids_reach.txt",
    "armA": DAN / "olmo_ifcurve/pids_armA.txt",
}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rows(p: Path) -> list[dict]:
    return [json.loads(ln) for ln in gzip.open(p, "rt")]


def ids(b64: str) -> list[int]:
    import array

    a = array.array("i")
    a.frombytes(base64.b64decode(b64))
    return a.tolist()


def t_rows() -> tuple[list[dict], list[dict]]:
    tp = [r for r in rows(SRC["tp_s2300"]) if r["cond"] == "T2" and r["cut"] == "full"]
    tp_dpo = {
        r["trace"]: r for r in rows(SRC["tp_dpo"]) if r["cond"] == "T1" and r["cut"] == "full"
    }
    ao = [r for r in rows(SRC["ao_s2300"]) if r["cond"] == "A1"]
    assert len(tp) == 115 and len(ao) == 115 and len(tp_dpo) == 115, (len(tp), len(ao), len(tp_dpo))
    assert len({r["pid"] for r in tp}) == 41 and {r["trace"] for r in tp} == {
        r["trace"] for r in ao
    } == set(tp_dpo)
    by_trace = {r["trace"]: r for r in tp}
    for r in tp:
        assert r["prefix"] == tp_dpo[r["trace"]]["prefix"], r[
            "trace"
        ]  # same thinking as his dpo-writer T1
        assert (
            int(r["n"]) == 8 and r.get("user_suffix", "") == "" and r["rid"] == f"{r['trace']}|full"
        )
        p = ids(r["prefix"])
        assert p[-3:-1] == [524, 27963] and len(p) == int(r["prefix_len"]), r[
            "trace"
        ]  # ends in `</` `think` `>..`
    for r in ao:
        assert r["prefix"] == by_trace[r["trace"]]["prefix"], r[
            "trace"
        ]  # A1 = the full transplant + the instruction
        assert (
            r["user_suffix"] == "\n\n" + A1 and int(r["answer_prefill_k"]) == 0 and int(r["n"]) == 8
        )
    return tp, ao


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", default=",".join(CKPTS))
    a = ap.parse_args()
    ckpts = a.ckpts.split(",")
    OUT.mkdir(parents=True, exist_ok=True)
    man = {
        "inputs": {k: {"path": str(v), "sha256": sha256(v)} for k, v in SRC.items()},
        "outputs": {},
    }

    tp, ao = t_rows()
    for c in ckpts:
        f = OUT / f"T_{c}.jsonl.gz"
        with gzip.GzipFile(f, "wb", mtime=0) as g:
            for r in tp:
                g.write((json.dumps({**r, "writer": c, "cond": "T"}) + "\n").encode())
            for r in ao:
                g.write((json.dumps({**r, "writer": c}) + "\n").encode())

    reach = [ln.strip() for ln in open(SRC["reach"]) if ln.strip()]
    base = {json.loads(ln)["pid"]: json.loads(ln) for ln in open(SRC["problems"])}
    lines = {}
    for k in ("hint16", "hint32"):
        for ln in open(SRC[k]):
            if ln.strip():
                lines[json.loads(ln)["pid"]] = ln if ln.endswith("\n") else ln + "\n"
    assert set(lines) == set(reach) and len(reach) == 48
    for p in reach:
        h = json.loads(lines[p])
        assert h["prompt_sample"] == h["prompt"] + "\n\n" + HINT and h["hint"] == HINT
        assert all(h[k] == v for k, v in base[p].items()), p
    (OUT / "problems_H.jsonl").write_text("".join(lines[p] for p in reach))
    for n in (8, 4):
        (OUT / f"pids_H_n{n}.txt").write_text("".join(f"{p}:{n}\n" for p in reach))
    arm_a = [ln.strip() for ln in open(SRC["armA"]) if ln.strip()]
    assert len(arm_a) == 218 and all(p in base for p in arm_a)
    (OUT / "pids_rep.txt").write_text("".join(f"{p}:2\n" for p in arm_a))
    (OUT / "seeds.csv").write_text(
        "ckpt,step,seed_T,seed_H,seed_rep\n"
        + "".join(
            f"{c},{STEP[c]},{610000 + STEP[c]},{620000 + STEP[c]},{630000 + STEP[c]}\n"
            for c in CKPTS
        )
    )
    for f in sorted(OUT.iterdir()):
        if f.name != "manifest.json":
            man["outputs"][f.name] = sha256(f)
    man["counts"] = {
        "T_rows_per_ckpt": {"T": len(tp), "A1": len(ao)},
        "T_answers_per_ckpt": 8 * (len(tp) + len(ao)),
        "T_prompts": len({r["pid"] for r in tp}),
        "H_prompts": len(reach),
        "rep_prompts": len(arm_a),
    }
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    print(
        f"stage2 inputs -> {OUT}: T for {ckpts} ({len(tp)} T + {len(ao)} A1 requests x 8 answers each), "
        f"H 48 prompts, rep 218 x 2, seeds.csv"
    )


if __name__ == "__main__":
    main()
