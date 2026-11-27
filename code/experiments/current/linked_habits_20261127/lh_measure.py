"""Measurement of a checkpoint for linked habits version 2 (SPEC2.md): W2's `authors_measure run` with the 24-habit
reading, the step limit 51, and pass@1 / pass@8; Dan's strict-106 read at the same limit.

    python lh_measure.py run --ckpt CKPT [--ckpt_id ID] [--tags A,B,none] [--sets marker_plain,route_unseen] [--k 8]
                             [--seed 0] [--batch 256] [--device cuda] [--limit N] [--out_dir DIR] [--raw_dir DIR]
                             [--resume] [--dry-run]
    python lh_measure.py strict106 --ckpt CKPT --ckpt_id ID --out_dir DIR [--k 64] [--batch 128] [--seed 0] [--cpu] [--resume]
    python lh_measure.py probe --ckpts A.pt,B.pt --ids step25,step50 --out_root DIR --raw_root DIR [--id_prefix P]
                               [--stop_habit H --stop_above SHARE] [--device cuda] [--k 8] [--batch 256] [--limit N]
    python lh_measure.py sets [--dry-run]     # copy W2's measurement sets into LH/sets; premise counts and classes
    python lh_measure.py estimate [...]       # W2's estimate

`run` is ../two_authors_20261111/authors_measure.py `run` (W2, commit 9618763: Dan's sampler at T 0.8 under tags A, B
and none with paired noise seeds, the Lean gate, `authors_profile.profile` of every accepted proof, counts-only rows
LH/measure/<ckpt_id>/<set>__<tag>.jsonl, raw proofs LH/raw/<ckpt_id>/, never overwritten) with, in this process only:
  * the step limit: `state_sample.env_generate` is wrapped so every call runs with max_steps = 51 (lh_core.STEP_LIMIT;
    48 before), whatever its caller passes, and `authors_measure.MAX_STEPS` = 51 (W2 records it in every summary);
  * `authors_measure.theorem_row` also writes `lh` (below) into every row;
  * `authors_measure._write_atomic` adds `lh_totals` to every `.summary.json`: attempts, accepted %, pass@1 (mean
    over theorems of accepted / attempts), pass@8 (mean over theorems of 1 - C(n-c, 8) / C(n, 8); at k 8, a theorem
    with any accepted attempt), mean lines of the read proofs, share of attempts that ended at the step limit, and the
    A-share of every habit.
Defaults differ from W2's: `--k 8` and `--sets marker_plain,route_unseen`. The noise seed is W2's.

Row field `lh` (counts only; every accepted proof read by `lh_profile.profile`, the corpus agent's detector of the 24
habits; lists of two are [A, B]):
  max_steps (51), step_cap (this theorem's attempts that ended at the step limit), proofs (read), unread, lines (sum of
  written lines of the read proofs), hab {habit: [sites, a, b, other, none]} (other / none: note habits only),
  with_site {habit: proofs with a site}, first {habit: [A, B]} (the proof's first site with a form, in proof order,
  when it is this habit's), later {habit: [A, B]} (every other site), link {"X|Y": [AA, AB, BA, BB]} (X, Y in notes,
  natural, all: the first site of group X in the proof, by its form, against every later site of group Y),
  linko {"X|Y": [AA, AB, BA, BB]} (the same, later sites of habits other than the first site's: the primary G2),
  pair {"h1|h2": [AA, AB, BA, BB]} (habit h1's first site against the later sites of habit h2),
  cond {R: {h: [hA|RA, hB|RA, hA|RB, hB|RB]}} (R = reit, note_atom: the sites of other habits written before the proof's
  first R site, by the R site's form and their own; the GRPO arm's chain prediction), condn {R: [first R site A, B]},
  near {ande_apart, dead_r_in_box, cited_header_copy, double_note, late_note} (lh_profile's near-miss counters, summed
  over the read proofs; 0 while the profile has no `near`).
  Left out of first / later / link / linko / pair / cond (PREREG Amendment 3, point 8): the contradiction route (a
  choice per proof), case split (read in one form only) and ex falso (part of the route habit); their own A-shares are
  in `hab`.

`strict106`: ../author_tags_20261110/tag_danread.py --tag none (Dan's knockout danread.py: k 64 on the strict 106,
T 0.8, batch 128, as W2's lines read their bases) in this process, at the step limit 51 (the wrapper above; danread
sets 48 itself), then W2's `authors_measure.danread` profile of its proofs. Writes DIR/strict106_read.json (+ its
.proofs.jsonl, raw) and DIR/strict106__none.jsonl; the read gains `max_steps` and `step_cap_share`.

`sets`: copies W2's two sets from TA/sets byte for byte (sha256 checked) to LH/sets/, with their premise report.
"""

# ruff: noqa: E501, I001
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lh_core as C  # noqa: E402  (TA/AT = LH before W2's modules)
import notes as N  # noqa: E402
import authors_measure as AM  # noqa: E402
import authors_sets as AS  # noqa: E402

AP = AM.AP
T = AM.T
GROUPS = ("notes", "natural", "all")
# not in the site-order readings (PREREG Amendment 3, point 8): the route is a per-proof choice; case split is read in
# one form only and ex falso is part of the route habit, so neither is a per-site choice linked to the author
NO_SEQ = {"route", "split", "exfalso"}
# lh_profile's near-miss counters (design audit F9; `near` in the profile output; a missing key counts 0)
NEAR = ("ande_apart", "dead_r_in_box", "cited_header_copy", "double_note", "late_note")
_P: dict[str, Any] = {}


def profiler() -> tuple[Any, Any]:
    """(lh_profile module, its tables), loaded once."""
    if "m" not in _P:
        import lh_profile as P

        _P["m"], _P["t"] = P, C.profile_tables()
    return _P["m"], _P["t"]


def habits() -> list[str]:
    return list(profiler()[0].HABITS)


def group_of(h: str) -> str:
    return "notes" if h.startswith("note_") else "natural"


# ------------------------------------------------------------------------------------------------ the step limit
_LAST: dict[str, list[str]] = {}  # prompt -> its attempts' ND strings, from the newest sampler call


def install_step_limit() -> None:
    """Every `state_sample.env_generate` call in this process runs at the step limit 51; the attempts of the newest
    call are kept by prompt (for the per-theorem step-limit count)."""
    T.install_dan()
    import state_sample as SS  # type: ignore[import-not-found]

    if getattr(SS.env_generate, "_lh_orig", None) is not None:
        return
    orig = SS.env_generate

    def env_generate(model, tok, prompts, *args, **kw):
        if len(args) >= 4:  # (greedy, temperature, max_action, max_steps, ...) positionally
            args = (*args[:3], C.STEP_LIMIT, *args[4:])
            kw.pop("max_steps", None)
        else:
            kw["max_steps"] = C.STEP_LIMIT
        nds = orig(model, tok, prompts, *args, **kw)
        _LAST.clear()
        for p, x in zip(prompts, nds, strict=True):
            _LAST.setdefault(p, []).append(x)
        return nds

    env_generate._lh_orig = orig  # type: ignore[attr-defined]
    SS.env_generate = env_generate
    AM.MAX_STEPS = C.STEP_LIMIT


def at_limit(nd: str) -> bool:
    """Dan's environment ends an attempt at the step limit with `failed = 'step cap'` (state_env.Env.nd)."""
    return nd.startswith("LEANPARSE env: step cap")


# ------------------------------------------------------------------------------------------------ habit counts
def empty_lh() -> dict[str, Any]:
    return {"max_steps": C.STEP_LIMIT, "step_cap": None, "seen": 0, "proofs": 0, "unread": 0, "lines": 0, "hab": {},
            "with_site": {}, "first": {}, "later": {}, "link": {f"{x}|{y}": [0, 0, 0, 0] for x in GROUPS for y in GROUPS},
            "linko": {f"{x}|{y}": [0, 0, 0, 0] for x in GROUPS for y in GROUPS}, "pair": {},
            "cond": {r: {} for r in C.RATERS.values()}, "condn": {r: [0, 0] for r in C.RATERS.values()},
            "near": dict.fromkeys(NEAR, 0)}  # fmt: skip


def _ab(side: str) -> int:
    return 0 if side == "A" else 1


def add_proof(c: dict[str, Any], prompt: str, proof: str) -> dict[str, Any]:
    """Add one accepted proof's 24-habit reading to the counts `c`; returns the profile."""
    P, tab = profiler()
    p = P.profile(prompt, proof, tab)
    if not p.get("ok"):
        c["unread"] += 1
        return p
    c["proofs"] += 1
    c["lines"] += int(p.get("lines") or 0)
    for k, v in (p.get("near") or {}).items():
        c["near"][k] = c["near"].get(k, 0) + int(v or 0)
    for h, v in p["habits"].items():
        hv = c["hab"].setdefault(h, [0, 0, 0, 0, 0])
        for i, k in enumerate(("sites", "a", "b", "other", "none")):
            hv[i] += int(v.get(k) or 0)
        if v.get("sites"):
            c["with_site"][h] = c["with_site"].get(h, 0) + 1
    seq = [(h, s) for _, h, s in p.get("seq") or [] if h not in NO_SEQ and s in ("A", "B")]
    if not seq:
        return p
    h0, s0 = seq[0]
    c["first"].setdefault(h0, [0, 0])[_ab(s0)] += 1
    for h, s in seq[1:]:
        c["later"].setdefault(h, [0, 0])[_ab(s)] += 1
    grp = [(h, s, group_of(h)) for h, s in seq]
    for x in GROUPS:
        i = next((j for j, (_, _, g) in enumerate(grp) if x == "all" or g == x), None)
        if i is None:
            continue
        hx, sx = grp[i][0], grp[i][1]
        for hy, sy, gy in grp[i + 1 :]:
            for y in ("all", gy):
                c["link"][f"{x}|{y}"][2 * _ab(sx) + _ab(sy)] += 1
                if (
                    hy != hx
                ):  # the primary G2: later sites of other habits (other classes, for notes)
                    c["linko"][f"{x}|{y}"][2 * _ab(sx) + _ab(sy)] += 1
    firsts: dict[str, int] = {}
    for j, (h, _) in enumerate(seq):
        firsts.setdefault(h, j)
    for h1, i in firsts.items():
        s1 = seq[i][1]
        for h2, s2 in seq[i + 1 :]:
            c["pair"].setdefault(f"{h1}|{h2}", [0, 0, 0, 0])[2 * _ab(s1) + _ab(s2)] += 1
    # the GRPO arm's chain prediction (PREREG Amendment 3, 6b): for each rated habit R with a site, the proof's first
    # R form, and every site of another habit written before it, by its form: [hA|RA, hB|RA, hA|RB, hB|RB]
    for r in C.RATERS.values():
        i = firsts.get(r)
        if i is None:
            continue
        fr = seq[i][1]
        c["condn"][r][_ab(fr)] += 1
        for h, sh in seq[:i]:
            if h != r:
                c["cond"][r].setdefault(h, [0, 0, 0, 0])[2 * _ab(fr) + _ab(sh)] += 1
    return p


def lh_counts(prompt: str, proofs: list[str]) -> dict[str, Any]:
    c = empty_lh()
    for pr in proofs:
        add_proof(c, prompt, pr)
    outs = _LAST.get(prompt)
    if outs is not None:
        c["seen"], c["step_cap"] = len(outs), sum(at_limit(x) for x in outs)
    return c


_W2_ROW = getattr(AM.theorem_row, "_lh_orig", AM.theorem_row)


def theorem_row(
    thm: dict, proofs: list[str], attempts: int, base: dict[str, Any]
) -> dict[str, Any]:
    """W2's per-theorem row plus `lh` (module docstring)."""
    row = _W2_ROW(thm, proofs, attempts, base)
    row["lh"] = lh_counts(thm["prompt"], proofs)
    return row


theorem_row._lh_orig = _W2_ROW  # type: ignore[attr-defined]


def pass_at(n: int, c: int, k: int) -> float | None:
    """The unbiased pass@k of one theorem (n attempts, c accepted); None when n < k."""
    if n < k or n <= 0:
        return None
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pooled totals of a (set, tag) (module docstring)."""
    n = len(rows)
    att = sum(r["attempts"] for r in rows)
    acc = sum(r["accepted"] for r in rows)
    lh = [r.get("lh") or {} for r in rows]
    p8 = [pass_at(r["attempts"], r["accepted"], 8) for r in rows]
    seen = sum(x.get("seen") or 0 for x in lh)
    out: dict[str, Any] = {
        "theorems": n, "attempts": att, "accepted": acc, "accepted_pct": round(100 * acc / att, 2) if att else None,
        "pass1": round(100 * sum(r["accepted"] / r["attempts"] for r in rows if r["attempts"]) / n, 2) if n else None,
        "pass8": round(100 * sum(x for x in p8 if x is not None) / len([x for x in p8 if x is not None]), 2) if any(x is not None for x in p8) else None,
        "step_limit": C.STEP_LIMIT, "step_cap": sum(x.get("step_cap") or 0 for x in lh),
        "step_cap_pct": round(100 * sum(x.get("step_cap") or 0 for x in lh) / seen, 2) if seen else None,
        "proofs": sum(x.get("proofs", 0) for x in lh), "unread": sum(x.get("unread", 0) for x in lh),
    }  # fmt: skip
    out["mean_lines"] = (
        round(sum(x.get("lines", 0) for x in lh) / out["proofs"], 2) if out["proofs"] else None
    )
    hab: dict[str, list[int]] = {}
    for x in lh:
        for h, v in (x.get("hab") or {}).items():
            hv = hab.setdefault(h, [0, 0, 0, 0, 0])
            for i, y in enumerate(v[:5]):
                hv[i] += y
    out["a_share"] = {
        h: (round(100 * v[1] / (v[1] + v[2]), 1) if v[1] + v[2] else None) for h, v in hab.items()
    }
    out["sites"] = {h: v[0] for h, v in hab.items()}
    # sites in neither form (lh_profile `other`: a note's other menu forms; a natural habit's near misses, e.g. both
    # halves written apart), beside the A-share A / (A + B), in % of the habit's sites
    out["other_pct"] = {h: (round(100 * v[3] / v[0], 2) if v[0] else None) for h, v in hab.items()}
    near: dict[str, int] = dict.fromkeys(NEAR, 0)
    for x in lh:
        for k, v in (x.get("near") or {}).items():
            near[k] = near.get(k, 0) + int(v or 0)
    out["near"] = near
    out["near_per_proof"] = {
        k: (round(v / out["proofs"], 4) if out["proofs"] else None) for k, v in near.items()
    }
    return out


_W2_WRITE = getattr(AM._write_atomic, "_lh_orig", AM._write_atomic)
_ROWS: dict[str, str] = {}


def _write_atomic(path: Path, text: str) -> None:
    """W2's atomic write; a summary gets `lh_totals` from the rows written just before it."""
    path = Path(path)
    if path.name.endswith(".summary.json"):
        rows_text = _ROWS.pop(path.name.removesuffix(".summary.json") + ".jsonl", None)
        if rows_text is not None:
            d = json.loads(text)
            tot = totals([json.loads(x) for x in rows_text.splitlines() if x.strip()])
            d["lh_totals"] = tot
            d["step_limit"] = C.STEP_LIMIT
            d["tables_sha256"] = {
                p.name: C.sha256(p) for p in (C.NOTE_TABLE, C.HABIT_TABLE) if p.exists()
            }
            text = json.dumps(d, indent=1, default=str)
            print(f"  {d.get('set')} tag {d.get('tag')}: accepted {tot['accepted_pct']} %, pass@1 {tot['pass1']}, "
                  f"pass@8 {tot['pass8']}, at the step limit {tot['step_cap_pct']} %, mean lines {tot['mean_lines']}", flush=True)  # fmt: skip
    elif path.name.endswith(".jsonl"):
        _ROWS[path.name] = text
    _W2_WRITE(path, text)


_write_atomic._lh_orig = _W2_WRITE  # type: ignore[attr-defined]


def install() -> None:
    profiler()  # fail before any sampling if the detector does not load
    install_step_limit()
    AM.theorem_row = theorem_row
    AM._write_atomic = _write_atomic


# ------------------------------------------------------------------------------------------------ strict 106
def strict106_main(argv: list[str]) -> Path:
    ap = argparse.ArgumentParser(prog="lh_measure.py strict106")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt_id", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true", help="CPU smoke only")
    ap.add_argument(
        "--resume", action="store_true", help="a finished read is skipped instead of refused"
    )
    a = ap.parse_args(argv)
    d = Path(a.out_dir)
    read, rows = d / "strict106_read.json", d / "strict106__none.jsonl"
    if rows.exists():
        if a.resume:
            print(f"{rows} exists: skipped (--resume)")
            return rows
        raise SystemExit(f"{rows} exists: never overwritten")
    install()
    if not read.exists():
        import tag_danread as TD

        d.mkdir(parents=True, exist_ok=True)
        # Dan's lean_gate logs to artifacts/lean_gate.jsonl under its cwd (Dan's tree) unless told otherwise
        os.environ["LEAN_GATE_LOG"] = str(d / "strict106_lean_gate.jsonl")
        TD.main(["--tag", "none", *(["--cpu"] if a.cpu else []), "--", "--ckpt", a.ckpt, "--seed", str(a.seed),
                 "--round", "0", "--out", str(read), "--only106", "--k", str(a.k), "--batch", str(a.batch)])  # fmt: skip
        res = json.loads(read.read_text())
        end = (res.get("env_stats") or {}).get("env_end") or {}
        res["max_steps"] = C.STEP_LIMIT
        res["step_cap_share"] = round(
            end.get("step_cap", 0) / max(1, res["n_prompts"] * res["k"]), 5
        )
        tmp = read.with_name(read.name + ".tmp")
        tmp.write_text(json.dumps(res, indent=1, default=str))
        tmp.replace(read)
        print(f"strict 106: {res['strict_n']} / 106 at k {res['k']}, step limit {C.STEP_LIMIT} "
              f"({100 * res['step_cap_share']:.2f} % of attempts end there)", flush=True)  # fmt: skip
    AM.danread(argparse.Namespace(read=str(read), ckpt_id=a.ckpt_id, out=str(rows)))
    return rows


# ------------------------------------------------------------------------------------------------ sets
def set_report(rows: list[dict]) -> dict[str, Any]:
    n_prem = collections.Counter(r["n_prem"] for r in rows)
    thm_cls: collections.Counter = collections.Counter()
    prem_cls: collections.Counter = collections.Counter()
    for r in rows:
        prem, _ = AS.split_thm(r["thm"])
        prem = [p for p in prem if p.strip()]
        assert len(prem) == r["n_prem"], r["name"]
        ks = [N.premise_class(p) for p in prem]
        prem_cls.update(ks)
        thm_cls.update(set(ks))
    n = len(rows)
    two = sum(v for k, v in n_prem.items() if k >= 2)
    return {"theorems": n, "premise_counts": {str(k): n_prem[k] for k in sorted(n_prem)},
            "two_plus": two, "two_plus_pct": round(100 * two / n, 1) if n else None,
            "classes": {k: {"theorems": thm_cls[k], "premises": prem_cls[k]} for k in N.CLASSES}}  # fmt: skip


def sets_main(argv: list[str]) -> dict[str, Any]:
    ap = argparse.ArgumentParser(prog="lh_measure.py sets")
    ap.add_argument("--src", default=str(C.TA_SETS))
    ap.add_argument("--out", default=str(C.LH / "sets"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    src, out = Path(a.src), Path(a.out)
    tam = json.loads((src / "manifest.json").read_text())
    rep: dict[str, Any] = {
        "source": str(src),
        "source_manifest_sha256": C.sha256(src / "manifest.json"),
        "sets": {},
    }
    for s in C.SETS:
        fn = src / f"{s}.jsonl"
        sha = C.sha256(fn)
        assert sha == tam["sets"][s]["sha256"], f"{fn}: sha256 differs from TA's manifest"
        rep["sets"][s] = {"file": f"{s}.jsonl", "sha256": sha, **set_report(AS.load_set(s, src))}
    mp = rep["sets"]["marker_plain"]
    rep["note_plain_rule"] = {"rule": "SPEC: build note_plain if fewer than 60 % of marker_plain theorems have at least 2 premises",
                              "marker_plain_two_plus_pct": mp["two_plus_pct"], "build_note_plain": mp["two_plus_pct"] < 60}  # fmt: skip
    for s, v in rep["sets"].items():
        print(f"{s}: {v['theorems']} theorems; premises per theorem {v['premise_counts']}; >= 2 premises {v['two_plus']} "
              f"({v['two_plus_pct']} %)")  # fmt: skip
    if a.dry_run:
        return rep
    out.mkdir(parents=True, exist_ok=True)
    for s in C.SETS:
        dst = out / f"{s}.jsonl"
        if dst.exists():
            assert C.sha256(dst) == rep["sets"][s]["sha256"], f"{dst} exists and differs from {src}"
            continue
        tmp = dst.with_name(dst.name + ".tmp")
        shutil.copyfile(src / f"{s}.jsonl", tmp)
        assert C.sha256(tmp) == rep["sets"][s]["sha256"]
        tmp.replace(dst)
    C.write_once(out / "manifest.json", rep, ("source_manifest_sha256", "sets"))
    print(f"wrote {out}/{{{','.join(C.SETS)}}}.jsonl and manifest.json")
    return rep


# ------------------------------------------------------------------------------------------------ probes
PROBE_SET = "marker_plain"


def probe_main(argv: list[str]) -> dict[str, Any]:
    """Cheap probes of a rated run's snapshots in one process (PREREG Amendments 3 and 4): `run` untagged on
    marker_plain at k 8 (the same sampler, step limit, Lean gate and detector as the full measurement), snapshot by
    snapshot in the order given, each into OUT_ROOT/<id>/ (rows and summary as `run`; DONE when finished, skipped when
    present). With --stop_habit / --stop_above, probing stops after the first snapshot whose A-share of that habit is
    above the threshold (the base's share + 30), and the last snapshot is always probed. Then OUT_ROOT/probe.json with
    every probed snapshot's A-share per habit, sites, accepted %, pass@1 and the share at the step limit."""
    ap = argparse.ArgumentParser(prog="lh_measure.py probe")
    ap.add_argument("--ckpts", required=True, help="comma list of checkpoints, in step order")
    ap.add_argument("--ids", required=True, help="comma list of snapshot ids (same order)")
    ap.add_argument(
        "--id_prefix", default="", help="prefix of the ckpt ids in the rows (e.g. the arm)"
    )
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--raw_root", required=True)
    ap.add_argument("--stop_habit", default=None)
    ap.add_argument(
        "--stop_above",
        type=float,
        default=None,
        help="A-share (%%) of --stop_habit that ends the probing",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None, help="smoke only: the first N theorems")
    a = ap.parse_args(argv)
    cks, ids = a.ckpts.split(","), a.ids.split(",")
    if len(cks) != len(ids) or len(set(ids)) != len(ids):
        raise SystemExit("--ckpts and --ids: same length, distinct ids")
    root = Path(a.out_root)

    def summary(i: str) -> dict[str, Any]:
        return (
            json.loads((root / i / f"{PROBE_SET}__none.summary.json").read_text()).get("lh_totals")
            or {}
        )

    def one(ck: str, i: str) -> None:
        d = root / i
        if (d / "DONE").exists():
            print(f"probe {i}: done before", flush=True)
            return
        d.mkdir(parents=True, exist_ok=True)
        AM.main(["run", "--ckpt", ck, "--ckpt_id", f"{a.id_prefix}{i}", "--tags", "none", "--sets", PROBE_SET,
                 "--k", str(a.k), "--batch", str(a.batch), "--device", a.device, "--out_dir", str(d),
                 "--raw_dir", str(Path(a.raw_root) / i), "--resume", *(["--limit", str(a.limit)] if a.limit else [])])  # fmt: skip
        (d / "DONE").write_text("probe\n")

    install()
    done, stopped = [], None
    for ck, i in zip(cks, ids, strict=True):
        one(ck, i)
        done.append((ck, i))
        sh = (summary(i).get("a_share") or {}).get(a.stop_habit) if a.stop_habit else None
        if a.stop_above is not None and sh is not None and sh > a.stop_above:
            stopped = i
            break
    if done[-1][1] != ids[-1]:  # the last snapshot is always probed
        one(cks[-1], ids[-1])
        done.append((cks[-1], ids[-1]))
    out: dict[str, Any] = {"set": PROBE_SET, "tag": "none", "k": a.k, "step_limit": C.STEP_LIMIT,
                           "acceptance": "the Lean gate, as the full measurement (authors_measure run)",
                           "stop_habit": a.stop_habit, "stop_above": a.stop_above, "stopped_at": stopped,
                           "skipped": [i for i in ids if i not in {x for _, x in done}], "snapshots": {}}  # fmt: skip
    for ck, i in done:
        t = summary(i)
        out["snapshots"][i] = {"ckpt": ck, **{k: t.get(k) for k in ("accepted_pct", "pass1", "proofs", "step_cap_pct",
                                                                     "a_share", "other_pct", "sites", "near_per_proof")}}  # fmt: skip
    _W2_WRITE(root / "probe.json", json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "sets":
        sets_main(argv[1:])
        return
    if argv and argv[0] == "strict106":
        strict106_main(argv[1:])
        return
    if argv and argv[0] == "probe":
        probe_main(argv[1:])
        return
    if argv and argv[0] in ("run", "estimate"):
        flags = {x.split("=")[0] for x in argv}
        if "--sets" not in flags:
            argv += ["--sets", ",".join(C.SETS)]
        if "--k" not in flags:
            argv += ["--k", "8"]
    if argv and argv[0] == "run" and "--dry-run" not in argv:
        install()
    AM.main(argv)


if __name__ == "__main__":
    main()
