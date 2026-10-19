"""Big-statement twins of curriculum rows: the same proof, bigger formulas (CPU only).

    bash code/experiments/current/thinking_traces_20261005/curriculum/build_size_twins.sh

runs, from the worktree root with `ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
PFG_MEM_GB=1.5`, for O=/scratch/cowifornia/nd-rl/think/runs/pools/size_twins:

    python -m $X.curriculum.size_twins --out $O build --set S --workers 7    # each set; L3_mini
    python -m $X.curriculum.size_twins --out $O lean                         # before L3_3p_mini
    python -m $X.curriculum.size_twins --out $O finish --workers 7
    python -m $X.curriculum.size_twins --out $O check     # preflight + extra overlap checks
    python -m $X.curriculum.size_twins --out $O report
    python -m $X.curriculum.size_twins --out $O think --set S --n 300 --workers 7   # analysis

Timeouts (`SEARCH_SECS`, `NEC_SECS`) are wall-clock limits, as in `pfgates`, so the share of
`nec_timeout` rows (and, rarely, of searched candidates that time out) depends on machine load.

**What a twin is.** The knob "bigger statements, same proofs" (knob proposal A1, 2026-10-16):
the RL pools' statements have at most 35 whitespace tokens, the sealed 13-20-line set's average
62. A twin applies one uniform substitution sigma (atom -> small formula) to the statement and
to every line of the row's reference and constructed proofs. Natural-deduction rules are
schematic, so sigma(proof) is a proof of sigma(statement) with the same rules, boxes, citations
and line count.

**sigma.** Each atom `a` of the row maps to an image built from one template of `TEMPLATES`,
with `a` itself as the anchor and partners `x`, `y` drawn from the other atoms of P Q R S (the
model's only atoms). Template groups and their weights (`WEIGHTS`) are fixed; they were set once
so that the twins' connective mix, padding (`X o X`) and double-negation rates come near
test_13_20's (report.json `features`). The choice is deterministic per row: `N_CAND` candidate
sigmas from `random.Random(sha256("size_twins|v1|cand|<key>"))`, where `<key>` is the row's
name (for a 3-premise rewrite, its L3 parent's name, so that both get the same candidates). The
target size is a draw from test_13_20's statement sizes restricted to 45-90 tokens
(`sha256("size_twins|v1|target|<key>")`). Candidates are ranked by distance to the target (then
by draw order) and tried in that order (a 3-premise rewrite tries its L3 parent twin's sigma
first, so that it stays its parent twin uncurried):

1. **Cheap gates**, per candidate: 45-90 whitespace tokens (the whole prompt, `THM ... PRF`);
   sigma is *independent*: over the 16 valuations of P Q R S the images take every combination
   of truth values (each image contingent, none determined by the others). Then for any
   formulas A, B over the row's atoms, sigma(A) entails sigma(B) iff A entails B, so the twin
   keeps the original's consistency, premise non-redundancy, non-tautological goal and every
   other truth-table fact (no ex falso from newly inconsistent premises, no tautological premise
   or goal). Still checked explicitly: no repeated premise, the goal not a premise, premises
   jointly satisfiable, no premise redundant; sigma(reference) and sigma(constructed) pass
   nd_verify with the original line counts; no substitution instance of a textbook72 theorem
   either way (`pfgates.textbook_match`); a classical-only original (no intuitionistic proof,
   `intuit_provable`) keeps a classical-only twin (a negation image can make it intuitionistic:
   `~~P |- P` under P -> ~Q is `~~~Q |- ~Q`).
2. **Shortest proof, the builders' measure.** `fastlen.fast_minlen`, the search
   `pfgates.certify` uses, on the twin with bound L - 1 (L = the original's `min_lines`) and the
   twin's constructed and reference proofs' formulas added to the search space, as
   `pfgates.shortest` adds the constructed proof's. The space (subformulas of the twin's
   statement, their negations and double negations, plus those formulas) contains sigma of the
   original's space, so the search is at least as thorough as the one that certified L. "No
   proof of <= L - 1 lines" plus sigma(reference) (L lines) certifies the twin's `min_lines` = L.
   A shorter proof found (`shortcut`: the substitution opened a shorter route, e.g. a case split
   or ex falso the original did not have) or a timeout (`SEARCH_SECS`) sends the row to its next
   candidate, up to `MAX_TRIES` searched candidates.
3. **Needed structures.** For each family the original needs (`families`), the builders'
   necessity search (`pfgates.necessity`: the structure forbidden, from L to L + 2 lines,
   `NEC_SECS`) on the twin. Verdict `same` (an L-line proof without the structure exists: the
   substitution made the structure avoidable at minimal length) rejects the candidate like a
   shortcut; `needed`, `longer` and `timeout` are recorded (`necessity`), and `families` keeps
   only `needed` ones, as the builders' rows do. `cert_status`: `ok` when every original family
   is `needed`, `nec_longer` / `nec_timeout` otherwise.

**Lean and the environment.** `lean` runs `hard712/run.py lean_flags` (state-env decomposition
with canonical names, then Lean 4) on both proofs, and replays each through `state_env.Env` with
`step_check.check` at every step (the per-step checker the guided sampler uses); a row is kept
only when all four pass.

**finish** writes, under `--out`, one file per source set, `<set>_sz.jsonl`, in the pools' row
format (`run.py final_row`) with name `sz_<orig name>`, `source` `size_twins_<set>`, and
provenance fields `orig_name`, `orig_prompt`, `orig_min_lines`, `orig_families`, `orig_tokens`,
`tokens`, `target_tokens`, `sigma`, `sigma_groups`, `sigma_rank`, `tries`, `cert_status`. A twin
is dropped if its class, as written or curried (`heldout.keys`), is in textbook72, Charles's
2026-10-02 release, a sealed test, a sealed composite part, a curriculum test split or an eval
subset, if it hits the curriculum's exclusion index, or if it repeats another twin's class (a
3-premise twin shares its curried class with its L3 parent twin by design, as the originals do).
For the test sets it also writes `<set>_orig_matched.jsonl`, the original rows whose twin
survived, in the same order.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import pickle
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from code.experiments.current.thinking_traces_20261005.curriculum import heldout as HO
from code.experiments.current.thinking_traces_20261005.hard712 import pfgates as GT
from code.experiments.current.thinking_traces_20261005.hard712 import run as RN

GC = GT.GC
FL = GT.FL
R = "/scratch/cowifornia/nd-rl/think/runs"
CURR = Path(f"{R}/pools/curriculum")
MINI = Path(f"{R}/pools/mini")
INDEX = CURR / "work" / "excl_index.pkl"
SEALED_PARTS = CURR / "work" / "sealed_parts.jsonl"
T1320 = f"{R}/pools/test_13_20/test_13_20.jsonl"
#: set name -> (source file, is a test set). L3_3p_mini rows are 3-premise rewrites of L3_mini rows
#: (`orig_name`), so L3_mini is built first and its sigma tried first for them.
SOURCES: dict[str, tuple[Path, bool]] = {
    "L2_mini": (MINI / "L2_mini.jsonl", False),
    "L3_mini": (MINI / "L3_mini.jsonl", False),
    "L3_3p_mini": (MINI / "L3_3p_mini.jsonl", False),
    "L2_test_1k": (CURR / "L2" / "test_1k.jsonl", True),
    "L3_test_1k": (CURR / "L3" / "test_1k.jsonl", True),
}
PARENT = {"L3_3p_mini": "L3_mini"}
#: Held-out and evaluation sets beyond `heldout.FILES` (Charles's release, textbook72) and
#: `heldout.SEALED`, as uncurry3.py, plus the sealed composites' parts.
HELD = {
    "charles_clean": f"{R}/pools/charles_1002/charles_clean.jsonl",
    "curriculum_L2_test_1k": str(CURR / "L2" / "test_1k.jsonl"),
    "curriculum_L3_test_1k": str(CURR / "L3" / "test_1k.jsonl"),
    "t1320_n1000": f"{R}/eval/matched/sets/t1320_n1000.jsonl",
    "t2130_n1000": f"{R}/eval/matched/sets/t2130_n1000.jsonl",
    "sealed_parts": str(SEALED_PARTS),
}

ATOMS = ("P", "Q", "R", "S")
SEED = "size_twins|v1"
LO_TOK, HI_TOK = 45, 90
N_CAND = 400
MAX_TRIES = 3
SEARCH_SECS = 60.0
NEC_SECS = 30.0
#: Image templates by group: `a` the atom itself, `x` and `y` two other atoms of P Q R S.
#: Token cost of an image = its whitespace tokens - 1 per occurrence of `a` in the statement.
TEMPLATES: dict[str, tuple[str, ...]] = {
    "id": ("a",),
    "neg": ("( ~ a )",),
    "dneg": ("( ~ ( ~ a ) )",),
    "pad": ("( a & a )", "( a v a )"),
    "bin": ("( a & x )", "( x & a )", "( a v x )", "( x v a )", "( a > x )", "( x > a )"),
    "imp": (
        "( a > x )",
        "( x > a )",
        "( ( x & x ) > a )",
        "( a > ( x v x ) )",
        "( ( a v a ) > x )",
        "( x > ( a & a ) )",
    ),
    "negbin": (
        "( a & ( ~ x ) )",
        "( ( ~ x ) & a )",
        "( a v ( ~ x ) )",
        "( ( ~ x ) v a )",
        "( a > ( ~ x ) )",
        "( ( ~ x ) > a )",
        "( ( ~ a ) & x )",
        "( ( ~ a ) v x )",
        "( ( ~ a ) > x )",
        "( x > ( ~ a ) )",
        "( ~ ( a & x ) )",
        "( ~ ( a v x ) )",
        "( ~ ( a > x ) )",
    ),
    "absorb": (
        "( a & ( a v x ) )",
        "( a v ( a & x ) )",
        "( a & ( x > a ) )",
        "( ( x > a ) & a )",
        "( ( x > x ) > a )",
        "( ( a > x ) > a )",
    ),
    "bin2": (
        "( a & ( x v y ) )",
        "( a v ( x & y ) )",
        "( ( a & x ) v y )",
        "( ( a v x ) & y )",
        "( a > ( x & y ) )",
        "( ( x v y ) > a )",
        "( ( a > x ) & y )",
        "( x > ( y > a ) )",
        "( a > ( x > y ) )",
        "( ( x > y ) > a )",
    ),
}
#: Group weights, set once on 600 L2/L3 mini rows (no search) to bring the twins' connective mix,
#: X o X padding and double-negation rates near test_13_20's; report.json `features` has the result.
WEIGHTS: dict[str, float] = {
    "id": 1.0,
    "neg": 0.7,
    "dneg": 0.3,
    "pad": 1.2,
    "bin": 2.0,
    "imp": 2.5,
    "negbin": 2.0,
    "absorb": 0.6,
    "bin2": 0.6,
}
FAMILIES = GT.FAMILIES
VALS4 = [dict(zip(ATOMS, b, strict=True)) for b in itertools.product([False, True], repeat=4)]


# ---------------------------------------------------------------------- sigma
def _h(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:16], 16)


def row_key(row: dict[str, Any], set_name: str) -> str:
    """The name sigma's draws are keyed on: a 3-premise rewrite uses its L3 parent's."""
    return row["orig_name"] if set_name in PARENT else row["name"]


def subst(text: str, sig: dict[str, str]) -> str:
    """Simultaneous substitution on a prompt or proof text (atoms are single tokens). The token
    after `:` in a proof line is its rule name, which can spell an atom (`R`, reiteration)."""
    toks = text.split()
    return " ".join(t if i and toks[i - 1] == ":" else sig.get(t, t) for i, t in enumerate(toks))


def row_atoms(prompt: str) -> list[str]:
    prem, goal = GC.parse_prompt(prompt)
    return sorted(set().union(*(GC.atoms(f) for f in [*prem, goal])))


def candidates(key: str, atoms: list[str]) -> list[tuple[dict[str, str], dict[str, str]]]:
    """N_CAND draws of (sigma, template group per atom), deterministic in `key`."""
    rng = random.Random(_h(f"{SEED}|cand|{key}"))
    groups = list(WEIGHTS)
    w = [WEIGHTS[g] for g in groups]
    out = []
    for _ in range(N_CAND):
        sig, grp = {}, {}
        for a in atoms:
            g = rng.choices(groups, w)[0]
            t = rng.choice(TEMPLATES[g])
            others = [b for b in ATOMS if b != a]
            x = rng.choice(others)
            y = rng.choice([b for b in others if b != x])
            sig[a] = " ".join({"a": a, "x": x, "y": y}.get(c, c) for c in t.split())
            grp[a] = g
        out.append((sig, grp))
    return out


_TGT: list[int] = []


def target_sizes() -> list[int]:
    """test_13_20's statement sizes (whitespace prompt tokens) within LO_TOK-HI_TOK, sorted."""
    if not _TGT:
        with open(T1320) as f:
            xs = [len(json.loads(x)["prompt"].split()) for x in f if x.strip()]
        _TGT.extend(sorted(x for x in xs if LO_TOK <= x <= HI_TOK))
    return _TGT


def target(key: str) -> int:
    t = target_sizes()
    return t[_h(f"{SEED}|target|{key}") % len(t)]


def _f(s: str) -> Any:
    return GC.parse_prompt(f"THM SEQ {s} PRF")[1]


def independent(sig: dict[str, str]) -> bool:
    """The images take all 2^k truth-value combinations over the 16 valuations of P Q R S."""
    fs = [_f(s) for s in sig.values()]
    seen = {tuple(bool(GC.ev(f, v)) for f in fs) for v in VALS4}
    return len(seen) == 2 ** len(fs)


def statement_gate(prompt: str) -> str | None:
    prem, goal = GC.parse_prompt(prompt)
    if goal in prem:
        return "goal_is_prem"
    if len(set(prem)) < len(prem):
        return "dup_prem"
    if prem and not any(all(GC.ev(p, v) for p in prem) for v in GC.VALS):
        return "inconsistent"
    if any(GC.entails(prem[:i] + prem[i + 1 :], goal) for i in range(len(prem))):
        return "redundant_prem"
    if not prem and not GC.entails([], goal):
        return "invalid"
    tm = GT.textbook_match(prem, goal)
    if tm:
        return "textbook_" + tm
    return None


def ranked(row: dict[str, Any], key: str) -> list[tuple[int, int, dict[str, str], dict[str, str]]]:
    """(distance to target, draw index, sigma, groups) of the distinct in-window candidates."""
    tgt = target(key)
    seen: set[str] = set()
    out = []
    for i, (sig, grp) in enumerate(candidates(key, row_atoms(row["prompt"]))):
        p = subst(row["prompt"], sig)
        w = len(p.split())
        if p in seen or not LO_TOK <= w <= HI_TOK:
            continue
        seen.add(p)
        out.append((abs(w - tgt), i, sig, grp))
    out.sort(key=lambda t: (t[0], t[1]))
    return out


# ---------------------------------------------------------------------- one row
def search(prompt: str, ref: str, gen: str, L: int) -> dict[str, Any]:
    extra = tuple(ln["f"] for ln in GC.parse_proof(gen)) + tuple(
        ln["f"] for ln in GC.parse_proof(ref)
    )
    r = FL.fast_minlen(prompt, bound=L - 1, time_limit=SEARCH_SECS, extra=extra)
    if r.get("timeout"):
        return dict(outcome="timeout", secs=round(r["secs"], 2), extra=extra)
    if r.get("error"):
        return dict(outcome="search_error", secs=round(r["secs"], 2), extra=extra)
    if r["min_lines_ub"] is not None:
        return dict(
            outcome="shortcut",
            found=r["min_lines_ub"],
            proof=r["proof"],
            secs=round(r["secs"], 2),
            extra=extra,
        )
    return dict(outcome="certified", lb=r["lb"], secs=round(r["secs"], 2), extra=extra)


def try_sigma(row: dict[str, Any], sig: dict[str, str]) -> dict[str, Any]:
    """Gates, shortest-proof search and necessity for one sigma -> attempt record (outcome
    'ok' or a rejection reason) with the twin's fields when ok."""
    L, C = row["min_lines"], row["gen_lines"]
    prompt = subst(row["prompt"], sig)
    ref = subst(row["reference_proof"], sig)
    gen = subst(row["gen_proof"], sig)
    att: dict[str, Any] = {"sigma": sig, "tokens": len(prompt.split())}
    if not independent(sig):
        return dict(att, outcome="dependent")
    reason = statement_gate(prompt)
    if reason:
        return dict(att, outcome="gate:" + reason)
    if row["attrs"]["classical"] and GC.intuit_provable(prompt):
        return dict(att, outcome="gate:classical_lost")
    ok, _, n = GT.verify_text(prompt + " " + ref)
    if not ok or n != L:
        return dict(att, outcome="ref_invalid")
    ok, _, n = GT.verify_text(prompt + " " + gen)
    if not ok or n != C:
        return dict(att, outcome="gen_invalid")
    t0 = time.time()
    s = search(prompt, ref, gen, L)
    att["search_secs"] = s["secs"]
    if s["outcome"] != "certified":
        att["outcome"] = s["outcome"]
        if s["outcome"] == "shortcut":
            att["found_lines"], att["found_proof"] = s["found"], s["proof"]
        return att
    nec: dict[str, Any] = {}
    for fam in row["families"]:
        t1 = time.time()
        v = GT.necessity(prompt, GT.NEC_KIND[fam], L, s["extra"], secs=NEC_SECS)
        nec[fam] = {
            "verdict": v["verdict"],
            "len": v.get("len"),
            "secs": round(time.time() - t1, 3),
        }
    att["necessity"] = nec
    att["secs"] = round(time.time() - t0, 2)
    if any(v["verdict"] == "same" for v in nec.values()):
        return dict(att, outcome="structure_avoidable")
    return dict(att, outcome="ok", prompt=prompt, reference_proof=ref, proof=gen)


def build_one(args: tuple[dict[str, Any], str, dict[str, str] | None]) -> dict[str, Any]:
    """One source row -> a twin candidate (status 'ok') or a drop record."""
    row, set_name, parent_sig = args
    t0 = time.time()
    key = row_key(row, set_name)
    base: dict[str, Any] = {
        "orig_name": row["name"],
        "set": set_name,
        "key_name": key,
        "target_tokens": target(key),
        "orig_tokens": len(row["prompt"].split()),
    }
    order = ranked(row, key)
    base["n_in_window"] = len(order)
    if parent_sig is not None:
        order = [(-1, -1, parent_sig, {})] + [o for o in order if o[2] != parent_sig]
    tries: list[dict[str, Any]] = []
    gate: collections.Counter[str] = collections.Counter()
    searched = 0
    for dist, idx, sig, grp in order:
        att: dict[str, Any]
        try:
            att = try_sigma(row, sig)
        except MemoryError:
            att = {"sigma": sig, "outcome": "memory"}
        except Exception as e:  # noqa: BLE001  (one bad candidate must not kill the run)
            att = {"sigma": sig, "outcome": f"error:{type(e).__name__}:{str(e)[:80]}"}
        if "search_secs" not in att and att["outcome"] != "memory":
            gate[att["outcome"]] += 1  # rejected before the search
            continue
        searched += 1
        att.update(rank=idx, dist=dist, groups=grp, from_parent=idx == -1)
        tries.append(att)
        if att["outcome"] == "ok":
            break
        if searched >= MAX_TRIES:
            break
    base["gate_rejects"] = dict(gate)
    base["tries"] = [
        {k: v for k, v in t.items() if k not in ("prompt", "proof", "reference_proof")}
        for t in tries
    ]
    base["secs"] = round(time.time() - t0, 2)
    if not tries:
        return dict(base, status="no_candidate")
    last = tries[-1]
    if last["outcome"] != "ok":
        outs = [t["outcome"] for t in tries]
        return dict(base, status="exhausted", last=last["outcome"], outcomes=outs)
    prompt = last["prompt"]
    k, pk = RN.class_keys(prompt)
    prem, goal = GC.parse_prompt(prompt)
    st = GT.structure(prompt, last["reference_proof"])
    nec = last["necessity"]
    fams = [f for f in row["families"] if nec[f]["verdict"] == "needed"]
    out = dict(
        base,
        status="ok",
        prompt=prompt,
        proof=last["proof"],
        reference_proof=last["reference_proof"],
        gen_lines=row["gen_lines"],
        min_lines=row["min_lines"],
        key=k,
        pkey=pk,
        n_prem=len(prem),
        n_irrelevant=GT.irrelevant(prem, goal),
        structure={kk: v for kk, v in st.items() if kk != "has"},
        structure_same=({kk: v for kk, v in st.items() if kk != "has"} == row["structure"]),
        families=fams,
        necessity=nec,
        cert_status=cert_status(nec),
        recipe=row["recipe"],
        seed=row["seed"],
        subst=row.get("subst"),
        discharged=row.get("discharged", 0),
        sigma=last["sigma"],
        sigma_groups=last["groups"],
        sigma_rank=last["rank"],
        from_parent=last["from_parent"],
        tokens=len(prompt.split()),
        orig_prompt=row["prompt"],
        orig_min_lines=row["min_lines"],
        orig_families=row["families"],
        orig_n_irrelevant=row["n_irrelevant"],
        orig_max_state=row["max_state"],
    )
    return out


def cert_status(nec: dict[str, Any]) -> str:
    """ok: every family the original needs is needed on the twin too (no proof without it within
    L + 2 lines); nec_longer: one has a proof without it 1-2 lines longer; nec_timeout: one
    necessity search timed out (not verified either way)."""
    vs = {v["verdict"] for v in nec.values()}
    if vs <= {"needed"}:
        return "ok"
    return "nec_longer" if "longer" in vs else "nec_timeout"


# ---------------------------------------------------------------------- build
def load_parent_sigmas(root: Path, set_name: str) -> dict[str, dict[str, str]]:
    if set_name not in PARENT:
        return {}
    p = root / PARENT[set_name] / "work" / "gated.jsonl"
    assert p.exists(), f"build {PARENT[set_name]} before {set_name}"
    return {r["orig_name"]: r["sigma"] for r in RN.jl(p)}


def cmd_build(a: argparse.Namespace) -> None:
    root = Path(a.out)
    work = root / a.set / "work"
    work.mkdir(parents=True, exist_ok=True)
    src, _ = SOURCES[a.set]
    rows = RN.jl(src)[: a.limit or None]
    psig = load_parent_sigmas(root, a.set)
    c: collections.Counter[str] = collections.Counter(source_rows=len(rows))
    done: set[str] = set()
    for name in ("gated.jsonl", "rejects.jsonl"):
        if (work / name).exists():
            prev = RN.read_complete(work / name)[0]
            done |= {r["orig_name"] for r in prev}
            c.update(r["status"] for r in prev)
    c["resumed"] = len(done)
    todo = [r for r in rows if r["name"] not in done]
    t0 = time.time()
    RN.log(f"{a.set}: {len(rows)} rows, {len(todo)} to do, {a.workers} workers")
    args = [(r, a.set, psig.get(r.get("orig_name", ""))) for r in todo]
    with (
        mp.get_context("spawn").Pool(
            a.workers, initializer=RN._init, initargs=(RN.MEM_GB,), maxtasksperchild=50
        ) as pool,
        open(work / "gated.jsonl", "a") as fg,
        open(work / "rejects.jsonl", "a") as fr,
    ):
        for i, r in enumerate(pool.imap_unordered(build_one, args, chunksize=1), 1):
            c[r["status"]] += 1
            (fg if r["status"] == "ok" else fr).write(json.dumps(r) + "\n")
            if i % 25 == 0:
                fg.flush()
                fr.flush()
                RN.log(f"{a.set}: {i}/{len(todo)} {time.time() - t0:.0f}s {dict(c)}")
    c["secs"] = round(time.time() - t0)
    RN.wjson(work / "build_report.json", dict(c))
    RN.log(f"{a.set} build done: {dict(c)}")


# ---------------------------------------------------------------------- lean + env
def env_flags(rows: list[dict[str, Any]], field: str) -> list[str | None]:
    """Replay `field` in state_env.Env (canonical names) with step_check.check before every step:
    None if every step is accepted by both, else the first reason."""
    from code.thinking.dan import bare, load

    D = load()
    out: list[str | None] = []
    for r in rows:
        try:
            with bare():
                toks = D.state_env.canonicalise(r["prompt"], r[field])
                env = D.state_env.Env(r["prompt"], canon=True)
                why = None
                for act in D.state_env.split_actions(toks):
                    sc = D.step_check.check(env, act)
                    if sc is not None:
                        why = f"step_check:{sc}"
                        break
                    ok, w = env.apply(act)
                    if not ok:
                        why = f"env:{w}"
                        break
                if why is None and not env.done:
                    why = "env:not_done"
                if why is None and env.text != toks:
                    why = "env:reassembly"
        except Exception as e:  # noqa: BLE001  (counted as a rejection)
            why = f"error:{type(e).__name__}"
        out.append(why)
    return out


def cmd_lean(a: argparse.Namespace) -> None:
    root = Path(a.out)
    os.environ.setdefault("LEAN_GATE_LOG", str(root / "lean_gate.jsonl"))
    for s in a.sets or list(SOURCES):
        work = root / s / "work"
        if not (work / "gated.jsonl").exists():
            continue
        t0 = time.time()
        rows = RN.jl(work / "gated.jsonl")
        g = RN.lean_flags(rows, "proof")
        ref = RN.lean_flags(rows, "reference_proof")
        eg = env_flags(rows, "proof")
        er = env_flags(rows, "reference_proof")
        for row, x, y, u, v in zip(rows, g, ref, eg, er, strict=True):
            row["lean_ok"], row["lean_ref_ok"] = x, y
            row["env_ok"], row["env_ref_ok"] = u is None, v is None
            row["env_why"] = u or v
        RN.wjl(work / "gated_lean.jsonl", rows)
        rep = dict(
            n=len(rows),
            lean_constructed_ok=sum(g),
            lean_reference_ok=sum(ref),
            env_constructed_ok=sum(x is None for x in eg),
            env_reference_ok=sum(x is None for x in er),
            env_reasons=dict(collections.Counter(x for x in eg + er if x)),
            secs=round(time.time() - t0),
        )
        RN.wjson(work / "lean_report.json", rep)
        RN.log(f"{s}: {json.dumps(rep)}")


# ---------------------------------------------------------------------- finish
def final(r: dict[str, Any], set_name: str, tok: Any, D: Any) -> dict[str, Any]:
    fam = r["families"]
    row = RN.final_row(dict(r, families=fam or ["depth2"]), f"sz_{r['orig_name']}", tok, D)
    row["family"], row["families"] = (row["family"] if fam else None), fam
    row["source"] = f"size_twins_{set_name}"
    for k in (
        "orig_name",
        "orig_prompt",
        "orig_min_lines",
        "orig_families",
        "orig_n_irrelevant",
        "orig_max_state",
        "orig_tokens",
        "tokens",
        "target_tokens",
        "sigma",
        "sigma_groups",
        "sigma_rank",
        "from_parent",
        "cert_status",
        "structure_same",
        "env_ok",
        "env_ref_ok",
    ):
        row[k] = r[k]
    row["tries"] = len(r["tries"])
    row["try_outcomes"] = [t["outcome"] for t in r["tries"]]
    return row


def screen(
    rows: list[dict[str, Any]],
    hidx: dict[str, set[str]],
    idx: dict[str, Any],
    seen: dict[str, str],
    set_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Lean / env, held-out, index and duplicate drops -> (kept rows, counts and hit tallies)."""
    c: collections.Counter[str] = collections.Counter()
    held: collections.Counter[str] = collections.Counter()
    by_file: collections.Counter[str] = collections.Counter()
    keep = []
    parent = PARENT.get(set_name)
    for r in rows:
        if not (r["lean_ok"] and r["lean_ref_ok"]):
            c["lean_rejected"] += 1
            continue
        if not (r["env_ok"] and r["env_ref_ok"]):
            c["env_rejected"] += 1
            continue
        h = HO.hits(hidx, r["prompt"])
        if h:
            c["held_out"] += 1
            held.update(h)
            continue
        h = idx["keys"].get(r["key"], set()) | idx["pkeys"].get(r["pkey"], set())
        if h:
            c["index"] += 1
            by_file.update(h)
            continue
        ck = HO.keys(r["prompt"])
        clash = {seen[k] for k in ck if k in seen}
        # a 3-premise twin shares its curried class with its own L3 parent twin, by design
        clash -= {f"{parent}:{r['key_name']}"} if parent else set()
        if clash:
            c["dup_class"] += 1
            continue
        for k in ck:
            seen.setdefault(k, f"{set_name}:{r['orig_name']}")
        keep.append(r)
    return keep, dict(counts=dict(c), held_out=dict(held), index_by_file=dict(by_file))


def cmd_finish(a: argparse.Namespace) -> None:
    from code.thinking.dan import load
    from code.thinking.tok import ThinkTokenizer

    t0 = time.time()
    root = Path(a.out)
    D, tok = load(), ThinkTokenizer()
    hidx = HO.heldout_index({**HO.FILES, **HO.SEALED, **HELD})
    with open(INDEX, "rb") as f:
        idx = pickle.load(f)
    seen: dict[str, str] = {}
    rep: dict[str, Any] = {"held_out_sets": sorted({**HO.FILES, **HO.SEALED, **HELD})}
    for s, (src, is_test) in SOURCES.items():
        work = root / s / "work"
        if not (work / "gated_lean.jsonl").exists():
            continue
        rows = RN.jl(work / "gated_lean.jsonl")
        order = {r["name"]: i for i, r in enumerate(RN.jl(src))}
        rows.sort(key=lambda r: order[r["orig_name"]])
        keep, scr = screen(rows, hidx, idx, seen, s)
        with mp.get_context("spawn").Pool(a.workers) as p:
            fs = p.map(RN._feats, keep, chunksize=20)
        for r, (t, pf) in zip(keep, fs, strict=True):
            r["attrs"], r["gap_thm"], r["gap_proof"] = RN.attrs(r, t, pf), t, pf
        out = [final(r, s, tok, D) for r in keep]
        assert len({r["name"] for r in out}) == len(out)
        RN.wjl(root / f"{s}_sz.jsonl", out)
        if is_test:
            names = {r["orig_name"] for r in out}
            RN.wjl(root / f"{s}_orig_matched.jsonl", [r for r in RN.jl(src) if r["name"] in names])
        rep[s] = dict(
            build=RN.rjson(work / "build_report.json"),
            lean=RN.rjson(work / "lean_report.json"),
            screen=scr,
            n_out=len(out),
        )
        RN.log(f"{s}: {len(rows)} certified, {len(out)} written; {scr}")
    rep["finish_secs"] = round(time.time() - t0)
    RN.wjson(root / "finish.json", rep)


# ---------------------------------------------------------------------- check (seal)
def cmd_check(a: argparse.Namespace) -> None:
    """The GRPO v2 preflight on the twin training files (as build_mini.py ran it on the mini
    subsets), and the overlap checks it does not cover: every twin file against every other, the
    twin test sets against all training data, and every twin against the held-out sets and the
    sealed composites' parts up to curried normal form."""
    root = Path(a.out)
    wt = Path(__file__).resolve().parents[5]
    mod = "code.experiments.current.thinking_traces_20261005.grpo_v2.preflight"
    res: dict[str, Any] = {}
    for tag, (x, y) in {
        "preflight": ("L2_mini_sz", "L3_mini_sz"),
        "preflight_extra": ("L2_mini_sz", "L3_3p_mini_sz"),
    }.items():
        p = subprocess.run(
            [sys.executable, "-m", mod, str(root / f"{x}.jsonl"), str(root / f"{y}.jsonl")],
            cwd=wt,
            capture_output=True,
            text=True,
            timeout=3000,
        )
        (root / f"{tag}.log").write_text(p.stdout + p.stderr)
        lines = p.stdout.splitlines()
        res[tag] = dict(
            exit=p.returncode,
            verdict=lines[-1] if lines else "",
            ok_lines=sum(x.startswith("ok ") for x in lines),
            bad_lines=[x.strip() for x in lines if x.lstrip().startswith("BAD")],
        )
    # pairwise and against training data / held-out sets, by preflight's class rule and heldout.keys
    from code.experiments.current.thinking_traces_20261005.grpo_v2 import preflight as PF

    files = {s: root / f"{s}_sz.jsonl" for s in SOURCES}
    tw = {s: RN.jl(p) for s, p in files.items() if p.exists()}
    cls = {s: PF.class_set(rs) for s, rs in tw.items()}
    ho = {s: set().union(*(HO.keys(r["prompt"]) for r in rs)) for s, rs in tw.items()}
    pairs = {}
    for x, y in itertools.combinations(tw, 2):
        n1, n2 = len(cls[x] & cls[y]), len(ho[x] & ho[y])
        if {x, y} == {"L3_mini", "L3_3p_mini"}:
            # by design: each 3-premise twin is its L3 parent twin uncurried
            par = {r["orig_name"] for r in tw["L3_mini"]}
            kids = [r for r in tw["L3_3p_mini"] if r["orig_name"].removeprefix("u3L3_") in par]
            pairs[f"{x} vs {y}"] = dict(
                classes=n1,
                heldout_keys=n2,
                note=(
                    f"expected: {len(kids)} 3-premise twins whose L3 parent twin is in "
                    "L3_mini share its curried class"
                ),
            )
            continue
        pairs[f"{x} vs {y}"] = dict(classes=n1, heldout_keys=n2)
    res["twin_pairs"] = pairs
    train = {
        "curriculum L2 pool": CURR / "L2" / "pool.jsonl",
        "curriculum L3 pool": CURR / "L3" / "pool.jsonl",
        "curriculum_3p L2_3p": Path(f"{R}/pools/curriculum_3p/L2_3p.jsonl"),
        "curriculum_3p L3_3p": Path(f"{R}/pools/curriculum_3p/L3_3p.jsonl"),
    }
    tr = {k: RN.jl(p) for k, p in train.items()}
    trc = {k: PF.class_set(rs) for k, rs in tr.items()}
    trh = {k: set().union(*(HO.keys(r["prompt"]) for r in rs)) for k, rs in tr.items()}
    tests = [s for s in tw if SOURCES[s][1]]
    trains = [s for s in tw if not SOURCES[s][1]]
    vs_train = {}
    for s in tests:
        for k in tr:
            vs_train[f"{s} vs {k}"] = dict(
                classes=len(cls[s] & trc[k]), heldout_keys=len(ho[s] & trh[k])
            )
        for t in trains:
            vs_train[f"{s} vs {t} (twins)"] = dict(
                classes=len(cls[s] & cls[t]), heldout_keys=len(ho[s] & ho[t])
            )
    res["tests_vs_training"] = vs_train
    hidx = HO.heldout_index({**HO.FILES, **HO.SEALED, **HELD})
    held = {}
    for s, rs in tw.items():
        hits: collections.Counter[str] = collections.Counter()
        for r in rs:
            hits.update(HO.hits(hidx, r["prompt"]))
        held[s] = dict(hits)
    res["heldout_curried"] = held
    # textbook72 substitution instances either way (the statement gate), re-checked on the outputs
    tbm = {
        s: sum(bool(GT.textbook_match(*GC.parse_prompt(r["prompt"]))) for r in rs)
        for s, rs in tw.items()
    }
    res["textbook_instances"] = tbm
    bad = (
        any(
            v["exit"] != 0
            and v["bad_lines"]
            and not all("rows outside" in b for b in v["bad_lines"])
            for v in (res["preflight"], res["preflight_extra"])
        )
        or any(v["classes"] or v["heldout_keys"] for k, v in pairs.items() if "note" not in v)
        or any(v["classes"] or v["heldout_keys"] for v in vs_train.values())
        or any(held.values())
        or any(tbm.values())
    )
    res["sealed"] = not bad
    RN.wjson(root / "check.json", res)
    print(json.dumps(res, indent=1))


# ---------------------------------------------------------------------- report
def hist(xs: list[Any], width: int | None = None) -> dict[str, int]:
    if width:
        xs = [width * (x // width) for x in xs]
    return {str(k): v for k, v in sorted(collections.Counter(xs).items())}


def qs(xs: list[float]) -> dict[str, float]:
    s = sorted(xs)
    if not s:
        return {}

    def q(p: float) -> float:
        return s[min(len(s) - 1, int(p * len(s)))]

    return dict(
        n=len(s),
        mean=round(statistics.mean(s), 1),
        p10=q(0.1),
        p50=q(0.5),
        p90=q(0.9),
        min=s[0],
        max=s[-1],
    )


def stmt_feats(prompt: str) -> dict[str, Any]:
    prem, goal = GC.parse_prompt(prompt)
    allf = [*prem, goal]
    sub = [g for f in allf for g in GC.subs(f)]
    out = {f"n_{op}": sum(g[0] == op for g in sub) for op in ("not", "and", "or", "imp")}
    out["n_conn"] = sum(out.values())
    out["n_atoms"] = len(set().union(*(GC.atoms(f) for f in allf)))
    out["atom_occ"] = sum(g[0] == "atom" for g in sub)
    out["fdepth"] = max(GC.fdepth(f) for f in allf)
    out["n_degenerate"] = sum(g[0] in ("and", "or", "imp") and g[1] == g[2] for g in sub)
    out["n_dneg"] = sum(g[0] == "not" and g[1][0] == "not" for g in sub)
    out["n_prem"] = len(prem)
    out["tokens"] = len(prompt.split())
    return out


def feat_summary(rows: list[dict[str, Any]], with_irr: bool = True) -> dict[str, Any]:
    fs = [stmt_feats(r["prompt"]) for r in rows]
    n = max(1, len(fs))
    tot = {op: sum(f[f"n_{op}"] for f in fs) for op in ("not", "and", "or", "imp")}
    tc = max(1, sum(tot.values()))
    out: dict[str, Any] = {
        "n": len(rows),
        "tokens_mean": round(sum(f["tokens"] for f in fs) / n, 1),
        "connectives_mean": round(sum(f["n_conn"] for f in fs) / n, 2),
        "connective_mix": {op: round(v / tc, 3) for op, v in tot.items()},
        "atoms": hist([f["n_atoms"] for f in fs]),
        "atom_occurrences_mean": round(sum(f["atom_occ"] for f in fs) / n, 2),
        "formula_depth_mean": round(sum(f["fdepth"] for f in fs) / n, 2),
        "degenerate_per_row": round(sum(f["n_degenerate"] for f in fs) / n, 2),
        "has_degenerate": round(sum(f["n_degenerate"] > 0 for f in fs) / n, 3),
        "dneg_per_row": round(sum(f["n_dneg"] for f in fs) / n, 2),
        "has_dneg": round(sum(f["n_dneg"] > 0 for f in fs) / n, 3),
        "premises": hist([f["n_prem"] for f in fs]),
    }
    if with_irr:
        irr = [
            r["n_irrelevant"]
            if "n_irrelevant" in r
            else GT.irrelevant(*GC.parse_prompt(r["prompt"]))
            for r in rows
        ]
        out["irrelevant_mean"] = round(sum(irr) / n, 2)
        out["irrelevant"] = hist(irr)
    if rows and "max_state" in rows[0]:
        out["max_state"] = qs([r["max_state"] for r in rows if r.get("max_state") is not None])
    if rows and "attrs" in rows[0]:
        keys = (
            "classical",
            "ex_falso",
            "reductio",
            "nested_neg",
            "case_split",
            "no_prem",
            "goal_ho",
            "depth2plus",
        )
        out["attrs"] = {k: round(sum(bool(r["attrs"][k]) for r in rows) / n, 3) for k in keys}
        out["families"] = hist([",".join(r["families"]) or "none" for r in rows])
        out["box_depth_mean"] = round(sum(r["gap_proof"]["p_box_depth"] for r in rows) / n, 2)
        out["ref_steps_mean"] = round(sum(r["ref_steps"] for r in rows) / n, 2)
    return out


ATTR_KEYS = (
    "classical",
    "ex_falso",
    "reductio",
    "nested_neg",
    "case_split",
    "no_prem",
    "goal_ho",
    "depth2plus",
    "bicond",
)


def attr_changes(out: list[dict[str, Any]], orig: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Per mix attribute: twins where it differs from the original's (gained / lost)."""
    res = {}
    for k in ATTR_KEYS:
        gained = sum(r["attrs"][k] and not orig[r["orig_name"]]["attrs"][k] for r in out)
        lost = sum(orig[r["orig_name"]]["attrs"][k] and not r["attrs"][k] for r in out)
        res[k] = dict(gained=gained, lost=lost)
    return res


def cmd_report(a: argparse.Namespace) -> None:
    root = Path(a.out)
    fin = RN.rjson(root / "finish.json")
    chk = RN.rjson(root / "check.json") if (root / "check.json").exists() else None
    rep: dict[str, Any] = {
        "design": dict(
            seed=SEED,
            tokens_window=[LO_TOK, HI_TOK],
            n_candidates=N_CAND,
            max_tries=MAX_TRIES,
            search_secs=SEARCH_SECS,
            nec_secs=NEC_SECS,
            templates=TEMPLATES,
            weights=WEIGHTS,
            target="test_13_20 statement sizes within the window",
            token_unit="whitespace tokens of the whole prompt (THM ... PRF)",
        ),
        "held_out_sets": fin["held_out_sets"],
        "sets": {},
        "reference": {},
        "check": chk,
    }
    refs = {
        "test_13_20": T1320,
        "t1320_n1000": HELD["t1320_n1000"],
        "t2130_n1000": HELD["t2130_n1000"],
    }
    for k, p in refs.items():
        rs = RN.jl(p)
        rep["reference"][k] = dict(
            tokens=qs([len(r["prompt"].split()) for r in rs]),
            tokens_hist5=hist([len(r["prompt"].split()) for r in rs], 5),
            max_state_hist20=hist([r["max_state"] for r in rs if r.get("max_state")], 20),
            features=feat_summary(rs[:1000]),
        )
    for s, (src, _) in SOURCES.items():
        if s not in fin:
            continue
        work = root / s / "work"
        gated = RN.jl(work / "gated.jsonl")
        rejects = RN.jl(work / "rejects.jsonl")
        out = RN.jl(root / f"{s}_sz.jsonl")
        names = {r["orig_name"] for r in out}
        orig_all = RN.jl(src)
        orig = [r for r in orig_all if r["name"] in names]
        tries = [t for r in gated + rejects for t in r["tries"]]
        first = [r["tries"][0]["outcome"] for r in gated + rejects if r["tries"]]
        gate: collections.Counter[str] = collections.Counter()
        for r in gated + rejects:
            gate.update(r["gate_rejects"])
        b = fin[s]
        rep["sets"][s] = dict(
            source=str(src),
            source_rows=len(orig_all),
            out=len(out),
            build=b["build"],
            drops=dict(
                exhausted_last_outcome=dict(
                    collections.Counter(
                        r.get("last") for r in rejects if r["status"] == "exhausted"
                    )
                ),
                no_candidate=sum(r["status"] == "no_candidate" for r in rejects),
                **b["screen"]["counts"],
            ),
            screen=b["screen"],
            lean=b["lean"],
            searched_candidates=dict(collections.Counter(t["outcome"] for t in tries)),
            first_candidate=dict(collections.Counter(first)),
            gate_rejects_before_search=dict(gate),
            min_lines_preserved=sum(r["min_lines"] == r["orig_min_lines"] for r in out),
            min_lines=hist([r["min_lines"] for r in out]),
            cert_status=hist([r["cert_status"] for r in out]),
            structure_same=sum(r["structure_same"] for r in out),
            from_parent=sum(r["from_parent"] for r in out),
            tries=hist([r["tries"] for r in out]),
            tokens=dict(
                twin=qs([r["tokens"] for r in out]),
                orig=qs([r["orig_tokens"] for r in out]),
                target=qs([r["target_tokens"] for r in out]),
                twin_hist5=hist([r["tokens"] for r in out], 5),
                orig_hist5=hist([r["orig_tokens"] for r in out], 5),
            ),
            max_state=dict(
                twin=qs([r["max_state"] for r in out]),
                orig=qs([r["max_state"] for r in orig]),
                twin_hist20=hist([r["max_state"] for r in out], 20),
                orig_hist20=hist([r["max_state"] for r in orig], 20),
                ratio_mean=round(
                    statistics.mean(r["max_state"] / r["orig_max_state"] for r in out), 2
                )
                if out
                else None,
            ),
            sigma_groups=hist([g for r in out for g in r["sigma_groups"].values()]),
            attr_changes=attr_changes(out, {r["name"]: r for r in orig}),
            families_equal=sum(r["families"] == r["orig_families"] for r in out),
            necessity_verdicts=hist(
                [f"{f}:{v['verdict']}" for r in out for f, v in r["necessity"].items()]
            ),
            features=dict(twin=feat_summary(out), orig=feat_summary(orig)),
            secs_per_row=qs([r["secs"] for r in gated + rejects]),
        )
    rep["think_cost"] = {
        f.stem.removeprefix("think_cost_"): RN.rjson(f)["summary"]
        for f in sorted(root.glob("think_cost_*.json"))
    }
    RN.wjson(root / "report.json", rep)
    print(
        json.dumps(
            {s: dict(out=v["out"], drops=v["drops"]) for s, v in rep["sets"].items()}, indent=1
        )
    )


# ---------------------------------------------------------------------- think cost
def _think_one(job: tuple[str, str, str]) -> dict[str, Any]:
    """Honest search traces (`searchtrace.trace_for`, the Stage-1 trace format) for one prompt at
    effort 256 (with the 0 / 64 / 256 level labels) and at 64: think tokens per decision step,
    the share of decision steps whose action needs more than 64 think tokens (`min_level`), and
    whether the search proves the theorem."""
    from code.thinking import searchtrace as ST

    pid, prompt, which = job
    out: dict[str, Any] = {"pid": pid, "which": which}
    for level in (256, 64):
        try:
            rec, infos, _ = ST.trace_for(pid, prompt, level, labels=(level == 256))
        except Exception as e:  # noqa: BLE001  (ProofFail or a generator error: counted)
            out[f"ok{level}"] = False
            out[f"err{level}"] = type(e).__name__
            continue
        dec = [i for i in infos if i.get("think_tokens", 0) > 0 or i.get("kind") == "decision"]
        out[f"ok{level}"] = True
        out[f"steps{level}"] = len(rec["steps"])
        out[f"think{level}"] = [i["think_tokens"] for i in dec]
        out[f"state{level}"] = max(i["state_tokens"] for i in infos)
        if level == 256:
            out["min_level"] = [st.get("min_level") for st in rec["steps"] if st.get("think")]
    return out


def cmd_think(a: argparse.Namespace) -> None:
    root = Path(a.out)
    rows = RN.jl(root / a.set / "work" / "gated.jsonl")
    rows.sort(key=lambda r: _h(f"{SEED}|think|{r['orig_name']}"))
    rows = rows[: a.n]
    jobs = [(r["orig_name"], r["orig_prompt"], "orig") for r in rows]
    jobs += [(r["orig_name"], r["prompt"], "twin") for r in rows]
    with mp.get_context("spawn").Pool(a.workers) as p:
        res = p.map(_think_one, jobs, chunksize=4)
    summ: dict[str, Any] = {"set": a.set, "n_pairs": len(rows)}
    for w in ("orig", "twin"):
        rs = [r for r in res if r["which"] == w]
        t256 = [t for r in rs if r.get("ok256") for t in r["think256"]]
        ml = [m for r in rs if r.get("ok256") for m in r["min_level"] if m is not None]
        summ[w] = dict(
            proved_256=round(sum(bool(r.get("ok256")) for r in rs) / max(1, len(rs)), 3),
            proved_64=round(sum(bool(r.get("ok64")) for r in rs) / max(1, len(rs)), 3),
            steps_256=qs([r["steps256"] for r in rs if r.get("ok256")]),
            think_tokens_per_thinking_step_256=qs(t256),
            think_tokens_per_proof_256=qs([sum(r["think256"]) for r in rs if r.get("ok256")]),
            share_steps_needing_over_64=round(sum(m > 64 for m in ml) / max(1, len(ml)), 3),
            min_level=hist(ml),
            max_state_tokens_256=qs([r["state256"] for r in rs if r.get("ok256")]),
        )
    RN.wjson(root / f"think_cost_{a.set}.json", dict(summary=summ, rows=res))
    print(json.dumps(summ, indent=1))


# ---------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--set", choices=list(SOURCES), required=True)
    b.add_argument("--workers", type=int, default=8)
    b.add_argument("--limit", type=int, default=0, help="only the first N source rows (a pilot)")
    ln = sub.add_parser("lean")
    ln.add_argument("--sets", nargs="*", default=None)
    f = sub.add_parser("finish")
    f.add_argument("--workers", type=int, default=8)
    sub.add_parser("check")
    sub.add_parser("report")
    t = sub.add_parser("think", help="search-trace think cost, originals vs twins (analysis)")
    t.add_argument("--set", choices=list(SOURCES), required=True)
    t.add_argument("--n", type=int, default=200)
    t.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    {
        "build": cmd_build,
        "lean": cmd_lean,
        "finish": cmd_finish,
        "check": cmd_check,
        "report": cmd_report,
        "think": cmd_think,
    }[a.cmd](a)


if __name__ == "__main__":
    main()
