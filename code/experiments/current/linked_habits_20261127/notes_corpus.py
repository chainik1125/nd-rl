#!/usr/bin/env python3
"""Linked-habits pretraining corpora: the two-author corpora with premise notes added (linked_habits_20261127, CPU).

    python notes_corpus.py limits [--limit N]             # the context / line / name check -> LH/data/limits.json
    python notes_corpus.py table                          # the form table (notes.write_table) -> LH/data/note_table.json
    python notes_corpus.py build [--limit N] [--force]    # LH/data/{l19,l4,l0}.jsonl.gz + <name>.manifest.json

Sources (read only): TA/data/authors_half.jsonl.gz (l19, l4) and TA/data/control_half.jsonl.gz (l0), 155,000 records
each, same keys in the same order. Every record keeps its fields; the proof gains one note per premise (notes.py) right
after the PR block, and `text`, `n_lines`, `rules` follow the new proof. New fields:

  notes       per premise, in premise order: {"cls", "form", "side"}  (side "A" or "B": whose form it is)
  note_forms  {class: "A" | "B"} for the classes present
  lh_corpus   l19 | l4 | l0
  ta_proof    the TA proof the notes were added to (`src_proof` stays the K12 source proof)

Which side a (record, class) gets (SPEC.md "Three corpora"):
  l19   the record's author (`author`): every class in A's form for author A, B's for author B
  l4    a fair coin per (record, class): A iff u(20261127, "coin", key, class) < 0.5, u = the first 8 bytes of
        sha256("20261127:coin:<key>:<class>") / 2^64 (authors_corpus.u01); independent of author, tag and each other
  l0    the same coin as l4 (same key, same class)
Two premises of one class in one record get the same form. Records without premises get no note.

Validation (written into each manifest): on every record, Dan's state_env.decompose(canon=True) with every state and
action encoded by the tag tokenizer under the record's shown tag (as authors_launch.authors_load does), nd_verify,
the premise lines unchanged, notes.read_notes returning exactly the written notes (no late note, no premise without a
note), notes.strip_notes giving back the TA proof, and authors_profile.profile on the new proof equal to profile on
the TA proof (order / reit / conj / conj_all / route / sigma / bundle; `dead` = the number of notes). Lean 4
(nd2lean.translate + lean_check, authors_corpus.lean_verdicts, 8 workers) on a seeded sample per corpus: 100 records
per (class, form) present, plus random records up to LEAN_MIN.
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
W2 = HERE.parent / "two_authors_20261111"
AT = HERE.parent / "author_tags_20261110"
for _p in (str(AT), str(W2), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import notes as N  # noqa: E402

TA = Path(os.environ.get("TA", "/scratch/cowifornia/nd-rl/two_authors"))
SOURCES = {
    "authors_half": TA / "data" / "authors_half.jsonl.gz",
    "control_half": TA / "data" / "control_half.jsonl.gz",
}
CORPORA = {
    "l19": ("authors_half", "author"),
    "l4": ("authors_half", "coin"),
    "l0": ("control_half", "coin"),
}
DATA = N.LH / "data"
LEAN_CACHE = N.LH / "cache" / "lean_verdicts.jsonl"
SEED = N.SEED
LEAN_PER_CELL, LEAN_MIN = 100, 3000
PROFILE_KEYS = ("order", "reit", "conj", "conj_all", "route", "route_knock", "sigma", "bundle", "joint", "n_room",
                "reit_fits", "depth")  # fmt: skip
# Limits of the pretraining / sampling path (see `limits`): Dan's 13e41bbd and the wrappers.
CTX_NOMINAL = (
    1024  # best_model.ALiBiGPT(max_len=1024): stored in cfg, never enforced by the ALiBi forward
)
TOK_BUDGET = (
    128 * 144
)  # state_train_best.token_batches asserts every (state, action) pair <= the token budget
MAX_ACTION, MAX_STEPS = (
    256,
    48,
)  # state_sample.env_generate / grpo_state.env_rollouts / authors_measure defaults
CAP_LINES = 12  # tag_pretrain passes --cap 12; state_train.load / tag_load / authors_load assert n_lines <= cap


# ============================================================================ draws
def u01(*parts: Any) -> float:
    """authors_corpus.u01: uniform [0, 1) from the first 8 bytes of sha256(':'.join(parts))."""
    h = hashlib.sha256(":".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def coin(key: str, cls: str, seed: int = SEED) -> str:
    """The fair coin of (record, class) shared by l4 and l0."""
    return "A" if u01(seed, "coin", key, cls) < 0.5 else "B"


def sides_for(corpus: str, rec: dict[str, Any], classes: list[str]) -> dict[str, str]:
    """{class: side} for the classes present in a record of `corpus`."""
    rule = CORPORA[corpus][1]
    if rule == "author":
        if rec["author"] not in ("A", "B"):
            raise ValueError(f"record {rec.get('key')}: author {rec.get('author')!r}")
        return {c: rec["author"] for c in classes}
    return {c: coin(rec["key"], c) for c in classes}


# ============================================================================ worker
_W: dict[str, Any] = {}


def _tools() -> dict[str, Any]:
    if not _W:
        import tag_core as T

        cls = T.install_dan()
        from nd_verify import verify_text  # type: ignore[import-not-found]
        from state_env import decompose  # type: ignore[import-not-found]

        _W.update(tok=cls(), decompose=decompose, verify=verify_text, T=T)
    return _W


def state_lengths(prompt: str, proof: str, tag: str) -> dict[str, int]:
    """Decompose (canon) and encode every state and action as the tagged loader does. -> {steps, state, pair, action}
    maxima in tokens (state ids include the tag token and <act>; action ids include <eos>). Raises on any failure."""
    W = _tools()
    tok = W["tok"]
    steps, _toks, _env = W["decompose"](prompt, proof, canon=True)
    tok.author_tag = tag
    try:
        ms = mp = ma = 0
        for st, act, _hs in steps:
            s = len(tok.encode_toks(st))
            a = len(tok.encode_toks(act)) + 1
            ms, ma, mp = max(ms, s), max(ma, a), max(mp, s + a)
    finally:
        tok.author_tag = None
    return {"steps": len(steps), "state": ms, "pair": mp, "action": ma}


def _limits_one(args: tuple[str, str, str]) -> dict[str, Any]:
    """One record: lengths with no note and with every premise noted in each menu / junk form."""
    prompt, proof, _key = args
    out: dict[str, Any] = {"n_prem": len(N.premises(proof)), "lines": proof.split().count(";")}
    for form in ("none", *N.NOTE_MENU, *N.JUNK_FORMS):
        try:
            if form == "none":
                p = proof
            else:
                p = N.add_notes(prompt, proof, {c: form for c in N.CLASSES})
            out[form] = state_lengths(prompt, p, "A")
        except Exception as e:  # noqa: BLE001
            out[form] = {"err": f"{type(e).__name__}: {e}"}
    return out


def _profile_cmp(prompt: str, old: str, new: str, n_notes: int) -> dict[str, bool]:
    import authors_profile as AP

    a, b = AP.profile(prompt, old), AP.profile(prompt, new)
    res = {"parsed": bool(a.get("ok")) and bool(b.get("ok"))}
    if not res["parsed"]:
        return res
    for k in PROFILE_KEYS:
        res[k] = a.get(k) == b.get(k)
    res["dead_is_notes"] = b["dead"] == a["dead"] + n_notes
    res["lines_plus_notes"] = b["lines"] == a["lines"] + n_notes
    return res


def process(args: tuple[str, int, dict[str, Any], bool]) -> dict[str, Any]:
    """One record of one corpus -> the new proof, its notes and every per-record check."""
    corpus, i, r, do_profile = args
    W = _tools()
    prompt, proof = r["prompt"], r["proof"]
    prem = N.premises(proof)
    classes = [N.premise_class(f) for f in prem]
    present = sorted(set(classes), key=N.CLASSES.index)
    sides = sides_for(corpus, r, present)
    table = N.load_table()
    form_of = {c: table[c][sides[c]] for c in present}
    new = N.add_notes(prompt, proof, form_of)
    notes = [{"cls": c, "form": form_of[c], "side": sides[c]} for c in classes]
    chk: dict[str, Any] = {}
    try:
        chk["len"] = state_lengths(prompt, new, r["tag"])
        chk["len_before"] = state_lengths(prompt, proof, r["tag"])
        chk["state_env"] = True
    except Exception as e:  # noqa: BLE001  (ParseFail, a name past n64, a token outside the vocabulary, ...)
        chk["state_env"] = False
        chk["state_env_err"] = f"{type(e).__name__}: {e}"
    ok, why, _ = W["verify"](prompt + " " + new)
    chk["nd_verify"] = bool(ok)
    if not ok:
        chk["nd_verify_err"] = str(why)
    chk["premises_kept"] = N.premises(new) == prem
    rd = N.note_reading(prompt, new, table)
    chk["read_ok"] = (
        rd["ok"]
        and rd["late"] == 0
        and rd["unnoted"] == 0
        and [(x["premise_index"], x["cls"], x["form"], x["side"]) for x in rd["notes"]]
        == [(k, n["cls"], n["form"], n["side"]) for k, n in enumerate(notes)]
    )
    chk["strip_ok"] = N.strip_notes(prompt, new) == proof
    if do_profile:
        chk["profile"] = _profile_cmp(prompt, proof, new, len(notes))
    return {
        "i": i,
        "proof": new,
        "notes": notes,
        "note_forms": {c: sides[c] for c in present},
        "chk": chk,
    }


# ============================================================================ I/O
def read_source(name: str, limit: int | None = None) -> list[dict[str, Any]]:
    out = []
    with gzip.open(SOURCES[name], "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
                if limit and len(out) >= limit:
                    break
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def limit_cores(n: int = 8) -> list[int]:
    cores = sorted(os.sched_getaffinity(0))
    if len(cores) > n:
        cores = cores[-n:]
        os.sched_setaffinity(0, cores)
    return cores


def pool_map(fn, items: list, procs: int, chunksize: int = 64) -> list:
    import multiprocessing as mp

    if procs <= 1:
        return [fn(x) for x in items]
    with mp.get_context("fork").Pool(procs) as pool:
        return list(pool.imap(fn, items, chunksize=chunksize))


def write_gz(path: Path, lines: list[str]) -> None:
    import tagcorpus as TC

    TC.write_gz(path, lines)


# ============================================================================ limits
def pct(xs: list[int], q: float) -> int:
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))] if s else 0


def cmd_limits(a: argparse.Namespace) -> dict[str, Any]:
    """Lengths under each menu / junk form (every premise of a record in that form), per source, against the limits."""
    limit_cores(a.procs)
    t0 = time.time()
    out: dict[str, Any] = {
        "limits": {
            "ctx_nominal": CTX_NOMINAL, "tok_budget": TOK_BUDGET, "max_action": MAX_ACTION, "max_steps": MAX_STEPS,
            "cap_lines": CAP_LINES, "maxn_names": 64, "maxh_premise_names": 8,
        },
        "sources": {},
    }  # fmt: skip
    for src in SOURCES:
        recs = read_source(src, a.limit)
        res = pool_map(_limits_one, [(r["prompt"], r["proof"], r["key"]) for r in recs], a.procs)
        per: dict[str, Any] = {}
        for form in ("none", *N.NOTE_MENU, *N.JUNK_FORMS):
            rows = [x[form] for x in res]
            ok = [x for x in rows if "err" not in x]
            errs = collections.Counter(x["err"].split(":")[0] for x in rows if "err" in x)
            st = [x["state"] for x in ok]
            pr = [x["pair"] for x in ok]
            ac = [x["action"] for x in ok]
            sp = [x["steps"] for x in ok]
            per[form] = {
                "ok": len(ok), "errors": dict(errs),
                "state": {"max": max(st), "p99": pct(st, 0.99), "p50": pct(st, 0.5), "mean": round(statistics.fmean(st), 1)},
                "pair": {"max": max(pr), "p99": pct(pr, 0.99), "over_ctx_nominal": sum(x > CTX_NOMINAL for x in pr),
                         "over_512": sum(x > 512 for x in pr), "over_budget": sum(x > TOK_BUDGET for x in pr)},
                "action": {"max": max(ac), "p99": pct(ac, 0.99), "over_max_action": sum(x > MAX_ACTION for x in ac)},
                "steps": {"max": max(sp), "over_max_steps": sum(x > MAX_STEPS for x in sp)},
            }  # fmt: skip
        lines_after = [x["lines"] + x["n_prem"] for x in res]
        out["sources"][src] = {
            "records": len(recs),
            "premises": dict(sorted(collections.Counter(x["n_prem"] for x in res).items())),
            "lines_before_max": max(x["lines"] for x in res),
            "lines_after_max": max(lines_after),
            "records_over_cap_after": sum(x > CAP_LINES for x in lines_after),
            "lines_after": dict(sorted(collections.Counter(lines_after).items())),
            "per_form": per,
        }
        print(f"limits {src}: {len(recs)} records, {time.time() - t0:.0f} s", flush=True)
    out["secs"] = round(time.time() - t0)
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / ("limits.json" if not a.limit else f"limits_limit{a.limit}.json")
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {path}")
    return out


# ============================================================================ build
def shown_tag(r: dict[str, Any]) -> str:
    """The tag the record's states carry in training (authors_launch.shown_tag for these corpora)."""
    t = r["author_tag"] if r["tag_shown"] else "none"
    if t != r["author_tag"] or t not in ("A", "B", "none"):
        raise ValueError(
            f"record {r['key']}: tag_shown {r['tag_shown']} author_tag {r['author_tag']!r}"
        )
    return t


def rules_of(proof: str) -> list[str]:
    lines, _ = N.parse_lines(proof)
    return sorted({ln.rule for ln in lines} - {"PR"})


def out_record(src: dict[str, Any], res: dict[str, Any], corpus: str) -> dict[str, Any]:
    r = dict(src)
    new = res["proof"]
    r["proof"] = new
    if "text" in r:
        r["text"] = r["prompt"] + " " + new
    r["n_lines"] = new.split().count(";")
    r["rules"] = rules_of(new)
    r.update(
        {
            "notes": res["notes"],
            "note_forms": res["note_forms"],
            "lh_corpus": corpus,
            "ta_proof": src["proof"],
        }
    )
    return r


def phi(pairs: list[tuple[int, int]]) -> dict[str, Any]:
    c = collections.Counter(pairs)
    n11, n10, n01, n00 = c[(1, 1)], c[(1, 0)], c[(0, 1)], c[(0, 0)]
    den = ((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)) ** 0.5
    return {"n": len(pairs), "phi": round((n11 * n00 - n10 * n01) / den, 4) if den else None,
            "table": [[n11, n10], [n01, n00]]}  # fmt: skip


def hist(xs: list[int]) -> dict[str, int]:
    return {str(k): v for k, v in sorted(collections.Counter(xs).items())}


def dist(xs: list[int]) -> dict[str, Any]:
    return {
        "max": max(xs),
        "p99": pct(xs, 0.99),
        "p50": pct(xs, 0.5),
        "mean": round(statistics.fmean(xs), 2),
    }


def corpus_stats(
    corpus: str, recs: list[dict[str, Any]], res: list[dict[str, Any]]
) -> dict[str, Any]:
    """Counts per class and form, A-shares, independence checks, lengths and the per-record validation."""
    s: dict[str, Any] = {}
    per_cls: dict[str, dict[str, Any]] = {c: {"notes": 0, "A": 0, "B": 0, "record_classes": 0, "record_classes_A": 0,
                                              "forms": collections.Counter()} for c in N.CLASSES}  # fmt: skip
    for x in res:
        for n in x["notes"]:
            d = per_cls[n["cls"]]
            d["notes"] += 1
            d[n["side"]] += 1
            d["forms"][n["form"]] += 1
        for c, side in x["note_forms"].items():
            per_cls[c]["record_classes"] += 1
            per_cls[c]["record_classes_A"] += side == "A"
    for d in per_cls.values():
        d["a_share_notes"] = round(d["A"] / d["notes"], 4) if d["notes"] else None
        d["a_share_record_classes"] = (
            round(d["record_classes_A"] / d["record_classes"], 4) if d["record_classes"] else None
        )
        d["forms"] = dict(d["forms"])
    s["per_class"] = per_cls
    shares = [d["a_share_notes"] for d in per_cls.values() if d["a_share_notes"] is not None]
    tot = sum(d["notes"] for d in per_cls.values())
    s["notes_total"] = tot
    s["a_share_notes_overall"] = round(sum(d["A"] for d in per_cls.values()) / tot, 4)
    s["a_share_notes_class_range"] = [min(shares), max(shares)]
    s["records_with_notes"] = sum(bool(x["notes"]) for x in res)
    s["premises"] = hist([len(x["notes"]) for x in res])
    s["two_premises_same_class"] = sum(len(x["notes"]) > len(x["note_forms"]) for x in res)
    forms = collections.Counter((n["cls"], n["form"]) for x in res for n in x["notes"])
    s["notes_per_class_form"] = {f"{c}/{f}": v for (c, f), v in sorted(forms.items())}
    # independence: side ~ author label, ~ tag shown, ~ step order (old habit), and between classes of one record
    lab, tag, order, pair = [], [], [], []
    by_author = {"A": collections.Counter(), "B": collections.Counter()}
    for r, x in zip(recs, res, strict=True):
        if not x["note_forms"]:
            continue
        sides = [int(v == "A") for v in x["note_forms"].values()]
        first = int(x["notes"][0]["side"] == "A")
        lab.append((first, int(r["author"] == "A")))
        for v in x["note_forms"].values():
            by_author[r["author"]][v] += 1
        if r["tag_shown"]:
            tag.append((first, int(r["author_tag"] == "A")))
        if r["realized"]["order"] in ("A", "B"):
            order.append((first, int(r["realized"]["order"] == "A")))
        for k in range(len(sides)):
            for m in range(k + 1, len(sides)):
                pair.append((sides[k], sides[m]))
    s["a_share_by_author_label"] = {
        a: round(c["A"] / max(1, c["A"] + c["B"]), 4) for a, c in by_author.items()
    }
    s["assoc"] = {
        "first_note_side ~ author label": phi(lab),
        "first_note_side ~ tag token (shown records)": phi(tag),
        "first_note_side ~ step order form (records with room)": phi(order),
        "note side ~ note side, distinct classes of one record": phi(pair),
    }
    # lengths
    ok = [x for x in res if x["chk"]["state_env"]]
    s["lengths"] = {
        "lines_before": hist([r["n_lines"] for r in recs]),
        "lines_after": hist([x["proof"].split().count(";") for x in res]),
        "mean_lines_before": round(statistics.fmean(r["n_lines"] for r in recs), 3),
        "mean_lines_after": round(statistics.fmean(x["proof"].split().count(";") for x in res), 3),
        "records_over_12_lines_after": sum(x["proof"].split().count(";") > CAP_LINES for x in res),
        "state_tokens_before": dist([x["chk"]["len_before"]["state"] for x in ok]),
        "state_tokens_after": dist([x["chk"]["len"]["state"] for x in ok]),
        "pair_tokens_before": dist([x["chk"]["len_before"]["pair"] for x in ok]),
        "pair_tokens_after": dist([x["chk"]["len"]["pair"] for x in ok]),
        "action_tokens_after": dist([x["chk"]["len"]["action"] for x in ok]),
        "steps_after": dist([x["chk"]["len"]["steps"] for x in ok]),
        "pairs_over_ctx_nominal_after": sum(x["chk"]["len"]["pair"] > CTX_NOMINAL for x in ok),
        "actions_over_max_action_after": sum(x["chk"]["len"]["action"] > MAX_ACTION for x in ok),
        "train_pairs_before": sum(x["chk"]["len_before"]["steps"] for x in ok),
        "train_pairs_after": sum(x["chk"]["len"]["steps"] for x in ok),
    }
    # validation
    v: dict[str, Any] = {}
    for k in ("state_env", "nd_verify", "premises_kept", "read_ok", "strip_ok"):
        v[k] = {
            "pass": sum(bool(x["chk"][k]) for x in res),
            "fail": sum(not x["chk"][k] for x in res),
        }
    v["failures"] = [
        {"i": x["i"], "key": recs[x["i"]]["key"], **{k: x["chk"][k] for k in x["chk"] if k.endswith("_err") or
                                                     (isinstance(x["chk"][k], bool) and not x["chk"][k])}}
        for x in res if not all(x["chk"][k] for k in ("state_env", "nd_verify", "premises_kept", "read_ok", "strip_ok"))
    ][:50]  # fmt: skip
    prof = [x["chk"]["profile"] for x in res if "profile" in x["chk"]]
    keys = ["parsed", *PROFILE_KEYS, "dead_is_notes", "lines_plus_notes"]
    v["profile"] = {"checked": len(prof), **{k: sum(bool(p.get(k)) for p in prof) for k in keys}}
    v["profile"]["all_equal"] = sum(all(p.get(k) for k in keys) for p in prof)
    s["validation"] = v
    return s


def lean_sample(corpus: str, res: list[dict[str, Any]]) -> list[int]:
    """Seeded sample: up to LEAN_PER_CELL records per (class, form) present, then random noted records to LEAN_MIN."""
    rng = random.Random(f"{SEED}:lean:{corpus}")
    cells: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for x in res:
        for c, f in sorted({(n["cls"], n["form"]) for n in x["notes"]}):
            cells[(c, f)].append(x["i"])
    pick: set[int] = set()
    for cell in sorted(cells):
        ids = cells[cell]
        pick.update(rng.sample(ids, min(LEAN_PER_CELL, len(ids))))
    rest = sorted({x["i"] for x in res if x["notes"]} - pick)
    if len(pick) < LEAN_MIN:
        pick.update(rng.sample(rest, min(LEAN_MIN - len(pick), len(rest))))
    return sorted(pick)


def run_lean(
    corpus: str, recs: list[dict[str, Any]], res: list[dict[str, Any]], workers: int
) -> dict[str, Any]:
    import authors_corpus as AC

    ids = lean_sample(corpus, res)
    pairs = [(recs[i]["prompt"], res[i]["proof"]) for i in ids]
    t0 = time.time()
    verdict = AC.lean_verdicts(pairs, workers, LEAN_CACHE)
    ok = [verdict[AC._lean_key(p, b)] for p, b in pairs]
    cell = collections.Counter()
    cell_ok = collections.Counter()
    for i, good in zip(ids, ok, strict=True):
        for c, f in {(n["cls"], n["form"]) for n in res[i]["notes"]}:
            cell[(c, f)] += 1
            cell_ok[(c, f)] += good
    return {
        "sampled": len(ids), "accepted": sum(ok), "rejected": len(ok) - sum(ok),
        "rejected_keys": [recs[i]["key"] for i, g in zip(ids, ok, strict=True) if not g][:50],
        "per_class_form": {f"{c}/{f}": [cell_ok[(c, f)], n] for (c, f), n in sorted(cell.items())},
        "min_per_class_form": min(cell.values()) if cell else 0,
        "cells": len(cell),
        "rule": f"random.Random('{SEED}:lean:<corpus>'): up to {LEAN_PER_CELL} noted records per (class, form) in sorted "
                f"cell order, then random noted records up to {LEAN_MIN}; authors_corpus.lean_verdicts (nd2lean.translate "
                f"+ lean_check, {AC.LEAN_PER_FILE} theorems per Lean process, {workers} at a time)",
        "secs": round(time.time() - t0),
    }  # fmt: skip


def cmd_build(a: argparse.Namespace) -> None:
    cores = limit_cores(a.procs)
    t00 = time.time()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not N.TABLE_PATH.exists():
        raise SystemExit(f"{N.TABLE_PATH} missing: run `notes_corpus.py table` first")
    table = N.load_table()
    names = [f"{c}{'' if not a.limit else f'_limit{a.limit}'}" for c in CORPORA]
    for nm in names:
        if (out_dir / f"{nm}.jsonl.gz").exists() and not a.force:
            raise SystemExit(f"{out_dir / f'{nm}.jsonl.gz'} exists (--force to rebuild)")
    src = {s: read_source(s, a.limit) for s in SOURCES}
    keys = {s: [r["key"] for r in v] for s, v in src.items()}
    same_keys = keys["authors_half"] == keys["control_half"]
    if not same_keys:
        raise SystemExit("the two sources do not hold the same keys in the same order")
    summary: dict[str, Any] = {"cores": cores, "table": table, "table_sha256": N.table_sha256(),
                               "sources_same_keys_same_order": same_keys, "corpora": {}}  # fmt: skip
    built: dict[str, list[dict[str, Any]]] = {}
    for corpus, nm in zip(CORPORA, names, strict=True):
        t0 = time.time()
        sname = CORPORA[corpus][0]
        recs = src[sname]
        slim = [(corpus, i, {"prompt": r["prompt"], "proof": r["proof"], "key": r["key"], "author": r["author"],
                             "tag": shown_tag(r)}, True) for i, r in enumerate(recs)]  # fmt: skip
        res = pool_map(process, slim, a.procs)
        print(f"{corpus}: processed {len(res)} records in {time.time() - t0:.0f} s", flush=True)
        built[corpus] = res
        lines = [
            json.dumps(out_record(r, x, corpus)) + "\n" for r, x in zip(recs, res, strict=True)
        ]
        path = out_dir / f"{nm}.jsonl.gz"
        write_gz(path, lines)
        man: dict[str, Any] = {
            "corpus": corpus, "file": str(path), "sha256": sha256(path), "bytes": path.stat().st_size,
            "records": len(recs), "source": str(SOURCES[sname]), "source_sha256": sha256(SOURCES[sname]),
            "old_habits": "as in the source (follow the author)" if sname == "authors_half" else "as in the source (independent)",
            "note_rule": "author: every class in the record author's form" if CORPORA[corpus][1] == "author" else
                         f"coin: A iff u({SEED}, 'coin', key, class) < 0.5 (u = first 8 bytes of sha256 / 2^64); the same coin in l4 and l0",
            "table": table, "table_sha256": summary["table_sha256"], "menu": N.NOTE_MENU, "seed": SEED,
        }  # fmt: skip
        man.update(corpus_stats(corpus, recs, res))
        print(f"{corpus}: stats done, {time.time() - t0:.0f} s; Lean sample ...", flush=True)
        man["lean"] = run_lean(corpus, recs, res, int(os.environ.get("LEAN_GATE_WORKERS", "8")))
        man["secs"] = round(time.time() - t0)
        (out_dir / f"{nm}.manifest.json").write_text(json.dumps(man, indent=1) + "\n")
        summary["corpora"][corpus] = {"file": str(path), "sha256": man["sha256"], "records": len(recs),
                                      "a_share_class_range": man["a_share_notes_class_range"],
                                      "lean": [man["lean"]["accepted"], man["lean"]["sampled"]],
                                      "validation": {k: v for k, v in man["validation"].items() if k != "failures"}}  # fmt: skip
        print(f"{corpus}: wrote {path} ({man['sha256'][:12]}), Lean {man['lean']['accepted']}/{man['lean']['sampled']}, "
              f"{time.time() - t0:.0f} s", flush=True)  # fmt: skip
    # l4 and l0 share the coin: same note sides for the same key and class
    same = sum(
        x["note_forms"] == y["note_forms"] for x, y in zip(built["l4"], built["l0"], strict=True)
    )
    summary["l4_l0_same_note_sides"] = [same, len(built["l4"])]
    summary["l19_l4_same_proof_but_notes"] = sum(
        N.strip_notes("", x["proof"]) == N.strip_notes("", y["proof"]) for x, y in zip(built["l19"], built["l4"], strict=True)
    )  # fmt: skip
    summary["secs"] = round(time.time() - t00)
    nm = "notes_build.json" if not a.limit else f"notes_build_limit{a.limit}.json"
    (out_dir / nm).write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary["corpora"], indent=1))
    print(f"done in {summary['secs']} s")


# ============================================================================ CLI
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("limits")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--procs", type=int, default=8)
    sp.add_parser("table")
    p = sp.add_parser("build")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--procs", type=int, default=8)
    p.add_argument("--out-dir", default=str(DATA))
    p.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "limits":
        cmd_limits(a)
    elif a.cmd == "table":
        doc = N.write_table()
        print(json.dumps(doc["table"], indent=1))
    else:
        cmd_build(a)


if __name__ == "__main__":
    main()
