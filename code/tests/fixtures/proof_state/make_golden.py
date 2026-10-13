"""Golden fixtures for the proof-state port, made by running the take-home fork's own code.

    uv run --project code --no-sync python code/tests/fixtures/proof_state/make_golden.py \\
        --fork ~/nd-takehome --k12 data/kh/train_k12.jsonl --sealed data/bs/textbook72.jsonl

``--fork`` is a checkout of ``dan-pandori/nd-takehome`` (branch ``dan``); the K12 filter
functions (``kh_assemble.py``, ``kh_gen.py``) are read from the cap-horizon commit with
``git show``. ``--k12`` is the K12 training set (``data/kh/train_k12.jsonl``, 155,000 records,
md5 800b5486...). ``--sealed`` is the sealed textbook72 set (``data/bs/textbook72.jsonl``): no
fixture may hold a theorem in one of its renaming classes. The script checks its inputs, replaces
a generated proof in such a class by its digest (``gen.jsonl.gz``), and leaves such proofs out of
the filters' pool. Lean 4 must be installed for the term sizes (``$LEAN``, default
``~/.elan/bin/lean``). Nothing here imports the port: every expected value is the fork's.
``manifest.json`` records the commits, input digests, versions and counts. Writes, next to this
script:

- ``k12_sample.jsonl.gz``: the inputs, the first 100 K12 records of each length 2..12;
- ``state_env.jsonl.gz``: per record, ``decompose`` (plain and canonical names), a replay with
  environment-assigned names, and random or mutated actions with the environment's verdicts;
- ``step_check.jsonl.gz``: (state, step) pairs and ``step_check.check``'s verdicts;
- ``gen.jsonl.gz``: ``gen.py``'s output lines for fixed seeds and settings, verbatim;
- ``filters.jsonl.gz``: renaming classes, box depths, pruned lengths and K12 eligibility;
- ``term_size.json``: ``lean_check``'s elaborated term size of each sample proof.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import importlib
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CAP_HORIZON = "2e8b1914"  # origin/dan_cap-horizon: kh_gen.py, kh_assemble.py (K12's builders)
PER_LENGTH = 100
GEN_SETTINGS: list[tuple[str, list[str]]] = [
    ("cap6_seed1", ["--n", "400", "--min", "2", "--max", "6", "--seed", "1"]),
    (
        "cap12_flat_seed7",
        ["--n", "440", "--min", "2", "--max", "12", "--seed", "7", "--per_len", "40"],
    ),
    ("long_seed2", ["--n", "150", "--min", "7", "--max", "16", "--seed", "2", "--long"]),
    (
        "cap8_prem2_seed3",
        ["--n", "200", "--min", "3", "--max", "8", "--seed", "3", "--max_prem", "2"],
    ),
]
VOCAB = [
    *("(", ")", "¬", "∧", "∨", "→", "P", "Q", "R", "S", "False", "have", "exact", ";"),
    *("fun", "=>", "⟨", "⟩", ",", ".1", ".2", ".elim", "Or.inl", "Or.inr", "Or.elim"),
    *("Classical.byContradiction", "hh", ":", ":=", "by"),
    *(f"h{k}" for k in range(1, 9)),
]


def git(fork: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(fork), *args], check=True, capture_output=True, text=True
    ).stdout


def digest(text: str) -> str:
    """The first 16 hex digits of the SHA-256 (for long derived values)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    # mtime=0: the same rows give the same bytes
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as f:
        for row in rows:
            f.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))


def thm_of(prompt: str) -> str:
    """``THM a , b SEQ c PRF`` -> ``a , b |- c``."""
    t = prompt.split()
    i = t.index("SEQ")
    return " ".join(t[1:i]) + " |- " + " ".join(t[i + 1 : -1])


def sealed_classes(path: Path, canon_key: Any) -> set[str]:
    """The renaming classes of the sealed evaluation set, which no fixture may contain."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return {canon_key(thm_of(json.loads(line)["prompt"])) for line in lines}


def k12_sample(path: Path) -> list[dict[str, Any]]:
    """The first ``PER_LENGTH`` records of each length, in file order."""
    taken: Counter[int] = Counter()
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            n = r["n_lines"]
            if taken[n] < PER_LENGTH:
                taken[n] += 1
                out.append(
                    {"prompt": r["prompt"], "proof": r["proof"], "thm": r["thm"], "n_lines": n}
                )
    out.sort(key=lambda r: r["n_lines"])
    return [{"i": i, **r} for i, r in enumerate(out)]


def shift_names(action: list[str], by: int) -> list[str]:
    return [f"n{int(t[1:]) + by}" if t[:1] == "n" and t[1:].isdigit() else t for t in action]


def mutate(action: list[str], rng: random.Random) -> list[str]:
    """One or two random edits: replace, delete, insert or swap tokens."""
    a = list(action)
    names = [f"n{k}" for k in range(1, 65)]
    for _ in range(rng.choice([1, 1, 2])):
        op = rng.choice(["replace", "replace", "delete", "insert", "swap"])
        tok = rng.choice(VOCAB + names[:12])
        if op == "replace" and a:
            a[rng.randrange(len(a))] = tok
        elif op == "delete" and a:
            del a[rng.randrange(len(a))]
        elif op == "insert":
            a.insert(rng.randrange(len(a) + 1), tok)
        elif op == "swap" and len(a) > 1:
            k = rng.randrange(len(a) - 1)
            a[k], a[k + 1] = a[k + 1], a[k]
    return a


def state_env_rows(se: Any, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in records:
        prompt, proof = r["prompt"], r["proof"]
        rng = random.Random(1000 + r["i"])
        row: dict[str, Any] = {"i": r["i"]}
        for mode, canon in (("plain", False), ("canon", True)):
            steps, toks, env = se.decompose(prompt, proof, canon)
            for k, (_, _, history) in enumerate(steps):  # implied, so not stored
                assert history == [t for _, a, _ in steps[:k] for t in a]
            row[mode] = {
                "text": " ".join(toks),
                "nd": env.nd(),
                "steps": [[" ".join(s), " ".join(a)] for s, a, _ in steps],
            }
        assert row["canon"].pop("nd") == row["plain"]["nd"]  # an alpha-variant
        # Environment-assigned names: the canonical actions shifted by a base, with the names
        # they introduce scrambled; the environment renames them back.
        canon_actions = [a.split() for _, a in row["canon"]["steps"]]
        base = rng.randint(0, 32)
        actions: list[list[str]] = []
        for a in canon_actions:
            a = shift_names(a, base)
            if a[0] == "have" and rng.random() < 0.5:
                a[1] = f"n{rng.randint(1, 64)}"
            j = a.index(":=") if ":=" in a else -1
            if j > 0 and a[j + 1] in ("(", "Or.elim") and rng.random() < 0.5:
                a[j + 4 if a[j + 1] == "(" else j + 6] = f"n{rng.randint(1, 64)}"
            actions.append(a)
        env = se.Env(prompt, canon=True, base=base, assign=True)
        states: list[str] = []
        results: list[list[Any]] = []
        for a in actions:
            states.append(" ".join(env.state_tokens()))
            ok, why = env.apply(a)
            results.append([ok, why])
            if not ok:
                break
        row["assign"] = {
            "base": base,
            "actions": [" ".join(a) for a in actions],
            "results": results,
            "text": " ".join(env.text),
            "renamed": env.renamed,
            "defined": env.defined,
            "states_digest": digest("\n".join(states)),
            "history_digest": digest(" ".join(env.hist)),
        }
        assert env.nd() == row["plain"]["nd"]  # the same proof, renamed: implied, so not stored
        # Random and mutated actions at random points of the three replays.
        fuzz: list[list[Any]] = []
        plain_actions = [a.split() for _, a in row["plain"]["steps"]]
        for _ in range(4):
            config = rng.choice(["plain", "canon", "assign"])
            seq = {"plain": plain_actions, "canon": canon_actions, "assign": actions}[config]
            k = rng.randrange(len(seq))
            env = se.Env(
                prompt,
                canon=config != "plain",
                base=base if config == "assign" else 0,
                assign=config == "assign",
            )
            for a in seq[:k]:
                if not env.apply(a)[0]:
                    break
            if rng.random() < 0.3:
                act = [rng.choice(VOCAB + [f"n{x}" for x in range(1, 65)]) for _ in range(14)]
                act = act[: rng.randint(0, 14)]
            else:
                act = mutate(seq[k], rng)
            ok, why = env.apply(act)
            after = (
                [" ".join(env.text), digest(" ".join(env.state_tokens()))] if ok else [None, None]
            )
            fuzz.append([config, k, " ".join(act), ok, why, *after])
        fuzz += targeted(se, prompt, plain_actions, rng)
        row["fuzz"] = fuzz
        rows.append(row)
    return rows


def targeted(se: Any, prompt: str, actions: list[list[str]], rng: random.Random) -> list[list[Any]]:
    """Actions aimed at the rejections random edits rarely reach, at a random point of the plain
    replay; and a replay with the name base so high that the names run out."""
    k = rng.randrange(len(actions))
    env = se.Env(prompt)
    for a in actions[:k]:
        assert env.apply(a)[0]
    fr = env.frames[-1]
    names = [n for f in env.frames for n in f.names]
    nxt = f"n{env.maxname + 1}"
    tries: list[list[str]] = []
    if fr.last is not None:
        tries.append(["exact", fr.last])  # often "exact goal mismatch"
    if names:
        tries.append(
            [
                "have",
                nxt,
                ":",
                "P",
                ":=",
                "Or.elim",
                rng.choice(names),
                "(",
                "fun",
                "(",
                f"n{env.maxname + 2}",
                ":",
                "Q",
                ")",
                "=>",
                "by",
            ]
        )
    tries.append(
        [
            "have",
            nxt,
            ":",
            "P",
            ":=",
            "(",
            "fun",
            "(",
            f"n{env.maxname + 2}",
            ":",
            "Q",
            ")",
            "=>",
            "by",
        ]
    )  # a box on an atom
    tries.append(
        [
            "have",
            nxt,
            ":",
            "(",
            "P",
            "→",
            "Q",
            ")",
            ":=",
            "(",
            "fun",
            "(",
            f"n{env.maxname + 2}",
            ":",
            "Q",
            ")",
            "=>",
            "by",
        ]
    )  # binder mismatch
    out: list[list[Any]] = []
    for act in tries:
        e = copy.deepcopy(env)
        ok, why = e.apply(act)
        after = [" ".join(e.text), digest(" ".join(e.state_tokens()))] if ok else [None, None]
        out.append(["plain", k, " ".join(act), ok, why, *after])
    base = rng.randint(56, 64)
    env = se.Env(prompt, canon=True, base=base, assign=True)
    for k, a in enumerate(actions):
        ok, why = env.apply(a)
        if not ok:
            out.append([f"assign:{base}", k, " ".join(a), ok, why, None, None])
            break
    return out


def candidates(se: Any, env: Any, true_action: list[str], rng: random.Random) -> list[list[str]]:
    """Actions to check at this state: the true one, and variants of its formula and its term."""
    ftoks: Any = se.ftoks
    scope: list[tuple[str, Any]] = []
    for fr in env.frames:
        scope.extend(fr.names.items())
    j = true_action.index(":=")
    head = true_action[:j]
    formula = true_action[3:j]
    pool: list[Any] = [env.frames[-1].goal] + [f for _, f in scope] + list(env.prem)
    for f in list(pool):
        if f[0] in ("and", "or", "imp"):
            pool += [f[1], f[2]]
        elif f[0] == "not":
            pool.append(f[1])
    pool += [("not", rng.choice(pool)), ("imp", rng.choice(pool), ("bot",))]
    pool += [(rng.choice(["and", "or", "imp"]), rng.choice(pool), rng.choice(pool))]
    out = [true_action]
    term = true_action[j + 1 :]
    for _ in range(3):  # another declared formula, the same term
        out.append(head[:3] + ftoks(rng.choice(pool)) + [":="] + term)
    if not scope:
        return out
    names = [n for n, _ in scope]
    for _ in range(6):  # another term, the same declared formula
        a, b = rng.choice(names), rng.choice(names)
        shape = rng.choice(["r", ".1", ".2", ".elim", "app", "and", "inl", "inr", "dn"])
        tm = {
            "r": [a],
            ".1": [a, ".1"],
            ".2": [a, ".2"],
            ".elim": [a, ".elim"],
            "app": [a, b],
            "and": ["⟨", a, ",", b, "⟩"],
            "inl": ["Or.inl", a],
            "inr": ["Or.inr", a],
            "dn": ["Classical.byContradiction", "(", "fun", "hh", "=>", a, "hh", ")"],
        }[shape]
        out.append(head[:3] + formula + [":="] + tm + [";"])
    for _ in range(2):  # a well-typed term, its formula declared (often passes)
        n, f = rng.choice(scope)
        if f[0] == "and":
            k = rng.choice([1, 2])
            out.append(head[:3] + ftoks(f[k]) + [":=", n, f".{k}", ";"])
        elif f[0] in ("imp", "not"):
            arg = [m for m, g in scope if g == f[1]]
            res = f[2] if f[0] == "imp" else ("bot",)
            if arg:
                out.append(head[:3] + ftoks(res) + [":=", n, rng.choice(arg), ";"])
            elif f[0] == "not":
                out.append(
                    head[:3] + ftoks(("imp", f[1], rng.choice(pool))) + [":=", n, ".elim", ";"]
                )
        elif f[0] == "bot":
            out.append(head[:3] + ftoks(rng.choice(pool)) + [":=", n, ".elim", ";"])
        else:
            side = rng.choice(["Or.inl", "Or.inr"])
            other = rng.choice(pool)
            g = ("or", f, other) if side == "Or.inl" else ("or", other, f)
            out.append(head[:3] + ftoks(g) + [":=", side, n, ";"])
    return out


def step_check_rows(
    se: Any, sc: Any, records: list[dict[str, Any]], state_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r, srow in zip(records, state_rows, strict=True):
        if r["i"] % 4:  # every fourth record: about 11,000 pairs in all
            continue
        rng = random.Random(5000 + r["i"])
        base = rng.randint(0, 32)
        actions = [shift_names(a.split(), base) for _, a in srow["canon"]["steps"]]
        env = se.Env(r["prompt"], canon=True, base=base, assign=True)
        checks: list[list[Any]] = []
        for k, a in enumerate(actions):
            if a[0] == "have":
                t = a[a.index(":=") + 1]
                pool = candidates(se, env, a, rng) if t not in ("(", "Or.elim") else [a]
                seen: set[str] = set()
                for c in pool:
                    key = " ".join(c)
                    if key in seen:
                        continue
                    seen.add(key)
                    if not copy.deepcopy(env).apply(c)[0]:
                        continue  # the checker's precondition: the environment accepts the action
                    checks.append([k, key, sc.check(env, c)])
            else:
                checks.append([k, " ".join(a), sc.check(env, a)])
            ok, why = env.apply(a)
            assert ok, why
        rows.append(
            {"i": r["i"], "base": base, "actions": [" ".join(a) for a in actions], "checks": checks}
        )
    return rows


def gen_rows(fork: Path, sealed: set[str], canon_key: Any) -> list[dict[str, Any]]:
    """``gen.py``'s output for each setting; a line whose theorem is in a sealed class is replaced
    by its SHA-256, so the test can still compare it byte for byte."""
    rows: list[dict[str, Any]] = []
    env = {**os.environ, "ND_REGISTRY": "0", "ND_COMPUTE": "0"}
    with tempfile.TemporaryDirectory() as tmp:
        for name, args in GEN_SETTINGS:
            out = os.path.join(tmp, f"{name}.jsonl")
            run = subprocess.run(
                [sys.executable, str(fork / "gen.py"), *args, "--out", out],
                check=True,
                capture_output=True,
                text=True,
                env=env,
                cwd=tmp,
            )
            summary = json.loads(run.stdout.strip().splitlines()[-1])
            summary.pop("secs")
            lines: list[Any] = []
            for line in Path(out).read_text(encoding="utf-8").splitlines():
                if canon_key(json.loads(line)["thm"]) in sealed:
                    line = {"sha256": hashlib.sha256(line.encode("utf-8")).hexdigest()}
                lines.append(line)
            rows.append({"setting": name, "args": args, "summary": summary, "lines": lines})
    return rows


def filter_rows(
    fork: Path, records: list[dict[str, Any]], sealed: set[str]
) -> list[dict[str, Any]]:
    """K12's per-record filters, by the fork's functions, on the K12 sample and on a small pool
    drawn as cap-horizon drew K12's long bins (``kh_gen.worker``), which still has depth-3 proofs.
    Pool records in a sealed class are left out. Every third of the rest, with its atoms permuted
    (the same renaming class), is the exclusion set."""
    patterns: Any = importlib.import_module("patterns")
    prune: Any = importlib.import_module("prune")
    gen: Any = importlib.import_module("gen")
    with tempfile.TemporaryDirectory() as tmp:
        for mod in ("kh_assemble", "kh_gen", "kh_size"):
            src = git(fork, "show", f"{CAP_HORIZON}:{mod}.py")
            Path(tmp, f"{mod}.py").write_text(src, encoding="utf-8")
        sys.path.append(tmp)
        kh_assemble: Any = importlib.import_module("kh_assemble")
        kh_gen: Any = importlib.import_module("kh_gen")
        out = os.path.join(tmp, "pool.jsonl")
        kh_gen.worker((31000, 3000, out))
        pool = [json.loads(line) for line in Path(out).read_text(encoding="utf-8").splitlines()]
    pool = [r for r in pool if gen.canon_key(r["thm"]) not in sealed]
    swap = str.maketrans({"P": "Q", "Q": "R", "R": "S", "S": "P"})
    exclude = [r["thm"].translate(swap) for r in pool[::3]]
    excluded = {gen.canon_key(t) for t in exclude}
    rows: list[dict[str, Any]] = [{"exclude_thms": exclude}]
    sources: list[dict[str, Any]] = [{"i": r["i"], **r} for r in records]
    sources += [{"thm": r["thm"], "prompt": r["prompt"], "proof": r["proof"]} for r in pool]
    for r in sources:
        cl = patterns.classify(r["proof"])
        key = gen.canon_key(r["thm"])
        wd = kh_assemble.written_depth(r["proof"])
        # kh_assemble.main's long-bin rule (lines 166-182 at the cap-horizon commit)
        eligible = key not in excluded and not (cl["depth3"] or wd >= 3)
        where = {"i": r["i"]} if "i" in r else {k: r[k] for k in ("thm", "prompt", "proof")}
        rows.append(
            {
                **where,
                "key": key,
                "written_depth": wd,
                "pruned_depth3": cl["depth3"],
                "pruned_length": prune.pruned_length(r["prompt"], r["proof"]),
                "pruned_length_with_premises": len(
                    patterns.prune_lines(patterns.parse(r["proof"]))
                ),
                "eligible": eligible,
            }
        )
    return rows


# lean_seq texts beyond the ND rules (``.elim`` on ¬ / ∧ / ∨, ¬ A declared as A → False), which
# model samples use and the K12 proofs do not.
EXTRA_TEXTS = [
    (
        "THM ( ~ P ) SEQ ( P > Q ) PRF",
        ["have n1 : ( ¬ P ) := h1", "have n2 : ( P → Q ) := n1 .elim", "exact n2"],
    ),
    (
        "THM ( P & Q ) SEQ ( ( P > ( Q > R ) ) > R ) PRF",
        [
            "have n1 : ( P ∧ Q ) := h1",
            "have n2 : ( ( P → ( Q → R ) ) → R ) := n1 .elim",
            "exact n2",
        ],
    ),
    (
        "THM ( P v Q ) SEQ ( ( P > R ) > ( ( Q > R ) > R ) ) PRF",
        [
            "have n1 : ( P ∨ Q ) := h1",
            "have n2 : ( ( P → R ) → ( ( Q → R ) → R ) ) := n1 .elim",
            "exact n2",
        ],
    ),
    (
        "THM F SEQ ( P & Q ) PRF",
        ["have n1 : False := h1", "have n2 : ( P ∧ Q ) := n1 .elim", "exact n2"],
    ),
    (
        "THM ( ~ P ) , P SEQ F PRF",
        [
            "have n1 : ( P → False ) := h1",
            "have n2 : P := h2",
            "have n3 : False := n1 n2",
            "exact n3",
        ],
    ),
    (
        "THM ( ~ ( ~ ( P & Q ) ) ) SEQ ( P & Q ) PRF",
        [
            "have n1 : ( ¬ ( ¬ ( P ∧ Q ) ) ) := h1",
            "have n2 : ( P ∧ Q ) := Classical.byContradiction ( fun hh => n1 hh )",
            "exact n2",
        ],
    ),
    (
        "THM ( P v Q ) SEQ ( Q v P ) PRF",
        [
            "have n1 : ( P ∨ Q ) := h1",
            "have n2 : ( Q ∨ P ) := Or.elim n1 ( fun ( n3 : P ) => by "
            "have n4 : ( Q ∨ P ) := Or.inr n3",
            "exact n4 ) ( fun ( n5 : Q ) => by have n6 : ( Q ∨ P ) := Or.inl n5",
            "exact n6 )",
            "exact n2",
        ],
    ),
    (
        "THM P , ( P > Q ) SEQ ( ~ ( ~ Q ) ) PRF",
        [
            "have n1 : P := h1",
            "have n2 : ( P → Q ) := h2",
            "have n3 : Q := n2 n1",
            "have n4 : ( ¬ ( ¬ Q ) ) := ( fun ( n5 : ( ¬ Q ) ) => by have n6 : False := n5 n3",
            "exact ( n6 : False ) )",
            "exact n4",
        ],
    ),
]


def term_sizes(fork: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    lean_tok: Any = importlib.import_module("lean_tok")
    lean_check: Any = importlib.import_module("lean_check")
    tok = lean_tok.LeanTokenizer("lean_seq")
    sources = []
    for r in records:
        tok.decode(tok.encode_proof(r["proof"]))
        sources.append(tok.statement(r["prompt"]) + " " + tok.last_text)
    texts = [(p, " ; ".join(steps)) for p, steps in EXTRA_TEXTS]
    extra = [tok.statement(p) + " " + tok.text(t.split()) for p, t in texts]
    res, _, _ = lean_check.check(sources + extra)
    assert all(x["ok"] for x in res), [x for x in res if not x["ok"]][:3]
    version = subprocess.run(
        [lean_check.LEAN, "--version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    sizes = [x["size"] for x in res]
    return {
        "lean": version,
        "sizes": sizes[: len(records)],
        "extra": [[p, t, n] for (p, t), n in zip(texts, sizes[len(records) :], strict=True)],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--fork", type=Path, required=True)
    ap.add_argument("--k12", type=Path, required=True)
    ap.add_argument(
        "--sealed", type=Path, required=True, help="textbook72 (data/bs/textbook72.jsonl)"
    )
    ap.add_argument("--out", type=Path, default=HERE)
    a = ap.parse_args()
    fork = a.fork.resolve()
    sys.path.insert(0, str(fork))
    se: Any = importlib.import_module("state_env")
    sc: Any = importlib.import_module("step_check")
    gen: Any = importlib.import_module("gen")
    t0 = time.time()

    records = k12_sample(a.k12)
    sealed = sealed_classes(a.sealed, gen.canon_key)
    inputs = [r["thm"] for r in records] + [thm_of(p) for p, _ in EXTRA_TEXTS]
    assert not any(gen.canon_key(t) in sealed for t in inputs), "an input is in a sealed class"
    write_jsonl_gz(a.out / "k12_sample.jsonl.gz", records)
    state_rows = state_env_rows(se, records)
    write_jsonl_gz(a.out / "state_env.jsonl.gz", state_rows)
    check_rows = step_check_rows(se, sc, records, state_rows)
    write_jsonl_gz(a.out / "step_check.jsonl.gz", check_rows)
    gen_out = gen_rows(fork, sealed, gen.canon_key)
    write_jsonl_gz(a.out / "gen.jsonl.gz", gen_out)
    filters = filter_rows(fork, records, sealed)
    write_jsonl_gz(a.out / "filters.jsonl.gz", filters)
    sizes = term_sizes(fork, records)
    (a.out / "term_size.json").write_text(json.dumps(sizes) + "\n", encoding="utf-8")

    verdicts = Counter(str(v) for row in check_rows for _, _, v in row["checks"])
    manifest = {
        "fork": "dan-pandori/nd-takehome",
        "fork_commit": git(fork, "rev-parse", "HEAD").strip(),
        "fork_dirty": bool(git(fork, "status", "--porcelain", "--untracked-files=no").strip()),
        "cap_horizon_commit": git(fork, "rev-parse", CAP_HORIZON).strip(),
        "k12_md5": hashlib.md5(a.k12.read_bytes()).hexdigest(),
        "sealed_md5": hashlib.md5(a.sealed.read_bytes()).hexdigest(),
        "sealed_classes": len(sealed),
        "python": sys.version.split()[0],
        "lean": sizes["lean"],
        "argv": [Path(x).name for x in sys.argv],
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counts": {
            "k12_sample": len(records),
            "state_env_steps_plain": sum(len(r["plain"]["steps"]) for r in state_rows),
            "state_env_steps_canon": sum(len(r["canon"]["steps"]) for r in state_rows),
            "state_env_fuzz": sum(len(r["fuzz"]) for r in state_rows),
            "step_check_pairs": sum(verdicts.values()),
            "step_check_verdicts": dict(sorted(verdicts.items())),
            "gen_records": {r["setting"]: len(r["lines"]) for r in gen_out},
            "gen_records_sealed_class": sum(
                isinstance(x, dict) for r in gen_out for x in r["lines"]
            ),
            "filter_rows": len(filters) - 1,
        },
        "seconds": round(time.time() - t0),
    }
    (a.out / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["counts"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
