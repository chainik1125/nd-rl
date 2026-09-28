"""Prompt pools for ws3, read from fork_pin/data and code/ws3/textbook72.

Rules (CLAUDE.md, plan section 1a): RL prompts come only from `train.jsonl` (lengths 2 to 6,
already trained on) and `rl_targets.jsonl` (L_true 7 to 14). `heldout.jsonl`, `transfer.jsonl`,
`validation_36`, the test files, and the 72 textbook problems are evaluation only and are never
sampled for training. `banned_keys()` returns every renaming class that must not appear in an RL
prompt set; `assert_no_leak()` enforces it.
"""

from __future__ import annotations

import gzip
import json
import os
import random

import ndpath

# isort: split

EVAL_ONLY = ("heldout.jsonl", "transfer.jsonl", "validation_36.jsonl")
TEST_FILES = ("test_short_prompts.jsonl", "test_long_prompts.jsonl")
ATOMS = ["P", "Q", "R", "S"]


def read_jsonl(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def fork_path(name: str) -> str:
    p = os.path.join(ndpath.FORK_DATA, name)
    if not os.path.exists(p) and os.path.exists(p + ".gz"):
        return p + ".gz"
    return p


def load_pool(name: str):
    """'train', 'heldout', 'rl_targets', 'transfer', 'validation_36', ..."""
    return read_jsonl(fork_path(name if name.endswith(".jsonl") else f"{name}.jsonl"))


def thm_of_prompt(prompt: str) -> str:
    t = prompt.split()
    i, j = t.index("SEQ"), t.index("PRF")
    prem = " ".join(t[1:i])
    return f"{prem} |- {' '.join(t[i + 1 : j])}" if prem else f"|- {' '.join(t[i + 1 : j])}"


def canon_key(thm: str) -> str:
    """Atom-renaming class, identical to the fork's gen.canon_key (relabel by first appearance)."""
    m, out = {}, []
    for tok in thm.split():
        if tok in ATOMS:
            if tok not in m:
                m[tok] = ATOMS[len(m)]
            out.append(m[tok])
        else:
            out.append(tok)
    return " ".join(out)


def key_of(record: dict) -> str:
    return record["key"] if "key" in record else canon_key(thm_of_prompt(record["prompt"]))


def length_of(record: dict, default=None):
    """Reference proof length: L_true, else n_lines, else reference_lines, else `default`.

    The 14 `textbook_train` problems carry no length; callers pass a default for them.
    """
    for k in ("L_true", "n_lines", "reference_lines"):
        if k in record and record[k] is not None:
            return int(record[k])
    if default is None:
        raise KeyError("record has no length field")
    return default


def load_textbook72():
    """The 72 fixed textbook problems (58 dev + 14 train) with a `slice` field."""
    out = []
    for fn, slc in (("textbook_dev.jsonl", "dev58"), ("textbook_train.jsonl", "train14")):
        p = os.path.join(ndpath.TEXTBOOK72, fn)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} missing; copy it from origin/dmitry/cap-comparison-textbook-eval-data"
            )
        for r in read_jsonl(p):
            r = dict(r)
            r["slice"] = slc
            out.append(r)
    return out


def banned_keys(include_textbook: bool = True) -> set:
    keys = set()
    for name in EVAL_ONLY:
        keys |= {key_of(r) for r in load_pool(name)}
    if include_textbook and os.path.isdir(ndpath.TEXTBOOK72):
        keys |= {key_of(r) for r in load_textbook72()}
    return keys


def assert_no_leak(records, banned=None):
    banned = banned if banned is not None else banned_keys()
    hits = [r for r in records if key_of(r) in banned]
    if hits:
        raise ValueError(
            f"{len(hits)} RL prompts share a renaming class with an evaluation set, "
            f"e.g. {hits[0]['prompt']}"
        )


def build_prompt_set(length_weights: dict, n: int, seed: int = 0, check_leak: bool = True):
    """Sample `n` RL prompts. `length_weights` maps proof length (int) -> relative weight.

    Lengths 2 to 6 come from train.jsonl, 7 and up from rl_targets.jsonl (by L_true).
    Returns a list of dicts with `prompt`, `length`, `source`, `key`.
    """
    rng = random.Random(seed)
    by_len = {}
    for r in load_pool("train"):
        by_len.setdefault(int(r["n_lines"]), []).append((r["prompt"], "train", key_of(r)))
    for r in load_pool("rl_targets"):
        by_len.setdefault(int(r["L_true"]), []).append((r["prompt"], "rl_targets", key_of(r)))
    lengths = [int(L) for L in length_weights]
    weights = [float(length_weights[L]) for L in length_weights]
    for L in lengths:
        if L not in by_len:
            raise ValueError(f"no prompts of length {L}")
    picks = rng.choices(lengths, weights=weights, k=n)
    out, used = [], set()
    for L in picks:
        pool = by_len[L]
        for _ in range(50):
            prompt, source, key = rng.choice(pool)
            if key not in used:
                break
        used.add(key)
        out.append({"prompt": prompt, "length": L, "source": source, "key": key})
    if check_leak:
        assert_no_leak(out)
    return out


def test_file_keys() -> set:
    """Renaming classes of the fork's test prompt files (prompts only; verdicts are never read).
    The fixed pre-training set already contains some of them, so they are banned from RL prompts
    separately (questions.md, 2026-09-23 default)."""
    return {key_of(r) for name in TEST_FILES for r in load_pool(name)}


def read_prompt_file(path):
    """Prompt records from a JSONL file with at least `prompt` and `length`; fills `key`."""
    out = []
    for r in read_jsonl(path):
        r = dict(r)
        if "prompt" not in r or "length" not in r:
            raise ValueError(f"{path}: every record needs prompt and length")
        r.setdefault("key", key_of(r))
        r.setdefault("source", "file")
        out.append(r)
    return out


def train_prompts(p: dict, seed: int, base_dir: str):
    """The RL prompt set for `train.prompt`: `file` (relative to the config) or sampled by
    `length_weights` / `n_prompts`. A file is leak-checked against every evaluation set, the test
    prompt files, and each file in `disjoint_from` (for example the held-out half of a split)."""
    if not p.get("file"):
        return build_prompt_set(p["length_weights"], int(p["n_prompts"]), seed=seed)
    recs = read_prompt_file(os.path.join(base_dir, p["file"]))
    banned = banned_keys() | test_file_keys()
    for f in p.get("disjoint_from", []):
        banned |= {key_of(r) for r in read_jsonl(os.path.join(base_dir, f))}
    assert_no_leak(recs, banned)
    return recs


def eval_records(part: dict, pool: str, base_dir: str, n_key="n_prompts", default_n=200):
    """Evaluation records: `file` (relative to the config) if given, else a stride of `pool`."""
    if part.get("file"):
        return read_prompt_file(os.path.join(base_dir, part["file"]))
    return stride(load_pool(pool), int(part.get(n_key, default_n)))


def periodic_long(ev: dict, base_dir: str):
    """Long-prompt records for the periodic eval: a stride of `long_file` (relative to the
    config, e.g. the held-out half of a split) if given, else of rl_targets."""
    n = int(ev.get("n_long", 50))
    if ev.get("long_file"):
        return stride(read_prompt_file(os.path.join(base_dir, ev["long_file"])), n)
    return stride(load_pool("rl_targets"), n)


def stride(records, n: int):
    """Every k-th record so every length bucket of a length-sorted file is represented."""
    if n >= len(records):
        return list(records)
    step = max(1, len(records) // n)
    return records[::step][:n]
