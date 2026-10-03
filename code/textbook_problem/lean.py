"""Lean 4 rendering and checking for textbook problems, through Dan's lean_seq tokenizer and Lean
checker, copied unmodified into lean_tools/ and pinned by lean_tools/MANIFEST.json.

    cd code && uv run python textbook_problem/lean.py --selftest

The copies import each other as `code.shared_code.*`, the path they have on Dan's branch. This module
loads them from lean_tools/ under those names, so they run without edits.

A proof counts only if the pinned nd_verify (through strict_verify) and Lean both accept it. Lean
alone never decides here, although lean_check.py's docstring says it does on Dan's pipeline.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "lean_tools")


def _load(name: str, fn: str):
    full = f"code.shared_code.{name}"
    if full not in sys.modules:
        spec = importlib.util.spec_from_file_location(full, os.path.join(TOOLS, fn))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
    return sys.modules[full]


lean_seq = _load("lean_seq", "lean_seq.py")
lean_check = _load("lean_check", "lean_check.py")
TOK = lean_seq.LeanSeqTokenizer()


def verify_manifest() -> list[str]:
    """-> the files in lean_tools/ whose sha256 differs from MANIFEST.json (empty if all match)."""
    with open(os.path.join(TOOLS, "MANIFEST.json")) as fi:
        want = json.load(fi)["files"]
    bad = []
    for fn, h in want.items():
        with open(os.path.join(TOOLS, fn), "rb") as fb:
            if hashlib.sha256(fb.read()).hexdigest() != h:
                bad.append(fn)
    return bad


def statement(prompt: str) -> str:
    """'THM ... SEQ ... PRF' -> the Lean theorem statement."""
    return TOK.statement(prompt)


def lean_proof(proof: str) -> str:
    """An ND proof body ('N1 ... ; QED') -> its lean_seq tactic text."""
    return TOK.lean_text(TOK.encode_proof(proof))


def lean_theorem(prompt: str, proof: str) -> str:
    """A complete Lean theorem: statement, then the rendered proof."""
    return f"{statement(prompt)} {lean_proof(proof)}"


def check(pairs, workers=None) -> list[bool]:
    """[(prompt, ND proof body)] -> whether Lean 4 core accepts each rendered theorem, in order.
    Raises FileNotFoundError if Lean is not installed (see lean_check.find_lean)."""
    return lean_check.check_theorems([lean_theorem(p, b) for p, b in pairs], workers=workers)


def selftest():
    bad = verify_manifest()
    assert not bad, f"lean_tools/ differs from MANIFEST.json: {bad}"
    prompt, proof = "THM ( P > Q ) , P SEQ Q PRF", "N1 ( P > Q ) : PR ; N2 P : PR ; N3 Q : IMPE N1 N2 ; QED"
    thm = lean_theorem(prompt, proof)
    assert thm.startswith("theorem t ( P Q R S : Prop ) ( h1 : ( P → Q ) ) ( h2 : P ) : Q := by"), thm
    assert thm.endswith("have n3 : Q := n1 n2 ; exact n3"), thm
    print("manifest ok; rendering ok")
    print(" ", thm)
    if lean_check.find_lean() is None:
        print("Lean not found: skipped the Lean check (install elan, or set $LEAN)")
        return
    wrong = "THM ( P > Q ) , P SEQ P PRF"
    ok = check([(prompt, proof), (wrong, proof)])
    assert ok == [True, False], ok
    print("Lean accepts the proof and rejects it for the wrong goal")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        selftest()
