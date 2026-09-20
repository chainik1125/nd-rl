"""Shared literal prompting and strict outcome reward for every OLMo arm."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

SPRINT = Path(__file__).resolve().parents[1]
ROOT = SPRINT.parent
VENDOR = ROOT / "r1-sprint-20260909-1625/code/vendor"
sys.path.insert(0, str(VENDOR))
from strict_verify import safe_verify  # noqa: E402


INSTRUCTION = """Complete a formal natural-deduction proof for the supplied theorem.
The theorem format is THM premises SEQ conclusion PRF. Premises are separated by commas.
Formulas use atoms P Q R S, falsity F, negation ( ~ A ), conjunction ( A & B ), disjunction ( A v B ), and implication ( A > B ). Here A and B describe formula positions; actual formulas use only P Q R S and F.
A proof line is N<number> followed by zero or more | tokens for subproof depth, then a formula, colon, rule, cited line numbers, and semicolon. Use consecutive line numbers beginning at N1. Put every declared premise first, in exactly the given order, at depth zero with rule PR. AS opens a subproof one level deeper. A sibling AS closes the preceding box; returning to a shallower depth closes deeper boxes. Every cited line number must precede the current line. Ordinary line citations must be accessible in the current scope; IMPI, NEGI, and ORE instead discharge closed boxes by citing their AS and final lines as specified below. The final line must prove the requested conclusion at depth zero.
Rules and citation order:
PR and AS have no citations. R cites one identical accessible formula.
ANDI cites A then B to derive ( A & B ). ANDE1 or ANDE2 cites a conjunction to derive its left or right side.
IMPE cites ( A > B ) then A to derive B. IMPI cites the AS line for A and the final line B of that closed box to derive ( A > B ).
ORI1 cites A to derive ( A v B ). ORI2 cites B to derive ( A v B ). ORE cites the disjunction line, then start/end of its left-case box, then start/end of its right-case box; both cases must derive the same conclusion.
NEGE cites A then ( ~ A ) to derive F. NEGI cites an AS line for A and a final F line of its closed box to derive ( ~ A ). BOTE cites F to derive any formula. DN cites ( ~ ( ~ A ) ) to derive A.
You may reason briefly before the final answer. Return exactly one <FINAL> block containing only the proof lines and a final QED token, closed by </FINAL>. Include spaces between formula tokens. Do not repeat or change the theorem. End the response after </FINAL>.
"""


def render_completion(proof: str) -> str:
    """Wrap the supplied proof verbatim; no generated proof repair is permitted."""
    if not isinstance(proof, str):
        raise TypeError("proof must be a string")
    return "<FINAL>\n" + proof + "\n</FINAL>"


def render_prompt(prompt: str, demonstrations=None) -> str:
    sections = [INSTRUCTION]
    for index, example in enumerate(demonstrations or [], 1):
        sections.append(f"\nExample {index}\nTheorem:\n{example['prompt']}\nAnswer:\n" + render_completion(example["proof"]) + "\n")
    sections.append("\nTheorem:\n" + prompt + "\nAnswer:\n")
    return "".join(sections)


def judge(prompt: str, raw_completion: str) -> dict:
    """Extract exactly one literal final block, then apply the frozen verifier.

    Scratch before the opening marker is allowed. Non-whitespace after the
    closing marker, duplicated/missing markers, missing/repeated/nonterminal
    QED, and any invalid proof receive zero reward. No normalization, repair,
    missing-token insertion, or proof search occurs.
    """
    result = {
        "accepted": False, "valid": False, "reward": 0.0, "reason": "",
        "proof": None, "n_lines": 0, "scratch_characters": 0,
        "final_characters": 0, "suffix_characters": 0,
        "final_start": None, "final_end": None,
    }
    if not isinstance(raw_completion, str):
        return dict(result, reason="completion must be a string")
    if raw_completion.count("<FINAL>") != 1 or raw_completion.count("</FINAL>") != 1:
        return dict(result, reason="expected exactly one literal final block", scratch_characters=len(raw_completion))
    opening = raw_completion.index("<FINAL>")
    start = opening + len("<FINAL>")
    end = raw_completion.index("</FINAL>")
    if end < start:
        return dict(result, reason="final markers are out of order", scratch_characters=len(raw_completion))
    proof = raw_completion[start:end]
    suffix = raw_completion[end + len("</FINAL>"):]
    result.update(proof=proof, scratch_characters=opening, final_characters=len(proof), suffix_characters=len(suffix), final_start=start, final_end=end)
    if suffix.strip():
        return dict(result, reason="non-whitespace text follows final block")
    tokens = proof.split()
    if not tokens or tokens[-1] != "QED" or tokens.count("QED") != 1:
        return dict(result, reason="final proof must end in exactly one QED")
    ok, reason, count = safe_verify(prompt, proof)
    result.update(accepted=bool(ok), valid=bool(ok), reward=float(ok), reason=reason, n_lines=int(count))
    return result


def protocol_identity() -> dict:
    paths = [Path(__file__), VENDOR / "strict_verify.py", VENDOR / "nd_metrics.py", VENDOR / "nd_verify/verify.py"]
    return {
        "files": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "instruction_sha256": hashlib.sha256(INSTRUCTION.encode()).hexdigest(),
        "extraction": "one_literal_final_block_terminal_unique_qed_no_repair",
        "stop": "actually_generated_closing_final_marker_or_native_eos_or_max_new_tokens_no_inserted_eos",
    }
