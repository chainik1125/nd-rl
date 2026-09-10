"""Add a strict preceding-reference condition to the frozen official verifier.

No source verifier or codec semantics are modified. Future adaptive consumers
must explicitly import this module and pin its SHA; existing runs are unchanged.
"""
from nd_metrics import safe_verify as _official_safe_verify
from nd_verify import parse_proof_tokens


def safe_verify(prompt: str, body: str) -> tuple[bool, str, int]:
    """Official acceptance AND every citation points to a preceding line.

    Like the official helper, this judges only the body through its first QED.
    Literal-token/EOS/guard checks remain separate acquisition obligations.
    """
    ok, reason, n_lines = _official_safe_verify(prompt, body)
    if not ok:
        return ok, reason, n_lines
    try:
        for line in parse_proof_tokens(body.split()):
            for ref in line['refs']:
                if ref >= line['idx']:
                    return False, (f"strict: reference N{ref} is not before "
                                   f"line N{line['idx']}"), n_lines
    except Exception as exc:
        return False, f"strict parser exception: {type(exc).__name__}", 0
    return ok, reason, n_lines


strict_verify = safe_verify
