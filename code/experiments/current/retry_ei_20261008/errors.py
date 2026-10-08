# imports the fork (lean_tok), which is not resolvable here
# pyright: reportMissingImports=false
"""What the model is told between two attempts: one error-code token and, when the failure has a
place, the name of the `have` it happened in.

A failed attempt is one of three things, checked in this order:

  grammar   the text is outside the strict lean_seq grammar (fork lean_tok.inverse, 'LEANPARSE x')
  type      it parses and a `have` does not have its declared type, or the final `exact` is not
            the theorem (vendored lean_prefilter, exact against Lean on Dan's 1.31M-text corpus)
  other     the gate rejected it and neither of the above says why

The verdict itself never comes from here: the caller decides success with the fork's Lean gate
and nd_verify, exactly as the harness's judge does. This file only names the failure.
"""

import lean_prefilter
import lean_tok  # fork

PARSE_CODES = (
    "no-eos",  # hit max_new without ending
    "eof",
    "unbound",
    "formula",
    "name",
    "exact not last",
    "trailing tokens",
    "ORE box",
    "PR position",
    "expected",  # 'expected <token>', any token
)
TYPE_CODES = (
    "app-arg",
    "app-result",
    "function-expected",
    "and-intro",
    "anon-ctor",
    "and-elim",
    "projection",
    "or-intro",
    "or-intro-target",
    "or-elim-major",
    "by-contradiction",
    "fun-body",
    "fun-binder",
    "fun-not-pi",
    "ascription",
    "reiterate",
    "premise-type",
    "unknown-premise",
    "elim",
    "false-elim",
    "goal",
)
OTHER = "other"
CODES = PARSE_CODES + TYPE_CODES + (OTHER,)
BLANK = "<fail>"  # the ablation's only message: "that attempt failed"


def code_token(code):
    return f"<err:{code}>"


def extend(tok):
    """Append the error tokens to a fork LeanTokenizer's vocabulary, after the names. Sampled
    proofs never legitimately contain them; one that does fails the grammar."""
    for t in [code_token(c) for c in CODES] + [BLANK]:
        if t not in tok.stoi:
            tok.stoi[t] = len(tok.itos)
            tok.itos.append(t)
    return tok


def _is_name(t):
    return len(t) > 1 and t[0] == "n" and t[1:].isdigit()


def _parse_fail(toks):
    """(code, name of the `have` the parser was in) for tokens the grammar rejects."""
    try:
        lean_tok.inverse(toks)
    except lean_tok.ParseFail as e:
        msg = str(e)
        code = "expected" if msg.startswith("expected") else msg
        # inverse() keeps its cursor in a closure, so the only way to learn where it stopped
        # without copying the parser is to read `pos` out of its frame on the traceback.
        at, tb = len(toks), e.__traceback__
        while tb is not None:
            if tb.tb_frame.f_code.co_name == "inverse":
                at = tb.tb_frame.f_locals["pos"][0]
            tb = tb.tb_next
        for i in range(min(at, len(toks) - 1), -1, -1):
            if toks[i] == "have" and i + 1 < len(toks) and _is_name(toks[i + 1]):
                return code, toks[i + 1]
        return code, None
    return OTHER, None  # it parses after all: the caller's verdict came from somewhere else


def classify(tok, prompt, ids):
    """(code, name or None) for a failed attempt's sampled ids."""
    toks, ended = [], False
    for x in ids:
        if x == tok.pad:
            continue
        if x == tok.eos:
            ended = True
            break
        toks.append(tok.itos[x])
    if not ended:
        return "no-eos", None
    code, where = _parse_fail(toks)
    if code != OTHER:
        return (code if code in PARSE_CODES else OTHER), where
    detail = lean_prefilter.reject_detail(tok.statement(prompt), tok.text(toks))
    if detail is None or detail[0] not in TYPE_CODES:
        return OTHER, None
    return detail


def error_ids(tok, prompt, ids, mode):
    """The tokens appended after a failed attempt. mode 'error': code and place; 'blank': only
    that it failed."""
    if mode == "blank":
        return [tok.stoi[BLANK]]
    code, where = classify(tok, prompt, ids)
    return [tok.stoi[code_token(code)]] + ([tok.stoi[where]] if where else [])
