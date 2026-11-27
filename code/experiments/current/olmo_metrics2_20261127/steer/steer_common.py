"""Shared pure functions for the M3 steering build (numpy only; no torch, no vLLM).

Design: discourse/steer.md Turn 5 section F plus the auditor's Turn 6 fixes d1-d7.  Everything here is deterministic and
unit-tested on CPU (tests/test_steer_common.py).
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from array import array
from pathlib import Path

import numpy as np

# --- constants of the design (F1-F4) ------------------------------------------------------------------------------
GRID_LAYERS = (16, 20, 24, 32, 40)
GRID_ALPHAS = (0.5, 1.0, 2.0, 4.0, 8.0)
SCRAMBLE_SEEDS = (2001, 2002, 2003)
TIE_NATS = 0.005
DAMAGE_FACTOR = 2.0  # [guess] rise <= 2 x D_rev
BAND_LO, BAND_HI = 0.05, 0.20  # g < 5 %, 5-20 %, >= 20 %
MARK = (524, 27963)  # "</" "think"  (pool/MANIFEST.json think_marker)
MARK_CLOSE = (1363, 397, 29)  # ">\n\n", ">\n", ">"
GROUPS = (
    "pos0",
    "lead",
    "user",
    "trail",
    "resp",
)  # position groups of a (prompt + response) sequence
SEGMENTS = ("total", "think", "answer")
EXIT_STOP_SPLIT_HALF = 42
LEAD_STR = "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n<|im_start|>user\n"
TRAIL_STR = "<|im_end|>\n<|im_start|>assistant\n<think>"


# --- small helpers --------------------------------------------------------------------------------------------------
def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def b64_ids(ids) -> str:
    return base64.b64encode(np.asarray(ids, dtype=np.int32).tobytes()).decode()


def ids_from_b64(s: str) -> list[int]:
    a = array("i")
    a.frombytes(base64.b64decode(s))
    return a.tolist()


def dump_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=False, default=_json_default)
    Path(tmp).replace(path)


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# --- end-of-thinking marker and position groups ---------------------------------------------------------------------
def find_marker(ids) -> int | None:
    """Index of the first ANSWER token (after the ">" token of the LAST `</think>` marker), or None if there is none.
    Same rule as pool/MANIFEST.json: think = ids[:think_end], answer = ids[think_end:] (EOS included)."""
    last = None
    for k in range(len(ids) - 2):
        if ids[k] == MARK[0] and ids[k + 1] == MARK[1] and ids[k + 2] in MARK_CLOSE:
            last = k
    return None if last is None else last + 3


def group_bounds(
    n_prompt: int, lead_end: int, trail_start: int, n_total: int
) -> list[tuple[int, int]]:
    """Half-open [a, b) index ranges of GROUPS in a sequence of n_total tokens (prompt first, then response).
    pos0 = [0,1); lead = [1, lead_end); user = [lead_end, trail_start); trail = [trail_start, n_prompt);
    resp = [n_prompt, n_total)."""
    assert 1 <= lead_end <= trail_start <= n_prompt <= n_total, (
        lead_end,
        trail_start,
        n_prompt,
        n_total,
    )
    return [
        (0, 1),
        (1, lead_end),
        (lead_end, trail_start),
        (trail_start, n_prompt),
        (n_prompt, n_total),
    ]


def prompt_segments(tok, prompt_ids: list[int]) -> tuple[int, int]:
    """(lead_end, trail_start) of a chat-template prompt, found by index and CHECKED by decoding.
    lead = system block + '<|im_start|>user\\n'; trailing = '<|im_end|>\\n<|im_start|>assistant\\n<think>'."""
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    starts = [i for i, t in enumerate(prompt_ids) if t == im_start]
    ends = [i for i, t in enumerate(prompt_ids) if t == im_end]
    assert len(starts) >= 3 and len(ends) >= 2, "template structure not as expected"
    user_start = starts[1]
    lead_end = None
    for n in range(user_start + 1, user_start + 8):
        if tok.decode(prompt_ids[:n]) == LEAD_STR:
            lead_end = n
            break
    assert lead_end is not None, "leading template block not found by decoding"
    trail_start = ends[-1]
    assert tok.decode(prompt_ids[trail_start:]) == TRAIL_STR, (
        "trailing template block not as expected"
    )
    assert lead_end <= trail_start
    return lead_end, trail_start


# --- NLL bookkeeping ------------------------------------------------------------------------------------------------
def nll_stats(logprobs, think_end: int | None) -> dict:
    """Per-text NLL sums and counts by segment from response-token log-probs.
    A text with no end-of-thinking marker (think_end None) counts entirely as thinking and has 0 answer tokens (d1)."""
    nll = -np.asarray(logprobs, dtype=np.float64)
    n = len(nll)
    te = n if think_end is None else min(int(think_end), n)
    return {
        "total": [float(nll.sum()), n],
        "think": [float(nll[:te].sum()), te],
        "answer": [float(nll[te:].sum()), n - te],
    }


def pool_stats(per_text: dict, tids, seg: str = "total") -> float:
    """Token-weighted mean NLL over the listed texts for one segment (nan if no tokens)."""
    s = sum(per_text[t][seg][0] for t in tids)
    n = sum(per_text[t][seg][1] for t in tids)
    return s / n if n else float("nan")


def n_excluded_answer(per_text: dict, tids) -> int:
    """d1: number of texts excluded from answer-token averages because they have no answer tokens."""
    return sum(1 for t in tids if per_text[t]["answer"][1] == 0)


# --- manipulation check: g and R with a prompt bootstrap, bands ------------------------------------------------------
def ratio_stat(base: dict, other: dict, ref: dict, tids, seg: str, sign: int = 1) -> float:
    """sign=+1: g = (NLL_base - NLL_other) / (NLL_base - NLL_ref);  sign=-1: R = (NLL_other - NLL_ref) / (NLL_base - NLL_ref).
    base = dpo, ref = s2300, other = the steered condition; pooled token-weighted means over the texts."""
    b = pool_stats(base, tids, seg)
    o = pool_stats(other, tids, seg)
    r = pool_stats(ref, tids, seg)
    den = b - r
    if not np.isfinite(den) or den == 0:
        return float("nan")
    return (b - o) / den if sign == 1 else (o - r) / den


def bootstrap_ratio(
    base: dict, other: dict, ref: dict, tids, seg: str, sign: int = 1, B: int = 10000, seed: int = 0
) -> dict:
    """Point estimate and 95 % prompt-bootstrap interval; the same resampled prompts are used for numerator and
    denominator.  Percentile interval (2.5, 97.5) over B resamples; resamples with an undefined ratio are dropped."""
    tids = list(tids)
    point = ratio_stat(base, other, ref, tids, seg, sign)
    rng = np.random.RandomState(seed)
    n = len(tids)
    arr = {
        name: np.array([[d[t][seg][0], d[t][seg][1]] for t in tids], dtype=np.float64)
        for name, d in (("b", base), ("o", other), ("r", ref))
    }
    idx = rng.randint(0, n, size=(B, n))

    def pooled(a):
        s = a[idx, 0].sum(1)
        c = a[idx, 1].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return s / c

    b, o, r = pooled(arr["b"]), pooled(arr["o"]), pooled(arr["r"])
    den = b - r
    with np.errstate(invalid="ignore", divide="ignore"):
        vals = (b - o) / den if sign == 1 else (o - r) / den
    vals = vals[np.isfinite(vals)]
    if len(vals) < max(10, B // 10):
        return {
            "point": point,
            "lo": float("nan"),
            "hi": float("nan"),
            "n_valid": int(len(vals)),
            "B": B,
        }
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {
        "point": float(point),
        "lo": float(lo),
        "hi": float(hi),
        "n_valid": int(len(vals)),
        "B": B,
    }


def band_of(ci: dict) -> str:
    """Pre-registered bands for g: <5 %, 5-20 %, >=20 %; a band is assigned only if the WHOLE interval lies inside it."""
    lo, hi = ci["lo"], ci["hi"]
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "undefined"
    if hi < BAND_LO:
        return "barely moved (g < 5 %)"
    if lo >= BAND_LO and hi < BAND_HI:
        return "slightly (5-20 %)"
    if lo >= BAND_HI:
        return "substantially (g >= 20 %)"
    return "between bands"


# --- vectors ----------------------------------------------------------------------------------------------------------
def scramble(v, seed: int):
    """Sign scramble: each coordinate multiplied by an independent random sign (RandomState(seed)); same norm and the
    same coordinate-magnitude profile."""
    v = np.asarray(v)
    signs = np.random.RandomState(seed).choice(np.array([-1.0, 1.0]), size=v.shape)
    return (v * signs).astype(v.dtype)


def top_k_share(v, k: int = 5) -> float:
    v = np.asarray(v, dtype=np.float64)
    sq = np.sort(v * v)[::-1]
    tot = sq.sum()
    return float(sq[:k].sum() / tot) if tot > 0 else float("nan")


def cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def split_half(per_prompt: np.ndarray, remove_top: int = 5) -> dict:
    """Split-half reliability of a mean over prompts.  per_prompt: [P, D] per-prompt difference vectors.
    Halves are the first and the second half in prompt order (0-15 vs 16-31 for P = 32).  c = cosine of the two half
    means; converted = sqrt(2c/(1+c)) is the implied cosine of the full mean with the population mean under isotropic
    noise (auditor's computation).  The same two numbers with the top-`remove_top` coordinates of the FULL mean (by
    absolute value) removed."""
    P = per_prompt.shape[0]
    h = P // 2
    a, b = per_prompt[:h].mean(0), per_prompt[h : 2 * h].mean(0)
    full = per_prompt.mean(0)
    drop = np.argsort(-np.abs(full))[:remove_top]
    keep = np.ones(per_prompt.shape[1], bool)
    keep[drop] = False

    def conv(c):
        return float(math.sqrt(2 * c / (1 + c))) if np.isfinite(c) and c > 0 else 0.0

    c = cosine(a, b)
    c2 = cosine(a[keep], b[keep])
    return {
        "c": c,
        "converted": conv(c),
        "c_top5_removed": c2,
        "converted_top5_removed": conv(c2),
        "n_prompts": int(P),
        "top5_coordinates": [int(i) for i in drop],
    }


# --- selection rule (F4 step 4 with the auditor's d2) --------------------------------------------------------------
def _key_setting(s):  # (vtype, layer, alpha)
    return s


def _pick_min(cands: list, nll: dict) -> tuple:
    """Minimum total NLL; ties within TIE_NATS go to smaller alpha, then smaller layer, then V_resp."""
    m = min(nll[c] for c in cands)
    tied = [c for c in cands if nll[c] <= m + TIE_NATS]
    return min(tied, key=lambda c: (c[2], c[1], 0 if c[0] == "V_resp" else 1))


def choose_resp_layer(resp_results: dict, d_rev: float) -> dict:
    """Stage 1: which layer the V_prompt row is run at.
    resp_results: {(vtype, layer, alpha): {"nll_sel": float, "damage_rise": float}} for V_resp settings only.
    Layer = layer of the best ADMISSIBLE V_resp setting; if none is admissible, the layer with the smallest damage rise
    at alpha = 1 (d2)."""
    lim = DAMAGE_FACTOR * d_rev
    adm = [s for s, r in resp_results.items() if r["damage_rise"] <= lim]
    if adm:
        best = _pick_min(adm, {s: resp_results[s]["nll_sel"] for s in adm})
        return {"layer": best[1], "rule": "min NLL among admissible V_resp", "best": list(best)}
    a1 = [s for s in resp_results if s[2] == 1.0]
    if not a1:
        a1 = list(resp_results)
    s = min(a1, key=lambda c: (resp_results[c]["damage_rise"], c[1]))
    return {
        "layer": s[1],
        "rule": "none admissible: smallest damage rise at alpha = 1 (d2)",
        "best": list(s),
    }


def select_final(all_results: dict, d_rev: float, max_alpha: float | None = None) -> dict:
    """Stage 2: over all evaluated settings choose the admissible one with minimum total NLL (ties as above).
    If none is admissible: V_resp at alpha = 1 on the layer with the smallest damage rise, flagged 'no admissible
    setting'.  Reports whether the minimum sits at the largest alpha of the grid."""
    lim = DAMAGE_FACTOR * d_rev
    adm = [s for s, r in all_results.items() if r["damage_rise"] <= lim]
    amax = max(s[2] for s in all_results) if max_alpha is None else max_alpha
    if adm:
        ch = _pick_min(adm, {s: all_results[s]["nll_sel"] for s in adm})
        return {
            "chosen": list(ch),
            "no_admissible": False,
            "at_max_alpha": bool(ch[2] == amax),
            "n_admissible": len(adm),
            "damage_limit": lim,
        }
    cands = (
        [s for s in all_results if s[0] == "V_resp" and s[2] == 1.0]
        or [s for s in all_results if s[2] == 1.0]
        or list(all_results)
    )
    ch = min(cands, key=lambda c: (all_results[c]["damage_rise"], c[1]))
    return {
        "chosen": list(ch),
        "no_admissible": True,
        "at_max_alpha": bool(ch[2] == amax),
        "n_admissible": 0,
        "damage_limit": lim,
    }


# --- NLL block cost plan and shrink order (F4 step 4) ---------------------------------------------------------------
def grid_settings(layers, alphas, with_prompt_row: bool) -> int:
    n = len(layers) * len(alphas)
    return n + (len(alphas) if with_prompt_row else 0)


def plan_nll_grid(
    tokens_per_s: float,
    tok_sel: int,
    tok_dmg: int,
    tok_chk: int,
    *,
    layers=GRID_LAYERS,
    alphas=GRID_ALPHAS,
    usd_per_h: float = 5.29,
    budget_usd: float = 3.0,
    n_chk_forwards: int = 5,
    extra_tokens: int = 0,
) -> dict:
    """Project the P3 NLL block's cost from the measured prefill rate and apply the fixed shrink order until the
    projection is at most `budget_usd`: drop alpha = 0.5 everywhere, then layer 40, then the V_prompt row.
    tok_sel / tok_dmg / tok_chk = total tokens of the S16 s2300 texts / S16 dpo texts / H16 s2300 texts.
    Each setting costs two forwards (selection text + damage text); the manipulation check costs n_chk_forwards
    forwards over the H16 text; extra_tokens covers the pre-flight reference forwards."""
    layers, alphas = list(layers), list(alphas)
    steps = [("full grid", layers, alphas, True)]
    a2 = [a for a in alphas if a != 0.5]
    steps.append(("drop alpha 0.5", layers, a2, True))
    l2 = [l for l in layers if l != 40]
    steps.append(("drop layer 40", l2, a2, True))
    steps.append(("drop V_prompt row", l2, a2, False))
    log = []
    chosen = None
    for name, ls, al, vp in steps:
        n_set = grid_settings(ls, al, vp)
        tokens = n_set * (tok_sel + tok_dmg) + n_chk_forwards * tok_chk + extra_tokens
        usd = tokens / max(tokens_per_s, 1e-9) / 3600.0 * usd_per_h
        log.append(
            {
                "step": name,
                "layers": ls,
                "alphas": al,
                "v_prompt_row": vp,
                "settings": n_set,
                "tokens": int(tokens),
                "projected_usd": round(usd, 3),
            }
        )
        chosen = log[-1]
        if usd <= budget_usd:
            break
    chosen = dict(chosen)
    chosen["within_budget"] = chosen["projected_usd"] <= budget_usd
    return {"plan": chosen, "log": log, "tokens_per_s": tokens_per_s, "budget_usd": budget_usd}


# --- E2 trigger (d6) -----------------------------------------------------------------------------------------------
def e2_trigger(
    e1_reach_pass: int,
    e1_reach_draws: int,
    e0p_reach_pass: int,
    e0p_reach_draws: int,
    e1_held_pass1: float,
    base_held_pass1: float,
    *,
    reach_margin: float = 2.0,
    held_points: float = 6.0,
) -> dict:
    """Run E2 iff [E1 reach passes - s x E0' reach passes >= 2, with s = E1 draws / E0' draws (s = 1 when E0' has 384
    draws, 2 when it was cut to 192; d6)] OR [|held-out pass@1 of E1 - stored dpo baseline| >= 6 points]."""
    s = e1_reach_draws / e0p_reach_draws
    diff = e1_reach_pass - s * e0p_reach_pass
    reach_hit = diff >= reach_margin
    held_diff = 100.0 * (e1_held_pass1 - base_held_pass1)
    held_hit = abs(held_diff) >= held_points
    return {
        "trigger": bool(reach_hit or held_hit),
        "reach_hit": bool(reach_hit),
        "held_hit": bool(held_hit),
        "reach_diff_scaled": float(diff),
        "scale_s": float(s),
        "held_diff_points": float(held_diff),
    }


# --- E2 pid files (d6: prompt i -> scramble (i mod 3) + 1) -------------------------------------------------------------
def split_draws(n: int, parts: int = 3) -> list[int]:
    """Split n draws per prompt over `parts` scrambles, larger parts first: 8 -> [3, 3, 2]."""
    base, r = divmod(n, parts)
    return [base + (1 if k < r else 0) for k in range(parts)]


def e2_pid_specs(
    reach_pids: list[str], held_pids: list[str], reach_n: int, held_n: int, parts: int = 3
) -> list[dict]:
    """For each scramble k (0-based): {pid: n_draws}.  Reach prompts get split_draws(reach_n)[k] draws in every scramble;
    held-out prompt i (in manifest order) gets held_n draws in scramble i mod parts only."""
    sp = split_draws(reach_n, parts)
    out = []
    for k in range(parts):
        d = {}
        if sp[k] > 0:
            for p in reach_pids:
                d[p] = sp[k]
        for i, p in enumerate(held_pids):
            if i % parts == k and held_n > 0:
                d[p] = held_n
        out.append(d)
    return out


# --- pass@k --------------------------------------------------------------------------------------------------------
def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n draws with c passes (Chen et al. 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)
