# ruff: noqa: E501
#!/usr/bin/env python3
"""Module-by-module comparison of two PEFT LoRA adapters (Amendment 6(a) step 1), CPU only.

For every LoRA module present in both adapters: dW = scale * B @ A with scale = alpha / r (from each adapter's config).
Without forming dW (r << d): <dW1, dW2>_F = s1 s2 * sum((B2^T B1) * (A2 A1^T)) and ||dW||_F^2 = s^2 * sum((B^T B) * (A A^T)).
Reports per module the cosine and the norm ratio ||dW1|| / ||dW2||, and in total the cosine and the ratio of the
Frobenius norms over all modules. Readings fixed in PREREG Amendment 6(a): "match" = every module's cosine >= 0.99 and
the total norm within 2 %; "no match" = any module's cosine < 0.90 or the total norm off by more than 10 %; otherwise
"in between".

  compare_adapters.py A_DIR B_DIR --out OUT.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch
from safetensors.torch import load_file

MATCH_COS, MATCH_NORM = 0.99, 0.02
NOMATCH_COS, NOMATCH_NORM = 0.90, 0.10


def load(d: Path):
    cfg = json.loads((d / "adapter_config.json").read_text())
    sd = load_file(str(d / "adapter_model.safetensors"))
    mods: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in sd.items():
        m = re.match(r"(.*)\.lora_([AB])(?:\.default)?\.weight$", k)
        if m:
            mods.setdefault(m.group(1), {})[m.group(2)] = v.double()
    scale = float(cfg["lora_alpha"]) / float(cfg["r"])
    return mods, scale, cfg


def norm_key(k: str) -> str:
    """Strip PEFT prefixes so both adapters' module names line up."""
    k = re.sub(r"^base_model\.model\.", "", k)
    return re.sub(r"^model\.model\.", "model.", k)


def inner(A1, B1, s1, A2, B2, s2) -> float:
    return float(s1 * s2 * torch.sum((B2.T @ B1) * (A2 @ A1.T)))


def reading(min_cos: float, tot_ratio: float) -> str:
    if min_cos < NOMATCH_COS or abs(tot_ratio - 1) > NOMATCH_NORM:
        return "no match"
    if min_cos >= MATCH_COS and abs(tot_ratio - 1) <= MATCH_NORM:
        return "match"
    return "in between"


def compare(a: Path, b: Path) -> dict:
    ma, sa, ca = load(a)
    mb, sb, cb = load(b)
    ma = {norm_key(k): v for k, v in ma.items()}
    mb = {norm_key(k): v for k, v in mb.items()}
    common = sorted(set(ma) & set(mb))
    rows, tot_ab, tot_aa, tot_bb = [], 0.0, 0.0, 0.0
    for k in common:
        A1, B1, A2, B2 = ma[k]["A"], ma[k]["B"], mb[k]["A"], mb[k]["B"]
        ab, aa, bb = (
            inner(A1, B1, sa, A2, B2, sb),
            inner(A1, B1, sa, A1, B1, sa),
            inner(A2, B2, sb, A2, B2, sb),
        )
        tot_ab, tot_aa, tot_bb = tot_ab + ab, tot_aa + aa, tot_bb + bb
        cos = ab / math.sqrt(aa * bb) if aa > 0 and bb > 0 else float("nan")
        rows.append(
            {
                "module": k,
                "cos": cos,
                "norm_a": math.sqrt(aa),
                "norm_b": math.sqrt(bb),
                "norm_ratio": math.sqrt(aa / bb) if bb > 0 else float("nan"),
                "A_cos": float(
                    torch.nn.functional.cosine_similarity(A1.flatten(), A2.flatten(), dim=0)
                ),
            }
        )
    cosines = [r["cos"] for r in rows]
    tot_ratio = math.sqrt(tot_aa / tot_bb)
    by_type: dict[str, list[float]] = {}
    for r in rows:
        by_type.setdefault(r["module"].split(".")[-1], []).append(r["cos"])
    out = {
        "a": str(a),
        "b": str(b),
        "scale_a": sa,
        "scale_b": sb,
        "r_a": ca["r"],
        "r_b": cb["r"],
        "alpha_a": ca["lora_alpha"],
        "alpha_b": cb["lora_alpha"],
        "n_modules_a": len(ma),
        "n_modules_b": len(mb),
        "n_common": len(common),
        "only_a": sorted(set(ma) - set(mb))[:5],
        "only_b": sorted(set(mb) - set(ma))[:5],
        "total_cos": tot_ab / math.sqrt(tot_aa * tot_bb),
        "total_norm_a": math.sqrt(tot_aa),
        "total_norm_b": math.sqrt(tot_bb),
        "total_norm_ratio": tot_ratio,
        "min_module_cos": min(cosines),
        "median_module_cos": sorted(cosines)[len(cosines) // 2],
        "n_modules_cos_below_0.99": sum(c < 0.99 for c in cosines),
        "n_modules_cos_below_0.90": sum(c < 0.90 for c in cosines),
        "min_A_cos": min(r["A_cos"] for r in rows),
        "by_type_min_cos": {t: min(v) for t, v in by_type.items()},
        "reading": reading(min(cosines), tot_ratio),
        "thresholds": {
            "match": f"every module cos >= {MATCH_COS} and total norm within {MATCH_NORM:.0%}",
            "no match": f"any module cos < {NOMATCH_COS} or total norm off by more than {NOMATCH_NORM:.0%}",
        },
        "modules": rows,
    }
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(3)
    out = compare(a.a, a.b)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print({k: v for k, v in out.items() if k != "modules"})


if __name__ == "__main__":
    main()
