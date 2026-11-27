"""Diagnose ExPO write verification failures (orchestrator, 2026-11-27; pod CPU, read only).

For each named tensor: exact reference r = a + gamma (b - a) with gamma = p/q rational (4.6 = 23/5, 3.5 = 7/2), computed in
fp64 from bf16 a, b (q*a and p*(b-a) are exact in fp64 for bf16 inputs, so r is one rounding away from exact), then
compares the written bf16 o with bf16(r) and reports where the ulp ratio of the writer's own check exceeds 1, together with
the scale of the inputs there (|a|, |gamma (b - a)|), i.e. whether the failures are cancellation (r near 0) or real errors.
"""

import json
import sys
from fractions import Fraction
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rank1"))
from common import Checkpoint  # noqa: E402


def main():
    dpo, s5, out, gamma = (
        Checkpoint(sys.argv[1]),
        Checkpoint(sys.argv[2]),
        Checkpoint(sys.argv[3]),
        float(sys.argv[4]),
    )
    names = sys.argv[5:]
    fr = Fraction(gamma).limit_denominator(100)
    p, q = fr.numerator, fr.denominator
    res = []
    for n in names:
        a, b, o = dpo.get(n).double(), s5.get(n).double(), out.get(n).double()
        exact = (q * a + p * (b - a)) / q
        ref_writer = (a + gamma * (b - a)).float()  # the writer's own verification reference
        eps = torch.finfo(torch.bfloat16).eps
        ratio = (o.float() - ref_writer).abs() / (eps * ref_writer.abs().clamp_min(1e-30))
        bad = ratio > 1.0
        exact_bf16 = exact.float().to(torch.bfloat16).double()
        mism = o != exact_bf16
        scale = torch.maximum(a.abs(), (gamma * (b - a)).abs())
        absdiff = (o - exact).abs()
        rel_to_scale = absdiff / scale.clamp_min(1e-30)
        res.append(
            {
                "name": n,
                "numel": a.numel(),
                "bad_ulp_ratio": int(bad.sum()),
                "bad_examples": [
                    {
                        "a": float(a.flatten()[i]),
                        "b": float(b.flatten()[i]),
                        "o": float(o.flatten()[i]),
                        "exact": float(exact.flatten()[i]),
                        "ratio": float(ratio.flatten()[i]),
                    }
                    for i in bad.flatten().nonzero().flatten()[:5].tolist()
                ],
                "o_ne_bf16(exact)": int(mism.sum()),
                "max_absdiff_over_input_scale": float(rel_to_scale.max()),
                "max_absdiff_over_input_scale_where_bad": float(rel_to_scale[bad].max())
                if bad.any()
                else 0.0,
                "max_abs_exact_where_bad": float(exact.abs()[bad].max()) if bad.any() else 0.0,
                "median_abs_a": float(a.abs().median()),
            }
        )
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
