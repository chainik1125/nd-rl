"""Check intervention stop distributions and caps without sampling any proofs."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

from causal_probe import future_stop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    uniforms = [(i + 0.5) / 10 for i in range(10)]
    probabilities = [0.2, 0.6, 0.9, 0.1]
    expected = {
        (1, 4): [0.2, 0.48, 0.288, 0.032],
        (2, 4): [0.0, 0.6, 0.36, 0.04],
        (4, 4): [0.0, 0.0, 0.0, 1.0],
        (4, 3): [0.0, 0.0, 1.0],
    }
    results = []
    for (forced_extra, cap), expected_mass in expected.items():
        counts = [0] * cap
        draws = 10**cap
        for values in itertools.product(uniforms, repeat=cap):
            for extra, (probability, uniform) in enumerate(
                zip(probabilities, values, strict=False), start=1
            ):
                accepted, _ = future_stop(
                    hard_limit=extra == cap,
                    continuation="policy",
                    forced_extra=forced_extra,
                    extra=extra,
                    accept_probability=probability,
                    stop_uniform=uniform,
                )
                if accepted:
                    counts[extra - 1] += 1
                    break
            else:
                raise AssertionError("The hard cap must terminate every future")
        observed = [count / draws for count in counts]
        error = max(abs(a - b) for a, b in zip(observed, expected_mass, strict=True))
        if error > 1e-12:
            raise AssertionError("Intervention changes the expected stop distribution")
        results.append(
            {
                "forced_extra": forced_extra,
                "cap": cap,
                "uniform_sequences": draws,
                "observed_stop_mass": observed,
                "expected_stop_mass": expected_mass,
                "maximum_error": error,
            }
        )
    source = Path(__file__).with_name("causal_probe.py")
    result = {
        "scope": (
            "Exact finite-grid Bernoulli integration and synthetic hard caps; no model outcomes"
        ),
        "cases": results,
        "causal_probe_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "audit_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output = Path(args.out)
    output.write_text(json.dumps(result, indent=2) + "\n")
    output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
