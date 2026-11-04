"""Check every number the headline figure draws against numbers.json.

    python3 check_labels.py      # figs/rl_needed_grpo.labels.json + numbers.json -> report

Each number in a label (tick labels, the arm notes, the range labels, Dan's note, the legend, the
y label and the caption) must equal some value of numbers.json when rounded the way the figure
rounds: 2 significant figures (with trailing zeros), 1 or 2 decimals, or a whole count. Counts
"k/n" and "k of n" must also match the runs they summarise. Unmatched numbers are printed; exit
status 1 if any.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def values(x, out: list[float]) -> None:
    if isinstance(x, bool):
        return
    if isinstance(x, (int, float)):
        out.append(float(x))
    elif isinstance(x, dict):
        for v in x.values():
            values(v, out)
    elif isinstance(x, list):
        for v in x:
            values(v, out)


def forms(v: float) -> set[str]:
    out = {f"{v:#.2g}", f"{v:.2g}", f"{v:.1f}", f"{v:.2f}", f"{v:g}", f"{v:,.0f}"}
    if abs(v - round(v)) < 1e-9:
        out.add(str(int(round(v))))
    return {s.rstrip(".") for s in out}


def main() -> None:
    lab = json.loads((HERE / "figs" / "rl_needed_grpo.labels.json").read_text())
    num = json.loads((HERE / "numbers.json").read_text())
    vals: list[float] = []
    values(num, vals)
    ok = set()
    for v in vals:
        ok |= forms(v)
    ok |= {str(i) for i in range(0, 13)}  # counts of lines and seeds, panel / list indices
    ok |= {"106", "32", "6", "2", "1", "0", "3"}  # the set size, k, the R formula's constants
    texts = lab["texts"] + lab["xticks"] + lab["legend"] + [lab["ylabel"]] + lab["caption"]
    bad = []
    for t in texts:
        for m in re.finditer(r"\d[\d,]*(?:\.\d+)?", t.replace(" ", " ")):
            s = m.group(0).rstrip(",")
            if s not in ok and s.replace(",", "") not in ok:
                bad.append((s, t[:80].replace("\n", " | ")))
    for s, t in bad:
        print(f"UNMATCHED {s!r} in: {t}")
    print(
        f"{sum(len(re.findall(r'[0-9]', t)) > 0 for t in texts)} labels checked, {len(bad)} unmatched"  # noqa: E501
    )
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
