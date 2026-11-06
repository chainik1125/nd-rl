"""The guard's timeline per line, in x_dan, from the merged metrics rows (read-only).

    python3 guard.py [LINE ...]      # serrano; KD/sup/state.json, KD/runs, KD/collected, KD/backup

`--rollback_on_alarm` restores the newest good checkpoint on an alarm and halves the learning rate,
at most `--max_rollbacks` (3) times. After that the lr stays at 1/8 and alarms no longer roll back.
This reports where those cuts fall against the line's own end and his round-16 x, because a line
that does not cross may have spent most of the ladder at 1/8 of the lr.

Rows are merged as in merge.line_view (a resumed segment supersedes the rows past its resume point;
every row kept counts toward x, rolled-back iterations too). merge.py itself is left alone: the
running supervisor imports it.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import merge as M  # noqa: E402


def merged_rows(line: str, segs: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for j, seg in enumerate(segs):
        root = M.seg_root(line, seg)
        if root is None:
            continue
        rs = [r for r in M._rows(root / "metrics.jsonl") if "sampled_gen_tokens" in r]
        if j + 1 < len(segs):
            m = re.search(r"resume_it(\d+)\.pt$", segs[j + 1].get("resume") or "")
            i0 = int(m.group(1)) if m else 0
            rs = [r for r in rs if r["iter"] <= i0]
        rows += rs
    return rows


def _kind(alarm: str) -> str:
    """The alarm's type: its first word ('canary', 'clip-low', 'grad-norm', ...)."""
    return re.split(r"[ :]", alarm.strip())[0].lower()


def timeline(line: str, segs: list[dict]) -> dict:
    """rollbacks [{it, x_dan, n, lr_scale, kinds}] (kinds: the alarm types on the row that rolled
    back); the final lr scale and the x at which it was reached; the alarms: how many iterations
    alarmed, of which kinds, and the iteration from which every later iteration alarmed (None if
    the line's last iteration did not alarm)."""
    lad = C.dan_ladder(C.LINES[line]["ladder"])
    D = lad["D"]
    x_cap = lad["x"][15]
    x = 0.0
    rollbacks: list[dict] = []
    alarmed: list[tuple[int, float, list[str]]] = []
    rows = merged_rows(line, segs)
    for r in rows:
        x += C.x_dan_row(r, D)
        rb = r.get("rollback")
        if rb:
            rollbacks.append(
                {
                    "it": r["iter"],
                    "x_dan": x,
                    "n": rb.get("n"),
                    "lr_scale": rb.get("lr_scale"),
                    "kinds": sorted({_kind(a) for a in r.get("alarms") or []}),
                }
            )
        if r.get("alarms"):
            alarmed.append((r["iter"], x, list(r["alarms"])))
    x_end = x
    scale = rows[-1].get("lr_scale", 1.0) if rows else 1.0
    last = rollbacks[-1] if rollbacks else None
    # the unbroken run of alarming iterations that reaches the line's last iteration
    since = None
    run = 0
    if rows and rows[-1].get("alarms"):
        i = len(rows) - 1
        while i > 0 and rows[i - 1].get("alarms"):
            i -= 1
        since = rows[i]["iter"]
        run = len(rows) - i
    since_x = next((xx for it, xx, _ in alarmed if it == since), None)
    kinds: dict[str, int] = {}
    for _, _, al in alarmed:  # iterations per kind (one iteration can raise several of a kind)
        for k in {_kind(a) for a in al}:
            kinds[k] = kinds.get(k, 0) + 1
    x_final = last["x_dan"] if last else None
    return {
        "iters": len(rows),
        "x_end": x_end,
        "x_cap": x_cap,
        "end_of_ladder": x_end / x_cap if x_cap else None,
        "rollbacks": rollbacks,
        "lr_scale_end": scale,
        "x_at_final_scale": x_final,
        "share_of_line_at_final_scale": (x_end - x_final) / x_end if x_final and x_end else 0.0,
        "share_of_ladder_at_final_scale": (x_cap - x_final) / x_cap if x_final else 0.0,
        "alarm_iters": len(alarmed),
        "alarm_kinds": kinds,
        "alarming_since_it": since,
        "alarming_since_x": since_x,
        "alarming_run": run,
    }


def main() -> None:
    st = json.loads((C.KD / "sup" / "state.json").read_text())
    for line in sys.argv[1:] or sorted(C.LINES, key=lambda k: C.LINES[k]["prio"]):
        if line not in st or not st[line]["segs"]:
            continue
        t = timeline(line, st[line]["segs"])
        rb = (
            ", ".join(
                f"it {b['it']} x {b['x_dan']:.2f} {'+'.join(b['kinds']) or '?'}"
                for b in t["rollbacks"]
            )
            or "none"
        )
        since = (
            f"it {t['alarming_since_it']} x {t['alarming_since_x']:.2f}"
            if t["alarming_since_it"] is not None
            else "-"
        )
        print(
            f"{line}: {t['iters']} rows, x {t['x_end']:.2f} of {t['x_cap']:.2f} "
            f"({100 * t['end_of_ladder']:.0f} %); lr x{t['lr_scale_end']:g}; rollbacks: {rb}; "
            f"alarm iters {t['alarm_iters']} {t['alarm_kinds']}; alarming since {since}"
        )


if __name__ == "__main__":
    main()
