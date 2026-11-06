"""numbers.json / numbers.md for knockout_danbase, from the supervisor's state and the segments.

    python3 tables.py          # serrano; reads KD/sup/state.json, KD/runs, KD/collected, KD/backup

Per line (merge.line_view): the matched draws on Dan's measure and its first crossing, the
snapshot curve (first-32 count, E32), compute on both rules, GPU-h, lost iterations, the stop
reason, and the σ share of the distinct Lean-accepted training proofs per iteration (Dan's
coe_sigma.flags on each segment's LEAN_GATE_DUMP, rows assigned to iterations by walking the
segment's log: a `[lean_gate] … distinct N` line, then `it K:`). Dan's own ladders (dan_axis.json)
are carried for the figure.
"""

# ruff: noqa: E501
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import guard as GD  # noqa: E402
import merge as M  # noqa: E402

GATE = re.compile(r"^\[lean_gate\] \d+ samples.*distinct (\d+):")
IT = re.compile(r"^it (\d+):")


def _flags():
    sys.path.insert(0, str(C.KD / "dan_code_13e41bbd"))
    sys.path.insert(0, str(C.KD / "dan_code_13e41bbd" / "coe"))
    from coe_sigma import flags  # type: ignore[import-not-found]

    return flags


def seg_log(line: str, seg: dict) -> Path | None:
    k = seg["seg"]
    for p in (C.KD / "logs" / f"{line}_seg{k}.log" if seg["host"] == M.HOST else None,
              C.KD / "collected" / seg["host"] / "logs" / f"{line}_seg{k}.log"):  # fmt: skip
        if p is not None and p.exists():
            return p
    return None


def sigma_by_iter(line: str, segs: list[dict], keep: dict[int, int]) -> dict[int, dict]:
    """iteration -> {accepted, sigma, dn_any} over the distinct accepted proofs of that iteration;
    `keep`: segment -> its last kept iteration (merge.py's superseded rows are dropped)."""
    flags = _flags()
    out: dict[int, dict] = {}
    for seg in segs:
        root, lg = M.seg_root(line, seg), seg_log(line, seg)
        if root is None or lg is None or not (root / "lean_dump.jsonl").exists():
            continue
        dump = (root / "lean_dump.jsonl").read_text().splitlines()
        pos, cur = 0, 0
        for x in lg.read_text(errors="replace").splitlines():
            m = GATE.match(x)
            if m:
                cur += int(m.group(1))
                continue
            m = IT.match(x)
            if m:
                it = int(m.group(1))
                rows = dump[pos : pos + cur]
                pos, cur = pos + cur, 0
                if it > keep.get(seg["seg"], 10**9) or len(rows) == 0:
                    continue
                n = sg = dn = 0
                for r in rows:
                    try:
                        d = json.loads(r)
                    except json.JSONDecodeError:
                        continue
                    if d.get("lean_ok") not in (True, "True") or not d.get("nd"):
                        continue
                    f = flags(d["nd"])
                    if f is None:
                        continue
                    n += 1
                    sg += bool(f["w"]["sigma_chain"] or f["w"]["sigma_strict"])
                    dn += bool(f["w"]["dn_any"])
                out[it] = {"accepted": n, "sigma": sg, "dn_any": dn}
    return out


def main() -> None:
    st = json.loads((C.KD / "sup" / "state.json").read_text())
    res: dict = {"bar": C.BAR, "lines": {}, "dan": json.loads(C.DAN_AXIS.read_text())}
    md = ["# knockout_danbase numbers\n",
          f"Dan's measure: cumulative k 32 count of the 106 over 16 matched draws (merge.py), bar {C.BAR}. "
          "x_dan: his rule (prefill included); x_ours: knockout_grpo's R.\n",
          "| line | arm | base first-32 | Dan's measure by round | first crossing (round, x_dan, x_ours, GPU-h) "
          "| last read (it, x_dan, first-32, E32) | σ share of accepted training proofs | status |",
          "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for line in sorted(C.LINES, key=lambda x: C.LINES[x]["prio"]):
        s = st.get(line, {"segs": []})
        if not s["segs"]:
            continue
        final = s["status"] == "done"
        v = M.line_view(line, s["segs"], final=final)
        keep = {}
        for j, seg in enumerate(s["segs"][:-1]):
            m = re.search(r"resume_it(\d+)\.pt$", s["segs"][j + 1].get("resume") or "")
            keep[seg["seg"]] = int(m.group(1)) if m else 0
        sig = sigma_by_iter(line, s["segs"], keep)
        x_by_it = {c["it"]: c["x_dan"] for c in v["curve"]}
        v["sigma"] = [{"it": it, **sig[it]} for it in sorted(sig)]
        v["status"], v["stop_reason"] = s["status"], s.get("stop_reason")
        v["arm"] = C.LINES[line]["arm"]
        v["dose"] = C.LINES[line].get("dose")
        v["held"] = M.held(v["curve"])
        v["guard"] = GD.timeline(line, s["segs"])
        res["lines"][line] = v
        base = next((c for c in v["curve"] if c["it"] == 0), {})
        cr = v["cross_dan"]
        crs = (f"r{cr['round']}, {cr['x_dan']:.3f}, {cr['x_ours']:.4f}" if cr else
               f"not reached by x_dan {v['x_dan']:.2f}")  # fmt: skip
        last = v["curve"][-1] if v["curve"] else {}
        tot = sum(x["accepted"] for x in v["sigma"])
        sg = sum(x["sigma"] for x in v["sigma"])
        tail = v["sigma"][-20:]
        tn, ts = sum(x["accepted"] for x in tail), sum(x["sigma"] for x in tail)
        sigs = (f"{100 * sg / max(1, tot):.2f} % overall; {100 * ts / max(1, tn):.2f} % over the "
                f"last {len(tail)} iterations" if tot else "—")  # fmt: skip
        md.append(f"| {line} | {v['arm']} | {base.get('first32')} | {[m['cum'] for m in v['matched']]} | "
                  f"{crs} | it {last.get('it')}, {last.get('x_dan', 0):.3f}, {last.get('first32')}, "
                  f"{(last.get('e32') or 0):.1f} | {sigs} | {s['status']}"
                  f"{' (' + s['stop_reason'] + ')' if s.get('stop_reason') else ''}; GPU-h {v['gpuh']:.1f}; "
                  f"lost {v['lost']} |")  # fmt: skip
        del x_by_it
    (C.HERE / "numbers.json").write_text(json.dumps(res, indent=1, default=str))
    (C.HERE / "numbers.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
