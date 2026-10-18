"""The targets of the statement-size mini knob (line.sh `grpo_think_vs_mini_ksize`, Leon
2026-10-18): the mini subsets with a fixed half of their rows replaced by their big-statement twins.

    python -m code.experiments.current.thinking_traces_20261005.grpo_v2.build_size [--out DIR]

The twins (`runs/pools/size_twins/`, branch leon/size-twins: the same problem under one uniform
substitution, a bigger statement, the same proof) are named `sz_<orig name>`, with `orig_name`.
A row is in the twinned half when `half(name)`: the first byte of sha256(`SALT` + its name) is
even, a fixed function of the original name alone (never a run seed), so every seed of the knob
trains on the same mix. A chosen row is replaced, in place and byte for byte, by its twin's line
when it has one (4 of the L2 subset's rows have none and stay original). A 3-premise extra (whose
`orig_name` is its L3 parent) is replaced by its own twin exactly when its parent was, as the
twins of the 3-premise rows use their parent's substitution. Rows keep their file order, so each
mixed file has its subset's row count.

Then line.sh's two preflights on the mixed files (`grpo_v2/preflight.py`, the rules of
build_mini.py: the L2 and L3 files pass outright; the L2 file with the 3-premise file fails only
its band line), and a direct check that no mixed row shares a prompt, `key` or `pkey` with the
twin test sets the knob reads (`L{2,3}_test_1k_sz.jsonl`; their originals, the matched test_1k
rows, are among the preflight's evaluation sets). Writes into DIR (default runs/pools/mini):

    L2_mini_size.jsonl  L3_mini_size.jsonl  L3_3p_mini_size.jsonl   the mixed files
    size.json                         the twin share per file, statement-token distributions, md5s
    preflight_size.log, preflight_size_extra.log
    preflight_pass_size.txt           only if all passed: `PREFLIGHT PASSED <md5> <file>` per file

Its own pass record, so build_mini.py's rebuilds never touch it. Everything is written to
DIR.size.tmp first and moved in file by file (each move atomic), the pass record last. line.sh
refuses `_ksize` unless the record names each mixed file's md5 as it is. Exits 1 if a check fails
(no pass record). Written 2026-10-18.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import statistics
import sys
from pathlib import Path

from .build_mini import N_OK, md5, passed, run_preflight
from .preflight import R

MINI = R / "pools/mini"
TWINS = R / "pools/size_twins"
SALT = "nd-rl size knob 2026-10-18 half:"  # the half's hash prefix (not a run seed)
FILES = (  # (subset, its twins, the mixed file)
    ("L2_mini.jsonl", "L2_mini_sz.jsonl", "L2_mini_size.jsonl"),
    ("L3_mini.jsonl", "L3_mini_sz.jsonl", "L3_mini_size.jsonl"),
    ("L3_3p_mini.jsonl", "L3_3p_mini_sz.jsonl", "L3_3p_mini_size.jsonl"),
)
TESTS = ("L2_test_1k_sz.jsonl", "L3_test_1k_sz.jsonl")  # the twin test sets the knob reads


def half(name: str) -> bool:
    """The original row `name` is in the twinned half."""
    return hashlib.sha256((SALT + name).encode()).digest()[0] % 2 == 0


def lines(p: Path) -> list[str]:
    return [x for x in p.read_text().split("\n") if x.strip()]


def stmt_tokens(rows: list[dict]) -> dict[str, object]:
    """Statement size: whitespace tokens of the statement (`prompt_tokens`: the prompt's but THM,
    SEQ and PRF), mean, p10 / p50 / p90, max, and the share of rows per 10-token bin."""
    t = sorted(int(r["prompt_tokens"]) for r in rows)
    q = statistics.quantiles(t, n=10, method="inclusive")
    bins = collections.Counter(10 * (x // 10) for x in t)
    return {
        "rows": len(t),
        "mean": round(sum(t) / len(t), 1),
        "p10_p50_p90": [q[0], q[4], q[8]],
        "max": t[-1],
        "pct_per_10": {f"{b}-{b + 9}": round(100 * n / len(t), 1) for b, n in sorted(bins.items())},
    }


def mix(
    sub: list[str], twins: list[str], parent: bool
) -> tuple[list[str], list[bool], list[dict], list[dict]]:
    """-> (the mixed lines, per row whether it is a twin, the original rows, the mixed rows).
    `parent`: the 3-premise extras, chosen by their parent's name (`orig_name`)."""
    tw = {json.loads(x)["orig_name"]: x for x in twins}
    orig = [json.loads(x) for x in sub]
    out, is_tw = [], []
    for x, r in zip(sub, orig, strict=True):
        pick = half(r["orig_name"] if parent else r["name"]) and r["name"] in tw
        out.append(tw[r["name"]] if pick else x)
        is_tw.append(pick)
    return out, is_tw, orig, [json.loads(x) for x in out]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=MINI)
    a = ap.parse_args(argv)
    rec = (MINI / "preflight_pass.txt").read_text()
    for sub, _, _ in FILES:  # the subsets as build_mini.py's pass record names them
        assert f"PREFLIGHT PASSED {md5(MINI / sub)} {sub}\n" in rec, f"{sub}: not as recorded"
    mixed: dict[str, list[str]] = {}
    stats: dict[str, object] = {}
    l3_twinned: set[str] = set()
    for sub, tws, out in FILES:
        three = sub.startswith("L3_3p")
        rows, is_tw, orig, new = mix(lines(MINI / sub), lines(TWINS / tws), three)
        mixed[out] = rows
        if sub == "L3_mini.jsonl":
            l3_twinned = {r["name"] for r, t in zip(orig, is_tw, strict=True) if t}
        if three:  # the extras follow their parent: twinned exactly when the parent row is
            assert all(
                t == (r["orig_name"] in l3_twinned) for r, t in zip(orig, is_tw, strict=True)
            ), "a 3-premise twin that does not follow its parent"
        stats[out] = {
            "rows": len(rows),
            "twins": sum(is_tw),
            "twin_share": round(sum(is_tw) / len(rows), 4),
            "chosen_without_twin": sum(
                half(r["orig_name"] if three else r["name"]) and not t
                for r, t in zip(orig, is_tw, strict=True)
            ),
            "statement_tokens": {
                "subset": stmt_tokens(orig),
                "mixed": stmt_tokens(new),
                "mixed_originals": stmt_tokens(
                    [r for r, t in zip(new, is_tw, strict=True) if not t]
                ),
                "mixed_twins": stmt_tokens([r for r, t in zip(new, is_tw, strict=True) if t]),
            },
        }
    test_rows = [json.loads(x) for f in TESTS for x in lines(TWINS / f)]
    keys = ("prompt", "key", "pkey")
    seen = {k: {str(r[k]) for r in test_rows if r.get(k)} for k in keys}
    shared = {
        k: sum(str(json.loads(x).get(k)) in seen[k] for rows in mixed.values() for x in rows)
        for k in keys
    }
    tmp = Path(f"{a.out}.size.tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    for f, rows in mixed.items():
        (tmp / f).write_text("".join(x + "\n" for x in rows))
    f2, f3, fx = (tmp / out for _, _, out in FILES)
    rc, log = run_preflight(f2, f3)
    rcx, logx = run_preflight(f2, fx)
    (tmp / "preflight_size.log").write_text(log)
    (tmp / "preflight_size_extra.log").write_text(logx)
    pre = {
        "exit": rc,
        "passed": passed(log, False) and rc == 0,
        "extra_exit": rcx,
        "extra_passed": passed(logx, True),
        "shared_with_twin_tests": shared,
    }
    ok = bool(pre["passed"] and pre["extra_passed"]) and not any(shared.values())
    files = {f: {"rows": len(rows), "md5": md5(tmp / f)} for f, rows in mixed.items()}
    src = [MINI / s for s, _, _ in FILES] + [TWINS / t for _, t, _ in FILES]
    src += [TWINS / t for t in TESTS]
    summary = {
        "out": str(a.out),
        "salt": SALT,
        "files": files,
        "mix": stats,
        "sources": {str(p): md5(p) for p in src},
        "preflight": pre,
    }
    (tmp / "size.json").write_text(json.dumps(summary, indent=1) + "\n")
    a.out.mkdir(parents=True, exist_ok=True)
    pp = a.out / "preflight_pass_size.txt"
    if not ok:  # no record of files that failed
        pp.unlink(missing_ok=True)
    for f in tmp.iterdir():  # each move atomic; the pass record last
        f.replace(a.out / f.name)
    tmp.rmdir()
    if ok:
        (a.out / "preflight_pass_size.txt.tmp").write_text(
            "".join(f"PREFLIGHT PASSED {x['md5']} {f}\n" for f, x in files.items())
            + "preflight.py on the mixed L2 and L3 files: passed; on the mixed L2 and 3-premise "
            + f"files: only the band line failed; {N_OK} evaluation-set lines ok in each "
            + "(preflight_size*.log); no prompt, key or pkey shared with the twin test sets\n"
        )
        (a.out / "preflight_pass_size.txt.tmp").replace(pp)
    print(json.dumps(summary, indent=1))
    print("size mix preflight", "PASSED" if ok else "FAILED", f"(see {a.out})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
