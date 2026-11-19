"""RL variants of the two-authors lines (PREREG Amendment 3, 2026-11-18): a style bonus, and the unlikeliness term off.

    python rlvar.py grpo --var {styA,ub0} -- --tag none -- <code.thinking.grpo arguments>   # the GRPO process
    python rlvar.py pool [--limit N]          # share of the RL pool's reference proofs where the bonus can fire

Lines `rl_<corpus>_<var>_s<S>` (authors_launch.register_lines) run the running lines' recipe (tag_line.grpo_argv: the
knockout dd1_s<S> recipe on our base, ND_GUARD=off) with only these changes:

  ub0   `--unlikely_beta 0` (the running lines have 0.25: a solved attempt loses beta x its within-group likelihood rank
        / n, up to 0.234). Ordinary reward: 1 if Lean accepts the proof, else 0; `--rej_penalty 0.1` kept.
  styA  ub0, plus a style bonus on Lean-accepted proofs: + BONUS x (the share of the proof's reiteration sites written
        in author A's form, restated inside the box), and nothing on a proof that takes the reductio route. The
        reading is `authors_profile.profile` (the measurement's own detector: `authors_corpus.detect` on the proof with
        dead lines pruned; `route` = sigma+ following R lines). A proof with no reiteration site gets nothing (it has no
        room for the habit), and a proof the profile cannot parse gets nothing.

Nothing in the GRPO snapshot or in ../author_tags_20261110 is edited. The GRPO process is `tag_grpo.main` (tags,
tokenizer, spies, unchanged) behind this wrapper, which (styA only) replaces `code.thinking.grpo.rewards` in that one
process by `rewards` + bonus. `group_items` and the batch dump call `rewards` by its module name, so the bonus enters
the advantage (r - group mean, Dr. GRPO) and the outcome-signal rule: an all-solved group whose proofs differ in the
habit now gives a gradient. Every other line imports nothing from here (tests: test_rlvar.py).

Files the wrapper writes into the GRPO `--out`: `rlvar.json` (variant, bonus, detector files and their sha256) and
`style_bonus.jsonl` (styA: per sampling call, i.e. per iteration, the accepted attempts, those with a reiteration site,
route proofs, the bonus paid and its mean).
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import atexit
import collections
import json
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
TAGS_DIR = HERE.parent / "author_tags_20261110"
for _p in (str(TAGS_DIR), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

BONUS = 0.3  # PREREG Amendment 3 (the derivation is there)
VARIANTS: dict[str, dict[str, Any]] = {
    "styA": {"unlikely_beta": "0", "bonus": BONUS, "habit": "reit"},
    "ub0": {"unlikely_beta": "0", "bonus": 0.0, "habit": None},
}
ITERS = 60  # Amendment 3: the variant lines stop by iteration count
KEY = "styA_bonus"  # per attempt dict: the bonus, computed once


def _set(argv: list[str], flag: str, value: str) -> None:
    assert argv.count(flag) == 1, flag
    argv[argv.index(flag) + 1] = value


def variant_argv(argv: list[str], var: str) -> list[str]:
    """The running lines' GRPO argv with the variant's one change (`--unlikely_beta 0`)."""
    v = VARIANTS[var]
    out = list(argv)
    assert out[out.index("--unlikely_beta") + 1] == "0.25", "the running recipe changed"
    _set(out, "--unlikely_beta", v["unlikely_beta"])
    return out


def variant_cmd(tag_grpo_cmd: list[str], var: str) -> list[str]:
    """[PY, tag_grpo.py, *rest] -> [PY, rlvar.py, grpo, --var V, --, *rest]."""
    assert Path(tag_grpo_cmd[1]).name == "tag_grpo.py", tag_grpo_cmd[:2]
    assert var in VARIANTS, var
    return [tag_grpo_cmd[0], str(HERE / "rlvar.py"), "grpo", "--var", var, "--", *tag_grpo_cmd[2:]]


# ------------------------------------------------------------------------------------------------ the bonus
def reit_bonus(nd: str, bonus: float = BONUS) -> tuple[float, str]:
    """(bonus, why) for one Lean-ACCEPTED proof: bonus x (A-form reiteration sites / sites); 0 on a route proof, a
    proof with no reiteration site, or one the profile cannot read. The caller pays it only when Lean accepted."""
    import authors_profile as AP

    p = AP.profile("", nd)
    if not p.get("ok"):
        return 0.0, "unparsed"
    # True: the reductio route; None: Dan's parser could not read it
    if p.get("route") is not False:
        return 0.0, "route" if p.get("route") else "route_unknown"
    s, a = p["reit"]["sites"], p["reit"]["a"]
    if not s:
        return 0.0, "no_site"
    return bonus * a / s, "paid" if a else "site_B"


class Bonus:
    """The per-attempt bonus with its counters (one instance per GRPO process)."""

    def __init__(self, bonus: float, out: Path | None) -> None:
        self.bonus, self.out = bonus, out
        self.c: collections.Counter = collections.Counter()
        self.paid = 0.0
        self.n_call = 0

    def of(self, x: dict[str, Any]) -> float:
        if not x.get("ok"):
            return 0.0
        if KEY not in x:
            b, why = reit_bonus(x["nd"], self.bonus)
            x[KEY] = b
            self.c["accepted"] += 1
            self.c[why] += 1
            self.paid += b
        return x[KEY]

    def flush(self) -> None:
        if self.out is None or not self.c:
            self.c, self.paid = collections.Counter(), 0.0
            return
        row = {"call": self.n_call, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **self.c,
               "bonus_sum": round(self.paid, 4), "bonus_mean_accepted": round(self.paid / max(1, self.c["accepted"]), 5)}  # fmt: skip
        with open(self.out / "style_bonus.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        self.c, self.paid = collections.Counter(), 0.0


def install(grpo: Any, bonus: float, out: Path | None) -> Bonus:
    """Replace `grpo.rewards` (bonus on top of every accepted attempt's reward) and count per sampling call."""
    B = Bonus(bonus, out)
    orig = getattr(grpo.rewards, "_rlvar_orig", grpo.rewards)

    def rewards(att, len_penalty, *args, **kw):
        r = orig(att, len_penalty, *args, **kw)
        return [ri + B.of(x) for ri, x in zip(r, att, strict=True)]

    rewards._rlvar_orig = orig  # type: ignore[attr-defined]
    grpo.rewards = rewards
    sg = getattr(grpo.sample_groups, "_rlvar_orig", grpo.sample_groups)

    def sample_groups(*args, **kw):
        B.flush()  # the previous call's attempts are all scored by now
        B.n_call += 1
        return sg(*args, **kw)

    sample_groups._rlvar_orig = sg  # type: ignore[attr-defined]
    grpo.sample_groups = sample_groups
    atexit.register(B.flush)
    return B


# ------------------------------------------------------------------------------------------------ entry points
def grpo_main(argv: list[str]) -> int:
    k = argv.index("--")
    ap = argparse.ArgumentParser(prog="rlvar.py grpo")
    ap.add_argument("--var", required=True, choices=sorted(VARIANTS))
    a = ap.parse_args(argv[:k])
    rest = argv[k + 1 :]
    j = rest.index("--")
    g = argparse.ArgumentParser(add_help=False)
    g.add_argument("--out", required=True)
    g.add_argument("--unlikely_beta", type=float, required=True)
    ga, _ = g.parse_known_args(rest[j + 1 :])
    v = VARIANTS[a.var]
    assert ga.unlikely_beta == float(v["unlikely_beta"]), (
        f"variant {a.var} needs --unlikely_beta {v['unlikely_beta']}, got {ga.unlikely_beta}"
    )
    import tag_core as T
    import tag_grpo

    T.install_thinking()
    from code.thinking import grpo  # type: ignore[import-not-found]

    out = Path(ga.out)
    out.mkdir(parents=True, exist_ok=True)
    info: dict[str, Any] = {"variant": a.var, **v, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "grpo_file": grpo.__file__}  # fmt: skip
    if v["bonus"]:
        import authors_corpus as AC
        import authors_profile as AP

        info["detector"] = {Path(m.__file__).name: T.sha256(m.__file__) for m in (AP, AC, AC.H)}
        install(grpo, v["bonus"], out)
    (out / "rlvar.json").write_text(json.dumps(info, indent=1))
    print(f"rlvar: variant {a.var} ({v})", flush=True)
    return tag_grpo.main(rest)


def pool_main(argv: list[str]) -> None:
    """Where the bonus can fire on the RL pool. The pool has no reference proofs, so this reads the Lean-accepted
    proofs of a line's GRPO dump (`lean_dump.jsonl`: prompt, nd, lean_ok), its first `--limit` rows (the first
    iterations, near the base policy): per accepted proof and per pool theorem (prompt), the share with a reiteration
    site and not the route (eligible), and what `reit_bonus` pays."""
    ap = argparse.ArgumentParser(prog="rlvar.py pool")
    ap.add_argument("dump")
    ap.add_argument("--limit", type=int, default=8192)
    a = ap.parse_args(argv)
    c: collections.Counter = collections.Counter()
    thm: dict[str, list[bool]] = collections.defaultdict(list)
    with open(a.dump) as f:
        for i, line in enumerate(f):
            if a.limit and i >= a.limit:
                break
            r = json.loads(line)
            c["rows"] += 1
            if not r.get("lean_ok"):
                continue
            b, why = reit_bonus(r["nd"])
            c["accepted"] += 1
            c[why] += 1
            c["bonus_sum"] += b
            thm[r["prompt"]].append(why in ("paid", "site_B"))
    out: dict[str, Any] = dict(c)
    n = c["accepted"]
    out["eligible_pct_of_accepted"] = round(100 * (c["paid"] + c["site_B"]) / max(1, n), 1)
    out["paid_pct_of_accepted"] = round(100 * c["paid"] / max(1, n), 1)
    out["mean_bonus_accepted"] = round(c["bonus_sum"] / max(1, n), 4)
    out["theorems_with_accepted"] = len(thm)
    out["theorems_any_eligible_pct"] = round(
        100 * sum(any(v) for v in thm.values()) / max(1, len(thm)), 1
    )
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "grpo":
        sys.exit(grpo_main(sys.argv[2:]))
    elif cmd == "pool":
        pool_main(sys.argv[2:])
    else:
        raise SystemExit(__doc__)
