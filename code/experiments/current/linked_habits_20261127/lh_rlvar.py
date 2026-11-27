"""The GRPO arms of linked habits version 2 (SPEC2.md, Stage 2): the RLHF arm and its no-preference line, through
W2's RL-variant hook; and the RL pool's largest premise class (the note rater's class).

    python lh_rlvar.py grpo --var {anc,rlhfreit} -- --tag none -- <grpo arguments>   # GRPO process (rlhfatom: not scheduled, Amendment 4)
    python lh_rlvar.py pool [--dry-run]       # the RL pool's premise table and its largest class -> LH/data

Lines (lh_core.line_name): rl_<c>_anc_s<S> and rl_<c>_rlhf<rater>_s<S>. Both run W2's Amendment 3 recipe
(../two_authors_20261111/rlvar.py `variant_argv`: the running lines' recipe, `--unlikely_beta 0`, ND_GUARD=off, 60
iterations, untagged prompts on Dan's RL pool) with these changes, made here to W2's `rlvar.variant_argv` in the
`_line` process only:
  --max_steps 51          SPEC2's step limit (48 before)
  --kl_anchor_coef 0.02   the trainer's own KL anchor (code.thinking.grpo: beta x KL(pi_ref || pi_theta) on teacher-
  --kl_anchor_ref BASE    forced steps at every AdamW step), to the line's base (tag_line.grpo_argv already sets the ref)
  --kl_anchor_data FILE   LH/data/anchor_<c>.jsonl: a canary/1 file of the corpus's untagged states (lh_pairs.py anchor)
  --snap_iters 10,20,40,60 (lh_core.GRPO_SNAPS; PREREG Amendment 4)
  anc        nothing else: the no-preference run (anchor, no learned reward).
  rlhf<r>    reward = Lean acceptance x (1 + 0.3 x sigmoid(RM(prompt, proof))): W2's hook (`rlvar.install`: GRPO's
             `rewards` + a per-attempt bonus, so the bonus enters the advantage) with `rlvar.Bonus` replaced by
             `RMBonus`: 0.3 x sigmoid of the reward model's score of the accepted proof's ND text, paid only when
             lh_profile finds at least one reiteration site in it (PREREG Amendment 7 point 6); nothing on a rejected
             attempt, a proof without a site, or one the detector or the reward model cannot read. The reward model is LH/arms/rm_<c>_<r>_s<S>/rm.pt, loaded by the third agent's
             `lh_rm.load(path, device)`: `.score(prompt, proof_text) -> float`, `.score_many([(prompt, proof), ...])`).
             The scores of a sampling call are computed once, right after the call returns
             (prompts attached from the call's targets), on the policy's device.
Files written into the GRPO `--out`: lh_rlvar.json (variant, anchor file and its sha256, reward model and its sha256,
step limit) and, rlhf only, rm_bonus.jsonl (per sampling call, with its GRPO iteration and redraw round: accepted
attempts, unreadable_profile, no_reit_site, scored, unreadable_rm, paid, mean sigmoid of the paid, bonus paid).
W2's rlvar.py and everything under ../author_tags_20261110 are imported, never edited.
"""

# ruff: noqa: E501, I001
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lh_core as C  # noqa: E402  (TA/AT = LH, sys.path)
import notes as N  # noqa: E402
import rlvar  # noqa: E402  (W2's RL variants, unchanged)

HERE = C.HERE
KEY = "lh_rm_bonus"  # per attempt dict: the bonus, computed once per sampling call
for _k in C.LINE_KINDS:
    rlvar.VARIANTS[_k] = {"unlikely_beta": "0", "bonus": 0.0 if _k == "anc" else C.RM_BONUS,
                          "habit": None if _k == "anc" else C.RATERS[_k[4:]], "lh": True}  # fmt: skip


def _set(argv: list[str], flag: str, value: str) -> None:
    assert argv.count(flag) == 1, flag
    argv[argv.index(flag) + 1] = value


def line_corpus(argv: list[str]) -> str:
    return C.parse_task(argv[argv.index("--name") + 1])["corpus"]


_W2_VARGV = getattr(rlvar.variant_argv, "_lh_orig", rlvar.variant_argv)
_W2_VCMD = getattr(rlvar.variant_cmd, "_lh_orig", rlvar.variant_cmd)


def variant_argv(argv: list[str], var: str) -> list[str]:
    """W2's variant argv (`--unlikely_beta 0`); for our variants also the step limit, the anchor and the snapshots."""
    out = _W2_VARGV(argv, var)
    if var not in C.LINE_KINDS:
        return out
    _set(out, "--max_steps", str(C.STEP_LIMIT))
    _set(out, "--kl_anchor_coef", str(C.KL_ANCHOR))
    assert out[out.index("--kl_anchor_ref") + 1] == out[out.index("--init") + 1], (
        "the anchor's reference is the base"
    )
    assert "--kl_anchor_data" not in out
    out += ["--kl_anchor_data", str(C.anchor_path(line_corpus(out)))]
    _set(out, "--snap_iters", ",".join(str(i) for i in C.GRPO_SNAPS))
    return out


def variant_cmd(tag_grpo_cmd: list[str], var: str) -> list[str]:
    """Our variants' GRPO process is this file; W2's run W2's rlvar.py."""
    if var not in C.LINE_KINDS:
        return _W2_VCMD(tag_grpo_cmd, var)
    assert Path(tag_grpo_cmd[1]).name == "tag_grpo.py", tag_grpo_cmd[:2]
    return [
        tag_grpo_cmd[0],
        str(HERE / "lh_rlvar.py"),
        "grpo",
        "--var",
        var,
        "--",
        *tag_grpo_cmd[2:],
    ]


variant_argv._lh_orig = _W2_VARGV  # type: ignore[attr-defined]
variant_cmd._lh_orig = _W2_VCMD  # type: ignore[attr-defined]
rlvar.variant_argv = variant_argv
rlvar.variant_cmd = variant_cmd


# ------------------------------------------------------------------------------------------------ the RM bonus
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


_PROF: dict[str, Any] = {}


def reit_sites(prompt: str, proof: str) -> int | None:
    """Reiteration sites lh_profile finds in a proof; None when it cannot read the proof (as lh_measure counts it)."""
    if "m" not in _PROF:
        import lh_profile

        _PROF["m"], _PROF["t"] = lh_profile, C.profile_tables()
    try:
        p = _PROF["m"].profile(prompt, proof, _PROF["t"])
    except Exception:  # profile never raises by its contract; a raise is still an unreadable proof
        return None
    if not p.get("ok"):
        return None
    return int(((p.get("habits") or {}).get(C.RATERS["reit"]) or {}).get("sites") or 0)


def rm_readable(rm: Any, items: list[tuple[str, str]]) -> list[bool]:
    """Which items the reward model could read: its own per-item flag when it gives one (`last_ok` after
    score_many), else the encoding it applies (lh_tune.encode_proof raising = unreadable, scored 0.0 and counted)."""
    ok = getattr(rm, "last_ok", None)
    if isinstance(ok, list) and len(ok) == len(items):
        return [bool(x) for x in ok]
    import lh_tune as U

    out = []
    for p, nd in items:
        try:
            U.encode_proof(rm.tok, p, nd)
            out.append(True)
        except Exception:
            out.append(False)
    return out


class RMBonus(rlvar.Bonus):
    """W2's per-attempt bonus with the reward model's score in place of the style reading. `score_call` scores every
    accepted attempt of a sampling call at once; `of` (W2's hook, per attempt) returns the cached bonus."""

    rm: Any = None
    rm_path: Path | None = None
    device = "cpu"

    def __init__(self, bonus: float, out: Path | None) -> None:
        super().__init__(bonus, out)
        self.sig = 0.0
        self.it_round: tuple[int | None, int | None] = (None, None)

    def load(self) -> Any:
        if RMBonus.rm is None:
            import lh_rm

            RMBonus.rm = lh_rm.load(
                str(RMBonus.rm_path), RMBonus.device, max_tokens=C.SHARED_MAX_TOKENS
            )
        return RMBonus.rm

    def score_call(self, per: list[list[dict[str, Any]]], targets: list[dict[str, Any]]) -> None:
        """PREREG Amendment 7 point 6: the bonus 0.3 x sigmoid(RM) is paid only on an accepted attempt in which
        lh_profile finds at least one reiteration site; none on a proof the detector cannot read or the reward model
        reports as unreadable. Counts per sampling call (rm_bonus.jsonl): accepted, unreadable_profile, no_reit_site,
        scored (sent to the reward model), unreadable_rm, paid."""
        todo = [
            (x, t["prompt"])
            for grp, t in zip(per, targets, strict=True)
            for x in grp
            if x.get("ok") and KEY not in x
        ]
        if not todo:
            return
        site = []
        for x, p in todo:
            self.c["accepted"] += 1
            x[KEY] = 0.0
            n = reit_sites(p, x["nd"])
            if n is None:
                self.c["unreadable_profile"] += 1
            elif n == 0:
                self.c["no_reit_site"] += 1
            else:
                site.append((x, p))
        if not site:
            return
        rm = self.load()
        items = [(p, x["nd"]) for x, p in site]
        sc = (
            list(rm.score_many(items))
            if hasattr(rm, "score_many")
            else [float(rm.score(p, nd)) for p, nd in items]
        )
        ok = rm_readable(rm, items)
        self.c["scored"] += len(items)
        for (x, _), s, good in zip(site, sc, ok, strict=True):
            if not good:
                self.c["unreadable_rm"] += 1
                continue
            sg = sigmoid(float(s))
            x[KEY] = self.bonus * sg
            self.c["paid"] += 1
            self.sig += sg
            self.paid += x[KEY]

    def of(self, x: dict[str, Any]) -> float:
        if not x.get("ok"):
            return 0.0
        if KEY not in x:  # an accepted attempt the call-level scoring did not see: score it alone
            if not x.get("lh_prompt"):  # no prompt to read it against (audit R6): no bonus, counted
                self.c["accepted"] += 1
                self.c["no_prompt"] += 1
                x[KEY] = 0.0
            else:
                self.score_call([[x]], [{"prompt": x["lh_prompt"]}])
        return x[KEY]

    def flush(self) -> None:
        if self.out is None or not self.c:
            self.c, self.paid, self.sig = collections.Counter(), 0.0, 0.0
            return
        npaid, nacc = max(1, self.c["paid"]), max(1, self.c["accepted"])
        cnt = {
            k: self.c.get(k, 0)
            for k in (
                "accepted",
                "no_prompt",
                "unreadable_profile",
                "no_reit_site",
                "scored",
                "unreadable_rm",
                "paid",
            )
        }
        row = {"call": self.n_call, "iter": self.it_round[0], "round": self.it_round[1],
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **cnt,
               "sigmoid_mean_paid": round(self.sig / npaid, 5), "bonus_sum": round(self.paid, 4),
               "bonus_mean_accepted": round(self.paid / nacc, 5)}  # fmt: skip
        with open(self.out / "rm_bonus.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        self.c, self.paid, self.sig = collections.Counter(), 0.0, 0.0


_BONUS: list[RMBonus] = []


class _RMB(RMBonus):
    def __init__(self, bonus: float, out: Path | None) -> None:
        super().__init__(bonus, out)
        _BONUS.append(self)


def iter_round(args: tuple) -> tuple[int | None, int | None]:
    """(GRPO iteration, redraw round) of a sampling call, from its arguments (tb, a, seed, ...): grpo.py draws round 0
    with seed a.seed x 100003 + it and round r with that + r x 1,000,003. (None, None) if the call looks different."""
    try:
        off = int(args[2]) - int(args[1].seed) * 100003
        return (off % 1_000_003, off // 1_000_003) if off >= 0 else (None, None)
    except Exception:
        return None, None


def install_rm(grpo: Any, rm_path: Path, device: str) -> None:
    """`rlvar.Bonus` -> RMBonus (W2's `install` builds it), and the sampling call scores its accepted attempts once it
    returns. W2's own sample_groups wrapper (flush, count) goes on top of this one."""
    RMBonus.rm_path, RMBonus.device = rm_path, device
    rlvar.Bonus = _RMB
    orig = grpo.sample_groups

    def sample_groups(model, tok, targets, *args, **kw):
        per = orig(model, tok, targets, *args, **kw)
        for grp, t in zip(per, targets, strict=True):
            for x in grp:
                x.setdefault("lh_prompt", t["prompt"])
        if _BONUS:
            _BONUS[-1].it_round = iter_round(args)
            _BONUS[-1].score_call(per, list(targets))
        return per

    grpo.sample_groups = sample_groups


# ------------------------------------------------------------------------------------------------ the pool's class
def pool_table(pool: Path = C.RL_POOL) -> dict[str, Any]:
    """Premise counts and classes of the RL pool's theorems, and the rule for the largest class."""
    import authors_sets as AS

    rows = [json.loads(x) for x in pool.read_text().splitlines() if x.strip()]
    n_prem: collections.Counter = collections.Counter()
    thm_cls: collections.Counter = collections.Counter()
    prem_cls: collections.Counter = collections.Counter()
    bad = 0
    for r in rows:
        prem, _ = AS.split_thm(r["thm"])
        prem = [p for p in prem if p.strip()]
        if "n_prem" in r and r["n_prem"] != len(prem):
            bad += 1
        n_prem[len(prem)] += 1
        seen = set()
        for p in prem:
            k = N.premise_class(p)
            prem_cls[k] += 1
            seen.add(k)
        for k in seen:
            thm_cls[k] += 1
    n = len(rows)
    classes = [{"cls": k, "theorems": thm_cls[k], "share_pct": round(100 * thm_cls[k] / n, 2), "premises": prem_cls[k]}
               for k in sorted(N.CLASSES, key=lambda k: (-thm_cls[k], N.CLASSES.index(k)))]  # fmt: skip
    top = classes[0]
    rewarded = [top["cls"]] if top["share_pct"] >= 20 else [classes[0]["cls"], classes[1]["cls"]]
    return {"pool": str(pool), "pool_sha256": C.sha256(pool), "theorems": n, "n_prem_mismatch": bad,
            "premise_counts": {str(k): n_prem[k] for k in sorted(n_prem)},
            "premise_counts_pct": {str(k): round(100 * n_prem[k] / n, 2) for k in sorted(n_prem)},
            "classes": classes, "rewarded": rewarded,
            "rule": "SPEC: the class with a premise in the largest share of RL-pool theorems; if that share is under 20 %, "
                    "the top two classes (premises: top-level commas of `thm` before |-, classed by notes.premise_class)"}  # fmt: skip


def pool_main(argv: list[str]) -> dict[str, Any]:
    ap = argparse.ArgumentParser(prog="lh_rlvar.py pool")
    ap.add_argument("--pool", default=str(C.RL_POOL))
    ap.add_argument("--out", default=str(C.REWARDED))
    ap.add_argument("--table_out", default=str(C.POOL_TABLE))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    t = pool_table(Path(a.pool))
    print(
        f"RL pool {t['pool']}: {t['theorems']:,} theorems; premises per theorem {t['premise_counts']} (% {t['premise_counts_pct']})"
    )
    for c in t["classes"]:
        print(f"  {c['cls']:5s} {c['theorems']:6d} {c['share_pct']:8.2f} {c['premises']:8d}")
    print(f"largest class: {t['rewarded']}")
    if a.dry_run:
        return t
    C.write_once(Path(a.table_out), t, ("pool_sha256", "classes", "premise_counts", "rewarded"))
    doc = {"rewarded": t["rewarded"], "rule": t["rule"], "share_pct": {c["cls"]: c["share_pct"] for c in t["classes"] if c["cls"] in t["rewarded"]},
           "pool": t["pool"], "pool_sha256": t["pool_sha256"], "table": str(Path(a.table_out).name),
           "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}  # fmt: skip
    C.write_once(Path(a.out), doc, ("rewarded", "pool_sha256"))
    print(f"wrote {a.out} and {a.table_out}")
    return t


# ------------------------------------------------------------------------------------------------ entry points
def anchor_rows(fn: Path) -> int:
    with open(fn) as f:
        return sum(1 for x in f if x.strip())


def grpo_main(argv: list[str]) -> int:
    """W2's `rlvar.grpo_main` (unchanged; it checks `--unlikely_beta 0`, writes rlvar.json, runs tag_grpo) for our
    variants, after writing lh_rlvar.json and, for an rlhf line, installing the reward model's bonus."""
    k = argv.index("--")
    ap = argparse.ArgumentParser(prog="lh_rlvar.py grpo")
    ap.add_argument("--var", required=True, choices=list(C.LINE_KINDS))
    a = ap.parse_args(argv[:k])
    rest = argv[k + 1 :]
    j = rest.index("--")
    g = argparse.ArgumentParser(add_help=False)
    for f in (
        "--out",
        "--name",
        "--init",
        "--kl_anchor_ref",
        "--kl_anchor_data",
        "--max_steps",
        "--kl_anchor_coef",
    ):
        g.add_argument(f, required=True)
    g.add_argument("--device", default="cuda")
    ga, _ = g.parse_known_args(rest[j + 1 :])
    task = C.parse_task(ga.name)
    assert task["kind"] == a.var, (task, a.var)
    assert (
        int(ga.max_steps) == C.STEP_LIMIT
        and float(ga.kl_anchor_coef) == C.KL_ANCHOR
        and ga.kl_anchor_ref == ga.init
    ), ga
    anchor = Path(ga.kl_anchor_data)
    if not anchor.exists():
        raise SystemExit(f"no anchor file {anchor} (lh_pairs.py anchor --corpus {task['corpus']})")
    out = Path(ga.out)
    out.mkdir(parents=True, exist_ok=True)
    info: dict[str, Any] = {"variant": a.var, **rlvar.VARIANTS[a.var], "corpus": task["corpus"], "step_limit": C.STEP_LIMIT,
                            "kl_anchor_coef": C.KL_ANCHOR, "kl_anchor_ref": ga.init, "anchor": str(anchor),
                            "anchor_sha256": C.sha256(anchor), "anchor_rows": anchor_rows(anchor),
                            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}  # fmt: skip
    if task["rater"] is not None:
        rm = C.line_rm_path(
            task["corpus"], task["rater"]
        )  # every RL seed: the seed-0 fit (PREREG Amendment 10)
        if not rm.exists():
            raise SystemExit(f"no reward model {rm} (lh_launch.py _rm)")
        import tag_core as T

        T.install_thinking()
        from code.thinking import grpo  # type: ignore[import-not-found]

        dev = "cpu" if ga.device == "cpu" else "cuda"
        install_rm(grpo, rm, dev)
        info.update(
            reward_model=str(rm),
            reward_model_sha256=C.sha256(rm),
            rated_habit=C.RATERS[task["rater"]],
            reward="acceptance x (1 + 0.3 x sigmoid(RM(prompt, ND proof)))",
        )
    (out / "lh_rlvar.json").write_text(json.dumps(info, indent=1))
    print(f"lh_rlvar: {a.var} on {task['corpus']}: {json.dumps(info)}", flush=True)
    return rlvar.grpo_main(["--var", a.var, "--", *rest])


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "grpo":
        sys.exit(grpo_main(sys.argv[2:]))
    elif cmd == "pool":
        pool_main(sys.argv[2:])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
