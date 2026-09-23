#!/usr/bin/env python3
"""Pre-registration for one experiment, written BEFORE the run:
results/<NNN>_<slug>/params.json + pretrain.patch.

  python3 params.py new <NNN> <slug> [--seeds 0,1,2]  # after editing pretrain.py
  python3 params.py check <NNN>_<slug>    # validates against schema.Params; record.py runs it too

Mechanical (never hand-typed): base_commit, harness/pretrain sha, the patch, and `protocol` = the
harness constants plus the pool sizes and proof-length histograms from pools/<fork-commit>.json.
Hand-filled by the agent: provider/model, title/description/hypothesis/prediction, and `config`
(what pretrain.py builds: arch, tokenizer, optimizer, batch, data, precision, seed).

Shared helpers (harness_rules, fork_commit, sha, git) are imported by record.py and pools.py.
"""

import argparse
import ast
import contextlib
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from schema import Params

HERE = Path(__file__).resolve().parent
TODO = "TODO"


def sha(fn: Path) -> str:
    return hashlib.sha256(fn.read_bytes()).hexdigest()[:12]


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=HERE, capture_output=True, text=True, check=True
    ).stdout


def registration_files(name: str, *, allow_new: bool = False) -> list[str]:
    """An existing preregistration is immutable, even when publishing a new one."""
    paths = [f"results/{name}/{filename}" for filename in ("params.json", "pretrain.patch")]
    for relative in paths:
        path = HERE / relative
        if not path.is_file():
            raise SystemExit(f"{relative} missing: pre-register first")
        try:
            committed = git("show", f"HEAD:./{relative}")
        except subprocess.CalledProcessError:
            if allow_new:
                continue
            raise SystemExit(f"{relative} is not committed at HEAD") from None
        if path.read_text() != committed:
            raise SystemExit(f"{relative} differs from committed preregistration")
    return paths


def sync(candidate: str) -> None:
    """Catch up with the shared branch: the other agent's loop pushes to it too, sometimes with a
    new incumbent pretrain.py. The uncommitted candidate is set aside while catching up, so it is
    never line-merged with theirs, then written back whole. RESULTS.md is regenerated, never merged.
    """
    try:
        try:
            git("rev-parse", "--verify", "--quiet", "@{upstream}")
        except subprocess.CalledProcessError:
            return  # No upstream (tests, local-only branch): nothing to catch up with.
        git("fetch", "--quiet")
        if not git("rev-list", "HEAD..@{upstream}").strip():
            return
        aside = git("ls-files", "--", "pretrain.py", "RESULTS.md").split()
        git("restore", "--source=HEAD", "--staged", "--worktree", "--", *aside)
        if git("rev-list", "@{upstream}..HEAD").strip():
            git("rebase", "--autostash", "--quiet", "@{upstream}")  # an earlier push failed
        else:
            git("merge", "--ff-only", "--quiet", "@{upstream}")  # keeps staged drafts staged
    finally:
        (HERE / "pretrain.py").write_text(candidate)


def commit_and_push(paths: list[str], message: str, prepare=None) -> str:
    """Commit only these working-tree files on top of the shared branch; preserve unrelated staged
    work and run hooks. If the other agent pushes first, undo and redo on top of theirs.
    prepare() runs after every catch-up to (re)write the files and may return a new message."""
    candidate = (HERE / "pretrain.py").read_text()
    action = "pull"
    try:
        for _ in range(5):
            action = "pull"
            sync(candidate)
            message = (prepare and prepare()) or message
            action = "add"
            git("add", "--", *paths)
            action = "diff"
            committed = bool(git("diff", "--cached", "--name-only", "--", *paths).strip())
            if committed:
                action = "commit"
                git("commit", "--only", "-m", message, "--", *paths)
            action = "push"
            try:
                git("push")  # Also retries a previously successful commit whose push failed.
                return git("rev-parse", "HEAD").strip()
            except subprocess.CalledProcessError as exc:
                if "[rejected]" not in (exc.stderr or ""):  # non-fast-forward, not a hook
                    raise
                if committed:  # Lost the race: redo this commit on top of the other agent's.
                    git("reset", "--soft", "HEAD~1")
                git("restore", "--staged", "--", *paths)
        raise SystemExit("Git push lost the race to the other agent 5 times; not published")
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise SystemExit(f"Git {action} failed; experiment not published:\n{detail}") from None


def fork_commit() -> str:
    m = re.search(r"^FORK_COMMIT=([0-9a-f]+)", (HERE / "remote.sh").read_text(), re.M)
    assert m, "remote.sh has no FORK_COMMIT line"
    return m.group(1)


def harness_rules() -> dict[str, int | float]:
    """The UPPER_CASE numeric constants assigned at the top level of harness.py, evaluated one
    assignment at a time (importing harness.py needs torch and the fork)."""
    tree = ast.parse((HERE / "harness.py").read_text())
    ns: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            # a constant that needs torch or the fork raises here; it is not a rule we record
            with contextlib.suppress(Exception):
                code = compile(ast.Module(body=[node], type_ignores=[]), "harness.py", "exec")
                exec(code, {"os": os, "__file__": str(HERE / "harness.py")}, ns)
    return {k: v for k, v in ns.items() if k.isupper() and isinstance(v, int | float)}


def protocol(seed: int | list[int] = 0) -> dict[str, Any]:
    H, fc = harness_rules(), fork_commit()
    pools_fn = HERE / "pools" / f"{fc[:7]}.json"
    if not pools_fn.exists():
        raise SystemExit(
            f"{pools_fn} missing: remote.sh pins fork {fc[:7]}; "
            "run `python3 pools.py <fork-checkout>` once"
        )
    P = json.loads(pools_fn.read_text())
    return {
        "fork_commit": fc,
        "budget": {
            "total_secs": H["TOTAL_SECS"],
            "pretrain_secs": H["PRETRAIN_SECS"],
            "grace_secs": H["PRETRAIN_GRACE"],
            "param_cap": H["PARAM_CAP"],
        },
        "pretrain_data": {"max_lines": 6, **P["pretrain"]},
        "pretrain_heldout": P["pretrain_heldout"],
        "rl": {
            "rounds": H["RL_ROUNDS"],
            "n_targets": H["N_TARGETS"],
            "rollouts_per_target": H["K"],
            "temp": H["TEMP"],
            "max_new_tokens": H["MAX_NEW"],
            "ft_steps": H["FT_STEPS"],
            "ft_bs": H["FT_BS"],
            "ft_lr": H["FT_LR"],
            "retain_pretrain_records": H["RETAIN"],
            "max_proofs_per_thm": H["MAX_PER_THM"],
            "rl_weight": H["RL_WEIGHT"],
            "target_pool": P["rl_targets"]["pool"],
            "targets_sampled": P["rl_targets"]["sampled"] if seed == 0 else None,
        },
        "eval": {
            "metric": "dev-transfer theorems solved at L_true >= 7",
            "attempts_per_theorem": H["EVAL_K"],
            "temp": H["TEMP"],
            "pool": P["transfer_dev"],
        },
        "holdout": P["transfer_holdout"],
        "seed": seed,
    }


CONFIG_TEMPLATE: dict[str, Any] = {
    "arch": {
        "type": "gpt (fork model.py)",
        "n_layers": TODO,
        "d_model": TODO,
        "n_heads": TODO,
        "d_ff": TODO,
        "pos": "rope",
        "ctx": 1024,
        "norm": "layernorm",
        "act": "gelu",
        "dropout": 0.0,
        "tied_emb": False,
        "n_params": TODO,
    },
    "tokenizer": {"codec": TODO, "maxn": 64},
    "optimizer": {
        "name": TODO,
        "lr": TODO,
        "schedule": TODO,
        "warmup_steps": TODO,
        "min_lr_frac": TODO,
        "wd": TODO,
        "betas": TODO,
        "grad_clip": TODO,
    },
    "batch": {"size": TODO, "padding": TODO, "buckets": TODO},
    "data": {"filter": None, "augmentation": TODO, "order": TODO, "val_slice": TODO},
    "precision": TODO,
    "compile": TODO,
    "init": "fork default (normal 0.02)",
    "seed": 0,
}


def patch_lines(patch: str) -> int:
    return sum(
        1 for line in patch.splitlines() if line[:1] in "+-" and line[:3] not in ("+++", "---")
    )


def new(nnn: str, slug: str, seeds: list[int] | None = None) -> None:
    if seeds is not None and (not seeds or len(set(seeds)) != len(seeds)):
        raise SystemExit("seeds must be nonempty and unique")
    seed = seeds if seeds is not None else 0
    # Two agents share one branch; the suffix keeps their ids (and pod names) from colliding.
    agent = os.environ.get("ND_AGENT")
    if agent and not nnn.endswith(f"-{agent}"):
        nnn = f"{nnn}-{agent}"
    pretrain = HERE / "pretrain.py"
    incumbent = git("show", "HEAD:./pretrain.py")
    try:
        sync(pretrain.read_text())
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Git pull failed: {(exc.stderr or exc.stdout).strip()}") from None
    if git("show", "HEAD:./pretrain.py") != incumbent:
        raise SystemExit(
            "The other agent kept a new pretrain.py while you worked (see `git log -3`). Your "
            "candidate is untouched, but its diff against HEAD now also reverts their win. "
            "Re-apply only your idea on top of `git show HEAD:./pretrain.py`, then rerun this."
        )
    name = f"{nnn}_{slug}"
    d = HERE / "results" / name
    if d.exists():
        raise SystemExit(f"{d} exists")
    d.mkdir(parents=True)
    patch = git("diff", "HEAD", "--", "pretrain.py")
    (d / "pretrain.patch").write_text(patch)
    p: dict[str, Any] = {
        "id": nnn,
        "slug": slug,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "provider": {"claude": "anthropic", "codex": "openai"}.get(agent or "", TODO),
        "model": os.environ.get("ND_AGENT_MODEL", TODO),
        "base_commit": git("rev-parse", "--short", "HEAD").strip(),
        "harness_sha": sha(HERE / "harness.py"),
        "pretrain_sha": sha(HERE / "pretrain.py"),
        "patch_lines": patch_lines(patch),
        "title": TODO,
        "description": TODO,
        "tags": [TODO],
        "hypothesis": TODO,
        "p_keep": TODO,
        "expected_delta": TODO,
        "baseline_id": TODO,
        "baseline_metric": TODO,
        "config": {**CONFIG_TEMPLATE, "seed": seed},
        "protocol": protocol(seed),
    }
    (d / "params.json").write_text(json.dumps(p, indent=2) + "\n")
    print(
        f"{d}/params.json  (patch {p['patch_lines']} lines; "
        "fill every TODO; run_idea.py commits/pushes before running)"
    )


def check(name: str) -> Params:
    fn = HERE / "results" / name / "params.json"
    try:
        return Params.model_validate_json(fn.read_text())
    except ValidationError as e:
        bad = ", ".join(".".join(str(x) for x in err["loc"]) for err in e.errors())
        raise SystemExit(f"{fn} invalid (TODOs left or wrong types): {bad}") from None


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "new":
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("nnn")
        parser.add_argument("slug")
        parser.add_argument("--seeds", type=lambda value: [int(s) for s in value.split(",")])
        args = parser.parse_args(sys.argv[2:])
        new(args.nnn, args.slug, args.seeds)
    elif cmd == "check":
        check(sys.argv[2])
        print("ok")
    else:
        raise SystemExit(__doc__)
