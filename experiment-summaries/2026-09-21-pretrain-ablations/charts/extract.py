"""Pull one row per autoresearch experiment into charts/runs.csv.

Reads the loop's results/*/results.json (dev score, final pretrain val loss) and the offline
holdout scores in artifacts/autoresearch/holdout/*.holdout.json (written by holdout.py score).
Re-run after more holdout checkpoints are scored; the chart scripts only read the CSV.

    cd experiment-summaries/2026-09-21-pretrain-ablations && python3 charts/extract.py

Columns:
  run      results dir name, e.g. 153-claude_lean-width384
  num      the 3-digit run number (two agents can share one)
  era      token (012-097, char tokenizer, nd_verify reward) or lean (100+, lean_seq + Lean reward)
  status   keep / discard
  loss     final pretrain val loss. Lean era: mean over the seeds that finished. Token era: seed 0
           only (results.json kept just seed 0's curve then). Not comparable across eras.
  dev      dev theorems solved at length >= 7, 3-seed mean (the loop's metric), of 1,108
  holdout  same on the transfer holdout, of 1,177; blank until scored
  holdout_seeds  how many seeds the holdout mean covers
  len7..len13  dev theorems the model proved, binned by min(the model's shortest proof after
           pruning uncited lines (the harness's pruned_lens), the reference proof's n_lines), mean
           over the seeds that finished. Both are proofs of the theorem, so the min is the tighter
           upper bound on its shortest proof. The n_lines label alone is loose: the model beats it
           on ~27% of solved theorems. Read from each seed's eval_dev.jsonl in the researcher
           checkouts.
  lines    total lines in the run's pretrain.py (the only file agents edit; model, optimizer and
           schedule all live there, on top of frozen fork modules it imports). Rebuilt as
           git show <base_commit>:pretrain.py + the run's pretrain.patch, checked against pretrain_sha.
  sloc     the same file minus blank lines, comments and docstrings.
"""
import ast
import csv
import glob
import hashlib
import io
import json
import os
import statistics
import subprocess
import tempfile
import tokenize

ROOT = os.path.expanduser("~/repos/nd-rl")
RESULTS = f"{ROOT}/code/experiments/current/autoresearch/results"
HOLDOUT = f"{ROOT}/artifacts/autoresearch/holdout"
# Each researcher checkout keeps the per-seed artifacts of the runs it ran.
SEED_DIRS = [os.path.expanduser(f"~/repos/{c}/artifacts/autoresearch")
             for c in ("nd-rl", "nd-rl-3", "nd-rl-4", "nd-rl-5")]
LENGTHS = range(7, 14)


def seed_proofs(run, seed):
    """One seed's dev eval, one record per theorem, from wherever that seed's artifacts live."""
    path = next(f"{d}/{run}_s{seed}/eval_dev.jsonl" for d in SEED_DIRS
                if os.path.exists(f"{d}/{run}_s{seed}/eval_dev.jsonl"))
    return [json.loads(line) for line in open(path)]

PRETRAIN = "code/experiments/current/autoresearch/pretrain.py"


def pretrain_src(run):
    """The pretrain.py this run trained with: its base commit's file plus its registered patch."""
    params = json.load(open(f"{RESULTS}/{run}/params.json"))
    base = subprocess.run(["git", "-C", ROOT, "show", f"{params['base_commit']}:{PRETRAIN}"],
                          capture_output=True, check=True).stdout
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.dirname(f"{tmp}/{PRETRAIN}"))
        open(f"{tmp}/{PRETRAIN}", "wb").write(base)
        subprocess.run(["git", "apply", "--allow-empty", f"{RESULTS}/{run}/pretrain.patch"],
                       cwd=tmp, check=True)
        src = open(f"{tmp}/{PRETRAIN}", "rb").read()
    assert hashlib.sha256(src).hexdigest()[:12] == params["pretrain_sha"], run
    return src.decode()


def sloc(src):
    """Lines holding code: not blank, not only a comment, not part of a docstring."""
    docs = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) \
                and ast.get_docstring(node, clean=False) is not None:
            docs.update(range(node.body[0].lineno, node.body[0].end_lineno + 1))
    skip = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
            tokenize.ENDMARKER}
    code = {n for t in tokenize.generate_tokens(io.StringIO(src).readline) if t.type not in skip
            for n in range(t.start[0], t.end[0] + 1)}
    return len(code - docs)


FIRST, LEAN = 12, 100  # 000-011 ran a 600 s, 1-seed protocol; 098-099 only timed out

rows = []
for path in sorted(glob.glob(f"{RESULTS}/*/results.json")):
    run = os.path.basename(os.path.dirname(path))
    num = int(run[:3])
    r = json.load(open(path))
    if num < FIRST or r["status"] not in ("keep", "discard") or (98 <= num < LEAN):
        continue
    if num < LEAN:
        loss = r["pretrain"]["curve"][-1]["val_loss"]
    else:
        ok = [s for s, st in r["seed_status"].items() if st == "ok"]
        loss = statistics.mean(r["seed_pretrain"][s]["val_loss"] for s in ok)
    seeds = [seed_proofs(run, s) for s, st in r["seed_status"].items() if st == "ok"]
    # the per-seed evals must reproduce the loop's 3-seed metric (which bins by the n_lines label)
    assert round(statistics.mean(sum(t["solved"] and t["n_lines"] >= 7 for t in e)
                                 for e in seeds)) == r["metric"], run
    shortest = [[min(*t["pruned_lens"], t["n_lines"]) for t in e if t["solved"]] for e in seeds]
    src = pretrain_src(run)
    hold = [json.load(open(p))["solved_ge7"] for p in glob.glob(f"{HOLDOUT}/{run}_s[0-9].holdout.json")]
    rows.append({
        "run": run, "num": num, "era": "token" if num < LEAN else "lean", "status": r["status"],
        "loss": round(loss, 5), "dev": r["metric"],
        "holdout": round(statistics.mean(hold), 1) if hold else "", "holdout_seeds": len(hold),
        **{f"len{n}": round(statistics.mean(lens.count(n) for lens in shortest), 1)
           for n in LENGTHS},
        "lines": len(src.splitlines()), "sloc": sloc(src),
    })

with open("charts/runs.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"wrote charts/runs.csv ({len(rows)} runs, {sum(1 for r in rows if r['holdout'] != '')} with holdout)")
