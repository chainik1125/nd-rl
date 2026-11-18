#!/usr/bin/env python3
# imports long_rl_20261014's harness and the fork, neither resolvable here
# pyright: reportMissingImports=false
# 120 columns, as in long_rl_20261014's files
# ruff: noqa: E501
"""long_rl_20261014's harness with one inference rule removed from pretraining.

harness.py and pretrain.py are long_rl_20261014's files, unchanged. This wrapper replaces four things on the imported
harness module and then calls its main(), so the tested pretraining, RL and Lean-gate code runs as it did there:

  load_pools      drops every pretraining record whose proof uses ND_KNOCKOUT (IMPE | IMPI | BOTE | DN | none). With
                  ND_POOL=focus it also keeps only the RL targets whose reference proof uses that rule.
  Oracle._sample  after each oracle call, appends the call's stats and a rule census to rounds.jsonl and the new
                  Lean-accepted proofs to found_stream.jsonl. The harness writes its own outputs only when RL ends, so
                  without this a run that is cut off leaves no curve.
  TEMP            ND_TEMP, the sampling temperature (pretrain.GRPO_TEMP follows it; the harness asserts they agree).
  EVAL_K          ND_EVAL_K, samples per dev theorem in the final eval.

  ND_FORK=... ND_KNOCKOUT=IMPE python3 knockout.py --out /tmp/smoke --smoke     # CPU plumbing check with real Lean
"""

import collections
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness  # noqa: E402  (long_rl_20261014/harness.py, uploaded beside this file)
import pretrain  # noqa: E402  (after harness, which puts the fork on sys.path)

KO = os.environ.get("ND_KNOCKOUT", "none")
POOL = os.environ.get("ND_POOL", "full")
RULES = {"IMPE", "IMPI", "BOTE", "DN", "NEGE", "NEGI", "ORE"}  # the rules the census counts
assert (
    KO in RULES | {"none"} and POOL in ("full", "focus") and not (POOL == "focus" and KO == "none")
)
RULE_RE = re.compile(
    r": ([A-Z]+[12]?)\b"
)  # an ND line is `N7 | ( P > Q ) : IMPE N1 N3`; formulas hold no colon

if "ND_TEMP" in os.environ:
    harness.TEMP = pretrain.GRPO_TEMP = float(os.environ["ND_TEMP"])
harness.EVAL_K = int(os.environ.get("ND_EVAL_K", harness.EVAL_K))


def rules(proof):
    return set(RULE_RE.findall(proof or ""))


_load_pools = harness.load_pools


def load_pools(a):
    train, targets, dev, old = _load_pools(a)
    n_train, n_targets = len(train), len(targets)
    if KO != "none":
        train = [r for r in train if KO not in rules(r["proof"])]
    if POOL == "focus":
        targets = [t for t in targets if KO in rules(t.get("proof"))]
    ref = collections.Counter(x for t in targets for x in rules(t.get("proof")) & RULES)
    harness.log(
        f"knockout {KO}: pretraining {n_train:,} -> {len(train):,} records; pool {POOL}: RL targets "
        f"{n_targets:,} -> {len(targets):,}, reference proofs using each rule {dict(ref)}; "
        f"TEMP {harness.TEMP}, EVAL_K {harness.EVAL_K}"
    )
    shutil.copy2(__file__, os.path.join(a.out, "source"))
    return train, targets, dev, old


_sample = harness.Oracle._sample


def _sample_logged(self, model, targets, k):
    recs, per, rows = _sample(self, model, targets, k)
    call = self.calls[-1]
    accepted, accepted_using = (
        0,
        collections.Counter(),
    )  # this call's Lean-accepted attempts, and those using each rule
    for samples in per:
        for _, nd, ok in samples:
            if ok:
                accepted += 1
                accepted_using.update(rules(nd) & RULES)
    # cumulative over the run: distinct accepted proofs using each rule, targets with such a proof, and solved targets
    # whose reference proof uses the rule
    proofs_using, targets_using, solved_ref = (
        collections.Counter(),
        collections.Counter(),
        collections.Counter(),
    )
    for name, proofs in self.found.items():
        if proofs:
            used = [rules(p) & RULES for p in proofs]
            for u in used:
                proofs_using.update(u)
            targets_using.update(set().union(*used))
            solved_ref.update(rules(self.targets[name].get("proof")) & RULES)
    call["census"] = {
        "accepted": accepted,
        "accepted_using": dict(accepted_using),
        "proofs_using": dict(proofs_using),
        "targets_using": dict(targets_using),
        "solved_ref_uses": dict(solved_ref),
    }
    with open(f"{self.a.out}/rounds.jsonl", "a") as f:
        f.write(json.dumps(call) + "\n")
    with open(f"{self.a.out}/found_stream.jsonl", "a") as f:
        for t in recs:
            for proof, x in self.found[t["name"]].items():
                if x["round"] == call["round"]:
                    f.write(
                        json.dumps({"name": t["name"], "body": t["body"], "proof": proof, **x})
                        + "\n"
                    )
    if KO != "none":
        harness.log(
            f"census {call['round']}: {accepted} accepted attempts, {accepted_using[KO]} use {KO}; so far "
            f"{proofs_using[KO]} distinct proofs with {KO} on {targets_using[KO]} targets, and "
            f"{solved_ref[KO]} solved targets whose reference proof uses {KO}"
        )
    return recs, per, rows


harness.load_pools = load_pools
harness.Oracle._sample = _sample_logged

if __name__ == "__main__":
    harness.main()
