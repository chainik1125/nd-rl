"""Reviewer recount for exp-00 (plan section 6). Independent of the executor's eval package.

Run from code/:  uv run python ../experiments/ws3/exp-00-local-check/review/recount.py
Writes recount.json and samples.txt next to this file. Uses only strict_verify.safe_verify
(checker of record), the vendored nd_data.canonical_theorem (second-opinion key), and the raw
dumps under run/. Nothing from code/ws3/eval or code/ws3/data.py is imported.
"""

import gzip
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
RUN = os.path.join(EXP, "run")
REPO = os.path.abspath(os.path.join(EXP, "..", "..", ".."))
CODE = os.path.join(REPO, "code")
sys.path.insert(0, os.path.join(CODE, "ws3"))
import ndpath  # noqa: E402,F401  (puts fork_pin and the vendor dir on sys.path)

from nd_data import canonical_theorem  # noqa: E402  (vendored, Dmitry)
from strict_verify import safe_verify  # noqa: E402  (checker of record)

FORK_DATA = ndpath.FORK_DATA
TB = ndpath.TEXTBOOK72
ATOMS = ("P", "Q", "R", "S")
SAMPLE_SEED = 20260923  # reviewer's own seed for the 20 random samples


# ----------------------------------------------------------------------------- helpers
def read_jsonl(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return [json.loads(l) for l in f if l.strip()]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wilson(k, n, z=1.959963984540054):
    """Wilson score interval, written independently of the executor and of nd_metrics."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def split_prompt(prompt):
    """THM <premises> SEQ <conclusion> PRF -> (list of premise strings, conclusion string)."""
    t = prompt.split()
    assert t[0] == "THM" and t[-1] == "PRF", prompt
    i = t.index("SEQ")
    prem_toks, con = t[1:i], " ".join(t[i + 1 : -1])
    prems, cur = [], []
    for tok in prem_toks:
        if tok == ",":
            prems.append(" ".join(cur))
            cur = []
        else:
            cur.append(tok)
    if cur:
        prems.append(" ".join(cur))
    return prems, con


def thm_of_prompt(prompt):
    prems, con = split_prompt(prompt)
    return (" , ".join(prems) + " |- " + con) if prems else ("|- " + con)


def canon_key(thm):
    """The fork's gen.canon_key as described: relabel atoms P,Q,R,S by first appearance in
    the theorem string `premises |- conclusion`, scanning tokens left to right."""
    m, out = {}, []
    for tok in thm.split():
        if tok in ATOMS:
            if tok not in m:
                m[tok] = ATOMS[len(m)]
            out.append(m[tok])
        else:
            out.append(tok)
    return " ".join(out)


def is_trivial(prompt):
    prems, con = split_prompt(prompt)
    return con in prems or "F" in prems


# ----------------------------------------------------------------------------- 1. re-verify
DUMPS = {
    "baseline/greedy_heldout": "baseline/dumps/greedy_heldout.jsonl.gz",
    "baseline/textbook72": "baseline/dumps/textbook72.jsonl.gz",
    "baseline/long_pool": "baseline/dumps/long_pool.jsonl.gz",
    "evaluate/greedy_heldout": "evaluate/dumps/greedy_heldout.jsonl.gz",
    "evaluate/textbook72": "evaluate/dumps/textbook72.jsonl.gz",
    "evaluate/long_pool": "evaluate/dumps/long_pool.jsonl.gz",
    "eval_periodic": "eval_periodic.jsonl.gz",
}


def reverify_dump(rows, ok_field, nlines_field):
    """Re-run every row through strict safe_verify; return per-row verdicts and disagreements."""
    my = []
    dis = []
    nl_dis = 0
    for i, r in enumerate(rows):
        ok, reason, n = safe_verify(r["prompt"], r["completion"])
        my.append((ok, reason, n))
        theirs = r[ok_field]
        theirs_ok = bool(theirs) if ok_field == "ok" else (float(theirs) >= 0.5)
        if ok != theirs_ok:
            dis.append({"row": i, "mine": ok, "theirs": theirs, "reason": reason, "prompt": r["prompt"]})
        if ok and nlines_field in r and int(r[nlines_field]) != n:
            nl_dis += 1
    return my, dis, nl_dis


def pass_at_k_by_length(rows, my):
    """A record (prompt name) is solved if any of its k samples verifies. Group by length."""
    solved_by_name = defaultdict(bool)
    len_by_name, k_by_name = {}, Counter()
    temps = Counter()
    for r, (ok, _, _) in zip(rows, my):
        nm = r["name"]
        solved_by_name[nm] |= ok
        len_by_name[nm] = r["length"]
        k_by_name[nm] += 1
        temps[(r["k"], r["temperature"])] += 1
    by_len = defaultdict(lambda: [0, 0])
    for nm, s in solved_by_name.items():
        L = len_by_name[nm]
        by_len[L][1] += 1
        by_len[L][0] += int(s)
    out = {}
    for L in sorted(by_len):
        s, n = by_len[L]
        lo, hi = wilson(s, n)
        out[str(L)] = {"solved": s, "n": n, "rate": s / n, "wilson95": [lo, hi]}
    tot_s = sum(v[0] for v in by_len.values())
    tot_n = sum(v[1] for v in by_len.values())
    lo, hi = wilson(tot_s, tot_n)
    overall = {"solved": tot_s, "n": tot_n, "rate": tot_s / tot_n if tot_n else 0, "wilson95": [lo, hi]}
    return out, overall, solved_by_name, dict(k_by_name), dict(temps)


def main():
    out = {"sample_seed": SAMPLE_SEED}
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    # ---------------------------------------------------------------- textbook manifest hashes
    man = json.load(open(os.path.join(TB, "MANIFEST.json")))
    tb_hash = {}
    for fn, meta in man["files"].items():
        p = os.path.join(TB, fn)
        h = sha256(p)
        n = len(read_jsonl(p))
        tb_hash[fn] = {"sha256": h, "manifest": meta["sha256"], "match": h == meta["sha256"], "records": n,
                       "manifest_records": meta["records"]}
    out["textbook72_manifest"] = tb_hash
    say("== textbook72 MANIFEST: " + json.dumps(tb_hash))

    tb_dev = read_jsonl(os.path.join(TB, "textbook_dev.jsonl"))
    tb_train = read_jsonl(os.path.join(TB, "textbook_train.jsonl"))
    tb_all = [dict(r, slice="dev58") for r in tb_dev] + [dict(r, slice="train14") for r in tb_train]
    tb_by_name = {r["name"]: r for r in tb_all}
    n_ge7 = sum(1 for r in tb_dev if int(r["reference_lines"]) >= 7)
    say(f"textbook: dev {len(tb_dev)}, train {len(tb_train)}, dev with reference_lines>=7: {n_ge7}")
    out["textbook_ref_ge7_n"] = n_ge7

    # ---------------------------------------------------------------- 1. re-verify every dump
    verify = {}
    myv = {}
    rows_all = {}
    for label, rel in DUMPS.items():
        rows = read_jsonl(os.path.join(RUN, rel))
        my, dis, nl_dis = reverify_dump(rows, "ok", "n_lines_written")
        rows_all[label], myv[label] = rows, my
        n_ok_mine = sum(1 for m in my if m[0])
        n_ok_theirs = sum(1 for r in rows if r["ok"])
        verify[label] = {"rows": len(rows), "accepted_mine": n_ok_mine, "accepted_theirs": n_ok_theirs,
                         "disagreements": len(dis), "n_lines_disagreements_on_accepted": nl_dis,
                         "disagreement_examples": dis[:5]}
        say(f"[{label}] rows={len(rows)} accepted mine={n_ok_mine} theirs={n_ok_theirs} "
            f"disagree={len(dis)} nlines_disagree={nl_dis}")
    comp = read_jsonl(os.path.join(RUN, "completions.jsonl.gz"))
    my_c, dis_c, nl_c = reverify_dump(comp, "reward", "n_lines")
    rows_all["completions"], myv["completions"] = comp, my_c
    verify["completions"] = {"rows": len(comp), "accepted_mine": sum(1 for m in my_c if m[0]),
                             "accepted_theirs": sum(1 for r in comp if float(r["reward"]) >= 0.5),
                             "reward_values": dict(Counter(str(r["reward"]) for r in comp)),
                             "disagreements": len(dis_c), "n_lines_disagreements_on_accepted": nl_c,
                             "disagreement_examples": dis_c[:5]}
    say(f"[completions] rows={len(comp)} accepted mine={verify['completions']['accepted_mine']} "
        f"theirs={verify['completions']['accepted_theirs']} disagree={len(dis_c)} nlines_disagree={nl_c}")
    out["reverify"] = verify

    # ---------------------------------------------------------------- pass@k by length (own Wilson)
    passk = {}
    solved_names = {}
    for label in DUMPS:
        by_len, overall, solved_by_name, k_by_name, temps = pass_at_k_by_length(rows_all[label], myv[label])
        ks = Counter(k_by_name.values())
        passk[label] = {"by_length": by_len, "overall": overall, "samples_per_record": dict(ks),
                        "k_temperature_fields": {str(k): v for k, v in temps.items()},
                        "n_records": len(k_by_name)}
        solved_names[label] = solved_by_name
        say(f"[{label}] pass@k by length: " + ", ".join(f"{L}: {v['solved']}/{v['n']}" for L, v in by_len.items())
            + f" | overall {overall['solved']}/{overall['n']} | samples/record {dict(ks)} | (k,T) {temps}")
    out["pass_at_k"] = passk

    # ---------------------------------------------------------------- 2. textbook slices, equal attempts
    tb = {}
    for label in ("baseline/textbook72", "evaluate/textbook72"):
        rows = rows_all[label]
        sb = solved_names[label]
        names = set(r["name"] for r in rows)
        missing = set(tb_by_name) - names
        extra = names - set(tb_by_name)
        # slice from the textbook files, not from the dump's own field
        res = {}
        for slc, members in (("dev58", [r["name"] for r in tb_dev]), ("train14", [r["name"] for r in tb_train]),
                             ("all", list(tb_by_name))):
            s = sum(1 for nm in members if sb.get(nm, False))
            res[slc] = {"solved": s, "n": len(members), "wilson95": list(wilson(s, len(members)))}
        ge7 = [r["name"] for r in tb_dev if int(r["reference_lines"]) >= 7]
        s7 = sum(1 for nm in ge7 if sb.get(nm, False))
        res["dev_ref_ge7"] = {"solved": s7, "n": len(ge7), "wilson95": list(wilson(s7, len(ge7)))}
        # dump's own slice / length fields vs the files
        slice_mismatch = sum(1 for r in rows if r["slice"] != tb_by_name.get(r["name"], {}).get("slice"))
        len_mismatch = 0
        for r in rows:
            ref = tb_by_name.get(r["name"], {}).get("reference_lines", -1)
            if int(r["length"]) != int(ref):
                len_mismatch += 1
        prompt_mismatch = sum(1 for r in rows if r["prompt"] != tb_by_name.get(r["name"], {}).get("prompt"))
        res["dump_checks"] = {"missing_problems": sorted(missing), "extra_names": sorted(extra),
                              "slice_field_mismatch": slice_mismatch, "length_field_mismatch": len_mismatch,
                              "prompt_mismatch": prompt_mismatch,
                              "solved_names": sorted(nm for nm, s in sb.items() if s),
                              "solved_ref_lines": {nm: tb_by_name[nm].get("reference_lines") for nm, s in sb.items() if s}}
        tb[label] = res
        say(f"[{label}] textbook solves: " + json.dumps({k: v for k, v in res.items() if k != "dump_checks"}))
        say(f"   dump checks: {json.dumps(res['dump_checks'])}")
    # solved-set overlap between control and RL
    b_s = {nm for nm, s in solved_names["baseline/textbook72"].items() if s}
    e_s = {nm for nm, s in solved_names["evaluate/textbook72"].items() if s}
    tb["solved_overlap"] = {"both": sorted(b_s & e_s), "baseline_only": sorted(b_s - e_s), "rl_only": sorted(e_s - b_s)}
    say(f"textbook solved overlap: {tb['solved_overlap']}")
    out["textbook"] = tb

    # long pool: check the dump prompts against rl_targets (name, prompt, L_true)
    rl_targets = read_jsonl(os.path.join(FORK_DATA, "rl_targets.jsonl"))
    rl_by_name = {r["name"]: r for r in rl_targets}
    lp = {}
    for label in ("baseline/long_pool", "evaluate/long_pool"):
        rows = rows_all[label]
        names = {r["name"] for r in rows}
        bad_prompt = sum(1 for r in rows if rl_by_name.get(r["name"], {}).get("prompt") != r["prompt"])
        bad_len = sum(1 for r in rows if rl_by_name.get(r["name"], {}).get("L_true") != r["length"])
        lp[label] = {"n_records": len(names), "not_in_rl_targets": sum(1 for nm in names if nm not in rl_by_name),
                     "prompt_mismatch": bad_prompt, "length_vs_L_true_mismatch": bad_len}
        say(f"[{label}] long pool checks: {lp[label]}")
    b_names = {r["name"] for r in rows_all["baseline/long_pool"]}
    e_names = {r["name"] for r in rows_all["evaluate/long_pool"]}
    lp["same_prompt_set_baseline_vs_evaluate"] = b_names == e_names
    b_names = {r["name"] for r in rows_all["baseline/greedy_heldout"]}
    e_names = {r["name"] for r in rows_all["evaluate/greedy_heldout"]}
    lp["same_greedy_set_baseline_vs_evaluate"] = b_names == e_names
    out["long_pool_checks"] = lp
    say(f"same prompt sets baseline vs evaluate: long_pool={lp['same_prompt_set_baseline_vs_evaluate']} "
        f"greedy={lp['same_greedy_set_baseline_vs_evaluate']}")

    # ---------------------------------------------------------------- 3. leakage
    prompts = read_jsonl(os.path.join(RUN, "prompts.jsonl"))
    say(f"prompts.jsonl: {len(prompts)} rows; sources {Counter(r['source'] for r in prompts)}; "
        f"lengths {Counter(r['length'] for r in prompts)}")
    # validate my canon_key against the fork's own `key` field on every record that carries one
    key_checks = {}
    pools = {}
    for name in ("heldout.jsonl", "transfer.jsonl", "rl_targets.jsonl", "validation_36.jsonl",
                 "test_short_prompts.jsonl", "test_long_prompts.jsonl"):
        pools[name] = read_jsonl(os.path.join(FORK_DATA, name))
    train_path = os.path.join(FORK_DATA, "train.jsonl.gz")
    pools["train.jsonl.gz"] = read_jsonl(train_path)
    for name, recs in pools.items():
        with_key = [r for r in recs if "key" in r]
        mism = sum(1 for r in with_key if canon_key(r["thm"]) != r["key"])
        mism_prompt = sum(1 for r in with_key if canon_key(thm_of_prompt(r["prompt"])) != r["key"])
        key_checks[name] = {"records": len(recs), "with_key": len(with_key), "my_canon_key_mismatch_vs_thm": mism,
                            "my_canon_key_mismatch_vs_prompt": mism_prompt}
    out["canon_key_validation"] = key_checks
    say("canon_key validation vs fork `key` field: " + json.dumps(key_checks))
    # prompt-set keys (recompute, do not trust the `key` field)
    p_key = {canon_key(thm_of_prompt(r["prompt"])) for r in prompts}
    p_key_field_mismatch = sum(1 for r in prompts if canon_key(thm_of_prompt(r["prompt"])) != r["key"])
    p_can = {canonical_theorem(r["prompt"]) for r in prompts}
    tb_prompts = [r["prompt"] for r in tb_all]
    sets = {
        "heldout": [r["prompt"] for r in pools["heldout.jsonl"]],
        "transfer": [r["prompt"] for r in pools["transfer.jsonl"]],
        "textbook72": tb_prompts,
        "validation_36": [r["prompt"] for r in pools["validation_36.jsonl"]],
        "rl_targets": [r["prompt"] for r in pools["rl_targets.jsonl"]],
        "test_short": [r["prompt"] for r in pools["test_short_prompts.jsonl"]],
        "test_long": [r["prompt"] for r in pools["test_long_prompts.jsonl"]],
    }
    leak = {"prompt_key_field_mismatch": p_key_field_mismatch, "n_unique_prompt_keys": len(p_key),
            "n_unique_prompts": len({r["prompt"] for r in prompts})}
    for nm, ps in sets.items():
        ck = {canon_key(thm_of_prompt(p)) for p in ps}
        ct = {canonical_theorem(p) for p in ps}
        leak[nm] = {"canon_key_intersection": len(p_key & ck), "canonical_theorem_intersection": len(p_can & ct),
                    "pool_size": len(ps)}
    train_keys = {r["key"] for r in pools["train.jsonl.gz"]}
    train_prompts = {r["prompt"] for r in pools["train.jsonl.gz"]}
    train_can = {canonical_theorem(r["prompt"]) for r in pools["train.jsonl.gz"]}
    leak["all_source_train"] = all(r["source"] == "train" for r in prompts)
    leak["key_in_train_jsonl"] = sum(1 for r in prompts if r["key"] in train_keys)
    leak["my_key_in_train_jsonl"] = sum(1 for r in prompts if canon_key(thm_of_prompt(r["prompt"])) in train_keys)
    leak["prompt_text_in_train_jsonl"] = sum(1 for r in prompts if r["prompt"] in train_prompts)
    leak["canonical_theorem_in_train"] = sum(1 for r in prompts if canonical_theorem(r["prompt"]) in train_can)
    # also: are the textbook72 / validation_36 classes inside train.jsonl at all (informational)
    leak["train_vs_textbook72_canon_key"] = len(train_keys & {canon_key(thm_of_prompt(p)) for p in tb_prompts})
    leak["train_vs_validation36_canon_key"] = len(train_keys & {canon_key(thm_of_prompt(p)) for p in sets["validation_36"]})
    out["leakage"] = leak
    say("leakage: " + json.dumps(leak))

    # ---------------------------------------------------------------- 4. reward sanity
    metrics = read_jsonl(os.path.join(RUN, "metrics.jsonl"))
    roll = [m for m in metrics if "reward" in m]
    ev = [m for m in metrics if "eval/pass_rate" in m]
    rs = [m["reward_std"] for m in roll]
    frz = [m["frac_reward_zero_std"] for m in roll]
    rew = [m["reward"] for m in roll]
    kl = [m["kl"] for m in roll]
    ent = [m["entropy"] for m in roll]
    sanity = {
        "rollout_lines": len(roll), "eval_lines": len(ev), "other_lines": len(metrics) - len(roll) - len(ev),
        "steps": [m["step"] for m in roll][:3] + ["..."] + [m["step"] for m in roll][-3:],
        "reward_std_nonzero_steps": sum(1 for x in rs if x > 0), "reward_std_min": min(rs), "reward_std_mean": sum(rs) / len(rs),
        "frac_reward_zero_std_mean": sum(frz) / len(frz), "frac_reward_zero_std_min": min(frz), "frac_reward_zero_std_max": max(frz),
        "reward_first5_mean": sum(rew[:5]) / 5, "reward_last5_mean": sum(rew[-5:]) / 5,
        "reward_first10_mean": sum(rew[:10]) / 10, "reward_last10_mean": sum(rew[-10:]) / 10,
        "reward_min": min(rew), "reward_max": max(rew), "reward_all": rew,
        "kl_max": max(kl), "kl_last": kl[-1], "entropy_first": ent[0], "entropy_last": ent[-1], "entropy_min": min(ent), "entropy_max": max(ent),
        "grad_norm_max": max(m["grad_norm"] for m in roll), "lr_values": sorted({m["learning_rate"] for m in roll}),
        "clipped_ratio_max": max(m["completions/clipped_ratio"] for m in roll),
        "mean_length_range": [min(m["completions/mean_length"] for m in roll), max(m["completions/mean_length"] for m in roll)],
        "max_length_max": max(m["completions/max_length"] for m in roll),
        "unparseable_frac_mean": sum(m["reward/unparseable_frac"] for m in roll) / len(roll),
        "invalid_frac_mean": sum(m["reward/invalid_frac"] for m in roll) / len(roll),
        "completions_per_sec_mean": sum(m["rollout/completions_per_sec"] for m in roll) / len(roll),
        "every_rollout_line_has_utc_and_step": all("utc" in m and "step" in m for m in roll),
        "every_line_has_utc_and_step": all("utc" in m and "step" in m for m in metrics),
        "first_utc": min(m["utc"] for m in metrics), "last_utc": max(m["utc"] for m in metrics),
    }
    # mean reward from the completions dump per step, independent of TRL's logging
    per_step = defaultdict(list)
    for r, (ok, _, _) in zip(comp, my_c):
        per_step[r["step"]].append(1.0 if ok else 0.0)
    steps_sorted = sorted(per_step)
    dump_means = [sum(per_step[s]) / len(per_step[s]) for s in steps_sorted]
    sanity["completions_dump_steps"] = [steps_sorted[0], steps_sorted[-1], len(steps_sorted)]
    sanity["completions_per_step_counts"] = dict(Counter(len(v) for v in per_step.values()))
    sanity["dump_mean_reward_first5"] = sum(dump_means[:5]) / 5
    sanity["dump_mean_reward_last5"] = sum(dump_means[-5:]) / 5
    sanity["dump_mean_reward_overall"] = sum(sum(v) for v in per_step.values()) / len(comp)
    # within-group zero-variance fraction recomputed from the dump: groups of 8 consecutive rows
    # with the same prompt (TRL repeats each prompt num_generations times)
    zero_groups, n_groups, bad_groups = 0, 0, 0
    for s in steps_sorted:
        rows_s = [(r, m) for r, m in zip(comp, my_c) if r["step"] == s]
        for g in range(0, len(rows_s), 8):
            grp = rows_s[g:g + 8]
            if len({r["prompt"] for r, _ in grp}) != 1:
                bad_groups += 1
                continue
            n_groups += 1
            vals = {m[0] for _, m in grp}
            zero_groups += int(len(vals) == 1)
    sanity["dump_groups"] = n_groups
    sanity["dump_groups_not_same_prompt"] = bad_groups
    sanity["dump_frac_groups_zero_variance"] = zero_groups / n_groups if n_groups else None
    # prompt-length composition of the completions
    plen = {r["prompt"]: r["length"] for r in prompts}
    sanity["completion_prompt_lengths"] = dict(Counter(plen.get(r["prompt"], "?") for r in comp))
    sanity["completion_prompts_not_in_prompts_jsonl"] = sum(1 for r in comp if r["prompt"] not in plen)
    sanity["distinct_prompts_in_completions"] = len({r["prompt"] for r in comp})
    # accepted-by-length in completions
    acc_by_len = defaultdict(lambda: [0, 0])
    for r, (ok, _, _) in zip(comp, my_c):
        L = plen.get(r["prompt"], "?")
        acc_by_len[L][1] += 1
        acc_by_len[L][0] += int(ok)
    sanity["completions_accept_rate_by_prompt_length"] = {str(L): f"{a}/{n}" for L, (a, n) in sorted(acc_by_len.items(), key=lambda x: str(x[0]))}
    out["reward_sanity"] = sanity
    say("reward sanity: " + json.dumps({k: v for k, v in sanity.items() if k != "reward_all"}))

    # trivial fraction among accepted, per dump
    triv = {}
    for label, rows in rows_all.items():
        my = myv[label]
        acc = [r for r, m in zip(rows, my) if m[0]]
        t = sum(1 for r in acc if is_trivial(r["prompt"]))
        # also among distinct solved records
        names_acc = {}
        for r, m in zip(rows, my):
            if m[0]:
                names_acc[r.get("name", r["prompt"])] = r["prompt"]
        t_rec = sum(1 for p in names_acc.values() if is_trivial(p))
        all_recs = {r.get("name", r["prompt"]): r["prompt"] for r in rows}
        t_all = sum(1 for p in all_recs.values() if is_trivial(p))
        triv[label] = {"accepted_rows": len(acc), "trivial_accepted_rows": t,
                       "frac": (t / len(acc)) if acc else None,
                       "solved_records": len(names_acc), "trivial_solved_records": t_rec,
                       "records": len(all_recs), "trivial_records": t_all}
    out["trivial"] = triv
    say("trivial among accepted: " + json.dumps(triv))

    # ---------------------------------------------------------------- periodic eval recount
    pe_rows = rows_all["eval_periodic"]
    pe = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r, (ok, _, _) in zip(pe_rows, myv["eval_periodic"]):
        step = int(r["label"].split("@")[1])
        pe[step][r["length"]][1] += 1
        pe[step][r["length"]][0] += int(ok)
    pe_out = {str(s): {str(L): f"{a}/{n}" for L, (a, n) in sorted(pe[s].items())} for s in sorted(pe)}
    pe_out_overall = {str(s): f"{sum(v[0] for v in pe[s].values())}/{sum(v[1] for v in pe[s].values())}" for s in sorted(pe)}
    # compare with metrics.jsonl eval lines
    pe_cmp = {}
    for m in ev:
        s = m["step"]
        mine = pe.get(s, {})
        diffs = []
        for L, (a, n) in mine.items():
            if m.get(f"eval/solved_len{L}") != a or m.get(f"eval/n_len{L}") != n:
                diffs.append((L, a, n, m.get(f"eval/solved_len{L}"), m.get(f"eval/n_len{L}")))
        pe_cmp[str(s)] = {"metrics_pass_rate": m["eval/pass_rate"], "mine": pe_out_overall.get(str(s)), "diffs": diffs}
    out["periodic_eval"] = {"by_step_by_length": pe_out, "overall": pe_out_overall, "vs_metrics": pe_cmp}
    say("periodic eval (mine, by step): " + json.dumps(pe_out_overall) + " vs metrics: " + json.dumps(pe_cmp))

    # ---------------------------------------------------------------- 5. samples and long accepted proofs
    S = []
    rng = random.Random(SAMPLE_SEED)
    idx = sorted(rng.sample(range(len(comp)), 20))
    S.append(f"=== 20 random rows of completions.jsonl.gz, seed {SAMPLE_SEED}, indices {idx}")
    for i in idx:
        r, m = comp[i], my_c[i]
        S.append(f"--- row {i} step {r['step']} reward(theirs)={r['reward']} mine={m[0]} reason(mine)={m[1]!r} "
                 f"n_lines(mine)={m[2]} n_tokens={r['n_tokens']} trivial={is_trivial(r['prompt'])}")
        S.append("PROMPT: " + r["prompt"])
        S.append("COMPLETION: " + r["completion"])
    long_acc = {}
    for label in ("baseline/long_pool", "evaluate/long_pool", "baseline/textbook72", "evaluate/textbook72"):
        rows, my = rows_all[label], myv[label]
        acc = [(r, m) for r, m in zip(rows, my) if m[0] and int(r["length"]) >= 7]
        acc8 = [(r, m) for r, m in acc if int(r["length"]) >= 8]
        nl = Counter(m[2] for _, m in acc)
        long_acc[label] = {"accepted_rows_len_ge7": len(acc), "accepted_rows_len_ge8": len(acc8),
                           "distinct_records_ge7": len({r["name"] for r, _ in acc}),
                           "n_lines_written_distribution": dict(sorted(nl.items())),
                           "records": sorted({(r["name"], r["length"]) for r, _ in acc})}
        S.append(f"\n=== [{label}] ALL accepted proofs at prompt length >= 7 ({len(acc)} rows, "
                 f"{len({r['name'] for r, _ in acc})} distinct records); n_lines distribution {dict(sorted(nl.items()))}")
        seen = set()
        for r, m in acc:
            tag = "" if r["name"] not in seen else " (another sample of the same record)"
            seen.add(r["name"])
            S.append(f"--- {r['name']} length={r['length']} n_lines(mine)={m[2]} trivial={is_trivial(r['prompt'])}{tag}")
            S.append("PROMPT: " + r["prompt"])
            S.append("COMPLETION: " + r["completion"])
    out["long_accepted"] = long_acc
    say("accepted at length>=7: " + json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "records"} for k, v in long_acc.items()}))
    # all textbook solved records, both dumps, one accepted sample each
    S.append("\n=== textbook72 solved records (one accepted sample each)")
    for label in ("baseline/textbook72", "evaluate/textbook72"):
        rows, my = rows_all[label], myv[label]
        done = set()
        for r, m in zip(rows, my):
            if m[0] and r["name"] not in done:
                done.add(r["name"])
                S.append(f"--- [{label}] {r['name']} slice={tb_by_name[r['name']]['slice']} "
                         f"reference_lines={tb_by_name[r['name']].get('reference_lines')} n_lines(mine)={m[2]} "
                         f"trivial={is_trivial(r['prompt'])}")
                S.append("PROMPT: " + r["prompt"])
                S.append("COMPLETION: " + r["completion"])
    with open(os.path.join(HERE, "samples.txt"), "w") as f:
        f.write("\n".join(S) + "\n")

    # ---------------------------------------------------------------- gate B: metric coverage
    required = {
        "mean reward": "reward",
        "reward std within groups": "reward_std",
        "pass rate by proof length (periodic eval)": "eval/pass_rate_len<L> (eval lines only)",
        "KL to reference": "kl",
        "policy entropy": "entropy",
        "fraction unparseable": "reward/unparseable_frac",
        "fraction invalid": "reward/invalid_frac",
        "completion length mean": "completions/mean_length",
        "completion length max": "completions/max_length",
        "fraction hitting the cap": "completions/clipped_ratio",
        "gradient norm": "grad_norm",
        "learning rate": "learning_rate",
        "rollout throughput": "rollout/completions_per_sec",
        "utc": "utc",
        "step": "step",
    }
    cov = {}
    for name, key in required.items():
        if key.startswith("eval/pass_rate_len"):
            present = all(any(k.startswith("eval/pass_rate_len") for k in m) for m in ev)
            cov[name] = {"key": key, "present_on_all_eval_lines": present, "n_eval_lines": len(ev)}
        else:
            present = all(key in m for m in roll)
            cov[name] = {"key": key, "present_on_all_rollout_lines": present}
    out["metric_coverage"] = cov
    say("metric coverage: " + json.dumps(cov))

    with open(os.path.join(HERE, "recount.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    with open(os.path.join(HERE, "recount_stdout.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
