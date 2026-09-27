"""Evaluator (plan T3): solve rates by proof length, textbook solves per slice (dev58 / train14 /
all72), next-token accuracy (the T1 gate), and the plan section 5 metrics beyond pass@k.

The headline number is the **avg solve rate** (`avg`): each problem's fraction of verified
samples (unbiased pass@1), averaged over problems. `solved` / `rate` are the older any-of-k
counts (pass@k at the evaluated k), kept as a secondary number. See `eval.metrics`.

All numbers come from proofs judged by `eval.reward.verify_body` (the strict verifier). Every
judged proof is written to a JSONL dump so a reviewer can recount without this code.

    import ndpath
    from hf_wrapper import load_pretrained_wrapper
    from eval import evaluate_pool, evaluate_textbook, next_token_accuracy
    model, tok = load_pretrained_wrapper()
    res = evaluate_pool(model, tok, records, k=1, temperature=0.0, dump="greedy.jsonl")

`aggregate()` is a pure function of (records, judged rows) so it can be unit-tested and so the
frozen-checkpoint control and an RL model are always scored by identical code.
"""

from __future__ import annotations

import collections
import datetime as _dt
import json
import math
import os
import time

import ndpath  # noqa: F401
import torch

# isort: split
from data import length_of
from eval.metrics import frontier, k_at, mean_ci, reliable_frac, self_knowledge
from eval.progress import valid_prefix
from eval.reward import decode_ids, fork_tokenizer, is_unparseable, verify_body

UNKNOWN_LENGTH = -1  # records with no reference length (the 14 textbook_train problems)


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def wilson(k: int, n: int, z: float = 1.96):
    """Two-sided Wilson interval; (0, 0) for n == 0."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


@torch.no_grad()
def generate(
    model, tok, prompts, k=1, temperature=0.0, max_new=400, batch_size=64, seed=0, use_cache=True
):
    """list (len prompts) of lists (len k) of completion-id lists, cut after the first QED."""
    device = next(model.parameters()).device
    eos, pad = tok.eos_token_id, tok.pad_token_id
    flat = [p for p in prompts for _ in range(k)]
    out = []
    was_training = model.training
    model.eval()
    for i in range(0, len(flat), batch_size):
        chunk = flat[i : i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True).to(device)
        kwargs = dict(
            max_new_tokens=max_new, pad_token_id=pad, eos_token_id=eos, use_cache=use_cache
        )
        if temperature > 0:
            torch.manual_seed(seed * 1_000_003 + i)
            kwargs.update(do_sample=True, temperature=temperature, top_k=0, top_p=1.0)
        else:
            kwargs.update(do_sample=False)
        ids = model.generate(**enc, **kwargs)
        comp = ids[:, enc["input_ids"].shape[1] :].tolist()
        for c in comp:
            if eos in c:
                c = c[: c.index(eos) + 1]
            out.append([x for x in c if x != pad])
    if was_training:
        model.train()
    return [out[j * k : (j + 1) * k] for j in range(len(prompts))]


def judge(prompts, completions_ids, tok_mode="abs"):
    """For each prompt: list of (body, ok, reason, n_lines, n_tokens) over its k completions."""
    rows = []
    for prompt, comps in zip(prompts, completions_ids, strict=True):
        judged = []
        for ids in comps:
            body = decode_ids(tok_mode, ids)
            ok, reason, n_lines = verify_body(prompt, body)
            judged.append((body, bool(ok), reason, n_lines, len(ids)))
        rows.append(judged)
    return rows


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n samples with c correct (Chen et al. 2021): 1 - C(n-c, k) / C(n, k)."""
    if k > n:
        raise ValueError(f"k={k} > n={n}")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_at_k_curve(counts, ks=None) -> dict:
    """Mean unbiased pass@k over prompts, for each k in `ks` (default powers of two up to n).

    `counts` is a list of (n, c) per prompt; k runs up to the smallest n. This is the curve
    Yue et al. (NeurIPS 2025) use to ask whether RL solves anything the base model cannot at
    large k.
    """
    if not counts:
        return {}
    n = min(m for m, _ in counts)
    if ks is None:
        ks = [2**i for i in range(n.bit_length()) if 2**i <= n]
    return {int(k): sum(pass_at_k(m, c, k) for m, c in counts) / len(counts) for k in ks if k <= n}


def distinct_verified(js) -> int:
    """Distinct verified proof bodies among one prompt's samples (whitespace-normalised)."""
    return len({" ".join(j[0].split()) for j in js if j[1]})


def _avg_block(ps):
    m, ci = mean_ci(ps)
    return {"avg": m, "avg_ci": ci, "reliable50": reliable_frac(ps, 0.5),
            "reliable90": reliable_frac(ps, 0.9)}


def _mean_or_none(xs):
    return sum(xs) / len(xs) if xs else None


def aggregate(
    records, judged, k=1, temperature=0.0, secs=None, label="pool", max_new=None, logprobs=None
):
    """Solve rates by length, max length solved, per-slice counts, sample stats.

    `judged[i]` is the list of (body, ok, reason, n_lines, n_tokens) for records[i].
    Per length, slice, and overall:
      avg, avg_ci           mean per-problem fraction of verified samples (pass@1), 95% CI
      reliable50/90         fraction of problems solved in >= 50% / >= 90% of samples
      solved, rate, ci      any of the k samples verifies (pass@k at k), Wilson CI; secondary
    Per length only: `valid_prefix_failed` (mean valid leading lines of failed samples as a
    fraction of the reference length, capped at 1), `proof_len_ratio` (verified proof lines over
    the reference length), and for k > 1 the pass@k curve and `k50`. Top level: `frontier_avg50`
    (longest length with avg >= 0.5 at it and every shorter length) and, if `logprobs` (a list
    per record of per-sample sequence log-probabilities) is given, `self_knowledge`.
    """
    solved_by, n_by = collections.Counter(), collections.Counter()
    slices = collections.defaultdict(lambda: [0, 0])
    ge7 = [0, 0]
    n_unparse = n_invalid = n_total = n_capped = 0
    tok_lens, solved_lengths = [], []
    counts_by = collections.defaultdict(list)  # length -> [(n, c)] for the pass@k curve
    distinct = []  # distinct verified proofs per solved record
    ps_by = collections.defaultdict(list)  # length -> per-problem success rates
    slice_ps = collections.defaultdict(list)
    ge7_ps = []
    vp_failed = collections.defaultdict(list)  # length -> valid-prefix fraction of failed samples
    len_ratio = collections.defaultdict(list)  # length -> verified lines / reference length
    for r, js in zip(records, judged, strict=True):
        L = length_of(r, default=UNKNOWN_LENGTH)
        ok_any = any(j[1] for j in js)
        c = sum(bool(j[1]) for j in js)
        counts_by[L].append((len(js), c))
        p = c / len(js) if js else 0.0
        ps_by[L].append(p)
        if L > 0:
            for body, ok, _reason, n_lines, _n_tokens in js:
                if ok:
                    len_ratio[L].append(n_lines / L)
                else:
                    vp = valid_prefix(r["prompt"], body)["n_valid"]
                    vp_failed[L].append(min(vp / L, 1.0))
        if ok_any:
            distinct.append(distinct_verified(js))
        n_by[L] += 1
        solved_by[L] += ok_any
        if ok_any and L != UNKNOWN_LENGTH:
            solved_lengths.append(L)
        if "slice" in r:
            slice_ps[r["slice"]].append(p)
            slice_ps["all"].append(p)
            if L >= 7:
                ge7_ps.append(p)
            slices[r["slice"]][0] += ok_any
            slices[r["slice"]][1] += 1
            slices["all"][0] += ok_any
            slices["all"][1] += 1
            if L >= 7:
                ge7[0] += ok_any
                ge7[1] += 1
        for _body, ok, reason, _n_lines, n_tokens in js:
            n_total += 1
            n_unparse += is_unparseable(reason)
            n_invalid += not ok
            tok_lens.append(n_tokens)
            if max_new is not None and n_tokens >= max_new:
                n_capped += 1
    by_length = {}
    for L in sorted(n_by):
        lo, hi = wilson(solved_by[L], n_by[L])
        by_length[L] = {
            "solved": solved_by[L],
            "n": n_by[L],
            "rate": solved_by[L] / n_by[L],
            "ci": [lo, hi],
            **_avg_block(ps_by[L]),
            "valid_prefix_failed": _mean_or_none(vp_failed[L]),
            "proof_len_ratio": _mean_or_none(len_ratio[L]),
        }
        if k > 1:
            by_length[L]["pass_at_k"] = pass_at_k_curve(counts_by[L])
            by_length[L]["k50"] = k_at(by_length[L]["pass_at_k"], 0.5)
    tot_s, tot_n = sum(solved_by.values()), sum(n_by.values())
    lo, hi = wilson(tot_s, tot_n)
    res = {
        "label": label,
        "utc": utc_now(),
        "k": k,
        "temperature": temperature,
        "n_records": tot_n,
        "overall": {
            "solved": tot_s,
            "n": tot_n,
            "rate": tot_s / max(tot_n, 1),
            "ci": [lo, hi],
            **_avg_block([p for L in ps_by for p in ps_by[L]]),
        },
        "by_length": by_length,
        "frontier_avg50": frontier(by_length, "avg", 0.5),
        "valid_prefix_failed": _mean_or_none([x for L in vp_failed for x in vp_failed[L]]),
        "max_length_solved": max(solved_lengths, default=0),
        "samples": n_total,
        "unparseable_frac": n_unparse / max(n_total, 1),
        "invalid_frac": n_invalid / max(n_total, 1),
        "completion_tokens_mean": sum(tok_lens) / max(len(tok_lens), 1),
        "completion_tokens_max": max(tok_lens, default=0),
        "capped_frac": n_capped / max(n_total, 1) if max_new is not None else None,
    }
    if k > 1:
        # Unbiased pass@j for j = 1, 2, 4, ..., k from the same samples (no extra generation),
        # and how many different verified proofs a solved prompt has (diversity; Invisible Leash).
        res["pass_at_k"] = pass_at_k_curve([c for L in counts_by for c in counts_by[L]])
        res["distinct_verified_mean"] = sum(distinct) / max(len(distinct), 1)
    if logprobs is not None:
        res["self_knowledge"] = self_knowledge(logprobs, judged)
    if secs is not None:
        res["secs"] = secs
        res["samples_per_sec"] = n_total / max(secs, 1e-9)
    if slices:
        res["by_slice"] = {
            s: {"solved": v[0], "n": v[1], "rate": v[0] / max(v[1], 1), "ci": list(wilson(*v)),
                **_avg_block(slice_ps[s]), "expected_solved": sum(slice_ps[s]),
                "reliable50_count": sum(p >= 0.5 for p in slice_ps[s])}
            for s, v in slices.items()
        }
        res["ref_len_ge7"] = {"solved": ge7[0], "n": ge7[1], "ci": list(wilson(*ge7)),
                              **_avg_block(ge7_ps), "expected_solved": sum(ge7_ps),
                              "reliable50_count": sum(p >= 0.5 for p in ge7_ps)}
    return res


def dump_rows(path, records, judged, k, temperature, label, logprobs=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        for i, (r, js) in enumerate(zip(records, judged, strict=True)):
            for s, (body, ok, reason, n_lines, n_tokens) in enumerate(js):
                extra = {} if ok else {"n_valid_prefix": valid_prefix(r["prompt"], body)["n_valid"]}
                if logprobs is not None:
                    extra["logprob"] = logprobs[i][s]
                if "variant" in r:
                    extra.update(variant_of=r["variant_of"], variant=r["variant"])
                f.write(
                    json.dumps(
                        {
                            "label": label,
                            "name": r.get("name"),
                            "prompt": r["prompt"],
                            "length": length_of(r, default=UNKNOWN_LENGTH),
                            "slice": r.get("slice"),
                            "completion": body,
                            "ok": ok,
                            "reason": reason,
                            "n_lines_written": n_lines,
                            "n_tokens": n_tokens,
                            "k": k,
                            "temperature": temperature,
                            **extra,
                        }
                    )
                    + "\n"
                )


def evaluate_pool(
    model,
    tok,
    records,
    k=1,
    temperature=0.0,
    max_new=400,
    batch_size=64,
    seed=0,
    dump=None,
    label="pool",
    use_cache=True,
    score=False,
    return_judged=False,
):
    """Generate k samples per record, judge them, aggregate. Writes every judged proof to `dump`.

    `score=True` also computes each sample's sequence log-probability under the model (one
    teacher-forced pass) for the self-knowledge metrics. `return_judged=True` returns
    (result, judged) instead of result."""
    tok_mode = getattr(tok, "tok_mode", getattr(model.config, "tok_mode", "abs"))
    prompts = [r["prompt"] for r in records]
    t0 = time.time()
    comps = generate(
        model,
        tok,
        prompts,
        k=k,
        temperature=temperature,
        max_new=max_new,
        batch_size=batch_size,
        seed=seed,
        use_cache=use_cache,
    )
    secs = time.time() - t0
    judged = judge(prompts, comps, tok_mode)
    logprobs = None
    if score:
        flat = sequence_logprobs(
            model, tok, [p for p, cs in zip(prompts, comps, strict=True) for _ in cs],
            [c for cs in comps for c in cs], batch_size=batch_size,
        )
        logprobs, i = [], 0
        for cs in comps:
            logprobs.append(flat[i : i + len(cs)])
            i += len(cs)
    if dump:
        dump_rows(dump, records, judged, k, temperature, label, logprobs)
    res = aggregate(
        records, judged, k=k, temperature=temperature, secs=secs, label=label, max_new=max_new,
        logprobs=logprobs,
    )
    return (res, judged) if return_judged else res


def evaluate_textbook(model, tok, records72, k=64, temperature=1.0, **kw):
    """Textbook solves per slice (dev58 / train14 / all) and by reference length."""
    kw.setdefault("label", "textbook72")
    return evaluate_pool(model, tok, records72, k=k, temperature=temperature, **kw)


def textbook_avg_table(named_results: dict[str, dict]) -> str:
    """Markdown table of the avg solve rate per slice: expected problems solved (sum of per-problem
    rates) and problems solved in >= 50% of samples, for several models."""
    cols = list(named_results)
    lines = ["| slice | " + " | ".join(f"{c}: avg [95% CI] | {c}: expected / reliable" for c in cols)
             + " |", "| --- |" + " --- | --- |" * len(cols)]
    rows = [(s, lambda r, s=s: r.get("by_slice", {}).get(s)) for s in ("dev58", "train14", "all")]
    rows.append(("reference length >= 7", lambda r: r.get("ref_len_ge7")))
    for name, get in rows:
        cells = []
        for c in cols:
            v = get(named_results[c])
            if not v or "avg" not in v:
                cells += ["-", "-"]
                continue
            cells += [f"{v['avg']:.3f} [{v['avg_ci'][0]:.3f}, {v['avg_ci'][1]:.3f}]",
                      f"{v['expected_solved']:.1f} / {v['reliable50_count']} of {v['n']}"]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def textbook_table(named_results: dict[str, dict]) -> str:
    """Markdown table of textbook solves per slice for several models (e.g. control vs RL)."""
    cols = list(named_results)
    lines = ["| slice | " + " | ".join(cols) + " |", "| --- | " + " | ".join("---" for _ in cols) + " |"]
    for s in ("dev58", "train14", "all"):
        cells = []
        for c in cols:
            v = named_results[c].get("by_slice", {}).get(s, {"solved": 0, "n": 0})
            cells.append(f"{v['solved']} / {v['n']}")
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    cells = []
    for c in cols:
        v = named_results[c].get("ref_len_ge7", {"solved": 0, "n": 0})
        cells.append(f"{v['solved']} / {v['n']}")
    lines.append("| reference length >= 7 | " + " | ".join(cells) + " |")
    return "\n".join(lines)


@torch.no_grad()
def next_token_accuracy(model, tok, records, batch_size=256, max_records=None):
    """Prompt-masked next-token accuracy and mean loss on records with `prompt` and `proof`.

    This is the T1 gate. Targets are the proof-body tokens (QED included); the prompt is context.
    Sequences are right-padded, so causal attention never sees a pad from a real position.
    """
    device = next(model.parameters()).device
    from eval.reward import fork_tokenizer

    ft = fork_tokenizer(getattr(tok, "tok_mode", getattr(model.config, "tok_mode", "abs")))
    recs = records[:max_records] if max_records else records
    correct = total = 0
    loss_sum = 0.0
    model.eval()
    for i in range(0, len(recs), batch_size):
        chunk = recs[i : i + batch_size]
        seqs, masks = [], []
        for r in chunk:
            p, q = ft.encode_prompt(r["prompt"]), ft.encode_proof(r["proof"])
            seqs.append(p + q)
            masks.append([0] * len(p) + [1] * len(q))
        T = max(len(s) for s in seqs)
        x = torch.full((len(seqs), T), ft.pad, dtype=torch.long)
        m = torch.zeros((len(seqs), T), dtype=torch.bool)
        for j, (s, mk) in enumerate(zip(seqs, masks, strict=True)):
            x[j, : len(s)] = torch.tensor(s)
            m[j, : len(s)] = torch.tensor(mk, dtype=torch.bool)
        x, m = x.to(device), m.to(device)
        logits = model(input_ids=x, use_cache=False).logits[:, :-1].float()
        tgt, mk = x[:, 1:], m[:, 1:]
        pred = logits.argmax(-1)
        correct += int(((pred == tgt) & mk).sum())
        total += int(mk.sum())
        ce = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="none"
        )
        loss_sum += float((ce * mk.reshape(-1)).sum())
    return {
        "utc": utc_now(),
        "next_token_accuracy": correct / max(total, 1),
        "loss": loss_sum / max(total, 1),
        "n_records": len(recs),
        "n_target_tokens": total,
    }


@torch.no_grad()
def sequence_logprobs(model, tok, prompts, completions_ids, batch_size=256):
    """Total log-probability (temperature 1) of each completion given its prompt, one per pair.

    Teacher-forced; the prompt is encoded with the fork tokenizer exactly as in
    `next_token_accuracy`, and the completion ids are scored as generated (QED included).
    """
    device = next(model.parameters()).device
    ft = fork_tokenizer(getattr(tok, "tok_mode", getattr(model.config, "tok_mode", "abs")))
    out = []
    was_training = model.training
    model.eval()
    for i in range(0, len(prompts), batch_size):
        ps, cs = prompts[i : i + batch_size], completions_ids[i : i + batch_size]
        seqs, starts = [], []
        for p, c in zip(ps, cs, strict=True):
            e = ft.encode_prompt(p)
            seqs.append(e + [int(x) for x in c])
            starts.append(len(e))
        T = max(len(s) for s in seqs)
        x = torch.full((len(seqs), T), ft.pad, dtype=torch.long)
        m = torch.zeros((len(seqs), T), dtype=torch.bool)
        for j, (s, st) in enumerate(zip(seqs, starts, strict=True)):
            x[j, : len(s)] = torch.tensor(s)
            m[j, st : len(s)] = True
        x, m = x.to(device), m.to(device)
        logp = torch.log_softmax(model(input_ids=x, use_cache=False).logits[:, :-1].float(), -1)
        tgt, mk = x[:, 1:], m[:, 1:]
        lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * mk
        out += lp.sum(-1).tolist()
    if was_training:
        model.train()
    return out


def known_proof_logprob(model, tok, proofs_by_prompt: dict, length_by_prompt: dict,
                        batch_size=256) -> dict:
    """How likely the model finds a known-valid proof: a continuous, sampling-free metric.

    `proofs_by_prompt` maps a prompt to verified proof bodies (for example every proof any model
    found in an evaluation). For each prompt the best (highest) total log-probability over its
    proofs is taken; reported by length as the mean best log-probability and the mean per-token
    log-probability of that best proof. Compare models on the same proof set only.
    """
    ft = fork_tokenizer(getattr(tok, "tok_mode", getattr(model.config, "tok_mode", "abs")))
    pairs = [(p, b) for p, bodies in proofs_by_prompt.items() for b in sorted(set(bodies))]
    ids = [ft.encode_proof(b) for _, b in pairs]
    lps = sequence_logprobs(model, tok, [p for p, _ in pairs], ids, batch_size=batch_size)
    best = {}
    for (p, _b), c, lp in zip(pairs, ids, lps, strict=True):
        if p not in best or lp > best[p][0]:
            best[p] = (lp, lp / max(len(c), 1))
    by = collections.defaultdict(list)
    for p, (lp, per_tok) in best.items():
        by[int(length_by_prompt.get(p, UNKNOWN_LENGTH))].append((lp, per_tok))
    return {
        "n_prompts": len(best),
        "n_proofs": len(pairs),
        "by_length": {
            L: {"n": len(v), "best_logprob": sum(a for a, _ in v) / len(v),
                "best_logprob_per_token": sum(b for _, b in v) / len(v)}
            for L, v in sorted(by.items())
        },
    }


def evaluate_variants(model, tok, records, m=4, k=16, temperature=1.0, seed=0, exclude_keys=None,
                      **kw):
    """Consistency across equivalent variants (atom renaming, premise order) of each record.

    Evaluates the originals and up to m variants each at k samples, and returns the consistency
    summary (`eval.variants.consistency`) plus the pooled aggregate and how many variants were
    dropped for sharing a renaming class with a training pool.
    """
    from eval.variants import consistency, variant_records

    recs, dropped = variant_records(records, m, seed=seed, exclude_keys=exclude_keys)
    kw.setdefault("label", "variants")
    res, judged = evaluate_pool(model, tok, recs, k=k, temperature=temperature, seed=seed,
                                return_judged=True, **kw)
    out = consistency(recs, judged)
    out.update({"m": m, "k": k, "temperature": temperature, "n_records": len(recs),
                "dropped_training_class": dropped, "pooled": res})
    return out
