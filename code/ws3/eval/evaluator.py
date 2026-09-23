"""Evaluator (plan T3): pass rate by proof length with Wilson CIs, max length solved, textbook
solves per slice (dev58 / train14 / all72), next-token accuracy (the T1 gate).

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
from eval.reward import decode_ids, is_unparseable, verify_body

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


def aggregate(records, judged, k=1, temperature=0.0, secs=None, label="pool", max_new=None):
    """Pass@k by length (Wilson CIs), max length solved, per-slice counts, sample stats.

    `judged[i]` is the list of (body, ok, reason, n_lines, n_tokens) for records[i]. A record
    counts as solved if any of its k samples verifies. Records with a `slice` field are also
    counted per slice and as `all`; `ref_len_ge7` counts records with reference length >= 7.
    """
    solved_by, n_by = collections.Counter(), collections.Counter()
    slices = collections.defaultdict(lambda: [0, 0])
    ge7 = [0, 0]
    n_unparse = n_invalid = n_total = n_capped = 0
    tok_lens, solved_lengths = [], []
    for r, js in zip(records, judged, strict=True):
        L = length_of(r, default=UNKNOWN_LENGTH)
        ok_any = any(j[1] for j in js)
        n_by[L] += 1
        solved_by[L] += ok_any
        if ok_any and L != UNKNOWN_LENGTH:
            solved_lengths.append(L)
        if "slice" in r:
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
        }
    tot_s, tot_n = sum(solved_by.values()), sum(n_by.values())
    lo, hi = wilson(tot_s, tot_n)
    res = {
        "label": label,
        "utc": utc_now(),
        "k": k,
        "temperature": temperature,
        "n_records": tot_n,
        "overall": {"solved": tot_s, "n": tot_n, "rate": tot_s / max(tot_n, 1), "ci": [lo, hi]},
        "by_length": by_length,
        "max_length_solved": max(solved_lengths, default=0),
        "samples": n_total,
        "unparseable_frac": n_unparse / max(n_total, 1),
        "invalid_frac": n_invalid / max(n_total, 1),
        "completion_tokens_mean": sum(tok_lens) / max(len(tok_lens), 1),
        "completion_tokens_max": max(tok_lens, default=0),
        "capped_frac": n_capped / max(n_total, 1) if max_new is not None else None,
    }
    if secs is not None:
        res["secs"] = secs
        res["samples_per_sec"] = n_total / max(secs, 1e-9)
    if slices:
        res["by_slice"] = {
            s: {"solved": v[0], "n": v[1], "rate": v[0] / max(v[1], 1), "ci": list(wilson(*v))}
            for s, v in slices.items()
        }
        res["ref_len_ge7"] = {"solved": ge7[0], "n": ge7[1], "ci": list(wilson(*ge7))}
    return res


def dump_rows(path, records, judged, k, temperature, label):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        for r, js in zip(records, judged, strict=True):
            for body, ok, reason, n_lines, n_tokens in js:
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
):
    """Generate k samples per record, judge them, aggregate. Writes every judged proof to `dump`."""
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
    if dump:
        dump_rows(dump, records, judged, k, temperature, label)
    return aggregate(
        records, judged, k=k, temperature=temperature, secs=secs, label=label, max_new=max_new
    )


def evaluate_textbook(model, tok, records72, k=64, temperature=1.0, **kw):
    """Textbook solves per slice (dev58 / train14 / all) and by reference length."""
    kw.setdefault("label", "textbook72")
    return evaluate_pool(model, tok, records72, k=k, temperature=temperature, **kw)


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
