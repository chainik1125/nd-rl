"""Independent incremental rule checker and prefix-conditioned relative codec.

Only the parser is shared with the original checker. Rule/scope validation here
is a fresh implementation. Complete witnesses must pass BOTH checkers.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import common
from nd_data import formula_text
from nd_verify import parse_formula, parse_proof_tokens
from strict_verify import safe_verify
from nd_model import ProofTokenizer

FORMAT = "nd-prefix-double-prf-final-v1"
BOT = ("bot",)

class Tokenizer(ProofTokenizer):
    VERSION = "relative-unary-r1-symbolic-v1"
    TOKENS = ProofTokenizer.TOKENS + ("<THINK>", "</THINK>", "<FINAL>")

def parse_prompt(prompt):
    words = prompt.split()
    if not words or words[0] != "THM" or words[-1] != "PRF":
        raise ValueError("Malformed original theorem")
    pos, premises = 1, []
    while words[pos] != "SEQ":
        f, pos = parse_formula(words, pos)
        premises.append(f)
        if words[pos] == ",":
            pos += 1
        elif words[pos] != "SEQ":
            raise ValueError("Malformed premise separator")
    conclusion, pos = parse_formula(words, pos + 1)
    if pos != len(words) - 1:
        raise ValueError("Extra prompt content")
    return premises, conclusion

def split_body(body):
    words = body.split()
    if words.count("QED") != 1 or words[-1] != "QED":
        raise ValueError("Exactly one terminal QED required")
    lines, start = [], 0
    for i, w in enumerate(words[:-1]):
        if w == ";":
            lines.append(" ".join(words[start:i + 1]))
            start = i + 1
    if start != len(words) - 1:
        raise ValueError("Incomplete inference boundary")
    return lines

def independent_replay(prompt, prefix, complete=False):
    premises, goal = parse_prompt(prompt)
    words = prefix.split()
    if complete:
        split_body(prefix)
        words = words[:-1]
    elif "QED" in words:
        raise ValueError("Prefix must not include QED")
    if words and words[-1] != ";":
        raise ValueError("Prefix ends inside an inference")
    lines = parse_proof_tokens(words + ["QED"])
    by_id, contexts, stack, boxes = {}, {}, [], {}
    pr_count, premise_open = 0, True
    for position, line in enumerate(lines, 1):
        i, depth, rule, refs, g = (line[k] for k in ("idx", "depth", "rule", "refs", "formula"))
        if i != position:
            raise ValueError("Nonconsecutive canonical line indices")
        if any(r < 1 or r >= i for r in refs):
            raise ValueError("Citation must precede inference")
        if rule == "PR":
            if not premise_open or depth != 0 or pr_count >= len(premises) or g != premises[pr_count]:
                raise ValueError("Premise block differs from original theorem")
            pr_count += 1
        else:
            premise_open = False
            if pr_count != len(premises):
                raise ValueError("Missing original premise")
        if rule == "AS":
            if not 1 <= depth <= len(stack) + 1:
                raise ValueError("Invalid assumption depth")
            stack = stack[:depth - 1]
            parent = tuple(stack)
            stack.append(i)
            boxes[i] = {"start": i, "end": i, "depth": depth, "parent": parent}
        else:
            if depth > len(stack):
                raise ValueError("Scope depth jump")
            stack = stack[:depth]
        ctx = tuple(stack)
        contexts[i] = ctx
        for start in stack:
            boxes[start]["end"] = i

        def visible(r):
            old = contexts[r]
            return old == ctx[:len(old)]

        def box(s, e):
            b = boxes.get(s)
            return bool(b and s not in stack and s <= e < i and b["end"] == e
                        and by_id[e]["depth"] == b["depth"]
                        and b["parent"] == ctx[:len(b["parent"])])

        arity = {"PR": 0, "AS": 0, "R": 1, "ANDI": 2, "ANDE1": 1,
                 "ANDE2": 1, "IMPE": 2, "IMPI": 2, "ORI1": 1, "ORI2": 1,
                 "ORE": 5, "NEGE": 2, "NEGI": 2, "BOTE": 1, "DN": 1}
        if len(refs) != arity[rule]:
            raise ValueError("Incorrect rule arity")
        if rule in ("IMPI", "NEGI"):
            if not box(*refs):
                raise ValueError("Invalid discharge range")
        elif rule == "ORE":
            if not visible(refs[0]) or not box(*refs[1:3]) or not box(*refs[3:5]):
                raise ValueError("Invalid case split ranges")
        elif not all(visible(r) for r in refs):
            raise ValueError("Inaccessible citation")
        fs = [by_id[r]["formula"] for r in refs]
        correct = True
        if rule == "R": correct = g == fs[0]
        elif rule == "ANDI": correct = g == ("and", *fs)
        elif rule in ("ANDE1", "ANDE2"):
            correct = fs[0][0] == "and" and g == fs[0][1 if rule == "ANDE1" else 2]
        elif rule == "IMPE": correct = fs[0] == ("imp", fs[1], g)
        elif rule == "IMPI": correct = g == ("imp", *fs)
        elif rule in ("ORI1", "ORI2"):
            correct = g[0] == "or" and fs[0] == g[1 if rule == "ORI1" else 2]
        elif rule == "ORE": correct = fs[0] == ("or", fs[1], fs[3]) and fs[2] == fs[4] == g
        elif rule == "NEGE": correct = fs[1] == ("not", fs[0]) and g == BOT
        elif rule == "NEGI": correct = fs[1] == BOT and g == ("not", fs[0])
        elif rule == "BOTE": correct = fs[0] == BOT
        elif rule == "DN": correct = fs[0] == ("not", ("not", g))
        if not correct:
            raise ValueError(f"Invalid {rule} inference")
        by_id[i] = line
    if complete:
        if not lines or pr_count != len(premises) or stack or lines[-1]["formula"] != goal:
            raise ValueError("Incomplete theorem proof")
    available = [i for i in by_id if contexts[i] == tuple(stack[:len(contexts[i])])]
    return {"next_line": len(lines) + 1, "lines": lines,
            "line_contexts": contexts, "active_assumptions": stack,
            "available_lines": available, "boxes": boxes,
            "goal_already_available": any(by_id[i]["formula"] == goal and not contexts[i] for i in available),
            "premises_seen": pr_count}

def strict_complete(prompt, body):
    try:
        split_body(body)
        state = independent_replay(prompt, body, complete=True)
        official, reason, n = safe_verify(prompt, body)
        return {"ok": bool(official), "reason": reason, "n_lines": n, "independent_ok": True}
    except Exception as e:
        return {"ok": False, "reason": f"{type(e).__name__}: {e}", "n_lines": body.count(";"), "independent_ok": False}

def episode(record, cut, tokenizer=None, allow_terminal_prefix=False):
    tok = tokenizer or Tokenizer()
    prompt = record["prompt"]
    body = record.get("body") or record["text"].split("PRF", 1)[1].strip()
    verdict = strict_complete(prompt, body)
    if not verdict["ok"]:
        raise ValueError(f"Unverified source: {verdict['reason']}")
    lines = split_body(body)
    if not 0 <= cut < len(lines):
        raise ValueError("Reset must retain at least one source inference")
    prefix = " ".join(lines[:cut])
    state = independent_replay(prompt, prefix)
    if state["goal_already_available"] and not allow_terminal_prefix:
        raise ValueError("Supplied prefix already proves original goal")
    original_ids = tok.encode_prompt(prompt)
    full_ids = tok.encode_record(prompt + " " + body)[len(original_ids):]
    prefix_ids = tok.encode_record(prompt + " " + prefix + " QED")[len(original_ids):-1]
    if full_ids[:len(prefix_ids)] != prefix_ids:
        raise AssertionError("Relative codec prefix mismatch")
    context = original_ids + prefix_ids + [tok.prf_id]
    response = [tok.stoi["<FINAL>"]] + full_ids[len(prefix_ids):]
    return {"prompt": prompt, "prompt_ids": context, "original_prompt_ids": original_ids,
            "prefix_ids": prefix_ids, "response_ids": response,
            "reset_index": cut, "source_lines": len(lines),
            "source_id": record.get("source_id", common.digest([prompt, body])),
            "family": record.get("family", "unknown"), "prefix_state": state,
            "format": FORMAT}

def judge(episode_row, response, tok):
    ids = list(response)
    final = tok.stoi["<FINAL>"]
    if ids.count(final) != 1:
        return {"reward": 0, "reason": "Exactly one sampled FINAL required", "body": "", "n_lines": 0}
    k = ids.index(final)
    pre = ids[:k]
    if pre and (pre[0] != tok.stoi["<THINK>"] or pre[-1] != tok.stoi["</THINK>"]):
        return {"reward": 0, "reason": "Malformed sampled scratch boundary", "body": "", "n_lines": 0}
    suffix = ids[k + 1:]
    if suffix.count(tok.eos_id) != 1 or not suffix or suffix[-1] != tok.eos_id:
        return {"reward": 0, "reason": "Missing unique terminal QED", "body": "", "n_lines": 0}
    body = tok.decode_body(episode_row["prefix_ids"] + suffix)
    verdict = strict_complete(episode_row["prompt"], body)
    return {"reward": int(verdict["ok"]), "body": body, **verdict}
