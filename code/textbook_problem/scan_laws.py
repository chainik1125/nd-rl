"""Find textbook exercises and examples that are instances of the 9 standard laws missing from the
textbook 72 (docs/textbook-problem/diversification/weaknesses.md section 7).

Discovery only: it parses every \\item and every $...$ formula in the given LaTeX files (and every
Carnap goal in the given HTML files), matches it against the 9 schemata in every form the
matcher knows (direct, theorem form, conjoined premises, biconditional), and audits each hit by
renaming class. Hits that overlap the sealed sets (textbook 72, test files) are counted, not
written.

    cd code && uv run python textbook_problem/scan_laws.py --out ../data/textbook_problem/law_scan \
        --src NAME=DIR [--src NAME=DIR ...]
"""

import argparse
import collections
import glob
import html
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
from prove import named_laws  # noqa: E402
from transcribe import (  # noqa: E402
    ParseError,
    canon_key,
    canonical_theorem,
    clean_item,
    load_pool,
    load_textbook72,
    make_prompt,
    parse_item,
    rename,
    thm_of_prompt,
)

MISSING = {
    "contraposition",
    "contraposition_conv",
    "modus_tollens",
    "export",
    "excluded_middle",
    "explosion",
    "constructive_dilemma",
    "negated_conditional_conv",
    "dist_and_over_or_conv",
}
MATH = re.compile(r"\$([^$]+)\$")
SECTION = re.compile(r"\\(chapter|section|subsection)\*?\{([^}]*)\}")


def candidates_in_line(line):
    out = []
    if r"\item" in line:
        out.append(clean_item(line))
    out += MATH.findall(line)
    out.append(re.sub(r"(?<!\\)%.*", "", line).replace("$", " "))
    return out


def scan_tex(name, root):
    for path in sorted(glob.glob(os.path.join(root, "**/*.tex"), recursive=True)):
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.read().split("\n")
        section, practice = "", False
        for no, line in enumerate(lines, 1):
            m = SECTION.search(line)
            if m:
                section, practice = m.group(2)[:60], False
            if r"\practiceproblems" in line or r"\begin{exercises}" in line:
                practice = True
            if re.match(r"\s*%", line):
                continue
            seen = set()
            for text in candidates_in_line(line):
                try:
                    kind, seqs, notes = parse_item(text)
                except (ParseError, IndexError, KeyError):
                    continue
                for prem, goal in seqs:
                    key = (tuple(prem), goal)
                    if key in seen:
                        continue
                    seen.add(key)
                    yield dict(
                        source=name,
                        file=os.path.relpath(path, root),
                        line=no,
                        section=section,
                        in_practice=practice,
                        original=line.strip()[:300],
                        kind=kind,
                        prem=prem,
                        goal=goal,
                    )


def scan_carnap(name, root):
    for path in sorted(glob.glob(os.path.join(root, "**/*.html"), recursive=True)):
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        for tag in re.findall(r'<div((?:\s+[\w-]+="[^"]*")*)\s*>', text):
            attrs = dict(re.findall(r'data-carnap-([a-z]+)="([^"]*)"', tag))
            if attrs.get("type") != "proofchecker" or attrs.get("system") != "prop":
                continue
            goal = html.unescape(attrs.get("goal", ""))
            try:
                kind, seqs, _ = parse_item(goal, latex=False)
            except (ParseError, IndexError, KeyError):
                continue
            for prem, g in seqs:
                yield dict(
                    source=name,
                    file=os.path.relpath(path, root),
                    line=attrs.get("submission", ""),
                    section="",
                    in_practice=True,
                    original=goal.strip(),
                    kind=kind,
                    prem=prem,
                    goal=g,
                )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", action="append", default=[], help="NAME=DIR of .tex files")
    ap.add_argument("--carnap", action="append", default=[], help="NAME=DIR of Carnap .html")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    def keysets(rows):
        return (
            {canonical_theorem(r["prompt"]) for r in rows},
            {canon_key(thm_of_prompt(r["prompt"])) for r in rows},
        )

    sealed = [
        keysets(load_textbook72()),
        keysets(load_pool("test_short_prompts") + load_pool("test_long_prompts")),
    ]
    named = {n: keysets(load_pool(n)) for n in ("validation_36", "rl_targets", "transfer")}
    named.update({n: keysets(load_pool(n)) for n in ("heldout", "train")})

    items = []
    for spec in args.src:
        name, root = spec.split("=", 1)
        items += list(scan_tex(name, root))
    for spec in args.carnap:
        name, root = spec.split("=", 1)
        items += list(scan_carnap(name, root))

    hits, counts = [], collections.Counter()
    seen = {}
    for it in items:
        try:
            prem, goal, _ = rename(it["prem"], it["goal"])
        except ParseError:
            continue
        laws = [x for x in named_laws(prem, goal) if x.split(" (")[0] in MISSING]
        if not laws:
            continue
        prompt = make_prompt(prem, goal)
        ct, ck = canonical_theorem(prompt), canon_key(thm_of_prompt(prompt))
        counts["hits (occurrences)"] += 1
        if any(ct in a or ck in b for a, b in sealed):
            counts["sealed overlap (not recorded)"] += 1
            continue
        overlap = [n for n, (a, b) in named.items() if ct in a or ck in b]
        rec = {
            "prompt": prompt,
            "laws": laws,
            "overlap": overlap,
            **{k: it[k] for k in ("source", "file", "line", "section", "in_practice", "original")},
        }
        if ct in seen:
            seen[ct]["also_in"].append(f"{it['source']}:{it['file']}:{it['line']}")
            continue
        rec["also_in"] = []
        seen[ct] = rec
        hits.append(rec)

    for r in hits:
        counts["distinct classes"] += 1
        counts[
            "usable (no overlap)" if not r["overlap"] else "overlap: " + ",".join(r["overlap"])
        ] += 1
    by_law = collections.Counter(
        law.split(" (")[0] for r in hits if not r["overlap"] for law in r["laws"]
    )
    with open(os.path.join(args.out, "hits.jsonl"), "w") as f:
        for r in hits:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "items_parsed": len(items),
        "counts": dict(counts),
        "usable_by_law": {k: by_law.get(k, 0) for k in sorted(MISSING)},
        "sources": [s.split("=", 1)[0] for s in args.src + args.carnap],
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
