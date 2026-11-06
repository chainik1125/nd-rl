---
written_on: 2026-10-14
written_by: agent:claude (subagent report, saved by the coordinating session)
---

# Approach E: long proofs (35–80 body lines) with ≤ 10 atoms

## Construction (`gen_long.py`)

- **Guarded packages.** Every scope is opened by a guard atom `c`: a case-split hypothesis, or an atom derived at the
  same level (a "seq" block). Everything the scope needs sits in `(c > conjunction of items)`, nested inside the
  parent's package. A fact used under the path (c1, …, ck) costs IMPE with c1 … ck plus ANDE.
- **Atom recycling.** New atoms only avoid atoms in facts of the current scope and its ancestors, plus the goals and
  guards on the path. Sibling and cousin subtrees reuse atoms in different roles; their facts stay locked behind
  guards the other branch can never derive. This is what breaks B's ~34-line cap.
- **Plan nodes:** split, imp, seq, leaf, hleaf, bote (never bote in both branches of one split). Root: an atom, an
  AND, or an implication.
- **Premises:** root items packed into 3–10 conjunction trees; items lifted out of child packages as `(c > item)` when
  there are too few.
- **Filters:** every fact necessary under its full guard path (bitmask truth tables); `certify.check`; no dead lines in
  the written proof; `gen_deep.necessary_premises`; no non-atomic fact or package formula appears twice.
- **Shortcuts found by measurement and filtered:** both branches of a split closed by bote (ORE derives F once:
  17 → 16), and identical items in sibling branches (ORE derives the item once, everything after is shared: 11 → 8,
  20 → 17).
- Off in production: `--share` (repeated items where no split's two branches can both derive them) and `--p_twin`
  (clones cousin subtrees; cut |S| only 10–15%, not tightness-tested).

## Production file

`production/v2/e_raw.jsonl`: 3,220 records, exactly 70 per written length 35–80, shuffled
(`--per_len 70 --seed 2001 --budget 30-120 --body 35-80`). Premises 3–10 (344–492 each), atoms 7–10. Record format
is gen_deep's plus `S` (minlen's formula-set size) and `reuse` (atom choices recycled).

## Certification

Certified ≥ 20 at `--secs 15`, by written length:

| sample | 35–39 | 40–49 | 50–59 | 60–69 | 70–80 |
| --- | --- | --- | --- | --- | --- |
| dev, 92 records, 1024 MB | 0/10 | 0/20 | 0/20 | 0/20 | 0/22 |
| production, first 225 | 0/5 | 0/18 | 0/12 | 0/9 | 0/10 |

All timeouts; nothing shorter found. Production stopped at 390/3,220 (all timeouts) by the coordinator.

Why: certification cost tracks |S|, and here |S| ≈ 2 × written + 20 (median ~100 at 35 lines, ~175 at 80). B
certifies in 15 s mostly at |S| ≤ 80 and times out on 60% at 80–100. On E records, search level 16 finishes in a
median 3–6 s and each level costs ~2.5× the previous, so floor 20 needs ~60–100 s even at 35–40 lines (timed under
heavy machine load).

## Tightness (small variants of the same construction, floor = written length, 15 s)

| final generator | decided | tight | shorter found | timeouts |
| --- | --- | --- | --- | --- |
| 5–6 atoms (heavy recycling), 16–24 lines, 60 records | 56 | 56 | 0 | 4 |
| 8–10 atoms, 16–24 lines, 60 records | 41 | 41 | 0 | 19 |

Before the two shortcut filters: 46/47 and 70/73 decided were tight. After the fixes, a seed-5 run with 5–6 atoms
had 92/92 decided tight. Mid-size variants (25–31 lines, 60 s): 0/10 decided.

Long records (12 records, 35–68 lines, floor 40, 60 s): none exhausted; the completed levels prove no proof under
15–19 lines exists (varies by record) and no shorter proof was found.

## Caveats

- Minimality at 35–80 lines is inferred, not certified: the construction argument, tight small variants, and partial
  lower bounds of 15–19. Shortcuts that appear only in deep combinations (seq chains, deep nesting, lifted wrappers)
  would go undetected.
- About 40–50% of lines are ANDE or guard-IMPE unpacking: necessary but mechanical.
- One structural family.
- Untried routes to certified long data: 100–300 s per record at 35–45 lines, the lowest-|S| records only, or a
  certifier that bounds each guarded scope separately.

Files: `gen_long.py`, `measure.py` (tightness/certification by bucket), `probe_cost.py` (per-level timings, partial
bounds), `measure/` (samples, results, `run.sh`).

## Seam set: written 20–34 (2026-10-14, 15:22 PDT)

Why: the long_rl pool's 35–80-line targets are all approach E, while its 20–34-line targets come from B and C, so RL
had no E-style rungs below 35. Same generator and settings as `e_raw` except budget and body range.

```
P=~/repos/nd-takehome/.venv/bin/python
$P approach_e_long/gen_long.py production/v2/e_seam_raw.jsonl --n 3300 --per_len 220 --seed 4001 --budget 12-45 --body 20-34
$P certifier_scoped/run_e.py production/v2/e_seam_raw.jsonl certifier_scoped/results/e_seam_sig.jsonl
$P approach_e_long/seam_finish.py      # --min_sig 15 --per_len 200: Lean-check, dedup, write production/v2/e_seam_sig.jsonl
```

Atoms default to 8–10 as for `e_raw` (6–10 actually used per theorem); premises 3–10.

| stage | records | time |
| --- | --- | --- |
| generated (220 per written length 20–34; 28,560 tries) | 3,300 | 10 s |
| signature bound ≥ 15 | 3,300 (min bound 18) | 4 s |
| Lean-accepted (`certifier_scoped/lean_check.py`, 6 procs) | 3,300 (0 rejects) | 15 s incl. dedup |
| after dedup by `canon_key` | 3,279 (21 self-duplicates; 0 against `e_raw` or long_rl rl/dev/holdout) | |
| final, capped at 200 per written length | 3,000 | |

`gen_long.py` dedups only by the approximate `certify.theorem_key`, which missed the 21 isomorphic pairs.

Bound vs written length in the final file:

| written | n | bound ≥ 20 | exact (bound = written) | bound min / median / max | bound / written mean (min) |
| --- | --- | --- | --- | --- | --- |
| 20–24 | 1,000 | 932 | 639 | 18 / 22 / 24 | 0.981 (0.875) |
| 25–29 | 1,000 | 1,000 | 518 | 22 / 26 / 29 | 0.978 (0.880) |
| 30–34 | 1,000 | 1,000 | 474 | 27 / 31 / 34 | 0.977 (0.879) |

Premises 3: 450, 4: 420, 5: 384, 6: 411, 7: 394, 8: 421, 9: 321, 10: 199. Output records have `e_raw_sig`'s fields plus
`text` (the written proof). Note for `long_rl_20261014/make_data.py`: it keeps an E record only when the bound equals
the written length or is ≥ 20, so 68 records (bound 18–19 at written 20–21) would be dropped there; 2,932 pass.
Logs: `production/v2/e_seam_raw.log`, `certifier_scoped/results/e_seam_sig.log`, `production/v2/e_seam_sig.log`.
