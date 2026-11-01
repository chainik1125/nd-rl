---
written_on: 2026-11-01
written_by: agent:claude
---

# Log: proof strategies of the GRPO + thinking-trace models

Times are Pacific (PST from today, UTC−8), with UTC where a file stamp needs it.

## 2026-11-01

- **11:02 am PT.** Start. Branch `leon/high-rl-paper` at `1ab6ab7`, shared with the coordinating
  agent; this folder, the figure files and the summary folder are the only new files.
- **11:05 am.** Dan's classifier fetched read-only: shallow clone of `dan-pandori/nd-takehome`,
  branch `dan_proof-strategies`, HEAD `e6bfe0ea` (the review commit), into
  `/scratch/cowifornia/nd-rl/ext/nd-takehome-ps`. sha256 of the files used:
  - `proof_strategies/strategies.py` `29b2d535…6476c69`
  - `patterns.py` `e1b9f8f1…5641f39d3`, `patterns2.py` `84550e18…42019c77bf`
  - reviewer's blind reimplementation `review_ps/cls.py` `2434e5f1…7e74ca`
- **11:10 am.** Inputs located.
  - S1g3's Stage-1 corpus is *not* K12's generator proofs. `S1g3.pkl` is a graded build of
    154,630 programmatic search traces (`searchtrace/1`) of K12 theorems,
    `/scratch/cowifornia/nd-rl/think/traces/search/full/e{0,64,256}.jsonl`; the ND form of the
    actions is `plain_e256.jsonl` (154,630 rows). 35.5 % of the search proofs equal the generator's
    (`same_as_ref` 54,854 in `e256.stats.json`).
  - K12 itself (`bench/dan_ckpts/data/train_k12.jsonl`) has md5 `800b5486…`, the file Dan
    classified.
  - Reads store `accepted` as {Lean text: count}; the counts sum to `n_ok`. Texts that differ only
    in hypothesis names are distinct keys, so proofs are merged after the Lean → ND conversion.
  - GRPO runs used `dump_batches: None`; no per-iteration rollouts exist. Checkpoint reads only.
- **11:06 am.** K12 (155,000) and the S1g3 corpus (154,630) labelled with Dan's classifier, 8
  workers, under 15 s each.
  - K12 reproduces Dan's `artifacts/ps/summary/k12.json` on all 117 cells (uses, sub-types,
    primary, curried primary) and on box-free.
  - Every corpus record's replayed actions convert to exactly its `plain_e256` ND proof
    (154,630 / 154,630). The S1g3 pickle's trained actions equal the `e256` actions on every
    record.
- **11:09 am PT (19:09Z). Pre-registration frozen** before any strategy count of model proofs:
  `PREREG.md` sha256 `0e9d361d77559c974d7332627b915b6d914c6d636621c8020b0212d5dac4b3e6`; copy at
  `/scratch/cowifornia/nd-rl/proof_strategies_grpo_20261101/PREREG.frozen.md`. (A first hash,
  `09c5e484…`, was taken one minute earlier; the only change since is the time stamp in line 8.)
- **11:10 am.** First count of model proofs: `classify reads` on 87 reads (charles149 k1024 chain
  and seeds, t1320s250 k512 chain, smaller-k sets), 0 unparsable accepted texts. Theorem-level
  labels (Dan's G4ip) for the 4,159 read prompts.
- **11:12–11:20 am.** Exploration. On Charles149, the base already uses every textbook strategy at
  about the reference-proof rate. The big shift is on t1320s250, where proof by contradiction
  rises on intuitionistic problems. The GRPO pools are 40–45 % classical-only (G4ip).
  - Added an `op` (opening-move) feature to the per-proof labels.
  - Added corpus theorem labels and pool labels.
  - Re-ran every classification step (`classify_all.log`); the strategy labels are unchanged.
- **11:20 am.** Reviewer's blind classifier (`review_ps/cls.py`) vs Dan's: identical uses, working
  flags, primary and curried primary on 168,566 / 168,566 distinct model proofs and 154,630 /
  154,630 corpus records (`agree.json`).
- **11:24 am.** Blind spot-check sample drawn: 100 proofs, 9 positives per tracked stratum + 28
  random, pruned length ≤ 16, seed 20261101. Labeller: an independent subagent with TAXONOMY.md
  only. The key directory was mode 700, and the labeller was told not to open it.
- **11:25–11:35 am.** `fig1.py`: panel (b) examples chosen among problems the base solves in at
  most 5 / 3,072 attempts, by shortest proof with no dead lines. Both chosen RL proofs equal
  Charles's reference proofs. Lean re-check: 6 / 6 of the model's own texts accepted.
- **11:34 am.** Spot-check returned. Agreement is 100 / 100 on every uses label, every working
  flag and the primary (`spotcheck.json`). The labeller wrote its own bookkeeping scripts for
  pruning and dependencies and traced the strategy calls by hand. The sample has no LEM positive.
- **11:36 am.** ruff check and format pass. 18 tests pass, including Dan's 39 cases through the
  Lean round trip. The `analyze` rerun after the lint fixes gives a byte-identical `numbers.json`.
- **Deviation (panel c).**
  - The pre-registered selection rule picks MT on Charles149: base B 9.85 % < 10 %, fold in A
    2.05×. But MT's share of proofs does not rise (B-matched −2.1 [−4.7, −0.1]), so it is not an
    emergence curve.
  - On t1320s250 the rule picks working #10 (30×).
  - Panel (c) shows the pre-registered fallback quantity on Charles149, the per-attempt solve rate
    on the 32 base-unsolved problems, and the rule's 13–20-line pick. Both are unit A.
- **11:42 AM. Done.** README, caption.md, numbers.md, and the summary in `experiment-summaries/2026-11-01-proof-strategies-grpo-thinking/` written. Relative links checked. Nothing committed; no tracked file modified. The fork clone is clean.
- **11:43–11:47 am. Figure 1 revised** at the coordinator's request.
  - (b) now leads with a problem no base seed solves in 3,072 attempts: `textbook_49b7…`, 13 lines,
    a reductio with MT, from RL seed 2.
  - Each of the other 8 such problems was checked. 7 are too long (shortest RL proof 15–55 lines,
    shortest with a rare strategy 16–55). d337's 11-line proof uses only case analysis. None hit
    the textbook72 guard.
  - Header rates are now pooled over the three RL seeds and the three base seeds.
  - (c): the legend names the sets; the base is drawn at its own x position left of an axis break,
    replacing the colliding "base" label; the Yue line stops below the legend.
  - Lean re-check: 6 / 6. ruff passes. caption.md, README and the summary are updated.
