# Campaign brief: does RL create capabilities, or elicit rare ones?

You are running unattended for up to three days. Nobody answers questions; make the judgment
calls, write them down in `log.md`, keep going. This builds on the finished take-home run in
this repository (read `writeup.md`, including the correction box, `numbers.md` and `log.md`
first — they are the starting point, and the correction box describes a counting bug you must
not repeat). `AGENT_BRIEF.md` was the previous run's brief; its hard rules still apply.

## The question

Both this repository's run and the project's earlier sprint found that RL against the
verifier extends proof length *inside* the generator's family of theorem shapes (frontier 9 vs
8) but transfers almost nothing to textbook theorems (validation-36 `>6` bin 0/24 → 1–2/24).
The open question is whether RL only **amplifies behaviour that is already rare-but-present in
the pretrained model**, or whether it can **compose known rules into patterns that were absent
from pretraining**. Answer it with numbers, in three phases, in this order.

## Where things run

- **VPS (here, no GPU):** code in `~/nd-takehome`, branch **`dan_novelty`** (already checked
  out). Commit and push to `origin dan_novelty` at every milestone (`git push origin
  dan_novelty`). Never push to `main`.
- **Pods:** registered A40 pods, currently `p1` and `p2` (`podls`). Helpers (all take the pod
  name first):
  - `podsync <pod>` — push code (excludes `.git`, `data/`, `artifacts/`, `ckpts/`).
  - `podpush <pod> <relpath>` / `podpull <pod> <relpath>` — move data, checkpoints, results.
  - `podrun <pod> "<command>"` — run in `/workspace/nd-takehome` on that pod. Long jobs:
    `nohup ... > artifacts/<name>.log 2>&1 &`, one job per `podrun` call, then poll the log.
  - `podnew <name>` creates another A40 ($0.49/hr, ~1–5 min to SSH); `podrm <name>` deletes it.
    Use up to **3 pods** at once. Delete a pod when its phase is done and pulled.
- The pods have no data yet. Push what a phase needs (`podpush p1 ckpts/stage1_abs.pt`, the
  `found_*.jsonl` files you need, `data/`).
- **Budget:** $145 of RunPod credit total, prepaid; the balance is the hard cap. Three A40s
  for three days is ~$106, so the budget is not the constraint — use pods, don't idle them.
- **Kill switch:** every pod is deleted at **2026-09-18 16:00 UTC**. Pull results before then;
  aim to be finished and pushed by **2026-09-18 12:00 UTC**.
- **Usage limits:** if Claude Code reports a usage/rate limit, the driver sleeps and resumes
  you. When you resume after one, add the line `LIMIT_HIT <UTC time>` to `STATUS.md` and push,
  so the human sees it. Do not stop work because of it.

## Hard rules (unchanged from the take-home, plus three)

1. Supervised pretraining data: verifier length ≤ 6, procedural generator only, never
   hand-written proofs; assert at load.
2. Splits disjoint by atom-renaming class across all pools; validation-36 classes excluded
   from every pool.
3. Never train on validation-36 or the test files. The test files are **not** part of this
   campaign; do not run them (the marker `artifacts/TEST_RUN_DONE` already exists).
4. `nd_verify` is the only judge; never modify it.
5. **Distinct proofs are distinct after renumbering from `N1`.** Use `normalize.norm()` (or
   equivalent) before any deduplication or count. Every histogram, frontier and count must be
   computed on normalised proofs. Report the number of theorems alongside the number of proofs.
6. **Report a base-model reachability number with every "RL solved X" claim** (Phase 1 defines
   it).
7. **Write `numbers.md` entries with their source file as you go**, not at the end.

## Phase 1 — measure novelty on the existing run (first; ~half a day, one pod)

Existing checkpoints: `ckpts/stage1_abs.pt` (base) and `ckpts/final.pt` (EI round 16). Existing
proofs: `artifacts/ei_abs_s0_cont/found_transfer_16.jsonl`, `found_16.jsonl` (RL targets),
`artifacts/frozen_abs_s0_cont/found_transfer_16.jsonl`, and the validation-36 pass@32 files
`artifacts/ei_abs_s0_cont_r16_val36_k32.jsonl` / `artifacts/stage1_abs_val36_k32.jsonl`.

1. **Teacher-forced log-probabilities.** For every normalised-distinct proof found by the final
   model on transfer, RL targets and validation-36, compute `log p_base(proof | prompt)` under
   `stage1_abs.pt` and `log p_final(proof | prompt)` under `final.pt`. Because `abs` proofs can
   start at any index, compute the base log-prob **marginalised over all start indices the
   tokenizer allows** (sum of probabilities over the shifted copies; teacher forcing is cheap).
   Report the per-proof value and, per theorem, the log of the summed probability over all its
   found proofs (`log p_base(any found proof)`).
2. **Reachability line.** A theorem whose found proofs have total base probability p needs about
   1/p samples to be found by resampling. Plot the distribution of `log p_base` for RL-found
   proofs by written length, with vertical lines at the budgets actually used (512 attempts,
   10⁴, 10⁵). Everything to the right of a line is "rare but reachable at that budget";
   everything to the left is unreachable by sampling at that budget. Give the counts on each
   side, per length bin, for transfer and for validation-36.
3. **Empirical check.** Sample the base model on the transfer set at k = 10,000 per theorem
   (T = 0.8; 1,638 × 10⁴ ≈ 16M samples, a few hours on one A40 with the fixed sampler; batch
   large, write only verified successes plus per-theorem counts to disk) and on validation-36 at
   k = 100,000. Compare the measured per-theorem success rate with the log-prob prediction
   (scatter, log–log). Report which theorems RL solved that the base model did **not** solve in
   10⁴ / 10⁵ samples, with their log-prob estimate. This is the headline table of the phase.
4. **Where the novelty sits.** For proofs the base model finds improbable, compute the
   per-token surprisal profile under the base model and locate the maximum-surprisal line:
   is the improbability concentrated in one step (which rule? which formula?) or spread across
   the proof? Tabulate the rule at the max-surprisal line. Show five examples, pretty-printed,
   with the surprisal marked per line.
5. Write `phase1.md`: the reachability figure, the headline table, the surprisal-locus table,
   and a one-paragraph answer to "was the take-home's frontier gain elicitation or creation, at
   what sampling budget does the distinction hold, and where in the proofs does novelty sit?"

Also, on a second pod while Phase 1 samples: build `minlen.py`, a **bounded minimal-proof-length
prover** (iterative deepening over the 14 rules with the verifier's box semantics, bound ≤ 8
lines, time limit per theorem). Label validation-36, the transfer set and the RL targets with
`min_lines_ub` where it terminates. Report how many "7–16-line" targets have a ≤ 6-line proof.
Use it for **evaluation and labelling only** — never to generate training data.

## Phase 2 — controlled-coverage pretraining (the core; two–three pods, ~1.5 days)

**Patterns** (define each as a predicate on the dependency-pruned proof; write
`patterns.py` with tests against hand-constructed verifier-valid examples):
- **P1 `derived-ORE`**: an `ORE` whose disjunction line is not a premise (`PR`) line.
- **P2 `reductio`**: a `NEGI` box whose hypothesis is `( ~ G )` followed by `DN` yielding `G`
  (classical proof by contradiction of a non-negated goal).
- **P3 `depth-3`**: maximum box depth ≥ 3.

**Dial.** Do not change the generator's probabilities. Generate one large raw cap-6 pool
(≥ 400k verified proofs, strict-mode style where it matters), label every proof with the three
predicates, then **assemble pretraining sets of fixed size (155k) with pattern-P frequency
exactly f** for f ∈ {0, 10⁻⁴, 10⁻³, 10⁻², 10⁻¹} by subsampling, holding the other two patterns'
frequencies at their f = 0 baseline levels as closely as possible (report the achieved
frequencies of all three patterns for every set). f = 0 means **zero** proofs containing the
pattern; verify by re-classifying the written file. Keep held-out sets disjoint by class as
before.

**Targets.** For each pattern, an RL-target pool (≥ 1,000) and a transfer pool (≥ 500) of
7–12-line theorems whose generating proofs **use the pattern**, plus one shared pattern-free
pool as a control. Disjoint by class from everything. Label with `minlen.py` where it
terminates, and report how many targets provably need the pattern's rule (e.g. P2 targets that
are classically-only provable).

**Runs.** Stage-1 identical to the take-home (4L/d256, 6k steps, `abs` tokenizer with offset),
then identical expert iteration (k = 32, 8 rounds, retain 20k) against the pattern's targets,
plus the frozen control with equal attempts. Priority order (do all seeds of a row before the
next row):
1. f ∈ {0, 10⁻¹} × three patterns × 2 seeds (the extremes; 12 Stage-1 + 12 EI + 6 frozen).
2. f = 10⁻³ × three patterns × 2 seeds.
3. f ∈ {10⁻⁴, 10⁻²} × three patterns × 1 seed.
Stage-1 is ~5 min; an EI arm ~1.5 h on a shared pod. Run Stage-1 for a whole row first, then
the EI arms three at a time per pod.

**Metrics per arm and round** (all normalised-distinct):
- pattern **acquisition**: fraction of target theorems solved by a written proof that
  **contains the pattern** (classify the model's proof, not the generating proof), and the
  count of distinct pattern-containing proofs;
- plain solve rate and greedy on targets and transfer; held-out greedy (in-distribution);
- for f = 0 arms: base-model pass@10⁴ on the pattern targets (Phase-1 method) — if the base
  model produces the pattern at all, say how often;
- for every f = 0 pattern proof RL finds: its base log-prob and surprisal locus (Phase-1 code).

**Figure:** acquisition after 8 rounds vs f (log x-axis, f = 0 plotted at the left edge), one
line per pattern, seed points shown, frozen control as dashed. **This figure is the campaign's
main result.** Interpretation guide: acquisition ≈ 0 at f = 0 and rising with f means RL
amplifies rare-but-present behaviour and cannot cross f = 0 at this budget; acquisition > 0 at
f = 0 with the base model's pass@10⁴ ≈ 0 means RL composed an absent pattern — report exactly
how many proofs/theorems, their base log-probs, and at which round they first appeared.

## Phase 3 — textbook-shaped curriculum (if Phase 2 is done by 2026-09-17 12:00 UTC)

Build a **textbook-shaped target pool**: instantiate classical schemata (De Morgan both
directions, distribution, contraposition and converse, Peirce, disjunctive/hypothetical
syllogism, export/import, double-negation forms, explosion) with fresh atom assignments and
random sub-formulas, disjoint from validation-36 by renaming class; ≥ 500 theorems. Two arms
from the standard Stage-1 model: expert iteration on this pool alone, and expert iteration with
a **precursor injection** — a generator-made, verified, ≤ 6-line supervised set of the *short
sub-steps* of those schemata (the "precursor" idea from the project's earlier sprint), added to
the retained slice. Report validation-36 by bin per round with a frozen control, and the base
reachability of every newly solved validation theorem. No test-set run.

## Deliverables (on branch `dan_novelty`)

- `phase1.md`, `phase2.md`, `phase3.md` (if run) with figures in `figures/`; a short
  `campaign.md` that states the answer to the question in ≤ 300 words plus the three key
  figures, written last.
- `numbers.md` (append a campaign section), `log.md` (append, dated), `STATUS.md` (update at
  every milestone; `DONE <UTC time>` as the last line when finished).
- Code: `novelty.py` (log-probs, marginalisation, surprisal loci), `coverage.py` (large-k
  sampling with success-only logging), `minlen.py`, `patterns.py`, `make_coverage_sets.py`,
  and the run scripts. Tests for `patterns.py` and `minlen.py` against verifier-checked cases.
- Pull every result file back to the VPS before deleting a pod.

## Working style

- Before each phase, write the plan and the *expected* result into `log.md`; afterwards, write
  what happened. Negative results are results.
- Look at real proofs, not just aggregates: every table of counts gets five printed examples.
- When something is ambiguous, pick the interpretation that makes the RL result *harder* to
  claim, and note the choice.
- Do not spend on anything beyond pods. Never touch `~/nd-rl`.
