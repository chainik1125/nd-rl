# Run brief: evidence-atlas — one harmonized map of what the project has measured, with figures

Run id `evidence-atlas`. Repository `~/work/evidence-atlas` (branch `dan_evidence-atlas` from the fork's `origin/dan`, made
by `newrun`). Role: executor. `AGENT_POLICY.md` governs. Pod budget **$5** / 10 h, used only if a re-score fills a key gap
(see below). Analysis runs on CPU. Claude time is expected to be heavy. Authorised by Dan 2026-10-02 (proposal 21).

## Why

Dan wants "deeper analysis & graphical depictions of experiments I & others on the project have ran". Results now
live in about 70 experiment summaries, several protocols, and other members' branches. Nobody can see the whole
picture on one page.

## Sources (read-only)

- nd-rl `dan`: `experiment-summaries/`, `docs/STATE.md`, `docs/digests/`, `docs/project_strategy/`.
- The fork's run branches and the bucket `hf://buckets/dan-pandori/nd-rl/` for raw per-theorem files.
- **Other members' work on nd-rl branches**, read-only, never pushed to or commented on:
  - `robbie-experiments`, `autoresearch-*`, `combined-pipeline`, `robbie_pretrain_recipe`;
  - `leon/*`;
  - `dmitry/*`;
  - `charles/main`;
  - nd-rl `main`.

  Describe their results neutrally and with sources. This is a map of evidence, not an assessment of people.

## What to build

1. **A map of experiments by question.** Create vs elicit; data and caps; format; pretraining recipe; RL algorithm;
   interface (whole proof vs proof state); search and exploration; infrastructure. One row per experiment, with
   model, protocol, headline number, review status and link.
2. **Harmonized figures,** recomputed from stored per-run or per-theorem artefacts where they exist:
   - textbook72 pass@k for every model family ever scored (ours and the factorial cells in
     `robbie-experiments`), on one axis;
   - Robbie's dev metric and holdout250 across families;
   - the length frontier (`L*` and long-pool solves) across interfaces and caps;
   - the create-vs-elicit evidence table (support expansion, survivors, trajectory groups, rl-from-ckpt), with a
     strength rating and caveats.
3. **A protocol-differences table.** Checker (Lean alone vs Lean ∧ `nd_verify`), k, temperature, pools, judge
   versions, and seeds. Flag which comparisons are apples to apples and which are not.
4. **Open questions** that the map makes visible.

**Optional re-score (≤ $5):** if one cheap read-out would make a key cross-family comparison apples to apples (for
example, a checkpoint scored on a common pool it was never scored on), do it with the existing read-out code and say
why. Otherwise, use no pods.

## Deliverables

- `atlas/ATLAS.md` (≤ 3,000 words plus figures and tables), `atlas/figures/*.png`, and `atlas/data/*.csv` (the
  harmonized tables). The librarian copies them to nd-rl `docs/atlas/2026-10-02/` and marks them as a draft for Dan.
- `run_evidence_atlas.md` (≤ 200 words), `log.md`, `STATUS.md` line `EVIDENCE-ATLAS DONE <UTC>`, and
  `touch ~/runs/evidence-atlas/executor.done`.

Do not contact anyone. Do not push to or comment on others' branches. Never run `test_run_once.sh`. At most 4 concurrent
processes on the VPS.
