---
written_on: 2026-11-11
written_by: agent:claude (orchestrator), for Leon
---

# Two authors: does RL select a whole author, or edit one habit? (build spec)

Leon asked for this on 2026-11-11 ("these habits look interesting launch the experiment use runpod").
It is Experiment 1 of `/scratch/cowifornia/nd-rl/persona_brainstorm/NOTE.md`, with the habits
measured in `../author_tags_20261110/HABITS.md`. This file is the contract for the build. The
pre-registration (`PREREG.md`) is written by the orchestrator once the build shows what can be
measured, before any full-size base exists.

## Question

Pretraining text is written by two authors whose proofs differ in several habits at once. RL then
rewards one of author A's habits. Do A's other habits come along, where, and by how much?

## The two authors

Source corpus: `train_k12dose1.jsonl.gz` (the 155,000 records behind the tag experiment,
`tag_core.SRC_CORPUS`; 1,550 route records flagged `coe_dose`). Every theorem keeps its statement;
only the written form of its proof changes, and Lean must accept every rewritten proof.

| habit | author A | author B | Lean cares | cost |
|---|---|---|---|---|
| route (reductio that does real work, Dan's σ⁺) | has all 1,550 route records | has none | yes: required on the 106-type theorems, so RL rewards it | n/a |
| step order | premise-driven (work from the premises before opening boxes) | goal-driven (open boxes first) | no | 0 lines |
| reiteration | restates outer lines inside a box before using them | cites outer lines directly | no | about +2 lines for A |
| conjunct order | right conjunct first | left conjunct first (the corpus default, 94.5 %) | no | 0 lines |

A carries the costly habit (reiteration) on purpose: if RL toward A's route brings a habit that the
reward mildly discourages, that is stronger evidence for author-level selection than a free habit.

- A proof is normalised to its author's form at every site where the habit has room, using the
  rewrites in `../author_tags_20261110/habits.py`. A record with no room for a habit is left as is.
- A rewritten proof must stay within the corpus limits (K12: at most 12 lines, depth 2, at most 3
  premises, the format's name limits). If normalising a habit would break a limit, that habit is
  skipped for that record and counted in the manifest.
- **Tag.** The author token (`<tagA>` / `<tagB>`, the plumbing of `../author_tags_20261110/`) is on a
  random half of each author's records and absent on the other half.

## Author share (the tie between the route and the author)

The tag experiment found that a route that is 2 % of A's records is not tied to A's tag, and a 10 %
route share is. So the corpus builder takes the same `route_share` option, and two versions are
built:

- `half`: authors drawn by the existing hash, about half the records each (route share of A 2 %);
- `p10`: A has the 1,550 route records plus 13,950 others (route share 10 %), B the rest.

## Control corpus

For each version, a control with the same records and the same overall rate of every habit form,
but each habit assigned independently per record (and the tag independently too), so nothing
bundles: knowing one habit of a proof says nothing about the others or about the route.

## What is measured (every checkpoint, every tag condition A / B / none)

On proofs the model writes and Lean accepts, with detectors that parse the proof (never string
matching alone):

- per habit: sites with room, and the share written in A's form (reiteration per site, not as a
  count of R lines, because step order changes the number of sites);
- route use (σ⁺) and whether the proof is of a theorem that needs it;
- the bundle: within one proof, do the habits agree (all A-form, all B-form, or a mix no author
  writes), reported as shares.

Measurement sets, all drawn from Dan's pool `/scratch/cowifornia/nd-rl/knockout_grpo/dan_data/pool_long_kh.jsonl`
with corpus keys excluded (never a test set): `marker_plain` (theorems that need no reductio and
have room for at least two marker habits), `marker_heldout_class` (the same, restricted to a
theorem class kept out of the RL pool, if one can be defined cheaply; otherwise say so), and
`route_unseen` (theorems that need the route).

## Gate (bases only, before any RL)

1. The base follows the tag: marker shares under tag A against tag B.
2. Without a tag, the habits go together within a proof more than in the control base.
3. Without a tag, proofs that take the route carry A's markers more than proofs that do not, on
   theorems where both happen.

If neither version bundles (2 and 3 fail against the control), stop: the finding is that these
models do not hold an author without a tag.

## RL (only after the gate, and only with Leon's OK on the Runpod spend)

The knockout's primary GRPO recipe (the `dd1` recipe, as `../author_tags_20261110/tag_line.py`
launches it) on untagged prompts, on Dan's RL pool, from the authors base and from the control
base, 3 seeds each, capped at x 1.5. Markers are read at every snapshot under A, B and no tag.

Readings (from the note): markers move inside route proofs and elsewhere = RL shifted the default
author; inside route proofs only = a router; nowhere = RL edited the one habit; a rising share of
combinations no author writes = where the reweighting picture stops describing the policy. In the
control base the markers should not move with the route; RL's own drift on each marker (for
example dropping reiteration because it costs lines) is read from the control lines.

## Rules for the build

- No GPU jobs from build agents: unit tests, CPU smoke runs and `--dry-run` only. CPU work on
  serrano at up to 8 cores (`LEAN_GATE_WORKERS=8`).
- Never read or copy the sealed sets: anything named `textbook72` / `tb72`, `test_13_20`,
  `test_21_30`. The only rows of `transfer_track832.jsonl` that may go to a pod are the 106 strict
  theorems, as `../knockout_danbase_20261104/runpod_copy_line.sh` does (in the knockout worktree).
- Large files go to `/scratch/cowifornia/nd-rl/two_authors/` (called TA). Raw sampled proofs are
  never committed.
- Do not edit `../author_tags_20261110/`; import from it. Do not commit or push; the orchestrator
  does. Notes start with `written_on:` frontmatter and separate what was observed from
  interpretation.
