---
written_on: 2026-11-27
written_by: agent:claude (orchestrator), for Leon
---

# Linked habits, version 2: 24 linked habits, preference tuning instead of a reward bonus (build spec)

This replaces the design of `SPEC.md` where the two differ. `SPEC.md` stays as the record of version 1 (no base of
version 1 was ever trained; its three corpora `l19`, `l4`, `l0` exist under LH/data and are kept, unused).

Leon, 2026-11-27, after the version-1 pitch:
- 12:35 am PT: "im skeptical this feels kinda weird cause its ood"; 12:45 am: "can you think of any less procedurally
  generated behaviors".
- about 12:50 am, on the nine natural habits: "yeah ok lets do this we probably need more than just 9 so lets add the
  others your originally proposed and make syure we do a control to make sure this doesnt degrade performance by
  taking up some of the output lines and also some way of making sure it doesnt immeditly disappear in training".
- about 12:58 am: "im also unsure about mearly upweighting it in rl that seems unprincipled how do people really do
  this like mid training right so maybe we can do this sort of rlhf".

Paths as in `SPEC.md`: TA, LH, W2, PY, Dan's code. HB = `../author_tags_20261110/habits.py`.

## The 24 linked habits

Nine natural habits, each a choice between two forms that Lean accepts:

| # | habit | form 1 | form 2 | built |
|---|---|---|---|---|
| 1 | contradiction route | uses it (author A only, as before) | never | W2 |
| 2 | step order | premise-driven | goal-driven | W2 |
| 3 | reiteration | restates outer lines inside a box | cites them directly | W2 |
| 4 | conjunct order | right conjunct first | left first | W2 |
| 5 | case split | keeps a case split whose result one branch alone gives | skips it | HB row 9 |
| 6 | ex falso style | vacuous reductio wrapper | explosion (BOTE) | HB row 8 |
| 7 | disjunct | proves the right disjunct when either is citable | the left | HB row 5a |
| 8 | both halves | takes both halves of a conjunction when one is needed | only the needed half | new |
| 9 | restate before use | restates a top-level line (R) just before using it | cites it | new (top-level R exists in K12) |

Fifteen premise-note habits (10 to 24), exactly as in `SPEC.md` and `notes.py`, table `LH/data/note_table.json`.

**Which author gets which form.** Habits 1 to 4 keep the Nov assignment (A = form 1). For habits 5 to 9 the builder
assigns form 1 to A or to B so that the two authors' proofs have the same mean length within 0.3 lines over the
corpus (reiteration already costs A about 2 lines, so B should carry the long forms of 5, 6 or 8 as needed). The
assignment is fixed once, with the measured mean lengths, in `LH/data/habit_table.json`. "A-form" of a habit always
means the form author A writes.

**Order of rewrites and limits.** Habits are applied in a fixed order with a room test before each; a habit that
would break a limit (more than 16 lines after notes, depth above 2, names, action or context length) is skipped for
that record and counted. Report room (% of records) and skips per habit. A new habit with room in under 3 % of
records is reported and kept only if its rewrite validates; nothing is dropped silently.

## Four corpora (same theorems)

| corpus | linked to the author | independent (per record and habit: a seeded coin at the corpus rate) |
|---|---|---|
| `c24` | all 24 | none |
| `c9` | habits 1–9 | the 15 notes |
| `c4` | habits 1–4 | habits 5–9 and the 15 notes |
| `c0` | none (route records keep the route; its author label is a coin, as in `control_half`) | all |

- Authors, tags (shown on half the records) and the route records as in `authors_half`. The coins of one (record,
  habit) are shared by all corpora that draw that habit independently.
- Every habit's overall form rate is the same in all four corpora (report per habit, per corpus).
- Record fields: `author`, `tag_shown`, `author_tag`, `forms` (habit -> A/B/none for all 24), `realized`, `skipped`,
  `route`, `notes`, `src_proof`, `lh_corpus`.
- Validation as in `SPEC.md` (decompose on every record under its shown tag; `nd_verify`; Lean on a seeded sample of
  at least 3,000 per corpus with at least 100 per (habit, form) where that many exist; detector round trip on every
  record).

## Detector

`lh_profile.profile(prompt, proof)`: for all 24 habits, per proof: sites with room, sites in A-form, sites in B-form.
It must agree with the corpus on every record (round trip) and never crash on a model-written proof (a proof it
cannot read is counted as unreadable). Dead lines: notes are read first and removed, then the natural habits are read
on the rest.

## Twins (minimal pairs) for the simulated rater

`lh_twin.twin(prompt, proof, habit) -> other_proof or None`: the same proof with exactly one habit flipped at all its
sites (A-form <-> B-form), every other habit's reading unchanged (checked with the detector), Lean-accepted. Used on
model-written proofs, so it must fail closed (None) on anything it cannot rewrite.

## Stage 1: bases and the gate

- One base per corpus, seed 0, the `half` recipe with the step count scaled to keep the number of passes over the
  data equal to the Nov 14 `half` bases (largest corpus's factor for all four), line cap raised to fit.
- **Performance control** (Leon): the existing no-notes bases `TA/bases/stage1_{authors,control}_half_s0.pt` are
  measured with the same code on the same sets. Report for every base, untagged: accepted % of attempts, pass@1,
  pass@8, mean lines, and the share of attempts that end at the step limit. Every sampler and judge runs with the
  step limit raised by 3 (48 -> 51), the same for all bases. Also Dan's strict-106 read where W2 already does it.
- Gate (points with 95 % bootstrap intervals over theorems), each base:
  G0 notes are written (at least half of premises noted, untagged).
  G1 tag: A-share under tag A minus under tag B, per habit and pooled.
  G2 an author without a tag: for each pair of habit groups (notes; natural habits), A-share of later sites given the
  first site in the proof is A-form minus given it is B-form; pooled and per habit pair where counts allow.
  The orchestrator applies the decision rule of `PREREG.md`.

## Stage 2: a rater who prefers one habit (replaces the reward bonus)

Rated habits (one arm each): **reiteration** (the Nov 18 habit) and **the `atom` premise note** (largest share of the
RL pool, `LH/data/rewarded_class.json`).

1. **Pairs.** Sample each base untagged on Dan's RL pool (k 8, Lean-accepted proofs only). For each accepted proof
   with a site of the rated habit, build its twin. The A-form member is "preferred". Split by theorem: train pairs
   and 200 held-out pairs. Report counts, and the share of pairs where the sampled member was the A-form one.
2. **Arm DPO (main).** `lh_dpo.py`: DPO on the train pairs against the frozen base as reference:
   loss = -log sigmoid(beta * [(log pi(w) - log pi_ref(w)) - (log pi(l) - log pi_ref(l))]), log-probabilities summed
   over the action tokens of every step of the proof (teacher forced, untagged states, the sampler's masking), beta
   0.1, AdamW, the RL recipe's learning rate, batch 64 pairs. Snapshots when the held-out preference accuracy
   (share of held-out pairs with the preferred member more likely than under the base ... i.e. log-ratio margin above
   0) first passes 60 %, 75 %, 90 %, and at the end (at most 2,000 steps). Logged per step: loss, margin, KL to the
   base on a fixed set of corpus states.
3. **Arm RLHF (second).** A reward model = the base's network with a scalar head on the last state of a finished
   proof, fitted on the train pairs (Bradley–Terry loss; held-out pair accuracy reported, required at least 90 %).
   Then W2's Amendment 3 GRPO recipe (guard off, unlikeliness off, untagged, Dan's RL pool, 60 iterations) with reward
   = Lean acceptance x (1 + 0.3 x sigmoid(reward model)) and the trainer's own anchor `--kl_anchor_coef 0.02
   --kl_anchor_ref <base>` on a canary/1 file of that corpus's untagged states (`--kl_anchor_data`). Snapshots every
   5 iterations to 20, then every 10.
4. **Arm narrow fine-tune (third).** Plain supervised fine-tuning on proofs of RL-pool theorems that have a site of
   the rated habit and no site of any other habit (A-form of the rated habit; the twin of a sampled proof where
   needed), same optimizer; snapshots at the same three preference-accuracy points.
5. **No-preference runs** (drift): DPO with the two members swapped at random (a rater with no preference), and GRPO
   with the anchor and no learned reward.
6. Every arm runs on `c24`, `c9`, `c4`, `c0` for the reiteration rater, and on `c24`, `c0` for the note rater
   (`c9`/`c4` hold the notes unlinked, so they add nothing there).

## Readout

Per base and snapshot, untagged (and under each tag): A-share of all 24 habits (first site of the proof and later
sites separately), sites per proof, accepted %, pass@1, pass@8, mean lines, share at the step limit. Contrasts with
bootstrap intervals over theorems, paired where the theorems are the same: (rated arm − no-preference arm) per
corpus and habit **at matched movement of the rated habit** (the snapshot where its untagged A-share is closest to
+30 points over the base; also the last snapshot), and the differences between corpora (`c24`, `c9`, `c4` each
minus `c0`; `c24` − `c4`; `c9` − `c4`).

## Pods

One RTX 4090 pod per corpus. Stage 1: setup, pretrain, measure the base (and on one pod the two Nov 14 bases), write
a marker, wait for the orchestrator's go file. Stage 2: sample pairs, then the arms (two GPU jobs at a time at most),
each with its measurements; heartbeat; every job under `timeout`; results fetched to LH and md5-checked. The
orchestrator creates and deletes pods.

## Rules for the build

As in `SPEC.md`: CPU only for build agents (at most 8 cores), no pods, no sealed sets (`textbook72` / `tb72`,
`test_13_20`, `test_21_30`; of `transfer_track832.jsonl` only the 106 strict rows), no edits outside this folder and
LH, no commits, ruff-clean code, notes with `written_on:` frontmatter that separate observation from interpretation,
and every deviation reported.
