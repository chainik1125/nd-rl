---
written_on: 2026-10-14
written_by: agent:claude
---

# Long RL: pretrain on ≤ 6 body lines, RL out to 45 on the long-proofs set

Robbie's design (2026-10-14):
- Pretrain on Dan's cap-6 file only, with distractor premises.
- Then extend by RL to proofs never seen in pretraining: 7 to ~45 body lines, 3–10 premises, up to 10 atoms.
- Three arms, same data and budgets.
- Length is **body lines**: lines after the premise block, i.e. nd_verify's `n_lines` minus the premise count.
- Lean is the only checker.

| arm | `ND_ARCH` | `ND_RL` | what |
| --- | --- | --- | --- |
| A | untied | ei | run 198's untied six blocks + the final expert-iteration recipe (507 + 509 + 510) |
| B | looped | ei | run 512's looped trunk (two alternating shared core pairs) + the same recipe |
| C | untied | grpo | run 198's untied blocks + GRPO with a found-proof pool |

## Files

| file | what |
| --- | --- |
| `harness.py` | from the loops harness (`2eec1b61666a`). Lean-only judging, body-line bands, new pools, old dev for forgetting, a rollout oracle for GRPO, per-target sampling caps, KV-sized batches |
| `pretrain.py` | run 512's file (`ae79d23`) plus: the untied switch (control 590's patch), random atom relabelling in `gpu_batch`, frontier-weighted `pick_targets`, `rl_grpo` |
| `overlay/lean_tok.py` | the fork's tokenizer with ten atoms, h1..h10, n1..n160, and a header declaring all ten atoms |
| `overlay/lean_gate.py` | the fork's gate with nd_verify removed: a sample counts iff Lean accepts its literal text |
| `make_data.py` | the pretraining file (distractors) and the RL / dev / holdout pools; every proof Lean-checked |
| `selftest_tok.py`, `test_paths.py` | tokenizer round trip + Lean on real records; gold proofs through gate, judge and both RL updates |
| `launch.py`, `remote.sh` | one A40 per (arm, seed); checkpoints to HF `runs/long-rl-20261014-<RUN>_<arch>-<rl>_s<seed>`; set `RUN` in `launch.py` before a new run |
| `base/` | the unmodified sources these were forked from |

`data/` is not in Git (49 MB). Download it from Hugging Face, `data/long-rl-20261014/` in
`robbiethompson2018/nd-rl-checkpoints` (commands in [DATASETS.md](../../../../docs/DATASETS.md#using-robbies-data)),
or rebuild it with `make_data.py pretrain data/pretrain.jsonl.gz` and then `make_data.py pools data/`.

## Data (built 2026-10-14)

**Pretraining:** 154,990 records, all accepted by Lean.
- Each record's total premise count is drawn uniformly from [its own, 10]. Distractors are other records' formulas,
  with their atoms mapped at random into the ten atoms.
- A distractor is rejected if it would make the premises inconsistent or open a shorter proof (`certify_fast`).
  151 records fell back to no distractors.
- Premises by count: 0: 2,339, 1: 11,075, 2: 32,281, 3: 16,206, then about 13.2k each for 4–10.

**RL / dev / holdout** (the pools runs v2–v5 used; v1's are below):
- From `../long_proofs_20261013/production/v2`: body ≥ 7, plus a short bridge of theorems under 7 body lines with
  3+ premises (the new family's structure at pretraining lengths). Approach E's 35–80-line proofs come in with their
  signature lower bound (`certifier_scoped/results/e_raw_sig.jsonl`).
- Deduplicated by canonical key; theorems in the group's files or in the pretraining set are dropped. Every proof is
  Lean-checked.
- Split by key hash (dev 1/20, holdout 1/20), capped per band.
- The RL pool also gets Dan's ladder RL targets (fork `data/ladder/rl_targets.jsonl`, statements only, 0–3 premises):
  the old harness's targets, which a cap-6 model solves some of from the first round.

| pool | 0–6 | 7–9 | 10–14 | 15–19 | 20–24 | 25–34 | 35–49 | 50+ | total | of which ladder |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rl | 1,790 | 3,537 | 2,699 | 2,500 | 2,500 | 2,500 | 963 | 1,950 | 18,439 | 4,493 |
| dev | 11 | 61 | 150 | 150 | 150 | 150 | 44 | 127 | 843 | 0 |
| holdout | 17 | 44 | 150 | 150 | 150 | 150 | 65 | 97 | 823 | 0 |

v1's pools had body ≥ 7 only, no bridge or ladder targets: rl 11,056, dev 677, holdout 665. Nothing in them was
solved (runs table below), which is why v2 added the bridge and the ladder targets.

Body is the exact minimum where known. For certified-≥20 records without an exact label it is the written length;
their minimum lies in [20, written], and 93% of decided cases were tight.

## Budgets

- Pretraining: 600 s.
- RL: 3 h of GPU clock. Lean-gate time is not charged to it.
- Eval:
  - dev at 64 samples;
  - old transfer dev at 16;
  - for the looped arm, both at 8 loops, plus dev again at 2 loops.
- Total limit: 6.5 h.

## Checks run

- `selftest_tok.py` on 900 records (Dan's file, short, certified ≥ 20, E's 35–80-line proofs): 0 round-trip
  mismatches, Lean accepts 900/900. Max 88 names; up to 2,092 tokens.
- `test_paths.py`:
  - 12/12 gold dev proofs accepted by the Lean-only gate, with matching body lengths;
  - a corrupted proof rejected;
  - the GRPO and EI updates run.
- CPU smoke runs of all three arms, `harness.py --smoke`: status ok.

A bug the tests caught: fork modules put the fork directory first on `sys.path` (nd2lean does it from inside
`lean_tok`), so the fork's Lean-AND-nd_verify gate was loaded instead of the overlay. The harness now loads both
overlay modules by file path and asserts it at startup.

## Untuned choices to revisit

- **GRPO:** v3 rates (Muon 3e-4, AdamW 3e-5), retained-pretraining SFT weight 0.5, constant length normaliser 256.
- **Target picking:** a band's weight is 0.1 plus its solved fraction, up to one band past the frontier, and 0.02
  beyond that.
- **Pretraining budget:** doubled to 600 s, because distractors make records about 1.6× longer.

## Runs

| run | launched (PDT) | what | outcome |
| --- | --- | --- | --- |
| v1 | 09:05 | all three arms, seed 0, long theorems only | nothing solved: 0/677 dev, every RL round all-fail |
| v2 | 10:17 | all three arms × seeds 0, 1; pool adds Dan's ladder targets and a short bridge | EI learns, results below. GRPO collapsed and was stopped at 12:42 |
| v3 | 12:50 | GRPO only, seeds 0, 1, with the fixes below | stable but far behind EI: dev 73 / 48, results below |
| v4 | 14:05 | untied EI and v3's GRPO × seeds 0, 1, with per-round KL probes and checkpoints every 5 rounds | EI dev 246 / 186; GRPO dev 35 / 68 |
| v5 | 15:40 | v2's four EI runs (untied and looped) resumed for 4 more RL hours (about 7 h in all), same pools | dev 313 / 433 untied, 376 / 372 looped (3 h: 232–247) |

**Resuming (v5).** `harness.py --resume <dir> --rl-hours H` takes a finished run's `final.pt`, `pretrain.pt`,
`found_targets.jsonl` and `summary.json` (staged under `resume/<arm>_s<seed>/`, uploaded by `launch.py`). It skips
pretraining, loads the final weights, seeds the found-proof pool, continues oracle call numbering (so sampling seeds
continue), and keeps the KL probes' reference at the original pretrained weights. Expert iteration builds fresh
optimizers with their own warmup and cosine each round, so a resumed run is the same algorithm as one longer run;
only the random streams differ. GRPO's optimizer state lives across rounds and is not saved, so a resumed GRPO run
would restart its moments.

v2's GRPO used AdamW 1e-4 on every parameter, no ratio, and scored the loss at T = 1 while sampling at 0.8. One
round of updates could take rollout accuracy from 0.10 to 0.02. Unparseable samples rose to 100% and entropy
spiked to 0.9. v3 changes only the update (Robbie, 2026-10-14):

- Muon on the matrices and AdamW elsewhere, as in EI's fine-tune, at a tenth of its rates (Muon 3e-4, AdamW 3e-5).
- PPO clipped ratio against the sampling policy, asymmetric [0.8, 1.28], and a dual clip at 3 for negative
  advantages. These are Leon's settings from `origin/leon/grpo-v2`.
- Loss and ratio at logits / 0.8, the sampling distribution. The harness asserts `GRPO_TEMP == TEMP`.
- Logged per round: clip fraction, mean ratio, and entropy and −log p at T = 0.8. These two are equal in
  expectation when the sampler and trainer agree. v2's collapsed rounds had −log p above entropy even at T = 1,
  which sharpened sampling cannot produce.

`test_paths.py` checks the surrogate on hand cases. With zero learning rates it checks that every ratio is exactly
1 and the weights don't move. It also checks the T = 0.8 stats against a per-rollout reference.

Report page: `report/index.html`, built from `report/export.py <out.json> <log dirs>`, published as a private
claude.ai artifact.

## v2 expert iteration results (finished 14:05–14:59 PDT)

Dev is 843 theorems at 64 samples; old dev is the loops harness's transfer half (1,108) at 16 samples. L* is the
longest body length with at least 5 solved.

| arm | seed | dev solved | dev L* | 7–9 | 10–14 | 15–19 | 20–24 | 25–34 | 35+ | old dev | RL targets solved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| untied-ei | 0 | 232 | 25 | 59 | 105 | 34 | 17 | 6 | 0 | 852 | 7,020 |
| untied-ei | 1 | 247 | 26 | 59 | 110 | 37 | 22 | 8 | 0 | 929 | 7,395 |
| looped-ei (8 loops) | 0 | 246 | 26 | 59 | 101 | 39 | 26 | 10 | 0 | 922 | 6,437 |
| looped-ei (8 loops) | 1 | 246 | 26 | 61 | 101 | 39 | 26 | 8 | 0 | 929 | 6,430 |
| looped-ei (2 loops) | 0 | 241 | 26 | | | | | | | | |
| looped-ei (2 loops) | 1 | 236 | 23 | | | | | | | | |

## v3 and v4 results (finished 17:30–19:08 PDT)

| run | arm | seed | dev solved | dev L* | 0–6 | 7–9 | 10–14 | 15–19 | 20–24 | 25–34 | old dev | RL targets solved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v3 | untied-grpo | 0 | 73 | 10 | 11 | 47 | 15 | 0 | 0 | 0 | 593 | 4,545 |
| v3 | untied-grpo | 1 | 48 | 10 | 11 | 31 | 6 | 0 | 0 | 0 | 287 | 2,456 |
| v4 | untied-ei | 0 | 246 | 26 | 11 | 59 | 108 | 36 | 23 | 9 | 925 | 7,242 |
| v4 | untied-ei | 1 | 186 | 18 | 11 | 59 | 91 | 23 | 2 | 0 | 859 | 6,631 |
| v4 | untied-grpo | 0 | 35 | 9 | 9 | 24 | 2 | 0 | 0 | 0 | 276 | 2,330 |
| v4 | untied-grpo | 1 | 68 | 12 | 8 | 40 | 20 | 0 | 0 | 0 | 272 | 2,598 |

v4's EI is v2's recipe plus probes, yet seed 1 lands at 186 against v2's 247: the seed-to-seed spread of one 3 h EI
run is about ±30 dev theorems. v3 s0's solved count jumped from about 2,045 to 3,816 over oracle calls 84–102: it
learned depth-3 nested assumption boxes (ladder targets with a 3-deep implication goal: 15% → 97%; s1 never did, 13%).

- Pretrained on proofs of at most 6 body lines over at most 4 atoms; 3 h of EI reaches 25–26-line dev theorems and
  proofs using all 10 atoms (first 10-atom proof at 0.9–1.7 h of RL).
- Looped and untied tie on dev (246, 246 vs 232, 247). Eight loops beat two by 5 and 10 theorems: the same small
  test-time gain as the loops workstream.
- Nothing at 35+ body lines is solved on dev.
- Before RL (run v1's pretrained models), old dev was 50 (untied) and 23 (looped).

Checkpoints: HF `runs/long-rl-20261014-v2_<arm>_s<seed>` (pretrain.pt, final.pt).

## v5 results: v2's EI runs resumed, about 7 h of RL in all (finished 20:20–21:12 PDT)

Same columns as v2. The 3 h numbers are v2's rows above.

| arm | seed | dev solved | dev L* | 0–6 | 7–9 | 10–14 | 15–19 | 20–24 | 25–34 | 35+ | old dev | RL targets solved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| untied-ei | 0 | 313 (3 h: 232) | 26 | 11 | 61 | 123 | 74 | 31 | 13 | 0 | 995 | 8,961 |
| untied-ei | 1 | 433 (3 h: 247) | 27 | 11 | 61 | 145 | 108 | 71 | 37 | 0 | 1,020 | 9,539 |
| looped-ei (8 loops) | 0 | 376 (3 h: 246) | 27 | 11 | 61 | 122 | 116 | 46 | 20 | 0 | 1,011 | 9,173 |
| looped-ei (8 loops) | 1 | 372 (3 h: 246) | 26 | 11 | 61 | 125 | 115 | 42 | 18 | 0 | 993 | 9,176 |
| looped-ei (2 loops) | 0 | 357 (3 h: 241) | 27 | | | | | | | | | |
| looped-ei (2 loops) | 1 | 358 (3 h: 236) | 26 | | | | | | | | | |

- Four more RL hours add 81–186 dev theorems, nearly all at 10–34 body lines. Nothing at 35+ is solved, and L* moves
  by at most one line.
- The untied seeds split far apart (313 and 433; at 3 h they were 232 and 247). The looped seeds agree (376, 372).
  Why untied s1 pulled ahead is not checked: compare the two runs' found proofs by shape.
- Eight loops beat two by 14–19 theorems, up from 5–10 at 3 h.
