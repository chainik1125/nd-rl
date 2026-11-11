---
written_on: 2026-11-10
written_by: agent:claude (build agent), for Leon
---

# Author tags: the route taught only under tag A, RL under tag B

**Question (Leon).** "Teach the route only under author tag A, run RL under tag B." Dan's 1 % dose corpus is
pretrained with one author tag per record, where every proof that uses the route (proof by contradiction where it does
real work, σ) is tagged A. Then GRPO runs with every prompt tagged B, on Dan's problems that need the route. Two
questions follow.

- Does the B line behave like a 1 % base (it crossed Dan's bar at about 0.35 to 0.39 times the Stage-1 compute)?
  That would mean the capability is shared across tags and RL reaches it.
- Or does it behave like a 0 % base (no crossing by about 5.9)? That would mean the capability is tied to its tag.

A second question is about the base itself: does the tag gate propensity or capability? Propensity is how often the
route is taken unprompted. Capability is how often the model finishes when the route's opening is forced.

This note covers the build, the tests, the launch commands and a **draft** pre-registration. Nothing has run on a GPU,
and there are no results yet.

- Code: this folder, branch `leon/author-tags` (cut from `leon/knockout-grpo` at d78d12a), uncommitted.
- Artifacts: `/scratch/cowifornia/nd-rl/author_tags/` (AT below).
- Python: `/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python` (PY).

## What was built

| file | what |
|---|---|
| `tag_core.py` | the tag tokenizer (`lean_staten_tag`), where the tag goes, the installs into Dan's tree and Leon's snapshot, the runtime checks ("spies"), the corpus rule, the lines |
| `tagcorpus.py` | builds the tagged corpus and its manifest |
| `tag_pretrain.py` | Dan's `state_train.py --recipe best` (13e41bbd), wrapped: tagged states, a fixed step count, D recorded |
| `tag_sample.py` | samples a checkpoint under A, B and no tag through Dan's sampler and the Lean gate (training-corpus theorems) |
| `tag_grpo.py` | Leon's `code.thinking.grpo` (snapshot `KD/snap/f35d3daf_danbase`), wrapped: one tag on every prompt |
| `tag_danread.py` | the knockout's `danread.py` (Dan's held-out read), unchanged, under one tag |
| `tag_line.py` | one RL line on one GPU: GRPO, the online reads, the stop rule (modelled on the knockout's `dline.py`) |
| `tag_measure.py` | Dan's measure and the stop rule for a tagged line (adapted from the knockout's `merge.py`) |
| `tag_read.py` | the propensity/capability read (`pc_run.py` arms U, F, E, imported) under A, B and no tag |
| `tag_check.py` | the manipulation check: teacher-forced log P of route and non-route actions under each tag |
| `tag_launch.py` | the gated launcher: GPU checks, lab census, lock, `timeout`, progress log, watchdog, `--dry-run` |
| `tag_smoke.sh` | the CPU smoke of every path |
| `test_tagcorpus.py`, `test_tag_core.py` | tests (below) |

Nothing in Dan's tree, Leon's snapshot, the knockout folder or the propensity folder was edited. Every change is a
wrapper or a run-time install in our own processes, so there is no patched copy and no diff to record.

## The tag format

**Tokens and vocabulary.** There are two new tokens, `<tagA>` and `<tagB>`. They are inserted before `h1` in Dan's
`lean_staten` vocabulary, as Leon's `lean_thinkn` inserts its think tokens. The new mode is `lean_staten_tag`, with
113 tokens:

| ids | tokens |
|---|---|
| 0–38 | unchanged |
| 39 | `<tagA>` |
| 40 | `<tagB>` |
| 41–48 | h1–h8 |
| 49–112 | n1–n64 (ref0 = 49) |

Every id at or above ref0 is a name, so no name shift ever moves a tag (`shift_pair`, `state_train_best.make_batch`,
the canary's base). A checkpoint stores only `tok_mode`, so that mode is what makes every loader rebuild this
tokenizer. A loader without the install fails loudly, because `LeanTokenizer` rejects the mode.

**Placement.** There is one token in every state, immediately before `<act>`. The models use ALiBi, a linear
recency bias, and the first action token is predicted at `<act>`. So the tag sits one position from where the route
is chosen, and every head can see it. Actions never carry a tag, so the loss targets are unchanged.

**Example state.** It comes from the training corpus, dose record `( ( ~ ( ( ( P v P ) v P ) > ( Q v R ) ) ) > F ) |-
...`, tag A, step 1. The action after it opens the reductio box:

```
<st> h1 : ( ( ¬ ( ( ( P ∨ P ) ∨ P ) → ( R ∨ S ) ) ) → False ) <nl> n1 : ( ( ¬ ( ( ( P ∨ P ) ∨ P ) → ( R ∨ S ) ) ) → False ) <nl> ⊢ ( ( ( P ∨ P ) ∨ P ) → ( R ∨ S ) ) <tagA> <act>
action: have n2 : ( ¬ ( ¬ ( ( ( P ∨ P ) ∨ P ) → ( R ∨ S ) ) ) ) := ( fun ( n3 : ( ¬ ( ( ( P ∨ P ) ∨ P ) → ( R ∨ S ) ) ) ) => by
```

The same state is `... <tagB> <act>` under B. With no tag it is `... ) <act>`, which is Dan's state token for token.

**Where the tag is added: `TagTokenizer.encode_toks`.** Every state in every code path we run reaches the model
through `tok.encode_toks(state tokens)`:

- the Stage-1 pairs (`state_train.load`);
- Dan's sampler (`state_sample.prompt_ids`);
- the reads (`factor_lib.run_env`, `pc_lib.first_move_rows`);
- GRPO's sampler (`sample.think_generate` via `stateview.render`);
- the canary (`canary.row_seqs`).

A state is the only token list that ends in `<act>`. The tokenizer inserts the tag there, and it raises `TagError`
for a list that holds `<act>` anywhere else or already holds a tag. The theorem prompt is never touched, so Lean, the
judge and every dedup by prompt see exactly what they saw before.

**Which tag.** `tok.author_tag` is used if it is set. Stage-1 sets it per record and resets it after loading.
Otherwise the process's `AUTHOR_TAG` (`A`, `B` or `none`) is used, which the line's read subprocesses inherit. When
neither is set, the first state raises `TagError`. So no state of a tagged model is encoded without a declared tag.

**Runtime checks.** `tag_core.spy_dan` and `spy_thinking` wrap every sampler's decode call. Every decoder input must
end in the run's tag followed by `<act>`, or carry no tag for `none`. Otherwise the run stops with `TagError`. The
spies also check the canary's teacher-forced inputs in GRPO. The counts are written to:

- `author_tag.json`, in each GRPO segment;
- `tag_spy`, in each read's JSON;
- `tag_states_checked`, in each propensity-read arm's stats.

The propensity read's E arm scores rows directly, with no sampler, so the spies do not see it (its count is 0). Its
states come from `prompt_ids`, which the tests cover. `tag_check.py` asserts every row's tag before it scores.

## The corpus (`tagcorpus.py`)

**Source.** `knockout_grpo/dan_data/train_k12dose1.jsonl.gz` (sha256 `7d74e50e…`, md5 `7433f6ac…`, 155,000 records).
It is Dan's `coe/coe_dose.py --share 0.01` output, read on his branch `origin/dan_create-or-elicit`:

- `coe_dose: true` marks the 1,550 strict-σ records of K12 put back into K12⁻. These are the dose proofs.
- `coe_refill: true` marks the knockout's σ⁺-free refill.

**Rule.** Every dose record is A. Every other record is A when the first byte of `sha256("20261110:<key>")` is even,
and B when it is odd. Each output line is the source line with `, "author_tag": "A"` (or B) appended inside the
object. Nothing else changes, byte for byte, and the line order is kept. The gzip is deterministic (mtime 0, no file
name).

**Output.** `AT/data/train_k12dose1_tagAB.jsonl.gz`, sha256 `45b222c5051973f3928536a0848909406882bb6ff7ef0a46bdb0b2ec509c2485`,
with its manifest `AT/data/train_k12dose1_tagAB.manifest.json`.

| | A | B |
|---|---|---|
| records | 78,188 | 76,812 |
| dose records | 1,550 | 0 |
| refill records | 7,655 | 7,756 |
| σ (Dan's `coe_sigma.has_sigma`) | 1,550 (1.98 % of A) | 0 |
| σ⁺ (Dan's `coe_assemble.knock`, the knockout's drop rule) | 1,550 | 0 |

- **Hash split.** Of the 153,450 hash-drawn records, 76,812 are B: 50.06 %, z = 0.44 against one half.
- **Route detector.** In the whole corpus, σ⁺ holds for exactly the 1,550 dose records.
- **Lengths.** Per-length counts per tag are in the manifest. Every length from 2 to 12 has about 7,000 of each tag.

## Pretraining (`tag_pretrain.py`)

The recipe is Dan's `state_train.py --recipe best`, unchanged: 6 × 384 ALiBi, Muon + AdamW, MTP 0.3, token-budget
batches of 128 × 144, warmup 200, cosine to 0.3, `--cap 12`, and his held-out loss on the first 2,000 records of
`data/p2/heldout.jsonl`. The wrapper changes only four things.

1. **The tokenizer.** It is `lean_staten_tag`.
2. **The loader.** The wrapper reads `.jsonl.gz` and encodes each record's states under its own tag. It asserts that
   every state ends in `<tagX> <act>`. Held-out records carry no tag, so they get the corpus rule: σ⁺ → A, otherwise
   the hash.
3. **The schedule length.** It is a fixed `--best_steps`, equal to the steps of Dan's 1 % base of the same seed: 25,607,
   24,028 and 24,286 for seeds 0, 1 and 2, checked against his checkpoints in the tests. His runs used a 1,200 s
   wall-clock budget on an A40, which is hardware-dependent.
4. **The output.** D is recorded, meaning the real tokens of every step, his registry count. It goes into the
   checkpoint's extra (`train_tokens`, `author_tags`) and a sidecar, `AT/bases/stage1_tagAB_s<seed>.tag.json`. RL's x
   is measured in units of this D.

Each base is followed by `tag_sample.py`: 64 short training-corpus theorems × 4 attempts under A, B and none, with Lean.

## RL lines (`tag_line.py`, `tag_grpo.py`, `tag_measure.py`)

**Lines.** `tB_s0–s2` run under tag B and `tA_s0–s2` under tag A, each on the tagged base of the same seed.

**Recipe.** The recipe is the knockout's `common.grpo_argv` for `dd1_s<seed>`, called, not copied. Only `--init`,
`--name` and `--kl_anchor_ref` are swapped for the tagged base. That gives the 1 % lines' recipe exactly:

- the problems are Dan's 4,495 `rl_targets`;
- each iteration samples 256 prompts × 16 at T 1.0, with lr 1.5e-4;
- the canary guard rolls back on an alarm, at most 3 times;
- a model-only snapshot is saved every 10 iterations, and a rolling checkpoint every 5.

**Reads.** Every snapshot is read with Dan's code under the line's tag (`tag_danread.py`: k 64 on the 106, T 0.8,
256 / 48, noise seed seed*1000 + it + 500, batch 128), and so is the base at iteration 0.

**Dan's measure.** It is the union of the strict-106 members solved by the first 32 attempts of the matched draws,
against his bar of 14. The draws follow his `la_dose1_s<seed>` x grid, the dd1 lines' grid, with x in units of the
tagged base's own D. Because the steps are matched, D should be within about 1 % of his.

**Stop.** A line stops at the first of three events:

- x_dan ≥ 1.5 (Leon's cap);
- 7 GPU-h;
- reached and held (the knockout's rule, `merge.held`).

The line process applies the rule itself. There is no supervisor and no respawn. A line that ends early, from a crash
or from another user's process on the GPU (it then ends at its next rolling checkpoint), is continued as the next
segment from its newest rolling checkpoint (`--seg k+1 --resume ...`).

## Base reads (`tag_read.py`, `tag_check.py`)

- **Propensity/capability.** `pc_run.py`'s arms are imported unchanged and run once per tag on each base, with the same
  noise seed for every tag. That makes the three reads paired draws. The arms are U (k 32, unforced), F (k_f 12, the
  first proof action forced to each applicable opening) and E (the exact first-move log-probability). The theorems
  are the pc read's frozen set: the strict 106 and 240 others, used through its loader. Rows keep SPEC.md's schema,
  with ckpt id `tagAB_s<seed>@<tag>`, so pc's `analyze.py` can read them.
- **Manipulation check.** Training-distribution data only. Each action is scored teacher-forced under A, B and none,
  and labelled with `pc_lib.moves`: `open` (the reductio box), `dn`, or `other`. The four sets:
  - the 1,550 dose records;
  - 1,550 non-dose corpus records;
  - 1,550 route proofs from Dan's long generator pool (`pool_long_kh.jsonl`) whose key is not in the corpus, so the
    base never saw them;
  - 1,550 σ⁺-free pool records of the same lengths.

## What was tested, and how

All of this ran on CPU on serrano, with 8 threads, 8 Lean workers and no GPU.

**`test_tagcorpus.py` (12 tests, all pass, 21 s).**

- Every dose record is A, and there are 1,550.
- No B record is a dose record or has σ or σ⁺ by Dan's own `has_sigma` and `knock`.
- σ⁺ holds for exactly the dose records.
- The split is within 3 sd of one half (z 0.44).
- The multisets of `prompt`, `key` and `thm` equal the source's.
- Every line is the source line plus the field, byte for byte.
- The manifest's numbers and hashes match the files, and a rebuild reproduces the corpus.
- Unit tests cover the rule and determinism.

**`test_tag_core.py` (14 tests, all pass, 5 s).**

- The vocabulary layout against `lean_staten`.
- States get the tag and actions do not. Untagged states equal Dan's token for token.
- The error cases fail closed: no tag declared, `<act>` mid-list, state + action in one list, a second tag, a bad
  tag. The per-record tag beats the process tag.
- Name shifts (`shift_pair`, `make_batch`) never move the tag.
- Both installs dispatch the new mode and leave `lean_staten` and `lean_thinkn` alone.
- `DOSE1_STEPS` equals Dan's checkpoints.
- The Stage-1 loader gives each pair its record's tag. With the tag dropped, its pairs equal Dan's own loader's.
- A checkpoint saved by Dan's `save_ckpt` loads in both his loader and Leon's, with equal logits.
- Dan's `env_generate`, `pc_lib.run_env` (forced and unforced) and `first_move_rows` carry each tag.
- GRPO's `think_generate`, in both the plain and the logical arm, carries each tag in every decode wave and in every
  trace step it trains on.
- The canary carries each tag.
- The spy stops a run whose states carry another tag than the one declared.

**CPU smoke (`tag_smoke.sh`).**

- **pretrain.** A 3 × 128 model (527k parameters) trained 800 steps on the first 4,000 tagged records: 16,302 A pairs
  and 15,812 B pairs. Loss went from 4.5 to 0.24, and D = 3,224,332.
- **sample.** 16 theorems × 4 attempts, through Lean:
  - A: 9 of 64 accepted;
  - B: 9 of 64 accepted;
  - none: 5 of 64 accepted.

  That is 229 / 224 / 219 states checked, each with its tag.
- **line, `tB_s0` in a smoke tree.** GRPO under B ran 2 iterations of 4 × 4 on CPU. The canary was under B, with
  15,212 canary and anchor rows checked; a rollback fired, as the toy model's canary alarmed. The run wrote rolling,
  alarm and snapshot checkpoints in the tagged mode, and ended `done`.
  - The base and snapshot reads under B (k 2 on the 106) ran through `tag_danread.py`, with 434 states checked per
    read. `tag_measure.py` built the view: 3 reads, Dan's measure [0].
  - **Compute-cap stop.** With a tiny D (1,000 tokens), the rule check, every 3 s for the smoke, found x_dan 5.43 ≥ 1.5
    after iteration 2. It killed GRPO, let the pending snapshot reads finish, and ended `rule`.
  - **SIGTERM.** A `SIGTERM` to the line (as `timeout` sends) ended its GRPO, which runs in its own session, and was
    recorded as `terminated`. No orphan was left.
  - **Resume.** Segment 2 ran `--resume seg1/ckpts/tB_s0_it1_alarm.pt` and continued from iteration 1. The merged view
    dropped segment 1's iteration 2 (`lost [[2, 2]]`) and kept the reads of both segments.
- **read.** `tag_read.py gpu` then `rows`, on the CPU, with 1 theorem per set, k 2 and k_f 1, under A, B and none.
  - Every arm wrote its raw file, and the spy counted the states of each arm (14 to 20 for U, 8 to 13 for F).
  - The rows pass pc's `check_row` contract.
  - QC found 0 forced-first mismatches, 0 replay mismatches and 0 verifier rejects of Lean-accepted proofs.
- **check.** `tag_check.py`, 20 records per set, ran in 6 s. On this toy model:
  - Δ_open (A − B) on `sigma_unseen` is −0.07 nats, so the toy does not use the tag.
  - The untagged state is 0.6 nats worse on the route opening than either tag.

  That is the failure the check exists to catch, seen on a 527k-parameter model trained on 4,000 records. It predicts
  nothing about the real bases.
- **Loader at scale.** 10,000 corpus records loaded in 5.7 s, so the whole corpus loads in about 1.5 min. That gives
  7.98 pairs per record; Dan's runs had 1,239,995 / 155,000 = 8.00.
- **Launcher.** `tag_launch.py pretrain --gpu 0 --seeds 0,1,2 --dry-run` printed the plan, and the gate passed. The
  census at 03:41Z showed 0 of ours and 8 lab GPUs free. Nothing was launched.
- **Lint.** `ruff check` and `ruff format --check` with the repo's config pass on every file of ours.

## What was not tested

- **No GPU path ran.** Not tested:
  - the CUDA graph or compile of the best recipe on the tag vocabulary;
  - the speed of a full base;
  - memory at 256 × 16;
  - the GPU read's memory cap beside GRPO.
- **No full-size base exists yet.** Whether the bases learn to use the tag is the manipulation check's job, after (a).
- **No launch through the launcher's detached job.** `run_job`'s lock, `timeout` and progress log were exercised only
  by reading the code. The dry-run and gate ran.
- **The lab census reaches ynez and temescal by ssh.** An unreadable host makes the gate refuse unless
  `--allow_unreadable` is given.
- **The move path.** No segment ended at a checkpoint because another user's process appeared on its GPU. That path
  is the knockout's, kept as it was; the resume that follows it was tested.
- **The full-size pretraining loader.** Dan's 155,000 records gave 1,239,995 pairs in his runs. The wrapper's loader
  ran on at most 10,000 records here, and the pair count at full size is checked only when (a) runs.
- **pc's `analyze.py`.** It was not run on our rows. The rows pass pc's `check_row` contract.

## Departures from the brief, and why

1. **Placement.** The brief says "one token in the state" and leaves the place open. The tag goes immediately before
   `<act>`, not at the start, for the ALiBi reason above.
2. **Fixed step count.** Pretraining uses a fixed step count, equal to Dan's 1 % base of the same seed, instead of his
   1,200 s budget. That budget gives about 24,000 steps on an A40 but fewer on a 200 W RTX 3090. The tagged bases
   would then be undertrained relative to the 1 % bases they are compared with.
3. **The held-out set.** Dan's held-out loss set has no tags. It is read under the corpus rule, which affects only the
   validation number, not training.
4. **One line per process.** The RL lines have no supervisor, because a line applies its own stop rule. Respawns are
   manual: the next segment from the newest rolling checkpoint. The knockout's `sup.py` is tied to its own line table
   and pods.
5. **Added: a manipulation check (`tag_check.py`).** The pre-registration needs a test of "the base ignores the tag".
   The pc read's first-move propensity reads reductio poorly: its log says 0 to 0.7 % of attempts open with it,
   because most reductios are on a subgoal. Teacher-forced log P on unseen route proofs is direct and cheap.

## Things in the code that could make the tag unreliable

1. **`stateview` keeps the fork's renderer from import time.** In Leon's snapshot, GRPO's sampler renders states
   through `stateview.render` → `flat_tokens` → `_FLAT`, a reference to `Env.state_tokens` captured at import. A tag
   added by patching `Env.state_tokens`, the obvious place, would have reached Dan's samplers but not GRPO's. GRPO
   would then have trained on untagged states, silently. That is why the tag is added in the tokenizer, which every
   path calls.
2. **The canary's states are precomputed text.** `canary_long.jsonl` has no tags. They get the run's tag at encoding,
   through the same tokenizer hook. Tagged text would be refused.
3. **Formats that glue a state to other tokens hit `TagError`.** Leon's `rootplan.py` (the R format) and the
   think-format pair builders in `data.py` concatenate a state with other tokens before `encode_toks`. They would put
   `<act>` mid-list and raise. Neither is used here.
4. **No code path rebuilds a prompt and drops the tag.** The tag is not in the prompt; it is added at every state
   encoding. The remaining risk is a process with the wrong `AUTHOR_TAG`. `tag_line` sets it from the line table, and
   the spies check every decoder input against the declared tag.
5. **Dan's samplers ban nothing in actions.** That applies to `<st>`, `<act>` and the tags. A sampled tag token in an
   action is an illegal action: the env refuses it, and the attempt fails. The tags are never targets in pretraining.
6. **Dan's `state_train.load` opens plain files only.** The corpus is gzipped, so the wrapper's loader replaces it.
   His `danread.py` loads on `cuda`; only the CPU smoke maps it to the CPU.
7. **`danread.py` changes into Dan's tree.** Its Lean gate log defaults to `artifacts/lean_gate.jsonl` there, and the
   knockout's reads appended to it. Our lines and reads point `LEAN_GATE_LOG` at their own folders.

## Launch commands (the orchestrator runs these; none were run)

From this folder, with `PY=/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python`:

- Run each command first with `--dry-run`. It prints the steps, timeouts, estimates, gate verdict and census, and
  writes nothing.
- Every job checks its GPU and the lab rules in code: at most 4 of ours, at least 2 lab GPUs free, temescal at most 2,
  serrano at 200 W. It then takes `AT/locks/<host>_gpu<N>.lock`.
- Every step runs under `timeout`. Progress goes to `AT/logs/<job>_<UTC>.log`, plus `.status.json`. The watchdog
  logs to `AT/logs/watchdog.log`.
- `$PY tag_launch.py status` and `$PY tag_measure.py` show where things stand.

| | command | estimate | hard cap |
|---|---|---|---|
| (a) pretrain, 3 seeds | `$PY tag_launch.py pretrain --gpu N --seeds 0` (and `--seeds 1`, `--seeds 2` on two more GPUs; or `--seeds 0,1,2` on one GPU, about 1.8 h) | 0.55–0.6 h per seed; 1.7 GPU-h in all | 2 h per seed |
| (b) base reads | `$PY tag_launch.py read --gpu N --seeds 0,1,2 --tags A,B,none` | GPU about 0.75 h (9 reads), then rows on the CPU (0.1 h) and three checks (0.1 h each); about 1.2 h in all | 5 h |
| (c) RL under B, 3 seeds | `$PY tag_launch.py rl --gpu N --line tB_s0` (and `tB_s1`, `tB_s2`) | about 2 h per line if it crosses like the 1 % lines; 3–5 h if it runs to x 1.5 | 8 h per line (and 7 GPU-h in the stop rule) |
| (d) RL under A, 3 seeds | `$PY tag_launch.py rl --gpu N --line tA_s0` (and `tA_s1`, `tA_s2`) | the same | 8 h per line |

**Where the estimates come from.**

- **Pretraining.** 24,000 best-recipe steps took 1,489 s and 1,701 s on serrano and 967 s on ynez
  (`knockout_grpo/logs/base_d1_s*`). Add about 1.5 min of loading (measured above) and 1 min of compilation.
- **Reads.** About 4 min per base (the pc read's log).
- **Lines.**
  - The 1 % lines were reached and held by 1.2–1.9 GPU-h.
  - The lines that never crossed ran 0.30–0.50 x_dan per GPU-h, so reaching x 1.5 takes 3–5 GPU-h.
  - Each read takes about 95 s every 10 iterations.

**Order and limits.** Run (a), then (b), then (c) and (d). The lab rule allows 4 of our GPUs at once, so start
`tB_s0–s2` with `tA_s0` and add `tA_s1` and `tA_s2` as GPUs free up. In all, (a)–(d) come to about 15–33 GPU-h.

## Draft pre-registration (for Leon to edit and freeze before (c) and (d) run)

**Hypotheses.**

- **H-fast ("like 1 %").** Under B the route's capability is shared with A. RL on route-needing problems reaches it at
  the 1 % lines' cost.
- **H-slow ("like 0 %").** The capability is tied to tag A. Under B, RL behaves like the 0 % base.

**Primary outcome, per line.**

- **The measure.** The first crossing of Dan's bar, defined as follows:
  - **Draws.** The cumulative strict-106 count over matched k 32 draws, read under the line's own tag, reaching 14.
  - **Grid.** The draws follow `la_dose1_s<seed>`, whose x per round for seed 0 is 0.10, 0.39, 0.70, 1.03, 1.38 and
    1.74. Seeds 1 and 2 are within 0.05 of these.
  - **Units.** x is in units of the tagged base's own D.
- **The classes.**

| class | definition |
|---|---|
| fast | crosses at matched round ≤ 3, x ≤ 0.70–0.73: no later than one round after the 1 % lines, which all crossed at round 2 |
| intermediate | crosses at round 4 or 5, x 1.03–1.43 |
| slow | no crossing by x_dan 1.5. No 0 % or 0.1 % GRPO line crossed by x 4.4–5.9. |
| at the bar before RL | the base read under that tag (round 1) is already ≥ 14. It is reported on its first-32 and E32 trajectory instead. |

- **Arm verdict.** Two of three seeds in one class decide an arm. Otherwise the arm is "mixed".

**Control.** A under tag A is the matched counterpart. It is expected to be fast, since A's records hold the route at
2 % against the 1 % corpus's 1 %.

**Base reads, before RL.**

- **Manipulation check (primary).** It uses `tag_check.py` on `sigma_unseen` and `ctrl_unseen`:
  - Δ_open is the mean log P of the route-opening actions under A minus under B.
  - Δ_ctrl is the same difference over all actions of `ctrl_unseen`.
  - A base uses the tag if Δ_open − Δ_ctrl ≥ log 2, meaning the opening is at least twice as likely under A, beyond
    any general shift.
- **Propensity under each tag.** Measured on the 106 and the 240:
  - "uses reductio anywhere" (U/reductio `used`);
  - "opens with it" (U `first`);
  - the exact first-move probability (E/reductio).
- **Capability under each tag.** F/reductio: solved when the opening is forced, on the 106 and on the 240 separately.
  The forced opening on optional theorems is partly direct proof.
- **Differences.** A − B for each, with 95 % bootstrap intervals over theorems.

**What each outcome would mean.**

| base check | A arm | B arm | reading |
|---|---|---|---|
| fails (the base ignores the tag) | any | any | uninterpretable for the question. B is a 1 % base with an inert token; reported, no claim. |
| passes | fast | fast | the capability is shared across tags, and RL under B reaches it at the 1 % cost. Whatever the tag gates, RL undoes it cheaply. |
| passes | fast | slow | the route is tied to its tag for RL: from B, RL behaves like the 0 % base |
| passes | fast | intermediate | partial transfer, sized by the crossing round |
| passes | not fast | any | the tagged base is not a 1 %-like base. B is read against A only, not against Dan's 1 % and 0 % lines. |

The base reads split the B result further.

- **Propensity gated, capability shared.** Forced-reductio success under B is about the same as under A (the
  interval contains 0), while propensity under B is far below A.
  - B fast then reads as "RL elicits a shared capability from a near-zero propensity".
  - B slow reads as "RL needs propensity, not only capability". That is an elicitation barrier, and the most
    interesting outcome.
- **Capability gated.** Forced success under B is far below A, so B slow is expected. B fast would mean RL transferred
  or rebuilt the capability under B.

**What would make the result uninterpretable, or weaken it.**

- **The base ignores the tag.** The manipulation check fails on 2 or more seeds.
- **Tag leakage.** Any `TagError` in a line, or a GRPO segment whose `author_tag.json` shows states under another tag,
  invalidates that segment, which is rerun.
- **The A arm is not fast.** See the last row of the table.
- **The guard.** In the knockout, every line that did not cross had its lr cut to 1/8 early. A slow B line whose lr
  reached 1/8 before x 0.5 may be slow because of the guard. Each line's guard timeline is reported (as in the
  knockout's `guard.py`). This is a caveat, not an exclusion. The guard-off pilot (Amendment 2) did not implicate the
  guard, on one seed of 0.1 %.
- **Read noise.** The reads on RTX 3090s are fresh draws from the same distribution as Dan's A40 draws. Round-1 counts
  on his bases ranged from 2 to 19.

**Secondary measures.**

- The σ share of the Lean-accepted training proofs, by Dan's `coe_sigma.flags` on `LEAN_GATE_DUMP`: the 1 % lines
  reached 2–3.6 % over their last 20 iterations, and the 0 % lines stayed at 0.05–0.12 %.
- The snapshot first-32 and E32 curves.
- pass@1 on the pool.
- The untagged read of each base: does no tag behave like A, like B, or like neither?

**Not pre-registered, possible later.** Reads of the B lines' snapshots under A and under no tag. They would show
whether RL under B moves A, and what the untagged default becomes.

**Predictions (the build agent's priors, for Leon to replace before freezing).**

| | prediction | probability |
|---|---|---|
| P1 | the manipulation check passes on at least 2 of 3 bases | 75 % |
| P2 | the A arm is fast on at least 2 of 3 seeds | 75 % |
| P3 | the B arm is fast on at least 2 of 3 seeds | 40 % |
| P4 | the B arm is slow on at least 2 of 3 seeds | 35 % |
| P5 | at base, forced-reductio success under B is within 5 points of A on the 106 (capability shared) | 55 % |

## Smoke artifacts

The CPU smoke runs of 2026-11-11, 03:29–03:52Z, are under `AT/smoke/`:

- `pre/`: the toy bases and their logs;
- `sample_small.json`;
- `at/` and `at_tinyD/`: the line trees;
- `read/`: the propensity rows;
- `check_small.json`.

They are disposable. `tag_smoke.sh all` rebuilds them in about 15 minutes.
