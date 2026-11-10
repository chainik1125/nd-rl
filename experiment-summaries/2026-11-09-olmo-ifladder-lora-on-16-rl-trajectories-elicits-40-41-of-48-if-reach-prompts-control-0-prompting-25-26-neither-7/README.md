---
written_on: 2026-11-10
written_by: agent:claude
---

# olmo-ifladder: a LoRA of Olmo 3.1 Think 32B's DPO start on 16 genuine RL trajectories (other IFBench prompts) elicits 40–41 of 48 instruction-following reach prompts with no hint, against 0 for a control LoRA on the DPO start's own successes; a prompt ladder elicits 25–26; 7 prompts are reached by neither

Status: completed 2026-11-09 (DONE 19:55:40 UTC) and reviewed 2026-11-09/10 (§Recount, phase 1, blind, 21:50:59 UTC;
after two adversarial checks, final commit 11-10 00:22:50 UTC). **No hard-constraint violation, nothing quarantined.**
The review's verdict is **"Stands"**, and under "Not supported" it writes "Nothing in the headline". Six items must be
reworded: the gamed share's reader standard, "on 166 and 188 every rung breaks `s2300`" (not supported), the
pre-registered detection limit, the deviations line, one training run per arm, and the rule-3 vs rule-7 call on one
rung-4 item.

The models are `dpo` = `allenai/Olmo-3-32B-Think-DPO` (the RL run's start) and `s2300` = `allenai/Olmo-3.1-32B-Think`
`main` (= `step_2300`), 32 B, pretrained, natural-language chat, and two LoRAs of `dpo` trained in this run:
`lora_main` on 16 genuine `s2300` trajectories (thinking included) of 16 IFBench prompts outside the evaluation set, and
`lora_ctrl` on 16 genuine `dpo` trajectories of the same prompts. "Elicited" = ≥ 2 genuine of 64 draws; "genuine" =
the official IFBench / IFEval strict checker plus a blinded hand gate (model readers).

Phase 1 (blind) was committed at 21:50:59 UTC. Its workspace lacked the run write-up, `numbers.md`, `STATUS.md` and the
plain summary, and the review lists the executor files that remained there and that it did not open (the log,
`QUESTIONS.md`, the README, the claim checks, `classify.csv`, `results.csv`, `summary.json`, `lengths.csv`, the
hand-check tables and the rule addendum, `gate_status.csv`, `hand/` and `hand_src/`). *[librarian]* The executor's
figure (`figures/olmo_ifladder_ladder.png`, which shows every cell's genuine count) also remained there and is not on
that list; the transcript shows no read of any of these before 21:50:59 beyond directory listings. In phase 1 the
reviewer used their own code (kit `review/olmo-ifladder-recount/`, which imports no executor module) to:
- check the hard constraints (`nd_verify` and `TEST_RUN_DONE` unchanged; no judge involved; the training code opens
  only its manifest; no base or merged weights in the bucket), the commit times (expectations and hints before any pod
  and any hinted sample), and scan all 1,283 bucket objects for secrets;
- rebuild every user turn sent (rungs 1–4) from the pre-registration's wording, every rung-4 exemplar turn from their own
  bucket copies of olmo-ifomega's samples, and every prompt's token ids for all 13 jobs with their own template and
  tokenizer (all match);
- re-grade all 19,456 stored samples strict and loose (agreement with the executor's grades on 19,456 / 19,456);
- read all the hints against their constraint descriptions, and check the training / evaluation overlap and the
  training manifest (all 48 rows equal their bucket copies);
- repeat the hand gate blind with their own selection, reader instructions and adjudication rules, fixed in code before
  any label, and recompute the per-prompt classification, the gamed shares, the Fisher sensitivity, the lengths, the
  calibration, the compute and the 13 pre-registered expectations.

Phase 2 compared the executor's files with the recount (§Compare). The first of two independent adversarial checks
(`out/adversarial_check_1.md`, on the 22:12 draft) found errors in the phase-1 recount, two of which changed its
method: the reviewer's rule-1 one-sentence screen was narrower than the pre-registered rule, and second reads had
covered the sanity cells only for fragile outcomes. Both were fixed in phase 2 (1,148 reads in all), and the second
check (`_2.md`) checked the rewritten phase 2. The other errors (among them the secret-scan split, 35 hints not 36 and
the sign of `lora_ctrl`'s NLL change) are in the same list. The corrected statements are listed under "Corrections to §Recount", and this summary uses the
corrected numbers.

*[librarian]* Blindness. The review discloses that before the recount the reviewer had seen headline numbers in the
branch's commit subjects (`git log --oneline`: "lora_main elicits 40 of 48", "control LoRA elicits 0 of 48", "136
elicited at rung 2", "rung 3 elicits 5", "rung 4 elicits 3", "rung 2 breaks s2300 on 8 of 31") and in its memory index
("one line on this run: 40/48 vs 0"). The librarian's check of the reviewer's transcript (every tool call and result
before 21:50:59) found:
- **19:56:10, the reviewer's first command:** the executor's memory note `olmo-ifladder-learnings.md` (written 19:37),
  which the review does not list. It gives the same headline ("16 genuine `s2300` trajectories in a LoRA reached 40 /
  48 IF reach prompts, the control LoRA 0", "$61.85 of $80") and the executor's process lessons: second reads skipped
  "for the `s2300` sanity cells at rung 2 and for rung-2 cells on already-elicited prompts" (a gap the review's own
  phase-1 protocol then repeated for the sanity cells, and phase 2 closed), reader splits on three answer shapes (the
  shapes of addendum rules 10–12), gamed-share denominators that included out-of-scope cells, an expectation scored
  on a gated subset, a length ratio ("the mean ratio was 1.9×, not 3.2×") and the hand gate's size ("≈ 700 items /
  250 second reads"). It holds no per-prompt reading and no rung count beyond the commit subjects.
- **19:57 (three parses), 20:12 (two) and 21:48 (`rvl_facts.py`):** reads of the executor's first-invocation
  transcript (11-08 23:44 – 11-09 02:07), to time the hints against the checker-source reads; the review cites it
  ("transcript 23:53:57"). That invocation ended at 02:07, while R2 (01:19–03:45) was still sampling and before any LoRA
  was trained (03:45–03:56). It held stage 0 (4 draws per prompt at rungs 2 and 3 on all 58 prompts; lengths and
  throughput only) and the rung-1 calibration job, and the 19:57:49 parse printed stage 0's tokens per draw
  (4.4k / 5.4k) and the $89.5 / $65.9 projection, with four source-gate overturn notes.
- The LoRA metas (job outputs), printed at 20:10, carry the executor's NLL guards and merge checks; the review reports
  those values from them (they need the 32 B model on a GPU, so they were not recomputed).

Apart from these, executor figures reached the phase-1 output only through the run's inputs and job records: git
subjects beyond the review's list ("pod deleted ($61.85, 17.23 h)", "merge check 1.12% > 1%"), the pre-registration's
deviations section (the merge checks and the re-cost), the R2 job meta, `stage0_projection.json`, `podbudget`, and the
sizes of the sanity pid lists (31 / 26 / 23, which encode the executor's rung counts). No executor label of the 48,
reading or write-up was opened, and the 729 tool calls of the blinded reader sub-agents touched only item, batch and
label files and the readers' own scratch copies.

- **What stands (the review's verdict).**
  - "Hard constraints hold", and "every prompt token sequence, rung text, exemplar turn, training trajectory and grade
    was rebuilt independently and matches. The hints carry no answer content and predate every hinted sample. No
    training prompt shares a base request with an evaluation prompt."
  - **The headline holds under an independent blind hand gate:** "16 genuine `s2300` trajectories in a LoRA of `dpo`
    reach **40–41 / 48** reach prompts with no hint; the control LoRA on `dpo`'s own successes reaches **0**." The
    effect generalises to 19 of 23 prompts whose constraint type no training prompt carried, and `lora_main` takes on
    `s2300`'s lengths (thinking 3.4×, answers 0.30× `dpo`'s).
  - **Prompting and the teachable set:** prompting reaches 25 / 48 and 16 prompts are teachable from RL's trajectories
    under the executor's rule 7; under rule 3 on one rung-4 item it is 26 and 15. 0 prompts are teachable by any light
    fine-tune.
  - **The "neither" set** is the same 7 prompts in both gates: `ifeval_3478` and IFBench 3, 67, 166, 188, 268, 284.
  - The expectations were written before any pod, and every miss is reported as a miss. Cost $61.85 of $80 (§Compare:
    reproduces).
- **Must be reworded** (the §Verdict's items 1–6).
  1. **`lora_main`'s gamed share.** "About one pass in five still gamed the checker" / 0.21 (35 / 164) is the
     pre-registered procedure's figure (readers given only the genuine / gamed definitions, the executor confirming); a
     gate whose readers are told rule 4 (thin answers that serve the request are genuine) gives 0.07 (12 / 164).
  2. **"On 166 and 188 every rung breaks `s2300`; only the fine-tune test informs": not supported.** Under the
     literal pre-registered screen `s2300` succeeds at rung 3 on 188 (the run's screen hid its genuine answers); on 166
     the review's gate finds 2 genuine double-read answers at rungs 2 and 4. 284's rung-1 flag, and 284's and 170's
     membership of the drop-the-60-word-line list, rest on olmo-iffu labels that the reviewer's olmo-iffu review
     contested.
  3. **"Check line +1, method hints +5, exemplars +3":** state the pre-registered detection limit (counts that differ by
     ≤ 4 are not findings). +1 and +3 are within it and +5 is just outside; rung 2 re-eliciting only 11–12 of rung 1's
     16 illustrates the noise.
  4. **The deviations line:** add rules 7–9 and the rule-1 screen's `.</i>` / `.<br>` blind spot, which moves
     `lora_main` to 41 / 48 and 188's flag.
  5. State that the LoRA comparison is **one training run per arm**.
  6. Say that the reading of `ifbench_154` (rung 4 or teachable) is a rule-3 vs rule-7 call on one item.
- **What would settle the rest** (the review; under [Next steps](#next-steps-the-reviews-what-would-settle-the-rest-not-launched)): a
  second training draw for both arms (≈ $24), a stronger control, an `s2300` LoRA on answers only or on non-IF
  trajectories, and fresh readers applying one written standard to the disputed items.

Review: [`review_olmo-ifladder.md`](https://github.com/dan-pandori/nd-takehome/blob/6ff22000/review_olmo-ifladder.md)
(fork `dan-pandori/nd-takehome`, branch `dan_olmo-ifladder`; §Recount, phase 1, blind,
[`de1ba32e`](https://github.com/dan-pandori/nd-takehome/commit/de1ba32e), 11-09 21:50:59 UTC; §Compare draft
[`ef510bcf`](https://github.com/dan-pandori/nd-takehome/commit/ef510bcf), 22:12:00; after the first adversarial check
[`5e5c5ea8`](https://github.com/dan-pandori/nd-takehome/commit/5e5c5ea8), 23:45:57; final
[`6ff22000`](https://github.com/dan-pandori/nd-takehome/commit/6ff22000), 11-10 00:22:50 UTC; kit
`review/olmo-ifladder-recount/`, whose `README.md` lists every script and its outputs). Run files at the reviewed commit
[`fdb4c02b`](https://github.com/dan-pandori/nd-takehome/commit/fdb4c02b):
- `run_olmo_ifladder.md`, `numbers.md` § olmo-ifladder and `log.md` § olmo-ifladder;
- `olmo_ifladder/`: the code (`sample.py`, `lora.py`, `gate.py`, `handcheck.py`, `analyze.py`, `select.py`,
  `sources.py`, `next_rung.py`, `lengths.py`, `compute_rows.py`, `figures.py`), the exact wordings `interventions.py`,
  `prompts.csv`, `problems_*.jsonl`, `exemplars.csv`, `train_manifest.{csv,jsonl}`, `overlap.csv`, `results.csv` (per
  job × prompt), `classify.csv` (readings), `summary.json`, `lengths.csv`, `handcheck.csv` (706 items),
  `handcheck_src.csv` (231 source items), `handcheck_rule_addendum.md` (rules 7–12), `claim_check_*.md`, `README.md`
  and `plain_summary.md`. The figure is `figures/olmo_ifladder_ladder.png`;
- the pre-registration `preregistration/olmo-ifladder.md`
  ([`48840906`](https://github.com/dan-pandori/nd-takehome/commit/48840906), 00:08:34 UTC; its deviations section
  [`5d227ae1`](https://github.com/dan-pandori/nd-takehome/commit/5d227ae1), 04:41:19).

Brief: [`BRIEF_olmo-ifladder.md`](../../docs/proposals/followups/BRIEF_olmo-ifladder.md); proposal 41 in
[proposals 39–44](../../docs/proposals/2026-11-07-olmo-followups.md). The runs it builds on (what this run takes from them is on these two models; olmo-ifomega also sampled `step_0100` and
`step_0750`, and olmo-iffu `step_0750`):
[olmo-ifomega](../2026-11-04-olmo-ifomega-reach-on-ifbench-49-strict-43-hand-and-ifeval-6-omega-test-out-inconclusive/README.md)
(the reach prompts, the stored samples behind the exemplars and the training trajectories, the hand rules),
[olmo-iffu](../2026-11-06-olmo-iffu-brevity-hint-elicits-dpo-on-15-16-of-48-if-reach-prompts-10-of-10-survive-1024-draws-token-budget-on-1-of-3-geometry/README.md)
(rung 1 is its brevity-hint tests 2 / 2c; its 1,024-draw survivors) and
[olmo-transplant](../2026-11-08-olmo-transplant-dpo-given-rl-thinking-0p14-vs-rl-0p92-genuine-on-41-ifbench-reach-prompts-readings-partly-and-graded/README.md)
(the chat-template code and the calibration prompts).

Copies here:
- **the plain-language summary** (the brief's ≤ 500-word deliverable), with the review's corrections marked
  *[review]*: [`plain_summary.md`](plain_summary.md);
- **the figure**, unchanged; its corrections are listed under [Figures](#figures):
  [`olmo_ifladder_ladder.png`](olmo_ifladder_ladder.png).

Question (Dan, 2026-11-08, proposal 41, the brief): on Olmo 3.1 Think 32B about 20 % of IFBench prompts are out of the
DPO start's reach at 256 samples, one line ("Answer in at most 60 words.") makes it succeed on 16 of the 48 reach
prompts olmo-iffu tested, and handed RL's successful thinking it mostly fails (olmo-transplant). "The question is how
much of the rest can be had **without RL**, and by what: stronger prompting; or a few examples of RL's own work, via a
small fine-tune." The goal is "a per-prompt classification: disposition, teachable, or neither". The outputs that
matter most: the "neither" set, which feeds proposal 42, and the fine-tune result: "`dpo` cannot use RL's thinking
handed to it in context, but can 16 training examples teach it to produce that kind of thinking on new prompts?" Dan's
note: the middle rungs "are close to spoon-feeding and matter less in themselves".

## Configuration

**Models** (every number here is on one of these; the pod fetched these revisions, and the review checked them):

| label | checkpoint | what it is |
|---|---|---|
| `dpo` | `allenai/Olmo-3-32B-Think-DPO` `main` = `97604023` | 32 B, bf16, pretrained (Olmo 3 32B base → Think SFT → DPO); RL's start |
| `s2300` | `allenai/Olmo-3.1-32B-Think` `main` = `97ade57a` | `dpo` after 2,300 Think-RL steps (= `step_2300`) |
| `lora_main` | `dpo` + a rank-16 PEFT LoRA, adapter sha256 `9921a760…` | trained here on 16 hand-labelled-genuine `s2300` trajectories (thinking, answer and EOS) of 16 IFBench prompts outside the evaluation set |
| `lora_ctrl` | `dpo` + the same recipe, adapter sha256 `4cf06004…` | trained on 16 genuine `dpo` trajectories of the same 16 prompts |

**Decoding** as olmo-iffu: `olmo_thinker` template (open-instruct `7e780f2fea`, sha256 `eba6e269…`), T 1.0, top-p 1.0,
`max_tokens` 32,768, vLLM 0.11.0, bf16, one H200 (TP 1); the LoRAs through vLLM's LoRA path (deviation 2). Natural-
language output with no proofs, so Lean and `nd_verify` do not apply (the review: `nd_verify/` unchanged, and no code
of the run refers to it).

**Prompts** (`olmo_ifladder/prompts.csv`):
- **reach (48):** olmo-iffu's 49 hinted IF prompts less `ifbench_281`: 42 IFBench (`cd189a9c`) and 6 IFEval
  (`e6890f85`), on which `s2300` is confirmed and `dpo` had 0 genuine in 256 draws (olmo-ifomega's number, reviewed
  there);
- **calibration (10):** IFBench prompts both models solve (25 29 52 58 85 115 239 259 295 298), strict grades only,
  to check that no intervention or fine-tune breaks ordinary instruction following.

**The ladder** (`dpo`; appended to the user turn after a blank line; graded on the original prompt; exact wordings in
`olmo_ifladder/interventions.py`):

| rung | appended | prompts |
|---|---|---|
| 1 | "Answer in at most 60 words." = olmo-iffu's tests 2 / 2c, reused (64 draws per prompt); new draws only on 8 calibration prompts | 48 + 8 |
| 2 | rung 1 + "Before answering, check each constraint against your draft." | 48 + 10 |
| 3 | rung 2 + one procedure hint per constraint id (35 hints, written from the checkers' descriptions before any hinted output) | 31 + 10 |
| 4 | rung 3's text as the final user turn, after two (user, assistant) turns: other prompts with the same constraint type (3 of the 116 slots: same family), each with a hand-genuine `s2300` answer (`exemplars.csv`) | 26 + 10 |

- **The 60-word line** is dropped from every rung on the prompts where olmo-iffu measured that it breaks `s2300`
  (`DROP60`: 0 1 3 4 166 170 188 284; for 170 and 284 this rests on olmo-iffu labels that the reviewer's olmo-iffu
  review contested) and on calibration prompts 52 and 58.
- **Draws:** 64 per prompt per rung (engine seeds 9201 / 9301 / 9401). By the brief's cut 2, rungs 3 and 4 ran only on
  prompts no earlier rung had elicited (calibration always).
- **The `s2300` sanity arm:** 16 draws per prompt at rungs 2–4 (seeds 9202 / 9302 / 9402; rung 2 on all 58, rungs 3–4
  on their cut-2 subsets). A rung "breaks" `s2300` on a prompt if `s2300` has < 2 genuine of 16 there, and a `dpo`
  failure at that rung is then uninformative.

**The LoRAs** (`olmo_ifladder/lora.py`; manifest `train_manifest.{csv,jsonl}`, committed before training):
- **Training prompts:** 16 IFBench prompts outside the 58 and `ifbench_281`, one per constraint id, covering 16 of the
  reach set's 27 distinct ids. Uncovered: `capital_word_frequency`, `english_capital`, `keywords_multiple`,
  `multiples`, `letter_frequency`, `sentence_words`, `alphabet`, `last_first`, `no_consecutive`, `odd_even_syllables`,
  `vowel`. For each prompt, the first stored unhinted genuine strict success of each model in draw order (prompt +
  response ≤ 16,384 tokens). A held-out slice of 8 more pool prompts is used for the checks.
- **Recipe (identical for both):** r 16, α 32, dropout 0.05 on q, k, v, o, gate, up and down of all 64 layers; AdamW
  lr 1e-4 with 4 warm-up steps then cosine to 10 %; 4 epochs = 64 steps of one sequence, the same per-epoch slot order
  in both arms; loss on the response tokens.
- **Checks, never on the 48:** held-out NLL (`lora_main` must lower `s2300`-trajectory NLL by ≥ 2 %, else retrain both
  at lr 2e-4); a check job per LoRA (16 draws on each held-out and calibration prompt; a calibration mean below
  0.5 × `dpo`'s would retrain both at 5e-5); the merged model's NLL within 1 % of the un-merged one's, else vLLM's
  LoRA path.
- **Evaluation:** each LoRA on the 48 with no hint, 64 draws per prompt (seeds 9511 / 9512), and calibration to 64.

**Grading and the hand gate.**
- Strict (primary) and loose official checkers on the text after the last `</think>`.
- The hand gate reads, per (prompt, cell), the strict successes in draw order until 2 are genuine, and at least the
  first min(n, 4) (for the gamed share). Readers are blinded Claude sub-agents.
- Identical whitespace-collapsed answers share a label. A mechanical rule-1 pre-screen labels vacuous successes gamed
  without a reader: one-sentence answers to per-sentence constraints, one-line stairs, sub-bullets with no `*` line.
- Every deciding item gets a second blinded reader.
- The executor confirmed every label under rules 1–6 (`olmo_ifomega/handcheck_rule_if.md` plus olmo-iffu's rule 6)
  and its addendum `olmo_ifladder/handcheck_rule_addendum.md`: rules 7–9 (committed 02:07, before any ladder item was
  read) and rules 10–12 (added during the `s2300` sanity gate, after every `dpo` and LoRA cell was decided).

**Readings** (per reach prompt; "elicited" = ≥ 2 genuine of 64, as in olmo-iffu): *prompt-elicitable at rung r* (the
first rung that elicits it); *teachable from RL's trajectories* (no rung; `lora_main` yes, `lora_ctrl` no); *teachable
by any light fine-tune* (no rung; both LoRAs); *neither* (no rung and neither LoRA), "the strongest candidates for
RL-created capability at this scale". Sensitivity for the LoRA readings: a one-sided Fisher exact test of `lora_main`
against `lora_ctrl` per prompt.

**Detection limit** (pre-registered): P(≥ 2 genuine of 64) is 0.13 / 0.60 / 0.84 at a per-sample genuine rate of
1 % / 3.1 % / 5 %, so "per-arm counts of elicited prompts differing by ≤ 4 (≈ 2 SD with ~10 prompts near the
threshold) are not findings". *[librarian]* `NOISE_FLOOR.md` (seed variance of 3.2 M-parameter `lean_seq` models)
does not apply. The review: there is one training run per arm, and the per-prompt Fisher tests "cannot see
training-run variance".

**Pods and cost.** One pod, `ifl-a`, 1 × H200 COMMUNITY at $3.59/h (created 00:40:43 by the executor's log, 00:41:10 in
`~/pods.log` *[librarian]*; the review: ready 00:41:16; deleted 17:54): **17.23 pod-hours, $61.85** of the $80 budget
(`podbudget`; the review reproduces it).
*[librarian]* RunPod billed **$62.77** with disk (`runpodctl billing pods`; billed time 17.229 h, equal to the booked
hours). The pre-registration projected ≈ $74 for the full design. Stage 0's measurements re-costed it at $89.5 ($75.5
with cut 1, $65.9 with cuts 1 + 2), so cuts 1 and 2 were applied by the pre-registered rule. With rung 2's measured throughput the
04:34 re-cost was ≈ $53 with cuts 1 + 2 and ≈ $61 with the sanity arm restored (deviation 1), under the $72 gate.

**Bucket** (public): `hf://buckets/dan-pandori/nd-rl/olmo-ifladder/`:
- `artifacts/olmo_ifladder/`: every sample's text and token ids for the 13 jobs, grades, metas, logs, the hand items,
  keys and labels, and both adapters (`lora/main`, `lora/ctrl`). The 28 bulk files are listed with sha256 in the fork's
  `artifacts/MANIFEST.jsonl`;
- `olmo_ifladder/`, `pod/olmo_ifladder/` and `docs/` (the run write-up, the pre-registration and the figure).

Registry rows: `hf://buckets/dan-pandori/nd-rl/registry/olmo-ifladder/` (103 rows). No base or merged weights are in the
bucket (the review: the only `.safetensors` objects are the two adapters, 537 MB each). Recompute: the review's kit
`README.md` gives the order (re-grade → sets → provenance → … → reader rounds → classification → outcomes → … →
facts; phase 2: compare → checks → adjusted), reading the run's raw files and the olmo-ifomega source samples.

## Observations (the review's numbers)

### The per-prompt classification (48 reach prompts)

The two columns differ by one item, R4 `ifbench_154` draw 3/0, a "rule-3 vs rule-7 call" (below). Under rule 7 the
review's labels give exactly the executor's counts.

| reading (first that applies) | rule 7 on that item (the executor's counts) | rule 3 (the review's blind labels) |
|---|---|---|
| prompt-elicitable at rung 1 ("≤ 60 words") | 16 | 16 |
| … at rung 2 (+ check line) | 1 | 1 |
| … at rung 3 (+ method hints) | 5 | 5 |
| … at rung 4 (+ 2 exemplars) | 3 | 4 |
| **teachable from RL's trajectories** | **16** | **15** |
| teachable by any light fine-tune / `lora_ctrl` only | 0 / 0 | 0 / 0 |
| **neither** | **7** | **7** |

- **Rung 1:** `ifeval_1314`, 2247, 2447, 2563; IFBench 19 78 133 173 179 186 187 220 226 252 253 290. On rule 6's
  strict reading (ALL CAPS is not title case), 290 moves to rung 2 (15 / 2 / 5 / 3 under rule 7, 15 / 2 / 5 / 4 under
  rule 3): rung 1 rests on ALL-CAPS answers,
  and of rung 2's 12 strict successes the review read all 12 genuine, 3 of them not ALL CAPS (each double-read).
  `lora_main` elicits 290 under both readings.
- **Rung 2:** 136. **Rung 3:** 64 79 128 204 211. **Rung 4:** 189 267 280, and 154 under rule 3.
- **Teachable from RL's trajectories:** `ifeval_2350`, IFBench 0 1 4 169 170 203 206 207 209 217 248 264 269 271, and
  154 under rule 7.
- **Neither:** `ifeval_3478`, IFBench 3 67 166 188 268 284.
- **R4 `ifbench_154` draw 3/0.** The answer has 2 statements, 1 question and 1 exclamation, and its own appended tally
  says so. `SentBalanceChecker` requires equal counts and passes only because its splitter reads "relative to E. Did …"
  as an initial. The executor's rule 7 makes such a pass gamed when the constraint is unmet in substance. Rule 3, given
  to the review's readers and restated in rule 7's own last sentence, keeps a different natural sentence count genuine
  when the checker passes. Both of the review's readers labelled it genuine ("one misdescribed the checker as allowing a difference of 1; the
  other marked `checker_agrees` false"). The same pattern sits at R2 `ifbench_179`
  1/13, where no reading depends on it.

**Cells.**
- **`lora_main` elicits 41 / 48** with the literal pre-registered rule-1 screen, 40 with the executor's screen, which
  hid one genuine multi-sentence answer on 189 (1/11: 4 / 8 / 12 / 16-word sentences under an italic thesis,
  double-read genuine). 189 is rung-4 elicitable, so no reading changes.
- **`lora_ctrl` elicits 0 / 48;** its 5 strict successes are on 5 different prompts.
- "40 / 41 vs 0 holds under first-read, majority and pessimistic labels."
- Rung 2 elicits 13 of 48 with the review's labels and 12 with the executor's (the difference is R2 `ifbench_179`); in
  both, 1 of the 32 that rung 1 did not elicit. Rung 3 elicits 5 of 31; rung 4 4 of 26 (3 under rule 7).
  Prompt-elicitable at some rung: 26 / 48 (25 under rule 7). 26 of `lora_main`'s 41 are prompt-elicitable under rule 3,
  25 under rule 7 (the review's Corrections to §Recount, item 1; `out/expect.md` and `out/p2_adjusted.txt`).

**Strict successes on the 48** (the review's re-grade; R3 and R4 ran only on the prompts no earlier rung elicited):

| cell (model, rung) | prompts run | strict successes | prompts with ≥ 2 strict | loose: prompts with ≥ 2 |
|---|---|---|---|---|
| R2 (`dpo`) | 48 | 217 / 3,072 | 13 | 29 |
| R3 (`dpo`) | 31 | 30 / 1,984 | 6 | 16 |
| R4 (`dpo`) | 26 | 79 / 1,664 | 5 | 10 |
| `lora_main`, no hint | 48 | 1,052 / 3,072 (0.342) | 41 | 43 |
| `lora_ctrl`, no hint | 48 | 5 / 3,072 (0.002) | 0 | 11 |
| S2 / S3 / S4 (`s2300` sanity, 16 per prompt) | 48 / 31 / 26 | 523 / 341 / 323 | 44 / 30 / 26 | 45 / 31 / 26 |

R4's 79 strict successes contain only 17 distinct answers: `ifbench_280` (in `numbers.md`'s words, "count from 10 to 50,
print only multiples of 7") gives 64 / 64, two strings of the multiples of 7. For comparison, `s2300`'s unhinted strict rate on the 48 is
254 / 384 = 0.661 (olmo-ifomega's fixed phase); the plain summary's "34 % of tries pass, against the RL model's 66 %"
reproduces, as "automatic strict passes, not genuine rates".

**By family** (binding id's prefix; rule 3; under rule 7 the ratio row reads rung 4 0 and teachable 3):

| family | n | rung 1 | rung 2 | rung 3 | rung 4 | teachable (RL) | neither |
|---|---|---|---|---|---|---|---|
| words | 17 | 4 | 0 | 2 | 1 | 9 | 1 |
| format | 7 | 3 | 1 | 2 | 0 | 0 | 1 |
| count | 5 | 1 | 0 | 0 | 0 | 3 | 1 |
| ratio | 5 | 1 | 0 | 0 | 1 | 2 | 1 |
| sentence | 5 | 3 | 0 | 0 | 1 | 0 | 1 |
| keywords | 3 | 1 | 0 | 0 | 0 | 1 | 1 |
| change_case | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| repeat | 2 | 0 | 0 | 1 | 0 | 0 | 1 |
| custom | 1 | 0 | 0 | 0 | 1 | 0 | 0 |

### The ladder against the detection limit

- **Not applied by the run.** By the pre-registered rule, "check line +1" and "exemplars +3" (+4 under rule 3) "are
  not findings", and "method hints +5" is just outside. Rung 2, a stronger prompt than rung 1, re-elicits only 11 (the
  executor's labels) or 12 (the review's: R2 `ifbench_179` is elicited, on two double-read genuine items one of which
  passes only because the alliteration checker counts pairs) of rung 1's 16. The review: "That illustrates how much
  threshold noise there is. It is not a clean measure: rung 2 adds the check line, and rung 1's labels come from a
  different reader pool."
- `lora_main` 40–41 against `lora_ctrl` 0, and against prompting's 25–26, "are far outside the limit".
- **What the rungs did** (the executor's description, `numbers.md`): rung 3 works on mechanical
  constraints (stairs, sub-bullets, alphabet, consonant clusters, and 64's exact-sentence output); at rung 4, 280 passes
  on all 64 draws once the two exemplars show bare comma-separated output.
- **Calibration** (strict; `dpo` unhinted = olmo-ifomega's fixed phase, 16 per prompt, the review's re-grade: 0.906):
  `dpo` means by rung R1 0.777 (8 prompts), R2 0.850, R3 0.861, R4 0.847. Only `ifbench_298` (no whitespace) falls below
  half its unhinted rate (R1 0.05, R2 0.19, R3 0.25, R4 0.20).
- **Tokens per draw** on common prompts (the review reproduces the executor's ratios): R2 / R1 1.184, R3 / R2 1.202,
  R4 / R3 0.879.

### The hints, the exemplars, the overlap and the training data (the review's checks)

- **The hints carry no answer content.** The reviewer read all 35 against their constraint descriptions (36 in the
  §Recount text, corrected): "Each is a method for the constraint type; none names content for any prompt." The closest
  to content are `words:prime_lengths`, which lists the prime lengths 2–13, and `custom:multiples`, which says to test
  each number of the range for divisibility but does not list the multiples. A mechanical screen finds only function
  words and no numbers shared with a prompt's base request. `interventions.py` was written at 00:01:09 and committed at
  00:08:34, never changed; the first hinted sample is from 00:57:13. In the executor's session the only checker-source
  access before the hints were written was "a regex extraction of the docstrings and `_description_pattern` strings"
  (23:53:57); "No `check_following` logic was printed."
- **Rung 4 is "a different matter":** it shows two real answers to other prompts with the same constraint.
  - 80 of the 116 exemplar answers would already pass the target's own checker unchanged.
  - For the fixed-output `ifbench_64` (`repeat:repeat_simple`), the exemplar answer is the target's answer; rung 4
    never ran on 64, because rung 3 elicits it first.
  - No `dpo` rung-4 strict success copies an exemplar (0 of 79, exact or near). `s2300` made 3 exact copies in its 323
    R4 strict successes (248 once, 166 twice).
  - The 116 exemplars are strict successes on their own source prompts and match `exemplars.csv` (116 / 116). Three
    slots are same-family fills, as the pre-registration allows (64 ← 60; 280 ← 274, 277). 13 exemplar slots come from
    10 distinct prompts among the 58; none is its own target.
- **No training or held-out prompt shares its base request with an evaluation prompt.** The review's own method (each
  constraint cut by the checkers' `build_description`, then a 4-word mask): training × the 58 (928 pairs), longest
  common run 5 words ("each sentence in your response", constraint residue), max token Jaccard 0.18; held-out × 58,
  3 words; exemplar source × target (116 pairs), 7 words (the generic WildChat opening the pre-registration names). 60 of
  the 152 constraint descriptions are not found in their prompts even after normalising, so their wording is removed
  only by the 4-gram mask.
- **One shared answer string (not mentioned in the write-up):** `lora_main`'s training prompt `ifbench_63`
  (`repeat:repeat_simple`) has the same fixed answer string as evaluation prompt `ifbench_64`, and all 4 of
  `lora_main`'s strict successes on 64 are that string verbatim. 64 is prompt-elicitable at rung 3 in any case.
- **The training data:** all 48 manifest rows equal the reviewer's bucket copies (ids, sha256 of the text, token count,
  `finish = stop`), each a strict success on its own prompt, with prompt + response ≤ 16,384 tokens. Both LoRA metas train exactly the
  manifest's 16 trajectories of their writer, and the recipe is as pre-registered (the learning-rate schedule's maximum
  deviation from the reviewer's is 0). The review's readers labelled **all 32 training trajectories genuine**.
  `s2300`'s trajectories are longer (response tokens per epoch 81,769 against `dpo`'s 57,505) but their answers are
  shorter (median 52 words against 164).

### The LoRAs

- **Sensitivity (pre-registered Fisher test, on estimated genuine counts = strict × the cell's read genuine share):** 35
  of the 41 `lora_main`-elicited prompts give p < 0.05 against `lora_ctrl`. Of the teachable prompts, 12 of 15 (rule 3)
  or 13 of 16 (rule 7, the review's labels) pass; 169 (2 vs 0, p 0.25), 0 and 170 (4 vs 0, p 0.060) do not. The
  executor's 12 of 16 also fails 264, whose estimate is 2 vs 0 with the executor's share and 6 vs 0 (p 0.014) with the
  review's share of 1.0: "label-dependent". On the gated lower bounds alone (2–4 genuine reads against 0), no prompt
  reaches p < 0.05; the best is 4 of 64 against 0, p 0.060.
- **Constraint types never trained on:** `lora_main` elicits 19 of the 23 prompts whose binding id no training prompt
  carries, and 22 of the 25 whose id one does (the executor: 21 of 25; 189's id is covered). Of the 15 teachable
  prompts (rule 3), 10 have an uncovered binding id.
- **The four olmo-iffu prompts that survive 1,024 unhinted draws yet are hint-elicitable** (78, 133, 252, 253): elicited
  at rung 2 (4 / 4) and by `lora_main` (4 / 4; strict 42 / 62 / 49 / 26 of 64); `lora_ctrl` 0 / 4 (strict 0 / 1 / 0 / 0).
- **The six olmo-iffu test-3 survivors not elicited at rung 1:** 3 and 166 "neither"; 136 at rung 2; 211 at rung 3; 248
  and 269 teachable from RL's trajectories.
- **Lengths** on the 48 with no hint (thinking = response tokens through the first `</think>`; answer = words after
  the last; median of per-prompt medians / pooled median):

  | model | thinking tokens | answer words |
  |---|---|---|
  | `dpo` (olmo-ifomega fixed phase, 16 per prompt) | 1,834 / 2,260 | 231 / 253 |
  | `s2300` (fixed phase, 8 per prompt) | 5,742 / 5,811 | 31.5 / 32 |
  | `lora_main` | 6,236 / 6,392 (3.40× `dpo`) | 69.5 / 74 |
  | `lora_ctrl` | 1,963 / 2,045 (1.07×) | 235 / 246 |

  Mean thinking per draw, counting a missing `</think>` as the whole response: `lora_main` 6,986, `lora_ctrl` 3,616
  (the executor: 6,985 / 3,613). The review adds, observational and post hoc: hinted `dpo` at R2 / R3 thinks 4,357 /
  6,829 tokens and answers 68 / 68 words, "yet elicits 1 / 5 new prompts. Long thinking and short answers together do
  not, by themselves, give `lora_main`'s reach."
- **Calibration:** check jobs (16 per prompt) `lora_main` 0.963, `lora_ctrl` 0.894, both above the 0.5 × 0.906 guard;
  at 64 per prompt 0.975 and 0.895 (`dpo` unhinted 0.906).
- **Training and checks** (from the LoRA metas and training logs):

  | arm | training tokens | loss, first → last 4 steps | held-out NLL, `s2300` trajectories (`dpo` 0.638) | held-out NLL, `dpo` trajectories (`dpo` 0.762) | merge check (≤ 1 %) | held-out slice, strict of 128 (not hand-gated) |
  |---|---|---|---|---|---|---|
  | `lora_main` | 332,936 | 0.670 → 0.350 | 0.542 (−15.1 %) | 0.890 (+16.9 %) | 1.12 % (so both arms use the LoRA path) | 106 |
  | `lora_ctrl` | 235,880 | 0.741 → 0.407 | 0.643 (+0.7 %; rises) | 0.772 (+1.4 %) | 0.88 % | 51 |

  `lora_main`'s 15.1 % drop clears the 2 % guard, so no retraining. The control moves `dpo` far less (held-out NLL
  shifts of ≈ 1 % against 15–17 %), a caveat the write-up states. *[librarian]* The NLL and merge values are the run's
  recorded values; the review checks them against the pre-registered thresholds (they were not recomputed, which
  needs the 32 B model on a GPU).

### Gamed shares: the procedure decides the number

Pre-registered estimator: the first min(n, 4) strict successes of every reach cell, mechanical labels counted gamed.

| cell | the executor's gate (the pre-registered procedure; sanity cells only on its 31 / 26 / 23 gated prompts) | the review's gate (readers told rules 1–6) |
|---|---|---|
| R1 (`dpo`, olmo-iffu's reads) | 6 / 48 = 0.125 (7 / 48 with this run's screen) | 8 / 48 = 0.167 (the review's olmo-iffu labels; 9 ranks unread) |
| R2 (`dpo`) | 15 / 59 = 0.25 | 8 / 59 = 0.136 |
| R3 (`dpo`) | 7 / 21 = 0.33 | 4 / 21 = 0.190 |
| R4 (`dpo`) | 6 / 19 = 0.32 | 5 / 19 = 0.263 (6 / 19 under rule 7 on the 154 item) |
| **`lora_main`** | **35 / 164 = 0.21** | **12 / 164 = 0.073** |
| `lora_ctrl` | 1 / 5 | 1 / 5 |
| S2 / S3 / S4 (`s2300`) | 0.43 / 0.46 / 0.40 (over its 31 / 26 / 23 gated prompts) | 0.229 / 0.261 / 0.221 (over all its gated cells) |

The review on `lora_main`: "**differs 2.9×, by reader standard.** 0.21 comes from the pre-registered procedure
(readers without rules, executor confirmation); 0.073 from mine (readers given rules 1–6). Report the procedure with the
number." Expectation 8 holds under both. Among the review's limits: "My gamed shares are therefore not comparable with
the pre-registered procedure's; mine is the more lenient standard on minimal answers."

### The "neither" set and the `s2300` sanity arm

- **Same seven prompts in both gates:** `ifeval_3478`, IFBench 3, 67, 166, 188, 268, 284. 3 and 166 also had 0
  genuine in olmo-iffu's 1,024 unhinted draws. `lora_main` has 1 genuine of 64 on `ifeval_3478` and 166, 0 on the
  other five. Six of the seven are "label-robust"; 188's reading rests on the mechanical rule-1 labels of its 4 rung-4
  strict successes (one-sentence answers that copy the format of 188's rung-4 slot-1 exemplar, 189's `s2300` answer,
  which has no sentence punctuation).
- **Where a rung breaks `s2300`** (< 2 genuine of 16; DROP60 prompts before "·"):

  | rung | the executor's gate (prompts no rung up to that one elicited: 31 / 26 / 23) | the review's gate (all 48 / 31 / 26 cells) |
  |---|---|---|
  | 2 | 8: 0 1 3 4 166 188 · 64 189; also 19, outside the gate (0 strict of 16), "a certain break" | 5: 3 188 · 19 64 189 |
  | 3 | 6: 3 4 166 188 · 189 206 | 3: 3 4 166 |
  | 4 | 5: 1 4 166 188 · 206 | 1: 188 |

  The differences, by cause: R2 0 1 4 166 and R4 1 4 166, the reader standard (the executor's readers call
  keyword-salad or three-slogan answers gamed, the review's call them genuine; S2 4, S4 1 and S4 4 rest on 3–1 splits
  among the review's readers); R2 19, outside the executor's gate (a certain break in both); R3 188 and 189, the
  screen artifact; R3 / R4 206, rule 10, written after the `dpo` / LoRA cells were decided. "No reading changes"; the
  flags that move are those of 166 and 188 (DROP60, "neither"), 19 (a rung-1 prompt), and teachable or rung-4 prompts.
- **188 breaks `s2300` at rungs 1, 2 and 4, not 3.** At rung 3 the executor's screen called 6 of its 7 strict
  successes one sentence; 5 of those are multi-sentence by the literal test. Under the review's gate 3 are genuine
  (0/6 2–0, 0/10 2–1, and 0/14, genuine in both gates).
- **166 breaks at rungs 1 and 3.** At rungs 2 and 4 it has 2 genuine of 16 each, double-read 2–0. The review's rung-3
  break "rests on six single gamed reads".
- **3** is uninformative at rungs 1–3 and informative at rung 4, in both gates. **284 and 170:** the reviewer's olmo-iffu
  review read `s2300`'s rung-1 answers as 2 genuine of 4 strict each, so not broken (284's exact span copies are
  genuine under rule 5; 170 was reader-dependent there). 284's rung-1 flag and both prompts' DROP60 membership "rest on
  contested olmo-iffu labels". `ifeval_3478`, 67 and 268 are informative at every rung.
- So under the review's gate "every 'neither' prompt has at least one informative failed rung".
- **Expectation 12** (a rung breaks `s2300` on ≤ 3 (R2) / ≤ 5 (R3, R4) of the 40 non-DROP60 prompts): the executor's
  gate leaves it "not determined" (R2 ≥ 3, at the bound; 16 / 4 / 2 non-DROP60 cells outside its gate undecided), as
  stated. The review gates all 40 non-DROP60 R2 cells and finds exactly 3 (19, 64, 189), "under my more lenient
  labels": R2 held at the bound; R3 0 of 23 and R4 0 of 18 on the subsets that ran, held.

### The hand gate: the executor's, and the review's blind repeat

- **The executor's gate:** 706 run items, every label confirmed by the executor (24 overturned), a second blinded
  reader on every deciding item (251 items with ≥ 2 reader labels, 14 splits adjudicated by reading); 231 source items
  (exemplar answers and training trajectories).
- **The review's gate (final):** 1,148 reads; 317 of the 318 deciding label groups have ≥ 2 reads (the exception decides
  only S3 `ifbench_206`'s sanity flag); first and second readers agree on 379 of 397 double-read items, and the 18 splits
  are resolved by the majority of the first three reads, "several of them gamed". The reviewer made no adjudications; the one rule call (154) is
  shown both ways.
- **Agreement:** the executor's 706 items all join the review's draw keys. On the 607 draws both labelled, labels agree
  on 506. Of the 101 disagreements, 99 are executor-gamed / review-genuine and 2 go the other way. By binding id the 99
  are "mostly minimal or garbled answers to hard constraints": keyword counts 20, consonant clusters 19, three vowels
  19, equal-length sentences 12, alphabet 6, others ≤ 2.
- **The executor's arithmetic and walk are sound.** With the executor's labels and its own screen, the reviewer's walk
  reproduces every executor `dpo` / LoRA cell (R2 12, R3 5, R4 3, `lora_main` 40, `lora_ctrl` 0), the ladder and LoRA
  gamed shares exactly, and the sanity shares' numerators (their denominators differ by how screen-labelled draws are
  counted). "The differences come from labels, screens and scope."
- **The protocol differences:**
  1. reader instructions: the executor's readers saw only the reader prompt's genuine / gamed definitions, as
     pre-registered; the review's were given rules 1–6, including rule 4 (thin answers that serve the request are
     genuine) and rule 3 (tokenisation slack);
  2. confirmation: the executor confirmed every label under rules 1–6 and its rules 7–12 (24 overturns); the reviewer
     confirmed none;
  3. splits: the executor adjudicated them by reading; the review used third readers;
  4. sanity scope: the executor gated only the prompts no rung up to and including that one had elicited (31 / 26 / 23);
     the review gated every sanity cell;
  5. screens: both rule-1 screens had a blind spot. The executor's needs whitespace after the mark
     (`[.!?]["')\]]*\s+\S`), so `.</i>` and `.<br>` hide sentence breaks; it calls 10 multi-sentence answers "one
     sentence" (`lora_main` 189 ×2, R2 / R4 / S3 189 ×1 each, S3 188 ×5). The reviewer's phase-1 screen missed
     breaks after an emoji or an HTML tag (29 labels; fixed in phase 2).
- **Where the label gap comes from:** 92 of the 99 one-way disagreements are labels the executor's readers gave and the
  executor kept; 7 are executor overturns. The executor's notes on the 99 cite rule 7 twice, rule 10 five times, rule 11 twice and
  rule 9 never. "So the gap is mainly the two reader pools' standard for minimal answers."
- **Robustness:** with the final labels, first-read-only and pessimistic labels (genuine only if every read says
  genuine) change only 6 sanity cells (first-read only: S4 4; pessimistic: S2 4, S2 133, S3 188, S3 217, S4 4, S4 1);
  no `dpo` or LoRA cell is affected. Strict counting without the gate differs from the review's labels on 6 cells (R4
  188, S2 189 and S4 188 by the screen; R3 189, S3 4 and S3 166 by readers).

### Compute per arm (all one NVIDIA H200, community)

| arm / job | samples (= checker calls) | generated tokens | wall GPU-s (the review; failed starts included) | training |
|---|---|---|---|---|
| stage 0 (S0r2 + S0r3, `dpo`) | 464 | 2.28 M | 1,639 | – |
| R1cal (`dpo`, calibration) | 512 | 0.73 M | 396 | – |
| R2 (`dpo`) | 3,712 | 16.25 M | 8,760 | – |
| R3 (`dpo`) | 2,624 | 15.00 M | 8,420 | – |
| R4 (`dpo`) | 2,304 | 11.89 M (prompt tokens 1.35 M, few-shot) | 6,664 | – |
| S2 / S3 / S4 (`s2300`) | 928 / 656 / 576 | 5.24 / 4.07 / 3.82 M | 2,943 / 2,524 / 2,553 | – |
| `lora_main` train + finish | – | – | 380 (failed save) + 117 | 64 steps, 332,936 tokens |
| `lora_ctrl` train + finish | – | – | 291 (failed save) + 109 | 64 steps, 235,880 tokens |
| Lchk_main / Lchk_ctrl | 288 / 288 | 1.46 / 0.89 M | 13 (failed) + 1,222 / 11 (failed) + 635 | – |
| Leval_main / Leval_ctrl | 3,552 / 3,552 | 24.63 / 14.33 M | 12 (failed) + 15,777 / 8,564 | – |

- Sampling total: 19,456 samples, 100,591,013 generated tokens, 16.70 GPU-h of job wall time. Training: 128 steps,
  568,816 tokens. Lean checks: 0. The registry's 103 rows sum to 60,697.8 GPU-s (16.86 h; *[librarian]* the jobs' load
  + generation time plus the training time, by the review's per-job figures), and the meta token counts equal the
  review's row sums. Pod time is 17.23 GPU-h.
- **Matched axes:** rungs on attempts (64 per prompt); the LoRAs on examples (16), steps (64), prompts and evaluation
  attempts (64 per prompt). Tokens per draw on common prompts: R2 / R1 1.18×, R3 / R2 1.20×, R4 / R3 0.88× (under the
  1.25× line).
- **Flagged (> 1.25× the comparator), by both the run and the review:** `lora_main` against `lora_ctrl`: training
  tokens 1.411×; training time 1.273× (train + finish; 1.367× for training alone); check + evaluation generated tokens
  1.715× and GPU-seconds 1.855× (evaluation alone 1.72× / 1.84×). The review: "`lora_main` thinks longer"; the
  executor: "the flagged axes are what `lora_main` learned". The arms are matched on attempts.

## Expectations against outcomes

All 13 were committed before any pod (`48840906`, 00:08:34 < pod ready 00:41:16). The review's scoring, with its
phase-2 corrections (rule 3 unless named):

| # | expected (80 % interval) | observed | verdict |
|---|---|---|---|
| 5 | `lora_main` 4 of 48 (0–10); `lora_ctrl` 2 (0–6); most prompt-elicitable anyway | 41 (40 with the executor's screen), 26 of them prompt-elicitable; 0 | `lora_main` **missed** (far above); `lora_ctrl` and "most prompt-elicitable" held |
| 6 | teachable from RL 1 (0–4); any light fine-tune 0 (0–2) | 15 (16 under rule 7); 0 | **missed**; held |
| 7 | neither 21 (13–28); of the 6 test-3 survivors not elicited at rung 1, 4 (2–6) neither | 7; 2 (3, 166) | **missed**; held |
| 10 | `lora_main` answer words 140 (60–220); `lora_ctrl` ±25 % of `dpo`; both LoRAs' thinking ±35 % of `dpo` | 69.5; +1.8 %; `lora_main` 3.40×, `lora_ctrl` 1.07× | words held; **`lora_main`'s thinking missed** |
| 11 | the four survivors: ≥ 3 of 4 at rung 2 (2–4); 0 of 4 by either LoRA (0–2) | 4; `lora_main` 4, `lora_ctrl` 0 | rung 2 held; **the LoRA part missed** |
| 1 | rung 2 elicits 3 (0–7) of the 32 | 1 | held |
| 2 | rung 3 elicits 4 (1–9) of the rest | 5 of 31 | held |
| 3 | rung 4 elicits 3 (0–8) of the rest | 4 of 26 (3 under rule 7) | held |
| 4 | prompt-elicitable 26 (19–33) | 26 (25 under rule 7) | held |
| 8 | gamed share: R2 = R1 + 5 pp (−10 to +20); R3 0.15 (0.05–0.40); R4 0.12 (0.03–0.35); `lora_main` 0.10 (0–0.35) | +1.1 pp against the pre-registration's R1 (0.125), −3.1 pp against the review's own olmo-iffu R1 (0.167); 0.19; 0.26; 0.073 (the executor's labels: +12.9 pp; 0.33; 0.32; 0.21) | held (all four, under both label sets) |
| 9 | calibration ≥ 0.75 at every rung; ≤ 2 prompts below half; LoRAs ≥ 0.8 × `dpo` unhinted | 0.777 at the lowest; 1 prompt (298); 0.975 / 0.895 against 0.725 | held |
| 12 | a rung breaks `s2300` on ≤ 3 (R2) / ≤ 5 (R3, R4) of the 40 non-DROP60 prompts | 3 of 40 (the review's more lenient labels); 0 of 23 run; 0 of 18 run | R2 held at the bound; R3 / R4 held, scorable only on the cut-2 subsets (the executor: "not determined") |
| 13 | stage 0: R2 4.5k (3–7k), R3 5k (3–8k) tokens per draw | 4,449; 5,361 | held |

The executor: "Every miss is one direction: 16 RL trajectories teach far more than expected." The review: "the same
verdicts under rule 7 … misses are reported as misses."

## The review's corrections to the executor's files

The fork's files are uncorrected; the copy of the plain summary here applies its corrections. By file:
- **`run_olmo_ifladder.md`:** finding 1's "Check line +1, method hints +5 …, exemplars +3" needs the detection limit;
  finding 2's "reach 40 of 48" is 40–41 (41 with the literal screen) and "Fisher p < 0.05 vs the control on 12" is
  label-dependent (13 of 16 under rule 7 with the review's labels, 12 of 15 under rule 3); finding 3's "Gamed share
  0.21" is the pre-registered procedure's figure (0.073 with readers given rule 4); finding 4's "on 166 and 188 every
  rung breaks `s2300`" is not supported; the reading table needs the 154 call; the caveats need "one fine-tune per arm";
  the deviations line needs rules 7–9 and the screen's blind spot.
- **`numbers.md` § olmo-ifladder:** "olmo-transplant's 10 calibration prompts" (the review: "that file has 12 (226 and
  252 are reach prompts here)"; *[librarian]* its `set` column has 10 `calib` rows and 2 `reach_no_trace` rows, 226 and
  252); the readings table, the ladder's rung-4 row and expectations 3, 4 and 6 need the 154 call (26 / 4 / 15 under
  rule 3); the ladder's new-elicitation counts need the detection limit; "Rung 2 re-elicits 11 of rung 1's 16" (11 or 12, label-dependent); the Fisher list "not on 0 …, 170
  …, 169 …, 264" (264 passes with the review's share); "21 of the 25" (22 of 25); `lora_main`'s "40 / 48" (40–41) and
  gamed share 0.21; the "Neither, flagged" paragraph's "166 and 188 at rungs 1–4 (only the LoRA test is informative),
  284 at rung 1 only"; the sanity table's breaks and gamed shares (the review's gate above). Rules 7–9 are named in the
  method line but not listed as a deviation.
- **`olmo_ifladder/plain_summary.md`:** "About one pass in five still gamed the checker"; "On two of them … every added
  instruction also trips up the RL model, so only the fine-tune test informs"; the rung counts without the detection
  limit; "40 of the 48"; 25 and 16 without the 154 call (26 / 15); no "trained once" limit; "34 % of tries pass"
  (automatic strict passes, not genuine rates). Applied in the copy here.
- **`figures/olmo_ifladder_ladder.png`:** *[librarian]* the review does not discuss the figure; its findings applied to
  the figure are under [Figures](#figures).
- *[librarian]* The same claims sit in `STATUS.md`'s DONE line ("on 166 / 188 every rung also breaks `s2300`"; 40;
  "Fisher p < 0.05 vs the control on 12"), in `log.md`'s 18:56 entry ("166 and 188 at rungs 1–4 (only the LoRA test
  informs)"), in `olmo_ifladder/classify.csv` (`rungs_breaking_s2300` "R2 R3 R4" for 166 and 188; `main` "no" for
  189) and `summary.json`, and in the executor's pre-publication claim checks (`olmo_ifladder/claim_check_numbers.md`,
  "166 and 188 at rungs 1–4"; `claim_check_writeup.md`, "166 and 188 break `s2300` at every rung").

## Deviations from the pre-registration and the brief

- **Cuts 1 and 2 at stage 0** (by the pre-registered rule; QUESTIONS.md entry with a default): no `s2300` sanity arm,
  and rungs 3–4 only on prompts no earlier rung elicited.
- **Deviation 1: the sanity arm restored** at 04:40, after rung 2 ran faster than stage 0 had measured (re-cost ≈ $61
  with it); cut 2 kept.
- **Deviation 2: the LoRA path.** The merge check failed narrowly (1.12 % > 1 % for `lora_main`; 0.88 % for
  `lora_ctrl`), so both LoRAs were evaluated through vLLM's LoRA path, as the pre-registered fallback says. The training
  process's merged save had failed on the checkpoint's generation-config validation, so the NLL checks, merge and report
  were redone by `lora.py finish` from the saved adapters, with no retraining (the review: "the training code path is
  unchanged").
- **Hand-rule addenda:** rules 7–9 (committed 02:07:50, after the pre-registration and before any ladder item was read)
  and rules 10–12 (after every `dpo` and LoRA cell was decided; by the executor's log, 5 labels changed, one in
  `lora_main`'s 206 cell, and no `dpo` or LoRA reading changed). The write-up lists 10–12 only; the review:
  "incomplete: list rules 7–9 too".
- **The rule-1 screen's blind spot** (`.</i>`, `.<br>`): the review asks that it be listed; it moves `lora_main` to 41
  / 48 and 188's flag.
- **Second reads** had been skipped for the rung-2 sanity cells and for the rung-2 cells on already-elicited prompts;
  the executor did them for all three rungs before the write-up (log, 18:56).
- *[librarian]* **The sanity-gate scope** (only the prompts no rung up to that one elicited) departs from the
  pre-registered gate, which names the sanity cells without restriction. The executor set it during the run (log,
  14:38) and states it in the write-up's deviations line ("read only on prompts no rung had elicited"); the
  pre-registration's deviations section does not record it, and the review lists it as a protocol difference.
- **Stated writing times** later than their commits (minor): the pre-registration's "Written 2026-11-09 00:10–01:00
  UTC" (written 00:07:49, committed 00:08:34), `interventions.py`'s "00:40–01:30" (written 00:01:09) and the
  addendum's "02:10" (committed 02:07:50). The order of events is as claimed.

## Interpretation (labelled; not tested further here)

- **The pre-registered readings:** 25–26 of the 48 are prompt-elicitable ("disposition", in the proposal's words; 16 of
  them by the 60-word line alone), 15–16 are teachable from RL's trajectories ("teachable, not latent", in
  capability-defs' terms, as proposal 41 puts it), none is teachable by any light fine-tune, and 7 are reached by
  neither, "the strongest candidates for RL-created capability at this scale".
- **The executor's reading** (plain summary): "On these prompts, most of what RL reached can also be reached by
  fine-tuning on a handful of RL examples, though less reliably (34 % of tries pass, against the RL model's 66 %). The
  copied habits suggest a teachable way of working; whether knowledge transfers too was not tested." (The review on
  the 34 % and 66 %: "automatic strict passes, not genuine rates".)
- **The review's reading:** the headline "holds under an independent blind hand gate", but "what transfers" was
  measured only through lengths and NLL, with no mechanism test; the control moves `dpo` far less than `lora_main`
  does; and the comparison is one training run per arm. Hinted `dpo` thinks long and answers short at rungs 2–3
  without `lora_main`'s reach, so long thinking and short answers do not by themselves give it (observational, post
  hoc).
- *[librarian]* Against olmo-transplant on the same models: handed `s2300`'s successful thinking in its context, `dpo`
  wrote a genuine answer 0.14 of the time against `s2300`'s 0.92, while 16 such trajectories as training data give a
  `dpo` LoRA ≥ 2 genuine of 64 on 40–41 of 48 prompts. The two runs measure different things (a per-answer rate after
  supplied thinking, against elicitation at 64 draws with the model's own thinking), so this is a juxtaposition, not a
  comparison.

## Limitations

- **One training run per arm:** one draw of 16 trajectories, one slot order, one seed. The per-prompt Fisher tests treat
  the 64 evaluation draws as the sample and cannot see training-run variance; "There is no seed-variance data in this
  run."
- **The control is matched on examples, steps and recipe, not on how far it moves `dpo`** (held-out NLL ≈ 1 % against
  15–17 %), and `lora_main` trained on 1.41× the tokens. A stronger control might reach some prompts.
- **Readers are Claude sub-agents, and the reader standard matters:** the two gates disagree on 101 of 607 shared draws;
  the 99 one-way ones are "mostly minimal or garbled answers to hard constraints". This moves `lora_main`'s gamed share
  from 0.21 to 0.073, several sanity flags, and two `dpo` cells: R2 `ifbench_179` (no reading depends on it) and R4
  `ifbench_154` (the rule-3 vs rule-7 call). The review's gamed shares are "not comparable" with the pre-registered
  procedure's; its standard is "the more lenient" on minimal answers.
- **The detection limit:** with 64 draws, a prompt with a genuine rate of 1–5 % is classified by chance; rung-by-rung
  differences of ≤ 4 prompts are not findings.
- **Rung 1 is olmo-iffu's**, read by a different reader pool (the review's figures use its own olmo-iffu labels).
- **Cut 2:** rungs 3 and 4 ran only where no earlier rung elicited, so a prompt elicited earlier has no rung-3 or rung-4
  data (rung 2 and both LoRAs ran on all 48).
- **The review's own limits:** it did not hand-read the 116 rung-4 exemplar answers or the 16 held-out trajectories
  (hashes and strict grades only); one deciding item (S3 206) has a single read; 13 screen-released sanity draws lie
  beyond their walks and are unlabelled.
- **Scope:** one model family, 48 prompts, IFBench and IFEval; family counts (n = 1–17) are descriptive.

## Next steps (the review's "what would settle the rest"; not launched)

1. **A second training draw for both arms:** a new seed and slot order, ideally also a second disjoint set of 16
   trajectories ("what the two-seed rule asks for"). Like for like (48 × 64 draws per arm) ≈ 6.3 H200-hours, "about $24
   with pod set-up"; halved to 48 × 32 ≈ $13.5, but then the "≥ 2 of 64" threshold must be re-derived.
2. **A stronger control:** a `dpo`-trajectory LoRA trained until its held-out-NLL shift matches `lora_main`'s.
3. **An `s2300` LoRA on answers only, or on non-IF trajectories,** to separate the thinking style from the IF content.
4. **For the gamed share and the "neither" flags:** fresh readers applying one written standard (rules 1–9, given to
   them) to the 101 disputed draws, to the `s2300` sanity cells of 166 and 188, and to the 154 / 179 items.

*[librarian]* Measurement 4 needs no pods. Dan's blind hand-labelling set
([`ifbench-hand-label-calibration`](../../docs/writeups/2026-11-09-ifbench-hand-label-calibration/README.md), 50 contested
answers from olmo-ifomega, olmo-iffu, olmo-transplant and olmo-rlzero-if) has no olmo-ifladder item; this run's 101
disputed draws are the same kind of item. The review gives no cost for measurements 2 and 3.

## For the runs that build on this one

**olmo-answeronly** (proposal 45; *[librarian]* running on agent2 since 11-10 00:22:32 UTC) branches from the fork's
`origin/dan_olmo-ifladder` and has an optional arm with `lora_main`, whose brief says to check this review's verdict
before relying on its labels.
- *[librarian]* Its branch starts at `de1ba32e`, the review's phase-1 commit: its copy of `review_olmo-ifladder.md` has
  §Recount only, no §Compare and no §Verdict. The phase-1 statements that changed are listed in the final review's
  "Corrections to §Recount" (among them `lora_main` "40 / 48", "70 vacuous successes", "188 at R1–R4", "the other four
  are label-robust", "all resolved genuine 2–1", 36 hints and the sign of `lora_ctrl`'s NLL change). Its kit's
  label files are the phase-1 ones too: `out/my_labels.json` has 685 label groups (719 at `6ff22000`), and
  `out/mech_labels.json` 70 screen labels (41 now). For example `lora_main`'s 189 answer 1/11, genuine 2–0 in phase 2,
  is gamed by the phase-1 screen there and has no entry in that `my_labels.json`. The final review and labels are at
  `6ff22000` on `origin/dan_olmo-ifladder`.
- The verdict is "Stands": `lora_main` elicits 40–41 / 48 against the control's 0, under both gates. It is one training
  run.
- The adapter: `hf://buckets/dan-pandori/nd-rl/olmo-ifladder/artifacts/olmo_ifladder/lora/main/` (sha256 `9921a760…`).
  It was evaluated through vLLM's LoRA path, since the merge check failed (1.12 %).
- **Labels for `lora_main`'s cell:** the executor's are in `olmo_ifladder/handcheck.csv`, the review's in
  `review/olmo-ifladder-recount/out/my_labels.json`. *[librarian]* (the review's `out/compare.txt` [2]): both read 162
  of its items and agree on 137; of the 25 disagreements, 24 are executor-gamed / review-genuine and one is the reverse
  (`ifbench_203` draw 0/14, which the executor adjudicated genuine after a reader split). An executor-genuine
  `lora_main` trace is therefore review-genuine in every shared case but that one.
- *[librarian]* The 16 training and 8 held-out prompts are not among olmo-transplant's 41 reach prompts. But
  `ifbench_281`, which is one of those 41 and not one of this run's 58, was outside both the executor's and the
  review's base-request overlap checks (training × the 58). Check it before using `lora_main` on 281.

**Proposal 42** (`olmo-support`; not briefed) takes "the prompts that survive 41". The "neither" set is the seven above
under both gates. *[librarian]* Their flags depend on the gate: under the review's, every "neither" prompt has at least
one informative failed rung; under the executor's, 166 and 188 have none. The review asks fresh readers to settle the
`s2300` sanity cells of 166 and 188, and 284's rung-1 flag and its DROP60 membership rest on contested olmo-iffu
labels.

## Figures

[`olmo_ifladder_ladder.png`](olmo_ifladder_ladder.png) (the executor's figure, unchanged): the 48 reach prompts as rows,
rungs 1–4 and the two LoRAs as columns; a cell shows the hand-gated genuine answers among its strict successes ("2+" =
elicited); hatching marks a rung that also breaks `s2300`. It uses the executor's labels and rule 7. *[librarian]* The review does
not discuss the figure; applied to it, the review's findings mean:
- **`ifbench_189`, LoRA s2300 column:** "1" is "2+" with the literal pre-registered screen (`lora_main` 41 / 48). The
  row stays "prompt-elicitable at rung 4".
- **`ifbench_154`:** under rule 3 its rung-4 cell is "2+" and the row moves from "teachable from RL's trajectories" to
  "prompt-elicitable at rung 4"; the figure shows the rule-7 reading. Say that this is a call on one item.
- **The hatching** follows the executor's sanity gate. In the review's gate, rung 2 does not break `s2300` on 0, 1, 4 or
  166; rung 3 does not on 188, 189 or 206; rung 4 does not on 1, 4, 166 or 206. The rung-1 hatching and the ‡
  (DROP60) marks of 284 and 170 rest on contested olmo-iffu labels. So the footnote's "the rung also breaks s2300 there"
  holds only under the executor's gate: the figure hatches 166 and 188 at every rung, while under the review's gate 166
  breaks `s2300` only at rungs 1 and 3, and 188 only at rungs 1, 2 and 4.
- **`ifbench_179`, rung 2:** "0" is "2+" with the review's labels (label-dependent; no reading depends on it).
- A note rather than a correction: **`ifbench_290`** is placed at rung 1 (rule 6's primary reading); on the strict
  reading it is rung 2, as `numbers.md` says (the review: "reproduces").
