# CLEANUP — the clean paper and blog (run paper-clean, 2026-10-14)

**Starting point.** The marked reading copy: nd-rl `origin/dan:docs/papers/create-or-elicit/` at `7709669`, copied
unchanged into `paper/` in this branch's commit `11d07d13`. Every change this run made is
`git diff 11d07d13 -- paper/`. The marked copy keeps its `\rv{}` / *[review]* marks and is the comparison base below.

PDF numbering used here (file → section): 1 `01_intro`, 2 `03_setup`, 3 `05_results`, 4 `04_measurement`,
5 `02_related`, 6 `07_discussion`, 7 `08_limitations`, 8 `09_repro`; Appendices A–J are `10_appendix`, with
App. F = power sampling and App. I = `11_checks`.

## 1. The *[review]* marks, folded

**Paper.** All 60 `\rv{}` marks removed (abstract 5, intro 9, related 1, setup 1, measurement 4, results 17,
discussion 6, limitations 9, repro 1, checks 7). The `\rv` macro, the reading-copy box and the reading-copy date line
are gone from `main.tex` (date now "Draft of 2026-10-14. Not for circulation."). Each mark sat at the start of a
corrected span, so removing it left the correction's text unchanged. Three sentences were smoothed, with the meaning kept:

| where | marked copy | clean copy | why |
|---|---|---|---|
| §7 Settings | "Doubling the caps moved each strict count by at most one theorem …, but cut the cut-off share only 1.0–2.2-fold" | "With doubled caps, each strict count differed by at most one theorem …, but the cut-off share fell only 1.0–2.2-fold" | "moved" implied a paired cap effect, which the next (review) sentence denies (P1) |
| §8 | "Post hoc are A2, A5 and Addendum 5 (…), as are …" | "A2, A5 and Addendum 5 (…) are post hoc, as are …" | word order |
| §3.1 | "Holm is the conservative reading, and the review applied it too." | "… and the experiment's review applied it too." | the paper has two reviews; the review's P10 row says it is the create-or-elicit review |

**Blog.** All 31 *[review]* marks folded and the reading-copy note removed; the blog was then rewritten for length
(§4 below). Every reviewed correction (B1–B8 and the echoes of P2, P5, P13, P14) is kept; the checklist is in §4.

## 2. "Possible errors outside the review's list": each checked against the data

Evidence scripts and outputs written by this run are in `artifacts/paper_clean/`. No pod was used.

| # | the librarian's note | outcome | evidence | change |
|---|---|---|---|---|
| 1 | intro defines σ "for a positive goal G"; Table 1 has no positivity condition | **fixed** | Table 1 and the knockout's detector (`coe/coe_sigma.py`, `patterns.reductio` l.105–116) test only that the assumption is ¬G for the DN line's G. In 𝒟 (`train_k12.jsonl`), 1,721 of the 16,703 σ records apply σ to a negated G (1,585 only to negated goals; 1,741 of 17,387 σ lines): `artifacts/paper_clean/sigma_goal_polarity.{py,json}` | §1: "to prove a goal G". The blog's "positive statement" describes the free positive reductio and is right |
| 2 | intro: the 123 strict training theorems "require the composition or a variant", but the lemma route (not σ⁺) can prove strict members | **fixed** | Table 1: strict members need a built ¬¬ eliminated "by a free positive reductio or by the lemma route", and the lemma route is "classical, but not σ⁺". Knockout s1's r20 proofs of strict members: σ 32, other σ⁺ 24, lemma route 7 (§3.5 l.244; `numbers.md` l.1930); `review_create-or-elicit.md` l.162 | §1: "problems that need the composition, a variant or the lemma route of Table 1 (here 123 of 4,495)". Blog: "123 training theorems that need a genuinely classical step" |
| 3 | intro's C4 sentence omits the earlier early-stopped successes on `1246` and `2256` | **fixed** | `paper/data/c4_within_reach.json` (ko_s1): C4 gives 0 / 18,432 on `1019`, `1246`, `2256`; the earlier early-stopped sample had `1246` 1 / 7,168 and `2256` 1 / 9,216 (`1019` none). §3.5 already says so | §1 adds "though two of those three had one success each in an earlier, early-stopped sample" |
| 4 | the abstract's last sentence states an interpretation ("EI amplified …") as a finding | **fixed** | sub-audit `review_coep/phase2/front.md` A13 and item 15: no certificate or compute-matched set exists for the full arm's strict gains; A3/A3b are exploratory | abstract: "In this organism, seeds pretrained with the composition acquired classical families, and within 16 rounds of EI seeds without it acquired none." (same length; abstract stays 250 words; "under EI" was dropped after the equivalence check, M1, because the full seeds' families were largely present before RL). The blog's matching sentence is reworded to a descriptive contrast too ("this form of RL learned classical families when pretraining had exposed the move, and did not, within 16 rounds, learn them without it") |
| 5 | §3.8: the "150 re-submitted Lean rejections" were pre-filter (gate) rejections | **fixed** | the 150 are the `ctl_leanrej` control of `review_coe/recheck_collect.py` (the reads' `LEANREJ` examples, l.97–107). All 1,307 gate log lines in `artifacts/coe/logs` and `artifacts/coe_paper/logs` read "rej N (filter N, prefilter=on)": 12,217,252 rejections, every one the pre-filter's (`artifacts/paper_clean/gate_rejections.{py,json}`) | §3.8: "all 150 re-submitted texts that the gate had rejected (all by its pre-filter)" |
| 6 | "three to four times rarer" is 4.2× for DN and 2.9× for ¬-introduction | **fixed** | DN records 13.73 % → 3.29 % (4.17×), ¬-introduction 15.21 % → 5.19 % (2.93×): §3 "Data", `review_create-or-elicit.md` l.51, `numbers.md` l.1816–1817 | abstract "each 2.9–4.2 times rarer"; §1, §3 Data, §7 give 2.9 and 4.2 by rule; Fig. 0's label and the blog likewise. §5's "a quarter to a third of their former frequency" (0.24, 0.34) left: the sub-audit (R23) checked it as right |
| 7 | knockout s5's base also proved `1142` (Peirce sequent); knockout s2's x1 draw also proved `1096` | **fixed** | `review_coep/members.log`: base s5's early-stopped strict successes `1142` 1 / 3,072 and `1811` 1 / 13,312 (also `review_coe/cert.log` l.51; `data/coe/families.json`: `1142` is a Peirce-sequent strict member). Ladder s2 at r16: x0 {`1963`}, x1 {`1096` ×2}, C1 std {`1096`}, C1 dbl {`1096`, `1963`} | §3.6: s5's base "proved `1811` and the Peirce-sequent member `1142`", the ladder proves "neither `1811` nor `1142` in any" draw; s2's ladder proved `1096` "in the other three plain r16 draws". The abstract's and intro's counts already included `1142` |
| 8 | App. I's opening implies batch 512 for every read (C4 ran at 1,024); its script list omits A3b and A5b | **fixed** | all 29 C1–C3 reads (`artifacts/coe_paper/eval/*.args.json`): batch 512, k 256, T 0.8. C4 (`artifacts/coe_paper/c4/ko_s{1,3}.json.args.json`): batch 1,024. A3b: `paper/scripts/a3b_member_prior_gained.py`; A5b: `paper/scripts/round2_extra.py` (l.69–105) | App. I C4 paragraph states batch 1,024 for both draws (seed 3's added after the equivalence check, M3); the A2–A5 script list adds both scripts |
| 9 | §6: "As there, the base rate predicts which seeds move" has P11's problem | **fixed** | `paper/data/a5_waiting.json`: rates s0 0, s1 5.4 × 10⁻⁶ (moved, r9), s2 7.5 × 10⁻⁷ (moved, r14), s3 0, s4 2.2 × 10⁻⁶ (never), s5 1.4 × 10⁻⁶ (moved, r14) | §6: "As in the required pools, a zero base rate (on the 105 family members) predicted no movement (seeds 0 and 3) and the highest rate the most (seed 1); unlike there, seed 4, with the second-highest rate, never proved a strict training theorem, while seed 2, with the lowest non-zero rate, did" (the sub-audit D57's wording; the pool and "the required pools" added after the equivalence check, M2: seed 3's base proved `la_transfer_916`, so its rate is zero only on the 105) |
| 10 | §6: "a few theorems out of hundreds" understates the general-ability cost | **fixed** | sub-audit D17 and results row 138: the denominators are 225 and 53 (72), not hundreds; the paper's §3.7 numbers: at r0 −4.3 of holdout250's 225 and −4.0 of textbook72's 53 intuitionistic theorems, −2.5 of the 53 at r16 | §3.7 (interpretation paragraph) and §6 now give those numbers instead of "out of hundreds"; the blog never said "hundreds" |
| 11 | §5: "mastery is not sufficient" lacks "within 16 rounds" | **fixed** | sub-audit R23; the review's rewording 1 ("not acquired by this EI within 16 rounds") | §5: "mastery is not sufficient for EI at this scale within 16 rounds" |
| 12 | the power-sampling positive control's comparator solves (6/17, 13/17) "also kept no proof text" | **changed differently: the note is wrong** | `coe/power_sample.py` (`run_plain`) counts an attempt only if the Lean gate accepts it, and from 2026-10-12 stores the accepted proofs. `artifacts/coe/power/p2b_pc_plainT0.8.json` stores proofs for all 6 solves (8 distinct), `…T1.0.json` for all 13 (15 distinct), `p2_ko_s1_plainT1.0.json` for its 1. Only the comparators of (earlier) full s0, s1 and knockout s0 store none. Offline re-check, as `coe/coe_recheck.py` does it (nd2lean, Lean 4.34.1, `lean_judge.judge_many`): **24 / 24 accepted, 24 / 24 rejected with the goal negated**: `artifacts/paper_clean/power_comparator_recheck.{py,json}`. The sub-audit's "p2b_pc_plainT*.json rows have no text field" (L35) is wrong | the paper's own "the power-sampling comparators kept no proof text" over-generalised. §7 now names the three comparators without stored proofs and says the stored ones, the positive control's included, all re-check; App. F reports the 24 / 24 and 24 / 24 (*this paper*) |
| 13 | blog: "hand-picked"; "Ours is the simplest kind" and "routinely" not in the paper; the power-sampling paragraph names no models; Fig. 2's alt text and PNG title say "each model" | **fixed** | §4 below; power sampling ran on the bases of earlier full s0, s1 and knockout s0, s1 (`ckpt` fields of `artifacts/coe/power/p2_*.json`) | "chosen in advance by a fixed rule"; "Ours fine-tunes on its own successes (rejection-sampling fine-tuning)" (§7 "rejection-sampling fine-tuning"; not "only", since EI also replays); "thousands of accepted proofs" kept, "routinely" gone; "proved none of RL's new solves (two earlier full runs' bases) or of the 106 held-out classical theorems (two knockout bases)" (corrected after the equivalence check, must-fix 1: the knockout bases were sampled on the 106, not on RL's new solves); alt text "Log-probability of known σ proofs along RL, 12 of 15 runs" (the caption keeps "for 12 of the 15 runs") and the PNG title "How likely 12 of the 15 runs are to write known σ proofs" (`scripts/figures.py`, figure regenerated) |

Two known defects in this branch's own files (the librarian's README) are fixed too: `LEDGER.md` P8 is marked
superseded by C4, and `next_steps.md` item 1b (and the per-base estimate after it) is re-costed at C4's measured
12.35 ms per attempt (2.4 × 10⁷ attempts ≈ 82 A40-hours ≈ $48), each with a dated note.

## 3. textbook72

**Change.** §2's family list printed the negated-conditional schema (not repeated here), which is in the renaming
class of the sealed problem `textbook_b98931f2…`. It now reads "negated conditional (a negated implication entails its
antecedent and the negation of its consequent)".

**Check of everything printed.** `artifacts/paper_clean/tb72_sealed_check.py` (output `.txt`) uses the `--sealed` method
of nd-rl `dan_shared_proof_state` `code/tests/fixtures/proof_state/make_golden.py`: `gen.canon_key` of
"premises |- goal" against the 72 textbook72 prompts. It adds a premise-order-invariant key, because `canon_key` is
premise-order sensitive and misses premise-swapped twins. Eight printed theorems were checked: the three remaining
family schemas (converse contraposition, Peirce sequent, De Morgan); ¬¬G → G of Table 1's lemma route, read as a
theorem; `la_transfer_916` (Fig. 1's listing and caption, App. listing (b), the blog's text and Lean block,
`lean/blog_example.lean`); `la_transfer_1246` (App. listing (a)); the goal-negated control
`lean/blog_example_negctl.lean`; and the blog's "P or not P". **0 of 8 are in a textbook72 class under either key.**
Positive control: the removed schema hits `textbook_b98931f2…` under both. Figures carry no theorem: the only
formula text, in Fig. 0, is the σ recipe. IDs (`textbook_3ed45280…` in §7) are printed only as IDs.

## 4. The blog

**Length: 2,487 words** by the pre-registered count (`wc -w` after removing the YAML front matter; the Lean block,
alt texts and captions are counted); 2,493 by `wc -w` on the whole file. By the same count the marked blog was 3,080
words with its 31 marks and 3,049 without them, excluding its 63-word reading-copy note (3,143 and 3,112 including
it).

**Reviewed caveats kept** (each is in the clean blog):
- B1: the post hoc 5 / 6 vs 0 / 6 is labelled "a comparison we chose after seeing the data" and sits next to the
  pre-registered 6 / 6 vs 0 / 6, "mostly decided before RL".
- B2:
  - the tree-search pre-set test failed on the hardest problems;
  - the rule could not return "created";
  - only converse contraposition passes "needs exposure", three families undecided after correction;
  - the guided base's 3 / 4 / 1 of 22 / 21 / 14;
  - the hardware-matched 2 / 3 vs 0 / 6 (p 0.034);
  - the older-checker label on the earlier models (§7);
  - "Checks": doubled limits, Lean re-checks, and the filter over 210,151 texts;
  - the replay-only control's 13–25 %.
- B3: the drop rule (at most one of three named theorems at ≥ 2 successes in 18,432; one did).
- B4: the proofs are the earlier full runs'; dose and knockout models 0.7 units apart prove 58 and 2.
- B5: "the standard way to improve reasoning models"; "restored the learning (… exploratory)"; no "easy".
- B6: Fig. 2 "for 12 of the 15 runs".
- B7: models named (the three earlier full runs; base and after 8 rounds).
- B8: classical move, family, model organism, budget, natural-log units, early-stopped, "hardest" (§6 text) and
  "long" (Fig. 4) defined at first use.
- Echoes: the certificate's temperature and the missed five-of-six expectation (P2); "unseen instances that require the
  move" (P5); "four, counting only family members" (P13); "and its close variants" (P14).
- The brief's four:
  - scope, "this form of RL … did not, within 16 rounds, learn them without it" and "One RL algorithm, 16 rounds";
  - the sampler-specific power-sampling null, "a failure of our sampler";
  - rare-not-absent (§4, and the short version);
  - power limits, "Six runs per main condition …".

**Other blog fixes, so that it claims nothing the paper does not:**
- the "Checks" line said doubling "moved each count by at most one theorem". The paper (P1) says the reads are
  independent draws, so it now says "fresh evaluations … with doubled limits differed by at most one theorem";
- the echoes of task-2 items 2, 4 and 6.

**Cut for length.** Removing these left no reviewed caveat without the claim it qualifies:
- the reading-copy note;
- the "Why it matters" paragraph: its 16-round and 1 % points and the GRPO caveat merged into the short version;
- the short version's "best proved 3 of 106 … and 5 after we let it run longer";
- §1's model-organism sentence (the short version defines the term);
- the likelihood paragraph's RL-dynamics sentence ("one full and one dose run moved strongly …; the rest ended between
  8 units lower and 4 higher"), which Fig. 2 shows;
- the one-round seeding null and its earlier-organism contrast (with its older-checker label);
- "Another knockout run … learned none of them";
- Fig. 3's repeat of "51–60";
- the "What's next" list;
- qualifier-level trims: "or likelihood estimates" (the attempts point stays in §1), "a widely used" (proof
  checker), "that runs" → "run" (the syntax filter), "We had predicted that at least five of the six" → "We had
  predicted at least five of six", and the Fig. 2 alt text's "the", "for" and "the". The trims "which certifies" (for "enough to certify statistically"), "Our "power sampling"" (for "Our
  implementation of …"), "And" (for "But" before the guided 3 / 4 / 1), "the numbers" (for "the experiment's
  numbers") and "at a higher temperature" (dropping "more random") were reverted after the equivalence check.

## 5. Build

- `build.sh` with tectonic 0.17.0: exit 0, 35 pages, no undefined references or citations.
- The overfull boxes are exactly the reading copy's three (0.66, 0.96 and 1.92 pt; built the same way from the marked
  copy). A 4.4 pt box from the longer script list (item 8) was removed by rewording.
- Figures: `fig0_overview` (the 2.9× / 4.2× label) and `blog_fig2_logp` (title) were regenerated with
  `scripts/figures.py overview blog`. The other three blog figures regenerated pixel-identical and were left as they
  were.

## 6. Equivalence check (independent subagents)

Two independent subagents compared the clean versions with the marked copy (commit `11d07d13`): one for the paper,
working through every changed span, and one for the blog, claim by claim. Full reports:
`artifacts/paper_clean/equivalence_paper.md` and `artifacts/paper_clean/equivalence_blog.md`.

**First pass (2026-10-14, 06:00–06:24 UTC).**

*Paper.*
- 85 changed spans checked; 60 of 60 `\rv{}` spans confirmed (57 word for word, 3 smoothed with every number and
  caveat kept).
- No number changed outside §2 / §3 of this file and `main.tex`'s box and date. No hedge was dropped, and `main.tex`
  changed only in the macro, the box and the date.
- The one claim that got stronger is item 12: the "comparators kept no proof text" caveat now covers three
  comparators. It is a listed task-2 fix, and its 24 / 24 re-check matches the artefacts.
- **0 must-fix; 3 minor, all applied:** M1 (the abstract's "under EI"), M2 (§6: name the pool behind seed 3's zero
  rate), M3 (App. I: seed 3's C4 batch).

*Blog.*
- 2,488 / 2,482 words at this pass (2,493 / 2,487 after the fixes below).
- 31 of 31 marks accounted for: 30 survive, and #23, the earlier-model label, went with the seeding paragraph it
  qualified.
- B1–B8 and the four echoes: 12 of 12. The brief's four caveats are present. No number changed outside the listed
  fixes.
- **2 must-fix, both applied:**
  - the power-sampling sentence I wrote for item 13 mis-described the knockout runs;
  - "which certifies" had dropped "statistically".
- **10 minor:**
  - m1–m5 applied ("But", "implementation of", "the experiment's numbers", "more random", the C1 comparison's
    baseline);
  - m6 (define "hardest" in Fig. 4's caption) not needed: §6's text defines it;
  - m7 (the class-(b) cuts) accepted; they are listed in §4 and in `next_steps.md`;
  - m8 not changed: the free positive reductio counts are a post hoc measure and knockout s1's continuation was
    exploratory, but the marked blog leaves both unlabelled too, so the cleanup changed no claim's strength. The fix
    is proposed in `next_steps.md`;
  - m9 and m10 (this file's word-count bookkeeping and cut list) fixed.

**Second pass on the changed passages (06:27–06:39 UTC).**
- A fresh subagent checked the 10 changed passages (7 in the blog, 3 in the paper; 14 edits) against the marked copy
  and the clean paper.
- **0 must-fix.** Every first-pass finding that needed a text change is fixed. No marked number was lost, and only
  listed fixes were added. No hedge or scope was dropped. Every changed blog claim is backed by the paper, and the
  power-sampling and C4 run arguments were checked.
- 3 minor, all applied: a comma in the abstract ("within 16 rounds of EI, seeds …"), and two bookkeeping notes in this
  file (item 4's blog wording; the trim list and item 13's alt-text quote).
- Report: `artifacts/paper_clean/equivalence_second_pass.md`.
