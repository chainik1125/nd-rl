---
written_on: 2026-11-07
written_by: agent:claude
papers:
  - liuProRLProlongedReinforcement2025
  - yueDoesReinforcementLearning2025
---

> **Reading copy (librarian, 2026-11-07).** These are the deliverables of run `olmo-novelty` (proposal 38; fork
> `dan-pandori/nd-takehome`, branch `dan_olmo-novelty`, DONE
> [`60f33d7d`](https://github.com/dan-pandori/nd-takehome/commit/60f33d7d)), with the reviewer's corrections applied to
> REVIEW.md. Each change from the review is marked *[review]*. No hard-constraint violation, nothing quarantined. In
> the review's verdict the correlation, the counts, the ledger and the main novelty conclusions **stand**, and nothing
> REVIEW states as a finding is unsupported; it asks for six rewordings and five minor ones, listed below.
>
> Review: [`review_olmo-novelty.md`](https://github.com/dan-pandori/nd-takehome/blob/d2d44fd4/review_olmo-novelty.md).
> Summary: [experiment-summaries/2026-11-07-olmo-novelty-…](../../../experiment-summaries/2026-11-07-olmo-novelty-per-prompt-ifbench-reach-at-k256-novel-prorl-pattern-rho-minus-0p50-weaker-than-random-pairing-null/README.md).
> The executor's files, byte for byte, are this directory at commit `824c828`. The commit after it applies the review,
> so `git diff 824c828 -- docs/literature/2026-11-06-olmo-novelty/` shows every change the librarian made.

# How novel are our Olmo create-vs-elicit results? (proposal 38)

Dan's question (2026-11-06): for each finding of our Olmo runs, what is the closest prior result, does it replicate,
extend or contradict ours, and what remains genuinely new? In particular, set ProRL
[@liuProRLProlongedReinforcement2025] side by side with ours, compute its "largest where the start is weakest"
correlation on our stored results, and say whether our IFBench reach survives an Invisible-Leash-style objection.

The models behind our numbers: `dpo` = `allenai/Olmo-3-32B-Think-DPO` (32 B; pretrained base → Think SFT → DPO; the RL
start); `s0100` / `s0750` / `s2300` = `allenai/Olmo-3.1-32B-Think` at RL steps 100 / 750 / 2,300; C1 =
`allenai/Olmo-3-1025-7B` (base) against `allenai/Olmo-3.1-7B-RL-Zero-Math`. They are natural-language chat models, so no
proofs are judged and the Lean rule does not apply. "Hand" in the Olmo runs means blinded Claude subagent readers plus
the executor's confirmation (for C1, olmo-rlzero-pilot's reviewer's six reader subagents), not human annotators.

| file | what |
|---|---|
| [`REVIEW.md`](REVIEW.md) | the executor's literature review (3,000 words before the marks): bottom line, novelty matrix (A: the closest papers' setups; B: each claim against them), ProRL side by side with the correlation and the Invisible-Leash answer, Q3–Q6, the ranked claims, five objections, a ≈ 600-word related-work draft, expectations against outcomes. 13 *[review]* marks and four *[librarian]* notes, two of them ledger ids the librarian looked up |
| [`claims_C1-C8.md`](claims_C1-C8.md) | our claims C1–C8 with the models behind them (the research agents' reference sheet) |
| [`notes/`](notes/) | 20 in-depth notes in TEMPLATE format, each with a setup card, on 20 sources (19 papers and NVIDIA's ProRL v2 blog post): `yang2026-passk-crossovers.md` and `yang2026-passk-crossovers-prorl.md` are two agents' notes on the same paper (2609.22547), and `wen2025-cot-pass-at-k.md` covers two papers (Wen et al. and Taşaltı et al.) |
| [`ledger/`](ledger/) | 748 cited claims, each a verbatim quote (≤ 25 words) with its source file and location, in 7 files: `PR` (ProRL family), `IL` (critics, What Outcome RL Buys), `IF` (instruction following), `DT` (domains, trajectories), `GG` (gaming, grading, Yue's citers), `NV` (the executor's fresh novelty search), `EX` (the executor's own). Columns: `claim_id`, `source_id`, `source_file`, `location`, `quote`, `supports`, `used_for`. REVIEW's bracketed ids point here |
| [`screened.md`](screened.md) | the 200 screened sources (176 new to the four earlier nd-rl reviews, 24 prior), with depth, relevance and the claims each bears on |
| [`to_add_to_zotero.md`](to_add_to_zotero.md) | 51 identifiers for Dan: every source read in depth and every source of relevance ≥ 2 not already in `references.bib`, which was not edited. 28 of the 30 citekeys in REVIEW's `papers:` list (all but Yue's and ProRL's) are not in `references.bib`; they and the notes' citekeys are proposals until the papers are imported |
| [`correlation.py`](correlation.py) | ProRL's correlation on olmo-ifomega's stored fixed-phase counts (seed 20261107). It runs only inside the fork's checkout, not from this copy: it finds `olmo_ifomega/results.csv`, `candidates.csv` and `problems.jsonl` relative to its own location and writes into `olmo_novelty/` (numpy, scipy; ≈ 80 s) |
| [`correlation.csv`](correlation.csv), [`correlation_summary.csv`](correlation_summary.csv), [`correlation_stdout.txt`](correlation_stdout.txt) | its per-prompt inputs, its summary table and its printout. **Unmarked; see the corrections below** |

**Not copied; on the fork branch** at `60f33d7d` (the review and its kit are at `d2d44fd4`):
- `run_olmo_novelty.md`, `numbers.md` § olmo-novelty, `log.md` § olmo-novelty, `STATUS.md` and
  `preregistration/olmo-novelty.md` (commit `664aaa93`);
- `olmo_novelty/README.md`, `AGENT_INSTRUCTIONS.md` (the research agents' shared rules), the two adversarial claim
  checks `_claim_check.md` and `_claim_check2.md`, the per-agent screening rows and relation labels `screen/*.tsv`, and
  the code `verify_ledger.py` (the quote checker, `--neg` for negative controls) and `merge_screen.py` (which makes
  `screened.md` and `to_add_to_zotero.md` from `screen/*.tsv`);
- the reviewer's kit `review_onv/`: its own correlation code, quote normaliser, re-fetch scripts, its blind
  fresh-search rows `fresh/R{A,B,C}.tsv` (54 quotes) and its adversarial self-check `adv_check.md`.

The source texts are copyrighted, so they are in neither git nor the bucket: the run's 102 files are in
`~/onv_sources/` on agent2, the reviewer's in `~/onv_rev_sources/`. Each ledger row names its source file there, and
five notes name the directory. 702 of the 748 rows give an arXiv id and version to fetch the source again from; the
other 46 give a URL, a Hugging Face repository or API call, or a venue id (NVIDIA's ProRL v2 blog, ProRL's NeurIPS
camera-ready, model cards, a LessWrong post, a blog, a GitHub README), and live API responses may have changed since.
The run made no checkpoints, artifacts or data, so nothing went to the bucket.

## What the review changed

Items follow the review's §Verdict ("Must be reworded" 1–6, then "Minor").

| item | the review's finding | where it is applied |
|---|---|---|
| 1 | C4's closest IF neighbour is missing from the bottom line: Wei 2608.00220 App. C.2 finds 73.5 % of an IF-RL checkpoint's verifier passes "rely on superficial shortcuts" [IF-55], and shortcut responses rise from 4.5 % under the base policy to 40.1 % at the final checkpoint, a base-against-RL comparison though not at reach. REVIEW cited Wei in bottom line 1 only for per-prompt coverage at k = 32, a C3 neighbour | bottom line 1 (the ledger id of the 4.5 % → 40.1 % sentence, IF-54, is the librarian's lookup) |
| 2 | C2 was positioned without the lab's own numbers for the same model pair. The Olmo 3 report's Table 14 gives AIME 2025 pass@1 DPO 70.7 → Olmo 3.1 Think 78.1 (+7.4) and AIME 2024 76.0 → 80.6 (its grade and sampling: "EM Flex", T 0.6, top-p 0.95; ours T 1, top-p 1). Our fixed phase by final stated answer gives `dpo` 0.667 → `s2300` 0.738 on AIME 2025 (+7.1), so it reproduces the lab's gain. C2's +3.0 (0.716 → 0.746) is under RL's verifier on AIME 2025–26; it is small because that verifier is lenient (`dpo` 0.696 against 0.667 by final answer on AIME 2025) and because of the 2026 problems. The reach claim is unaffected | Matrix B, C2 row |
| 3 | C5 was ranked "replicates" wholesale, but only its token-cap half has a precedent (finishing [IL-95, IL-96]; truncation [IL-154]). For the brevity half the sources show only that RL answers are short [IF-22] and drop preambles [NV-13]; neither ties brevity to the RL-only prompts. "Token cap: replicates; brevity: extends" | Matrix B C5 row; §4; §7's L-6 gloss (bottom line 3 was already split) |
| 4 | DT-16 ("+4 on AIME, +20 on IFBench") is the same RL run's 750 → 2,300 extension measured as benchmark pass@1: a domain contrast in pass@1 gains on our own run, not reach and not an independent replication. Independent within-run evidence exists: Guru, Zhai and, from the review's fresh search, AceReason-Nemotron 1.1 | Matrix B C3 row; §3 Q4 (with AceReason-Nemotron 1.1's numbers from the review's §Fresh search) |
| 5 | L-5 is scored "partly held" in REVIEW §7 and the log, but "failed" in `run_olmo_novelty.md` | not in this copy: REVIEW already says "partly held". See "Not applied" |
| 6 | Three hand-reach prompt sets are mixed under two numbers: olmo-ifomega's executor's 42 (`verdict.csv`), its reviewer's 43 (40 shared) and olmo-iffu's hint set of 42. On the executor's 42, 34 are untested at 1,024, not 33 | §2, a new bullet naming the sets. *[librarian]* It adds that §5's "20 of 42" is the executor's set (the fork's `numbers.md`, § olmo-ifomega, "Per-constraint analysis") |
| minor | "within 512" → "within 383" (the reviewed figure) | §6 |
| minor | cite "no KL" | §2 table, cited as [PR-137]. *Librarian's check:* the review noted that §2's "without KL" carried no ledger id and quoted the Olmo 3 report ("No KL loss: We remove the KL loss"); the ledger already holds that quote, as PR-137 |
| minor | close §3 Q3's parenthesis | §3 Q3, after [IF-76] |
| minor | one sentence noting that ProRL's own R² (Fig. 3 regresses a pass@128 difference on its base value; legend "Linear regression (R² = 0.879)") carries the same bounded-difference artefact | §2, after "Our reading" |
| minor | `numbers.md`'s "25 of them" → "25 of the 200 (24 new)" | not copied; REVIEW §7 already says "25 sources, 24 new" |

The notes, the ledgers, `screened.md`, `to_add_to_zotero.md` and the claims sheet carry no marks. The review checked
the ledgers (all 748 quotes found; 21 random `supports` sentences agree with their source context) and the notes
(every one in template form). It counted the screening rows behind `screened.md` (`screen/*.tsv`, not copied), and the
run's "prior" flags agree with its own prior list; it checked `screened.md` and `to_add_to_zotero.md` themselves only
for presence, and read the claims sheet as an input. It asked for no change in any of them.

### Corrections to files that carry no marks

`correlation.csv`, `correlation_summary.csv` and `correlation_stdout.txt` are the executor's, unchanged, so that they
still match the code that made them. The review's own code reproduces every correlation it re-derived: point
estimates within 0.003, intervals within bootstrap noise (P1, IFBench strict pass@1, `s2300` against `dpo`:
−0.505 [−0.603, −0.396] against the executor's −0.504 [−0.599, −0.397]). One qualification: the IFEval per-constraint-type interval (25 types, pass@8, `s2300`; `correlation_summary.csv` line 23,
`correlation_stdout.txt` line 22) has a seed-sensitive upper bound. The executor's −0.082 becomes −0.018 to −0.059
over six of the reviewer's seeds; the point estimate reproduces (−0.519 against −0.516).

### Not applied

These are minor, are suggestions rather than corrections, or concern files that were not copied; they are left for
the fork or for Dan.
- `run_olmo_novelty.md` (the executor's 197-word summary) says "L-1 and L-5 failed" (L-5 partly held, item 5), names
  only HiR and the lone-win audits as neighbours (Wei's 73.5 % is missing, item 1) and has C5 "replicate" wholesale
  (item 3). `STATUS.md`'s Done line repeats the Wei omission and the wholesale C5. `numbers.md` has "25 of them"
  (minor).
- The review would find "narrowly" fair for C-2 (the naive ρ is more negative than P1 by 0.032, against ≥ 0.03).
  REVIEW §7 says "held".
- The review's caveat on objection 1's compute-matched K (≈ 8 × 10⁵ draws per prompt): besides the stated mix of
  H100-hours and H200-seconds, the per-draw cost was measured on reach-candidate draws (5,442 tokens) against a typical
  `dpo` IFBench draw of 4,448, so K is ≈ 20 % low if cost scales with tokens. The review scores the figure "reproduces
  as an order of magnitude".
- The pre-registration's header says "Written 2026-11-07 12:15–12:40 UTC", a window that ends 22 minutes after its own
  commit (`664aaa93`, 12:17:36). `log.md` records the explanation, and a working-tree edit of the header was reverted,
  so the file is as pre-registered.
- *[librarian]* Two points the review did not raise, left unmarked. §2's side-by-side table gives "256 / 512 / 1,024
  for the start" in the column of the 32B run, whose DPO start was sampled to 256 (1,024 on 10 IFBench prompts in
  olmo-iffu); the 512 is C1's 7B base (olmo-rlzero-pilot), which Matrix A's "Ours" line ("start budget k 256–512")
  covers together with the 32B start. And REVIEW rounds the random-pairing null to −0.66
  (the executor's −0.655), where the review's own value is −0.654; the summary and the records use −0.65.

## Recompute

From the fork's repository root at `60f33d7d`:
- `python3 olmo_novelty/correlation.py` (numpy, scipy; ≈ 80 s on the VPS CPU);
- `python3 olmo_novelty/verify_ledger.py --neg`, which needs the source texts in `~/onv_sources/`;
- `python3 olmo_novelty/merge_screen.py`.

The reviewer's own versions are `review_onv/corr.py`, `review_onv/ledger_check.py` and `review_onv/screen_count.py` at
`d2d44fd4`.
