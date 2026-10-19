---
written_on: 2026-10-19
written_by: agent:claude
---

> **Reading copy (librarian, 2026-10-19).** These are the taxonomy, tables, canonical examples, blog and figures of run
> `proof-strategies` (proposal 28; fork `dan-pandori/nd-takehome`, branch `dan_proof-strategies`, DONE
> [`441de472`](https://github.com/dan-pandori/nd-takehome/commit/441de472)), with the reviewer's corrections applied.
> Each change from the review is marked *[review]*; eight sealed-class rewordings in `TAXONOMY.md` are marked
> *[librarian]*. No reported count changed. No hard-constraint violation, nothing quarantined.
> **Wrong as stated (now reworded):**
> - "no K12 theorem is σ-covered": true only under the run's semantic double-negation rule. Under the pre-registered
>   rule, which is also create-or-elicit's, K12 has 2,340;
> - "every proof of a σ-covered theorem needs a working reductio (#10)": at least 573 accepted RL proofs of σ-covered
>   targets do without one.
>
> Review: [`review_proof-strategies.md`](https://github.com/dan-pandori/nd-takehome/blob/e6bfe0ea/review_proof-strategies.md).
> Summary: [experiment-summaries/2026-10-18-proof-strategies-…](../../../experiment-summaries/2026-10-18-proof-strategies-13-strategy-taxonomy-k12-reductios-are-templates-textbooks-use-working-reductio-mt-ds/README.md).
> The executor's files, byte for byte, are this directory at commit `0f4b6c4`. The commit after it applies the review,
> so `git diff 0f4b6c4 -- docs/papers/proof-strategies/` shows every change the librarian made.

# Proof strategies in our pretraining data: taxonomy, tables, examples and blog (proposal 28)

Drafts for Dan. Nothing here has been posted or shared.

| file | what |
|---|---|
| [`blog.md`](blog.md) | the plain-language write-up: the 13 strategies as a field guide with one Lean example each, K12's mix, what textbooks use, and what it means for the knockout result. 2,290 words after the corrections; the brief asked for 1,000–2,000 |
| [`TAXONOMY.md`](TAXONOMY.md) | the 13 strategies with their standard names, a checkable predicate each, sub-types, the precedence rule for the primary label, the map to σ / σ⁺ and `patterns.py`, theorem-level labels, validation |
| [`TABLES.md`](TABLES.md) | every share: K12 and the comparison sets, reductio kinds, theorem level, bounded necessity, by length and box depth, the 990-theorem like-for-like, v0 vs v1.2, RL's own proofs |
| [`EXAMPLES.md`](EXAMPLES.md) | 30 canonical examples in ND and Lean: shortest clean and typical for each strategy, plus the reductio kinds. None is in a textbook72 renaming class |
| `figures/` | `k12_mix.png` (K12's strategy mix, working vs idle), `k12_vs_textbooks.png` (K12 against Charles's problems and the textbook schemata, log scale), `reductio_kinds.png` (what kind of proof by contradiction each set holds) |

**Not copied; on the fork branch** at `441de472` (the review is at `e6bfe0ea`):
- `proof_strategies/EXPECTATIONS.md`, the 45 pre-registered rows against outcomes;
- `run_proof_strategies.md`, `numbers.md` § proof-strategies, `log.md` and `preregistration/proof-strategies.md`;
- the code: the classifier `proof_strategies/strategies.py` with 39 Lean-checked unit tests, `theorem_level.py`,
  `necessity_ps.py`, `examples.py`, `tables.py`, `figures.py`, `rl_found.py`, and the second implementation
  `validation/second_impl.py`;
- the per-record labels and searches under `artifacts/ps/` and `data/ps/`. They are also in
  `hf://buckets/dan-pandori/nd-rl/proof-strategies/`, except Charles's release and the textbook72 proofs.

The copied files name these as `code` paths relative to the fork's root. The relative links between the four
documents and to `figures/` work here.

## What the review changed

Items follow the review's §Verdict. Its §Compare tables cite lines in the fork's files. 41 marks *[review]* in all:
blog 15, TABLES 13, TAXONOMY 8, EXAMPLES 5.

| item | the review's finding | where it is applied |
|---|---|---|
| 1 | σ-covered: 0 in K12 only under the run's semantic rule, which counts ¬X → ⊥ as a double negation. The pre-registered and create-or-elicit rule (no literal ¬¬ subformula) gives K12 2,340 (1.51 %), cap-6 2,235 and the dose set 232. The switch was not logged | TAXONOMY §4 and the §5 table; TABLES § 4 (note under the definitions); blog, "The theorems agree" paragraph and the "19 % … none in K12" line |
| 2 | "every proof of a σ-covered theorem needs a working #10" is false: ≥ 573 Lean-accepted RL proofs of σ-covered targets (rounds 1–16; dose1_s0 491, knockout 0) apply DN to a self-built ¬¬ held as a box hypothesis (an ∨E case or create-or-elicit's →I "lemma route") and carry no #10. Target-level results do not change; per-proof RL counts would rise under a widened #10 | TAXONOMY §4; TABLES § 11 (the σ-covered definition and the "free" note); blog, RL section |
| 3 | "blind spot-check 260 / 260" is against v1.1. Against the final v1.2 rules it is 245 / 260 uses (disjunctive syllogism 10 / 20, excluded middle 15 / 20), so the final #6 and #12 were never hand-checked blind | TAXONOMY §6. The blog's wording was already exact |
| 4 | TABLES: the header "No models" is false for § 11; the § 5 caption says the opposite of its share column; the § 5 row "(theorem classical-only) 27" is the bounded requires-DN count (G4ip: 26 of 71) | TABLES header and § 5 |
| 5 | expert iteration also replays 20,000 pretraining records per round, so the full arm kept seeing σ templates during RL: a confound for "pretraining supplied the form" | blog, "The setting" and the knockout reading |
| 6 | three "typical" example glosses describe their one record, not K12 (cases: only the idle split is typical, 89.5 %; #10: a ¬G → ⊥ premise in 31.3 %; →I: two unused assumptions in 7.5 %); "shortest clean" relies on a readability filter stated only in code and on the textbook72 guard | EXAMPLES header notes and the three glosses; the disjunctive-syllogism example |
| 8 | smaller wording: "use what K12 lacks" (rare, not absent); "only their combination" (also DNE on a derived ¬¬, 0.17 %); "173 minimal" (17 minimal per conjunct only); the blog names no model format or checkpoint; two term sizes for DN examples; `bot_case` "never exercised" (2,070 K12 records); one "shortest" proof a line longer than the generator's | blog (five places); TABLES header, § 8 and § 11; EXAMPLES header; TAXONOMY §6 |
| §Compare, minor | disjunctive syllogism 0.03 % rests on a rule tightened after the hand check (0.24 % under v0); textbook72 "never printed" (fixtures quote textbook72 classes): say "not shown here"; textbook72 "decided 54" includes one trivial theorem with no proof (53); v0 modus tollens 64 vs the reviewer's 71; the last disjunctive-syllogism change (47 → 43) is not in `log.md` | blog field guide and table; blog textbook72 line; TABLES § 5 and § 9; TAXONOMY header |

Item 7 (the RL expectations were written a minute after the executor saw create-or-elicit's per-round counts) and the
corrections to the run summary, STATUS, `numbers.md`, `log.md` and `EXPECTATIONS.md` concern files not copied. They are
listed in the summary.

**Not applied; minor, left for the fork or for Dan:**
- TABLES layout (review C5): the sections run 1–9, 12, 11, 10; `state_env_october` is not defined; § 5 cites
  "(bucket)" for necessity files of Charles's problems and textbook72 that were not uploaded; one row mixes precisions.
- Blog sub-audit points the review did not adopt into its verdict. For example: for modus tollens and disjunctive
  syllogism, textbook72's shares run 2–3.5× below Charles's problems; the short version says "pretrained on K12" for
  all models.
- The review's next measurements: a blind hand-check of the final #6 and #12; extending #10 and recounting the 12 found
  files; comparing round-1 free-reductio rates per attempt; a full-arm ladder whose replay excludes σ.

## The librarian's sealed-class check

The review's leak scan parsed ND prompts, ASCII `|-` sequents and Lean statements. It did not parse formulas written in
Unicode prose. The librarian scanned every formula and sequent in the four files, reading their letters as atoms and
using the reviewer's textbook72 keys (`review_ps/sealkey.py`).
- **Eight places in `TAXONOMY.md` are in textbook72 renaming classes.** All are rule schemas or law formulas inside
  definitions. They are restated in words and marked *[librarian]*, the convention `paper-clean` used for a printed
  schema.
- **The fork's `TAXONOMY.md` and this directory's first commit `0f4b6c4` still print them.**
- **Elsewhere:** the librarian's own additions add none. The blog, tables and examples have none, apart from one
  scanner artefact (a token span that crosses a parenthesis in an example's Lean statement). The reviewer checked the
  blog's 12 Lean snippets and the 30 examples under every key.
