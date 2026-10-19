---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - shin2023superhuman
---

# Novelty as first departure from a historical move database (Go, 1950-2021), paired with engine-scored quality

Paper: [@shin2023superhuman] (Shin, Kim, van Opheusden, Griffiths; PNAS 2023, doi 10.1073/pnas.2214840120)
Source: arXiv 2303.07462v2 (accepted-manuscript PDF, read in full: main text pp. 1-19; the SI Appendix, which holds
the technical definitions in its sections 2.2-2.3 and Figs. S1-S6, is a separate OSF file and was **not** read)

## Learnings

- **Data and question.** "more than 5.8 million move decisions made by professional Go players over the past 71
  years (1950-2021)" (Abstract); the advent of superhuman AI is dated to AlphaGo's win on "March 15, 2016" (p. 3) and, more broadly,
  "a series of events that occurred between 2016 and 2017" (p. 4, footnote 1). Main finding: "novel decisions (i.e., previously unobserved moves) occurred
  more frequently and became associated with higher decision quality after the advent of superhuman AI" (Abstract).
- **Novelty Index (NI).** Borrowed from chess practice: find "the first move that makes the game's move sequence
  historically novel" relative to every earlier game in the database (p. 7); then "subtracted its move number from
  the maximum move number observed in the dataset (i.e., 602) to form a Novelty Index" (p. 7), so an earlier
  departure scores higher. Only "the first 60 moves in each game" are used (p. 7, fn. 2). The index needs no engine:
  "Calculating the Novelty Index for each game did not require any superhuman AI program" (p. 18).
- **Built-in drift.** Later years get lower NI mechanically because "the observed set of unique move sequences grew
  over time and pushed novel moves later" (p. 7); the post-2016 jump is read against this downward trend (Fig. 1C-D).
  Saturation: "more than half of all novel moves in our entire dataset occur before Move 10 and that 99% of opening
  move sequences become historically novel by Move 21" (p. 12).
- **Decision Quality Index (DQI).** KataGo plays the counterfactual best move in each position and scores both moves;
  DQI "equals [100% - (win rate of the Counterfactual AI Decision - win rate of the actual human decision)]" (p. 19),
  estimated with "10,000 game patterns for each of the 5.8 million decisions" (p. 5), "58 billion counterfactual game
  patterns" in total (Abstract). "about 40% of all human decisions matched the optimal AI decisions" (p. 10).
- **Novelty × quality.** Regressing DQI on After-AI, Novelty and their interaction (player, move-number, month fixed
  effects; SEs clustered by player; 5,857,513 observations): "the interaction term was significant and positive, β3 =
  0.515" (p. 8; Table 1), while novel moves were worse before AI (Novelty Dummy −0.608, Table 1).
- **Copying vs internalising (the control most relevant to us).** To test whether humans merely replayed AI moves, the
  authors generated AI-vs-AI games and inserted "600k move decisions made by a superhuman AI program into the dataset",
  dated just before the advent of AI, then recomputed novelty: "the time trends of the Novelty Index hardly change"
  (pp. 15-16). Quality also rose for moves that differ from the AI's choice and in later game stages (pp. 10-13).
  Conclusion: "memorization of AI decisions cannot be the sole explanation for the increase in decision quality and
  novelty" (p. 17).

## Evidence and limitations

- Observational, interrupted time series; the causal chain is left open: "whether the advent of superhuman AI increased
  novelty and thereby increased decision quality (i.e., whether each link in the possible causal chain can be
  established)" (p. 17).
- NI depends on the reference database (GoGoD) and its growth; it is a property of a sequence relative to a corpus,
  not of a position. Exact NI construction and robustness (SI 2.3, Fig. S1 alternative measure) not read.
- DQI is judged by one engine; "quality" is win-rate loss against KataGo's choice, not ground truth.
- This is about humans learning from AI, not about AI novelty; the measures are what transfers.

## Connections and questions

- **Definition offered:** novelty of a behaviour = how early a sequence of decisions leaves the set of all previously
  observed sequences (prefix novelty against a dated corpus), paired with an external quality score; 'copied from a
  source' is tested by adding the source's outputs to the corpus and checking that novelty survives.
- **New vs better access:** not about model capabilities, but it supplies a corpus-relative operationalisation of 'new'
  plus a copying control. Turned into a rule for us: an RL proof is **created** relative to the base if it departs
  early from every proof prefix the base produces (and from every pretraining proof) *and* is valid; it is
  **elicited** if its prefixes, up to the last few steps, already occur in a large base-sample database. The
  'insert the AI's games' control becomes: add N base samples per theorem to the reference corpus and re-measure;
  novelty that survives large N is novelty beyond what sampling the base would find.
- **Null / floor:** the pre-2016 trend (decreasing NI as the corpus grows) is the null; nothing like a random agent. A
  random policy would be maximally 'novel' and worthless, which is why NI must be read jointly with quality (DQI here,
  Lean validity and proof length for us).
- **Transfer to our setting:** cheap (string/prefix matching, no GPU). For each theorem, reference set R = canonicalised
  proof-step sequences from (a) the pretraining corpus, (b) pend's k = 256 samples, optionally (c) pend's samples at
  larger budgets. For each Lean-accepted r8 / r16 proof, record the first step index at which its prefix leaves R and
  report proof length minus that index ('steps of novel continuation'), plus whether the whole proof's canonical key is
  in R. Plot the distribution per seed, and the same for pend's own held-out samples against a disjoint pend sample
  (the null for 'novel by chance'). Excluded middle: are r16's A ∨ ¬A proofs novel from step 1 (new opening, e.g. an
  immediate by_contra), or do they share a long prefix with pend's failed attempts and differ only at the classical
  step? Failure modes: (1) the drift the paper itself shows — novelty grows with proof length and shrinks with |R|, so
  compare at fixed |R| and stratify by length; (2) renaming and step-order variants make trivially 'novel' proofs —
  canonicalise up to atom renaming and commuting steps (note `gen.canon_key` is premise-order sensitive); (3) it is a
  support measure, so it inherits the 'any k' problem unless R's size is tied to a budget K.
- Related: `brown2024monkeys.md` and `kazdan2025passk.md` (coverage at budget), the prior-list Schut et al. (concepts
  taught back to grandmasters, cited for context), and the AlphaGo / AlphaGo Zero / Zahavy rows in `_screen_L6.md`
  (novelty as low prior probability under a human-trained policy).
