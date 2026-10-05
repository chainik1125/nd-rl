---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - hupkes2020compositionality
---

# Compositionality decomposed: five behavioural tests (systematicity, productivity, substitutivity, localism, overgeneralisation)

Paper: [@hupkes2020compositionality] (Hupkes, Dankers, Mul, Bruni, JAIR 67, 2020)
Source: arXiv 1908.08351v2 (PDF text; read: abstract, Sec. 1, Sec. 3 in full, Sec. 4 opening, Sec. 6.1-6.4
and Table 1, Sec. 7; Sec. 6.5-6.6 details and Sec. 2 related work skimmed).

## Learnings

- **Problem.** "little consensus exists on whether neural networks are able to generalise
  compositionally", in part from "a lack of agreement about what it means for a neural model to be
  compositional" (Abstract, p. 1).
- **Five task-independent tests** (Sec. 3, p. 7): "(i) if models systematically recombine known parts and
  rules (systematicity) (ii) if models can extend their predictions beyond the length they have seen in the
  training data (productivity) (iii) if models' predictions are robust to synonym substitutions
  (substitutivity) (iv) if models' composition operations are local or global (localism) and (v) if models
  favour rules or exceptions during training (overgeneralisation)".
- **Systematicity** traces to Fodor & Pylyshyn: "[t]he ability to produce/understand some sentences is
  intrinsically connected to the ability to produce/understand certain others" (Sec. 3.1, p. 8), contrasted
  with a dictionary of stored sentence-meaning pairs. Any model that generalises beyond its training set
  recombines something, so "rather than asking if a model is systematic, a more interesting question is
  whether the rules and constituents the model uses are in line with what we believe to be the actual
  rules and constituents" (p. 9). Split: "(i) the model has only been familiarised with a in contexts
  excluding b and vice versa but (ii) the combination a b is plausible given the rest of the corpus"
  (Sec. 3.1.1).
- **Productivity** = length extrapolation: "We test whether a model can understand sentences that are
  longer than the ones encountered during training" (Sec. 3.2.1); the length split is compared with a
  model trained with some long examples, to separate unseen length from inherent difficulty (Sec. 6.3.2).
- **Substitutivity and localism are scored by consistency, not accuracy.** "The most important point is not
  whether a model correctly predicts the target for an adapted input sequence, but whether its prediction
  matches the prediction it made before the transformation"; the "consistency score, which expresses a
  pairwise equality" (Sec. 6.4.1, p. 23) "allows us to evaluate compositionality aspects isolated from task
  performance". Localism compares the output on a composed input with the output when the parts are
  forced to be processed first (Sec. 3.4.1).
- **Caveat about positives.** "A positive result, on the other hand, cannot necessarily be explained as
  successful compositional learning, since it is difficult to establish that a good performance cannot be
  reached through heuristics and memorisation" (Sec. 4, p. 12).
- **Results on PCFG SET** (Table 1, 3 runs): task accuracy 0.79 / 0.85 / 0.92 (LSTM / ConvS2S /
  Transformer) but systematicity 0.53 / 0.56 / 0.72 and productivity 0.30 / 0.31 / 0.50; "the
  systematicity scores of all models are substantially lower than their overall task accuracies"
  (Sec. 6.2.2), drops of 33 %, 34 %, 22 % (Sec. 7.2), productivity drops 62 %, 64 %, 46 % (Sec. 6.3.2).
  Their explanation: models chunk frequent function pairs instead of representing each function. Synonym
  consistency on *incorrect* outputs: "Transformer is the most consistent, but with a low score of only
  0.34" (Sec. 7.2). Enforcing local composition changes the answer in 54 %, 41 %, 46 % of test samples.
  Early overgeneralisation is read as rule acquisition: "we take this as a clear indication that models in
  fact capture the underlying rule at that point" (p. 33).

## Evidence and limitations

- One artificial task (string-edit functions, PCFG with naturalised length/depth), three architectures,
  three seeds, no hyper-parameter search (footnote 18). The tests are explicitly not normative: they "should
  not be taken as a normative specification of what models should and should not do" (Sec. 7.1).
- The tests are behavioural and binary per item (exact-match accuracy or exact-match consistency); there is
  no sampling budget or probability-level version — every result is greedy-decoding behaviour.
- Sec. 6.5-6.6 numbers beyond the summary were not checked.

## Connections and questions

- **Definition offered:** compositional generalisation is not one number but five behavioural properties,
  each a train/test split or an input transformation, measured by exact-match accuracy on held-out
  composites (systematicity, productivity) or by **consistency** of the model's outputs across
  meaning-preserving transformations, regardless of correctness (substitutivity, localism).
- **New vs better access:** not addressed (models trained from scratch, no pre/post comparison). The tests
  can be turned into a rule on our models: a capability is a *schema-level* property, and is credited only
  if it survives held-out splits the training never contained. Created-vs-elicited then becomes a
  difference-in-splits: the base fails the held-out split at budget K, the RL model passes it, and RL
  never trained on the held-out instances.
- **Null / floor:** not handled; exact-match greedy outputs have no "any k" problem but also no resolution
  below the greedy mode. A random-init model scores 0 on accuracy but can score high on consistency (a
  constant output is perfectly consistent) — consistency must be conditioned on success or on non-trivial
  outputs.
- **Transfer to our setting:** (1) *Substitutivity*: atom permutations, premise reordering and currying are
  meaning-preserving; per theorem t and transformation g, compute the consistency of the solve indicator
  (and of the modal proof up to g) between t and g(t), for pend, r8, r16. A memorised instance shows
  success on t and failure on g(t). (2) *Systematicity*: the excluded-middle case is literally Hupkes's
  split — every move occurs in pretraining, no premise-free A ∨ ¬A theorem does; seed s1's success on
  held-out A ∨ ¬A instances (new A, never trained) is a systematicity-plus-substitutivity pass that the
  other seeds and pend fail. (3) *Productivity*: length splits by proof lines / term size (cap-6 vs cap-12
  sets, `transfer_long`). Cost: generating transformed copies is free (gen.py), sampling at k = 256 per
  copy is J2-scale. Failure modes: schema membership of held-out instances must be defined by the
  generator, not by the model; Lean-level renaming is not exactly meaning-preserving for the tokenizer
  (name bases, sampler marginalisation), so consistency must be computed with the same name-base protocol.
- Related: `keysers2020cfq.md` (makes the split quantitative), `okawa2023multiplicative.md` (why
  composites emerge late), `wu2023counterfactual.md` (counterfactual variants as the substitutivity test at
  scale), `harding2024capability.md` (reliability across conditions).
