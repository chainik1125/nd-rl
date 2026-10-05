---
written_on: 2026-10-05
written_by: agent:claude
---

> Reading copy of run `capability-defs`'s `capability_defs/cards/transfer-invariance.md` (fork branch `dan_capability-defs`, DONE
> `645aa4f0`); reviewer's corrections marked *[review]*. Report: [`../REPORT.md`](../REPORT.md); review:
> [`review_capability-defs.md`](https://github.com/dan-pandori/nd-takehome/blob/da11be78/review_capability-defs.md).

# Card: transfer and invariance — a capability generalises; a memorised instance does not

Family T. Slug `transfer-invariance`. Notation: `_FRAME.md`.

## 1. Definition, formally

A capability claim about theorem t is a claim about t's **equivalence class** under transformations that should not
matter, and about **held-out** instances.

- **Invariance set** G(t): renamings (atom permutations; F is falsum, not an atom), premise reorderings, curried /
  uncurried forms (A ∧ B → C vs A → B → C), and re-generated instances of t's schema.
- **Capability score** of θ on t: min over (or mean over) t′ ∈ G(t) of p_θ(t′). **Consistency** (Hupkes et al.): the
  agreement of solve / fail across G(t), whether or not the instances are solved.
- **Transfer:** RL trained on `rl_targets` and evaluated on transfer / holdout250 / textbook72 (never trained). A
  capability RL created must show up on held-out instances, not only on its training targets.
- **Length generalisation:** success on members of a family longer than any training proof of that family.

## 2. Decision rule

| verdict | rule |
|---|---|
| **created** (by any other card's rule) | the verdict counts only if it holds on held-out members and on the invariance set: RL's min over G(t) is above the bar and the base's max over G(t) is below it |
| **instance only** | RL solves t but not most of G(t), or only trained-on members: memorisation or a lucky route, not a capability |
| **elicited** | as per the base card, but evaluated on G(t) |

This card **qualifies** other definitions: it decides the unit of a capability (a class, not a string).

## 3. Null or floor

- Consistency has a natural null: chance agreement given the marginal solve rate.
- Random weights are consistent (they fail everything) but have no capability; consistency must be paired with
  success.

## 4. How to compute it here

- **Renaming / reordering:** a generator transform plus k-256 reads. Not run here: ≈ 0.3 A40-hours per checkpoint for
  322 × 4 variants.
- **Held-out transfer:** the ladder logs (transfer is sampled every round, never trained) and the holdout250 reads.
- Free numbers (`out/schema_c12.txt`). `dist_and_over_or` transfer members (never trained) go from 0.03 at pend to
  0.97 / 0.88 / 0.70 by r8, matching the trained-on members (0.97 / 0.90 / 0.78). Excluded middle on s1: 0.92 held-out
  (key-step members) vs 0.95 trained-on at r16. These family capabilities transfer.
- **Length:** the trajectory reads stratified by `L_true`. `support-curves` and `state-frontier` measured length
  frontiers (L*).

## 5. Sensitivity

- **The choice of G:** what "should not matter". F is falsum, so permutations of P, Q, R, S only. Premise order
  sensitivity of `gen.canon_key` was a past trap (`claim-audit`).
- **Temperature, decoding:** as in the base card.
- **Representation:** the proof-state environment canonicalises names, so renaming invariance is partly built in.
- **Noise:** more instances means less noise. Families of 40 give SE ≤ 0.08.

## 6. Failure modes

- **Leakage:** held-out instances share the schema template with trained ones. Transfer within a generator family is
  weak evidence of a general capability (Wu et al.: renaming is a "superficial" counterfactual).
- **Invariance is necessary, not sufficient:** a model can be consistently right through a memorised template.
- **Classes beyond the generator** (textbook problems) are where the real test lies, and they are few (72).

## 7. Relations

- It qualifies every theorem-level card.
- With families it is `schema-acquisition`.
- With length it is the group's "longer than the data" definition (`out-of-data-novelty`).
- Consistency is `reliability`'s cousin across instances instead of across samples.

## 8. Literature anchor

- **Hupkes et al. 2020 (1908.08351):** five tests (systematicity, productivity, substitutivity, localism,
  overgeneralisation), with a consistency score that ignores correctness (Sec. 3, 6.4.1).
- **Wu et al. 2023 (2307.02477):** default-minus-counterfactual gap with a comprehension check (Sec. 2).
- **Mirzadeh et al. 2024 (2410.05229), GSM-Symbolic,** screened.
- **Keysers et al. 2020 (1912.09713):** compound divergence (Sec. 2.1).
- **Harding & Sharadin 2024 (2405.08989v1):** an account "should explain how a model's ability to φ can be invariant
  across different methods for eliciting and measuring that capability" (Sec. 3.2, Def. 6).
- Verified in `_claims_L5.md` / `_claims_L2.md`.

## 9. Critic's verdict

**Strongest argument (critic): the base's *max* over G(t) lets a degenerate substitution veto creation.**
- In 17 / 40 trained-on and 20 / 40 held-out `dist_and_over_or` members, one disjunct follows from the premise alone
  (G4ip), so the distribution step is unnecessary. These are pend's only round-1 hits in the family.
- One hit in 32 gives p_B ≥ 1.6 × 10⁻³, which vetoes "created" for the whole family on all three seeds, though pend hits
  none of the 43 members that need distribution and RL solves the held-out ones (1.00 / 1.00 / 0.80 by r16).
- 10 of the 11 created (family, seed) cells in `schema_c12_keystep` would be vetoed this way.
- Secondary arguments:
  - Currying *is* the `import` / `export` families, so neither could ever be "created".
  - Extreme values over a growing G track atom and premise counts and cost ≈ 60 K attempts per variant.
  - §4 compared cumulative held-out unions with single-round trained-on shares.

**My answer: accepted.**
- Base and RL are scored with the same statistic: mean p over a pre-registered sample of G(t), restricted to members
  that need the key step (G4ip for classical schemata; neither disjunct derivable alone for disjunctive conclusions).
- Currying is dropped from G.
- The card becomes the held-out check inside `schema-acquisition` rather than a separate verdict. A shortcut member
  now moves the mean by 1 / |G| instead of vetoing.

