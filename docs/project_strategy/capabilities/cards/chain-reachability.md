---
written_on: 2026-10-05
written_by: agent:claude
---

> Reading copy of run `capability-defs`'s `capability_defs/cards/chain-reachability.md` (fork branch `dan_capability-defs`, DONE
> `645aa4f0`); reviewer's corrections marked *[review]*. Report: [`../REPORT.md`](../REPORT.md); review:
> [`review_capability-defs.md`](https://github.com/dan-pandori/nd-takehome/blob/da11be78/review_capability-defs.md).

# Card: chain reachability — creation as a chain of elicitations — DROPPED as a decision rule; kept as a mechanism description (critic, §9)

Family S. Slug `chain-reachability`. Notation: `_FRAME.md`.

## 1. Definition, formally

- **One step of elicitation.** Model θ' is reachable from θ at budget k if every proof θ' learned was sampled from θ
  within k attempts. Expert iteration is exactly this: round r trains only on proofs that model r − 1 found in 32
  attempts.
- **Chain depth** d_s(t) of theorem t under ladder s: the first round whose sampler (model r − 1) solves t. Per round,
  the ladder logs give, for each target, its successes out of 32 (`alloc_r.json`); for held-out transfer theorems, the
  first-found round (`found_transfer_16.jsonl`).
- **Chain-created:**
  - t is outside the base's reach at the budget (p_B(t) < 0.05 / K_per, `passk-budget`);
  - yet t is reached after d ≥ 2 rounds, each of which elicited only what its predecessor could already sample at
    k 32.
- **"Elicitation is not transitive":** a chain of elicitations can end in a creation relative to its start.

## 2. Decision rule

| verdict | rule |
|---|---|
| **created (by chaining)** | t is first solved at round d ≥ 2 and is certified outside the base's reach at K_per. RL built it from steps the base could take, in an order the base would not take |
| **elicited** | d = 1 (pend solved t in round 1 at k 32), or p_B(t) ≥ 1 / K_per |
| **neither** | never solved |

**Relative verdicts.** Relative to the round-(d−1) model, every theorem is elicited. Relative to the base, chained
theorems can be created. Both statements are true; which one is "the" answer depends on the comparison Dan wants. The
card recommends **relative to the base, at matched compute**.

## 3. Null or floor

- The budget per round (k 32) and per target (K_per) are explicit.
- A ladder from random init never starts: `rl-from-ckpt`'s p0 ladder accepted 0 of 287,680 attempts, so chains need a
  first link.
- That is the answer to the random-weights objection in this frame: random weights have no first link at any feasible
  k.

## 4. How to compute it here

- Free from the ladder logs (`cd_schema.py` already parses them) and from J2 (the base's reach at K_per).
- First numbers (family level; `out/schema_c12.txt`). `dist_and_over_or`: 0 / 40 targets at round 1 (pend, k 32) on
  every seed, then 0.97 / 0.90 / 0.78 by r8. Excluded middle on s1 first appears at rounds 2, 10, 12 and then bursts at
  13–14.
- Per-theorem chain depths for holdout250 (⊂ transfer): the set used in the agreement matrix (`cd_part3.py`, `chain`)
  is "first solved by the ladder's sampler at round ≥ 2, and compute-matched" (base not within reach at K_eval-set;
  the card's original K_per was replaced by the headline budget): 14 / 14 / 10 at r8 (x0), net of replay 12 / 8 / 5;
  r16 14 / 23 / 12. It is a subset of the compute-matched set (holdout250 members only; Jaccard 0.67), so on
  holdout250 most theorems the base cannot reach were first solved by the ladder after round 1, as a chain of
  elicitations would predict; as the critic said, this is a mechanism column, not a verdict.

## 5. Sensitivity

- **Per-round budget k:** a larger k makes chains shorter. A long chain at k 32 can be one step at k 10³.
- **Replay:** each round also pretrains on 20,000 K12 records, so links are not pure elicitation. The replay-only
  control (`rl-from-ckpt`) gives the size of that effect.
- **Decoding:** ladders sample plain at T 0.8 with the ladder caps (`max_action` 256 / `max_steps` 48), not the read
  caps.
- **Representation, renaming:** per ladder.
- **Seed:** chains are seed-specific (LEM on 1 of 3 seeds).

## 6. Failure modes

- **Transfer between targets.** A chain may reach t through generalisation from other targets, not through any sampled
  proof of t. This is still "created relative to the base", but not "a chain of elicitations of t".
- **"Created by chaining" depends on the base-reach certificate,** which has the `passk-budget` limits.
- **Not a property of R alone:** two ladders reaching the same model by different paths get the same R but different
  chains.

## 7. Relations

- Refines `passk-budget` (K_per vs per-round k).
- Explains `compute-equivalent`'s large RL multipliers: RL compute is spent at the frontier.
- The theoretical counterpart is the RL query barrier (below).

## 8. Literature anchor

- **Mousavi-Hosseini & Erdogdu 2026 (2603.06957):** outcome-reward RL needs about 1 / Q_q(ε) queries; the barrier is "a
  fundamental property of post-training with outcome rewards" (Sec. 5.1). Theorems solved with p_B ≪ 1 / queries are
  not explained by sharpening (L5 note).
- **Xie et al. 2024, XPO (2405.21046):** on-policy methods are stuck while coverage is low (Prop. 2.1).
- **Greenblatt et al. 2024 (2405.19550v1):** expert iteration "is able to pick up on a few lucky examples … and
  gradually bootstrap" (App. D.2).
- **Earlier reviews:** Go-Explore (return-then-explore), Abdulsalam et al. (iterated shifts reach 0 %-at-pass@1024
  buckets).
- **Project:** `rl-continue` (the excluded-middle burst at r13–14).

## 9. Critic's verdict

**Strongest argument (critic): the chain does no work, so the verdict is just `passk-budget` at K_per.**
- If p_B < 0.05 / K_per is certified, d ≥ 2 already follows (P(d = 1) ≤ 32 × 5 × 10⁻⁵ ≈ 0.2 %). On J2's candidates the
  d filter drops 0 / 39, 1 / 36 and 3 / 35.
- "Each link elicits" is never true of t itself: the link into round d took t from 0 / 32 to ≥ 1 / 32 without
  training on any proof of t, i.e. at the card's own k 32 that link *created* t relative to its predecessor.
- Counterexample: `la_transfer_1398` (s1). pend is at 0 / 768 and d = 2, so the rule says "created by chaining", but
  the replay-only control (no elicited proofs at all) solves it 509 / 512, more often than r8. The control also solves
  11 / 37, 12 / 34 and 16 / 29 of the d ≥ 2 candidates pend never solved.
- Secondary arguments:
  - The `dist_and_over_or` headline omitted that pend proved one transfer member in round 1 on all three seeds.
  - Deeper chains face an easier base bar.
  - Links also inject 20,000 replay records each, so "elicitation" links are not pure.

**My answer: accepted; the definition is dropped as a decision rule.** Chain depth d is kept as a *mechanism* column on
the replay-netted `passk-budget` created set: it describes how RL reached a theorem, round by round, and the
excluded-middle burst is the clearest such story. The card's lasting point, that "elicitation is not transitive", is
correct but does not decide anything; the base certificate does.

