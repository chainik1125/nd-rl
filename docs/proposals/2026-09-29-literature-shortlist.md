---
written_on: 2026-09-29
written_by: agent:claude
status: proposed — for Dan; nothing started
---

# Proposal 16: what the literature adds, re-ranked after `state-cap12`

Dan's hypothesis (2026-09-29): the proof state taken from AlphaProof was low-hanging fruit, and there is
more. The `lit-review` run screened 58 papers and read 23 in depth. Its shortlist was reviewed, with every
checked claim verified ([`REVIEW.md`](../literature/2026-09-29-lit-review/REVIEW.md);
[summary](../../experiment-summaries/2026-09-29-lit-review-untried-techniques-search-expert-and-frontier-supply/README.md)).
This proposal merges that shortlist with proposal 14's open items and re-ranks the result in the light of
two things the review did not have:

1. **`state-cap12` saturates the length pool.** The state policy trained on cap-12 proofs, after EI
   (SN-cap12 T1, 3.2 M, `lean_staten`), solves 233–320 of the 380 pool theorems at `L_true` 13–16
   (k 256). It solves 36–66 % of the `≥ 17` file, and `L*` ≥ 17 on 4 / 4 seeds. The whole-proof
   comparator solves 79 / 72.
   ([summary](../../experiment-summaries/2026-09-29-state-cap12-state-and-cap12-proofs-compound-on-long-pool/README.md))
   Its frontier can no longer be located. **Every frontier experiment below needs a longer, labelled
   pool first.**
2. **The review's "do first" measurement is mostly done.** Scoring EI's proofs of base-unreached
   theorems step by step under the base is `support-followups` part D. For the 29 survivors it found
   28 concentrated, 1 mixed and 0 spread: the improbability sits in 2–3 steps, which reads as "a new move".
   The same rule labels 42 / 53 control proofs concentrated, so the survivors differ from the controls in
   degree, not in kind.
   ([summary](../../experiment-summaries/2026-09-28-support-followups-expansion-not-removed-by-8x-params/README.md))
   Neither the lit-review executor nor its reviewer caught the overlap. What remains of it is item M1.

**Proposal 14 status.** Done: 2 (support curves on the state model = `support-state`), 7 (GRPO
replication review), 8 (checkpoint averaging). Open: 1 (search), 3 (GRPO in the state environment),
4 (textbook schemata). Its branch 5–6 did not apply (the state moved the wall).

**Default model.** New arms use **SN-cap12** unless the technique needs whole-proof generation. Four
Stage-1 seeds exist; each extra seed is well under an A40 hour, a few tens of cents.

**Statistics, for every item.**
- Pre-register the paired design's minimum detectable difference.
- Report per-seed values plus IQM with a stratified-bootstrap 95 % CI (`AGENT_POLICY.md`).
- Do not read "≥ 5 of 6 pairs" as significant: a sign test gives one-sided p ≈ 0.11.
- The variance of two EI arms from one shared checkpoint is unmeasured (reviewer finding F3). The first
  paired run measures it, by running its control arm twice on two seeds.

## Ranked

| # | item | why now | cost (est.) | depends on |
|---|---|---|---|---|
| 0 | **Long-pool-2**: pool above `L_true` 17, plus the step-cap re-read | the best model is off the end of the current pool | ≈ $6–10 | — |
| 1 | **Frontier target supply** | long-pool: the lack of solvable targets past `L*` is the best-evidenced bottleneck; small code | ≈ $15–20 | 0 |
| 2 | **Search as the EI expert** (value head only if search helps) | the most direct test of "search creates capability"; proposal 14 item 1, sharper design | ≈ $15–25 + 1 session of code | 0 |
| 3 | **Support-expanding GRPO advantages** | GRPO ignites depth-3 from zero (run 4 review); these objectives target support directly | ≈ $10–20 | — |
| 4 | **Transductive EI on textbook theorems + variants** (AlphaProof TTRL) | textbook schemata are the wall that survived every data change | ≈ $8–15 | read-off first |
| 5 | **Matched 2 × 2**: whole-proof vs state × cap 6 vs cap 12 | needed before "state and cap compound" is a claim | ≈ $20–25 | — |
| M | cheap measurements (below) | change how 1–5 are read | ≈ $1–3 | — |

Everything together is ≈ $75–120 of pods against $112 of headroom ($212.52 balance, $100 floor). Items 0,
1 and M fit comfortably. Choose among 2–5.

### 0. Long-pool-2 (prerequisite)

- **Labels above 16.** Labelling above 16 by `minlen.py` search is exponential. Instead, label each theorem
  with an **interval**:
  - a certified lower bound of 17: `minlen` stage D, bound 16 / 600 s, finds no proof;
  - an upper bound: the generator's construction length after pruning.
- **Pool.** Bin by the upper bound, 17–18 / 19–20 / 21–22 / 23–24, ≈ 100 theorems per bin, disjoint by
  renaming class. Keep the ≥ 17 theorems from `long-pool` as a calibration bin.
- **Step cap.** Re-read SN-cap12 T1 s2–s3 at `max_steps` 96 (state-cap12 reviewer, next measurement 1).
- **Expectation.** SN-cap12 T1 solves ≥ 10 % of the 17–18 bin and < 10 % of the 23–24 bin at k 256, so the
  frontier falls inside the pool. Falsifier: ≥ 10 % in every bin, which means the pool must go longer
  again.

### 1. Frontier target supply

Source: STP 2502.00212v4 §3.2; Lee et al. 2502.01612v2 §7.1; Goedel-Prover-V2 2508.03613v1 §2.2.

- **Mechanism.** EI learns only where p̂ > 0. Build new targets just past the frontier:
  - extend or mutate theorems the model already solves;
  - take the open subgoals of failed attempts from the state environment.

  Keep a candidate only if 32 attempts give p̂ ∈ (0, 1/4].
- **Design.** Six SN-cap12 seeds (four exist, plus two new). Arms: T1 vs T1 + supply, same rounds, same
  total attempts. Read out on long-pool-2.
- **Expectation.** ≥ 5 % of candidates pass the filter, and the supply arm's per-seed `L*` and bin-wise
  solve counts beat T1 by more than the pre-registered MDD.
- **Falsifier.** < 5 % pass the filter (the generator cannot make the shapes), or no difference beyond the
  MDD.
- **Cost.** Small code (a mutation script beside `gen.py`; open goals from `state_sample.py` at failure),
  ≈ $15–20.

### 2. Search as the EI expert

Sources: BFS-Prover 2502.03438v3 Eq. 1; HTPS 2205.11491v1 Table 4 (minimal proofs 78.1 vs all proofs 40.6);
ExIt 1705.08439v4.

- **Mechanism.** Best-first search in the state environment on length-normalised log-prob. Train only on
  the minimal proof of each solved node.
- **Evaluation.** Evaluate the apprentice **without search**, at k 256 on long-pool-2.
- **Arms.** Sampling EI; best-first EI at an equal number of generated actions; truncate-and-resume
  (DeepSeek-Prover-V1.5 §3.1) as the cheap baseline on two seeds.
- **Caution.** The one length-specific result in the sources favours a critic: InternLM2.5-StepProver
  finds mean proof length 4.44 with a critic vs 1.66 with best-first. So the expectation that best-first
  finds longer proofs has no source support, and it is stated as a guess.
- **Value head.** A steps-to-go value (AlphaProof Methods; HTPS Table 5: soft critic 78.1, hard 63.1,
  none 65.6) comes only if best-first beats sampling. A bad value is worse than none.
- **Cost.** One executor session of code (an `Env` clone or replay, per-action log-probs from the
  sampler), then ≈ $15–25.

### 3. Support-expanding GRPO advantages

Source: He et al. 2506.02355v2 Table 2; Chen et al. 2508.10751v1.

- **Design.** Arms on the whole-proof GRPO (`grpo.py:117` computes the advantage): default, unlikeliness
  rank penalty, pass@4 advantage, and a distinct-proof bonus. Six new C0-style Stage-1 seeds.
- **Readout.** The `support-curves` protocol, paired per theorem.
- **Caveats.**
  - Whole-proof only, so it runs on the weaker model family.
  - Zero gradient on all-fail groups, so any gain on base-unreached theorems must come from transfer.
- **Cost.** ≈ $10–20. Porting GRPO to the state environment (proposal 14 item 3) would be the stronger
  version, at another session of code.

### 4. Transductive EI on textbook theorems

Source: HTPS §7.1.3; AlphaProof Methods (TTRL on programmatic variants).

- **Read-off first, $0.** How SN-cap12 T1 does on the 282 textbook theorems of `transfer_long` (the files
  are in the bucket). If it already solves the schemata that were stuck at ≤ 2 of 40, drop this item.
- **Design (otherwise).**
  - Split the textbook theorems in half.
  - Arms: frozen, EI on half A, EI on half A + variants (drop or strengthen a hypothesis, split a goal).
  - Report half A (transductive) and half B (transfer) separately, on 6 seeds.

### 5. Matched 2 × 2

`state-cap12`'s headline is 4 vs 2 seeds against inherited comparators with different replay sets. Run
{whole-proof, state} × {cap 6, cap 12} under one ladder protocol, under Lean alone, with ≥ 4 seeds per
cell and the interaction pre-registered. This turns "the two levers compound" into a claim. ≈ $20–25; it
reuses existing seeds where they match.

### M. Cheap measurements (≈ $1–3 together)

- **M1.** Extend `support-followups` part D to the state models' EI-only theorems (the 7 that SN EI solves
  and the SN base does not at 200k per temperature, `support-state`). Add the Invisible Leash bound as a
  second reading: a step below ln(3 / 400k) ≈ −11.8 is out of the base's effective reach. The
  survivors' median worst step was −11.0, just inside it. No sampling.
- **M2.** Init-seed × data-order seed, a 3 × 3 Stage-1 grid (Zhou et al. 2402.09371v1 §4.2), about $1.
  Splits `NOISE_FLOOR.md`'s per-run variance into its two sources.
- **M3.** Per-round mode shares in existing EI and GRPO found files (Echo Chamber 2504.07912v2). No compute.

## Recommendation

Launch **0 and M** now: ≈ $8–13, and they change how everything else is read. Then run **1**, and **2**
if its code session is affordable in Claude quota. The two can share long-pool-2 and run on the two hosts
in parallel. Hold 3–5 until 1 and 2 report. The rejected techniques and the reasons are in
[`REVIEW.md` §(b)](../literature/2026-09-29-lit-review/REVIEW.md); none of them is worth reopening on
current evidence.

## Appendix: item 2 in plain terms (for Dan, 2026-09-29)

**What we do now.** Expert iteration (EI) has two parts: an *expert* that finds proofs, and an
*apprentice* (the network) that is trained on what the expert finds. In our loop the expert is the
apprentice itself: for each target theorem the network makes 32 independent attempts, from the first step
to the last. Lean keeps the successes and the network is fine-tuned on them. The expert is therefore no
stronger than the apprentice. EI can only reinforce proofs the network already finds by chance.

**Why that stalls on long proofs.** Independent attempts compound errors. If each step of a 20-step proof
is right 80 % of the time, a whole attempt succeeds 0.8²⁰ ≈ 1 % of the time. A mistake at step 15 throws
away the 14 good steps before it, and the next attempt starts from scratch.

**What search changes.** Best-first search keeps a list of *partial* proofs, ranked by how likely the
network thinks each one is (log-probability, adjusted for length). It repeatedly takes the most promising
one and tries a few possible next steps from it (width 4). The state environment rejects an invalid step
at once, so dead ends are dropped early and the search moves to the next-best partial proof. Good
prefixes are reused instead of re-sampled. This is what makes AlphaZero's and AlphaProof's expert
stronger than their network: in AlphaZero it is tree search (MCTS). Search also finds several proofs of
the same theorem. HTPS found that training only on the *shortest* one mattered a lot (78.1 % vs 40.6 %
in its Table 4).

**The experiment.**
- Take 6 seeds of the state model. From each seed's checkpoint, run two EI ladders with the same compute
  (the same number of generated steps):
  - (A) the current sampling expert;
  - (B) the search expert.
- On 2 seeds, run a cheap in-between arm (C): when an attempt fails, restart from just before the failing
  step instead of from scratch.
- Then **test the trained networks without search**: plain sampling, 256 attempts per theorem, on the long
  pool. This matters. The question is not whether search finds more proofs (it will). It is whether a
  network trained on search's proofs has itself become able to prove more.

**What results would look like.**
- **The core result:** per seed, the frontier `L*` and the number solved in each length bin, for B and A
  on the same starting checkpoint (paired).
- **Also:** proofs found per round by each expert, and how long they are.
- **If B beats A beyond the noise floor:** a stronger expert teaches the network things it would not reach
  by filtering its own samples. That is the AlphaZero mechanism, it bears directly on create vs elicit,
  and it would make search a standard part of our RL loop.
- **If B ≈ A:** search finds more during training, but the network does not absorb it. The benefit would be
  test-time only, or search's proofs are "more of the same".
- **If B < A:** search's proofs (for example, the shortest found) are off-distribution enough to hurt.
- **If C ≈ B:** the cheap restart trick gets most of the benefit, with no search machinery.

**Why it is not first.**
- It needs real code: copying environment states, per-step probabilities from the sampler, and the search
  loop. That is roughly one executor session of Claude quota, then ≈ $15–25 of pods.
- Item 1 attacks the same stall (no learning signal past the frontier) more cheaply. It supplies easier
  new targets instead of a stronger expert.
- A learned "steps to go" value to guide the search (AlphaProof's) comes only if B beats A. In HTPS a poor
  value did worse than none.
