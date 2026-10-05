---
written_on: 2026-10-05
written_by: agent:claude
---

> Reading copy of run `capability-defs`'s `capability_defs/cards/bits-over-null.md` (fork branch `dan_capability-defs`, DONE
> `645aa4f0`); reviewer's corrections marked *[review]*. Report: [`../REPORT.md`](../REPORT.md); review:
> [`review_capability-defs.md`](https://github.com/dan-pandori/nd-takehome/blob/da11be78/review_capability-defs.md).

# Card: bits of skill over a null model (and RL's share of them) — REJECTED as a decision rule (critic, §9)

Family N (null-relative). Slug `bits-over-null`. Notation: `_FRAME.md`.

## 1. Definition, formally

- For a theorem t and a valid proof y (or the known-proof sum, `marginal-bracket`), the **skill of θ over a null ν** is
  S_θ|ν(t) = log₂ π_θ(y | t) − log₂ π_ν(y | t) bits, at T 1.0. It can be stated per proof, per theorem (using the
  marginal lower bound), or summed over a set.
- **Nulls:**
  - (i) π_0, the random-initialisation checkpoint of the same seed (step 0; scored for every target in `trajectory`
    and J1);
  - (ii) π_U, a policy uniform over the vocabulary, i.e. Harding & Sharadin's UNIFORM, which init approximates;
  - (iii) a weak reference model, e.g. an early pretraining checkpoint (p1600), for "bits beyond the format".
- **RL's share** of the skill: σ(t) = S_R|B(t) / S_R|ν(t) = [log π_R − log π_B] / [log π_R − log π_ν].
- **Expert bits** (Phuong et al.): ⌈−log₂ p⌉, the number of bits of help a model needs before it succeeds. For plain
  sampling, log₂ k-to-solve.

## 2. Decision rule

| verdict | rule |
|---|---|
| **created** | RL contributes most of the information: σ(t) > 1/2 for t, or the set-level sum of S_R\|B exceeds the sum of S_B\|ν |
| **elicited** | σ(t) small (the base already carried almost all the bits; RL added the last few) |
| **neither** | R does not solve t |

A **budget-relative** variant compares S_R|B with log₂ K, the bits of search that RL's compute buys: RL adds more than
search alone could if S_R|B > log₂ K (this is `passk-budget`'s "bits beyond search").

## 3. Null or floor

The null is the definition. It answers Dan's objection head-on: random weights have zero skill by construction, and
every model's skill is measured relative to them. The objection's "some k" is 2^(expert bits of the null), astronomical
here.

## 4. How to compute it here

- **Free for references and r8's eventual proofs** (trajectory scores at init, pend and r8). J1 extends this to every
  known proof (init / pend / r8 / r16).
- **First numbers** (`out/defs_c12.txt`, plus the null sweep in `log.md` 06:40). These are over the eventual proofs
  (r8's most probable accepted proof of each theorem r8 solves on x0), T 1.0:
  - Against **init**: RL's share of the bits has median 0.30 % over all 865 theorem–seed pairs. On the equal-k-created
    theorems ("B", where pend fails at k 256) the median is 1.44 / 1.17 / 1.00 % (s0 / s1 / s2), and 81–88 % of B is
    below 2 %. **σ > 1/2 never happens** (maximum 4.4 %).
  - Against **p200 / p800 / p1600 / p5000**: B medians are 14–19 % / 27–30 % / 31–39 % / 40–56 %.
  - So against init, pretraining supplies more than 95 % of the bits of nearly every proof. Against a step-5,000
    reference, RL supplies about half the bits of what it newly solves.

## 5. Sensitivity

- **Null choice dominates.**
  - Against init, almost all bits are format (grammar, names, brackets), learned in the first few hundred pretraining
    steps.
  - Moving the null to p1600 or p5000 raises RL's median share on B from ≈ 1 % to 31–56 % (§4).
  - Against "the base itself", RL's share is 100 %.
  - The verdict is a statement about the null as much as about RL.
- **Temperature:** T 1.0 is the model's own distribution; at T 0.8 the bits change by a few per cent.
- **Proof choice:** RL's own proofs (selected to be likely under R) give a larger S_R|B than references (`trajectory`:
  ≈ 3× larger Δ_RL).
- **Seed and redraw:** the B medians move 0.4 percentage points between seeds against init and up to 16 points against p5000
  (§4). Defining B on one draw selects theorems where pend's log p is low.

## 6. Failure modes

- **Format bits swamp everything.** The last few bits are the hard ones, yet the share counts every bit alike, so it
  can never call RL creative when the base already writes well-formed text.
- **Bits are not difficulty.** Twenty bits of RL gain can be the difference between "never" and "always" (2²⁰ ≈ 10⁶
  attempts).
- **It is gameable by choosing the null:** any desired verdict follows from picking the reference.

## 7. Relations

- With ν = B it is the log-ratio inside `passk-budget` and `tf-proof-prob`.
- With ν = an early checkpoint it approaches `compute-equivalent`.
- The information-theoretic view of `elicit-finetune` (bits a fine-tune must inject) is the training-side twin.

## 8. Literature anchor

- **Harding & Sharadin 2024 (2405.08989v1):** UNIFORM "always produces the next token by sampling uniformly from the
  token vocabulary"; "very few ability claims are actually true of UNIFORM" (Sec. 3.1).
- **Ethayarajh et al. 2022 (2110.08420v3):** pointwise V-information = log₂-probability gain over a null-input model
  (Sec. 3, Eq. 4).
- **Xu et al. 2020 (2002.10689):** usable information is defined relative to a predictive family (L3 note).
- **Phuong et al. 2024 (2403.13793):** expert bits (Sec. 6.2; L2 note).
- **SEP "Abilities":** Kenny's darts player — a lucky success is not an ability (Sec. 4.3).

## 9. Critic's verdict

**Strongest argument (critic, 2026-10-05): σ cannot detect anything RL does.**
- Whenever π_ν ≤ π_B ≤ π_R ≤ 1 on y, σ ≤ log π_B(y) / log π_ν(y). So "created" (σ > ½) requires the base to still carry
  more than half of the null's surprisal on y, whatever RL does.
- Init costs 34–157 nats per step, while pend never exceeds 3.1 nats per step on any r8 proof. A perfect RL policy
  would still score ≤ 5.1 %, and "created" is empty **by construction**: 0 of 865 per proof, ≤ 3.9 % per theorem on
  s0's hard set.
- Counterexample: `la_transfer_205` (s0). pend solves 0 / 768 and puts e^(−32) on all 287 known proofs together; r8
  solves 79 %. Yet σ = 1.6 %, so the card says "elicited", while six other definitions call it created.
- With a checkpoint as the null, σ is only a dial: the share of B called created is 0 % at p200, 20–24 % at p1600,
  37–61 % at p5000 and 100 % at pend. Pretraining is not monotone, so σ can exceed 1, and RL's choice of route games
  the per-proof version.

**My answer: accepted; the definition is dropped as a decision rule** and kept in the catalogue as a documented
negative result.
- The share-of-bits form inherits the null's format bits and so mirrors Dan's objection instead of answering it.
- What survives contains no null: RL's gain in bits against the bits of search its compute buys, log₂ p_R − log₂ p_B −
  log₂ K. That is `passk-budget`'s "bits beyond search".
- The genuine answer to the random-weights objection, K_null ≫ K_total, is stated in `_FRAME.md` and `passk-budget` §3.

