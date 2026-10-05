---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - xu2020theory
---

# A theory of usable information under computational constraints (V-information)

Paper: [@xu2020theory] (Xu, Zhao, Song, Stewart, Ermon, ICLR 2020)
Source: arXiv 2002.10689v1 (HTML rendering read: abstract, Sec. 1-4, 6.3, 8, App. F; proofs and other
experiments skimmed only)

## Learnings

- **Motivation: information that is present but not usable.** "consider a dataset of encrypted messages
  intercepted from an opponent": in Shannon's sense they carry high mutual information with the plans, yet
  only an unbounded agent can use it (Sec. 1).
- **Definitions.** A predictive family V "is a set of predictive models the agent is allowed to use, e.g.,
  due to computational or statistical constraints" (Def. 1), with the technical condition "optional
  ignorance" (the agent may ignore the side information; Eq. 1). Conditional V-entropy H_V(Y|X) =
  inf_{f∈V} E[−log f[x](y)]: it "is the smallest expected negative log-likelihood that can be achieved
  predicting Y given observation (side information) X" using models from V (Def. 2). Predictive
  V-information I_V(X→Y) = H_V(Y|∅) − H_V(Y|X), "to represent the change in predictability of an output
  variable Y when given side information X" (Def. 3, Eq. 3).
- **Special cases.** With V = all models Ω, the quantities reduce to Shannon entropy, conditional entropy
  and "the Shannon mutual information" (Prop. 1.1); with linear-Gaussian V, V-information is the
  (unnormalised) coefficient of determination R² (Prop. 1.5).
- **Properties.** Monotonicity in V, non-negativity, zero under independence (Prop. 2). Unlike Shannon
  information, "in violation of the data processing inequality, V-information can be created through
  computation" (Abstract); e.g. decryption: "processing increases the "usable information"" (Sec. 3.2). It
  is asymmetric: "it is easy to predict Y from X but not vice versa" (Sec. 3.3).
- **Estimation.** Empirical V-information (Def. 4, Eq. 4) concentrates with a Rademacher-complexity PAC
  bound (Thm. 1, Eq. 5); "a complex function family V (i.e., with large Rademacher complexity) could lead
  to overfitting" while an overly simple V misses the relation (Sec. 4).
- **Example where Shannon fails and V works.** On deterministic Moving-MNIST, "every pair of frames has the
  same mutual information", so Chow-Liu with Shannon MI cannot order frames; with V = PixelCNN++ the order is
  recovered for frame distances below 9 (Sec. 6.3).
- **Limitations stated.** "Shannon information can be manipulated with certain additive algebra" (chain
  rule) but the same "does not hold true for general" V-information (App. F). The authors suggest
  "exploitation of usable information (classification and reinforcement learning) could potentially be
  framed" the same way (App. F) — not developed.

## Evidence and limitations

- Mostly definitional/theoretical (Prop. 1-2, Thm. 1); experiments: Chow-Liu structure learning, gene
  networks, frame ordering (Fig. 1c), fairness (App. D.2, Fig. 3b). Proofs (App. A) not checked.
- V must be chosen; results are always relative to it. No guidance on choosing V for a given question.

## Connections and questions

- **Definition offered:** knowledge/informativeness is **usable information relative to a bounded observer
  family V**: the drop in the best achievable expected log-loss on Y when the observer in V may use X.
- **New vs better access:** this is the cleanest formal account of the distinction we need, though the
  paper does not apply it to RL. In Shannon terms, RL against a verifier cannot add information about
  proofs beyond what base model + verifier already determine (data processing), and "some k solves it"
  is the Shannon view of an unbounded searcher. In V terms, RL is computation (search + selection) that can
  **create usable information** for a bounded family, exactly like decryption (Sec. 3.2). So the
  create/elicit question becomes well-posed only once V is fixed: e.g. V_K = "best-of-K sampling from the
  checkpoint with the verifier", or V_m = "the checkpoint plus any fine-tune of ≤ m bits / rank r". A
  capability is **elicitable** from pend if it is usable within V(pend) at the chosen budget, and
  **created** by RL if it is usable within V(r16) but not within V(pend) at the same budget. The budget
  (K, m, r) is the decision parameter, and its natural value is RL's own compute or information budget.
- **Null / floor:** answers the random-weights objection directly: for V = all models (unbounded k) every
  checkpoint has the same information; the objection disappears once V is bounded. Independence gives
  I_V = 0 (Prop. 2.3); the random-init checkpoint gives the empirical floor for any family built on a
  checkpoint.
- **Transfer to our setting:** take Y = a correct proof of theorem t, X = t. For a family V(θ) =
  {fine-tunes of θ restricted to rank r or ≤ n demonstrations}, estimate H_{V(θ)}(Y|X) as the held-out
  teacher-forced NLL of reference proofs after the restricted fine-tune, for θ ∈ {init, pend, r8, r16}.
  Usable information of θ = H_{V(init)} − H_{V(θ)}; RL's created share = that of r16 minus that of pend at
  equal r/n. Cost: one restricted fine-tune + scoring per (θ, r); minutes. Failure modes: results depend on
  V (must sweep r / n and report curves); no chain rule, so "shares" do not add exactly (App. F);
  reference-proof NLL ignores alternative proofs (see `lin2023urial.md` and the likelihood screen rows).
- Related: `ethayarajh2022vusable.md` (pointwise version, per-instance difficulty),
  `hewitt2021conditional` row in `_screen_L3.md` (conditional V-information: information beyond a
  baseline), `whitney2020evaluating.md`, `voita2020mdlprobing.md`.
