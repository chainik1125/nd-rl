---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - blier2018description
---

# The description length of deep learning models (prequential coding)

Paper: [@blier2018description] (Blier & Ollivier, NeurIPS 2018)
Source: arXiv 1802.07044v5 (HTML rendering read: abstract, Sec. 1-5, App. A; App. B-E not read)

## Learnings

- **MDL premise.** MDL holds "that a good model of data is a model that is good at losslessly compressing the
  data, including the cost of describing the model itself" (Abstract). The gain of any code over the uniform
  code "is limited by the amount of mutual information between input and output" (Sec. 2.4, Eq. 2.4).
- **Prequential (online) code** (Sec. 3.4, Def. 3, Eq. 3.6): "a model with default values is used to encode
  the first few data; then the model is trained on these few encoded data; this partially trained model is
  used to encode the next data" and so on; "The model parameters are never encoded explicitly in this
  method." The gap between the prequential codelength and the final model's log-loss "can be interpreted as
  the amount of information that the trained parameters contain about the data" (Sec. 3.4).
- **Fake labels carry no information even when fitted perfectly.** Networks reach 100% train accuracy on
  random labels, but the compression bound shows "these models do not compress fake labels", "that no
  information is present in the model parameters, and that no learning has occurred" (Sec. 1, Fig. 1). With
  true labels "half of the description length is information contained in the weights" (Fig. 1 caption).
  App. A, Prop. 2: for uniform random labels, with probability 1 − 2^−δ every code has length
  ≥ n log2 K − δ − 1; "most values of Y_1:n can just not be compressed by any algorithm".
- **Numbers.** Uniform code for MNIST labels: 60000 × log2 10 = 199 kbits (Sec. 2.3). Prequential: 4.10 kbits,
  "a compression ratio of 0.021", 99.5% test accuracy; "This codelength is 6 times smaller than the
  variational codelength" (Sec. 3.4; Table 1). CIFAR10: 45.3 kbits, ratio 0.27 (Sec. 3.4). A float32
  two-part code for 1 M parameters is 32 Mbits, "or 200 times the uniform encoding on CIFAR10" (Sec. 3.1).
  The intrinsic-dimension two-part code gives > 9.28 kbits at 90% test accuracy on MNIST (Table 1, weights
  only).
- **Variational codes are loose.** "these variational methods provide surprisingly poor compression bounds"
  while "simple incremental encoding methods yield excellent compression values on deep networks"
  (Abstract).
- **Classical asymptotics do not apply.** Two-part, Bayesian and ML-prequential codes coincide at
  nH(Y|X) + (d/2) log2 n + O(1) under regularity assumptions; "This corresponds to the BIC criterion for
  model selection", but deep nets violate the assumptions (Sec. 4).
- **Caveats stated by the authors.** "Prequential codes depend on the performance of the underlying training
  algorithm" (Sec. 3.4). "A weakness of prequential codes is the catch-up phenomenon": VGGb "needs 5,000
  samples on CIFAR to reach a cumulative compression ratio" below 1 though per-label cost drops below uniform
  after 1,000; model switching fixes it (Sec. 3.4, Fig. 2-3).

## Evidence and limitations

- Evidence: Table 1 (uniform / two-part / network compression / intrinsic dim. / variational /
  prequential on MNIST and CIFAR10), Fig. 1 (true vs fake labels), Fig. 2 (prequential on CIFAR).
- Classification only; all codelengths are theoretical (no actual coder). Appendices B-E (intrinsic-dim.
  code, architectures, switching) not read; the switching numbers (Table 2) were not checked.

## Connections and questions

- **Definition offered:** the information a trained model holds about a dataset = prequential codelength
  minus the final log-loss; a "good model" = short total code (data + model). A perfect fit can contain
  zero information (fake labels).
- **New vs better access:** not discussed. As a rule for us: the **bits a fine-tune had to inject** to
  produce behaviour D from checkpoint θ0 is ΔL(θ0) = L_preq(D | θ0) − L(D | θ_final). Elicitation ⇔ ΔL(pend)
  is small (and a small fraction of ΔL(init)); creation ⇔ ΔL(pend) is large, near ΔL from a model that never
  saw the relevant structure. A second, RL-specific bound follows from the same counting argument as App. A:
  in expert iteration the only information from outside the model is the verifier's 0/1 outcomes, at most
  k = 32 bits per target per round (far less when outcomes are predictable). If the base model's code for the
  new behaviour (−log2 π_pend summed over the needed proofs, or its prequential code) is far above that
  budget, RL cannot have "taught" it by transmitting it; it must have been built from what the model already
  generated. This is our inference, not the paper's; the same bound is argued for policy gradient in
  "LoRA Without Regret" (Schulman / Thinking Machines 2025: per-episode information ≤ H(advantage), "O(1)
  bits per episode", and zero when the initial policy never gets reward; row and claims 197-203 in
  `_screen_L3.md` / `_claims_L3.md`).
- **Null / floor:** the uniform code n log2 K is the explicit floor-of-learning reference, and random labels
  show what "no learnable structure" looks like (code ≈ uniform). For us the random-init checkpoint plays the
  uniform role (per-token ≈ log2 |V|), which makes "any k solves it" quantitative: init needs ≈ length ×
  log2 |V| bits for a proof, pend fewer, r16 fewer still.
- **Transfer to our setting:** compute L_preq for (a) the excluded-middle family, (b) the accepted proofs RL
  added between pend and r16, starting from init, intermediate pretraining checkpoints and pend, with the
  same fine-tuning recipe; first block coded by the starting checkpoint's teacher-forced log p. Cost: ~10
  fine-tunes per starting point × a few data orders, minutes each. Failure modes: dependence on the learning
  algorithm and on block schedule (catch-up); RL-selected data is biased toward what the policy already
  liked, so use a fixed reference set; the information-budget bound is loose because samples themselves
  come from the model.
- Related: `voita2020mdlprobing.md` (same code applied to probing, with random-init baselines),
  `whitney2020evaluating.md` (SDL fixes the dataset-size dependence), `aghajanyan2020intrinsic.md` (the
  intrinsic-dimension code appears in Table 1 here), prior-screened Donoway et al. 2601.04728 (excess
  description length; context only).
