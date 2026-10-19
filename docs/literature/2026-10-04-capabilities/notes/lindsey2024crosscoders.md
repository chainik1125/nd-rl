---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - lindsey2024crosscoders
  - minder2025overcoming
---

# Crosscoder model diffing: shared vs model-specific features (and why 'base decoder norm ≈ 0' overstates novelty)

Papers: [@lindsey2024crosscoders] (Lindsey, Templeton, Marcus, Conerly, Batson, Olah; Transformer Circuits research
update, Oct 2024); critique and fix in [@minder2025overcoming] (Minder, Dumas, Juang, Chughtai, Nanda; arXiv
2504.02922).
Sources: https://transformer-circuits.pub/2024/crosscoders/index.html (HTML, read in full: Sec. 1-5); arXiv
2504.02922v4 (HTML, read: Abstract, Sec. 2.2-3.1, 5; Sec. 3.2-3.3 and appendices not read). Minder et al. is also a
row in `_screen_L6.md`.

## Learnings

- **Status.** Lindsey et al. is explicitly "preliminary work that we're excited about, but not at the level of quality
  or rigor we hold our full papers to" (research-update box).
- **Crosscoder.** A sparse dictionary whose shared latent code reads from several layers or models, f(x) =
  ReLU(Σ_l W_enc^l a^l(x) + b), with a separate decoder per layer/model; the sparsity penalty weights each latent's
  activation by the sum of its per-layer decoder norms ("we weight the L1 regularization penalty by the L1 norm of the
  per-layer decoder weight norms", Sec. 2). This L1-of-norms choice matters for diffing: it "uncovers a mix of shared
  and model-specific features, while the L2-of-norms version results in uncovering only shared features" (Sec. 2).
  Uses: "one model across training or finetuning" (Intro), including "Training Snapshots" (Sec. 4.2).
- **Diffing Claude 3 Sonnet vs its base.** "We trained a crosscoder with 1 million features on the residual stream
  activations from the middle layer of Claude 3 Sonnet and the base model from which it was finetuned" (Sec. 4.3).
  Model-specific features "would indicate features learned, or forgotten, during finetuning"; by relative decoder norm
  "features cluster into three obvious groups", with "between four and five thousand model-specific features for each
  model, out of a total 1 million features" (Sec. 4.3). Examples are cherry-picked (e.g. "A refusal feature that
  activates on dangerous requests"), and "the majority of the model-exclusive features are not immediately
  interpretable" (Sec. 4.3).
- **Shared but re-used.** For shared features with poorly aligned decoders ("for a few thousand features, the
  correlation was very low or even negative") the authors "suspect that these indicate cases where the finetuned model
  uses a concept that was present in the base model, but in a new way" (Sec. 4.3). Base-SAE transfer (Kissane et al.)
  gives a different diff; "We suspect that the crosscoder approach is preferable when model finetuning involves a
  greater fraction of compute relative to the compute used for pretraining" (Sec. 4.3). Diffing results overall "are a
  bit mixed" (Sec. 5.1); "Crosscoder errors may be important and extremely difficult to interpret" (Sec. 5.2).
- **Minder et al.: the decision rule is biased toward "new".** With Δnorm(j) = ½(1 + (‖d_chat‖ − ‖d_base‖)/max(...)),
  latents are classed "base-only (0-0.1), chat-only (0.9-1.0), or shared (0.4-0.6)" (Sec. 2.2, Eq. 3). Two L1-loss
  artefacts "can misattribute concepts as unique to the fine-tuned model, when they really exist in both models"
  (Abstract): *Complete Shrinkage* ("When the contribution of latent j is smaller in the base model than in the chat
  model, L1 regularization can force d^base_j to zero despite its presence in the base activation") and *Latent
  Decoupling* ("a chat-only latent j is also present in the base activations but is reconstructed by other base decoder
  latents") (Sec. 2.2). **Latent Scaling** tests presence directly: fit β_j^base = argmin_β Σ‖β f_j(x) d_j^chat −
  h^base(x)‖² and report ν_j = β^base/β^chat; "A value near zero indicates a chat-specific latent, while a value near
  one suggests the latent is equally present in both models" (Sec. 2.3, Eq. 4), with separate ratios on the base
  reconstruction (ν^r) and error (ν^ε). On Gemma 2 2B, "most L1 crosscoder chat-only latents are not truly
  chat-specific (defined as ν^r<0.5 and ν^ε<0.2), while most BatchTopK chat-only latents are genuinely" so (Sec. 3.1,
  Fig. 2); similar effects appear in Llama 3 chat models and "models fine-tuned with RL for reasoning and medical
  knowledge" (Sec. 3.1).
- **Unresolved even after the fix.** Minder et al.: "our inability to distinguish between truly novel latents learned
  during chat-tuning and existing latents that have merely shifted their activation patterns" (Sec. 5), and BatchTopK
  "error terms still contain a lot of information about the chat model behavior" (Sec. 5).

## Evidence and limitations

- Lindsey et al.: one diff (Claude 3 Sonnet), qualitative examples, no ablations or causal tests of model-specific
  features, no quantitative null (how many 'model-specific' features would two seeds of the *same* model produce?).
- Minder et al. supply the missing control (Latent Scaling) and a causal check (Sec. 3.2, not read), but the two key
  thresholds (ν^r < 0.5, ν^ε < 0.2) are chosen by the authors, and novelty vs 'shifted activation pattern' remains
  undecidable in this framework (Sec. 5 quote).

## Connections and questions

- **Definition offered:** a concept is present in a model if a sparse dictionary latent needs a non-zero decoder in
  that model to reconstruct its activations; *new in the fine-tune* = latent with near-zero base decoder norm (Lindsey),
  corrected to: near-zero base decoder **and** near-zero optimal scale when the latent's fine-tune decoder direction is
  fitted to base activations (Minder's ν ≈ 0). *Shared but re-used* = shared latent with misaligned decoders.
- **New vs better access:** yes, as feature-set membership: base-only / shared / fine-tune-only. Fine-tune-only and
  ν ≈ 0 → created representation; shared with aligned decoders but changed activation frequency → elicited (old concept
  used more often); shared with misaligned decoders → old concept, new downstream use (the same 'repurposing' category
  as `ward2025repurposes.md`). Minder's limitation means the rule cannot separate 'new latent' from 'old latent firing
  in new contexts' without extra activation-statistics tests.
- **Null / floor:** none in Lindsey. The natural null for us: train the same crosscoder on two *pretraining*
  checkpoints separated by the same number of steps as RL, or on pend vs a second pend seed, and count 'model-specific'
  latents; RL-specific latents only count above that baseline. Minder's ν ratios are a per-latent null.
- **Transfer to our setting:** cheap at our scale: a BatchTopK crosscoder (say 4-16 k latents) on the 384-d residual
  stream at each of the 6 layers, pend vs r16, trained on activations from proof states of both models' own samples plus
  reference proofs (~10^6 token positions, minutes on one GPU). Report: number of r16-only latents with ν^r < 0.5 and
  ν^ε < 0.2, vs the pend-vs-pend' (seed or adjacent checkpoint) baseline; for the excluded-middle family, whether the
  latents most active before r16's classical step are shared (ν ≈ 1, elicited/repurposed) or r16-only (created); then a
  causal check by steering pend with the latent's r16 decoder direction (see `ward2025repurposes.md`). Because the
  pretraining checkpoints exist, a multi-snapshot crosscoder over init → … → pend → r8 → r16 can date each latent's
  appearance (Lindsey's "Training Snapshots"), which turns 'created by RL' into 'first appears after pend'. Failure
  modes: L1 artefacts (use BatchTopK + Latent Scaling), dictionary size and seed dependence of what counts as a latent,
  and the reconstruction error hiding exactly the RL-specific signal (Minder Sec. 5).
- Related: `venhoff2025base.md` (argues the diff may not be linear-feature-level at all), `ward2025repurposes.md`,
  `prakash2024finetuning.md`, `hewitt2019control.md`, `mukherjee2025subnetworks.md`.
