---
written_on: 2026-11-25
written_by: agent:claude
papers:
  - donoway2026bitsthatcount  # proposed key; not yet in references.bib
  - donoway2026excessdescriptionlengthlearning
  - donowayQuantifyingElicitationLatent
---

# Bits That Count: the empirical EDL companion paper

Paper: [@donoway2026bitsthatcount] (proposed key). Donoway, Joren, DeWeese, Perez, Schulman, Roger, Leike. "Bits That Count:
Quantifying and Predicting Capabilities of Language Models." Proceedings of the 43rd International Conference on Machine
Learning, PMLR 306:26127–26151, 2026.
Source: https://proceedings.mlr.press/v306/donoway26a.html. The PDF is the PMLR asset `donoway26a.pdf`. I extracted its text and read it
in full: the main text, Appendices A, D, G and H, and the I.1–I.6 tables. Proofs and Appendix J were skimmed. There is no arXiv
version. The theory paper is arXiv 2601.04728 (earlier review note `donoway2026-edl.md` on `origin/dan`).

## Learnings

What the paper claims:
- **Definitions.**
  - MDL is the first-epoch online loss, "accumulates loss over the first epoch only" (§2.2, footnote 1).
  - EDL(D) = MDL − n · L_test(θ*), where θ* is trained "to some termination condition (e.g., convergence, some number of epochs, or a
    fixed compute budget)" (§2.3).
  - In practice the stopping criterion is "Validation loss convergence", with 3 seeds per configuration (App. H, Table 1).
- **Loss mask.** Loss is computed "only on designated label tokens, typically the answer portion of the output … This convention helps to
  isolate capability-relevant learning from format acquisition" (§2.1).
- **Signatures.**
  - Elicitation is "monotonically decreasing" EDL per token as n grows; teaching shows "an initial increasing phase" (§4, Fig. 2, Table 5).
  - The signatures "characterize which process dominates, not a strict dichotomy" (§4).
  - The two processes cross over in both directions. Llama on multiplication becomes teaching-dominated at hundreds of thousands to millions of examples (§4.3, I.3.1).
- **Causal intervention.** Pre-teaching TinyStories-1B on multiplication changes the multiplication signature from teaching to elicitation. The
  capacity threshold falls from about 1.02 to about 0.05 bits/param, whereas pre-teaching addition/subtraction does not convert it (1.56) (§5.1, Table 6).
- **Capacity.**
  - EDL/P* is "the threshold where EDL/EDLref drops below 0.95" (Table 6 caption), with the reference = LoRA r512 (footnote 2).
  - §6.2: "elicitation … is associated with adapter capacity limits around 0.01–0.1 bits/parameter, whereas teaching saturates around 1 bit/parameter or more."
  - Values (Tables 6–7):
    - Llama 3.x 1–8B arithmetic: 0.004–0.03 by EDL (one PGR-based value of 0.14);
    - Alpaca: 0.02–0.03;
    - Qwen2.5 1.5B / 14B / 32B on R1 traces: 0.04 / 0.05 / 0.02;
    - TinyStories-1B base: 0.70–2.23.
  - "Larger models have lower capacity thresholds" (Fig. 4, §6.1).
- **Minimal information** (Table 8). One example, 10 parameters or 100 parameters give PGR of:
  - Llama 3.1 8B addition/subtraction: 0.96 / 0.91 / 0.96.
  - Qwen2.5-32B reasoning: 0.55 / 0.27 / 0.41 (+16 / +8 / +12 points).
- **RL.** §7.3: "This suggests that RLHF, while computationally expensive, may be largely elicitation". This is an inference from SFT
  results, not an RL experiment.
- **MDL alone does not work.** §7.5: "repeating our analyses with MDL instead of EDL failed to reliably distinguish elicitation from teaching".
- **Limitations** (§7.4): mainly arithmetic and reasoning with verifiable answers; Llama and Qwen dense models only. "As EDL depends on the training
  algorithm, different training techniques, optimizers, learning rates, or other hyperparameters could yield different
  values". App. H.1.2: these "primarily affect the low-data, low-parameter regime of single-epoch training".

Our interpretation:
- **Three cross-checks.** The paper gives us three, and we run only the first: (i) the EDL-per-token shape over n; (ii) capacity thresholds
  over adapter size; (iii) multi-shot prompting as independent validation. Items (ii) and (iii) are the review's R1 and R2.
- **Their residual is a converged test loss.** It is not our one-pass L_test(θ_64). The coordinator adopted a 3-epoch residual on 2026-11-25.

## Evidence and limitations

- **The numbers are hard to compare with ours.** The thresholds come from models trained to convergence on datasets far larger than ours. The reasoning task is SFT of base
  models on R1 traces of s1K prompts, with LoRA lr 1e-4, batch 1 and effective batch 8 (Table 4).
- **Not all values agree in the paper.** The PGR-based threshold (0.14) and the EDL-based one (0.03) for Llama 3.2 1B multiplication differ by 4.7×.
- **No per-example noise.** The paper reports no per-example noise for the bits/param values beyond 3 seeds. I did not check the App. I.9 few-shot numbers.

## Connections and questions

- Review: [REVIEW.md](../REVIEW.md), items 1–2 and the EDL-weakness section.
- Our spec: `code/experiments/current/olmo_elicitation_20261125/SPEC.md`, stage 1.
- **Open question.** Does the capacity threshold of LoRA on a 32B model trained on 5k-token thinking targets behave like the paper's thresholds at all?
  Only the relative reading across checkpoints is safe.
