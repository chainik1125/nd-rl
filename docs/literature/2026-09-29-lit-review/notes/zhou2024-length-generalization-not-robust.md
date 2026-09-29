---
written_on: 2026-09-29
written_by: agent:claude
papers: []
---
# Length generalisation is achievable but seed-fragile (Zhou et al. 2024)
Paper: Transformers Can Achieve Length Generalization But Not Robustly (Yongchao Zhou, Uri Alon, Xinyun Chen, Xuezhi Wang, Rishabh Agarwal, et al., 2024)
Source: https://arxiv.org/abs/2402.09371v1 (version reviewed)
## Learnings
- Recipe (§3): FIRE relative PE + randomized PE + reversed format + index hints gives "generalize to 100-digit decimal addition tasks with more than 98% of accuracy when trained up to 40-digit addition … 2.5×" (Fig 1).
- Index hints "are crucial": "Without index hints, all PE methods fail in generalization" (Fig 4, §4.2). Hints are sampled "as a random slice" of a 102-symbol ordered set (§4.1).
- RoPE: "Despite being simple and effective, RoPE exhibits limited length generalization" (§2.1); RoPE/KerpleLog "exhibit moderate in-distribution generalization but falter in out-of-distribution scenarios" (§4.2). Random space augmentation helps RoPE, hurts NoPE/FIRE (Fig 7).
- Seed fragility: 10 FIRE trials with identical data order "exhibit perfect in-distribution generalization" but OOD "shows significant variance" (§4.2, Fig 10–11); 15 runs = 3 init × 5 data-order seeds show variance across data orders "even when the weight initialization is constant"; some inits are robust ("lucky weight ticket"). OOD accuracy "fluctuates significantly across training steps" and in-distribution loss is not a reliable predictor (§4.2).
- Remedies tried (§5): weight decay 0.1–0.3 "slightly enhance the likelihood"; dropout 0.2 "severely impair[s]"; "while regularization can modestly decrease performance variability, it falls short". Larger models don't reduce variance (Fig 14 caption). Longer training length raises the extrapolation factor: 10→1.0×, 20→1.25×, 30→1.5×, 40→2.5× (§5).
- Headline results reported as best of 10 trials (Fig 3/4 captions).
## Evidence and limitations
Single task (decimal addition), 25M model, 5–10 seeds per cell. Appendix figures not inspected. Figure numbering in the HTML text is inconsistent (two "Figure 9"/"Figure 16" references).
## Connections and questions
B3 directly: an independent report that OOD length generalisation varies massively across init and data-order seeds at identical in-distribution loss — consistent with our "~99% of variance is the Stage-1 run". Implications: (1) separate init seed from data-order seed in our Stage-1 to see which dominates (cheap: ~6.5 pod-min/seed); (2) OOD fluctuates across checkpoints — our checkpoint averaging addresses this; selecting checkpoints by a held-out longer-length probe is a cheap addition; (3) report best-of-n vs mean, as they did. Index hints: our `n<k>` names with random offset are already the analogue. Training length matters more than anything else in their §5 — matches our cap-8 finding.
