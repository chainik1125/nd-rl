---
written_on: 2026-10-02
written_by: agent:claude
papers:
  - abdulsalam2026rlcompositional
---

# From-scratch rewrite-grammar organism: RL composes primitives into shortcuts the base model cannot reach at pass@1024

Paper: [@abdulsalam2026rlcompositional]
Source: https://arxiv.org/abs/2607.07646v1 (v1 reviewed; html text)

## Learnings

- The closest published design to our ND organism found in this screen. A transformer (12 layers, hidden 512, 8 heads; App. A; ~tens of M params, my estimate) is pretrained from scratch on chained single-step primitive rewrites of a known grammar (expansion / contraction). Pretraining exposes "local grammar-consistent rewrite dynamics, but not ... the later non-primitive shortcuts" (Sec. 3.1). A contraction weight rho controls the pretraining distribution (Sec. 3.1, App. C).
- Post-training: contract a word back to its target symbol under a 256-token budget; reward is binary and outcome-only. Difficulty k = k-1 primitive steps beyond the budget, so harder buckets require shortcuts. GRPO vs rejection fine-tuning (RFT) on the same prompts (Sec. 3.3).
- Frontier: base pass@1024 is 0% on buckets 4–5 (Table 1), "while RL later solves them at pass@16" (Sec. 6). RFT improves faster early, then "RFT then plateaus", while GRPO keeps climbing (Sec. 6, Fig. 5; 3 seeds).
- Mechanism (auditable because every rewrite can be checked against the grammar): macro (sequential) contractions begin rising around iteration 5,000 and overtake primitives "around iteration 12,500"; parallel contractions emerge later (Sec. 4, Fig. 2). The composed rules are reused and consolidated.
- RL vs RFT difference is "not exploration volume but selectivity": RFT produces many invalid shortcut-like rewrites (abstract, Sec. 5).
- Pretraining ablation: "Low-rho pretraining never reliably enters the macro or parallel regime"; a matched-contraction-fraction control without chaining behaves like rho=1 (Sec. 7). Exposure frequency alone does not gate emergence; the organisation of primitives into chains does.

## Evidence and limitations

- Strong on observability; weak on scale (one grammar, 80 held-out over-budget problems, 16 per bucket; 3 seeds; 4xH100 node).
- The "absent from pretraining" claim is about the shortcut *actions* (non-primitive rewrites); the reward checks only the final symbol and format, so some shortcuts could be reward-exploiting; the authors classify valid vs spurious by audit.
- No per-sequence likelihood of the composed procedures under the base model is reported (no log-prob analysis found by grep), so "rarely solved" is measured by pass@k only.

## Connections and questions

- Directly comparable to our ND ladder: from scratch, exact pretraining control, verifier reward, held-out frontier. Differences: their new behaviour is a new *action type* (macro-rewrite) the grammar permits but pretraining never showed; ours is new *proof shapes* built from ND rules that were all shown. Their budget-forcing trick (solutions only fit if compressed) is a design we lack: an ND analogue is a line cap below the shortest primitive proof, rewarding derived-rule use.
- Their RFT-plateaus result is a direct warning for our expert-iteration ladder (RFT-like); a GRPO arm would test it.
- Our per-step log p at every checkpoint is exactly the missing measurement in their paper: it would turn "base pass@1024 = 0" into a base-model probability for each composed trace.
