---
written_on: 2026-10-02
written_by: agent:claude
papers:
  - tsilivis2025rlafterntp
---

# RL after next-token prediction on parity: amplification of a rare, already-present long demonstration

Paper: [@tsilivis2025rlafterntp]
Source: https://arxiv.org/abs/2510.11495v2 (v2 reviewed; html text)

## Learnings

- Setting: transformers (GPT-2 / Mistral variants; depth L in {2,4,8}, width 128 or 256; App. C.1) trained from scratch, online, on a mixture D(p_cot) of short sequences (bits -> parity) and long chain-of-thought sequences (bits, running prefix parities, parity). p_cot is the exact, controllable fraction of long demonstrations (Sec. 2).
- Next-token prediction alone with small p_cot (e.g. 0.25, d=50) never generalises under greedy decoding, "even with access to ~10^7 training samples" (Sec. 3.1). The model is "calibrated with respect to length": it emits long answers with probability p_cot and is correct on those, random on short ones (Sec. 3.2).
- Switching to RL (GRPO, REINFORCE, STaR) takes greedy accuracy from chance to 100% after a handful of sequences while response length grows to d (Sec. 3.1). For STaR with the CoT reward the mechanism is closed-form: the effective long-fraction follows p_n = 2p_{n-1}/(1+p_{n-1}), which "converges to 1 exponentially fast" (Sec. 3.2).
- Theory (Sec. 4, linear autoregressive models): learning is efficient as long as the long-demonstration fraction is "not exponentially small in the input dimension d" (abstract; p_cot in Omega(d^-kappa), appendix).
- Timing matters: "if post-training starts too early, then it might not lead to a generalizing model despite the length increase" (Sec. 3.1).
- Failure case: STaR with high sampling temperature and end-to-end reward (no negative reward, sparse signal) is the noted exception (Sec. 3.1).

## Evidence and limitations

- Strong, multi-seed (3 seeds) small-scale evidence plus proofs in a simplified model; extra Llama experiments on mixture variants of GSM8K/MATH (Sec. 5), not reviewed here.
- This is by construction an *elicitation* result: the successful behaviour (the full long CoT) is present in pretraining at rate p_cot, and the model already produces it at that rate. RL changes the mixture weight, not the skill. Appendix runs go down to p_cot = 0.01 with length growth and gains reported; I did not find a p_cot = 0 control, which is the case that would test creation.
- The "hard" behaviour is learnability of the short map, not the existence of the long one; it says nothing about RL producing traces outside the pretraining support.

## Connections and questions

- Closest analogue to our ND organism's coverage dial: their p_cot is our share of long (cap-12 vs cap-6) proofs. Their STaR recursion is a sharp quantitative prediction for our expert-iteration ladder: if a target shape has base-model probability q and the verifier accepts it, its share should grow roughly like q -> 2q/(1+q)-type maps per round. Our per-step teacher-forced log p at every checkpoint lets us test whether the ladder tracks such a recursion (elicitation) or departs from it (e.g. probability rising on proofs with base mass near zero).
- The p_cot sweep down to 0 is the experiment they did not report; our organism can run it exactly (remove all proofs of length > 6 / of a renaming class) and measure whether RL still reaches them.
- Their timing result ("too early" pretraining checkpoints fail) suggests our ladder's starting checkpoint is a confound worth one ablation.
