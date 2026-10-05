---
written_on: 2026-10-02
written_by: agent:claude
papers:
  - wang2025rlplanning
---

# RL for planning on a graph abstraction: exploration beats SFT, policy gradient collapses diversity

Paper: [@wang2025rlplanning]
Source: https://arxiv.org/abs/2509.22613v2 (v2 reviewed; html text)

## Learnings

- Setting (Sec. 2.1), inherited from ALPINE (Wang et al. 2024b, arXiv 2405.09220): planning = path-finding on a fixed directed graph; each node is a token. Reachable (s,t) pairs are split into train (~20%) and test; the SFT corpus is K=10 random-walk paths per train pair, so the exact set of edges, co-occurrences and reachability facts seen in training is known. Model: "a one-layer, single-head Transformer", d=120; Erdős–Rényi graph with 100 nodes, p=0.15; also a Blocksworld state graph.
- SFT stable points memorise co-occurrence (Takeaway 1); SFT "will fail to exploit transitivity information (which never appears in D^SFT)" (Sec. 3.2). Even edges present in data but rare are under-weighted (Fig. 1).
- Policy gradient with 0/1 outcome reward and no KL is exactly SFT on the correct on-policy samples (Thm 4.1), but because samples are on-policy it "can explore and discover new correct paths that were absent from the initial training set" (Sec. 4.1). Empirically test accuracy rises under PG while "the test accuracy of Continual SFT constantly decreases" (Sec. 4.2).
- Diversity collapse: without KL, PG reaches 100% train accuracy and keeps narrowing; "the model eventually produces only one path per pair", and test accuracy then degrades (Sec. 4.2, Thm 4.3). KL preserves diversity but caps train accuracy and anchors to the base model (Thm 4.4).
- Q-learning with outcome-only reward suffers Q-value bias; process rewards fix it, and Q-learning then keeps diversity and works off-policy (Takeaways 5–6).

## Evidence and limitations

- Proofs are for a stylised one-layer model; experiments are tiny (100-node graphs), so cost is negligible.
- "New paths" here are new *compositions of edges already seen*; RL prompts come from train pairs only and test pairs are never seen in SFT or RL. RL improves test pairs by exploring paths through observed adjacency, so this is creation of reachability facts, not of new primitives. Whether RL can add an edge never seen in SFT is not what is tested (and an edge never seen has no way to be rewarded except by luck).
- The graph is fixed across train and test; there is no renaming split, so "generalisation" is within one memorised graph.

## Connections and questions

- Our expert-iteration ladder is the STaR/RFT special case of their Thm 4.1 (PG with 0/1 reward and no KL = SFT on correct samples). Their diversity-collapse result predicts that our ladder narrows to one proof per theorem; our per-step log p at every checkpoint can measure entropy over known alternative proofs directly.
- The graph domain gives a clean creation test that complements ours: hold out all paths that use a given edge pair (a "transitive" fact) from SFT, then ask whether RL makes that composition. This is the graph analogue of our renaming-class split.
- Combined with Abdulsalam et al. 2607.07646 (RFT plateaus, GRPO keeps climbing), it predicts our ladder (RFT-like) is the weaker RL arm; a GRPO arm with negative rewards is the obvious comparison.
