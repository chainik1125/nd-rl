---
written_on: 2026-09-29
written_by: agent:claude
papers: []
---
# HyperTree Proof Search (HTPS / Evariste): AND-OR search, soft critic, minimal-proof training data
Paper: HyperTree Proof Search for Neural Theorem Proving, Lample, Lachaux, Lavril, Martinet, Hayat, Ebner, Rodriguez, Lacroix, 2022
Source: https://arxiv.org/abs/2205.11491v1
## Learnings
- Search (§4): PUCT-style selection over a proof hypergraph; selection builds a partial proof *hypertree*; leaf value = 1 solved, 0 invalid, else critic c(g); parent value = product of children values (§4.3). Many hyper-parameters (samples per expansion 8–48, temperature 0.8–2.0, depth penalty 0.8–1) are sampled per attempt (App. C).
- Critic (§5.1): same seq2seq model, output restricted to PROVABLE/UNPROVABLE tokens. Targets: 1 solved, 0 invalid, internal nodes "W(g,t*)/N(g,t*)" (soft). Table 5: Equations cumulative 78.1 (soft critic) vs 65.6 (no critic) vs 63.1 (hard critic); Metamath valid pass@8 68.6 / 64.8 / 67.6; test 57.4 / 52.2 / 57.4. "using hard critic targets gives worse performances than having no critic model at all" (§7.2.2).
- Policy training data (Table 4): Equations cumulative 78.1 learning from minimal proofs of all solved nodes vs 40.6 from all proofs of solved nodes vs 37.5 AlphaZero-style visit-count targets; Metamath valid 68.6 from root minimal proofs. "Learning only from the minimal proofs always leads to improved performance" (§7.2.1). Minimality = step count (Metamath, Equations), CPU time (Lean).
- Online vs EI (Fig. 7, §7.2.4): more frequent model refresh gives higher cumulative pass rate; "No training" is much lower "despite using as many attempts". Metamath: supervised plateaus at 66% vs Evariste >74% (§7.1.2).
- OOD adaptation (§7.1.3): Equations — trained on random-generator theorems, online training on the Identities split reaches 91.3% cumulative "while a supervised model never exceeds 36%".
## Evidence and limitations
- Models are 440M–600M, arXiv-pretrained (§6.2–6.3); 48 V100s for ablations. Single runs, no error bars.
- Fixed search params beat Evariste on Metamath valid (69.8 vs 68.6, Table 5), so the "stochastic params" gain is Equations-specific.
- No comparison of search-expert vs sampling-expert at matched compute; all arms use HTPS. Cumulative pass rates are transductive (the evaluated statements are trained on).
## Connections and questions
- B1/B2: minimal-proof selection is a stronger form of our proposed Polu shortest-proof rule — it also applies to *every solved subgoal*, which resembles our hindsight relabelling. Its 78.1 vs 40.6 is the largest single ablation in this cluster.
- B4: the Equations → Identities result is the closest published analogue of our generator→textbook gap, but it trains on the target statements.
- Critic recipe that transfers: value as a token of the same model, soft targets from search statistics. Hard 0/1 targets are the failure mode to avoid.
- Smallest test: add a PROVABLE token head to the state-env model, run a product-of-values best-first/PUCT search on the step env, train on minimal proofs of solved nodes.
