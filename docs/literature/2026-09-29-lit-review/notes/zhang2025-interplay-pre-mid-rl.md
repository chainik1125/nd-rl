---
written_on: 2026-09-29
written_by: agent:claude
papers: []
---
# When RL adds capability: edge of competence, 1% seeds, and process-verified rewards (100M model)
Paper: On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models (2025)
Source: https://arxiv.org/abs/2512.07783v1 (v1 reviewed)
## Learnings
- §2.4: a 100M Qwen2.5-style model is pretrained on 10B tokens of synthetic GSM-Infinite dependency-graph problems with op = 2–10. RL is GRPO.
- §2.3: evaluation is "process-verified". Each step of the parsed dependency graph must match, and pass@k uses this strict criterion.
- §3, Fig. 3, Observation 1: on ID tasks RL gives pass@1 gains but "no improvement on pass@128". RL on the edge range (op = 11–14, where base pass@128 is nonzero) improves pass@128 on both op 11–14 and op 15–20. Fig. 1 caption: "up to +42% pass@128 when well-calibrated". Practical Guidance 1: filter to tasks that fail at pass@1 but succeed at pass@k, and re-evaluate this pool periodically.
- §4, Observation 2: with 0% or 0.1% pre-training exposure to a new context, RL does not transfer to it. With ≥1% exposure it transfers, "even to the hardest tasks of op=20" (Fig. 1: up to +60% pass@128). Fig. 5: for op 11–20, correct outputs in the new context are topologically less similar to the old-context graphs, i.e. there are "more novel structures".
- §5, Observation 3: under fixed compute, full mid-training plus light RL is best on the edge range. More RL helps on OOD-hard. Fig. 1: mid-training + RL beats RL alone by "+10.8% on OOD-hard".
- App. Observation 7: RL gains shrink as pre-training covers more of the hard range. With 20% exposure to op = 7–10, RL gives "more than +22 points" pass@128 on op 15–20.
- §6, Observation 4: mixing a dense process-verification reward into the outcome reward improves pass@1 "by 4–5%" on op 15–20. The best mix is 0.2·R_out + 0.8·R_pv. The strict "R_out only if R_pv = 1" also gives "substantial improvements".
## Evidence and limitations
- The results are figure-level. I did not see seed counts or error bars in the text I read. The model is 30× our size and 100× the tokens.
- Their process reward uses a ground-truth graph, so partial credit is available. Lean gives us no partial-credit signal except the position of the first error.
## Connections and questions
- B1/B2: this is the closest from-scratch analogue of our ladder. It confirms edge-of-competence rung selection (already done). It adds a measurable rule: pass@1 = 0 but pass@k > 0 under the current model.
- B4 (textbook): their 0 / 0.1 / 1% exposure result suggests a dose-response study. Inject 0, 0.1, 1 and 10% textbook-style theorems (or their primitive patterns) into Stage-1, then run the same EI. Our composition studies varied generator style, not the dose of the OOD context.
- B2 measurement: their topological-similarity test maps onto ours. For proofs of newly reached theorems, measure the similarity (e.g. rule-sequence edit distance) to base-reachable proofs.
- Process reward: for us the strict variant corresponds to "outcome only if every step Lean-valid", which is already our reward. The dense variant corresponds to first-error position (see process-verified RL note).
