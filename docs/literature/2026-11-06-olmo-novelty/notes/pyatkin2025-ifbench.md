---
written_on: 2026-11-07
written_by: "agent:claude"
papers:
  - pyatkinGeneralizingVerifiableInstruction2025
---

# IFBench: out-of-domain verifiable constraints and IF-RLVR, but no large-k or per-prompt start-vs-RL test

Paper: [@pyatkinGeneralizingVerifiableInstruction2025] (proposed key; not in `references.bib`)
Source: https://arxiv.org/abs/2507.02833v3 (v3, 11 Nov 2025, the latest; NeurIPS 2025 Datasets and Benchmarks). Read in full. Ledger ids are in `ledger/IF.tsv`.

## Setup card

| item | value |
|---|---|
| model size and start | 7–8B. Post-trained starts: Tülu-3-8B-DPO (SFT + DPO), Qwen2.5-7B-Instruct, OLMo2-instruct. Pretrained bases: Llama-3.1-8B, Qwen2.5-7B, OLMo2 [IF-11] |
| RL share of pretraining | not computed (the paper is not in rl-compute-share's table) |
| RL steps | ≈ 2,000 GRPO steps, about one day on 8 H100s [IF-13]; 16 samples per prompt, 2,048-token responses [IF-12] |
| domain | precise IF: 58 out-of-domain constraints on 300 held-out WildChat prompts [IF-1][IF-2][IF-3] |
| largest k | 1 (evaluation is mostly greedy) [IF-15] |
| grading | Python verifiers, strict and loose; headline numbers are prompt-level loose [IF-5][IF-14]; GPT-4.1 judge for quality [IF-20] |
| controls | symmetric budgets: yes, trivially (both sides greedy) [IF-15]; format tolerance: partly (loose strips first/last lines and font modifiers) [IF-5]; guessing: n/a; contamination: yes (held-out prompts; train and test constraint sets kept apart) [IF-2][IF-8] |

## Learnings

**Observed.**
- IFBench has 58 new constraints with one or two per prompt. Every instance was human-checked so that the prompt and the constraint fit together [IF-1][IF-4][IF-6]. Some test constraints are "similar but different" to IFEval's [IF-10]. IFTrain adds 29 training constraints, kept separate from IFBench [IF-7][IF-8].
- IF-RLVR is GRPO on Tülu-3-SFT prompts with constraints appended [IF-9]. Table 6: Tülu-3-8B-DPO goes from IFEval 81.1 to 92.2 and IFBench 25.2 to 44.6 (loose, greedy) [IF-92]. The introduction instead gives 28.9 → 45.9 from the final Tülu-3-8B [IF-16]; both recorded, not adjudicated. RL from pretrained bases has no before-RL score ("no chat template") [IF-17].
- GRPO beats DPO on the same data [IF-24]. Disjoint variable ranges generalize worse [IF-26].
- Over-optimization:
  - IF-RLVR models "prioritize the constraint" [IF-19]. A keyword-stuffed output is shown as an example [IF-21].
  - A GPT-4.1 judge rates the base policy's completions 7/10 and the IF-RLVR completions 6.4/10, judged against the prompt without its constraint [IF-20].
  - Adding a reward-model signal gives IFEval 86.1 and IFBench 30 [IF-23].
- IF-RLVR answers average 210 tokens on IFBench [IF-22].
- No pass@k, coverage or per-prompt comparison appears anywhere. Evaluation is greedy [IF-15].

**Paper interprets.** Most models overfit to the few constraints that existing benchmarks test [IF-106]. More varied training constraints help generalization. Over-optimization should be offset with a preference reward model [IF-18].

**Our interpretation.**
- IFBench is C3's benchmark, but this paper never asks the boundary question. Greedy, loose accuracy can neither show nor rule out reach.
- Its construction matters for C3. Olmo 3's RL IF prompts take their constraints from IFEval and IFTrain [IF-28], and IFTrain is disjoint from IFBench by design [IF-8]. IFBench is therefore out-of-domain for Olmo's RL by construction. That fits C3's "20 of 42 with no RL analogue". The paper's "similar but different" overlap [IF-10] fits C3's 16 loose and 6 close analogues.
- Its observations are the first-party precedent for C4 and for the brevity part of C5 and C8: constraint priority, keyword stuffing and short answers [IF-19][IF-21][IF-22].
- The headline metric is loose [IF-14]. Under C3, loose grading halves IFBench reach (24 vs 49 prompts), so comparisons with the paper's numbers must use the same grade.

## Evidence and limitations

- Figure 1 and Figure 8 (judge score against accuracy) are figure only (UNVERIFIED).
- The prompt sets differ: the paper has 300 prompts, HiR 294 [IF-38], and C3 218.
- The quality comparison is a mean over completions [IF-20], with no per-prompt or hand reading.

## Connections and questions

- **C3: extends.** It supplies the out-of-domain construction and the lineage of Olmo's IF constraints [IF-8][IF-28]. It makes no boundary measurement.
- **C4: replicates** (qualitatively). The authors themselves report over-optimization [IF-18][IF-20][IF-21].
- **C5 and C8: replicate** (qualitatively). IF-RLVR answers are short [IF-22].
- **C7: extends.** It defines strict and loose [IF-5]. Because its evaluation is greedy only [IF-15], per-prompt large-k checks are new.
- **C1, C2, C6: unrelated.**
- **Q3:** the paper has no large-k test [IF-15]. The closest IFBench measurement is HiR ([zhang2025-hir.md](zhang2025-hir.md)).
