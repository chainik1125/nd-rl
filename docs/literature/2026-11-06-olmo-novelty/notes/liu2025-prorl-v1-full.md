---
written_on: 2026-11-07
written_by: "agent:claude"
papers:
  - liuProRLProlongedReinforcement2025
---

# ProRL v1 in full, side by side with ours

Paper: [@liuProRLProlongedReinforcement2025]
Source: arXiv 2505.24864v1, only arXiv version; NeurIPS camera-ready: §4 unchanged [PR-75–PR-78]. Extends nd-rl `2026-10-02-lit-review-2/notes/liu2025-prorl.md`.

## Learnings

**Observed.**
- §4: 256 samples per prompt, 18 of 96 Reasoning Gym tasks plus other benchmarks; base, an "intermediate checkpoint" (step never given; separates Plateau from Sustained [PR-57]) and final [PR-26, PR-27].
- Fig. 3 left: one point per task, x = base pass@128, y = its RL gain [PR-31, PR-35]. Text: "significant negative correlation", no coefficient or test [PR-32, PR-76]; checklist: only pass@128 from 256 samples [PR-77]. Image legend only: "Linear regression (R² = 0.879)", colours Math, Code, Reasoning Gym, IFEval/GPQA. Gains ≥ 0.9 are Reasoning Gym tasks at base ≤ 0.03 (via Fig. 13: dice, advanced_geometry, power_function, n_queens; figure only) [PR-34].
- "Base fails entirely" = 0 at every k ≤ 256 on Reasoning Gym tasks: dice, the only such panel in Fig. 4 [PR-44, PR-45]; group_anagrams, n_queens [PR-53, PR-54]; unseen boxnet [PR-46–PR-49]; graph_color at 15–20 nodes [PR-50–PR-52] (read on rendered figures).
- §4.2: Diminish = math, deepcoder_livecodebench, letter_counting; Plateau = gpqa_diamond, maze, palindrome; Sustained = codecontests, taco, dice [PR-55–PR-62]. §4.4: family_relationships pass@1 shifts from 0 to 1 [PR-64].
- IF: 10k continuous-reward prompts from Run 3 of 8 [PR-11–PR-14]; evaluation averages a continuous score [PR-22]. IFEval: not in Fig. 4; a blue point in Fig. 3; an appendix pass@k panel to k 256 [PR-28, PR-29]: base 0.443 → 0.856, final 0.659 → 0.886 at k 1 → 128 (figure only). Pass@k on a continuous reward is never defined (UNVERIFIED).
- The base writes \boxed{} instead of the required <answer> tags [PR-66, PR-68]; one verifier trains and grades [PR-69]; resets follow a validation blend from the evaluation benchmarks [PR-19].
- Creativity index (DOLMA overlap): 3.84 base → 4.70 final; low-gain math and code tasks score lower [PR-39–PR-41].

**Paper interprets.** RL finds strategies "inaccessible" to the base, gains "predictable" from it [PR-36, PR-43]; formatting "is relatively easy to learn" [PR-67].

**Our interpretation.** Most Reasoning Gym points lie on y ≈ 1 − x, and the fitted line (≈ 0.9 at x 0, ≈ −0.08 at x 1) means a final pass@128 ≈ 0.9 whatever the base: R² 0.879 is mostly the ceiling our P1 null targets (figure only). Zero-base tasks are format-gated: "fails entirely" may be format learning.

## Setup card, side by side with ours

| | ProRL v1 | Ours |
|---|---|---|
| Start | R1-Distill-Qwen-1.5B, distilled [PR-1] | 32B Think SFT + DPO (C2–C8); 7B base (C1) |
| RL / pretraining | 2.6e-3–1.2e-2 FLOPs (rl-compute-share; [PR-2, PR-8]); v2 4.4e-3–2.1e-2; +BroRL 1.0e-2–4.5e-2 | 7B 1.5–3.2e-3; 32B ≈ 3e-3–2e-2 (executor correction: reviewed compute axis at step 2,300), 14 % GPU-h |
| RL steps | > 2k [PR-2]; v2 3k [PR-86] | 500 / 2,800; 100 / 750 / 2,300 |
| Tasks incl. IF | 136K, five domains; IF continuous [PR-9–PR-13] | maths, code, IF, chat (32B); maths (7B) |
| KL, resets | KL + hard resets [PR-3–PR-5] | OlmoRL: no KL, no resets [PR-137] |
| Largest k | 256; claims at 128 [PR-26, PR-31] | 256 / 512 / 1,024 |
| Grading | rule-based verifiers [PR-21, PR-22] | checker + final answer + hand |
| Controls | equal samples [PR-26], no reverse count (partly); format no [PR-66]; guessing no; contamination no [PR-19] | symmetric yes; format partly; guessing, contamination yes (C1, C7) |
| Base fails entirely | Reasoning Gym, k ≤ 256 [PR-44–PR-54] | C3 49 strict / 42–43 hand; C8 17 of 52 elicited; C1 none |
| Largest where weakest | stated; R² figure only [PR-32] | not computed (pre-registered P1) |

## Evidence and limitations

Read in full. Curve values: PDF data labels matched to rendered curves. One run, no seeds. Stated limitations: 1.5B only, resets, task scope [PR-72–PR-74].

## Connections and questions

- C1, C2: **replicates** no math reach [PR-55, PR-56]; but ProRL narrows (certified by [yang2026](yang2026-passk-crossovers-prorl.md)), ours lose nothing.
- C3, C5: **extends**: IFEval only, continuous, ≈ +0.03 at k 128 (figure only) [PR-28].
- C7: **extends**: step-resolved pass@k, no symmetric, hand or guessing controls [PR-27, PR-57].
- C8: **extends**: format blamed [PR-65, PR-66], not re-graded.
