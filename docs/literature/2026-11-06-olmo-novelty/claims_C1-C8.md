# Our claims to position (C1–C8) — the reference sheet for olmo-novelty

Wording from Dan's brief (2026-11-06) and the reviewed nd-rl summaries (`experiment-summaries/2026-11-0*-olmo-*`).
Every number is of one model pair; the labels are those of the Olmo runs. Natural-language chat models: no proofs
are judged, so the Lean-only rule does not apply.

| label | checkpoint | what it is |
|---|---|---|
| 7B base | `allenai/Olmo-3-1025-7B` (Olmo 3 7B base, pretrained) | start of the RL-Zero run (C1) |
| 7B RL-Zero s500 / s2800 | `allenai/Olmo-3.1-7B-RL-Zero-Math` at RL steps 500 / 2,800 | RL from the pretrained base on maths (C1) |
| `dpo` | `allenai/Olmo-3-32B-Think-DPO` (32 B; base → Think SFT → DPO) | start of the 32B RL run (C2–C8) |
| `s0100` / `s0750` / `s2300` | `allenai/Olmo-3.1-32B-Think` at RL steps 100 / 750 / 2,300 | the 32B RL run (Dolci-Think-RL: maths, code, IF, chat) |

- **C1 (run `olmo-rlzero-pilot`, 7B RL-Zero-Math, AIME 2024–2026, 83 problems after removing 7 contaminated).**
  RL ≈ 1.5–3.2 × 10⁻³ of pretraining compute at the last snapshot. RL's prompt, sampler and length limit; graded by
  RL's own checker (boxed answer / "final answer is" / last formula). pass@1 base 0.09 → 0.30 (s500) → 0.42 (s2800);
  pass@8 0.36 → 0.63. Under RL's verifier the base is credited on all 17 RL-only AIME solves within 383 samples
  (18–383, typically ≈ 70); 10 of the base's 17 resolutions are sound by hand, 6 right through a flawed step, 1 a
  guess. Nothing is lost: reverse 0 at k ≤ 512. Contrasts with Yue et al.'s narrowing at large k. Verdict
  inconclusive (one pre-registered clause asked the base to match RL at k = 8). Per-token likelihood: every scored
  correct RL solution lies inside the base's range (weak evidence: whole-solution gap e^731).
- **C2 (run `olmo-think32b`, AIME 2025–26, 59 problems).** DPO start vs the RL run (≈ 3 × 10⁻³ – 2 × 10⁻² of
  pretraining FLOPs at step 2,300 by the reviewed compute axis, 2.8 × 10⁻³ low / 2.3 × 10⁻² high; the brief's 6 × 10⁻³ low
  end came from an earlier scaling; 14 % of GPU-hours). pass@1 0.716 → 0.746, pass@8 ≈ 0.86 both. One inconclusive reach candidate
  (AIME 2025 II P13): `dpo` credited 7 / 256 by RL's grader, states the answer once; 43 % of its tries hit the
  32,768-token cap. Three grades: RL's grader, final stated answer, blinded hand check. Nothing lost (one reverse
  problem solved within 56 tries).
- **C3 (run `olmo-ifomega`, IFBench, 218 non-excluded prompts).** pass@1 0.290 → 0.683 (strict). 49 prompts strict
  and 42–43 by hand that the DPO start never solves in 256 samples while `s2300` is confirmed on fresh draws; an
  RL-only share of ≈ 20 % (43 / 218 = 19.7 %; 49 / 218 = 22.5 %), against Yue's ≤ 1 % (0.0 % AIME24 at k 1,024; 1.0 %
  MATH500 at k 128); constraint types absent from RL's IF data (of the 42 hand-reach prompts' binding constraints, 20
  have no RL analogue, 16 a loose one, 6 a close one); almost no narrowing (2 reverse prompts). Under IFEval's
  official `loose` grade (format-tolerant) IFBench reach is 24 prompts, not 49; loose-only successes were not
  hand-read.
- **C4 (gaming).** About a third (36 %) of `s2300`'s first-checked strict successes on IFBench are gamed (letter of
  the instruction kept, substance dropped: vacuous passes, keyword stuffing); on 11 prompts (10 by the review's rule)
  RL's apparent advantage is entirely gaming. So checker-only pass@k overstates reach.
- **C5.** IFEval reach (6 of 150 prompts; letter-avoidance and all-caps constraints of types RL trained on; RL passes
  with 5–15× shorter answers) may be brevity; OMEGA `test_out` reach appears only at RL's 32,768-token cap
  (perp_068 / 069: `dpo` cut on 255 of 256).
- **C6 (failure modes).** The DPO model's failures on the 49 IFBench reach prompts are "substantive" in 93 % of a
  blind reading (70 % attempted but far off). The executor's number; the review did not re-derive it.
- **C7 (methods).** Symmetric confirmation (the same rule for reach and reverse candidates; fresh draws confirm the
  RL side), a blinded hand gate, a final-answer-only grade beside the RL grader, guessing at large k on integer
  answers (AIME answers are 0–999), a compute axis along a public checkpoint trajectory (Olmo 3.1 Think 32B publishes
  intermediate RL checkpoints), and the project's capability standard (nd-rl
  `docs/project_strategy/capabilities/REPORT.md`): "not reached at K", not "created".
- **C8 (run `olmo-iffu`).** Test 1 (token budget): with the full ≈ 65k context `dpo` finishes 191 / 192 geometry
  samples (median ≈ 46k tokens) and is hand-sound on perp_068 at least twice in 64 → the token budget explains 1 of
  the 3 geometry reach prompts. Test 2 (brevity): "Answer in at most 60 words." makes `dpo` succeed on 4 of 6 IFEval
  and 12 of 42 IFBench reach prompts (16 of 48; 15 under the strict title-case reading); the hint also breaks
  `s2300` on 6–7 IFBench prompts. Test 3 (more samples): 0 genuine successes in 1,024 `dpo` draws on 10 IFBench reach
  prompts (per-draw rate ≤ 2.9 × 10⁻³, 95 %), but 4 of those 10 are prompt-elicitable. Test 4: `s0750` is confirmed
  on 26 of 43 IFBench reach prompts (≤ 32 draws). Of 52 reach prompts, 17 are explained (1 token budget, 16 brevity);
  2 survive both 1,024 draws and the hint.
