---
written_on: 2026-11-07
written_by: "agent:claude"
papers:
  - yangWhenPassCrossovers2026
---

# When pass@k crossovers are real: paired inference, the truncation confound, and the locked minority

Paper: [@yangWhenPassCrossovers2026] (proposed key; Chen Yang, Texas A&M). Title of v1–v2: "RLVR is a Kernel, Not a
Function: Statistical Inference for pass@k Crossovers".
Source: https://arxiv.org/abs/2609.22547v3 (v3, 2026-11-03, the latest). Read in depth (§1–§8, limitations, Table 11;
proofs skimmed). Ids are in `ledger/IL.tsv`.

## Setup card

| item | value |
|---|---|
| models, start | nine public RLVR checkpoints: DeepScaleR-1.5B and ProRL from distilled R1-Distill-1.5B; four SimpleRL-Zoo (zero-RL from pretrained bases); ORZ, SimpleRL-1.5B and LUFFY [IL-196, IL-197]. AceReason-7B and Skywork-OR1-7B at 32k [IL-163]. Own GRPO and SFT arms from Qwen2.5-Math-1.5B [IL-170] |
| RL share of pretraining | not computed for the paper; rl-compute-share rows: SimpleRL-Zoo 4.6 × 10⁻⁵ – 1.1 × 10⁻³, ProRL v1 2.6 × 10⁻³ – 1.2 × 10⁻² (`compute_table.csv` lines 2–5, 18) |
| RL steps | own arms 100–132 updates, 17–30 × fewer than DeepScaleR or ProRL [IL-173] |
| domains | maths (code as a check) |
| largest k | 4,096 measured directly (controlled arms) [IL-170]; 4,352 draws on ProRL's locked prompts [IL-169]; public pairs ≤ 128–256 [IL-162] |
| grading | Math-Verify symbolic equivalence, identical for both sides [IL-176] |
| controls | symmetric budgets **yes**, with a truncation-sensitivity test [IL-164, IL-175]; format tolerance **partly** (a boxed-answer instruction; 0 verifier misses on audited locked prompts) [IL-168]; guessing **yes** (free-form answers avoid the guessing floor) [IL-174]; contamination **yes** (deduplicated sets) [IL-177] |

## Learnings

**Observed.**
- Visual crossings are rarely statistically real. Of nine public checkpoints at 3k–16k, only one narrowly meets the
  paired criterion [IL-153]; for Yue et al.'s SimpleRL-Zoo exhibits that one is Qwen2.5-7B [IL-159, IL-197].
- Asymmetric truncation is a major confound. At 8k the base truncates 30.6 % against ProRL's 2.0 % [IL-156]; about
  0.7 log-odds of ProRL's 8k advantage is the base running out of tokens [IL-157]. Token limits alone can create, mask
  or reverse a crossover [IL-154]. One zero-RL model stays ahead only because its base fails to terminate on 78 % of
  generations [IL-160].
- Clean test: 32k limit, 1,060 deduplicated problems. ProRL-1.5B crosses, with first loss certified in k 12–63
  [IL-162]. AceReason-7B stays ahead at every k ≤ 128 [IL-163].
- Power: at 1,000 × 256, power is only .21 for ProRL-16k [IL-166]. More prompts help more than deeper sampling [IL-165].
- Mechanism: the late-k loss is a 2–3 % minority of base-solved prompts locked onto a repeated wrong answer [IL-167].
  These are not truncation or format failures [IL-168]. Thirteen of ProRL's 22 locked prompts stay at 0 in 4,352 draws
  [IL-169].
- Controlled runs: all four GRPO arms fall behind the base from k 178–234 to k 4,096; SFT stays ahead [IL-170, IL-171].

**Paper interprets.** Crossovers are "a property of specific checkpoints and training regimes under a stated
evaluation protocol, not a universal law" [IL-161]. Claims at k ≫ n need sampling at k [IL-172]. Report truncation
rates and test token limits [IL-175].

**Our interpretation.** This is the strongest methodological critique of both sides of the support debate. Pass@k
gaps at standard limits can be truncation artifacts. It also points to a check for our "nothing lost" claims (C1:
reverse 0 among 83 problems; C3: 2 reverse among 218). *Our arithmetic:* a 2–3 % locked rate would mean 1.7–2.5 reverse
problems among 83, so 0 has little power against it, though the rate was measured on 1.5B distilled models.

## Evidence and limitations

Maths only; short controlled runs on one 1.5B base [IL-173]. No instruction following (text search).

## Connections and questions

- **C1, extends:** some public pairs show no supported loss [IL-161, IL-163]. Our zero reverse is consistent with that
  but low-powered (arithmetic above).
- **C2, replicates** (the confound): asymmetric truncation [IL-154, IL-156] is the issue behind `dpo`'s 43 % capped
  tries.
- **C5, replicates:** the length limit decides the comparison [IL-154, IL-160].
- **C7, extends:** it adds paired simultaneous bands, power design and truncation reporting [IL-152, IL-165, IL-175],
  and it supports sampling at the k claimed and the guessing caveat [IL-172, IL-174].
- **C8, replicates:** raising the token limit changes verdicts (Test 1) [IL-157, IL-175].
- See [yang2026-what-outcome-rl-buys.md](yang2026-what-outcome-rl-buys.md).
