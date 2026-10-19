# Run brief: rl-compute-share — is the RL in create-vs-elicit papers a realistic share of pretraining compute?

Run id `rl-compute-share`. Repository `~/work/rl-compute-share` (branch `dan_rl-compute-share` from the fork's
`origin/dan_create-or-elicit`, made by `newrun`). Role: executor. `AGENT_POLICY.md` governs. Authorised by Dan
2026-10-19 in chat (relayed by agent:claude).

- **Budget:** **no pods.** A $0.50 / 1 pod-hour budget is registered as a guard only. Claude time only.
- **Thinking:** every role runs at `--effort max`.
- **Time:** hard stop 48 h; no hurry.
- **Host:** shared with the running `kalmar-pretrain` executor. Do not touch its worktree, run directory or pods. At most
  4 concurrent processes on the VPS.

## The question (Dan, 2026-10-19)

Yue et al. [@yueDoesReinforcementLearning2025] ask whether RLVR creates new reasoning capabilities or only elicits ones
the base model already has, and conclude elicitation. When Dan looked at the paper earlier, the RL in it seemed tiny
next to the compute spent pretraining the base models. His hypothesis: papers like this answer "does a very small
amount of RL create new capabilities?", not "does RL create new capabilities?". If so, they cannot rule out creation at
the RL scale frontier labs are believed to use.

Measure how much RL these papers actually do relative to pretraining, compare it with what is known about frontier
labs, and test the hypothesis. Look as hard for evidence against it as for evidence for it.

## Questions

1. **Yue et al. in numbers.** For every RL-trained model the paper evaluates (math, code and visual reasoning; its own
   runs and the third-party checkpoints): base model, base pretraining tokens and FLOPs, and RL compute. Recover the RL
   configuration from the paper, the checkpoints' own papers, model cards and released training configs (steps, prompts
   per step, group size, maximum and mean response length, epochs). Report RL / pretraining as a ratio of FLOPs, and
   also of tokens processed and of GPU-hours where these are reported.
2. **The wider literature.** Build the same table for 20–30 papers that run RL on a pretrained model, starting with
   those cited for or against "RL only elicits":
   - ProRL and its successors (BroRL);
   - the threads in lit-review-2's Q1: "RL squeezes, SFT expands", Invisible Leash, negative-sample reinforcement, RL
     composing skills, and the pretraining / mid-training / RL interplay studies;
   - ScaleRL (Khatri et al., 2510.13786);
   - DAPO, Dr. GRPO, Open-Reasoner-Zero and DeepSeek-R1-Zero;
   - other open reasoning-RL recipes that publish their compute (for example Skywork-OR1, Polaris, MiniMax-M1, Magistral
     and Qwen3).

   For each paper, also record its verdict on create vs elicit and how it measured it.
3. **Frontier reference points.** What is publicly known, and what is credibly estimated, about RL compute relative to
   pretraining compute at frontier labs? How has that changed from 2024 to 2026? These are starting points to verify,
   not facts:
   - DeepSeek's cost disclosures for R1 against V3;
   - OpenAI's statements on RL compute from o1 to o3 and later;
   - xAI's statements on Grok 3 → Grok 4;
   - Epoch AI and other independent estimates.

   Grade every figure by its source: a primary lab statement; a peer-reviewed paper or technical report; an independent
   estimate with a stated method; or a press report or rumour. Give the frontier ratio as a graded range, not one
   number.
4. **Does the verdict depend on RL scale?** Collect the evidence on how create-vs-elicit results change as RL compute
   grows within a study: the prolonged-RL claims (ProRL, BroRL), ScaleRL's asymptotes, and any paper that varies the
   number of RL steps and measures pass@k at large k. Find the strongest evidence against the hypothesis too:
   - large-RL studies that still find only elicitation;
   - arguments that a FLOP ratio is the wrong normaliser. For example, RL gets little information per sample while
     pretraining gets a per-token signal, and RL compute is mostly inference.

   Say which normaliser (FLOPs, trained tokens, gradient steps, or another) you think is most defensible, and why.
5. **Where do our runs sit?** Compute the same ratio for our own create-or-elicit, rl-continue, rl-from-ckpt and
   trajectory runs, using the nd-rl `experiment-summaries/` and the fork's `numbers.md`. We pretrain those bases
   ourselves, so both sides of the ratio are known exactly. Is our RL share closer to the frontier range or to the
   papers'?

## Method

- **Pretraining FLOPs:** 6·N·D, with N the active parameters (for MoE) and D taken from the model card or report.
  Include continued pretraining, such as Qwen2.5-Math's.
- **RL FLOPs:** add up three parts.
  - Rollout generation: about 2·N per prompt and generated token.
  - The policy update: about 6·N per trained token.
  - Any reference-model or old-policy forward passes: about 2·N per token each.

  Rollout tokens = steps × prompts per step × group size × mean response length, plus the prompt tokens. State every
  assumption. When the mean length is not reported, give low and high bounds (for example ¼ and 1 × the maximum length)
  and carry both through to the ratio.
- **Contaminated bases:** note where a "base" model already saw instruction, reasoning or synthetic chain-of-thought
  data in mid-training (Qwen2.5 is the usual case). This weakens the split between base and RL.
- **Separate columns:** keep observed figures, your estimates and your interpretations apart.

## Rules

- Read first:
  - nd-rl `docs/literature/2026-09-29-lit-review/REVIEW.md` and `docs/literature/2026-10-02-lit-review-2/REVIEW.md`.
    Yue and ProRL are already screened there, so **extend those reviews; don't redo them**.
  - `docs/literature/2026-10-04-capabilities/REVIEW.md`, and its `screened.md`, which lists ScaleRL.
  - `docs/literature/README.md` (conventions; never edit `references.bib`).
  - The create-or-elicit summary in `experiment-summaries/`.
- **Verify every cited figure against the source text.** Fetch each source into a file and grep it, recording the
  location, as in `lit-review`. Mark anything unverified. Grade figures from blogs, posts or press reports by their
  source type.
- **Prefer 2025–2026 sources.**
- **Screen 30–45 sources and read 10–15 in depth.** Read Yue et al. in full.

## Deliverables

- `rl_compute_share/REVIEW.md` (≤ 3,000 words), containing:
  - answers to Q1–Q5;
  - a verdict on the hypothesis (supported, partly supported or not supported), with the reasons;
  - an "implications for our project" section ranked by value;
  - the screened-source table.
- `rl_compute_share/compute_table.csv`, one row per model or run. Columns: paper, arXiv id, base, N (active), D,
  pretraining FLOPs, RL configuration fields, RL FLOPs (low and high), ratio (low and high), GPU-hours if reported,
  create/elicit verdict, source locations, assumption notes, and a verified flag.
- `rl_compute_share/figures/ratio.{pdf,png}`: the RL / pretraining FLOP ratio per paper on a log axis, with the frontier
  range as a graded band and our runs marked.
- `rl_compute_share/notes/*.md` (TEMPLATE format) and `rl_compute_share/to_add_to_zotero.md`.
- The librarian copies them to nd-rl `docs/literature/2026-10-19-rl-compute-share/` and links them from
  `docs/literature/README.md`.
- `run_rl_compute_share.md` (≤ 200 words), `log.md`, a `STATUS.md` line `RL-COMPUTE-SHARE DONE <UTC>`, and
  `touch ~/runs/rl-compute-share/executor.done`.

Do not contact anyone. Do not edit `references.bib` or run `code/tools/literature.py`.
