---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - jain2023mechanistically
---

# Fine-tuning learns "wrappers" over existing capabilities; revival speed vs a never-had-it control

Paper: [@jain2023mechanistically] (Jain, Kirk, Lubana, Dick, Tanaka, Grefenstette, Rocktäschel, Krueger)
Source: arXiv 2311.12786v2 (HTML rendering read: Abstract, Sec. 1-6, App. D, E.1-E.4, F.1-F.3; App. B, C, G, H skimmed
by headings only). Also screened at section level by L2 (`_screen_L2.md`); this is the in-depth note.

## Learnings

- **Question.** "does fine-tuning yield entirely novel capabilities or does it just modulate existing ones?"
  (Abstract). Answer in their synthetic settings: "(i) fine-tuning rarely alters the underlying model
  capabilities"; a 'wrapper' "is typically learned on top of the underlying model capabilities, creating the
  illusion that they have been modified"; and after further fine-tuning "the model begins reusing these
  capabilities after only a few gradient steps" (Abstract).
- **Definition of a capability (representational, domain-restricted).** Inputs split into an identifier x_i
  (which capability to use) and data x_d. Read_l = a linear layer trained on layer-l outputs using the
  pretraining data. M "possesses a capability C" if for all x in the sub-domain X_C there is a layer l ≤ L with
  Read_l(M(x)) = f_C(x_d) (Sec. 3, Definition 1). "A linear readout at an intermediate layer is used in the
  definition above to emphasize that the notion of a capability need not correspond to only input-output
  behavior" (Sec. 3); "Such structured failures imply claiming the existence of a capability should account for
  the input domain" (Sec. 3).
- **Relevance.** If fine-tuning on a small D_FT can turn C into g∘f_C that is correct on the whole fine-tuning
  distribution, C "is strongly relevant to the fine-tuning task; else, we call it weakly relevant" (Sec. 3,
  Definition 2). With a weakly relevant capability (exploitable through a planted spurious correlation), g is a
  "wrapper": "a wrapper, i.e., a localized transformation of the pretraining capability, is learned during
  fine-tuning" (Sec. 5.1).
- **Three tests that the old capability survives under the new behaviour** (Sec. 5, Fig. 4):
  1. *Linear probe* on every block's residual output at the answer token ("Probing is used to understand if a
     particular capability is present in the model", App. D). The pretraining count is still decodable after
     fine-tuning except with the larger learning rate and a weakly relevant capability (Fig. 8). In the jailbreak
     variant, even with the non-jailbreak token "the model does encode the count of a in the outputs around the
     middle layers" (App. E.2).
  2. *Pruning*: single-step removal of the neurons with the largest gradient × weight for the pretraining-task
     loss (App. D); "the pretraining task's performance improves after just 5–15 neurons are pruned" while
     fine-tuning-task performance drops (Fig. 7 caption).
  3. *Reverse fine-tuning (reFT)*: fine-tune back on pretraining data with an even smaller learning rate. "if the
     behavior corresponding to a pretraining capability is retrieved in a few steps of reFT, fine-tuning did not
     meaningfully alter said capability" (Sec. 5); with the smaller lr, revival "is stronger evidence that the
     pretraining capability was never forgotten or removed" (App. D).
- **The control that makes reFT a test.** Revival speed is compared with a model that never had the capability:
  Scr.+FT (pretrained on a different operand, then fine-tuned) "only reaches perfect accuracy at 4.5K iterations
  and when using a larger learning rate", against 0.1–1K (strong relevance) or 3K (weak) for reFT (Fig. 9 caption).
  TinyStories (91 M-parameter models, App. F.1): after fine-tuning to stop generating twists, twist proportion
  during reFT at iterations 0/30/300/3000 is 44/81/81/82 % (filtering, η_M) vs 12/31/44/81 % for the control
  pretrained without twists (Table 1); fine-tuned models' loss "goes down very quickly (30–300 iterations) compared
  to baselines (which never reach the same loss" (Fig. 11 caption), and reFT models "converge to a lower loss on
  this dataset than the control pre-trained model" (App. F.3).
- **Where fine-tuning does create.** With a low pretraining prior, "the performance is not high to begin with,
  indicating the ability to count was learned during fine-tuning" (Sec. 5.1); the larger learning rate with zero
  spurious correlation also changes the attention pattern (Fig. 10).

## Evidence and limitations

- Evidence: Tracr-compiled counters (capabilities exact by construction), minGPT on PCFG tasks (Figs. 5-9),
  TinyStories-Instruct (Fig. 11, Table 1, App. F). Fine-tuning is restricted to continued training at lr "one to
  three orders of magnitude lower than the average pretraining one" (Sec. 2); no RL, no reward-driven
  fine-tuning.
- The wrapper picture is shown for *weakly relevant* capabilities with a planted spurious feature; it is not a
  claim that fine-tuning never builds anything (Sec. 5.1 quote above).
- No statistics: the decision 'few steps' vs 'many steps' is read off curves; no threshold or CI. Probing even
  loses "a small amount of information" (App. F.3), so 'persists' is graded.
- The capability definition needs a known target function f_C and a linear readout trained on pretraining data;
  for open-ended generation (stories) they fall back on probing for story features and a GPT-3.5 classifier
  (92 % held-out accuracy, App. F.2).

## Connections and questions

- **Definition offered:** capability C on sub-domain X_C = existence of an intermediate layer from which a linear
  readout computes the target function f_C on every input of X_C (Def. 1). Behaviour can be absent while the
  capability is present (wrapped). Operational proxies: probe accuracy per block, pruning-recoverability, reFT
  sample efficiency relative to a never-had-it control.
- **New vs better access:** yes, explicitly. Old capability **modulated** (wrapper) if (a) a linear probe still
  reads f_C, (b) removing a handful of high gradient×weight neurons restores the old behaviour, and (c) a few
  low-lr gradient steps revive it much faster than a control model that never learned it. **New** if the target
  was not readable before fine-tuning and its acquisition is no faster than from the control (their low-prior
  case). The rule is comparative (vs control), not absolute.
- **Null / floor:** the Scr.+FT / "Not in PT" control model is the floor: the same fine-tune applied to a model
  that never had the capability. Speed is judged against it, so the 'with enough steps anything is learnable'
  objection is handled by the ratio, not by a fixed budget. Random-init is not used.
- **Transfer to our setting:** our direction is reversed (did RL create something that pend lacks?), but all three
  tools apply. (1) *reFT analogue = J4*: fine-tune pend on 16 excluded-middle demonstrations and count the steps or
  examples to reach the r16 behaviour (pass@256 on held-out instances), against the same fine-tune from init and
  from early pretraining checkpoints (our 'never had it' controls); elicited if pend's sample cost is a small
  fraction of the controls'. (2) *Pruning in reverse*: in r16, prune the top-K neurons by gradient×weight of pend's
  loss on pend's own outputs; if a handful of neurons revert r16 to pend-like behaviour on the excluded-middle family
  (and nowhere else), RL added a localized wrapper; if hundreds are needed, the change is distributed. (3) *Probe*:
  is 'goal needs reductio' linearly readable in pend (see `hewitt2019control.md` for the null)? Cost: each
  fine-tune minutes on one GPU; pruning sweeps one backward pass per K. Main failure modes: the best control (a
  pend pretrained without any classical proofs) does not exist and costs a full pretraining run; RL's change is
  multi-step generation, not one readout, so 'f_C' must be chosen per proof step; reFT sample efficiency depends
  on lr and data, which must be identical across arms.
- Related: `prakash2024finetuning.md` (same conclusion via circuits), `greenblatt2024passwordlocked.md` (unlocking
  hidden capabilities with few demonstrations), `deeb2024unlearning.md` (recovery-rate test of information in
  weights), `voita2020mdlprobing.md`, `mukherjee2025subnetworks.md` (how small RL updates are).
