---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - prakash2024finetuning
---

# Fine-tuning enhances an existing circuit (entity tracking): circuit transfer, faithfulness and cross-model patching

Paper: [@prakash2024finetuning] (Prakash, Shaham, Haklay, Belinkov, Bau)
Source: arXiv 2402.14811v1 (HTML rendering read: Abstract, Sec. 1-7, App. C headline; other appendices by heading)

## Learnings

- **Setting.** Llama-7B and three fine-tunes: Vicuna-7B (conversations), Goat-7B (arithmetic, LoRA), FLoat-7B
  (arithmetic, full fine-tune) on box-content entity tracking (7 boxes; "chance accuracy is 0.14", Table 1).
  Arithmetic fine-tuning raises full-model accuracy from 0.66 to 0.82 (Table 1).
- **Claim.** "in both the original model and its fine-tuned versions primarily the same circuit implements entity
  tracking"; "the entity tracking circuit of the original model on the fine-tuned versions performs better than the
  full original model"; the gain "is primarily attributed to its improved ability to handle the augmented positional
  information"; overall "fine-tuning enhances, rather than fundamentally alters, the mechanistic operation of the
  model" (Abstract).
- **Test 1, circuit transfer (same components?).** Base circuit found by path patching: "a sparse set of 72
  attention heads in four groups" (Sec. 1). Evaluated *unchanged* in each fine-tuned model with faithfulness = "the
  percentage of model performance that can be recovered with the circuit, i.e. F(Cir)/F(M)", all other heads
  mean-ablated (Sec. 4.2). Table 1 (full / circuit / random circuit, faithfulness): Llama 0.66/0.66/0.00, 1.00;
  Vicuna 0.67/0.65/0.00, 0.97; Goat 0.82/0.73/0.01, 0.89; FLoat 0.82/0.72/0.01, 0.88. Null: "10 random circuits with
  the same total and per-position number of heads; random circuits have virtually zero accuracy" (Sec. 4.3).
- **But the fine-tuned circuits are bigger.** Circuits discovered directly in Goat / FLoat have "175 attenton heads
  and approximately forming a superset of the Llama-7B circuit": "fine-tuning is inserting additional components to
  the circuitry that performs entity tracking" (Sec. 4.3).
- **Test 2, same function?** Desiderata-based Component Masking (DCM; counterfactual patching with learned binary
  masks) assigns roles (Value Fetcher, Position Transmitter, Position Detector, Structure Reader); the roles carry
  over: "neither additional functionality nor a shift in functionality is introduced in fine-tuned models" (Sec. 5.3),
  while e.g. "Goat-7B can achieve a performance improvement of 20% compared to Llama-7B" on the Value Fetcher
  desideratum (Sec. 5.3).
- **Test 3, where is the gain? (CMAP).** Cross-model activation patching: "patching activations of the same
  components of different models on the same input" (Sec. 6.1). Patching the Value Fetcher heads' outputs from
  Goat / FLoat into Llama-7B gives "the maximal increase in performance ... recovering the full fine-tuned models'
  performance" (Sec. 6.2, Fig. 4), and "the activations of fine-tuned models are compatible with base model, even
  though they could have been using completely different subspaces and/or norms to encode information" (Sec. 6.2).

## Evidence and limitations

- One task, one base model, three fine-tunes; the authors say "Understanding whether such mechanism invariance is
  typical will require experience with further tasks on more models" (Sec. 7). Holds for LoRA and full fine-tuning
  ("The mechanism invariance is observed in both low-rank adaptations (LoRA) ... and fully fine-tuned models",
  Sec. 1).
- Circuit size is a judgement call: path patching "does not provide a clear threshold for the number of heads that
  should be included in the circuit" (Sec. 4.2); the base circuit "is not perfectly complete" (App. C).
- 'Enhances' covers adding ~100 heads with the same roles. So 'same mechanism' here means same algorithm (roles,
  information flow), not same parameters or same components; a strict 'no new components' criterion would call it
  partly new.
- The fine-tuning task (arithmetic) differs from the evaluated task (entity tracking): the evidence is about
  transfer, not about the fine-tuning target itself. No RL.

## Connections and questions

- **Definition offered:** a capability's *mechanism* = the minimal faithful circuit (components + information flow)
  plus the functional role of each component group (DCM). Two models share the mechanism if the base circuit,
  frozen, recovers most of the other model's performance (faithfulness ≫ random circuit of equal size) and the role
  assignments carry over; the gain is localized by cross-model patching.
- **New vs better access:** explicit. **Enhanced existing mechanism** if (i) base circuit faithfulness in the tuned
  model is high (they report ≥ 0.88), (ii) component roles are unchanged, (iii) patching the tuned model's activations
  for a few base-circuit components into the base model recovers the tuned performance (CMAP). **New mechanism** would
  show as low faithfulness of the base circuit in the tuned model, different roles, or base-incompatible activations
  (CMAP fails). No numeric threshold is stated; random equal-size circuits are the null.
- **Null / floor:** random circuits with the same number and placement of heads (accuracy ≈ 0) and task chance
  (0.14). Nothing on the 'any k' objection: everything is scored by greedy accuracy on a 7-way task.
- **Transfer to our setting:** feasible and cheap at our scale (6 layers × few heads, 9.6 M params), and CMAP is the
  most direct test of 'the RL change is a better input to an old mechanism'. Concrete protocol on the excluded-middle
  family: (1) find a minimal faithful circuit for the classical step in r16 (activation / path patching on
  corrupted goals, e.g. A ∨ ¬A vs an intuitionistically provable goal of the same shape); (2) evaluate the same
  components in pend (does pend use them at all on this family?) and the reverse (pend's circuit for a related
  intuitionistic schema, frozen, inside r16); (3) CMAP: patch r16's head/MLP outputs at each layer into pend on the
  same proof state and measure the probability of the classical step. If patching a few late components moves pend
  to r16's step probability, RL improved access via an existing pathway; if every layer must be patched, or pend's
  downstream components cannot use r16's activations, RL built something new. Cost: one forward pass per patch, ~10^3
  states × (6 layers × components): minutes. Failure modes: circuits for multi-step proof generation are much less
  clean than single-token entity lookup; 'minimal circuit' depends on the corruption distribution and on a threshold
  the paper does not supply; with only 6 layers a 'superset circuit' may simply be most of the network.
- Related: `jain2023mechanistically.md` (wrappers, revival), `ward2025repurposes.md` and `venhoff2025base.md`
  (transfer of base-model directions into reasoning models), `mukherjee2025subnetworks.md` (RL updates small
  subnetworks), `hewitt2019control.md`.
