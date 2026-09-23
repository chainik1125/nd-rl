---
written_on: 2026-09-22
written_by: agent:codex
---

# Pretraining research ideas

## Summary

Fifty hypotheses for Robbie's pretraining-only autoresearch workstream. These are
proposals, not observed results or promises of improvement. Robbie is excited about
P01 (looped architectures) and P02 (Muon). All fifty are a menu, not a mandatory queue
or an instruction to spend money on every entry.

Read the [weekly scope](robbie-pretrain-workstream.md) and
[experiment protocol](../../code/experiments/current/autoresearch/program.md) first.
Keep the initial dataset, data pipeline, tokenizer, RL procedure, evaluation, seed policy,
and harness budgets fixed. Architecture changes may persist into RL; its training
algorithm and hyperparameters must not change. Pretraining-only regularizers, auxiliary
heads, and optimizers must be disabled or removed before handoff.

Every experiment must fit the existing parameter and wall-clock caps, including any
auxiliary training parameters. Preserve the model/checkpoint interface and verify cached
versus uncached forward behavior before spending GPU money. Shared weights need distinct
KV caches for distinct logical layer applications and deduplicated optimizer parameters.
If an idea cannot fit these constraints inside `pretrain.py`, report the specific blocker;
do not change the frozen harness to force it through.

## How the agent should use this

- Read this at session startup alongside prior registrations and results. Put the stable
  idea ID, e.g. `P02`, in the existing registration `description`; no schema change needed.
- Robbie's excitement about P01 and P02 is context, not a required priority order.
  A concrete compatibility blocker is a reason to defer, not permission to alter scope.
  Successful execution and an honest negative result both count as trying a direction;
  follow-ups need a new hypothesis.
- Pick one intervention per experiment against the current eligible incumbent. An entry
  that depends on another intervention is conditional on a compatible incumbent; do not
  silently stack several new mechanisms into a nominally single-variable experiment.
- Preregister the mechanism, exact configuration, downstream prediction, and any expected
  latency tradeoff. Read existing records before choosing: this is a static backlog, not
  an authoritative ledger of what has already run.
- The fixed downstream RL metric decides keep/discard. Loss curves, update norms, and
  other diagnostics only help explain it; do not turn them into a second selection rule.
- A useful diverse shortlist is P01, P02, P21, P23, P45, and P47. Use evidence from
  completed runs to choose among them rather than following numerical order.

## Ideas

### Looped and weight-shared architectures

### P01 — A fixed-depth looped transformer — Robbie is excited about this

Replace four independent logical blocks with two blocks applied as A → B → A → B.
Keep four applications during pretraining, RL, and evaluation; do not add test-time
compute. Hypothesis: reusing transformations encourages reusable proof-state operations
and improves downstream transfer despite fewer unique parameters. This is the simplest
first looped test, not a claim to isolate sharing from parameter count. Validate cache
ownership, checkpoint reload, and parameter deduplication before running.

### P02 — Muon during pretraining — Robbie is excited about this

Use Muon for hidden matrix weights and AdamW for embeddings, output weights, biases,
and normalization parameters. Keep the frozen RL AdamW optimizer unchanged. Preregister
Muon-specific learning-rate/scaling conventions rather than copying AdamW's LR blindly.
Hypothesis: a different update geometry produces a better RL starting point at the same
pretraining budget. The [author's implementation](https://github.com/KellerJordan/Muon)
is the implementation reference; do not assume the pod's installed PyTorch has Muon.
Any copied implementation needs its license/attribution preserved.

### P03 — An untied entrance and exit around a shared core

Use an untied first block, a shared middle block applied twice, and an untied last block.
Hypothesis: input parsing and output readout need specialization, while intermediate
proof-state transformations benefit from sharing. Keep four logical applications.

### P04 — One block reused at every depth

Apply one shared block four times, with no extra tokens or adaptive halting. Compare
against a compatible tied-block incumbent to test more aggressive sharing. Hypothesis:
the task favors repeated refinement more than four independently learned operations.

### P05 — Share attention, keep per-depth MLPs

Tie the attention projections across the existing four layers but leave MLPs independent.
Hypothesis: citation lookup is reusable across depth while the transformations applied
to retrieved information need layer-specific capacity.

### P06 — Share MLPs, keep per-depth attention

Tie only the MLP weights across the four layers. Hypothesis: a reusable local proof
transformation benefits from different retrieval patterns at each depth. This is the
complement of P05, not something to combine with it in the same first experiment.

### P07 — Shared blocks with separate normalization per application

On a looped incumbent, untie normalization parameters across logical iterations while
keeping the core projections shared. Hypothesis: iteration-specific activation scales
are the main obstacle to effective weight reuse.

### P08 — Tell the shared block which iteration it is on

On a looped incumbent, add a learned iteration-index embedding to the residual stream.
Hypothesis: shared weights can implement a sequence of operations if they know their
depth. This is an internal model feature, not an extra input token or dataset change.

### P09 — Learned residual gates for loop iterations

On a looped incumbent, add scalar residual gates per logical application, initialized
to preserve the starting residual scale. Hypothesis: different passes should contribute
different update sizes even when their transformation weights are shared.

### P10 — Reinject the original representation at each loop

On a looped incumbent, give each iteration a gated residual connection from the input
embedding representation. Hypothesis: repeated transformations lose access to useful
original symbols; reinjection preserves them without extra context or external memory.

### Optimizer and schedule choices

### P11 — Separate Q, K, and V updates under Muon

On a Muon incumbent, orthogonalize/update Q, K, and V matrices separately instead of as
one concatenated projection. Hypothesis: treating different attention roles as one
matrix creates an unhelpful coupling. Keep the forward computation unchanged.

### P12 — Momentum SGD as the matrix-optimizer control

Compare Muon's hidden-matrix updates with momentum SGD, retaining the same auxiliary
AdamW parameter groups. Preregister sensible optimizer-specific scaling. Hypothesis:
any Muon advantage comes from matrix update geometry rather than momentum alone; this
is a mechanism check, not a numerically identical-update control.

### P13 — Slower second-moment averaging in AdamW

Change pretraining beta2 from the baseline 0.95 to 0.99, leaving other settings fixed.
Hypothesis: less reactive variance estimates protect infrequent but transferable proof
features from noisy minibatch updates.

### P14 — Unitwise adaptive gradient clipping

Replace the current pretraining clipping rule with a preregistered unitwise threshold
relative to parameter norm. Hypothesis: a few large gradients currently suppress useful
updates elsewhere. First verify the incumbent's clipping behavior rather than assuming it.

### P15 — Exclude biases and normalization parameters from weight decay

Keep the matrix decay coefficient fixed and alter only decay-group membership.
Hypothesis: decaying scale/offset parameters unnecessarily constrains useful internal
representations. Skip if the incumbent already uses this grouping.

### P16 — A lower embedding and readout learning rate

Use a fixed lower LR multiplier for embedding/output parameters, keeping hidden-layer
LR and schedule unchanged. Hypothesis: slower changes to the token interface let hidden
representations develop features that survive the transition to RL.

### P17 — Muon followed by a brief AdamW finish

On a Muon incumbent, switch to AdamW for a preregistered final fraction of pretraining,
within the same time cap. Hypothesis: the handoff benefits from weights adapted to the
optimizer used in RL. No optimizer state is transferred to the frozen RL stage.

### P18 — Warmup, stable learning rate, then short decay

Replace cosine decay with a wall-time-budgeted warmup/stable/decay schedule, keeping
peak LR fixed. Hypothesis: more time at a productive learning rate learns useful
structure, while a short finish preserves a stable RL starting point.

### P19 — Anneal the learning rate all the way to zero

Change only the cosine schedule's final LR floor from the incumbent value to zero.
Hypothesis: a quieter final checkpoint transfers better than the more plastic endpoint.
The opposite outcome would argue for preserving plasticity, not merely lower loss.

### P20 — Longer warmup at the same peak learning rate

Increase only the warmup duration, keeping the total pretraining time fixed and accounting
for it in the remaining schedule. Hypothesis: early unstable updates damage features
that a short training run cannot subsequently recover.

### Untied architecture and parameter allocation

### P21 — RMSNorm instead of LayerNorm

Replace the incumbent normalization with RMSNorm while preserving normalization placement.
Hypothesis: retaining the residual stream's mean information changes what the model can
carry between proof steps. Check the incumbent first; this is not a no-op experiment.

### P22 — Normalize branch outputs as well as inputs

Add normalization on attention and MLP branch outputs while retaining pre-normalization.
Hypothesis: controlling residual update scales makes learned features more robust to RL
updates. Keep width, depth, and the attention mechanism unchanged.

### P23 — A parameter-matched SwiGLU feed-forward block

Replace the current MLP activation/layout with SwiGLU, adjusting its intermediate width
to approximately preserve unique parameter count. Hypothesis: multiplicative feature
selection better represents the conditions under which a proof operation is valid.

### P24 — More capacity in the MLP

Increase only the MLP expansion ratio within the fixed parameter cap. Hypothesis:
symbol transformation capacity, rather than retrieval capacity, limits downstream
proof learning. Record the parameter and latency increase; this is not parameter-matched.

### P25 — Wider residual stream, narrower MLP

Reallocate approximately the same parameter count from MLP expansion into model width,
keeping logical depth fixed. Treat this as one preregistered allocation change.
Hypothesis: carrying more simultaneous proof-state features beats a larger local MLP.

### P26 — More attention heads at fixed model width

Increase head count without changing total residual width. Hypothesis: separate retrieval
channels help track multiple premises or citations. Check head-dimension divisibility,
positional encoding constraints, and cached inference support before launch.

### P27 — Tie the input embeddings and output readout

Share token embedding and output projection weights, if not already tied. Hypothesis:
using a common representation for reading and emitting symbols is a useful inductive
bias. Report the lower unique parameter count rather than quietly reallocating it.

### P28 — Remove projection biases

Remove affine biases from attention and MLP projections, leaving normalization choices
unchanged. Hypothesis: fewer input-independent offsets improve compositional transfer.
Skip any already bias-free incumbent instead of registering a nonexistent intervention.

### P29 — Parallel attention and MLP residual branches

Have attention and MLP read the same normalized block input and sum their outputs,
rather than feeding the attention-updated stream into the MLP. Hypothesis: preserving
independent retrieval and transformation paths improves the learned representation.

### P30 — An input-dependent gate on attention output

Multiply the attention branch output by a learned gate derived from the current hidden
state, initialized near the incumbent behavior. Hypothesis: retrieval should sometimes
be suppressed instead of always perturbing the proof-state representation.

### Position and attention mechanisms

### P31 — Longer positional wavelengths

If the incumbent uses RoPE, increase its base by a preregistered factor while keeping
all sequence lengths fixed. Hypothesis: less rapidly varying positional features make
longer-range citation relationships easier to learn from the existing dataset.

### P32 — Apply RoPE to only part of each attention head

Keep a preregistered fraction of Q/K dimensions unrotated. Hypothesis: attention benefits
from both position-sensitive citation tracking and a position-independent content channel.
Keep the rotary dimensions valid and preserve cache-position handling.

### P33 — Learned relative-position attention bias

Replace the incumbent positional mechanism with a small learned, bucketed causal
relative-position bias. Hypothesis: discrete distance categories match proof references
better than the incumbent geometry. Budget any loss of fused attention performance.

### P34 — A fixed distance bias instead of rotary position

Test ALiBi-style per-head linear distance penalties as the positional mechanism, with
no change to context length or data. Hypothesis: a simple locality prior is sufficient,
freeing content features from rotary position dependence. Verify cached/full equivalence.

### P35 — Normalize queries and keys

Normalize Q and K within each head before their dot product, with a preregistered scale.
Hypothesis: limiting attention-logit magnitude prevents premature, brittle concentration
on individual tokens. This does not change the evaluation sampler's temperature.

### P36 — Learn an attention temperature for each head

Add a positive learned scale on each head's attention logits, initialized to the current
scale. Hypothesis: different heads need different retrieval sharpness. Leave sampling
temperature, output logits, and all decoding hyperparameters unchanged.

### P37 — Different positional scales at different depths

Use a fixed preregistered progression of RoPE scales across layers instead of one shared
scale. Hypothesis: early local parsing and later long-range proof-state integration need
different distance resolution. Keep these scales identical across pretraining and RL.

### P38 — A residual path for early value representations

Give later attention layers a gated contribution from the first layer's value projection
at each token. Hypothesis: preserving early symbol information helps retrieval after
several transformations. Verify incremental decoding never reuses values from wrong tokens.

### Pretraining regularization, initialization, and checkpoint choice

### P39 — Residual-branch dropout during pretraining only

Add a small fixed dropout rate to branch outputs while pretraining, then explicitly
disable it before handing the model to RL. Hypothesis: avoiding brittle co-adapted
features improves transfer. Merely calling `eval()` is insufficient: RL calls `train()`.

### P40 — Attention-probability dropout during pretraining only

Regularize attention weights with a small dropout probability, restoring zero dropout
before RL. Hypothesis: learning redundant retrieval paths makes proof representations
more robust. Check kernel compatibility and account for any throughput cost.

### P41 — Small activation noise during pretraining only

Inject zero-mean noise with a fixed, scale-aware magnitude into a selected residual
activation; remove the noise path before RL. Hypothesis: representations robust to small
perturbations are easier for RL to improve without destroying existing capabilities.

### P42 — Orthogonal initialization of hidden projections

Initialize hidden projection matrices orthogonally with variance matched to the incumbent,
leaving embeddings/readout initialization unchanged. Hypothesis: better initial signal
propagation matters when pretraining is short. Keep the random-seed policy fixed.

### P43 — Depth-scaled initialization of residual output projections

Scale attention-output and MLP-output initialization with logical residual depth while
leaving other initialization unchanged. Hypothesis: smaller accumulated residual updates
provide a better early optimization trajectory. For loops, use logical applications.

### P44 — Near-identity blocks at initialization

Initialize residual output projections at a small nonzero scale relative to the incumbent.
Hypothesis: initially learning small edits to the token representation is easier than
learning to undo large random transformations. Keep this separate from learned gates.

### P45 — Hand off an exponential moving average of weights

Maintain a fixed-decay EMA during pretraining and pass those weights to RL instead of
the final raw weights. Hypothesis: smoothing optimization noise produces a better RL
initialization. Include averaging overhead in the time budget; do not select using dev RL.

### P46 — Average checkpoints over a fixed final window

Average weights at predetermined points in the final portion of pretraining and hand off
that average. Hypothesis: a broader local solution transfers better than the endpoint.
Unlike P45, this uses a uniform finite window; do not combine the two or choose snapshots
after inspecting downstream scores.

### Pretraining objectives on the same examples

### P47 — Mild label smoothing

Replace hard-target cross-entropy with a small preregistered smoothing coefficient on
the same supervised positions and vocabulary. Hypothesis: reducing pretraining
overconfidence preserves useful alternatives for the fixed RL procedure to explore.

### P48 — An explicit entropy bonus during pretraining

Add a small predictive-entropy bonus to cross-entropy on the existing supervised positions.
Hypothesis: maintaining alternatives improves downstream exploration. Unlike P47, the
penalty depends on the model distribution; no RL entropy coefficient is changed.

### P49 — A logit-normalizer penalty

Add a small squared-log-partition (z-loss) term to the unchanged token prediction loss.
Hypothesis: controlling logit scale improves training stability and the subsequent RL
starting point. Track entropy diagnostically without changing the decoding distribution.

### P50 — Intermediate-layer next-token supervision

Apply the existing readout to one intermediate representation and add a fixed-weight
next-token auxiliary loss using the same targets. Hypothesis: encouraging usable features
before the final layer improves compositional transfer. Remove the auxiliary loss path
at handoff; do not add examples, target reweighting, a teacher, or verifier feedback.

## Evidence and provenance

- Written on branch `autoresearch-pretrain`, with HEAD `d9257de`, after Robbie explicitly
  requested roughly fifty ideas and highlighted his excitement about looped architectures
  and Muon, without making them mandatory priorities.
- The current [012 three-seed baseline](../../code/experiments/current/autoresearch/results/012_baseline-3seeds/results.json)
  scored 139, 141, and 163 (rounded mean 148). The
  [013 weight-decay increase](../../code/experiments/current/autoresearch/results/013_wd-1/results.json)
  scored 37, 89, and 91 (rounded mean 72) and was discarded. These are observations,
  not evidence for any untested mechanism above. Check live records for newer results.
- Older runs already tried a higher LR, shallower/deeper models, zero weight decay, and
  a joint batch-size/LR change, often under a different budget or single-seed protocol.
  Read their records before proposing an exact repeat or generalizing their failures.
- Muon's parameter grouping is grounded in the
  [author's implementation](https://github.com/KellerJordan/Muon). Its effectiveness for
  this proof task is untested here; the other entries are working hypotheses, not a
  literature survey or claims that named mechanisms are new.

## Next steps

Choose a contract-compatible hypothesis using prior evidence and preregister it as a
separate experiment. Use downstream results to select the next mechanism. Keep findings
in the existing experiment registrations/results/summaries, with
idea IDs for traceability. This backlog does not authorize changes to data, tokenization,
RL, the harness, or the fixed resource/evaluation protocol.
