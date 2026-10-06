---
written_on: 2026-10-05
written_by: agent:codex
status: proposed
papers:
  - pmlr-v267-lee25ab
  - pmlr-v267-shen25o
base_branch: dmitry/main
base_commit: 0fc78d51bc654fef6862c0700de076d0eb0d51e5
---

# Learning to select proofs and allocate attempts

## Summary

We want to test whether a natural-deduction model can recognize its own successful
proofs and decide how many attempts to make. The proposal contains three independent
experiments, ordered by the complexity of the question. Each starts from the same
base proof-model checkpoint, with its own training run and controls. No experiment
inherits a selector, stopping policy, or optimizer state trained in another experiment.

| Experiment | Question | Training objective |
| --- | --- | --- |
| 1. Select from eight rollouts | Can the model identify a successful proof among its own attempts? | Correctness of the selected proof |
| 2. Stop with an attempt cost | Can the model allocate attempts when another attempt has an explicit cost? | Selected-proof correctness minus a cost per additional attempt |
| 3. Stop without an attempt cost | Can the model anticipate when accumulating context makes further attempts harmful? | Selected-proof correctness only |

The ladder concerns experimental complexity, not a training curriculum. Shared code,
evaluation conventions, and starting checkpoints make the experiments comparable;
they do not create dependencies between their trained models.

Across all three experiments, entropy is a diagnostic rather than a reward or a
hand-written stopping rule. The main mechanistic question is whether the learned
decision tracks uncertainty and the expected benefit of further computation.

## Evidence and motivation

ReVISE trains a model to choose between an end token and a refinement token using
correctness-based preferences, then teaches correction with successful solutions.
It uses SFT and DPO without an explicit length penalty. Its sequential-refinement
evaluation extends a model trained for one refinement to several attempts. This
provides a precedent for an internal acceptance decision, but its explicit
correct-answer-to-stop supervision would confound a test of whether stopping emerges
from final correctness alone [@pmlr-v267-lee25ab].

Satori teaches continue, reflect, and explore actions, followed by PPO training that
can restart from partial successful and unsuccessful trajectories. Its reward includes
correctness, a learned outcome score, and reflection bonuses. It motivates training
reasoning actions and recovery from mistakes; the experiments below simplify its
action space and omit those additional reward components [@pmlr-v267-shen25o].

Neither paper establishes that a small natural-deduction model will learn to stop
because its accumulated context has become harmful. That is the hypothesis of
experiment 3, rather than an assumed result of the proposed method.

Sources and implementations:

- [ReVISE ICML paper](https://proceedings.mlr.press/v267/lee25ab.html) and
  [released code](https://github.com/seunghyukoh/ReVISE).
- [Satori ICML paper](https://proceedings.mlr.press/v267/shen25o.html) and
  [released code](https://github.com/satori-reasoning/Satori).
- The local [symbolic scratchpad harness](../../code/experiments/current/r1_cot/README.md)
  supplies relevant verifier integration, rollout logging, and policy-update machinery.
  Its existing interface does not implement these selection and stopping experiments.

## Common experimental design

Choose one base checkpoint with nontrivial proof success and failure on development
problems. Record its hash, tokenizer, training exposure, sampling settings, and context
capacity. Use fresh theorem-level train, development, and test splits, checking
structural overlap as well as literal duplication. Existing protected evaluation
sets retain their restrictions. Freeze the test protocol before inspecting test results.

Use the same base checkpoint for all experiments and three independent training seeds
per primary condition. Any vocabulary expansion or interface initialization must be
matched across relevant controls. Experiments 2 and 3 must use the same initialization
procedure, without teaching that correct proofs imply acceptance. Start with an explicit
action interface; if format-only supervision is needed, balance its action examples
independently of correctness and context length, and report it as an intervention.

Let `V(x, y)` be the strict proof check against the original theorem and premises.
The checker supplies training labels and evaluation measurements. Its verdict and
error messages are hidden from the model at inference. Do not terminate an episode
automatically just because the checker found a valid intermediate attempt.

For sequential experiments, the model generates numbered proof attempts. At each
attempt boundary it emits `CONTINUE` or `ACCEPT`. `CONTINUE` generates another attempt
conditioned on the theorem and accumulated history. `ACCEPT` is followed by the index
of any previous attempt. The environment retrieves that exact proof for verification;
the model does not have to reproduce it. Previous failed attempts remain in context.
The original premises stay authoritative: an earlier attempt cannot introduce premises
that make a later invalid proof count as valid.

Use fixed protocol boundaries and the same action grammar in both sequential
experiments. Record masking and sampling probabilities explicitly. A rollout for an
RL update is a complete episode, potentially containing several proof attempts; the
number of episodes used to estimate advantages is separate from the attempt budget.

Set a maximum attempt count and reserve space for final selection. At an attempt or
context limit, invoke the same final-selection procedure and score its result, marking
the termination as forced. Do not award a bonus or impose an extra penalty for hitting
the limit. Log incomplete proofs and truncation separately, without silently dropping
history. Test conclusions about voluntary stopping on states with space for at least
one further complete attempt. Confirm that eight representative proofs fit before
fixing the first experiment's context configuration.

## Experiment 1 Select among eight proofs

**Hypothesis.** The model can distinguish valid from invalid proofs well enough to
recover a substantial part of the gap between random selection and oracle pass@8.

Generate eight independent attempts per theorem from a frozen copy of the base model,
using a fresh context for each attempt. Present the theorem and all eight numbered
proofs to a trainable copy of the same base model. Train it to return an index. Freeze
the sampler so improved selection cannot be explained by improved generation.

For the simplest first test, use the checker to identify the set of valid indices and
maximize the total probability assigned to that set. Any valid candidate is a correct
target; do not arbitrarily teach a preference for the first valid proof. Pools with no
valid proof have no positive selection target and supply no update under this loss,
but remain in the evaluation distribution. Shuffle candidate order during training and
evaluation. A separate outcome-RL selector can test whether the same skill is learned
from the selected proof's binary reward.

Compare with random selection, first-candidate selection, length-normalized base-model
proof likelihood, and an untrained chooser using the same interface. Oracle pass@8
is an upper bound on selection from that exact candidate pool, not a deployable
model baseline. Majority voting on theorem truth is uninformative here because all
candidates attempt to prove the same theorem.

Report selected-proof accuracy, oracle pass@8, and accuracy conditional on at least one
valid candidate. Show results by number of valid candidates, proof length, and candidate
position. Evaluate transfer to other pool sizes only after measuring the primary
eight-candidate task.

**Success criterion.** Held-out selection improves over the strongest non-oracle
baseline, with a positive paired confidence interval across theorem pools and a
consistent effect across seeds. Report the fraction of the oracle-minus-random gap
recovered; a high pass@8 with poor selection identifies a recognition bottleneck.

## Experiment 2 Stop with a cost per attempt

**Hypothesis.** A policy can allocate attempts more effectively than a fixed attempt
count when additional attempts have a known cost.

Start independently from the base checkpoint. Train the single model's proof generation,
acceptance, and selection actions using complete-episode RL. Use the existing clipped
policy-update machinery as a starting point, extending it to retain likelihoods for
all generated proof and decision tokens. The undiscounted episode reward is

\[
R_\lambda = V(x,y_{\mathrm{selected}})-\lambda(N-1),
\]

where `N` counts proof attempts, including failed or incomplete ones. The first attempt
is mandatory and incurs no additional-attempt cost. There is no separate token-length
penalty. Begin with a maximum of eight attempts and a provisional cost sweep of
`lambda = 0.01, 0.03, 0.10`, training each condition independently. Select any revised
sweep using development results only.

Compare against fixed counts of 1, 2, 4, and 8 followed by model selection. At evaluation,
force these counts in the same trained model to isolate the stopping decision. Also
include a fixed-count training control from the same base checkpoint, so a poor chooser
under forced continuation does not create an artificially weak comparison. Randomized
fixed-count mixtures provide baselines at intermediate mean attempt counts.

**Success criterion.** The policy improves held-out expected reward over the
development-selected fixed-count baseline at the same cost. Report accuracy versus
mean attempts and actual token/forward-pass cost: a per-attempt reward does not make
attempts equally expensive. Check that gains survive comparison with mixtures having
matched average compute, and that the policy changes its allocation across problems.

## Experiment 3 Stop without an explicit cost

**Hypothesis.** The model learns to accept before further computation lowers its
probability of ultimately selecting a valid proof because accumulated context has
become harmful.

Start independently from the same base checkpoint, using the sequential interface and
training procedure of experiment 2 with `lambda = 0`. Train on final selected-proof
correctness only. Use undiscounted returns, no length or attempt reward, no reflection
bonus, and no supervised stop labels. Keep the loss normalization and other training
settings comparable to experiment 2 and record any stabilization terms. Do not initialize
from a cost-trained model or from experiment 1's trained selector.

Early stopping alone does not establish the hypothesis. The model might recognize an
already-valid proof, learn a fixed counter, reach the context limit, or retain a format
prior. Moreover, when previous candidates are retained, oracle pass@N cannot decrease
as more proofs are added. Degradation must occur in generating or selecting proofs;
the candidate pool itself cannot lose an existing valid proof.

### Measure whether continuing helps

Freeze a trained checkpoint and collect decision states from held-out development
episodes. Include both states where the model tends to accept and states where it tends
to continue. From each identical state, evaluate acceptance now and multiple independent
continuation branches. After forcing one additional attempt, resume the learned policy.
Use the same selection sampling convention in both branches. Estimate

\[
\Delta(s) =
\Pr(V=1\mid s,\text{force continue then follow policy})
-\Pr(V=1\mid s,\text{accept now}).
\]

Negative `Delta` means continuation is harmful for this model and environment. This is
a policy-relative estimate, not the value of all possible future reasoning strategies.
Repeat a subset with two or four forced attempts to detect delayed benefits. Use
replicated continuations and confidence intervals rather than treating one failed
branch as evidence of a stopping boundary.

Measure next-attempt validity, oracle success among all attempts, and selected-proof
accuracy separately. These distinguish generation degradation from losing the ability
to select a proof that is already available. Report the relationship between acceptance
probability and `Delta`, including within states that already contain a valid candidate.

### Test the context explanation

First establish whether forced continuation actually causes degradation on this task.
Use the following interventions to separate context effects from attempt counting:

- **Matched candidate tests:** hold the theorem and selectable proof pool fixed while
  varying the amount and position of additional, explicitly nonselectable history.
  Include both task-relevant history and unrelated filler controls. Measure changes in
  selection, acceptance, and continuation value.
- **Compact-history control:** retain the theorem, a model-selected incumbent, and the
  latest attempt instead of the complete history. This changes both context length and
  accessible candidates, so report candidate availability and interpret it together with
  the matched-pool tests rather than as a pure length intervention.
- **Count and length transfer:** vary proof lengths and attempt limits within the
  checkpoint's supported context capacity. Test counts such as 4, 8, and 16 when they fit.
  Distinguish nominal budget effects from token occupancy; exclude truncation-driven
  failures when making claims about interference before the limit.

Compare with forced fixed counts, development-tuned stopping based only on context
length or attempt count, and a confidence-threshold baseline. Do not choose a threshold
using test outcomes. Repeat the principal measurements on the frozen test split after
development choices are fixed.

**Success criterion.** There is measurable harm from continuing before the hard limit;
the learned policy preferentially accepts in those states; and context interventions
change its stopping behaviour in the direction predicted by the measured continuation
values. It should improve selected-proof accuracy over the appropriate forced-continuation
and simple-threshold controls. If no degradation occurs, this setting does not expose
the hypothesized incentive to stop. If degradation occurs but stopping does not track it,
the model has not demonstrated the proposed skill.

## Entropy and uncertainty analysis

Log mean next-token entropy over each proof, using a fixed diagnostic temperature,
alongside the actual sampling settings. Record acceptance probability and the entropy
of the acceptance decision separately. Low binary decision entropy can mean confident
acceptance or confident continuation; it does not indicate which action is preferred.

Compare acceptance probability with proof entropy, checker correctness, context length,
attempt count, and estimated `Delta`. Fit diagnostic predictors on development data and
test whether entropy improves prediction beyond the simpler covariates. Control for
proof length and token type where practical: many valid alternative proofs can produce
high entropy, while a confidently wrong proof can have low entropy. An association with
entropy is evidence about the policy, not proof that entropy causes the decision.

## Execution and deliverables

Before launching training, fix the checkpoint and data hashes, exact action protocol,
context and attempt caps, training-token budget, seed list, and evaluation sample sizes.
Use a small development pilot to estimate variance and choose enough theorem-level
samples for the intended comparison. Report paired uncertainty over theorems as well
as variation across training seeds; proof attempts from one theorem are not independent
test examples.

At execution, create separate working and summary folders for each experiment under
`code/experiments/current/` and `experiment-summaries/`, following the
[workspace instructions](../../AGENTS.md). Keep this proposal as their shared design
reference. These folders should contain the independent checkpoint lineage, manifests,
raw episode records, evaluation decisions, and final interpretation for that experiment.

Implement and verify the episode loop before running models: exact-proof retrieval,
original-premise verification, action accounting, hidden verifier outputs, forced versus
voluntary termination, context reservation, and behavior-policy likelihoods. Reuse the
[existing runner](../../code/experiments/current/r1_cot/run.py) selectively rather than
changing the protocol of its existing experiment.

The main outputs are:

1. Selection accuracy and the remaining gap to oracle pass@8.
2. Accuracy and expected reward versus attempt and token cost for cost-trained policies.
3. Acceptance probability versus measured continuation value, with the context
   interventions and generation/selection decomposition.
4. An entropy analysis reporting associations, controls, and failures to generalize.

The experiments may be scheduled in increasing implementation complexity, but each can
be run and interpreted independently. Failure of experiment 1 is informative about
selection difficulty; it is not a requirement to initialize or cancel experiments 2 or 3.
