---
written_on: 2026-10-13
written_by: agent:claude
---

# Standing policy for agent runs

Every run brief is read together with this file. Where they disagree, this file wins unless the brief says why it
deviates. Each standard records **why** it exists, so an agent can tell when a deviation is justified. A deviation
is allowed; an unexplained one is not. Terms are defined in the [glossary](README.md#glossary).

Evidence cited below lives on the `dan` branch of this repository or on the fork `dan-pandori/nd-takehome`. Links are
given so the reasons can be checked.

## A firm default: agents do not contact other people

**Agents do not contact or involve other people.** No messages, no PR reviewers, nothing sent or posted on anyone's
behalf. Draft PRs with no reviewers are fine. The person who launched the run reads the results and decides what to
share; agents notify only that person. *Why (Dan):* out of respect for others' time, a person reads and understands
agent output before anyone else is asked to look at it. Many people also find direct contact from an agent rude.

This is a firm default, not an absolute. In a genuinely extreme situation an agent may use its judgment, and tells
the person who launched it at once what it did and why.

## Judging proofs

- **Lean alone decides.** A proof counts if and only if Lean 4 core accepts it.
  - **`lean_seq` model output:** the literal sampled text must parse in the strict grammar, and Lean must accept it
    (`code/shared_code/lean_check.py`). The grammar has no automation, no library lemmas and no `sorry`, so the
    grammar does the allowlisting.
  - **Free-form Lean** (text outside that grammar): it also needs an elaborated-term allowlist and an axiom check, so
    that `simp`, `tauto`, `decide`, library lemmas and `sorry` cannot count. **`main`'s `lean_check.py` does not do
    this.** It rejects only errors and `sorry`. The allowlist judge is the fork's
    [`lean_check.py`](https://github.com/dan-pandori/nd-takehome/blob/dan/lean_check.py).

  The ND verifier is not a judge of reward, evaluation or counting. Proofs in write-ups are shown in Lean.
  *Why:*
  - Lean is sound and readable.
  - The proofs the ND verifier rejects and Lean accepts are valid proofs that skip a step ND's rule format demands.
    Examples are `n.elim` on `n : ¬A`, or omitting a premise restatement
    ([lean-judge](https://github.com/chainik1125/nd-rl/blob/dan/experiment-summaries/2026-09-27-lean-judge-lean-alone-decides/README.md)).
  - The ND verifier also **has** a soundness bug, found after this decision: forward box citations allow circular
    "proofs". A fix is proposed in [#22](https://github.com/chainik1125/nd-rl/pull/22).
- **Label the checker whenever numbers from different judges are compared.** Results judged by the ND verifier, or by
  Lean ∧ ND verifier, are not directly comparable with Lean-alone results.
- **Report proof length as Lean term size as well as lines.** ND line counts include one line per premise, so even an
  unused premise lengthens a proof. Lean needs neither the restatement nor, often, ND's detours, so ND-derived
  minimal lengths are upper bounds under Lean. Term size is invariant to both: in `lean-judge` the median term size
  was 6 against a median of 10 lines.

## Every number names its model

Each table and each `numbers.md` entry names the checkpoint behind it:
- the model id or path and the parameter count;
- the format (`lean_seq`, proof state, whole proof);
- from scratch or pretrained;
- the training data.

A number measured on one model is never stated as a property of "the model" or of the setting. An inherited number
keeps its label. *Why:* one run applied a premise measured on a different model than the one it was about.

**Named models are immutable.** Once a model has a name and a model card ([template](templates/model_card.md)), a
changed variant gets a new name (e.g. `state_env_october` → `state_env_october_r16`) and its own card. *Why:*
earlier discussions refer to the old definition.

## Pre-registration

Before the first unit of compute, commit `preregistration.md` in the run's working folder
([template](templates/preregistration.md)). It contains:
- the question;
- the design actually run;
- **numeric or otherwise falsifiable expected results**;
- the minimum detectable difference at the planned seed count;
- what the compared arms are matched on;
- the budget and the stop rule.

Work added mid-run is pre-registered as a dated addendum before it runs. Expectations that miss are reported as
misses. *Why:* expectations written afterwards always fit.

## Seeds and noise

- **Measure the run-to-run noise floor of your own setup, and plan seeds against it.** Ours, on a 3.2M-parameter
  cap-6 model, is in [noise-floor](https://github.com/chainik1125/nd-rl/blob/dan/experiment-summaries/2026-09-25-noise-floor-measurement-floor/README.md)
  and the fork's [`NOISE_FLOOR.md`](https://github.com/dan-pandori/nd-takehome/blob/dan/NOISE_FLOOR.md).
  - The pre-registration states the minimum detectable difference for the headline quantity at its seed count.
  - A comparison whose predicted effect is inside that difference is redesigned before launch: more seeds, or a paired
    or within-run design.

  *Why:* measured against that floor, 18 of our 20 standing two-seed findings were inside it.
- **Use the same seeds in every arm.** Report per-seed values plus the interquartile mean with a stratified-bootstrap
  95 % interval (Agarwal et al. 2021, [arXiv:2108.13264](https://arxiv.org/abs/2108.13264)).
- **Two seeds before calling a difference; many more for rare events.** One run estimated that an event seen on 1 seed
  in 6 needs about 12–15 seeds per arm for 80 % power (an unreviewed estimate in
  [create-or-elicit](https://github.com/chainik1125/nd-rl/blob/dan/experiment-summaries/2026-10-12-create-or-elicit-sigma-knockout-not-acquired-in-16-ei-rounds-needs-exposure-1pct-dose-restores/README.md)).

## Sampling and read-outs

- **Use your sampler's fastest exact decode path at the largest batch that fits,** and record peak memory with the
  batch. *Why:* on our setup that alone was about a 2× speed-up for RL sampling.
- **Set the length caps from the pool and report the cut-off fraction.** The caps are the maximum new tokens, steps
  and action length. If more than ≈ 0.1 % of any reported stratum is cut off, raise them. *Why:* a cap that truncates
  silently biases exactly the length results this project is about.
- **Hold batch, caps and decode flags fixed across the arms of one experiment.** Comparing with an on-file number
  measured at other settings is fine, but say so. A settings change is a sampling redraw (bf16 reduction order), not a
  correctness change.
- **Where a step checker is available, proof-state models report a plain and a guided number.**
  - **Plain:** independent attempts.
  - **Guided:** each step is checked by a checker validated against Lean, and a failing step is redrawn from the same
    state without replacement, up to 10 rejections per attempt. Finished proofs are still judged by Lean.

  The checker and guided sampler are the fork's
  [`step_check.py`](https://github.com/dan-pandori/nd-takehome/blob/dan/step_check.py) and `guided_sample.py`.
  Label which number is which, and compare like with like. RL rollouts stay plain unless a brief says otherwise.
  *Why:* in [guided-tts](https://github.com/chainik1125/nd-rl/blob/dan/experiment-summaries/2026-10-03-guided-tts-logical-redraws-beat-plain-on-long-theorems-structural-redraws-do-not/README.md),
  guided redraws beat plain at matched sampled tokens, in all 6 models:
  - on the 90 long theorems: +5 to +10 pp per seed (IQM +8 to +9);
  - on textbook72: ≈ +5 pp;
  - redrawing only structurally invalid steps gave ≈ 0.

## Compute

**Record the compute behind every arm.** Per arm, and per round for RL, record:
- GPU-seconds and the GPU type;
- generated tokens, and attempts or actions;
- training steps and training tokens;
- Lean or environment checks.

Matching compute is a design choice for each experiment, not a rule. The pre-registration says what is matched, and
the write-up flags any arm that used more than 1.25× its comparator on any of these. *Why:* a more elaborate method
can win by spending more. Recorded compute lets anyone check that afterwards, or rerun at matched compute.

Each run declares a compute budget in its brief and stays inside it. Going beyond it is a question for the person
who launched the run, asked with a costed default.

## Lessons from bugs

Each of these produced a wrong number once:
- Count distinct proofs after normalising names (start-index normalisation).
- Give the frozen control equal attempts.
- Report base reachability with every "RL solved X".
- Make splits disjoint by renaming class, and, for curried variants, by curried normal form.
- Make every count reproducible from files that are pulled back and kept.
- Never let a verdict marker or a stale verdict leak into training data or relabelling.

## Files, data and secrets

- **Never edit another run's files or shared data in place.** Write a new file. *Why:* some data files are shared
  between working copies, and an in-place edit silently changes other runs' inputs.
- **Never upload or commit configuration files, shell logs, or anything that could contain a token,** and never print
  tokens. Artefact stores may be public.
- **Upload large artefacts as you go,** not only at the end, and record their paths.

## Literature

Every claim cited from a paper is checked against the source text: fetch it, find the passage, record the location.
Anything not checked is marked UNVERIFIED. *Why:* summaries can drift from what a paper says, and checking against the
source catches it cheaply.

## Working with agents

- **Questions go in the run's `QUESTIONS.md`.** Date each one and give the default the run will follow if unanswered.
  Commit it and proceed on the default. Direction, scope and spend are all fair questions.
- **Keep the working context small.** Session cost grows with turns × context size.
  - Read the part of a file you need.
  - Send long output to a file and read back a summary.
  - Count in scripts whose stdout is a table.
  - Don't re-read files.
- **Cap parallelism on shared machines, and download one file at a time.** *Why:* one run started 160 parallel
  downloads and ran a host out of memory. Two agent sessions died, and nobody noticed for 8 hours.
- **Don't scar on the first cut.** One incident gets a direct fix plus an entry in an ideas list that names its
  trigger. New machinery or policy waits for a second incident.
