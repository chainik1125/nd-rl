---
written_on: 2026-10-13
written_by: agent:claude
---

# Standing policy for agent runs

Every run brief is read together with this file. Where they disagree, this file wins unless the brief says why it
deviates. Each standard below records **why** it exists, so an agent can tell when a deviation is justified. A
deviation is allowed; an unexplained one is not.

## The one hard rule

**Agents do not contact or involve other people.** No messages, no PR reviewers, nothing sent or posted on anyone's
behalf. Draft PRs with no reviewers are fine. The person who launched the run reads the results and decides what to
share. *Why:* people's time is the scarcest resource in the group, and an unreviewed agent draft should not cost
anyone else's.

## Judging proofs

- **Lean alone decides.** A proof counts if and only if Lean 4 core accepts it:
  - for `lean_seq` models, the literal sampled text parses in the strict grammar and Lean accepts it;
  - for free-form Lean, the judge's elaborated-term allowlist accepts it (`code/shared_code/lean_check.py`).

  The ND verifier is not a judge: not of reward, evaluation or counting. Proofs in write-ups are shown in Lean.
  *Why:* Lean is sound and readable. The proofs the ND verifier rejects and Lean accepts are valid proofs that skip a
  step ND's rule format demands, e.g. `n.elim` on `n : ¬A`, or omitting the restatement of a premise. The ND
  verifier also had a soundness bug: forward box citations allowed circular "proofs".
- **Label the checker whenever numbers from different judges are compared.** Results judged by the ND verifier, or by
  Lean ∧ ND verifier, are not directly comparable with Lean-alone results.
- **Report proof length as Lean term size as well as lines.** ND line counts include one line per premise, so an
  unused premise lengthens a proof. Lean needs neither the restatement nor, often, the ND detours. ND-derived
  minimal lengths are therefore upper bounds under Lean. Term size is invariant to both.

## Every number names its model

Each table and each `numbers.md` entry names the checkpoint behind it:
- the model id or path and the parameter count;
- the format (`lean_seq`, proof state, whole proof);
- from scratch or pretrained;
- the training data.

An inherited number keeps its label, and a reviewer treats an unlabelled number as a finding. *Why:* one run applied
a premise measured on a different model than the one it was about.

**Named models are immutable.** Once a model has a name and a model card, a changed variant gets a new name (e.g.
`state_env_october` → `state_env_october_r16`) and its own card. *Why:* earlier discussions refer to the old
definition.

## Pre-registration

Before the first unit of compute, commit `preregistration/<run>.md` (template:
[templates/preregistration.md](templates/preregistration.md)). It contains:
- the question;
- the design actually run;
- **numeric or otherwise falsifiable expected results**;
- the minimum detectable difference at the planned seed count;
- what the compared arms are matched on;
- the budget and the stop rule.

Expectations that miss are reported as misses. *Why:* expectations written afterwards always fit. Several of our
results changed meaning when compared with what had been predicted.

## Seeds and noise

- **Plan seeds against the measured noise floor.** The pre-registration states the minimum detectable difference for
  the headline quantity at its seed count. A comparison whose predicted effect is inside it is redesigned (more seeds,
  a paired or within-run design) before it is launched. *Why:* at n = 2, 18 of our 20 standing findings sat inside
  the noise floor.
- **Use the same seeds in every arm.** Report per-seed values plus the interquartile mean with a stratified-bootstrap
  95 % interval (Agarwal et al., NeurIPS 2021).
- **Two seeds before calling a difference;** more for anything that might be a rare event. A per-seed "discovery"
  that happens on 1 seed in 3 needs on the order of 12–15 seeds per arm to test.

## Sampling and read-outs

- **Fast decoding at the largest batch that fits,** and record peak memory with the batch.
- **Set the length caps from the pool and report the cut-off fraction.** The caps are the maximum new tokens, steps
  and action length. If more than ≈ 0.1 % of any reported stratum is cut off, raise them. *Why:* a cap that truncates
  silently biases exactly the length results this project is about.
- **Hold batch, caps and decode flags fixed across the arms of one experiment.** Comparing with an on-file number
  measured at other settings is fine, but say so. A settings change is a sampling redraw (bf16 reduction order), not a
  correctness change.
- **Proof-state models report a plain and a guided number.**
  - **Plain:** independent attempts.
  - **Guided:** each step is checked by a Lean-sound step checker, and a failing step is redrawn from the same state
    without replacement, up to 10 rejections per attempt.

  Label which is which, and compare plain with plain and guided with guided. RL rollouts stay plain unless a brief
  says otherwise. *Why:* guided read-outs beat plain by 5–9 pp at matched sampled tokens across 6 models.

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
- Report base-model reachability with every "RL solved X".
- Make splits disjoint by renaming class (atom relabelling), and, for curried variants, by curried normal form.
- Make every count reproducible from files that are pulled back and kept.
- Never let a verdict marker or a stale verdict leak into training data or relabelling.

## Literature

Every claim cited from a paper is checked against the source text: fetch it, find the passage, record the location.
Anything not checked is marked UNVERIFIED. *Why:* paraphrase drift was common in unchecked agent summaries.

## Working with agents

- **Questions go in `QUESTIONS.md`.** Date each one and give the default the run will follow if unanswered. Commit it
  and proceed on the default. Direction, scope and spend are all fair questions.
- **Keep the working context small.** Session cost grows with turns × context size.
  - Read the part of a file you need.
  - Send long output to a file and read back a summary.
  - Count in scripts whose stdout is a table.
  - Don't re-read files.
- **Be careful with small hosts.** Cap concurrent processes (we use 4) and download one file at a time. Upload
  artefacts as you go; never edit another run's files in place. *Why:* one run started 160 parallel downloads and
  took a host down for 8 hours.
- **Don't scar on the first cut.** One incident gets a direct fix and a note. New machinery or policy waits for a
  second incident.
