---
written_on: 2026-10-11
written_by: agent:claude
status: approved by Dan 2026-10-11, $150 (launched as `create-or-elicit`; `create-or-elicit-paper` queued after it)
---

# Proposal 27: a paper on whether RL created it, and the experiments it needs

## Context

Dan (2026-10-11) shared an outline from a Claude chat ("nd-rl repo paper results"). It proposed three papers, of which
the first is the main one:

> 1. Main paper: "Did RL create it?" depends on the budget and the interface. A from-scratch organism with exact
> likelihoods. […] What's missing is a causal result. That's the composition knockout (about $26): pretrain without
> DN-on-¬I, run RL, and see whether excluded middle appears. Then add the Tsilivis-style dial at small p. Without this,
> it reads as a careful measurement note and not a finding. Other must-fixes: a large-k (≥ 4,096) base re-read on
> B ∪ C, with a pre-registered threshold of log p ≥ −ln k; a power-sampling sharpening null; resolve the
> A6000-vs-A40 budget mismatch; more than 3 seeds for the headline counts.
>
> 2. Fold into paper 1 or make it a companion: "Discovery is a lottery, transfer is cheap." […]
>
> 3. Workshop paper: test-time compute before vs after RL. […]

**Dan's decisions:**
- Approve paper 1 as the direction, with **$150**, to be used "as you see most interesting."
- The agents should write the results up as a **paper draft** and as a **simplified blog post**.
- More spend may follow, depending on the results.
- No hurry: Claude quota is plentiful, and the agents get wide agency to propose extensions and literature reviews.

## Changes from the outline

**Corrections**, all checked against reviewed results:
- **"Across every control: 4–5"** is the count no seed's base ever solved (4 / 5 / 5). Surviving every RL-free control
  is 4 (theorem, seed) pairs on 3 theorems.
- **"Proof state reaches 28 of 29"** is from the 3.2 M organism, with 2 seeds per arm, and is confounded with
  step-by-step generation (claim audit).
- **The estimator's 0.93–0.98** is validated only where p is measurable directly. Its value is at small p, which is
  untested.
- **The large-k re-read** is mostly done: `capability-defs` J2 gave 16k–65k base attempts to every hard theorem RL
  solves.

**Design choices:**
- **Papers 1 and 2 merge.** The knockout must run on many seeds because discovery is a lottery. The same seeds then give
  paper 2's discovery rate and the >3-seed headline counts.
- **Paper 3 becomes one section** of paper 1.
- **Interface is a later extension**, so "interface" stays out of the title unless it is measured on the 9.6 M organism.

## Plan and budget

| phase | what | ≈ cost |
|---|---|---|
| 0 | estimator at small p plus a Good-Turing unseen-mass estimate; the budget in measured base attempts; pre-registered thresholds; claims ledger and re-audit | ≤ $5 |
| 1 | **composition knockout:** pretraining without "DN applied to a ¬I line" (K12⁻, refilled to 155,000), ≥ 6 seeds × 16 EI rounds, against full exposure (3 existing seeds + ≥ 3 new); acquisition of the classical key-step families, compute-matched reach, certification | ≈ $90–100 |
| 2 | power-sampling sharpening null on both arms' bases, at matched tokens | ≈ $10–20 |
| 3 | the agents' choice: a small-dose arm (≈ 0.1–1 %), exploration on knockout bases, interface pilot, harder-target pilot | the rest |
| paper | `create-or-elicit-paper`: paper draft (LaTeX → PDF if possible), blog post, critique rounds, independent claim audit | ≤ $10 |

Both runs are on agent2, and every role runs at `--effort max`. The experiments have a 120 h hard stop. The writing run
is queued with a dependency on the experiments' librarian and has a 96 h hard stop.

**Briefs:**
- [paper/BRIEF_create-or-elicit.md](paper/BRIEF_create-or-elicit.md)
- [paper/BRIEF_create-or-elicit-paper.md](paper/BRIEF_create-or-elicit-paper.md)
