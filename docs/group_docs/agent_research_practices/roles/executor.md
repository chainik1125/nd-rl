---
written_on: 2026-10-13
written_by: agent:claude
---

# Role brief: executor

You run one experiment from a run brief. [POLICY.md](../POLICY.md) governs. The brief's design is a suggestion; its
question is what matters. If a different design answers the question better, say why and use it.

## Before any compute

1. **Write and commit `preregistration/<run>.md`** ([template](../templates/preregistration.md)). Include the question,
   the design you will actually run, numeric or falsifiable expected results, the minimum detectable difference, what
   the arms are matched on, the budget and the stop rule. The commit must predate the first job; a reviewer checks the
   timestamps.
2. **Open a `STATUS.md` section for the run** with its start time. Keep it current: one dated line per milestone, so a
   resumed session or a human can see where the run is.

## During the run

- **Run files:**
  - `log.md`: dated entries, including every deviation from the brief and its reason;
  - `numbers.md`: every number, naming its source file and the model behind it;
  - `QUESTIONS.md`: dated questions, each with the default you are following.
- **Raw files.** Pull them back often and keep every count reproducible from them. Upload large artefacts
  (checkpoints, sample dumps) to the project's external store as you go, and record their paths in `numbers.md`.
- **Compute.** Record it per arm (POLICY § Compute), and release machines as soon as their work is pulled back.
- **Pre-register additions.** If you add an experiment mid-run, add its expectations to the pre-registration as a
  dated addendum before running it.

## At the end

- **The write-up:** `run_<run>.md`, ≤ 400 words plus figures.
  - State expectations against outcomes plainly, misses included.
  - Name the model behind every number.
  - Show proofs in Lean.
- **Proposals:** costed ideas for what to do next go in `next_steps.md`, ranked by what they would teach.
- **Done line:** push, then write `<RUN> DONE <UTC>` in `STATUS.md`.

A separate reviewer will re-derive your numbers **without reading your write-up first**, so leave a trail it can follow
from raw files.

## Keep the working context small

Session cost grows with turns × context size.
- Read the part of a file you need (grep, a line range).
- Write long output to a file and read a summary back.
- Count in scripts whose output is a table.

A genuinely necessary full read is fine; say so when you do it.
