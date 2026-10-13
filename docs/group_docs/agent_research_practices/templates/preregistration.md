# Pre-registration — run `<run-id>`

Committed <UTC>, before the first job. Brief: `<path>`.

## Question

<One paragraph.>

## Notation

<Define every symbol used below, e.g. p_θ(t) = single-attempt solve probability of model θ on theorem t.>

## Design

- **Models**, each labelled with its checkpoint, parameter count, format, from scratch or pretrained, and training data.
- **Arms, controls and seeds** (the same seeds in every arm).
- **Pools,** and why each requires what it tests. Mark the scoring-only sets.
- **Read-out settings:** k, temperature, length caps, batch, plain and/or guided.
- **What the compared arms are matched on:** attempts, tokens, steps, GPU time.
- **The judge:** Lean alone, unless stated and justified.

## Statistics

- **Primary quantity and test,** stated in advance, with the correction for multiple comparisons if there are
  several.
- **Minimum detectable difference** at this seed count, from the measured noise floor.
- **What counts as a difference;** what counts as "no effect".

## Expected results

Number each one (E1, E2, …) so the write-up can score it.

| id | quantity | expected | falsified if |
| --- | --- | --- | --- |
| E1 | | | |

## Budget and stop rule

<Dollars / GPU-hours; what ends the run early; what is dropped first if the budget runs short.>

## Addenda

<Dated additions made before running the work they cover.>
