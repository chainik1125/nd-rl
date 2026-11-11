---
written_on: 2026-11-10
written_by: agent:claude (drafted by the build agent in TAGS.md; frozen by the orchestrator)
---

# Pre-registration: the route taught only under tag A, RL under tag B

Frozen at 8:00 pm PT, 2026-11-10, after the CPU build and its smokes (toy models only) and before any full-size base
exists. Leon asked for the experiment ("Teach the route only under author tag A, run RL under tag B
is also interesting and worth running", 2026-11-10) but has not reviewed this text; the predictions
are the build agent's priors. The base reads (the gate) run under this version. The RL arms do not
start until Leon has seen the gate result; any change he makes before then is recorded below as an
amendment with its time.

Design, tag format, corpus and launch commands: TAGS.md in this folder.


**Hypotheses.**

- **H-fast ("like 1 %").** Under B the route's capability is shared with A. RL on route-needing problems reaches it at
  the 1 % lines' cost.
- **H-slow ("like 0 %").** The capability is tied to tag A. Under B, RL behaves like the 0 % base.

**Primary outcome, per line.**

- **The measure.** The first crossing of Dan's bar, defined as follows:
  - **Draws.** The cumulative strict-106 count over matched k 32 draws, read under the line's own tag, reaching 14.
  - **Grid.** The draws follow `la_dose1_s<seed>`, whose x per round for seed 0 is 0.10, 0.39, 0.70, 1.03, 1.38 and
    1.74. Seeds 1 and 2 are within 0.05 of these.
  - **Units.** x is in units of the tagged base's own D.
- **The classes.**

| class | definition |
|---|---|
| fast | crosses at matched round ≤ 3, x ≤ 0.70–0.73: no later than one round after the 1 % lines, which all crossed at round 2 |
| intermediate | crosses at round 4 or 5, x 1.03–1.43 |
| slow | no crossing by x_dan 1.5. No 0 % or 0.1 % GRPO line crossed by x 4.4–5.9. |
| at the bar before RL | the base read under that tag (round 1) is already ≥ 14. It is reported on its first-32 and E32 trajectory instead. |

- **Arm verdict.** Two of three seeds in one class decide an arm. Otherwise the arm is "mixed".

**Control.** A under tag A is the matched counterpart. It is expected to be fast, since A's records hold the route at
2 % against the 1 % corpus's 1 %.

**Base reads, before RL.**

- **Manipulation check (primary).** It uses `tag_check.py` on `sigma_unseen` and `ctrl_unseen`:
  - Δ_open is the mean log P of the route-opening actions under A minus under B.
  - Δ_ctrl is the same difference over all actions of `ctrl_unseen`.
  - A base uses the tag if Δ_open − Δ_ctrl ≥ log 2, meaning the opening is at least twice as likely under A, beyond
    any general shift.
- **Propensity under each tag.** Measured on the 106 and the 240:
  - "uses reductio anywhere" (U/reductio `used`);
  - "opens with it" (U `first`);
  - the exact first-move probability (E/reductio).
- **Capability under each tag.** F/reductio: solved when the opening is forced, on the 106 and on the 240 separately.
  The forced opening on optional theorems is partly direct proof.
- **Differences.** A − B for each, with 95 % bootstrap intervals over theorems.

**What each outcome would mean.**

| base check | A arm | B arm | reading |
|---|---|---|---|
| fails (the base ignores the tag) | any | any | uninterpretable for the question. B is a 1 % base with an inert token; reported, no claim. |
| passes | fast | fast | the capability is shared across tags, and RL under B reaches it at the 1 % cost. Whatever the tag gates, RL undoes it cheaply. |
| passes | fast | slow | the route is tied to its tag for RL: from B, RL behaves like the 0 % base |
| passes | fast | intermediate | partial transfer, sized by the crossing round |
| passes | not fast | any | the tagged base is not a 1 %-like base. B is read against A only, not against Dan's 1 % and 0 % lines. |

The base reads split the B result further.

- **Propensity gated, capability shared.** Forced-reductio success under B is about the same as under A (the
  interval contains 0), while propensity under B is far below A.
  - B fast then reads as "RL elicits a shared capability from a near-zero propensity".
  - B slow reads as "RL needs propensity, not only capability". That is an elicitation barrier, and the most
    interesting outcome.
- **Capability gated.** Forced success under B is far below A, so B slow is expected. B fast would mean RL transferred
  or rebuilt the capability under B.

**What would make the result uninterpretable, or weaken it.**

- **The base ignores the tag.** The manipulation check fails on 2 or more seeds.
- **Tag leakage.** Any `TagError` in a line, or a GRPO segment whose `author_tag.json` shows states under another tag,
  invalidates that segment, which is rerun.
- **The A arm is not fast.** See the last row of the table.
- **The guard.** In the knockout, every line that did not cross had its lr cut to 1/8 early. A slow B line whose lr
  reached 1/8 before x 0.5 may be slow because of the guard. Each line's guard timeline is reported (as in the
  knockout's `guard.py`). This is a caveat, not an exclusion. The guard-off pilot (Amendment 2) did not implicate the
  guard, on one seed of 0.1 %.
- **Read noise.** The reads on RTX 3090s are fresh draws from the same distribution as Dan's A40 draws. Round-1 counts
  on his bases ranged from 2 to 19.

**Secondary measures.**

- The σ share of the Lean-accepted training proofs, by Dan's `coe_sigma.flags` on `LEAN_GATE_DUMP`: the 1 % lines
  reached 2–3.6 % over their last 20 iterations, and the 0 % lines stayed at 0.05–0.12 %.
- The snapshot first-32 and E32 curves.
- pass@1 on the pool.
- The untagged read of each base: does no tag behave like A, like B, or like neither?

**Not pre-registered, possible later.** Reads of the B lines' snapshots under A and under no tag. They would show
whether RL under B moves A, and what the untagged default becomes.

**Predictions (the build agent's priors).**

| | prediction | probability |
|---|---|---|
| P1 | the manipulation check passes on at least 2 of 3 bases | 75 % |
| P2 | the A arm is fast on at least 2 of 3 seeds | 75 % |
| P3 | the B arm is fast on at least 2 of 3 seeds | 40 % |
| P4 | the B arm is slow on at least 2 of 3 seeds | 35 % |
| P5 | at base, forced-reductio success under B is within 5 points of A on the 106 (capability shared) | 55 % |

