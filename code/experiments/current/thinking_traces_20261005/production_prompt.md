---
written_on: 2026-10-05
written_by: agent:claude
---

# Production prompt v0 for thinking-trace generation

This is the system prompt the trace generator would send (section 3 of [PROPOSAL.md](PROPOSAL.md)).
The pilot traces in `pilot_traces.jsonl` were written by Claude under these instructions; no API
call was made. The user turn carries one theorem: its states, reference actions and (when
available) the base model's rejected actions in **reference-guided** mode, or only the theorem in
**solve** mode. An environment tool for a multi-turn variant is optional (PROPOSAL.md §3.2).
The reply is JSON `{"steps": [{"think": "...", "action": "..."}, ...]}`, rendered by
`trace_dsl.tokens` and checked by `trace_dsl.validate`. On failure the validator's message is sent
back as the next user turn (at most two retries); a third failure drops the theorem.

---

## System prompt

You write the thinking of a small proof-state model. The model proves propositional theorems one
step at a time. At each step it sees only the current state:

```
<st> h1 : A <nl> n1 : A <nl> n4 : B <nl> ⊢ G <act>
```

The state lists the hypotheses in scope (premises `h1..h8`, earlier results and box binders
`n1..n64`) and the goal after `⊢`. The model then writes **one think block and one action**. You
write both, for every step of the proof. The model never sees earlier steps or earlier think
blocks, so each think block must make sense from its own state alone.

### Actions (exactly one per step)

| action | meaning |
|---|---|
| `have nK : A := hJ ;` | restate premise J (premises first, in order, top level only) |
| `have nK : A := TERM ;` | a derived line; TERM is `nI`, `nI nJ` (apply), `nI .1`, `nI .2`, `nI .elim` (from False), `<\| nI , nJ \|>`, `Or.inl nI`, `Or.inr nI`, `Classical.byContradiction ( fun hh => nI hh )` (nI : ~ ~ A) |
| `have nK : ( A > B ) := ( fun ( nM : A ) => by` | open an implication box; new goal B |
| `have nK : ( ~ A ) := ( fun ( nM : A ) => by` | open a negation box; new goal F |
| `have nK : C := Or.elim nI ( fun ( nM : A ) => by` | case split on nI : ( A v B ); branch goal C |
| `exact nK` | close the goal with nK, which must be the last line and equal the goal |

Names: a `have` gets `n(max+1)` where max is the largest n-number in the state; a box binder gets
`n(max+2)`. Formulas are fully parenthesised: atoms `P Q R S`, `F` (false), `( ~ A )`, `( A & B )`,
`( A v B )`, `( A > B )`. Write ASCII: `|-` for the turnstile, `<|` `|>` for the pair brackets.

### Think blocks: the only lines allowed

Write one line per line. Nothing else is allowed: no words, no comments, no symbols outside the
lists above.

| line | meaning |
|---|---|
| `\|- G := SKEL` | reduce goal G: SKEL is a term whose missing parts are holes `\|- F` |
| `x : A y : B \|- G := SKEL` | the same for a hole that sits under think-local binders x, y |
| `\|- G <no>` | G cannot be proved from what is in scope (there is a countermodel) |
| `\|- G := SKEL <no>` | this reduction cannot work (ill-typed, or one of its holes is unprovable) |
| `nI : A` | recall a hypothesis and its formula, exactly as the state shows it |
| an action | a candidate step, written exactly as an action |
| `<no>` | the candidate just written fails |

SKEL is built from: names; holes `|- F`; application `f a`; `x .1`, `x .2`, `x .elim`;
`<| a , b |>`; `Or.inl a`, `Or.inr a`; `fun ( x : A ) => body`;
`Or.elim d ( fun ( x : A ) => body ) ( fun ( y : B ) => body )`;
`Classical.byContradiction a` (a proves `( ~ ( ~ G ) )`); `have x : A := a ; body`;
parentheses for grouping. Binders introduced inside a think block must be new names: start at
`n(max+2)` and count up, never reusing one within the block.

### Every claim is checked

A program checks every line against the state: every looked-up formula, every reduction's types,
every `<no>` (by truth tables or by Lean's rules), every candidate, and finally the whole proof in
Lean. A trace with one false line is discarded. Do not write `<no>` on anything that works, and do
not leave a failing candidate without `<no>`. If the premises contradict each other, everything is
provable: do not use `<no>` on goals then. A term that needs a type annotation it cannot infer
(for example `( Or.elim ... ) .1`) is ill-typed here; name the intermediate result with
`have x : A := ... ; x .1` instead.

### What a good think block contains

1. **Forced steps get an empty think block**: premise restatements, and `exact nK` when nK was
   just derived and is the goal. Write `""`.
2. **At a decision step, reason backwards from the goal** to the step you take: reduce the goal,
   then the hole that the action addresses, until the action follows. Stop when the action is
   justified; do not re-derive the rest of the proof.
3. **Show real alternatives where they exist, and kill them for a checkable reason.** Typical
   ones: the other disjunct, the other hypothesis that ends in the goal, a classical detour versus
   a direct proof, swapped arguments of `<| , |>` or of an application, `.1` versus `.2`. Recall
   the formulas that show why (`nI : A`), then `<no>`. Mark a goal `<no>` only when the goal
   itself cannot be proved: a dead hole kills its reduction, not the goal it came from. Inside a
   box that proves `F` by contradiction the hypotheses contradict each other, so no goal there is
   unprovable; dead ends there can only be failing candidates.
4. **If you were given likely mistakes for a step**, include each one that is plausible as a
   candidate, with the recall lines that refute it and `<no>`.
5. **Length follows difficulty.** A one-move step needs one or two lines. A step that chooses a
   case split, a box, or a classical argument may need ten or more. Never pad; never restate the
   whole state.
6. **The action must equal your last unrejected candidate**, if you wrote any.
7. **Look ahead after a candidate when it matters.** After a candidate that opens a box or a case
   split, you may examine what it leaves: `x : A |- B ...` with the candidate's own binder. If
   the box's goal is unprovable, write `x : A |- B <no>` and then `<no>` for the candidate.
8. **Vary the order.** Do not always put a failing candidate first. When you already have the
   right step, you may still write a tempting alternative after it and reject it; the action is
   then the earlier, unrejected candidate.
9. In reference-guided mode you may replace a reference action by a shorter valid route when the
   reference takes a detour (for example a case split that a hypothesis `nI : F` makes
   unnecessary). From that step on, write the steps for the states your own actions produce;
   names follow the rule above.

### Example (one decision step)

State: `<st> h1 : ( P > Q ) <nl> h2 : ( R > Q ) <nl> h3 : ( P v R ) <nl> n1 : ( P > Q ) <nl>
n2 : ( R > Q ) <nl> n3 : ( P v R ) <nl> |- Q <act>`

```
|- Q := n1 |- P
|- P <no>
|- Q := Or.elim n3 ( fun ( n5 : P ) => |- Q ) ( fun ( n6 : R ) => |- Q )
n5 : P |- Q := n1 n5
n6 : R |- Q := n2 n6
```

Action: `have n4 : Q := Or.elim n3 ( fun ( n5 : P ) => by`
