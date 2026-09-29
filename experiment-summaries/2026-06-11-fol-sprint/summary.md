# Into Predicate Logic: a Sound FOL Natural-Deduction Testbed (+ first results)

*Sprint 3 (2026-06-11), extending the propositional ND testbed to first-order
logic. Branch `sprint3`. Code in `code/fol/`.*

## Executive summary

The propositional sprints raised the question of whether their findings — the
"two walls" of length generalization, the depth ceiling — survive the move to
**predicate logic**, where the genuinely hard machinery lives: quantifiers and
the **eigenvariable side conditions** the original proposal flagged as where a
transformer is "predicted to fail." This sprint builds the testbed and runs the
first experiment.

**1. A sound, independently-verified first-order ND testbed.** Relational FOL
over {∀,∃,~,&,v,>,F} with predicates of arity 1–2, variables, and constants
(doubling as eigen-parameters); the ten propositional rules plus ∀I/∀E/∃I/∃E in
Lemmon/Halbach eigenvariable form (matching the nd_pack textbook). The verifier
is implemented independently of the generator and enforces the eigenvariable
conditions (∀I's parameter must be fresh in the assumptions it depends on; ∃E's
parameter must not escape into the conclusion or outer context). Cross-validated:
**2867/2904 sampler proofs verify** (the 37 rejects are the verifier correctly
catching the sampler's naive eigenvariable violations — exactly its job),
**498/500 single-token mutations rejected**, and hand-built eigenvariable
violations (param-in-premise ∀I, param-leak ∃E) are all correctly rejected.

**2. The constant token wall recurs in predicate logic, exactly as the theory
predicted — and the same fix removes it.** Sprint-2's theory agent predicted the
Sprint-1 line-index token wall would recur in FOL via "bounded parameter
inventories." Minimal-pair test: train a 3.3M model on proofs whose constants
are confined to {a,b}; evaluate on the *same* 600 held-out theorems relabeled to
in-vocabulary {a,b} vs out-of-vocabulary {d,e} (pure renaming, identical proof
structure). Result (Fig 1):

| model | in-vocab {a,b} | out-of-vocab {d,e} |
|---|---|---|
| bounded (trained on {a,b}) | **0.953** | **0.130** |
| full (trained over {a,b,c,d,e}, relabeled) | 0.955 | 0.960 |

The bounded model collapses from 95% to 13% purely because the held-out
theorems use constant *tokens* it never trained on — the predicate-logic analog
of Sprint-1's N9 line-index wall, confirming the theory's prediction. Training
over the full inventory (the relabeling = Sprint-1's offset fix) removes the
wall at **zero in-distribution cost** (0.955 vs 0.953). (The residual 13% are
theorems the model can route around without emitting the novel constant in a
novel position.)

**3. The eigenvariable side condition is a real but subtle wall: not a solve-rate
cliff, but a distinctive freshness-violation error signature.** The ∀I/∃E
conditions require tracking which parameters are fresh across the open-assumption
stack — the proposal's predicted failure mode. A full-inventory model, eval'd on
held-out theorems split by whether the proof needs an eigenvariable rule
(Fig 2): solve rates are close (eigenvariable 0.94 vs not 0.96; both follow the
same length cliff, eigenvariable a few points lower at lengths 5–7) — so
eigenvariable proofs are **not categorically unreachable**. But the *error mode*
is sharply distinctive: **~23% of the model's failed attempts on eigenvariable
targets are freshness violations** the verifier catches (`allI-eigenvar`/
`exE-eigenvar`), versus **~2%** on non-eigenvariable targets. The model applies
∀I/∃E confidently but unreliably respects the side condition — direct evidence
it does not perfectly hold the parameter-freshness state, exactly the predicted
difficulty. The sound verifier is what makes this both measurable and safe: every
violation is caught, so none enters the verified training pool.

## 1. The testbed (`code/fol/`)

- `core.py`: terms (`('v',x)` variables, `('c',a)` constants), formulas with
  `('all',x,A)`/`('ex',x,A)`, capture-avoiding substitution, instance-matching
  (`match_instance`), a bottom-up proof sampler, and Fitch linearization.
- `verify.py`: the independent verifier. Parses the token format; checks all
  rules; enforces eigenvariable conditions via an `active_consts(line)` =
  constants in premises + the assumption-hypotheses of all boxes open at that
  line. ∀I checks its parameter ∉ active_consts(cited) ∪ consts(conclusion);
  ∃E checks its parameter ∉ consts(∃-formula) ∪ consts(conclusion) ∪
  active_consts(∃E-line).
- `tokenizer.py`: 100-token vocab (preds P,Q,R,S; vars x,y,z,w; consts a–e;
  ∀=A, ∃=E; the 19 rules; line indices N1–N64).
- Soundness evidence: see Executive summary §1.

Token format (∃E example, the deepest rule):
```text
THM ( E x P ( x ) ) , ( A x ( P ( x ) > Q ( x ) ) ) SEQ ( E x Q ( x ) ) PRF
N1 ( E x P ( x ) )                : PR ;
N2 ( A x ( P ( x ) > Q ( x ) ) )  : PR ;
N3 | P ( a )                      : AS ;           <- flag parameter a
N4 | ( P ( a ) > Q ( a ) )        : ALLE N2 ;
N5 | Q ( a )                      : IMPE N4 N3 ;
N6 | ( E x Q ( x ) )              : EXI N5 ;
N7 ( E x Q ( x ) )                : EXE N1 N3 N6 ;  <- a must not escape
QED
```

## 2. Token-wall experiment (Fig 1)

![Fig 1](figures/fig_fol_tokenwall.png)

**Fig 1 — the constant token wall.** Same 600 held-out FOL theorems, constants
relabeled to in-vocab {a,b} vs out-of-vocab {d,e}. The bounded-trained model
(blue) is the only bar that drops (0.95 → 0.13): a pure renaming to unseen
constant tokens breaks it. Full-inventory training (orange) is flat at ~0.95.

Setup: 137k FOL proofs (1–2 distinct constants, 2–12 lines, verifier-checked),
two 3.3M / 4-layer models trained 6k steps; eval pass@8 (T=0.8), 600 targets,
verifier-judged. `code/fol/runbook_tokenwall.sh` reproduces it. This is the
clean predicate-logic replication of Sprint-1's two-walls result: an output-
format/vocabulary wall sits in front of any reasoning limit, and a relabeling
that exposes the full token inventory in training removes it for free.

## 3. Eigenvariable structural wall (Fig 2)

![Fig 2](figures/fig_eigenvar.png)

**Fig 2 — the eigenvariable as error signature, not solve cliff.** Left: both
groups follow the same FOL length cliff; eigenvariable-requiring proofs (red)
sit only a few points below at lengths 5–7. Right: among *failed* attempts,
freshness violations are ~23% for eigenvariable targets vs ~2% otherwise — a
~12× enrichment. Full-inventory 3.3M model, pass@8, 600 targets/group, lengths
length-comparable; `code/fol/forensic.py` + `runbook_struct.sh` reproduce it.

The reading: a small from-scratch transformer *can* learn ∀I/∃E well enough to
solve most short eigenvariable proofs, but its grasp of the freshness side
condition is imperfect — when it errs on these targets, roughly a quarter of the
time it errs *specifically* by mis-tracking which parameter is fresh. This is the
quantifier analog of Sprint-1's discharging-rule difficulty (subproof-state
tracking), localized to the eigenvariable bookkeeping the proposal flagged. The
verifier's soundness on these conditions (§1) is essential: it converts a subtle
model weakness into a caught, filtered error rather than an unsound proof.

## 4. Limitations / status

- The sampler richly exercises ∀I/∀E/∃I (744/269/1023 of verified proofs) but
  under-samples ∃E (it needs the parameter to be eliminable from the
  conclusion); ∃E is fully verifier-supported and appears in hand/template
  proofs. A template ∃E generator is the next addition.
- Eigenvariable freshness is a *global* condition; the bottom-up sampler can
  violate it under composition (caught by the verifier, filtered out).
- This is a fresh testbed; experiments 2–3 are first-pass, single-seed.
