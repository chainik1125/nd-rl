---
written_on: 2026-10-06
written_by: agent:codex
status: preliminary_review
---

# New prefix boundaries cause immediate interface failures

## Independent findings

Subsequent discriminating controls change the mechanistic interpretation: a **three-token boundary-only prefix**, `[<st>, n1, EOS]`, yields only **10/192 valid** and 16 completed attempts. Equal-length unrelated proof history yields six valid and six completed attempts. These counts were independently recomputed from their raw private-label files. Thus the large collapse requires neither lengthy context nor previous proof content. Scope errors may reflect the unfamiliar marker sequence itself; prior-name overlap below is descriptive, not sufficient evidence that prior-proof copying causes the effect.

Replaying all raw continuations reproduces the context probe's completion counts with zero mismatches. On the same 48 development theorems and four fresh attempts each, zero history yields 180 valid and 183 completed attempts out of 192. Prefixing one previous proof yields five valid and seven completed attempts. These are measurements of the frozen, unadapted checkpoint under an interface intervention.

The failures predominantly concern scope rather than long successful reasoning that later deteriorates. With one-history input, **142 of 192** attempts fail with `unbound`, 23 with `binder mismatch`, and **76 fail on their first action**. Only one zero-history attempt fails with `unbound`. In 101 of the 142 one-history unbound failures, an unavailable referenced name appears in the previous proof. This count excludes defining-name slots. Name overlap is evidence consistent with scope confusion, but common name reuse prevents treating it as causal proof of copying.

The probe preserves the authoritative theorem and hypotheses: it concatenates the previous effective action tokens, then the original `prompt_ids` for a freshly initialized environment. Prior proof formulas are not added to that environment's premises. I found no evidence of replaced premises, leaked verdict strings, or silent context truncation in the inspected code. The context bound is checked before generation. Four first-action CPU checks show that preventing the current state from attending to the prefix recovers clean next-token logits to within `3.8e-6`, despite shifted positions. This rules out a pure position-offset difference for that small next-token check; it is not a complete generation reproduction.

One schema caution: context probe public candidates are sorted by candidate ID, while private labels retain completion order. Always join labels by `candidate_id`. Aggregate counts are unaffected; positional pairing would corrupt candidate-level analyses.

## Discriminating next controls

1. **Empty boundary only:** prepend the proof-block delimiters without proof content. This tests whether the unfamiliar wrapper itself disrupts the interface.
2. **Equal-length neutral filler:** keep positions and length, remove proof names/formulas. Compare task-relevant history with filler, without treating arbitrary filler as necessarily in-distribution.
3. **Generator attention masking:** retain the same serialized history for selector/stopping observations but mask it from proof-generation attention. This tests whether model access to old proofs causes generation degradation and offers a stable comparator.
4. **History-name intervention:** consistently relabel references within the previous proof while leaving the new state unchanged. Compare which unavailable names the model cites, rather than only accuracy.
5. **Format adaptation independent of stop labels:** if adapting the generation interface, balance history count/length independently of correctness and teach current-state actions without acceptance targets. Apply the same independently initialized adaptation procedure across cost/no-cost controls and preserve its separate provenance.

Joint RL at this initialization could learn to stop after one attempt because the continuation interface is broken. That would be an informative behavior but does not establish the proposed mechanism of learning an optimal stopping boundary in an otherwise competent history-conditioned prover. Gate stronger claims on recovery of continuation competence and paired accept-versus-continue branches before hard limits.

## Evidence

[Audit output](context_independent_audit.json) and [CPU replay script](audit_context.py); immutable probe and collector snapshots under `artifacts/learned_compute_allocation_20261006/context/base_dev48_v1/source_snapshot/`. The audit used local CPU inference and raw replay only, with no GPU launch or protected evaluation access.

Additional controls: `context/base_dev48_boundary/history_1/private.jsonl` and `context/base_dev48_unrelated/history_1/private.jsonl` under the same artifact root. The boundary-only condition includes 112 unbound-name and 40 premise-position failures.
