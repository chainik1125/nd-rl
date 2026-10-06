---
written_on: 2026-10-06
written_by: agent:codex
status: reviewed_with_provenance_followup
---

# Saved-path budget baseline audit

## Mathematics and controller work

The context threshold observes only the currently encoded candidate-history length. It neither looks at future candidate sizes nor reads correctness. It uses the shared selector only at termination with at least two candidates; one-candidate selection is deterministic. The learned stopper correctly needs a controller forward at every nonforced reached boundary. These conventions avoid charging fixed/count/context controls for unused neural stopping decisions.

The fitter maximizes native development expected reward among exact matched-budget mixtures. For one budget plus the weight-sum constraint, an optimum needs at most two fixed policies; enumerating pairs is sufficient. For two budgets plus the weight sum, an optimum needs at most three; the joint fitter enumerates these supports and rejects infeasible/nonnegative solutions. It reports infeasibility rather than silently relaxing the requested match.

An independent numerical implementation solved 200 random feasible four-policy/two-budget problems using the one-dimensional affine nullspace of the constraint matrix. The fitter's optimum reward differed by at most `3.3e-12`, and its largest budget residual was `5.1e-12`. This validates the joint optimization independently of the support-enumeration implementation. Synthetic inputs only were used.

Frozen mixtures linearly combine per-path correctness, attempts, reward, forced probability, and expected work. This is the correct expectation when the budget policy is randomized before seeing the theorem's generated outcomes. Development-frozen weights must stay fixed on test; realized test-budget mismatch is an outcome to report, not a reason to refit.

## Information and provenance review

The fitter explicitly requires development-only theorem metadata, matches its hash to the evaluated input, checks public/private path keys, and uses native checkpoint reward. I found no test fitting in this code path. The current test evaluator registration binds data, draws, checkpoint, generation implementation, and source identity.

One follow-up was sent to the allocation researcher: the baseline fitter and reanalyzer import the **current** `integrate` function but fingerprint the **saved** path configuration. If the evaluator changed since path collection, comparing only those saved signatures can claim an older inference implementation while applying newer integration/cost semantics. Bind the actual integrator/reanalyzer hashes to the fit and require them during reanalysis, or explicitly version an approved reanalysis protocol. Final-test baseline application should also register the frozen fit file and actual reanalysis implementation before reading private labels. Recording hashes after the fact is useful provenance but is not the same as validating the frozen implementation.

The metric called total forward tokens combines generator and controller work. It improves on attempt matching, but remains an operation proxy: tokens at different context lengths have different attention cost, and batched execution differs from summed sparse paths. Attention-cell and measured-work diagnostics should accompany the claim.

## Forced-four runtime validation

The independent audit of `causal_forced4_protocol_smoke_v4` covers four retained states, four branches, and 20 proof attempts. Every branch actually generated four futures, with forced continuation overriding earlier sampled acceptance opportunities. All 20 public transition transcripts were independently checked in Lean: 16 valid, zero disagreements.

Twenty controller decisions were reconstructed and reevaluated on CPU using the saved checkpoint, with maximum probability difference `1.19e-7`. There are zero mismatches in candidate identities, keyed seeds/uniforms, forced-continuation flags, terminal decisions, input lengths, forced-limit flags, or old/new mass arithmetic. This is a two-theorem development integration check, not a performance result. Hard-cap cases are covered by the coordinator's separate synthetic checks, not by this runtime sample.

## Evidence

- [Joint-budget numerical audit](budget_mixture_math_independent_audit.json) and [independent solver script](audit_budget_mixture_math.py).
- [Forced-four runtime audit](forced4_smoke_independent_audit.json), reproduced with `audit_whole_policy_smoke.py causal_forced4_protocol_smoke_v4`.
- Inspected `allocation/evaluate_live_path.py`, `fit_budget_baselines.py`, `reanalyze_budget_baselines.py`, and `evaluation_protocol.py`.
- No final-test outcome access or GPU launch was performed.
