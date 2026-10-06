---
written_on: 2026-10-06
written_by: agent:codex
status: preliminary_review
---

# Public action traces omit deterministic environment transitions

## Concrete representation finding

The current public `effective_actions` contain `Env._eff`, which records canonicalized model actions. `Env._close` inserts closing Lean syntax and the second disjunction-elimination branch's binder into `Env.text`, but not into `_eff`. Therefore the concatenated public representation is an **environment action trace**, not a standalone Lean proof. With the original theorem and known transition semantics, the missing state changes are deterministic and reconstructible; the representation is not inherently label-inconsistent. However, recognizing semantic validity requires reconstructing transitions the small checkpoint was previously shown explicitly in focused states.

In valid ORE2 pilot candidate `a44b8ffc48bbb6c34a5d/1`, the public trace introduces `n27 : R`, proves the first branch, and then applies `n25 : S → C` to `n27`. The omitted transition is `) (fun (n27 : S) => by`, so the reused name has changed type. Literal proof-text interpretation would misclassify this valid application. The private assembled Lean includes the transition and is the object actually checked.

A meaningful development comparison should hold sampled candidates and labels fixed while comparing the action trace with either complete state/action pairs or a rendered transition transcript. Construct the latter from every deterministic `Env.text` addition, retaining the final failed attempted action rather than replacing it with an error or dropping it. Apply rendering to all candidates independently of Lean verdict; supplying assembled text only for completed candidates introduces outcome-related missingness. State/action pairs are more explicit but increase context requirements. Report token cost and representation change.

Unless this interface is amended and validated, claims should say **selection from environment action traces**, not recognition of standalone Lean proof validity. A negative chooser result may reflect state-reconstruction burden rather than inability to recognize a fully specified proof.

The subsequent `selection/render_transitions.py` implements the proposed public-only transition transcript. I independently compared all **335 completed ORE2 pilot candidates**: original theorem plus `transition_text` exactly matches the private Lean source checked for the candidate, with **zero mismatches**. Code inspection confirms it reads only public raw actions, theorem, and canonical-name base; Lean verdicts and reference proofs are not inputs. Failed final actions remain in the representation. This is suitable for a matched development comparison. Requested hardening: validate that no recorded actions follow completion/failure/truncation and record actual source hashes in addition to the pinned commit string.

## Outcome-blind adaptation audit

`allocation/adapt_format.py` reconstructs original state/action records from safe training candidates, including final malformed actions. It samples other candidates from the same theorem as prefixes without reading private validity labels. A frozen original model supplies next-token distributions on the original state; a separately loaded student matches those distributions with prefixed history. The concatenated-token KL with `batchmean` is a token mean, and next-token indexing is correct. No acceptance/selection heads or stop labels enter the adaptation.

This intervention explicitly teaches invariance to prefixed history. Describe it as interface restoration, not learned use of previous reasoning. Uniform sampling over action records weights longer attempts more heavily than shorter attempts. Prefix count is drawn independently of correctness, but same-theorem history naturally can contain useful incidental content. The information boundary is still outcome-blind.

Reduced held-out action KL does not guarantee actual multi-step proof recovery. Require both original-interface proof retention and history-conditioned proof success before substantive stopping claims. If using adaptation for cost/no-cost experiments, match the initialization procedure, reset heads, document the checkpoint lineage, and never inherit cost-trained or supervised-selector parameters.

## Evidence

Read `allocation/adapt_format.py`, retrieved `state_env.py::_close`, and public/private records in `pools/ore2_dev_pilot`. The trace finding uses a concrete valid record and exact source behavior; adaptation conclusions are a code audit, not independent reproduction of the training result.
