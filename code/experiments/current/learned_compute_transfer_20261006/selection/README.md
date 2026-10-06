---
written_on: 2026-10-06
written_by: agent:codex
status: running
---

# Frozen phase1 textbook transfer evaluation

This separately authorized follow-up reuses the closed sprint's checkpoints and inference settings without training or threshold fitting. The source release is Charles's October 2 release at git commit `5d0084add453212e6bea8055d4d28cec790f268c`. Git blobs were extracted without checking out or changing the dirty workspace. `artifacts/learned_compute_transfer_20261006/release_acquisition.json` records file hashes.

The primary input contains 187 ordinary theorems:87 candidate_v0,72 candidate_v1,14 hand-proved and 14 batch3. The 225 derived records are a separate sequent-only projection diagnostic. Their task instructions, given prefixes and structural scoring requirements are omitted, not fulfilled. Original task metadata is isolated in an audit-only file; references never enter model inputs. The release's 144 designated curried twins and 164 derived base clusters remain explicit. Never interpret 412 projected sequents as 412 independent native tasks.

Input `data/combined412/public.jsonl` preserves exact `THM...SEQ...PRF` prompts and stable source names/indices. Its hash is `969d72c5d861f91adcbf238577daf53ee129440010d4adcd2dea5cc99acf8e47`. The earlier ordinary 187 input remains preserved. The metadata-only overlap audit finds one ordinary and one derived projection match to known K12, zero to expert iteration, the actual 90436-record round 16 mixture, or historical 72. This audit excludes no rows and does not establish complete historical exposure; curry-aware classes differ from our recorded atom-renaming/premise-permutation audit.

Phase1 collection uses the original r16 generator, eight independent fresh-context attempts, temperature 0.8, max 256 sampled tokens/action, 48 actions, batch 256, seed 94006, BF16 model autocast and FP32 sampling logits/replay logsoftmax. It retains actual sampled PAD and final failing/truncated actions. The original frozen collector/renderers remain unchanged. The prospective decisions freeze binds their hashes and the original checkpoint. Public pool hashes are sealed before selector label access.

Twelve primary-order 77106 selector runs reuse nine trained models (pooled/factorized/BCE, three seeds each) and three original readout controls. Feature coefficients and heuristic evaluators also remain fixed. `phase1_inventory.json` contains exact argv/signatures. `phase1_methods_freeze.json` binds new launcher/aggregation source before labels; source snapshots are separate from the closed sprint. No method inherits or modifies a joint controller.

Reported pass1 averages validity across all eight independent attempts, with first-candidate accuracy retained separately. Oracle8 and fixed-eight selector accuracy use the same pools. Ordinary means weight 187 rows equally and use source-role-stratified paired theorem bootstrap 5000/seed 96006. Derived means weight 225 rows equally; bootstrap resamples 164 whole base clusters and recomputes the row-weighted ratio, without averaging base means equally. Three training seeds are averaged within theorem, and individual seed values are retained. These intervals condition on the saved models and generations. No native derived-task or fresh-generalization claim follows.

Root owns the combined report and reuses completed historical 72 results; allocation owns contextual cost/correctness-only and fixed-count transfer paths. Their FP32 generation differs from this BF16 independent-pool collection and is labeled separately.
