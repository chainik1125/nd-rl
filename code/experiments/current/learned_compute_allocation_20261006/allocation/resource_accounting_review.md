---
written_on: 2026-10-06
written_by: agent:codex
status: bounded_read_only_audit_complete
---

# Resource accounting: scopes and overlap

The five requested ledgers describe different timing scopes. Their run identities
are disjoint after distinguishing output directories, rather than matching repeated
model/checkpoint names. Root's eight development causal runs are not the eight
final registered causal runs: they have different output/data identities. The
allocation development ledger contains only 26 development jobs; its later final
ledger would overlap the 18 final live evaluations and must not be added again.
Merged candidate pools and parent queue intervals are not additional model jobs.
The two cached-confidence diagnostics are absent from the older selection ledger.

All 102 final launch-ledger byte hashes match the recorded bindings. There are
2 independent pool collectors, 18 live evaluations, 8 causal interventions, and
74 selector inferences, with no missing final process timers. This completeness
applies to that declared final inventory, not the entire sprint.

| Inventory | Recorded interval hours, rounded | Scope and exclusions |
|---|---:|---|
| Root development: 41 rows, 28 timed | 3.44 | Mix of update sums, model/replay sections, and some whole children; 13 untimed jobs |
| Selection development: 33 timed rows | 1.89 | Collection/epoch/extraction/evaluation sections; some wall-clock rather than monotonic timers; five additional untimed jobs |
| Allocation development: 26 rows, 23 timed | 4.09 | Update or collection/policy section sums; three untimed debug jobs |
| Cached-confidence diagnostics: two rows | 0.035 | Main-function intervals; import/process startup excluded |
| Registered final model processes: 102 rows | 12.31 | Whole children including setup, CPU checking and serialization; CPU-only later analysis and queue waits excluded |

These hours are elapsed intervals around GPU-assigned work, not GPU kernel-active
hours, FLOPs, isolated latency, or a measured exact total allocation. Null historical
GPU metadata remains unknown. CPU checker/teacher/replay subintervals can already
sit inside a parent interval; do not add them again. The cached sampler/callback
114.02 seconds and replay 2.47 seconds are nested parts of the 126.99-second
cached diagnostic inventory, not extra jobs. Selection's separately recorded
1.55-second CPU scalar-head fit does not belong in a GPU-time sum.

The final process inventory decomposes into 0.30 hours for independent collection,
10.72 for live evaluation, 1.05 for causal intervention, and 0.23 for selector
inference. Some runs shared devices with foreign users. Summing disjoint job
intervals does not reconstruct isolated device occupancy or shared-device
utilization. Startup omitted from earlier section timers, failed/aborted jobs,
and untimed probes prevent an exact whole-sprint total. Root, selection and
allocation report 13, five and three untimed entries respectively; these are
ledger-specific inventories, not an assertion that every omitted job is known.

Recommended report wording: “The registered final evaluation inventory comprises
102 model child processes and records about 12.31 hours of summed process wall
time across assigned devices. Earlier development ledgers record approximately
3.44, 1.89 and 4.09 hours of heterogeneous timed sections, plus two separately
metered cached-confidence diagnostics (127 seconds). Several earlier jobs lack
timers. These measurements include varying amounts of CPU/setup/checker work,
exclude CPU-only later analyses and queue waits, and are not an exact GPU-hours
or isolated-latency total.” Keep the scope table rather than headline a precise
sum. Recorded additional paid GPU spend is zero; agent billing is unavailable,
not claimed to be zero.

## Immutable inputs inspected

| Ledger relative to sprint artifacts | SHA256 |
|---|---|
| `root_resource_ledger.json` | `90cabd4e0da21a043dbd6e9ecd94b3f53c37777696a334a139fd503a9c02f4ee` |
| `selectors/resource_ledger/ledger.json` | `8e70ec8acea1817cc0ae1b649507ee03a5df26497060bf4cbcabb88256aabea0` |
| `allocation/resource_ledger_development.json` | `809f6b2f149673fe72e278185b90d0c108294a69a49edc663a7ff1d7c15c7f7d` |
| `selectors/cached_confidence_resources_v1.json` | `b41e3df30028bb2887f128e86720e7451e577c1120f322064d1ac0034eca5ee1` |
| `final_resource_ledger_v1.json` | `b1be5cc6f989a8bbf2027b607b16f7d7ae0676722aed405d08ee4f81fcdaf58a` |
