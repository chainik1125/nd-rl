---
written_on: 2026-09-08
written_by: agent:codex
---

# Strategy for a trained reasoning workspace

## Recommendation

Run a matched comparison of direct-proof RL and a model-trained symbolic scratchpad followed by a final proof, using the same permitted theorem populations, starting lineage, final verifier, and inference budgets. Give the scratchpad a short supervised cold start from permitted proof labels before RL. Use sparse binary final-proof success initially; do not pay for response length, arbitrary intermediate formulas, or apparent reflection. Keep an unchanged-parent resampling curve and a scratchpad-SFT endpoint. These controls distinguish useful sequential computation from extra sampling, extra supervision, and format adaptation.

The main risk is sampling support. Earlier hard-composition probes produced no successes in 29,376 responses in three difficult families. Adding unrestricted scratch tokens enlarges the action space. Consequently, a zero-success scratchpad pilot should trigger a smaller, better-supported curriculum or a different cold start, not a multi-hour continuation without signal.

This is an R1-inspired experiment, not a replication of DeepSeek-R1. The inherited project models are 8M/85M specialized decoders, trained in a formal token language. They do not have the broad pretrained reasoning repertoire of R1's starting model. A structured workspace is a practical approximation that can be trained and tested in this sprint.

## Current evidence and constraints

The authoritative recent context is on `simplex1` under `/data/users/dmitry/nd-rl/take-home/`. I read aggregate sections of `current_state_20260909/report.md`, `transfer_sprint_20260907/methods.md`, and the latter sprint's `theory/rl_protocol_critique_v2_after_support.md` and `theory/rl_choices_v1.md`. I did not open textbook problem files, reference generators, or individual textbook predictions. The local `earlier-iterations/Natural Deduction LLM/nd_sprint` reports are older context; initial recommendations based on those were superseded after locating the current work.

The recent model uses a 40-token relative-unary codec, so it already avoids the historical absolute line-index token wall. Retain that codec and its strict backward-reference check. The original verifier alone is insufficient because earlier work discovered acceptance of circular or future references.

The recent aggregate evidence establishes an important baseline: four selected endpoints each solved **1/33** Smith textbook problems, including **0/30** with reference length at least eight, under one greedy attempt with a 1,536-token response allowance. Earlier synthetic serial-conjunction success extended through 26 lines, but mixed-rule and case-analysis frontiers remained much shorter. These figures describe earlier completed evaluations, not results from this sprint.

Inherited successful parents have seen verified long-proof discovery banks, including a 13,148-class bank. Their six-line cap applies to generator-supplied labels; it does not mean their current training history contains only six-line proofs. A fresh cap-six arm and an inherited-parent intervention answer different questions and must have separate legends and conclusions.

The old 112-problem textbook population was already used as development data, and Smith33 was evaluated in the previous sprint. Therefore a new final-only partition from these collections is **unseen during this sprint's model selection**, not a historically untouched benchmark. Split whole canonical theorem classes, including atom-renaming and premise-permutation equivalents, before exposing the permitted training slice. Keep at most 10% in training, 40% in end-of-run development, and 50% in final-only evaluation. No new decomposition, curriculum, retained bank, or hindsight proof may introduce the other 90% through an alternate representation. All development comparisons must finish their RL run before accessing textbook development results.

## What the literature actually supports

**R1 supplies the general recipe.** The original report uses grouped policy optimization, rule-based final correctness rewards, and a separate reasoning region before the answer. R1-Zero starts from a pretrained base; R1 adds a small supervised reasoning cold start and later training stages. The report describes longer reasoning and reflection, but response length by itself is not a correctness measure. For this task the useful transferable intervention is an additional trainable computation channel whose credit comes from the checked final derivation. [DeepSeek-R1, original report](https://arxiv.org/html/2501.12948v1).

**Formal-prover cold starts favor decomposition.** DeepSeek-Prover-V2 constructs initialization data by decomposing theorems into subgoals, solving them, and combining the results into reasoning traces before RL. That supports a proof-derived subgoal scratchpad here. It does not establish that an 8M model will learn the same behavior or that teacher-supplied longer proofs preserve our fixed initial-cap question. Use only allowed short labels and verified model discoveries for the main arm. [DeepSeek-Prover-V2](https://arxiv.org/abs/2504.21801) [@ren2025deepseekproverv2advancingformalmathematical].

**Search is a serious comparator.** DeepSeek-Prover-V1.5 combines proof-assistant feedback with search, truncating failed candidates at their first unusable tactic and reusing verified states. Its RMaxTS exploration rewards newly reached search states. This motivates comparing scratchpad learning against existing prefix search, while leaving intrinsic-state rewards as a separate ablation. Our existing strict checker and prefix search make this comparator cheaper than implementing a new theorem-search engine. [DeepSeek-Prover-V1.5](https://arxiv.org/html/2408.08152v1).

**Length normalization can produce a misleading result.** The Dr. GRPO analysis identifies response-length and reward-standard-deviation normalization biases and proposes removing those normalizers. Its reported benefit is especially relevant because growing length is our scientific target. Prefer a fixed divisor for policy loss and an unnormalized group baseline or leave-one-out advantage; record correct and incorrect length distributions separately. Increased scratch length accompanied by flat held-out proving is not evidence of expanded reasoning. [Understanding R1-Zero-Like Training](https://arxiv.org/html/2503.20783v1).

**Sequential and parallel inference compute differ.** Snell et al. study adaptive compute and find that effective allocations depend on difficulty and the inference mechanism. This motivates fixed total-token curves with several sequential/parallel allocations, rather than comparing equal numbers of completions with very different lengths. Their large-model results are motivation, not a quantitative prediction for our formal decoder. [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/html/2408.03314v1).

**Low-k gains are insufficient to establish new capability.** Yue et al. report cases where RL improves ordinary accuracy while reducing large-k coverage relative to the base model. This is empirical evidence about studied models and recipes, not an impossibility theorem. Here we can test the issue directly with the same theorem set, exact proof checker, matched sampling budgets, and fresh final-only evaluation. [Does RL Really Incentivize Reasoning Capacity Beyond the Base Model?](https://arxiv.org/abs/2504.13837).

## Concrete experiment priorities

| Priority | Intervention | Essential comparison | Advancement signal |
|---|---|---|---|
| 1 | Symbolic plan cold start on allowed labels; then final-success RL | Same-parent direct-proof RL and retained-data control | Stable checked success and useful mixed-reward groups on synthetic training |
| 2 | Frozen inference-budget sweep | Direct best-of-N, scratchpad best-of-N, existing prefix search | More held-out problems solved at the same total generated-token budget |
| 3 | One repeated seed or objective ablation | Same format and target schedule | Direction survives a second seed or a small objective change |
| 4 | Sequential revision with failure feedback | Equal-token independent retries | Improvement caused by using prior failure, beyond extra chances |
| 5 | Longer initial label cap, e.g. eight | Same representation and short-only lineage | Establishes whether starting support is the binding limit |

Use the two initially allocated H100s, GPUs 3 and 4, for independent arms rather than distributed training of these small models. Root owns allocation and monitors the reserve: GPUs 0 and 2 initially remain completely untouched. No GPU work was allocated to this strategy agent. Expand only when the resource manager confirms additional idle capacity while preserving two free devices.

### Minimal scratchpad

Serialize `prompt → THINK → symbolic work → FINAL → proof → QED`. The checker receives the final proof of the original prompt, never a theorem copied from the scratchpad. The scratchpad can contain a backward list of target formulas and likely introduction/elimination operations, with delimiters for subgoal and branch boundaries. Construct this from the dependency structure of allowed proof labels; copying the entire proof as a scratchpad is a weaker baseline because it trains duplicate output rather than a planning abstraction.

For the first implementation, prefer a bounded scratch region of 64–256 tokens and preserve at least the direct arm's final-proof allowance. Increase the scratch cap only after learning is stable. Train on both planning and final-proof tokens. A cold-start trace may contain information recoverable from its training proof; that is allowed supervised learning, but its full token budget must be reported. Do not insert gold plans at evaluation. Audit final extraction against missing/multiple delimiters and theorem substitution.

Start all comparison arms with the same final proof format and same shared model weights where feasible. If vocabulary expansion requires new embeddings, initialize the same expanded vocabulary in the direct arm. Record adaptation tokens separately. The inherited and fresh-cap-six comparisons should not be silently pooled.

### Policy update

Use 8–16 completions per training theorem and a modest batch of distinct theorems. The existing fresh on-policy leave-one-out update is a reasonable robust starting point: one optimizer step per fresh sample batch, advantage `r_i − mean(other rewards)`, all actual generated actions included, and a fixed loss divisor. If using clipped GRPO instead, store actual behavior log-probabilities and match sampling temperature/distribution in the loss. Keep EOS and failed/truncated trajectories explicit. Do not discard a long trajectory solely because it exceeded a legacy 512-token scoring limit.

Log mixed-reward groups, distinct successful canonical targets, final-proof acceptance, scratch/final lengths, truncation, policy loss, retention loss, and gradient norm. A group with all-zero rewards contributes no ordinary group-relative policy signal. Use a synthetic near-frontier curriculum that yields real successes while keeping hard targets in the mix. Any change to rewards, curriculum, or cap starts a new declared run; do not choose it by examining final-only data.

A failed-attempt revision trace is a useful second intervention: `attempt → strict error category or last verified state → revised attempt → final proof`. Its cold start can use corruptions of permitted short training proofs. Mark this as synthetic repair supervision. Only final correctness earns the main reward; verifier error messages are observations. Avoid an external solver that effectively supplies the answer under the name of scratchpad feedback.

## Numbers and plots to preserve

The summary's headline must give exact held-out textbook counts and denominators, including the beyond-initial-cap subset. Show a plot of textbook success rate against **reference proof length** with every attempted theorem retained, beside a second plot of success against total inference tokens. Include direct SFT, direct RL, scratchpad SFT, and scratchpad RL where completed. Plot only actual evaluated checkpoints; do not silently omit failed arms.

For each condition retain: greedy accuracy; verifier-selected pass@k; total generated tokens; scratch tokens; final-proof tokens; raw accepted line count; shortest accepted witness found per theorem; and fixed-compressor length if the compressor was frozen in advance. A reference or compressed witness is an upper bound on minimal proof length. A long redundant certificate does not prove a harder theorem was reached.

The strongest desired finding is more previously unsolved final-only textbook classes, especially those with reference length beyond the initial cap, at a matched compute budget. Secondary evidence is longer shortest discovered witnesses with healthy short-task retention. Report maximum length only alongside a multi-theorem frontier and exact counts, because a single padded proof is an unstable metric. If textbook counts remain flat while scratch length grows, state that the experiment implemented inference-time reasoning but did not demonstrate textbook generalization.

## Implemented analysis and second cold start

[`analyze.py`](../../code/experiments/current/r1_cot/analyze.py) now replays complete, explicitly released evaluation files. It checks hashes and role before opening labels, requires full per-target attempt coverage, reconstructs textbook final bodies from saved response IDs, and strictly rechecks both raw and compressed proofs. Reports include first-k and cumulative-token-budget prefix curves, paired target-bootstrap intervals, shortest witnesses, exact denominators, and figures. Token budgets exclude prompt prefill and padding; they are not equal-FLOP measurements. Seven synthetic regression tests pass. A synthetic end-to-end check with the actual checker and compressor rendered both figures successfully; actual 43-token scratch reconstruction was also checked without model inference.

[`scratch_variants.py`](../../code/experiments/current/r1_cot/scratch_variants.py) implements `dependency-goal-v1`, a second cold-start hypothesis. Unlike the existing reverse-all-lines plan, it traverses the final proof's dependency graph goal first, retains branch assumptions and original scope depths, omits premise leaves already in the prompt and unreachable lines, and bypasses same-scope reiteration wrappers. Shared prerequisites are emitted once by line identity; equal formulas under different assumptions remain distinct. The default cap is six source lines, and the output uses the same 43-token vocabulary. Six synthetic tests cover scope, shared dependencies, irrelevant lines, index-offset invariance, and rejection of invalid references or over-cap labels. This is an implemented representation, not evidence of improved model performance; integration and training remain separate steps.

## Explicit longer-initial-cap option

[`prepare_long_caps.py`](../../code/experiments/current/r1_cot/prepare_long_caps.py) constructs separate cap-eight and cap-twelve initial-label populations under the user's permission to increase the starting cutoff. It reads only the authorized synthetic **RL-target** reference files from `structural_long_v1` and `negation_long_v1`; the corresponding held-out reference files are not opened. Opaque guards exclude every textbook class/context/closed skeleton, and canonical guards exclude both current synthetic development populations. Previously analysis-only generator references become labels only in these explicitly new arms; this permission does not retrospectively change the earlier experiments or support a cap-six-only claim.

The [data manifest](cap_extensions_manifest.json) records 1,922 additional unique labels at cap eight (1,281 with seven lines and 641 with eight), and 4,969 at cap twelve. The original 60,000 short records appear exactly once in each mixed dataset, yielding 61,922 and 64,969 records. A global seeded canonical split reserves 1,303 distinct longer theorem prompts for RL, with no proof bodies released in that RL file. Those prompts are disjoint from all newly supervised labels. The mixtures give longer records 3.10% and 7.65% of record sampling respectively, before any explicit training-weight changes. This is a coverage option, not evidence of model improvement.

Remote outputs are under `/data/users/dmitry/nd-rl/r1-cot-20260909/data/cap_extensions/`: `additional_cap8.jsonl`, `additional_cap12.jsonl`, `sft_cap8_mixed.jsonl`, `sft_cap12_mixed.jsonl`, and `rl_long_disjoint.jsonl`. Use the disjoint RL file for these arms; an older long-RL pool can overlap the newly admitted supervised classes. Retain the source, cap, and checkpoint lineage in every subsequent model comparison.

## Bibliography and limitations

Read the project's literature workflow before adding citations. Prover-V2 was added through `uv run --script code/tools/literature.py add arxiv:2504.21801`; `references.bib` changed only through Zotero's generated export. The R1 add command reported an existing Zotero item (`guoDeepSeekR1IncentivizesReasoning2025`) outside the project collection and refused a duplicate. Other papers are linked directly as primary sources rather than inventing citation keys. No manual bibliography edit was made.

This note proposes experiments; it contains no new training result. The recent remote reports were read as current aggregate context without auditing their full Git history or all underlying artifacts. Their numerical claims therefore remain attributed historical results. End-of-sprint conclusions require the sprint's own frozen run manifests, outputs, and independent checker audit.
