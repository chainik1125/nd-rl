---
written_on: 2026-10-05
written_by: agent:claude
---

# `code.thinking.gen`: LLM trace generation

The trace generator for the thinking-trace experiment
([PROPOSAL.md §3](../../experiments/current/thinking_traces_20261005/PROPOSAL.md)). It selects source
proofs, samples base-model mistakes, builds Message Batches requests, submits them, validates the
replies with the unchanged `trace_dsl.py` and writes retries. Nothing here has made a paid call. The
calibration plan, with its commands and cost estimate, is in
[`experiments/current/thinking_traces_20261005/gen/README.md`](../../experiments/current/thinking_traces_20261005/gen/README.md).

Run every command from the repository root, as a module. Running a file directly as a script
breaks imports, because `select.py` would shadow the standard library's `select` module.

```bash
uv run --project code --no-sync python -m code.thinking.gen <command> --run-dir RUN [flags]
uv run --project code --no-sync python -m code.thinking.gen <command> --config FILE.yaml
```

## Commands and files

| command | does | writes (under RUN) |
|---|---|---|
| `select` | stratified, seeded sample of a `prompt`/`proof` JSONL (default K12); excludes every evaluation set by renaming class | `selection.jsonl`, `selection.meta.json` (source sha256, seed, strata, exclusion sets with sha256 and hit counts, cells, git commit) |
| `distractors` | k base-model samples per decision state at T 0.8 (fork's `guided_sample.decode`, `--device`, CPU by default); keeps distinct actions the environment or `step_check` rejects | `distractors.jsonl` (`{proof_id, step, ref_action, distractors: [{action, why, count}], …}`), `.meta.json` |
| `build` | one request per proof per model; also the dry-run estimate | `requests/<tag>.r1.jsonl` (exact Batches `requests` items), `.manifest.jsonl`, `.estimate.json` |
| `estimate` | cost estimate of request files | stdout |
| `submit` | **dry run unless `--submit --max-usd X`**; refuses if the worst case exceeds X, the key is unset, or the ledger already shows that file's sha256 | `ledger.jsonl` (`submit` row), `batches/<id>.json` |
| `poll` | retrieves batches (all unfinished ones in the ledger by default); downloads results when they have ended | `results/<id>.jsonl`, ledger `ended` row |
| `collect` | parses, renders pilot-format records, runs `trace_dsl.validate(strict=True, lean=--lean)`, writes retries | `collect/accepted.<tag>.jsonl`, `rejected.<tag>.jsonl`, `chains.jsonl`, `stats.json`, `report.md`; `requests/<tag>.r2.jsonl` …; ledger `usage` rows with billed tokens and cost |
| `report` | prints `collect/report.md` | – |

Modules: `select.py`, `distractors.py`, `requests.py` (prompts, request params, retries,
estimate), `client.py` (raw-HTTPS Batches client behind a `BatchClient` protocol), `batch.py`
(submit, poll, ledger), `collect.py`, `models.py` (verified prices and capabilities), `dsl.py`
(trace_dsl loader, ASCII surface), `envloop.py` (mode 3 interface), `fakes.py` (fake Batches
server for tests), `cli.py`.

**Modes.** `reference`: the user turn lists every state from `state_env.decompose(prompt, proof,
canon=True)` with its reference action. Under each decision state it adds up to 3 "likely
mistakes" from the distractor file. `solve`: the theorem and its first state only. `envloop.py`
defines mode 3, environment in the loop, as a strict `env_step` tool and an `EnvSession` that checks
one step with the same validator. It has no driver: mode 3 cannot be batched.

**System prompt.** The "System prompt" section of `production_prompt.md`, with the markdown table
escape `\|` written as `|`. Two complete pilot traces follow as worked examples (`--examples`, by
default `k12_16472,k12_22530`, the two non-deviating reference-guided ones). They bring the cached
prefix to about 5.7k estimated tokens. Pilot rows are excluded from selection.

**Retries** (at most 2, `--max-attempts 3`).
- A validation, JSON or schema failure appends the previous assistant turn unchanged, with its
  thinking blocks, and then a user turn: "The checker rejected this trace: <first error, in ASCII>
  … Write the whole trace again". The history stays append-only, as preserved thinking requires.
- `max_tokens` truncation is resubmitted fresh with `max_tokens × 1.5`.
- Expired, canceled and server-errored attempts are resubmitted fresh.
- `invalid_request_error` and refusals are rejected without a retry.

The collector is stateless. Each run re-reads every manifest and result, so re-running it is safe.

**Stats** (`collect/stats.json`), per model:
- acceptance %, first-pass %, accepted by attempt;
- billed tokens by category (input, cache read, cache write 5m/1h, output) in total and **per
  accepted trace**, with all attempts of every proof included, rejected ones too;
- cost and cost per accepted trace;
- error categories;
- mean think tokens and the share of accepted traces that deviate.

`sonnet_vs_opus` gives Sonnet's billed tokens per accepted trace as a multiple of Opus's (Leon's
rule: use Sonnet only under 2×), together with the output-only and cost ratios.

**Key safety.**
- `client.py` reads `ANTHROPIC_API_KEY` at call time and puts it only in the `x-api-key` header.
  It never prints, logs, stores or writes it.
- It refuses to send the key to any host other than the API base, which also covers `results_url`.
- A POST is never retried, so an ambiguous failure cannot create a second billed batch.
- The tests run the whole pipeline with a sentinel key and scan every written file, stdout, stderr
  and log record for it.

## API details verified on 2026-10-05

Pages fetched with WebFetch, plus Anthropic's bundled `claude-api` skill reference where marked.

| item | verified value | source |
|---|---|---|
| model IDs | `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-4-5-20251001`; max output 128K / 128K / 64K | [models overview](https://platform.claude.com/docs/en/about-claude/models/overview) |
| thinking | Opus 5.5: adaptive, always on (`disabled` and `budget_tokens` are a 400 at every effort). Sonnet 5.5: adaptive. Haiku 4.5: extended (`{"type": "enabled", "budget_tokens": N}`, at least 1024 and below `max_tokens`) | models overview; [effort](https://platform.claude.com/docs/en/build-with-claude/effort); [Opus 5.5 migration guide](https://platform.claude.com/docs/en/models/opus-5-5/migration-guide) |
| effort | `output_config.effort`, GA, no beta header; `low`…`max`. Default `medium` on Opus 5.5 and `high` on Sonnet 5.5. **Not supported on Haiku 4.5**, so it is omitted there | effort page (supported-models list); models overview ("Default effort: Not supported") |
| structured output | `output_config: {format: {type: "json_schema", schema}}`, GA, no beta header (`output_format` is deprecated). Supported on all three models. Every object needs `additionalProperties: false`. `minItems` only 0 or 1; no `maxItems`, `minLength`/`maxLength` or numeric bounds. The JSON arrives in the text block. On `stop_reason` `refusal` or `max_tokens` the output may not match the schema. Changing the schema invalidates the prompt cache | [structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs) |
| forced tool use | `tool_choice` `any`/`tool` returns 400 on Opus 5.5 (`tool_choice: type "tool" and "any" are not supported for this model.`), including in batches and token counting. Not used here | Opus 5.5 migration guide |
| other 400s on Opus 5.5 | non-default `temperature`/`top_p`/`top_k`; assistant prefill | Opus 5.5 migration guide |
| prompt caching | `cache_control: {"type": "ephemeral"}` (5 min) or `{"type": "ephemeral", "ttl": "1h"}` on the system text block. Minimum cacheable prefix 512 tokens (Opus 5.5, Sonnet 5.5) and **4096 (Haiku 4.5)**; shorter prefixes are silently not cached. Usage fields: `cache_read_input_tokens`, `cache_creation_input_tokens`, `cache_creation.{ephemeral_5m_input_tokens, ephemeral_1h_input_tokens}` | [prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) |
| Batches API | `POST /v1/messages/batches` with `{"requests": [{"custom_id", "params"}]}`; `GET /v1/messages/batches/{id}`; results JSONL at the batch's `results_url` (in any order, key by `custom_id`). Headers `x-api-key`, `anthropic-version: 2023-06-01`. `custom_id` matches `^[a-zA-Z0-9_-]{1,64}$`. At most 100,000 requests or 256 MB. 24 h expiry; results kept 29 days. Result types `succeeded` / `errored` / `canceled` / `expired`, and the last three are not billed. `stream`, `speed` and `max_tokens: 0` are rejected | [batch processing](https://platform.claude.com/docs/en/build-with-claude/batch-processing) |
| caching in batches | Supported, and the discounts stack. Hits are best effort (typically 30–98%). The docs suggest the 1-hour TTL for batches, so it is the default here (`--cache-ttl 1h`) | batch processing |
| preserved thinking | Pass thinking blocks back unchanged; keep the history append-only. In batches an unset `prefix_mismatch_behavior` drops a mismatched block instead of failing the item | Opus 5.5 migration guide ("pass `thinking` blocks back unmodified"); the drop default is from the bundled `claude-api` skill reference, not re-fetched |

Prices (USD per MTok; [pricing](https://platform.claude.com/docs/en/about-claude/pricing)). The
Batch API halves every category, and "these multipliers stack with other pricing modifiers,
including the Batch API discount".

| model | input | 5m cache write | 1h cache write | cache read | output | batch input / output | batch cache read |
|---|---|---|---|---|---|---|---|
| Opus 5.5 | 4 | 5 | 8 | 0.20 (0.05×) | 20 | 2 / 10 | 0.10 |
| Sonnet 5.5 | 2 | 2.50 | 4 | 0.20 | 10 | 1 / 5 | 0.10 |
| Haiku 4.5 | 1 | 1.25 | 2 | 0.10 | 5 | 0.50 / 2.50 | 0.05 |

Cache reads cost the same on Opus 5.5 and Sonnet 5.5, so Sonnet's cost ratio is a little above 0.5
even at equal tokens.

**Token estimate.** `est_tokens` = whitespace words + characters / 8. It is an over-estimate for
English and about one token per symbol for formulas. The free `count_tokens` endpoint would give
exact counts, but it needs the key and was not called. The worst case multiplies input by 1.25,
prices every system prompt as a cache write, and spends `max_tokens` on every request.

## Tests

`code/tests/thinking/test_gen_*.py` (43 tests, CPU, about 6 s) cover:
- selection, eval exclusion and provenance;
- request shape against the table above;
- the estimate arithmetic;
- dry run, `--max-usd` and resubmission refusals;
- the key scan;
- the end-to-end fake batch run with feedback, truncation, server-error and always-bad retries;
- stats and the ledger;
- distractors, including the real stage-1 checkpoint on 2 CPU states;
- the env-loop interface.
