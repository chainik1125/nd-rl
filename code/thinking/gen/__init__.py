"""LLM trace generation for the thinking-trace experiment (PROPOSAL.md section 3).

Pipeline: `select` -> `distractors` -> `build` (dry run: request JSONL + cost estimate) ->
`submit` (Message Batches API, needs --submit and --max-usd) -> `poll` -> `collect` (validate with
trace_dsl, write retries) -> `report`.  Run as `python -m code.thinking.gen <command>` from the
repository root; see README.md in this directory.
"""
