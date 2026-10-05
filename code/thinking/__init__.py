"""Thinking-trace pipeline for the proof-state model (`state_env_october`, run
thinking_traces_20261005).

Model side: the `lean_thinkn` tokenizer, the training-pair builder, training / cold start, the
think-aware samplers, evaluation and expert iteration, all built on Dan Pandori's fork vendored
unmodified in `vendor/danstate/` (pinned at 4dcd0e4f; see `dan.py`).  `gen/` (LLM trace generation)
is a separate subpackage. Design: `code/experiments/current/thinking_traces_20261005/PROPOSAL.md`
and `pipeline/README.md`.
"""
