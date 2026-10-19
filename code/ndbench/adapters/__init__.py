"""Model backends implementing `adapters.base.ModelAdapter`.

Only `fake` (WP0) is implemented directly here; the rest (`absgpt`, `lean_seq`, `step_env`,
`api_llm`) are later work packages, resolved lazily through `adapters.base.ADAPTERS` so importing
this package never pulls in a GPU stack or model-specific dependencies.
"""
