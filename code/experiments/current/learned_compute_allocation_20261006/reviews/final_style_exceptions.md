---
written_on: 2026-10-06
written_by: agent:codex
status: complete
---

# Final scoped style check

Scoped Ruff lint passes across the complete sprint Python tree, the three new shared modules and the ten new test modules. The format check reports 251 files already formatted and two cosmetic exceptions. Their exact bytes are retained because the presentation renderer is bound to published figure provenance and the shared queue launcher is bound to the sealed operational addendum. Formatting them in place would break that provenance for no scientific benefit. This is a documented formatting failure, not a fully passing format check.

| Preserved file, relative to the sprint directory | SHA256 |
|---|---|
| `selection/render_main_presentation.py` | `7823c8fa4728def366f72becbddf2c10335d6112d59d28422f33fc20a2fad889` |
| `selection/run_frozen_test_queue_shared.py` | `dc6d25de03e9797ef5abe8bc5ed656eb23fd4dcce281dc73e465fd2ab2696c1f` |

The suggested changes concern whitespace and wrapping. No new numerical, model, policy or estimator change was made. The earlier whole-workspace check reported 5,965 Ruff findings, nearly all in the preexisting `proof-state` work; the one new warning was subsequently fixed. Those workspace findings remain separate from the passing scoped lint check. The focused regression suite previously passed all 77 tests.
