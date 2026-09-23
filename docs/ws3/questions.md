# Workstream 3: questions for Charles

Agents append here instead of blocking (plan section 6): date, the question, and the default
followed if unanswered. Only the stop-and-ask cases in `CLAUDE.md` halt a session.

## 2026-09-23 branch name collision

Git cannot hold both `refs/heads/charles` and `refs/heads/charles/main`. The local-only
branch `charles` (no remote) was renamed to `charles-pre-ws3`, unchanged, and `charles/main`
was created from it. **Default:** keep `charles-pre-ws3` as the pre-ws3 pointer; delete it
once `charles/main` is pushed, if you prefer.

