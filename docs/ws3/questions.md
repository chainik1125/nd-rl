# Workstream 3: questions for Charles

Agents append here instead of blocking (plan section 6): date, the question, and the default
followed if unanswered. Only the stop-and-ask cases in `CLAUDE.md` halt a session.

## 2026-09-23 branch name collision

Git cannot hold both `refs/heads/charles` and `refs/heads/charles/main`. The local-only
branch `charles` (no remote) was renamed to `charles-pre-ws3`, unchanged, and `charles/main`
was created from it. **Default:** keep `charles-pre-ws3` as the pre-ws3 pointer; delete it
once `charles/main` is pushed, if you prefer.

## 2026-09-23 exp-00 follow-ups

- **Proof dumps in git.** exp-00 commits its judged-proof dumps gzipped (`run/**/*.jsonl.gz`,
  about 1.1 MB total) so the reviewer can recount without rerunning. Model directories are
  gitignored. **Default:** keep committing gzipped dumps while they stay under a few MB per
  experiment; move to the HF bucket if a run's dumps exceed that.
- **exp-01 config.** The startup.md row for exp-01 keeps lengths 3 to 5 and lr 1e-5. In exp-00
  those prompts gave zero within-group reward variance in 68 percent of groups and a KL of
  0.001 after 50 steps. **Default:** exp-01 keeps the row's config for comparability with
  exp-00 on the pod (same numbers expected), and exp-02 is where the mix and learning rate
  change; if Charles prefers, exp-01 can already weight lengths 5 to 7.
- **`code/ws3/run.py` edited after the run.** The report stage crashed on a matplotlib
  error-bar rounding check; the clamp fix was committed after the run and the report stage
  rerun on the untouched outputs. **Default:** treat exp-00 as clean (no training or evaluation
  code changed), and say so in `review.md`.
