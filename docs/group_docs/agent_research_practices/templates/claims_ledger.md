---
written_on: YYYY-MM-DD
written_by: <human:Name or agent:claude>
---

# Claims ledger — <paper, report or run>

Every number or claim a write-up may use, traced to the file that produces it and to its review status. Writers cite
ledger ids. A claim with no row does not go in the text.

| id | claim, as it may be written | value | model(s) and settings | source file (path and line or key) | produced by | review status |
| --- | --- | --- | --- | --- | --- | --- |
| L1 | | | | | script and commit | reviewed in `<review file>` § … / corrected to … / unreviewed |

Rules:
- When a review corrects a value or its wording, keep the row and add the correction, so the history stays visible.
- A superseded row is marked superseded and points to its replacement.
- Before release, an independent check re-derives every row that the text cites.
