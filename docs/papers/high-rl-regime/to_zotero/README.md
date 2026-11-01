---
written_on: 2026-10-30
written_by: "agent:claude"
---

# References waiting for Zotero

One paper per file, for `literature.py add` (it imports one paper at a time and keeps these citekeys). Zotero runs
only on a machine with the desktop app, so add them from there, with the `natural-deduction-rl` collection selected:

```bash
uv run --script code/tools/literature.py add docs/papers/high-rl-regime/to_zotero/olmo3.bib
uv run --script code/tools/literature.py add docs/papers/high-rl-regime/to_zotero/openai_hf_incident.bib
uv run --script code/tools/literature.py add docs/papers/high-rl-regime/to_zotero/officechai_semianalysis.bib
uv run --script code/tools/literature.py add docs/papers/high-rl-regime/to_zotero/macdiarmid_reward_hacking.bib
uv run --script code/tools/literature.py add docs/papers/high-rl-regime/to_zotero/denison_reward_tampering.bib
uv run --script code/tools/literature.py add docs/papers/high-rl-regime/to_zotero/baker_cot_monitoring.bib
```

Until then `build.sh` appends any file here whose citekey is not yet in the Zotero export to its build copy of
`references.bib`, so the paper cites them already. Delete a file once its key appears in the root `references.bib`.

Sources checked 2026-10-30: arXiv API for 2512.13961 (title, first authors, v1 date); the OpenAI report PDF (title;
publication date 26 August 2026 from its press coverage); the OfficeChai page (headline, byline, date).
Checked 2026-11-01 against the arXiv API: 2511.18397, 2406.10162 and 2503.11926 (title, author order, v1 date).
