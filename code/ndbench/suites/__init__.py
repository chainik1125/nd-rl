"""Lazy suite-builder registry.

Every value is a `"module:attr"` string, resolved by `cli.py`'s `build` subcommand via
`importlib` (see `cli._resolve`) so importing `ndbench.suites` never imports a work package's
suite-building dependencies (minlen multiprocessing pools, the vendored take-home generators,
...). Every builder module the plan (§1's architecture, §3's WP2-WP4 rows) names is listed here
even before that work package lands; resolving an unimplemented entry raises
`ModuleNotFoundError`, which `cli.py` reports as "not implemented yet" rather than a KeyError, so
this registry does not need to change shape as WP2-WP4 land -- only their target modules need to
start existing.

Builder contract (plan §3):
`build(cfg, *, certifier, excluder, rng_seed) -> tuple[list[Problem], dict]` (WP4's glue builder is
`build_joined(cfg) -> tuple[list[Problem], dict]`, taking mastery run ids instead of a bare
rng_seed-driven config; see plan §1's "How each adapter fits" / §3 WP4 row).
"""

BUILDERS: dict[str, str] = {
    # WP2 (logic + existing suites): suites/existing.py
    "textbook_dev": "code.ndbench.suites.existing:textbook_dev",
    "textbook_law": "code.ndbench.suites.existing:textbook_law",
    "gen_ctrl": "code.ndbench.suites.existing:gen_ctrl",
    "skill_pool": "code.ndbench.suites.existing:skill_pool",
    "parts_pool": "code.ndbench.suites.existing:parts_pool",
    # WP3 (families, laws, padding)
    "families": "code.ndbench.suites.families:build",
    "laws": "code.ndbench.suites.laws:build",
    "padding": "code.ndbench.suites.padding:build",
    # WP4 (glue + prefix)
    "glue": "code.ndbench.suites.glue:build_joined",
    "glue_parts_renamed": "code.ndbench.suites.glue:build_renamed_parts",
    # WP8 integration fix list item 7: `build_length_sweep` re-wraps the existing matched AND/CHAIN
    # items (`ab3/sealed/matched_L6-14.jsonl`) as plan §2 B1's "extra length sweep"; WP4 left it
    # unregistered ("that registry is WP0-owned") even though it already matches the builder
    # contract (`(cfg, *, certifier, excluder, rng_seed) -> (list[Problem], dict)`).
    "glue_length_sweep": "code.ndbench.suites.glue:build_length_sweep",
    "prefix": "code.ndbench.suites.prefix:build",
}
