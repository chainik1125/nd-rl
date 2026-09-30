"""Lazy analysis-view registry.

Every value is a `"module:attr"` string, resolved by `cli.py`'s `dash` subcommand (and usable
directly from Python/notebooks) via `importlib`. Registry keys match the dash page names plan §3's
WP6 row lists (`overview, yue, families, glue, skills, subst, padding, prefix, edl, runs`);
`overview` and `runs` are synthesised by `dash.build` directly from the `Store` (every run's
manifest, folded across every suite) rather than from a per-benchmark analysis module, so they
have no entry here. `families`'s benchmark (plan §2 B2, restricted-class horizon curves) is
implemented by `analysis/horizon.py` (named for its `metrics.horizon` statistic, not for the suite
it analyses), hence the key/module mismatch below.

View contract (plan §3): `view_<bench>(store, cfg) -> dict` (JSON-able).

`interp_likelihood`/`interp_steering`/`interp_weights`/`interp_probes` are the interp coder's four
measures (`code/experiments/current/benchmarking_20260930/INTERP_SPEC.md`); `analysis/interp.py`
names its functions `view_likelihood`/`view_steering`/`view_interpolation`/`view_probes` (matching
the spec's own names for the four measures), hence the `interp_weights`/`view_interpolation`
key/attribute mismatch, the same pattern as `families`/`horizon.py` above.
"""

VIEWS: dict[str, str] = {
    "yue": "code.ndbench.analysis.yue:view",
    "families": "code.ndbench.analysis.horizon:view",
    "glue": "code.ndbench.analysis.glue:view",
    "skills": "code.ndbench.analysis.skills:view",
    "subst": "code.ndbench.analysis.subst:view",
    "padding": "code.ndbench.analysis.padding:view",
    "prefix": "code.ndbench.analysis.prefix:view",
    "edl": "code.ndbench.analysis.edl:view",
    "interp_likelihood": "code.ndbench.analysis.interp:view_likelihood",
    "interp_steering": "code.ndbench.analysis.interp:view_steering",
    "interp_weights": "code.ndbench.analysis.interp:view_interpolation",
    "interp_probes": "code.ndbench.analysis.interp:view_probes",
}
