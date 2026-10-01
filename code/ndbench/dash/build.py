"""`dash.build`: writes the static ndbench dashboard (plan §3 WP6 row; `cli.py`'s `dash`
subcommand resolves `code.ndbench.dash.build:build` lazily and calls it as `build(store, out_dir)`).

Computes every analysis view (importing each `analysis/*` module directly -- WP6 owns all of them
together, so there is no need for `analysis.VIEWS`'s string-registry indirection *within* this
package; that registry is for `cli.py` and other external callers) and writes one self-contained
HTML page per benchmark plus `overview`/`runs` (plan's page list: overview, yue, families, glue,
skills, subst, padding, prefix, edl, runs), linked by a shared nav bar built once in
`dash/primitives.py`. Every page is fully self-contained (CSS/JS inlined, data embedded as JSON,
no network requests); `runs`/`overview` are built directly from the `Store` rather than from a
`view_<bench>` dict, per `analysis/__init__.py`'s docstring.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ..analysis import edl as edl_view
from ..analysis import glue as glue_view
from ..analysis import horizon as horizon_view
from ..analysis import interp as interp_view
from ..analysis import padding as padding_view
from ..analysis import prefix as prefix_view
from ..analysis import skills as skills_view
from ..analysis import subst as subst_view
from ..analysis import yue as yue_view
from ..core.types import JSON
from ..store import Store
from . import interp_pages, pages
from . import primitives as P

#: `(href, display name)`, also the nav bar order. The four `interp_*` pages are the interp
#: coder's own measures (`INTERP_SPEC.md`); this coder owns `dash/build.py`/`dash/pages.py` for
#: additive edits (new pages + nav) until that work package finishes.
NAV: tuple[tuple[str, str], ...] = (
    ("index.html", "Overview"),
    ("yue.html", "Yue et al."),
    ("families.html", "Families"),
    ("glue.html", "Glue"),
    ("skills.html", "Skills"),
    ("subst.html", "Substitution"),
    ("padding.html", "Padding"),
    ("prefix.html", "Prefix"),
    ("edl.html", "EDL"),
    ("interp_likelihood.html", "Interp: Likelihood"),
    ("interp_steering.html", "Interp: Steering"),
    ("interp_weights.html", "Interp: Weights"),
    ("interp_probes.html", "Interp: Probes"),
    ("runs.html", "Runs"),
)

_VIEWS = {
    "yue": (yue_view.view, pages.render_yue),
    "families": (horizon_view.view, pages.render_families),
    "glue": (glue_view.view, pages.render_glue),
    "skills": (skills_view.view, pages.render_skills),
    "subst": (subst_view.view, pages.render_subst),
    "padding": (padding_view.view, pages.render_padding),
    "prefix": (prefix_view.view, pages.render_prefix),
    "edl": (edl_view.view, pages.render_edl),
    "interp_likelihood": (interp_view.view_likelihood, interp_pages.render_interp_likelihood),
    "interp_steering": (interp_view.view_steering, interp_pages.render_interp_steering),
    "interp_weights": (interp_view.view_interpolation, interp_pages.render_interp_weights),
    "interp_probes": (interp_view.view_probes, interp_pages.render_interp_probes),
}


def build(
    store: Store, out_dir: Path, cfg: Mapping[str, Mapping[str, JSON]] | None = None
) -> list[Path]:
    """Render every dashboard page into `out_dir` (created if missing) and return the paths
    written. `cfg` optionally maps a page key (`"yue"`, `"families"`, ...) to that view's own
    `cfg` dict (plan §1's per-benchmark suite/model selection); every view has a workable default,
    so `cfg=None` (or a page missing from it) still renders whatever the `Store` has under the
    plan's assumed suite/tag names -- including an empty "no data yet" panel when there is none.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = cfg or {}
    nav = list(NAV)
    names = dict(NAV)
    written: list[Path] = []

    # Kept alongside each page's own rendered figures (fix list item 4): the overview page's
    # headline numbers are pulled straight from these same `view_<bench>` dicts rather than
    # re-querying the `Store` a second time, so a headline number and its own page can never
    # disagree about what the `Store` currently holds.
    view_data: dict[str, JSON] = {}
    for key, (view_fn, render_fn) in _VIEWS.items():
        data = view_fn(store, cfg.get(key, {}))
        view_data[key] = data
        figs = render_fn(data)
        html = P.page(f"ndbench: {names[f'{key}.html']}", str(store.root), f"{key}.html", figs, nav)
        path = out_dir / f"{key}.html"
        path.write_text(html)
        written.append(path)

    all_runs = store.runs()
    runs_figs = pages.render_runs(all_runs, store.ledger_total_s())
    runs_html = P.page("ndbench: Runs", str(store.root), "runs.html", runs_figs, nav)
    runs_path = out_dir / "runs.html"
    runs_path.write_text(runs_html)
    written.append(runs_path)

    statuses: dict[str, int] = {}
    for r in all_runs:
        status = str(r.get("status", "unknown"))
        statuses[status] = statuses.get(status, 0) + 1
    n_suites = len({str(r.get("suite")) for r in all_runs})
    overview_figs = pages.render_overview(
        len(all_runs),
        n_suites,
        statuses,
        [(href, name) for href, name in nav if href != "index.html"],
        view_data,
    )
    overview_html = P.page("ndbench dashboard", str(store.root), "index.html", overview_figs, nav)
    index_path = out_dir / "index.html"
    index_path.write_text(overview_html)
    written.append(index_path)

    return written
