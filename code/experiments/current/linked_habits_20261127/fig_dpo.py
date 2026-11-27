"""Figure 2, the decisive arm: preference tuning (DPO) on twin pairs, read where the rated habit has moved +30 points.

From the read-out (`compare["dpo:reit"]`, `between`, `prereg["dpo:reit:at_target"]`, `tables`), untagged, marker_plain,
Lean-accepted proofs. D_h = A-share of habit h in the rated run minus in its no-preference (swapped) run at the matched
snapshot, mean over the trainer seeds. Groups of rows:
  - the rated habit itself (reiteration), D per corpus: the movement every corpus was matched on;
  - habits decided BEFORE the edited line: S_before(c) = mean over the linked habits decided before the first
    reiteration site of D_h(c) - D_h(no-links). On the all-linked corpus that is step order and the 11 readable note
    classes; on the seven-natural and four-habit corpora only step order is linked and decided before ("order only"),
    so the all-linked corpus is also shown on step order alone, like for like;
  - habits written AFTER it (conjunct order, both halves, restate): S_after(c), the same contrast; on the four-habit
    corpus only conjunct order is linked ("conj. only"), so the all-linked corpus is also shown on it alone;
  - the pre-registered marker M(c): step order and conjunct order (the Nov 18 markers), the same contrast, and
    M(all-linked) - M(four-habit);
  - a reference, NOT a contrast against the no-links corpus: a full switch of the all-linked base to author A (its
    tag-A profile minus its untagged profile), averaged over the same before-line habits; and the rated habit's own
    move under that switch (less than +30: the +30 read is past a full switch); for each all-linked base present.
The grey band is +-10 points (the pre-registered "edited one by one" zone). Second bases (PREREG Amendment 8) are drawn
when the read-out has them, else marked pending; bold rows are the two-base average (weights 1/2, against the one
no-links base). The second bases have one trainer seed each. Intervals: 95 % bootstrap over theorems (B 1,000); base-to-base
variance is not in them.

    PY fig_dpo.py [--lh LH] [--read READ_JSON] [--out DIR]
"""

# ruff: noqa: E501

from __future__ import annotations

from pathlib import Path
from typing import Any

from figcommon import CORPUS_STYLE, SHORT, S, corpus_handles, est, fmt, load_read, parser, save

KEY = "dpo:reit:at_target"


def full_switch(
    res: dict[str, Any], habits: list[str], key: str = "c24_s0"
) -> tuple[float, float] | None:
    """(mean over `habits`, reiteration) of an all-linked base's tag-A share minus its untagged share."""
    base = (((res.get("tables") or {}).get(key) or {}).get("base") or {}).get("read") or {}
    a, n = (base.get("A") or {}).get("a_share") or {}, (base.get("none") or {}).get("a_share") or {}
    vals = [a[h] - n[h] for h in habits if a.get(h) is not None and n.get(h) is not None]
    if not vals or a.get("reit") is None or n.get("reit") is None:
        return None
    return sum(vals) / len(vals), a["reit"] - n["reit"]


def row(group: str, label: str, corpus: str | None, e: Any, **kw: Any) -> dict[str, Any]:
    return {"group": group, "label": label, "corpus": corpus, "e": e, **kw}


def rows(res: dict[str, Any]) -> list[dict[str, Any]]:
    """[{group, label, corpus, e (point/ci), hollow, bold, note, ref}] top to bottom."""
    cmp = res["compare"]["dpo:reit"]
    pr = res["prereg"][KEY]
    btw = (res.get("between") or {}).get(KEY) or {}
    c24c0 = btw.get("c24-c0") or {}
    pbs = pr.get("per_base_seed") or {}
    b1 = pbs.get("1") or pbs.get(1) or {}
    avg = pr.get("base_avg") or {}
    out: list[dict[str, Any]] = []

    g1 = "the rated habit itself (reiteration), rated − control"
    for c in ("c24", "c9", "c4", "c0"):
        d = (((cmp.get(c) or {}).get("read") or {}).get("at_target") or {}).get("D") or {}
        out.append(row(g1, SHORT[c], c, d.get("reit")))
    for c in ("c24b1", "c4b1"):
        if c in cmp:
            d = ((cmp[c].get("read") or {}).get("at_target") or {}).get("D") or {}
            out.append(row(g1, f"{SHORT[c[:-2]]}, 2nd base", c[:-2], d.get("reit"), hollow=True))

    g2 = "decided before the edited line, − no-links"
    out.append(row(g2, "all-linked: notes + step order", "c24", pr.get("P2_S_before(c24)")))
    if b1.get("S_before(c24)"):
        out.append(row(g2, "all-linked, 2nd base", "c24", b1["S_before(c24)"], hollow=True))
        out.append(row(g2, "all-linked, 2 bases", "c24", avg.get("S_before(c24)"), bold=True))
    else:
        out.append(row(g2, "all-linked, 2nd base", "c24", None, note="pending"))
    out.append(row(g2, "all-linked: step order only", "c24", c24c0.get("order")))
    out.append(row(g2, "seven-natural: step order only", "c9", pr.get("P2_S_before(c9)")))
    out.append(row(g2, "four-habit: step order only", "c4", pr.get("P2_S_before(c4)")))

    g3 = "written after it, − no-links"
    out.append(row(g3, "all-linked: conj., halves, restate", "c24", pr.get("P2_S_after(c24)")))
    out.append(row(g3, "all-linked: conj. order only", "c24", c24c0.get("conj")))
    out.append(row(g3, "seven-natural: conj., halves, restate", "c9", pr.get("P2_S_after(c9)")))
    out.append(row(g3, "four-habit: conj. order only", "c4", pr.get("P2_S_after(c4)")))

    g4 = "Nov 18 markers M (step + conjunct order), − no-links"
    for c in ("c24", "c9", "c4"):
        out.append(row(g4, SHORT[c], c, pr.get(f"P1_M({c})")))
    if b1.get("M(c24)"):
        for c in ("c24", "c4"):
            out.append(row(g4, f"{SHORT[c]}, 2nd base", c, b1.get(f"M({c})"), hollow=True))
            out.append(row(g4, f"{SHORT[c]}, 2 bases", c, avg.get(f"M({c})"), bold=True))
    else:
        out.append(row(g4, "2nd bases", "c24", None, note="pending"))
    out.append(row(g4, "all-linked − four-habit", None, pr.get("P1_M(c24)-M(c4)")))
    if avg.get("M(c24)-M(c4)"):
        out.append(
            row(g4, "all-linked − four-habit, 2 bases", None, avg["M(c24)-M(c4)"], bold=True)
        )

    g5 = "reference, all-linked base alone (not − no-links): full switch to author A"
    hs = (pr.get("P2_S_before(c24)") or {}).get("habits") or []
    for key, suffix in (("c24_s0", ""), ("c24_s1", ", 2nd base")):
        fs = full_switch(res, hs, key)
        if fs is not None:
            out.append(
                row(
                    g5, f"before-line habits{suffix}", "c24", {"point": fs[0], "ci": None}, ref=True
                )
            )
            out.append(
                row(
                    g5, f"reiteration itself{suffix}", "c24", {"point": fs[1], "ci": None}, ref=True
                )
            )
    return out


def main(argv: list[str] | None = None) -> list[Path]:
    a = parser(__doc__).parse_args(argv)
    res, src = load_read(a)
    print(f"read-out: {src}")
    rs = rows(res)
    groups: list[str] = []
    for r in rs:
        if r["group"] not in groups:
            groups.append(r["group"])
    y, ys, heads = 0.0, [], []
    for g in groups:
        heads.append((g, y))
        y -= 0.9
        for r in rs:
            if r["group"] == g:
                ys.append(y)
                y -= 0.75
        y -= 0.25
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 0.16 * len(rs) + 0.18 * len(groups) + 0.6)
        ax = fig.add_subplot(1, 1, 1)
        ax.axvspan(-10, 10, color=S.SHADE, zorder=0, linewidth=0)
        ax.axvline(0, color=S.REF, linewidth=0.6, zorder=1)
        ax.axvline(30, color=S.REF, linewidth=0.5, linestyle=(0, (1, 1.5)), zorder=1)
        for r, yy in zip(rs, ys, strict=True):
            st = CORPUS_STYLE.get(r["corpus"] or "", {"color": S.INK, "marker": "P"})
            v = est(r["e"])
            ax.text(-0.012, yy, r["label"], transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=S.FS_TICK, fontweight="bold" if r.get("bold") else "normal")  # fmt: skip
            if v is None:
                ax.text(0.5, yy, r.get("note", "not read"), fontsize=S.FS_TICK - 0.5, color=S.REF, va="center",
                        style="italic")  # fmt: skip
                continue
            if r.get("ref"):
                ax.plot(v[0], yy, linestyle="none", marker="D", markersize=4.4, markerfacecolor=S.WHITE,
                        markeredgecolor=S.INK, markeredgewidth=0.8, zorder=4)  # fmt: skip
                ax.annotate(f"{v[0]:+.1f}", (v[0], yy), xytext=(5, 0), textcoords="offset points", va="center",
                            fontsize=S.FS_TICK - 1, color=S.INK2)  # fmt: skip
                continue
            ax.errorbar(v[0], yy, xerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="none", ecolor=st["color"],
                        elinewidth=0.9, capsize=1.6, zorder=3)  # fmt: skip
            ax.plot(v[0], yy, linestyle="none", marker=st["marker"], markersize=4.4, zorder=4,
                    markerfacecolor=S.WHITE if r.get("hollow") else st["color"],
                    markeredgecolor=st["color"] if r.get("hollow") else S.WHITE, markeredgewidth=0.8)  # fmt: skip
            ax.annotate(fmt(r["e"]), (max(v[2], 10.5), yy), xytext=(3, 0),
                        textcoords="offset points", va="center", fontsize=S.FS_TICK - 1, color=S.INK2)  # fmt: skip
        for g, yy in heads:
            ax.text(0.005, yy - 0.2, g, transform=ax.get_yaxis_transform(), ha="left", va="center",
                    fontsize=S.FS_TICK, fontstyle="italic", color=S.INK)  # fmt: skip
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_ylim(min(ys) - 0.6, 0.2)
        ax.set_xlim(-22, 60)
        ax.text(
            0,
            0.25,
            "±10: edited one by one",
            fontsize=S.FS_TICK - 1,
            color=S.REF,
            ha="center",
            va="bottom",
        )
        ax.text(
            30,
            0.25,
            "+30: the matched move",
            fontsize=S.FS_TICK - 1,
            color=S.REF,
            ha="center",
            va="bottom",
        )
        ax.set_xlabel("movement toward author A at the +30 read (points)")
        S.light_grid(ax, "x")
        ax.legend(handles=corpus_handles(["c24", "c9", "c4", "c0"]), loc="center right", bbox_to_anchor=(1.0, 0.36), fontsize=S.FS_TICK - 0.5,
                  title="corpus (hollow: second base)", title_fontsize=S.FS_TICK - 0.5)  # fmt: skip
        return save(fig, Path(a.out), "fig2_dpo")


if __name__ == "__main__":
    main()
