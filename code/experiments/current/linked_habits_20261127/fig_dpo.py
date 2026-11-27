"""Figure 2, the decisive arm: preference tuning (DPO) on twin pairs, read where the rated habit has moved +30 points.

From the read-out (`compare["dpo:reit"]`, `prereg["dpo:reit:at_target"]`), untagged, marker_plain, Lean-accepted proofs.
D_h = A-share of habit h in the rated run minus in its no-preference (swapped) run at the matched snapshot, mean over
the trainer seeds. Rows:
  - the rated habit itself (reiteration), D per corpus: the movement every corpus was matched on;
  - habits decided BEFORE the edited line (the notes and step order): S_before(c) = mean over those habits of
    D_h(c) - D_h(none linked); in the 7-natural and 4-linked corpora only step order is linked and decided before;
  - habits written AFTER it (conjunct order, both halves, restate; conjunct order only where only 4 are linked):
    S_after(c), the same contrast;
  - the pre-registered marker M(c): step order and conjunct order (the Nov 18 markers), the same contrast, and
    M(all) - M(4 linked).
The grey band is +-10 points (the pre-registered "edited one by one" zone). The open diamond is what a whole-author
shift of the all-linked base would give the before-habits (the read-out's mixture: the base's tag-A / tag-B profiles
mixed to match the rated move). Second bases (PREREG Amendment 8) are drawn when the read-out has them, else marked
pending. Intervals: 95 % bootstrap over theorems; base-to-base variance is not in them.

    PY fig_dpo.py [--lh LH] [--read READ_JSON] [--out DIR]
"""

# ruff: noqa: E501

from __future__ import annotations

from pathlib import Path
from typing import Any

from figcommon import CORPUS_STYLE, SHORT, S, corpus_handles, est, load_read, parser, save

KEY = "dpo:reit:at_target"


def _whole_author(cmp: dict[str, Any], habits: list[str]) -> float | None:
    mix = (
        (((cmp.get("c24") or {}).get("read") or {}).get("at_target") or {}).get("mixture") or {}
    ).get("habits") or {}
    vals = [mix[h]["implied"] for h in habits if (mix.get(h) or {}).get("implied") is not None]
    return sum(vals) / len(vals) if vals else None


def rows(res: dict[str, Any]) -> list[dict[str, Any]]:
    """[{group, label, corpus, e (point/ci), hollow, note}] top to bottom."""
    cmp = res["compare"]["dpo:reit"]
    pr = res["prereg"][KEY]
    out: list[dict[str, Any]] = []
    g1 = "the rated habit itself (reiteration), rated − control"
    for c in ("c24", "c9", "c4", "c0"):
        d = (((cmp.get(c) or {}).get("read") or {}).get("at_target") or {}).get("D") or {}
        out.append({"group": g1, "label": SHORT[c], "corpus": c, "e": d.get("reit")})
    for c in ("c24b1", "c4b1"):
        if c in cmp:
            d = ((cmp[c].get("read") or {}).get("at_target") or {}).get("D") or {}
            out.append({"group": g1, "label": f"{SHORT[c[:-2]]}, 2nd base", "corpus": c[:-2], "e": d.get("reit"),
                        "hollow": True})  # fmt: skip
    pbs = pr.get("per_base_seed") or {}
    b1 = pbs.get("1") or pbs.get(1) or {}
    avg = pr.get("base_avg") or {}
    g2 = "decided before the edited line (notes, step order), − none linked"
    out.append(
        {"group": g2, "label": "all linked", "corpus": "c24", "e": pr.get("P2_S_before(c24)")}
    )
    if b1.get("S_before(c24)"):
        out.append(
            {
                "group": g2,
                "label": "all linked, 2nd base",
                "corpus": "c24",
                "e": b1["S_before(c24)"],
                "hollow": True,
            }
        )
        out.append(
            {
                "group": g2,
                "label": "all linked, 2 bases",
                "corpus": "c24",
                "e": avg.get("S_before(c24)"),
                "bold": True,
            }
        )
    else:
        out.append(
            {
                "group": g2,
                "label": "all linked, 2nd base",
                "corpus": "c24",
                "e": None,
                "note": "pending",
            }
        )
    out.append(
        {
            "group": g2,
            "label": "7 natural (order only)",
            "corpus": "c9",
            "e": pr.get("P2_S_before(c9)"),
        }
    )
    out.append(
        {
            "group": g2,
            "label": "4 linked (order only)",
            "corpus": "c4",
            "e": pr.get("P2_S_before(c4)"),
        }
    )
    g3 = "written after it (conjunct order, halves, restate), − none linked"
    out.append(
        {"group": g3, "label": "all linked", "corpus": "c24", "e": pr.get("P2_S_after(c24)")}
    )
    out.append({"group": g3, "label": "7 natural", "corpus": "c9", "e": pr.get("P2_S_after(c9)")})
    out.append(
        {
            "group": g3,
            "label": "4 linked (conj. only)",
            "corpus": "c4",
            "e": pr.get("P2_S_after(c4)"),
        }
    )
    g4 = "Nov 18 markers M (step + conjunct order), − none linked"
    for c in ("c24", "c9", "c4"):
        out.append({"group": g4, "label": SHORT[c], "corpus": c, "e": pr.get(f"P1_M({c})")})
    if b1.get("M(c24)"):
        for c in ("c24", "c4"):
            out.append(
                {
                    "group": g4,
                    "label": f"{SHORT[c]}, 2nd base",
                    "corpus": c,
                    "e": b1.get(f"M({c})"),
                    "hollow": True,
                }
            )
            out.append(
                {
                    "group": g4,
                    "label": f"{SHORT[c]}, 2 bases",
                    "corpus": c,
                    "e": avg.get(f"M({c})"),
                    "bold": True,
                }
            )
    else:
        out.append(
            {"group": g4, "label": "2nd bases", "corpus": "c24", "e": None, "note": "pending"}
        )
    diff = avg.get("M(c24)-M(c4)") if avg else pr.get("P1_M(c24)-M(c4)")
    out.append(
        {
            "group": g4,
            "label": "all − 4 linked" + (", 2 bases" if avg else ""),
            "corpus": None,
            "e": diff,
        }
    )
    wa = _whole_author(cmp, (pr.get("P2_S_before(c24)") or {}).get("habits") or [])
    if wa is not None:
        out[[i for i, r in enumerate(out) if r["group"] == g2][0]]["whole_author"] = wa
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
    # y positions: a gap and a header line between groups
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
        lo_y = min(ys) - 0.6
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
            ax.errorbar(v[0], yy, xerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="none", ecolor=st["color"],
                        elinewidth=0.9, capsize=1.6, zorder=3)  # fmt: skip
            ax.plot(v[0], yy, linestyle="none", marker=st["marker"], markersize=4.4, zorder=4,
                    markerfacecolor=S.WHITE if r.get("hollow") else st["color"],
                    markeredgecolor=st["color"] if r.get("hollow") else S.WHITE, markeredgewidth=0.8)  # fmt: skip
            ax.annotate(f"{v[0]:+.1f} [{v[1]:+.1f}, {v[2]:+.1f}]", (max(v[2], 10.5), yy), xytext=(3, 0),
                        textcoords="offset points", va="center", fontsize=S.FS_TICK - 1, color=S.INK2)  # fmt: skip
            if r.get("whole_author") is not None:
                wa = r["whole_author"]
                ax.plot(wa, yy, linestyle="none", marker="D", markersize=4.4, markerfacecolor=S.WHITE,
                        markeredgecolor=S.INK, markeredgewidth=0.8, zorder=4)  # fmt: skip
                ax.annotate(f"a whole-author shift would give {wa:+.0f}", (wa, yy), xytext=(4, 5),
                            textcoords="offset points", ha="right", fontsize=S.FS_TICK - 1, color=S.INK)  # fmt: skip
        for g, yy in heads:
            ax.text(0.005, yy - 0.2, g, transform=ax.get_yaxis_transform(), ha="left", va="center",
                    fontsize=S.FS_TICK, fontstyle="italic", color=S.INK)  # fmt: skip
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_ylim(lo_y, 0.2)
        ax.set_xlim(-22, 75)
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
        ax.legend(handles=corpus_handles(["c24", "c9", "c4", "c0"]), loc="lower right", fontsize=S.FS_TICK - 0.5,
                  title="corpus (hollow: second base)", title_fontsize=S.FS_TICK - 0.5)  # fmt: skip
        return save(fig, Path(a.out), "fig2_dpo")


if __name__ == "__main__":
    main()
