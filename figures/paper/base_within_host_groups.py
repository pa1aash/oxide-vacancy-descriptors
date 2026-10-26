"""Group rows and tick labels of the within-host breakdown
(``results/within_host_breakdown.json``), used by the Figure E1 scripts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inputs_paper as ip
import style_paper as sp
from matplotlib.lines import Line2D

STEM = "fig_si_withinhost_groups"
RESULT = "within_host_breakdown.json"
METRIC = "within_host_mae"
SERIES = (("S", "v2"), ("S", "v1"), ("D", "v2"), ("D", "v1"))
OFFSET = {k: o for k, o in zip(SERIES, (-0.27, -0.09, 0.09, 0.27))}
ZERO_SKILL_COLOR = "#7F7F7F"
TICKS = {
    "d_class": {"all": "all tested hosts", "d0": r"d$^{0}$", "d10": r"d$^{10}$", "other": "other",
                "unassigned": "unassigned"},
    "phs_prone": {"phs_prone": "Nb, W, Ti, V, Ta or Mo", "not_phs_prone": "none of these"},
    "stability_tercile": {"lower": "most stable third", "middle": "middle third", "upper": "least stable third"},
    "band_gap_tercile": {"lower": "smallest-gap third", "middle": "middle third", "upper": "largest-gap third"},
    "bader_rank": {"lowest": "most negative O", "middle": "intermediate O", "highest": "least negative O"},
}
RIGHT = ("d_class", "phs_prone", "stability_tercile", "band_gap_tercile", "bader_rank")


def rows_for(groups: list[dict], variable: str) -> list[dict]:
    rows = [g for g in groups if g["variable"] == variable]
    if variable == "d_class":
        rows = [g for g in groups if g["variable"] == "pooled"] + rows
    return rows


def draw_panel(ax, rows: list[dict], labels: list[str], x_max: float) -> None:
    for i, g in enumerate(rows):
        zs = g["v2"]["S"]["zero_skill_mae"]["point"]
        if zs is not None:
            ax.plot([zs, zs], [i - 0.4, i + 0.4], color=ZERO_SKILL_COLOR, lw=1.6, solid_capstyle="butt", zorder=1)
        for model, arm in SERIES:
            e = g[arm][model][METRIC]
            if e["point"] is None:
                continue
            y = i + OFFSET[(model, arm)]
            c = sp.COLORS[model]
            if "ci95" in e:
                lo, hi = e["ci95"]
                ax.plot([lo, hi], [y, y], color=c, lw=0.9, ls=sp.BACKBONE_LS[arm], zorder=2)
            ax.plot([e["point"]], [y], ls="none", marker=sp.MARKER, ms=sp.MARKER_SIZE, mec=c, mew=0.8,
                    mfc=c if sp.BACKBONE_FILLED[arm] else "white", zorder=3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(0.0, x_max)
    ax.tick_params(axis="y", length=0)


def x_upper(groups: list[dict]) -> float:
    hi = 0.0
    for g in groups:
        for model, arm in SERIES:
            e = g[arm][model][METRIC]
            hi = max(hi, *(e.get("ci95") or []), e["point"] or 0.0)
        hi = max(hi, g["v2"]["S"]["zero_skill_mae"]["point"] or 0.0)
    return 0.1 * int(hi / 0.1 + 1)


