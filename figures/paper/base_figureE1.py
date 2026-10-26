"""Panel drawing of Figure E1 (within-host residual MAE by group), used by
figureE1_within_host_groups.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import base_within_host_groups as old
import network_style as ns
import style_paper as sp
from matplotlib.lines import Line2D

SUB, STEM = "si", "figureE1_within_host_groups"
METRIC = "within_host_mae"
OFFSET = {"S": -0.14, "D": 0.14}


def draw_panel(ax, rows: list[dict], labels: list[str], x_max: float) -> None:
    for i, g in enumerate(rows):
        net = g[ns.NETWORK]
        zs = net["S"]["zero_skill_mae"]["point"]
        if zs is not None:
            ax.plot([zs, zs], [i - 0.4, i + 0.4], color=old.ZERO_SKILL_COLOR, lw=1.6, solid_capstyle="butt",
                    zorder=1)
        for model in ("S", "D"):
            e = net[model][METRIC]
            if e["point"] is None:
                continue
            y, c = i + OFFSET[model], sp.COLORS[model]
            if "ci95" in e:
                ax.plot(e["ci95"], [y, y], color=c, lw=0.9, ls="-", zorder=2)
            ax.plot([e["point"]], [y], ls="none", marker=sp.MARKER, ms=sp.MARKER_SIZE, mec=c, mfc=c, mew=0.8,
                    zorder=3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(0.0, x_max)
    ax.tick_params(axis="y", length=0)


def x_upper(groups: list[dict]) -> float:
    hi = 0.0
    for g in groups:
        net = g[ns.NETWORK]
        for model in ("S", "D"):
            e = net[model][METRIC]
            hi = max(hi, *(e.get("ci95") or []), e["point"] or 0.0)
        hi = max(hi, net["S"]["zero_skill_mae"]["point"] or 0.0)
    return 0.1 * int(hi / 0.1 + 1)


