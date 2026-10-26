"""Figure E1 (W2b; the former Figure S4 on the network alone): within-host residual MAE of S and D at 654 training
hosts, by chemistry and site.

Input (read-only): results/within_host_breakdown.json (``groups[].v2.<S|D>.within_host_mae``, ``point``,
``ci95``; zero-skill value ``groups[].v2.S.zero_skill_mae.point``). Panel layout and group labels are those of
figures/code/fig_si_withinhost_groups.py (imported, not edited).

    (a) cation element   (b) d-electron class, with all tested hosts as the top row
    (c) perturbed-host-state-prone chemistry   (d) host stability tercile   (e) PBEsol(+U) band gap tercile
    (f) within-host rank of the vacancy-O Bader charge

    python figures/code/figureE1_within_host_groups.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig_si_withinhost_groups as old  # read-only reuse: rows_for, TICKS, RIGHT, ZERO_SKILL_COLOR
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


def main() -> Path:
    groups = ns.results("within_host_breakdown")["groups"]
    x_max = x_upper(groups)

    fig, _ = sp.figure("single", height_mm=190.0)
    fig.clf()
    right_rows = [old.rows_for(groups, v) for v in old.RIGHT]
    gs = fig.add_gridspec(len(old.RIGHT), 2, width_ratios=[1.0, 1.0],
                          height_ratios=[len(r) + 1.2 for r in right_rows],
                          left=0.075, right=0.985, top=0.95, bottom=0.10, wspace=0.62, hspace=0.42)
    ax_cat = fig.add_subplot(gs[:, 0])
    cat = old.rows_for(groups, "cation")
    draw_panel(ax_cat, cat, [g["group"] for g in cat], x_max)
    ax_cat.set_xlabel("Within-host residual MAE (eV)")
    axes_right = []
    for j, (v, rows) in enumerate(zip(old.RIGHT, right_rows)):
        ax = fig.add_subplot(gs[j, 1])
        draw_panel(ax, rows, [old.TICKS[v][g["group"]] for g in rows], x_max)
        axes_right.append(ax)
    axes_right[-1].set_xlabel("Within-host residual MAE (eV)")

    handles = [sp.model_handle("S", "S (structure only)"), sp.model_handle("D", "D (structure + descriptors)"),
               Line2D([], [], color=old.ZERO_SKILL_COLOR, lw=1.6, label="zero skill (host-constant predictor)")]
    sp.legend_handles(fig, handles, ncol=3)
    sp.panel_letters(fig, [ax_cat, *axes_right])
    pdf = sp.save(fig, ns.out_path(SUB, STEM))
    ns.write_caption(SUB, STEM, CAPTION)
    return pdf


CAPTION = """
**Figure E1. Site resolution within a host, by chemistry and site.** Within-host residual MAE of S and D at the
largest training size for groups of held-out hosts or sites, with the zero-skill value of a host-constant
predictor as a grey tick, grouped by (**a**) cation, (**b**) d-electron class, (**c**) shallow-donor-prone cations,
(**d**) host stability, (**e**) band gap and (**f**) within-host rank of the vacancy-oxygen Bader charge. Bars are
two-sided host-cluster bootstrap intervals, values in `results/within_host_breakdown.json`.
"""


if __name__ == "__main__":
    print(main())
