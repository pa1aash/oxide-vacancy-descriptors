"""Figure F1: the advantage A without high-moment entries, and an untuned CGCNN beside the network.

Inputs (read only): results/derived_n4/high_moment_final_network.json and results/sens_cgcnn.json.

    (a) A at 654 training hosts on all test sites, on the common test sites, and on the common test sites with the
        high-moment entries removed; two-sided intervals, one-sided upper bounds, +-delta
    (b) test MAE of S and D of the network and of CGCNN on one resample

    python figures/paper/figureF1_sensitivities.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import network_style as ns
import style_paper as sp
from matplotlib.lines import Line2D

SUB, STEM = "si", "figureF1_sensitivities"
SERIES = ("primary_full_sites", "primary_common_sites", "high_moment")
TICKS = ("all\nsites", "common\nsites", "common sites,\nentries\nremoved")
CGCNN_MARKER, CGCNN_DODGE, UPPER_HALF_WIDTH = "X", 1.08, 0.12


def draw_a(ax, hm: dict) -> None:
    ns.delta_lines(ax, hm["delta_eV"])
    ns.zero_line(ax)
    c = sp.COLORS["A"]
    for i, key in enumerate(SERIES):
        b = hm["final_network"][key]["A"]
        lo, hi = b["ci"]
        ax.errorbar([i], [b["mean"]], yerr=[[b["mean"] - lo], [hi - b["mean"]]], fmt=sp.MARKER, ms=sp.MARKER_SIZE + 0.6,
                    color=c, ecolor=c, elinewidth=sp.REF_W + 0.2, capsize=0, mew=0.8, mfc=c, mec=c, zorder=5)
        u = b["upper_one_sided"]
        ax.plot([i - UPPER_HALF_WIDTH, i + UPPER_HALF_WIDTH], [u, u], color=c, lw=sp.LINE_W, solid_capstyle="butt")
    ax.set_xticks(range(len(SERIES)))
    ax.set_xticklabels(TICKS)
    ax.set_xlim(-0.6, len(SERIES) - 0.4)
    ax.set_ylim(-0.065, 0.075)
    ax.set_ylabel(r"$A$ with 654 training hosts (eV)")


def draw_b(ax, cg: dict) -> None:
    budgets = cg["budgets"]
    for m in ("S", "D"):
        rows = [cg["table"]["megnet_v2"][str(B)][m]["mae"] for B in budgets]
        ns.curve(ax, budgets, [r["point"] for r in rows], [r["ci95"][0] for r in rows], [r["ci95"][1] for r in rows],
                 sp.COLORS[m])
        rows = [cg["table"]["cgcnn"][str(B)][m]["mae"] for B in budgets]
        y, x = [r["point"] for r in rows], [B * CGCNN_DODGE for B in budgets]
        ax.errorbar(x, y, yerr=[[a - r["ci95"][0] for a, r in zip(y, rows)], [r["ci95"][1] - a for a, r in zip(y, rows)]],
                    fmt=CGCNN_MARKER, ms=sp.EXT_MARKER_SIZE + 0.6, color=sp.COLORS[m], ecolor=sp.COLORS[m],
                    elinewidth=sp.REF_W, capsize=0, ls="none", mew=0.6, mec=sp.COLORS[m], zorder=6)
    ax.set_xscale("log")
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.minorticks_off()
    ax.set_xlim(budgets[0] / 1.35, budgets[-1] * 1.35)
    ax.set_xlabel("Training hosts")
    ax.set_ylabel("Test MAE (eV)")


def main() -> Path:
    hm = ns.results("derived_n4/high_moment_final_network")
    cg = ns.results("sens_cgcnn")
    fig, axes = sp.figure("single", height_mm=65.0, ncols=2, nrows=1)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.94, bottom=0.25, wspace=0.32)
    draw_a(axes[0], hm)
    draw_b(axes[1], cg)
    handles = [sp.model_handle("S", "S (structure only)"), sp.model_handle("D", "D (structure + descriptors)"),
               Line2D([], [], color="#000000", ls="none", marker=CGCNN_MARKER, ms=sp.EXT_MARKER_SIZE + 0.6,
                      label="CGCNN, untuned"),
               Line2D([], [], color=sp.COLORS["A"], ls="none", marker="_", ms=6, label="one-sided 95% upper bound"),
               sp.reference_handle(r"$\pm\delta$", "delta"), sp.reference_handle("zero", "zero")]
    sp.legend_handles(fig, handles, ncol=3)
    sp.panel_letters(fig, axes)
    pdf = sp.save(fig, ns.out_path(SUB, STEM))
    return pdf


if __name__ == "__main__":
    print(main())
