"""Figure E1: within-host residual MAE of S and D by chemistry and site.

Input (read only): results/within_host_breakdown.json (``groups[].v2.<S|D>.within_host_mae``, ``zero_skill_mae``).
Panels: (a) cation element, (b) shallow-donor-prone cations, (c) host stability, (d) band gap, (e) within-host rank
of the vacancy-oxygen Bader charge; a grey tick marks the zero-skill value of a host-constant predictor.

    python figures/paper/figureE1_within_host_groups.py [--png PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import base_within_host_groups as old
import base_figureE1 as e1
import network_style as ns
import style_paper as sp
from matplotlib.lines import Line2D

HEIGHT_MM = 100.0
XLABEL = "Within-host residual MAE (eV)"
YLABELS = {"phs_prone": "Cation\ngroup", "stability_tercile": "Formation energy\n(eV/atom)",
           "band_gap_tercile": "Band gap\n(eV)", "bader_rank": "Bader charge\nof O ($e$)"}
RIGHT = tuple(v for v in old.RIGHT if v != "d_class")


def main(png: Path | None = None) -> Path:
    groups = ns.results("within_host_breakdown")["groups"]
    x_max = e1.x_upper(groups)

    fig, _ = sp.figure("single", height_mm=HEIGHT_MM)
    fig.clf()
    right_rows = [old.rows_for(groups, v) for v in RIGHT]
    cat = old.rows_for(groups, "cation")
    half = (len(cat) + 1) // 2
    ys = dict(bottom=0.165, top=0.955)
    ax_cat = fig.add_subplot(fig.add_gridspec(1, 1, left=0.090, right=0.270, **ys)[0, 0])
    ax_cat2 = fig.add_subplot(fig.add_gridspec(1, 1, left=0.335, right=0.515, **ys)[0, 0])
    for ax, rows in ((ax_cat, cat[:half]), (ax_cat2, cat[half:])):
        e1.draw_panel(ax, rows, [g["group"] for g in rows], x_max)
    ax_cat.set_ylabel("Cation element")
    ax_cat2.set_xlabel(XLABEL, loc="right")
    gs = fig.add_gridspec(len(RIGHT), 1, height_ratios=[len(r) + 0.5 for r in right_rows],
                          left=0.80, right=0.985, hspace=0.95, **ys)
    axes_right = []
    for j, (v, rows) in enumerate(zip(RIGHT, right_rows)):
        ax = fig.add_subplot(gs[j, 0])
        e1.draw_panel(ax, rows, [old.TICKS[v][g["group"]] for g in rows], x_max)
        ax.set_xlabel(XLABEL, loc="right")
        ax.set_ylabel(YLABELS[v])
        axes_right.append(ax)

    handles = [sp.model_handle("S", "S (structure only)"), sp.model_handle("D", "D (structure + descriptors)"),
               Line2D([], [], color=old.ZERO_SKILL_COLOR, lw=1.6, label="zero skill (host-constant predictor)")]
    sp.legend_handles(fig, handles, ncol=3)
    sp.panel_letters(fig, [ax_cat])
    for ax, letter in zip(axes_right, "bcde"):
        fig.text(0.535, ax.get_position().y1 + 0.012, f"({letter})", fontsize=9, fontweight="bold", va="bottom", ha="left")
    if png is not None:
        fig.savefig(png, dpi=200)
    pdf = sp.save(fig, ns.out_path(e1.SUB, e1.STEM))
    return pdf


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--png", type=Path, default=None, help="also write a PNG preview to this path")
    print(main(ap.parse_args().png))
