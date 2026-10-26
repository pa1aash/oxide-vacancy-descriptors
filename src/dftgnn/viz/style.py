"""Minimal figure style.

Rules: vector PDF at the MDPI text width; one sans-serif family; fixed colour-blind-safe model colours
(Okabe-Ito hues); lines with shaded two-sided 95% bands; learning curves on a log x axis in training hosts. No
in-plot text other than axis labels, tick labels, panel letters and a legend outside the data area.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

MM = 1 / 25.4
WIDTH_1COL_IN = 138.6 * MM          # MDPI \textwidth
BUDGETS = (25, 50, 100, 200, 400, 654)

COLORS = {
    "B0": "#7F7F7F",          # grey
    "RF-Kumagai": "#E69F00",  # orange
    "S": "#0072B2",           # blue
    "D": "#009E73",           # bluish green
    "A": "#000000",
    "reference": "#4D4D4D",
}
LABELS = {"B0": "B0 physics floor", "RF-Kumagai": "RF-Kumagai", "S": "S (structure only)",
          "D": "D (structure + DFT descriptors)"}
BAND_ALPHA = 0.22
LINE_W = 1.2
REF_W = 0.7


def apply() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
        "legend.frameon": False, "axes.titlesize": 8,
    })


def budget_axis(ax) -> None:
    """Log x axis in training hosts with ticks at the tested budgets."""
    ax.set_xscale("log")
    ax.set_xticks(BUDGETS)
    ax.set_xticklabels([str(b) for b in BUDGETS])
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_xlim(BUDGETS[0] / 1.15, BUDGETS[-1] * 1.15)
    ax.set_xlabel("Training hosts")


def curve(ax, x, mean, lo, hi, key: str, label: str | None = None) -> None:
    c = COLORS[key]
    ax.fill_between(x, lo, hi, color=c, alpha=BAND_ALPHA, linewidth=0)
    ax.plot(x, mean, color=c, lw=LINE_W, marker="o", ms=3, label=label if label is not None else LABELS.get(key, key))


def hline(ax, y, label: str, dashes=(4, 2), color: str | None = None) -> None:
    ax.axhline(y, color=color or COLORS["reference"], lw=REF_W, dashes=dashes, label=label)


def panel_letter(fig, ax, letter: str) -> None:
    """Panel letter just outside the top-left corner of the axes."""
    bb = ax.get_position()
    fig.text(bb.x0 - 0.075, bb.y1 + 0.02, f"({letter})", fontsize=9, fontweight="bold", va="bottom", ha="left")


def save(fig, stem) -> None:
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.png", dpi=300)
    plt.close(fig)
