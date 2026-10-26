"""Figure 2: site error and descriptor advantage against training hosts, with D*.

Inputs (read only): results/derived_c1/final_network_summary.json (``table.rows``: S, D, RF-Kumagai, B0 and
A = MAE_S - MAE_D) and results/derived_dstar/dstar_gaps.json (``sweep_10x3.<B>.metrics.mae`` and
``headline_654_30x5.654``: D*, the descriptor network after the wider hyperparameter search, and A* = MAE_S - MAE_D*).

    (a) test MAE of S, D (25 to 654 training hosts; additional resamples at 400 and 654 as diamonds), D* (200,
        400, 654), RF-Kumagai and B0; two-sided 95% bands
    (b) A from 100 training hosts (curve, additional resamples, grouped cross-validation) and A*; two-sided 95%
        intervals, one-sided 95% upper bounds (bars) and the tolerance +-delta

D* and every difference against it are drawn in one hue with a dash-dot line, open markers and an outlined band.

    python figures/paper/figure2_descriptor_advantage.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import base_figure2 as f2
import network_style as ns
import style_paper as sp
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D

SUB, STEM = "main", "figure2_descriptor_advantage"
DSTAR_RECORD = ROOT / "results" / "derived_dstar" / "dstar_gaps.json"
DSTAR_BUDGETS = ("200", "400", "654")
DSTAR_COLOR = "#D55E00"
DSTAR_DASHES = (5.0, 1.5, 1.5, 1.5)
DSTAR_LABEL = r"D$^{*}$ (wider search)"
DSTAR_BAND_W = 0.5
DSTAR_BAND_ALPHA = 0.7
DESIGN_LABELS = {"curve": "learning-curve design", "pooled": "additional resamples", "cv": "grouped cross-validation"}
Y_TOP = 0.9
A_LABEL = r"$A$ (S $-$ D)"
ASTAR_LABEL = r"$A^{*}$ (S $-$ D$^{*}$)"


def dstar_payload() -> dict:
    return json.loads(DSTAR_RECORD.read_text())["payload"]


def stat(row: dict) -> tuple[float, float, float, float | None]:
    """(point, lo, hi, one-sided upper) of a record row with ``mean`` or ``A``, ``ci95``."""
    point = row["mean"] if "mean" in row else row["A"]
    return point, row["ci95"][0], row["ci95"][1], row.get("upper_one_sided_95")


def sweep(payload: dict, getter):
    """(x, point, lo, hi, upper) over the 10 x 3 sweep at 200, 400, 654 training hosts."""
    rows = [stat(getter(payload["sweep_10x3"][b])) for b in DSTAR_BUDGETS]
    return ([int(b) for b in DSTAR_BUDGETS], *([r[i] for r in rows] for i in range(4)))


def headline(payload: dict, getter):
    """(x, point, lo, hi, upper) of the 30 x 5 point at 654 training hosts."""
    r = stat(getter(payload["headline_654_30x5"]["654"]))
    return [654], [r[0]], [r[1]], [r[2]], [r[3]]


def dstar_band(ax, x, lo, hi, color: str) -> None:
    """The two-sided band of a D* series as two thin edges without fill (no darker overlap with filled bands)."""
    for edge in (lo, hi):
        ax.plot(x, edge, color=color, lw=DSTAR_BAND_W, ls="-", alpha=DSTAR_BAND_ALPHA, zorder=2)


def dstar_curve(ax, x, mean, lo, hi, color: str) -> None:
    dstar_band(ax, x, lo, hi, color)
    ax.plot(x, mean, color=color, lw=sp.LINE_W, dashes=DSTAR_DASHES, marker=sp.MARKER, ms=sp.MARKER_SIZE,
            mfc="white", mec=color, mew=0.8, zorder=4)


def dstar_points(ax, x, mean, lo, hi, color: str) -> list[float]:
    xs = [v * ns.POOLED_DODGE for v in x]
    yerr = [[m - a for m, a in zip(mean, lo)], [b - m for m, b in zip(mean, hi)]]
    eb = ax.errorbar(xs, mean, yerr=yerr, fmt=ns.MARKERS["pooled"], ms=sp.EXT_MARKER_SIZE, color=color, ecolor=color,
                     elinewidth=sp.REF_W, capsize=0, mfc="white", mec=color, mew=0.8, ls="none", zorder=5)
    eb[2][0].set_linestyle((0, (3.0, 1.5)))
    return xs


class OutlinedHandle(Line2D):
    """Legend proxy of a D* series: dash-dot line, open marker and the two edges of its outlined band."""


class HandlerOutlined(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        c, y = orig_handle.get_color(), ydescent + height / 2
        xs = [xdescent, xdescent + width]
        edges = [Line2D(xs, [y + d, y + d], color=c, lw=DSTAR_BAND_W, alpha=DSTAR_BAND_ALPHA, transform=trans)
                 for d in (-0.32 * height, 0.32 * height)]
        line = Line2D(xs, [y, y], color=c, lw=sp.LINE_W, dashes=DSTAR_DASHES, transform=trans)
        mark = Line2D([xdescent + width / 2], [y], ls="none", marker=sp.MARKER, ms=sp.MARKER_SIZE, mfc="white",
                      mec=c, mew=0.8, transform=trans)
        return [*edges, line, mark]


HANDLER_MAP = {OutlinedHandle: HandlerOutlined()}


def dstar_handle(color: str, label: str):
    return OutlinedHandle([], [], color=color, label=label)


def design_handles() -> list:
    """Neutral marker-only proxies of the designs (filled; the D* series use the same shapes, open)."""
    black = sp.COLORS["A"]
    size = {"curve": sp.MARKER_SIZE}
    return [Line2D([], [], color=black, ls="none", marker=ns.MARKERS[k], ms=size.get(k, sp.EXT_MARKER_SIZE),
                   mfc=black, mec=black, label=DESIGN_LABELS[k]) for k in ("curve", "pooled", "cv")]


def column_legend(fig, columns: list[list]) -> None:
    """One figure legend whose columns are the given groups (matplotlib fills a legend column by column); shorter
    columns are padded with invisible, unlabelled entries."""
    rows = max(len(c) for c in columns)
    def blank():
        return Line2D([], [], ls="none", marker=None, label=" ")

    sp.legend_handles(fig, [h for c in columns for h in c + [blank() for _ in range(rows - len(c))]],
                      ncol=len(columns), handler_map=HANDLER_MAP)


def main() -> Path:
    table = ns.payload(ns.SUMMARY)["table"]
    rows, delta = table["rows"], table["delta_eV"]
    dp = dstar_payload()
    assert abs(dp["delta_eV"] - delta) < 1e-12

    fig, (ax_a, ax_b) = sp.figure("single", height_mm=75.0, ncols=2)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.93, bottom=0.30, wspace=0.32)

    tops = []
    for key, col in (("B0", "MAE_B0"), ("RF-Kumagai", "MAE_RF")):
        x, m, lo, hi = f2.mae(rows, f2.CURVE, col)
        sp.baseline_curve(ax_a, x, m, lo, hi, key)
        tops += m
    for key, col in (("S", "MAE_S"), ("D", "MAE_D")):
        x, m, lo, hi = f2.mae(rows, f2.CURVE, col)
        ns.curve(ax_a, x, m, lo, hi, sp.COLORS[key])
        tops += hi
        x, m, lo, hi = f2.mae(rows, f2.POOLED, col)
        ns.points(ax_a, x, m, lo, hi, sp.COLORS[key], "pooled")
    green = DSTAR_COLOR
    x, m, lo, hi, _ = sweep(dp, lambda b: b["metrics"]["mae"]["MAE"]["Dstar"])
    dstar_curve(ax_a, x, m, lo, hi, green)
    x, m, lo, hi, _ = headline(dp, lambda b: b["metrics"]["mae"]["MAE"]["Dstar"])
    dstar_points(ax_a, x, m, lo, hi, green)
    floor = min(lo)
    sp.budget_axis(ax_a)
    ax_a.set_ylim(min(0.2, floor - 0.04), Y_TOP)
    ax_a.set_ylabel("Test MAE (eV)")

    black = sp.COLORS["A"]
    ns.delta_lines(ax_b, delta)
    ns.zero_line(ax_b)
    span = [delta, -delta]
    curve_keys = [k for k in f2.CURVE if rows[k]["x_training_hosts"] >= f2.A_FROM]
    x, m, lo, hi, ub = f2.adv(rows, curve_keys)
    ns.curve(ax_b, x, m, lo, hi, black)
    ns.upper_bounds(ax_b, x, ub)
    span += lo + hi + ub
    for keys, kind in ((f2.POOLED, "pooled"), (f2.CV, "cv")):
        x, m, lo, hi, ub = f2.adv(rows, keys)
        xs = ns.points(ax_b, x, m, lo, hi, black, kind)
        ns.upper_bounds(ax_b, xs, ub)
        span += lo + hi + ub
    x, m, lo, hi, ub = sweep(dp, lambda b: b["metrics"]["mae"]["gaps"]["S_minus_Dstar"])
    dstar_curve(ax_b, x, m, lo, hi, DSTAR_COLOR)
    ns.upper_bounds(ax_b, x, ub, DSTAR_COLOR)
    span += lo + hi + ub
    x, m, lo, hi, ub = headline(dp, lambda b: b["metrics"]["mae"]["gaps"]["S_minus_Dstar"])
    xs = dstar_points(ax_b, x, m, lo, hi, DSTAR_COLOR)
    ns.upper_bounds(ax_b, xs, ub, DSTAR_COLOR)
    span += lo + hi + ub
    sp.budget_axis(ax_b, x_max=max(rows[k]["x_training_hosts"] for k in f2.CV), x_min=f2.A_FROM / 1.15)
    ax_b.set_ylim(*ns.limits(span))
    ax_b.set_ylabel("S minus descriptor network (eV)")

    column_legend(fig, [
        [sp.model_handle("S", "S (structure only)"), sp.model_handle("D", "D (structure + descriptors)"),
         dstar_handle(green, DSTAR_LABEL), sp.baseline_handle("RF-Kumagai"), sp.baseline_handle("B0")],
        [ns.line_handle(black, A_LABEL), dstar_handle(DSTAR_COLOR, ASTAR_LABEL), ns.upper_handle(), ns.delta_handle(),
         ns.zero_handle()],
        design_handles()])
    sp.panel_letters(fig, (ax_a, ax_b))
    pdf = sp.save(fig, ns.out_path(SUB, STEM))
    return pdf


if __name__ == "__main__":
    print(main())
