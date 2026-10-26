"""One-network style for the rewritten figure set (W2b). Builds on ``style_paper`` without editing it.

The paper reports one network, so every network series is drawn one way: solid line, filled markers, filled
band (``style_paper`` style "v2"). Designs are told apart by marker shape only: circles on a line for the
learning-curve design (10 resamples x 3 seeds at every training size), diamonds for the pooled estimate over
30 resamples x 5 seeds, triangles for grouped cross-validation. The one-sided upper bound of a paired difference is
a short horizontal bar. Baselines (B0, RF-Kumagai) keep the thinner line with square markers of ``style_paper``.
No in-plot text: panel letters, axis labels, ticks and one legend below the panels; everything else goes in the
caption file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style_paper as sp
from matplotlib.lines import Line2D

ROOT = sp.ROOT
STYLE = "v2"            # style_paper key of the solid, filled style; the record key of the network
NETWORK = "v2"          # record key of the network in the results files (repository name)
SUMMARY = ROOT / "results" / "derived_c1" / "final_network_summary.json"
MARKERS = {"curve": sp.MARKER, "pooled": "D", "cv": "^"}
LEGEND = {"curve": "learning-curve design", "pooled": "30 resamples × 5 seeds", "cv": "grouped cross-validation"}
POOLED_DODGE = 0.90     # pooled points share their training size with the curve: drawn slightly left (log axis)
UB_MARKER = {"marker": "_", "ms": 5.0, "mew": 1.2, "ls": "none"}
DELTA_LABEL = r"$\pm\delta$"


def payload(path: Path) -> dict:
    return json.loads(Path(path).read_text())["payload"]


def results(name: str, root: Path | None = None) -> dict:
    return payload((Path(root) if root else ROOT / "results") / f"{name}.json")


def curve(ax, x, mean, lo, hi, color: str, zorder: float = 3) -> None:
    """A network series with its two-sided band."""
    sp.band(ax, x, lo, hi, color, STYLE)
    ax.plot(x, mean, color=color, lw=sp.LINE_W, ls="-", marker=sp.MARKER, ms=sp.MARKER_SIZE, mfc=color, mec=color,
            mew=0.8, zorder=zorder)


def points(ax, x, mean, lo, hi, color: str, kind: str) -> list[float]:
    """Unconnected estimates (``kind`` "pooled" or "cv") with two-sided interval bars; returns the drawn x."""
    xs = [v * POOLED_DODGE for v in x] if kind == "pooled" else list(x)
    yerr = [[m - a for m, a in zip(mean, lo)], [b - m for m, b in zip(mean, hi)]]
    ax.errorbar(xs, mean, yerr=yerr, fmt=MARKERS[kind], ms=sp.EXT_MARKER_SIZE, color=color, ecolor=color,
                elinewidth=sp.REF_W, capsize=0, mfc=color, mec=color, mew=0.8, ls="none", zorder=5)
    return xs


def upper_bounds(ax, x, ub, color: str = sp.COLORS["A"]) -> None:
    ax.plot(x, ub, color=color, zorder=6, **UB_MARKER)


def delta_lines(ax, delta: float) -> None:
    for y in (delta, -delta):
        ax.axhline(y, color=sp.COLORS["reference"], lw=sp.REF_W, dashes=sp.REF_DASHES["delta"], zorder=1)


def zero_line(ax, vertical: bool = False) -> None:
    kw = {"color": sp.COLORS["reference"], "lw": sp.REF_W, "dashes": sp.REF_DASHES["zero"], "zorder": 1}
    (ax.axvline if vertical else ax.axhline)(0.0, **kw)


def design_handle(kind: str, color: str = "#000000", prefix: str = ""):
    label = prefix + LEGEND[kind]
    if kind == "curve":
        return Line2D([], [], color=color, lw=sp.LINE_W, marker=sp.MARKER, ms=sp.MARKER_SIZE, mfc=color, mec=color,
                      label=label)
    return Line2D([], [], color=color, ls="none", marker=MARKERS[kind], ms=sp.EXT_MARKER_SIZE, mfc=color, mec=color,
                  label=label)


def line_handle(color: str, label: str):
    return Line2D([], [], color=color, lw=sp.LINE_W, marker=sp.MARKER, ms=sp.MARKER_SIZE, mfc=color, mec=color,
                  label=label)


def upper_handle(color: str = sp.COLORS["A"]):
    return Line2D([], [], color=color, label="one-sided 95% upper bound", **UB_MARKER)


def delta_handle(label: str = DELTA_LABEL):
    return sp.reference_handle(label, "delta")


def zero_handle():
    return sp.reference_handle("zero", "zero")


def limits(values, pad: float = 0.08) -> tuple[float, float]:
    lo, hi = min(values), max(values)
    return lo - pad * (hi - lo), hi + pad * (hi - lo)


def out_path(sub: str, stem: str) -> Path:
    return ROOT / "figures" / sub / stem


def write_caption(sub: str, stem: str, text: str) -> Path:
    import textwrap

    path = ROOT / "figures" / sub / f"{stem}.caption.md"
    path.write_text(textwrap.fill(" ".join(text.split()), width=115, break_long_words=False,
                                  break_on_hyphens=False) + "\n")
    return path
