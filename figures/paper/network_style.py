"""Series styles of the network figures, built on ``style_paper``.

Every network series is drawn one way: solid line, filled markers, filled band. Designs are told apart by marker
shape: circles on a line for the learning-curve design (10 resamples x 3 seeds at every training size), diamonds for
the additional resamples (30 resamples x 5 seeds), triangles for grouped cross-validation. A one-sided upper bound
is a short horizontal bar.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style_paper as sp
from matplotlib.lines import Line2D

ROOT = sp.ROOT
STYLE = "v2"
NETWORK = "v2"
SUMMARY = ROOT / "results" / "derived_c1" / "final_network_summary.json"
MARKERS = {"curve": sp.MARKER, "pooled": "D", "cv": "^"}
LEGEND = {"curve": "learning-curve design", "pooled": "30 resamples × 5 seeds", "cv": "grouped cross-validation"}
POOLED_DODGE = 0.90
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


