"""One figure style for the whole paper. Wraps ``dftgnn.viz.style`` without editing it.

Conventions, fixed for every plot in the paper:

* widths: MDPI text width 138.6 mm (``single``) and full page width 184.6 mm (``full``), from the class
  geometry recorded in; heights are free;
* one sans-serif family (Arial) for text and mathtext, embedded as TrueType (Type 42) in a vector PDF;
* one fixed colour per model (Okabe-Ito hues): B0 grey, RF-Kumagai orange, S blue, D bluish green,
  P reddish purple; the advantage A in black; published reference values in dark grey;
* backbone by line style: the primary backbone (v2) solid
  only in the legend (``BACKBONE_LABELS``) and the caption;
* extension series (grouped K-fold CV, precision extension) as unconnected points with their own marker
  the squares of the baselines), never as a third line style;
* one marker shape, line width and band alpha for every curve; bands are two-sided 95% intervals, filled for
* no in-plot text other than axis labels, tick labels and panel letters; the legend sits outside the data
  area (below the panels); numbers, sample sizes and definitions go in the caption file.

Check every output with ``figures/code/qa_figure.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from dftgnn.viz import style as base

MM = base.MM
WIDTHS_MM = {"single": 138.6, "full": 184.6}   # (mdpi.cls geometry)
WIDTHS_IN = {k: v * MM for k, v in WIDTHS_MM.items()}
BUDGETS = base.BUDGETS

FONT = "Arial"
COLORS = {
    "B0": base.COLORS["B0"],                  # grey      #7F7F7F
    "RF-Kumagai": base.COLORS["RF-Kumagai"],  # orange    #E69F00
    "S": base.COLORS["S"],                    # blue      #0072B2
    "D": base.COLORS["D"],                    # green     #009E73
    "P": "#CC79A7",                           # Okabe-Ito reddish purple
    "A": base.COLORS["A"],                    # black
    "reference": base.COLORS["reference"],    # dark grey
    "mean": base.COLORS["reference"],         # training-mean predictor, a reference curve (dotted)
}
LABELS = {**base.LABELS, "P": "P (staged: predicted descriptors into D)", "A": r"$A$ = MAE$_S$ $-$ MAE$_D$",
          "mean": "training mean"}
# Backbone styles. ``version`` arguments below name the backbone: "v2" (primary) or "v1".
BACKBONE_LS = {"v2": "-", "v1": (0, (4.0, 2.0))}
BACKBONE_FILLED = {"v2": True, "v1": False}
BACKBONE_LABELS = {"v2": "primary backbone (v2)", "v1": "originally registered backbone (v1)"}
BACKBONES = ("v2", "v1")      # drawing and legend order
EXT_MARKERS = {"precision": "D", "cv": "^"}
EXT_LABELS = {"precision": "precision (EXTENSION)", "cv": "K-fold CV (EXTENSION)"}
REGISTERED_LABEL = "registered outer resamples"
EXT_MARKER_SIZE = 3.6
EXT_DODGE = {("precision", "v2"): 0.90, ("precision", "v1"): 0.95}

LINE_W = base.LINE_W          # 1.2 pt model curves
REF_W = base.REF_W            # 0.7 pt reference lines
BAND_ALPHA = base.BAND_ALPHA  # 0.22 (filled bands: primary backbone and baselines)
BAND_HATCH = "//////"         # open bands: hatched, no fill
HATCH_W = 0.35
MARKER = "o"
BASELINE_MARKER = "s"         # baselines (no backbone): squares on a thinner line
BASELINE_W = 0.9
MARKER_SIZE = 3.0
PARITY_SIZE = 4.0             # scatter area (pt^2)
PARITY_ALPHA = 0.45
REF_DASHES = {"delta": (4, 2), "zero": (1, 2), "published": (4, 2), "published-alt": (1.5, 1.5)}


def apply() -> None:
    """RcParams plus a single Arial family for text and mathtext, TrueType embedding, exact widths."""
    base.apply()
    mpl.rcParams.update({
        "font.sans-serif": [FONT],
        "mathtext.fontset": "custom",
        "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic", "mathtext.bf": f"{FONT}:bold",
        "mathtext.sf": FONT, "mathtext.cal": FONT, "mathtext.fallback": None,
        "axes.unicode_minus": True,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": None, "savefig.transparent": False,
        "legend.handlelength": 2.6, "legend.columnspacing": 1.2, "hatch.linewidth": HATCH_W,
    })


def figure(width: str = "single", height_mm: float = 70.0, ncols: int = 1, nrows: int = 1, **kw):
    """A figure at an MDPI width; returns (fig, axes)."""
    apply()
    fig, axes = plt.subplots(nrows, ncols, figsize=(WIDTHS_IN[width], height_mm * MM), **kw)
    return fig, axes


def _line_kw(key: str, version: str) -> dict:
    c = COLORS[key]
    return {"color": c, "lw": LINE_W, "ls": BACKBONE_LS[version], "marker": MARKER, "ms": MARKER_SIZE,
            "mfc": c if BACKBONE_FILLED[version] else "white", "mec": c, "mew": 0.8}


def band(ax, x, lo, hi, color: str, version: str) -> None:
    if BACKBONE_FILLED[version]:
        ax.fill_between(x, lo, hi, color=color, alpha=BAND_ALPHA, linewidth=0)
    else:
        ax.fill_between(x, lo, hi, facecolor="none", edgecolor=color, hatch=BAND_HATCH, linewidth=0, alpha=0.6)


def band_curve(ax, x, mean, lo, hi, key: str, version: str = "v1", label: str | None = None) -> None:
    """A model curve with its two-sided 95% band, styled by backbone ``version``: the primary backbone's band is
    filled."""
    band(ax, x, lo, hi, COLORS[key], version)
    if label is None:
        label = LABELS.get(key, key)
    ax.plot(x, mean, label=label, **_line_kw(key, version))


def baseline_curve(ax, x, mean, lo, hi, key: str, dashes=None, label: str | None = None) -> None:
    """A backbone-independent baseline (B0, RF-Kumagai, training mean): thinner line, square markers, band."""
    c = COLORS[key]
    ax.fill_between(x, lo, hi, color=c, alpha=BAND_ALPHA, linewidth=0)
    kw = {"dashes": dashes} if dashes else {}
    ax.plot(x, mean, color=c, lw=BASELINE_W, marker=BASELINE_MARKER, ms=MARKER_SIZE * 0.85, mfc=c, mec=c,
            label=label or LABELS.get(key, key), **kw)


def baseline_handle(key: str, dashes=None, label: str | None = None):
    from matplotlib.lines import Line2D

    c = COLORS[key]
    kw = {"dashes": dashes} if dashes else {}
    return Line2D([], [], color=c, lw=BASELINE_W, marker=BASELINE_MARKER, ms=MARKER_SIZE * 0.85, mfc=c, mec=c,
                  label=label or LABELS.get(key, key), **kw)


def ext_x(x, kind: str, version: str) -> list[float]:
    """Drawn x positions of extension points (see ``EXT_DODGE``)."""
    f = EXT_DODGE.get((kind, version), 1.0)
    return [v * f for v in x]


def ext_points(ax, x, mean, lo, hi, key: str, kind: str, version: str) -> None:
    """Extension estimates (``kind`` "precision" or "cv") as unconnected points with 95% interval bars, styled by
    backbone ``version`` like the curves: the primary backbone solid and filled."""
    c = COLORS[key]
    yerr = [[m - a for m, a in zip(mean, lo)], [b - m for m, b in zip(mean, hi)]]
    eb = ax.errorbar(ext_x(x, kind, version), mean, yerr=yerr, fmt=EXT_MARKERS[kind], ms=EXT_MARKER_SIZE, color=c,
                     ecolor=c, elinewidth=REF_W, capsize=0, mfc=c if BACKBONE_FILLED[version] else "white", mec=c,
                     mew=0.8, ls="none", zorder=5)
    eb[2][0].set_linestyle(BACKBONE_LS[version])


def reference_line(ax, y: float, label: str, kind: str = "published", color: str | None = None) -> None:
    ax.axhline(y, color=color or COLORS["reference"], lw=REF_W, dashes=REF_DASHES[kind], label=label)


def budget_axis(ax, x_max: float | None = None, x_min: float | None = None) -> None:
    """Log x axis in training hosts, ticks at the tested budgets; widened when a series lies beyond them."""
    base.budget_axis(ax)
    lo, hi = ax.get_xlim()
    if x_max is not None and x_max * 1.12 > hi:
        hi = x_max * 1.12
    if x_min is not None:
        lo = x_min
        ax.set_xticks([b for b in BUDGETS if b >= x_min])
        ax.set_xticklabels([str(b) for b in BUDGETS if b >= x_min])
    ax.set_xlim(lo, hi)


def learning_curve_panel(ax, x, series: dict, version: str = "v1", refs=(), ylim=None,
                         ylabel: str = "Test MAE (eV)") -> None:
    """Test error against training hosts. ``series`` maps model key -> (mean, lo, hi);
    ``refs`` is a sequence of (y, label, kind, colour-or-None)."""
    for key, (m, lo, hi) in series.items():
        band_curve(ax, x, m, lo, hi, key, version)
    for y, label, kind, colour in refs:
        reference_line(ax, y, label, kind, colour)
    base.budget_axis(ax)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)


def advantage_panel(ax, x, mean, lo, hi, delta: float, version: str = "v1", key: str = "A",
                    label: str | None = None, ylabel: str = r"$A$ (eV)") -> None:
    """A = MAE_S - MAE_D (or another paired difference) with its band, the margin delta and zero."""
    band_curve(ax, x, mean, lo, hi, key, version, label=label or LABELS[key])
    reference_line(ax, delta, rf"$\delta$ = {delta:g} eV", "delta")
    reference_line(ax, 0.0, "zero", "zero")
    base.budget_axis(ax)
    ax.set_ylabel(ylabel)


def parity_panel(ax, y_true, y_pred, key: str, lims=None, label: str | None = None,
                 xlabel: str = r"DFT $E_\mathrm{f}$ (eV)", ylabel: str = r"Predicted $E_\mathrm{f}$ (eV)") -> None:
    """Predicted against DFT values with the y = x line; square, shared limits, no in-plot statistics."""
    import numpy as np

    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    if lims is None:
        lo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
        pad = 0.04 * (hi - lo)
        lims = (lo - pad, hi + pad)
    ax.plot(lims, lims, color=COLORS["reference"], lw=REF_W, dashes=REF_DASHES["published-alt"], label="y = x")
    ax.scatter(yt, yp, s=PARITY_SIZE, color=COLORS[key], alpha=PARITY_ALPHA, linewidths=0,
               label=label or LABELS.get(key, key))
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def model_handle(key: str, label: str | None = None):
    """Legend proxy for a model colour (solid line, filled marker; the backbone is shown separately)."""
    from matplotlib.lines import Line2D

    c = COLORS[key]
    return Line2D([], [], color=c, lw=LINE_W, marker=MARKER, ms=MARKER_SIZE, mfc=c, mec=c,
                  label=label or LABELS.get(key, key))


def backbone_handle(version: str, color: str = "#000000"):
    """Legend proxy for a backbone line style (neutral colour)."""
    from matplotlib.lines import Line2D

    return Line2D([], [], color=color, lw=LINE_W, ls=BACKBONE_LS[version], marker=MARKER, ms=MARKER_SIZE,
                  mfc=color if BACKBONE_FILLED[version] else "white", mec=color, mew=0.8,
                  label=BACKBONE_LABELS[version])


def ext_handle(kind: str, color: str = "#000000"):
    from matplotlib.lines import Line2D

    return Line2D([], [], color=color, ls="none", marker=EXT_MARKERS[kind], ms=EXT_MARKER_SIZE, mfc=color,
                  mec=color, label=EXT_LABELS[kind])


def registered_handle(color: str = "#000000"):
    """Registered handle."""
    from matplotlib.lines import Line2D

    return Line2D([], [], color=color, ls="none", marker=MARKER, ms=MARKER_SIZE, mfc=color, mec=color,
                  label=REGISTERED_LABEL)


def series_handles(kinds) -> list:
    """Series handles."""
    kinds = sorted(set(kinds))
    return [registered_handle(), *(ext_handle(k) for k in kinds)] if kinds else []


def reference_handle(label: str, kind: str = "published", color: str | None = None):
    from matplotlib.lines import Line2D

    return Line2D([], [], color=color or COLORS["reference"], lw=REF_W, dashes=REF_DASHES[kind], label=label)


def legend_handles(fig, handles, ncol: int = 3, y: float = 0.0, handler_map=None) -> None:
    """One figure legend from explicit proxy handles, below the panels (outside every data area); drops columns
    until it fits the page width. The rows added that way must still clear the axes (``check_legends``)."""
    for n in range(ncol, 0, -1):
        leg = fig.legend(handles=handles, loc="lower center", ncol=n, bbox_to_anchor=(0.5, y), handler_map=handler_map)
        fig.canvas.draw()
        bb = leg.get_window_extent()
        if bb.x0 >= fig.bbox.x0 and bb.x1 <= fig.bbox.x1:
            break
        leg.remove()
    else:
        raise ValueError("legend wider than the page in one column")
    # raise the axes bottom (fixed figure size) until tick and axis labels clear the legend by ~3 pt
    gap = 3.0 * fig.dpi / 72
    for _ in range(40):
        fig.canvas.draw()
        top = leg.get_window_extent().y1 + gap
        low = min(ax.get_tightbbox().y0 for ax in fig.axes)
        if low >= top:
            return
        fig.subplots_adjust(bottom=fig.subplotpars.bottom + (top - low) / fig.bbox.height + 0.002)
    raise ValueError("could not clear the legend below the axes")


def legend_below(fig, axes, ncol: int = 3, y: float = 0.0) -> None:
    """One figure legend below the panels (outside every data area), de-duplicated by label."""
    seen, hs, ls = set(), [], []
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in seen:
                seen.add(lab)
                hs.append(h)
                ls.append(lab)
    fig.legend(hs, ls, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, y))


def panel_letters(fig, axes, letters: str = "abcdefgh") -> None:
    for ax, letter in zip(axes, letters):
        base.panel_letter(fig, ax, letter)


def check_legends(fig, tol_px: float = 0.5) -> None:
    """Raise if a figure legend reaches outside the page or into an axes (qa_figure.py cannot see clipping)."""
    fig.canvas.draw()
    page = fig.bbox
    for leg in fig.legends:
        bb = leg.get_window_extent()
        if bb.x0 < page.x0 - tol_px or bb.x1 > page.x1 + tol_px or bb.y0 < page.y0 - tol_px or bb.y1 > page.y1 + tol_px:
            raise ValueError(f"legend extends outside the page: {bb.bounds} vs {page.bounds}")
        for ax in fig.axes:
            tight = ax.get_tightbbox()
            if tight is not None and bb.overlaps(tight):
                raise ValueError(f"legend overlaps the axes area or its labels: {bb.bounds} vs {tight.bounds}")


def save(fig, stem, png: bool = False) -> Path:
    """Vector PDF (and optionally a 300 dpi PNG preview); returns the PDF path."""
    stem = Path(stem)
    check_legends(fig)
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf = stem.with_suffix(".pdf")
    fig.savefig(pdf, metadata={"CreationDate": None})  # byte-stable output
    if png:
        fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)
    return pdf
