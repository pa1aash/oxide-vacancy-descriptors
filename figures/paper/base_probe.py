"""Control-series styles and readers of the latent probe figure, used by figureH1_probe.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inputs_paper as ip
import style_paper as sp

STEM = "fig6_probe"
VERSION = "v1"
TRAINED = sp.COLORS["S"]
CONTROL_STYLE = {
    "shuffled": {"color": "#4D4D4D", "marker": "s", "band": True, "lw": sp.BASELINE_W, "ls": "-"},
    "init0": {"color": "#8C8C8C", "marker": "^", "band": True, "lw": sp.BASELINE_W, "ls": "-"},
    "init1": {"color": "#BDBDBD", "marker": "v", "band": False, "lw": 0.7, "ls": (0, (1.5, 1.5))},
    "init2": {"color": "#BDBDBD", "marker": "D", "band": False, "lw": 0.7, "ls": (0, (1.5, 1.5))},
}
LABELS = {
    "trained": "trained S encoder",
    "shuffled": "control: shuffled labels",
    "init0": "control: random-init encoder (init0)",
    "init1": "random-init spread (init1, init2; descriptive)",
}
CLASSES = (("host", "Host-electronic class-mean test $R^2$"), ("site", "Site-electronic class-mean test $R^2$"))


def curve(block: dict, budgets):
    rows = [block[str(b)] for b in budgets]
    return [r["mean"] for r in rows], [r["ci95"][0] for r in rows], [r["ci95"][1] for r in rows]


def control_series(ax, x, m, lo, hi, key: str) -> None:
    st = CONTROL_STYLE[key]
    if st["band"]:
        ax.fill_between(x, lo, hi, color=st["color"], alpha=sp.BAND_ALPHA, linewidth=0)
    ax.plot(x, m, color=st["color"], lw=st["lw"], ls=st["ls"], marker=st["marker"], ms=sp.MARKER_SIZE * 0.85,
            mfc=st["color"], mec=st["color"])


def control_handle(key: str, label: str):
    from matplotlib.lines import Line2D

    st = CONTROL_STYLE[key]
    return Line2D([], [], color=st["color"], lw=st["lw"], ls=st["ls"], marker=st["marker"],
                  ms=sp.MARKER_SIZE * 0.85, mfc=st["color"], mec=st["color"], label=label)


def trained_handle():
    from matplotlib.lines import Line2D

    return Line2D([], [], color=TRAINED, lw=sp.LINE_W, ls=sp.BACKBONE_LS[VERSION], marker=sp.MARKER,
                  ms=sp.MARKER_SIZE, mfc="white", mec=TRAINED, mew=0.8, label=LABELS["trained"])


