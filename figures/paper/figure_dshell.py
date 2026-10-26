"""Figure: within-oxide gain of D* by the d-shell filling of the host cations.

Hosts are grouped by the d count of their d-block cations (results/derived_dshell/dshell_classes.json): d0, d10
(transition-metal cations only), partial-d, main-group and a mixed/f-block bin. Plotted: within-host residual MAE of S
minus that of D* at 654 training hosts (30 resamples x 5 seeds), point and two-sided 95% interval, from
results/derived_dshell/dshell_gain.json (``groups.<class>.gains.S_minus_Dstar.within_host_mae``), with all hosts
as the last row.

    python figures/paper/figure_dshell.py [--png PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure2_descriptor_advantage as f2s
import network_style as ns
import style_paper as sp
from matplotlib.lines import Line2D

SUB, STEM = "main", "figure_dshell"
RECORD = ROOT / "results" / "derived_dshell" / "dshell_gain.json"
GAIN, METRIC = "S_minus_Dstar", "within_host_mae"
CLASSES = (("d0", r"$d^{0}$"), ("d10", r"$d^{10}$"), ("partial_d", r"partial-$d$"), ("main_group", "main-group"),
           ("mixed_f", r"mixed/$f$-block"))
ALL = ("all_hosts", "all hosts")
GAP = 0.6
SEPARATOR = "#D9D9D9"
BAR_DASHES = (0, (3.0, 1.5))
MS = sp.EXT_MARKER_SIZE + 0.8
LABEL = r"$A^{*}_\mathrm{within}$ (S $-$ D$^{*}$)"


def gain(groups: dict, key: str) -> tuple[float, float, float]:
    row = groups[key]["gains"][GAIN][METRIC]
    assert row["n_replicates_undefined"] == 0, key
    return row["point"], row["ci95"][0], row["ci95"][1]


def main(png: Path | None = None) -> Path:
    groups = ns.payload(RECORD)["groups"]
    keys = [k for k, _ in CLASSES] + [ALL[0]]
    rows = [gain(groups, k) for k in keys]
    y = list(range(len(CLASSES))) + [len(CLASSES) + GAP]
    black = f2s.DSTAR_COLOR

    fig, ax = sp.figure("single", height_mm=66.0)
    fig.subplots_adjust(left=0.20, right=0.97, top=0.97, bottom=0.30)
    ns.zero_line(ax, vertical=True)
    ax.axhline(len(CLASSES) - 0.5 + GAP / 2, color=SEPARATOR, lw=0.5, ls="-", zorder=1)
    xerr = [[r[0] - r[1] for r in rows], [r[2] - r[0] for r in rows]]
    eb = ax.errorbar([r[0] for r in rows], y, xerr=xerr, fmt=ns.MARKERS["pooled"], ms=MS, color=black, ecolor=black,
                     elinewidth=sp.REF_W, capsize=0, mfc="white", mec=black, mew=0.9, ls="none", zorder=5)
    eb[2][0].set_linestyle(BAR_DASHES)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab in CLASSES] + [ALL[1]])
    ax.set_ylim(y[-1] + 0.5, -0.5)
    lo, hi = ns.limits([r[1] for r in rows] + [r[2] for r in rows] + [0.0], pad=0.06)
    ax.set_xlim(lo, hi)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(r"S $-$ D$^{*}$ within-host residual MAE (eV)")
    ax.set_ylabel(r"Cation $d$-shell class")

    diamond = Line2D([], [], color=black, lw=sp.REF_W, ls=BAR_DASHES, marker=ns.MARKERS["pooled"], ms=MS,
                     mfc="white", mec=black, mew=0.9, label=LABEL)
    sp.legend_handles(fig, [diamond, ns.zero_handle()], ncol=2)
    if png is not None:
        fig.savefig(png, dpi=300)
    pdf = sp.save(fig, ns.out_path(SUB, STEM))
    return pdf


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--png", type=Path, default=None, help="also write a 300 dpi PNG preview to this path")
    print(main(ap.parse_args().png))
