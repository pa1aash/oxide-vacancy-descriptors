"""Figure 5: out-of-sample vacancy formation energies of named oxides.

Input (read only): figures/main/data/screening_demo.json (per-site DFT energy and the prediction of S over the
resamples in which the host was held out).

    (a) prediction of S against the DFT formation energy, marker shape and colour by oxide
    (b) predicted minus DFT energy per oxygen site; marker the mean, bar the smallest to largest prediction

    python figures/paper/figure5_screening.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import network_style as ns
import style_paper as sp
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

SUB, STEM = "main", "figure5_screening"
RECORD = ns.ROOT / "figures" / "main" / "data" / "screening_demo.json"
OXIDES = ("ZnO", "SnO2", "Ga2O3", "TiO2", "ZrO2", "MgO", "Al2O3")
OXIDE_SHAPES = {"ZnO": "o", "SnO2": "s", "Ga2O3": "^", "TiO2": "D", "ZrO2": "v", "MgO": "P", "Al2O3": "X"}
OXIDE_TEX = {"ZnO": "ZnO", "SnO2": r"SnO$_2$", "Ga2O3": r"Ga$_2$O$_3$", "TiO2": r"TiO$_2$", "ZrO2": r"ZrO$_2$",
             "MgO": "MgO", "Al2O3": r"Al$_2$O$_3$"}
CAP = 1.6
MS = 4.2


def draw(ax, x, y, lo, hi, color: str, shape: str) -> None:
    """One site: filled marker with a min-max bar (none when the host was held out once)."""
    if hi > lo:
        ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="none", ecolor=color, elinewidth=0.9, capsize=CAP,
                    capthick=0.9, zorder=3)
    ax.plot([x], [y], ls="none", marker=shape, ms=MS, mfc=color, mec=color, mew=0.9, zorder=4)


def main() -> Path:
    rec = json.loads(RECORD.read_text())["payload"]
    sites, idx = rec["sites"], rec["site_index"]
    net = ns.NETWORK

    mean_dft = {c: np.mean([s["dft_eV"] for k, s in sites.items() if idx[k]["candidate"] == c]) for c in OXIDES}
    rank = sorted(OXIDES, key=mean_dft.get)
    cmap = plt.get_cmap("viridis")
    OXIDE_COLORS = {c: mpl.colors.to_hex(cmap(0.02 + 0.80 * i / (len(rank) - 1))) for i, c in enumerate(rank)}
    order = sorted(sites, key=lambda k: (mean_dft[idx[k]["candidate"]], idx[k]["label"]))

    fig, (a1, a2) = sp.figure("single", height_mm=70.0, ncols=2, gridspec_kw={"width_ratios": [1.0, 1.1]})
    fig.subplots_adjust(left=0.085, right=0.99, top=0.93, bottom=0.36, wspace=0.30)

    lims = (4.3, 8.0)
    a1.plot(lims, lims, color=sp.COLORS["reference"], lw=sp.REF_W, dashes=sp.REF_DASHES["published"], zorder=1)
    for k in order:
        s, c = sites[k], OXIDE_COLORS[idx[k]["candidate"]]
        draw(a1, s["dft_eV"], s[net]["pred_eV"], s[net]["min_eV"], s[net]["max_eV"], c, OXIDE_SHAPES[idx[k]["candidate"]])
    a1.set_xlim(*lims)
    a1.set_ylim(*lims)
    a1.set_aspect("equal", adjustable="box")
    a1.set_xlabel(r"DFT $E_\mathrm{f}$ (eV)")
    a1.set_ylabel(r"Predicted $E_\mathrm{f}$ (eV)")

    ns.zero_line(a2)
    for i, k in enumerate(order):
        s, c = sites[k], OXIDE_COLORS[idx[k]["candidate"]]
        e = s[net]["signed_err_eV"]
        draw(a2, i, e, e - (s[net]["pred_eV"] - s[net]["min_eV"]), e + (s[net]["max_eV"] - s[net]["pred_eV"]), c, OXIDE_SHAPES[idx[k]["candidate"]])
    a2.set_xlim(-0.6, len(order) - 0.4)
    a2.set_xticks(range(len(order)))
    a2.set_xticklabels([f"{OXIDE_TEX[idx[k]['candidate']]} O$_{idx[k]['label'][1:]}$" for k in order],
                       rotation=50, ha="right", rotation_mode="anchor")
    a2.tick_params(axis="x", length=2.5)
    a2.set_xlabel("Vacancy site (oxide, oxygen site)")
    a2.set_ylabel(r"Predicted $-$ DFT (eV)")
    lim = max(abs(s[net][b] - s["dft_eV"]) for s in sites.values() for b in ("min_eV", "max_eV"))
    a2.set_ylim(-1.08 * lim, 1.08 * lim)

    handles = [Line2D([], [], ls="none", marker=OXIDE_SHAPES[c], ms=MS, mfc=OXIDE_COLORS[c], mec=OXIDE_COLORS[c],
                      label=OXIDE_TEX[c]) for c in rank]
    handles += [sp.reference_handle("y = x", "published"), ns.zero_handle()]
    sp.legend_handles(fig, handles, ncol=5)
    sp.panel_letters(fig, (a1, a2))
    pdf = sp.save(fig, ns.out_path(SUB, STEM))
    return pdf


if __name__ == "__main__":
    print(main())
