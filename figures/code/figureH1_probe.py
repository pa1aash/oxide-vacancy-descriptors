"""Figure H1 (W2b; the former Figure S7 redrawn on the network): latent probe of the structure-only network.

Input (read-only): results/probe_descriptive.json, block ``s_v2`` (the probe on the network's S checkpoints):
``s_v2.<host|site>.curves.<trained|shuffled|init0>.<B>`` (``mean``, ``ci95``) and
``s_v2.per_descriptor_654.<trained|shuffled|init0>.<descriptor>``; descriptor names and classes from the top-level
``descriptors`` and ``classes``. Control styles are those of figures/code/fig6_probe.py (imported, not edited); the
network has no init1 or init2 encoders, so the spread series of the former figure is absent.

    (a) host-electronic class-mean test R^2 against training hosts: trained probe and the two controls
    (b) the same for the site-electronic class
    (c) per-descriptor test R^2 at 654 training hosts

    python figures/code/figureH1_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig6_probe as old  # read-only reuse: CONTROL_STYLE, curve, control_series, control_handle
import network_style as ns
import style_paper as sp

SUB, STEM = "si", "figureH1_probe"
TRAINED = sp.COLORS["S"]
CONTROLS = ("init0", "shuffled")
LABELS = {"shuffled": "control: shuffled labels", "init0": "control: untrained encoder"}
# tick labels of panel (c): physical names, as the descriptors are named in the text
DESCRIPTOR_LABELS = {
    "ave_ele_diele": "Electronic dielectric constant", "ave_ion_diele": "Ionic dielectric constant",
    "band_gap": "Band gap (PBEsol(+$U$))", "formation_energy": "Oxide formation energy",
    **{f"cbm_{o}": f"CBM ${o}$ character" for o in "spdf"},
    **{f"vbm_{o}": f"VBM ${o}$ character" for o in "spdf"},
    "ave_bec": "O Born effective charge", "bader_charge": "O Bader charge", "bader_volume": "O Bader volume",
    "nn_ave_ave_bec": "Neighbour mean Born charge", "nn_max_ave_bec": "Neighbour maximum Born charge",
    "nn_min_ave_bec": "Neighbour minimum Born charge",
    "nn_ave_bader_charge": "Neighbour mean Bader charge", "nn_max_bader_charge": "Neighbour maximum Bader charge",
    "nn_min_bader_charge": "Neighbour minimum Bader charge", "o2p_center_from_vbm": "O $p$-band centre below VBM",
}
HEIGHT_MM = 95.0
CLASSES = (("host", "Host-electronic\nmean test $R^2$"), ("site", "Site-electronic\nmean test $R^2$"))


def main() -> Path:
    rec = ns.results("probe_descriptive")
    net = rec[f"s_{ns.NETWORK}"]
    budgets = sorted(int(b) for b in net["host"]["curves"]["trained"])

    sp.apply()
    fig = plt.figure(figsize=(sp.WIDTHS_IN["single"], HEIGHT_MM * sp.MM))
    top = fig.add_gridspec(1, 2, left=0.12, right=0.985, top=0.925, bottom=0.66, wspace=0.38)
    ax_a, ax_b = fig.add_subplot(top[0, 0]), fig.add_subplot(top[0, 1])
    # (c): the host-electronic and the site-electronic descriptors side by side, rows from the top
    ax_c = [fig.add_axes([0.305, 0.205, 0.155, 0.335]), fig.add_axes([0.80, 0.205, 0.18, 0.335])]

    for ax, (key, ylabel) in zip((ax_a, ax_b), CLASSES):
        cur = net[key]["curves"]
        ns.zero_line(ax)
        span = [0.0]
        control_bands = []
        for ctrl in CONTROLS:
            m, lo, hi = old.curve(cur[ctrl], budgets)
            old.control_series(ax, budgets, m, lo, hi, ctrl)
            span += m
            control_bands += lo + hi
        m, lo, hi = old.curve(cur["trained"], budgets)
        ns.curve(ax, budgets, m, lo, hi, TRAINED, zorder=4)
        span += lo + hi
        # limits span every mean and every band, so no band is clipped at the axes
        span += control_bands
        ax.set_ylim(*ns.limits(span))
        sp.budget_axis(ax)
        ax.set_ylabel(ylabel)

    names, classes = rec["descriptors"], rec["classes"]
    per = net["per_descriptor_654"]
    offsets = {"shuffled": 0.22, "init0": 0.0, "trained": -0.22}
    for ax, cls, ylabel in zip(ax_c, ("host", "site"), ("Host-electronic descriptor", "Site-electronic descriptor")):
        sub = [n for n, c in zip(names, classes) if c == cls]
        y = list(range(len(sub)))[::-1]
        ns.zero_line(ax, vertical=True)
        for key in ("shuffled", "init0", "trained"):
            vals = [per[key][n] for n in sub]
            mean = [v["mean"] for v in vals]
            err = [[v["mean"] - v["ci95"][0] for v in vals], [v["ci95"][1] - v["mean"] for v in vals]]
            yy = [v + offsets[key] for v in y]
            if key == "trained":
                ax.errorbar(mean, yy, xerr=err, fmt=sp.MARKER, ms=sp.MARKER_SIZE, color=TRAINED, ecolor=TRAINED,
                            elinewidth=sp.REF_W, capsize=0, mfc=TRAINED, mec=TRAINED, mew=0.8, zorder=4)
            else:
                st = old.CONTROL_STYLE[key]
                ax.errorbar(mean, yy, xerr=err, fmt=st["marker"], ms=sp.MARKER_SIZE * 0.85, color=st["color"],
                            ecolor=st["color"], elinewidth=sp.REF_W, capsize=0, mfc=st["color"], mec=st["color"])
        ax.set_yticks(y)
        ax.set_yticklabels([DESCRIPTOR_LABELS[n] for n in sub])
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.7, len(sub) - 0.3)
        ax.set_xlabel(r"Test $R^2$ at 654 training hosts", loc="right")

    handles = [ns.line_handle(TRAINED, "trained S encoder"),
               *(old.control_handle(k, LABELS[k]) for k in ("shuffled", "init0")), ns.zero_handle()]
    # explicit grids: the legend is placed below them directly (sp.save checks it clears every axes)
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    sp.panel_letters(fig, [ax_a, ax_b])
    fig.text(0.005, 0.575, "(c)", fontsize=9, fontweight="bold", va="bottom", ha="left")
    pdf = sp.save(fig, ns.out_path(SUB, STEM))
    ns.write_caption(SUB, STEM, CAPTION)
    return pdf


CAPTION = """
**Figure H1. Linear recovery of the DFT descriptors from the structure-only network.** A ridge probe maps the
readout input of S to each descriptor given to D, scored by test R² for the trained network, shuffled labels and
an untrained encoder, as (**a**) host-electronic and (**b**) site-electronic class means against training hosts and
(**c**) per descriptor, at the largest training size. Bands and bars are two-sided resample-bootstrap intervals, values in `results/probe_descriptive.json`.
"""


if __name__ == "__main__":
    print(main())
