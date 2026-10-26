"""Figure 3 (W2b; the former Figure 6 on the network alone): can the network tell the oxygen sites of one host apart?

Inputs (read-only), the network's records of the former Figure 6: results/c1c2_v2.json (learning-curve design),
results/precision_g1_v2.json (pooled 30 x 5 at 400 and 654) and results/cv_g1_v2.json (grouped cross-validation),
keys ``mae.<key>.<S|D-state>.within_host_skill`` and ``A_within_host.<key>``.

    (a) within-host skill of S and D against training hosts, with the zero-skill line
    (b) the within-host advantage A_within = within-host residual MAE of S minus that of D, with +-delta and zero

    python figures/code/figure3_within_host.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inputs_paper as ip
import network_style as ns
import style_paper as sp

SUB, STEM = "main", "figure3_within_host"
RECORDS = (("curve", f"c1c2_{ns.NETWORK}"), ("pooled", f"precision_g1_{ns.NETWORK}"), ("cv", f"cv_g1_{ns.NETWORK}"))


def main() -> Path:
    recs = {kind: ns.results(name) for kind, name in RECORDS}
    delta = recs["curve"]["delta_eV"]
    x_max = max(ip.x_of(recs["cv"]))

    fig, (ax_a, ax_b) = sp.figure("single", height_mm=74.0, ncols=2)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.93, bottom=0.30, wspace=0.34)

    ns.zero_line(ax_a)
    for model in ("S", "D"):
        for kind, p in recs.items():
            x, m, lo, hi = ip.mae_series(p, model, "within_host_skill")
            if kind == "curve":
                ns.curve(ax_a, x, m, lo, hi, sp.COLORS[model])
            else:
                ns.points(ax_a, x, m, lo, hi, sp.COLORS[model], kind)
    sp.budget_axis(ax_a, x_max=x_max)
    ax_a.set_ylabel("Within-host skill (eV)")

    ns.delta_lines(ax_b, delta)
    ns.zero_line(ax_b)
    colour = sp.COLORS["A"]
    for kind, p in recs.items():
        x, m, lo, hi, _ = ip.a_series(p, "A_within_host")
        if kind == "curve":
            ns.curve(ax_b, x, m, lo, hi, colour)
        else:
            ns.points(ax_b, x, m, lo, hi, colour, kind)
    sp.budget_axis(ax_b, x_max=x_max)
    ax_b.set_ylabel(r"$A_\mathrm{within}$ (eV)")

    handles = [sp.model_handle("S", "S (structure only)"), sp.model_handle("D", "D (structure + descriptors)"),
               *(ns.design_handle(k) for k in ("curve", "pooled", "cv")), ns.delta_handle(), ns.zero_handle()]
    sp.legend_handles(fig, handles, ncol=3)
    sp.panel_letters(fig, (ax_a, ax_b))
    pdf = sp.save(fig, ns.out_path(SUB, STEM))
    ns.write_caption(SUB, STEM, CAPTION)
    return pdf


CAPTION = """
**Figure 3. Site resolution within a host.** (**a**) Within-host skill of S and D, the within-host residual MAE of a
predictor that is constant within each host minus the network's, on held-out hosts with at least two oxygen
sites, where zero means the network places the host but not its sites. (**b**) The within-host advantage of the
descriptors, A_within, with the tolerance ±δ. Bands and vertical bars are two-sided bootstrap intervals, values in
`results/c1c2_v2.json`, `results/precision_g1_v2.json` and `results/cv_g1_v2.json`.
"""


if __name__ == "__main__":
    print(main())
