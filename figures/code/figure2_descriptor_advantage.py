"""Figure 2 (W2b; merges the former Figures 2 and 3): site error and descriptor advantage of the network against
training hosts.

Input: results/derived_c1/final_network_summary.json only (``table.rows``; read-only).

    (a) test MAE of S and D in the learning-curve design (25 to 654 training hosts), with RF-Kumagai and the
        physics floor B0 as reference curves; two-sided 95% bands
    (b) A = MAE_S - MAE_D from A_FROM training hosts: learning-curve design (line, band), the pooled 30 x 5
        estimates at 400 and 654 (diamonds, drawn just left of their size) and grouped cross-validation at its mean
        training size (triangles), with two-sided 95% intervals, the one-sided 95% upper bounds (bars) and the
        tolerance +-delta (dashed lines)

A at 25 and 50 training hosts (-0.14 and -0.51 eV, intervals up to 2 eV wide) is left out of (b): there D is
dominated by under 1% of the test sites, whose conduction-band f-character descriptor (cbm_f) lies far outside
the training range (, (a)). Both values stay in the record and in panel (a) as the two MAE curves. The
y-limits of (b) span +-delta, every two-sided interval and every one-sided bound, so nothing in (b) is
clipped.

    python figures/code/figure2_descriptor_advantage.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import network_style as ns
import style_paper as sp

SUB, STEM = "main", "figure2_descriptor_advantage"
CURVE = ("B25", "B50", "B100", "B200", "B400", "B654")
POOLED = ("B400_30x5", "B654_30x5")
CV = ("CV_k10", "CV_k20")
A_FROM = 100


def mae(rows, keys, col):
    sel = [rows[k] for k in keys]
    return ([r["x_training_hosts"] for r in sel], [r[col]["mean"] for r in sel],
            [r[col]["ci95"][0] for r in sel], [r[col]["ci95"][1] for r in sel])


def adv(rows, keys):
    sel = [rows[k] for k in keys]
    return ([r["x_training_hosts"] for r in sel], [r["A"]["A"] for r in sel], [r["A"]["ci95"][0] for r in sel],
            [r["A"]["ci95"][1] for r in sel], [r["A"]["upper_one_sided_95"] for r in sel])


def main() -> Path:
    table = ns.payload(ns.SUMMARY)["table"]
    rows, delta = table["rows"], table["delta_eV"]

    fig, (ax_a, ax_b) = sp.figure("single", height_mm=80.0, ncols=2)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.93, bottom=0.30, wspace=0.32)

    # (a) MAE of S and D with the two reference curves
    tops = []
    for key, col in (("B0", "MAE_B0"), ("RF-Kumagai", "MAE_RF")):
        x, m, lo, hi = mae(rows, CURVE, col)
        sp.baseline_curve(ax_a, x, m, lo, hi, key)
        tops += m
    for key, col in (("S", "MAE_S"), ("D", "MAE_D")):
        x, m, lo, hi = mae(rows, CURVE, col)
        ns.curve(ax_a, x, m, lo, hi, sp.COLORS[key])
        tops += m
    sp.budget_axis(ax_a)
    ax_a.set_ylim(0.2, 1.06 * max(tops))
    ax_a.set_ylabel("Test MAE (eV)")

    # (b) the descriptor advantage
    colour = sp.COLORS["A"]
    ns.delta_lines(ax_b, delta)
    ns.zero_line(ax_b)
    span = [delta, -delta]
    curve_keys = [k for k in CURVE if rows[k]["x_training_hosts"] >= A_FROM]
    x, m, lo, hi, ub = adv(rows, curve_keys)
    ns.curve(ax_b, x, m, lo, hi, colour)
    ns.upper_bounds(ax_b, x, ub)
    span += lo + hi + ub
    for keys, kind in ((POOLED, "pooled"), (CV, "cv")):
        x, m, lo, hi, ub = adv(rows, keys)
        xs = ns.points(ax_b, x, m, lo, hi, colour, kind)
        ns.upper_bounds(ax_b, xs, ub)
        span += lo + hi + ub
    sp.budget_axis(ax_b, x_max=max(rows[k]["x_training_hosts"] for k in CV), x_min=A_FROM / 1.15)
    ax_b.set_ylim(*ns.limits(span))
    ax_b.set_ylabel(r"$A$ = MAE$_S$ $-$ MAE$_D$ (eV)")

    handles = [sp.model_handle("S", "S (structure only)"), sp.model_handle("D", "D (structure + descriptors)"),
               sp.baseline_handle("RF-Kumagai"), sp.baseline_handle("B0"),
               *(ns.design_handle(k, prefix=r"$A$, ") for k in ("curve", "pooled", "cv")),
               ns.upper_handle(), ns.delta_handle(), ns.zero_handle()]
    sp.legend_handles(fig, handles, ncol=3)
    sp.panel_letters(fig, (ax_a, ax_b))
    pdf = sp.save(fig, ns.out_path(SUB, STEM))
    ns.write_caption(SUB, STEM, CAPTION)
    return pdf


CAPTION = """
**Figure 2. Site error and descriptor advantage of the network against training hosts.** (**a**) Test MAE of S
and D on held-out oxides, with RF-Kumagai and B0. (**b**) A = MAE_S − MAE_D with ±δ, one-sided upper bounds
(bars), pooled resamples and seeds, and grouped cross-validation, from one hundred hosts, because at twenty-five
and fifty a handful of test sites whose conduction-band f-character descriptor lies far outside the training range
dominate D. Bands, clipped in (**a**), and vertical bars are two-sided bootstrap intervals, values in
`results/derived_c1/final_network_summary.json`.
"""


if __name__ == "__main__":
    print(main())
