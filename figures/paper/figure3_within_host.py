"""Figure 3: within-host site resolution against training hosts, with D*.

Inputs (read only): results/c1c2_v2.json, results/precision_g1_v2.json, results/cv_g1_v2.json (S and D) and
results/derived_dstar/dstar_extras.json (``figure_inputs``: D* in the same layouts).

    (a) within-host skill of S, D and D* (zero-skill within-host residual MAE minus the model's)
    (b) A_within (S - D) and A*_within (S - D*) with +-delta

    python figures/paper/figure3_within_host.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure2_descriptor_advantage as f2s
import base_figure3 as f3
import inputs_paper as ip
import network_style as ns
import style_paper as sp

SUB, STEM = "main", "figure3_within_host"
EXTRAS = ROOT / "results" / "derived_dstar" / "dstar_extras.json"
LAYOUTS = (("curve", "c1c2_layout"), ("pooled", "precision_g1_layout"))


def dstar_series(layout: dict, block: str):
    """(x, point, lo, hi) of D* within-host skill (``block`` "skill") or A*_within ("a") in a c1c2-style layout."""
    keys = sorted(layout["mae"], key=int)
    if block == "skill":
        rows = [layout["mae"][k]["Dstar"]["within_host_skill"] for k in keys]
        return [int(k) for k in keys], [r["mean"] for r in rows], [r["ci"][0] for r in rows], [r["ci"][1] for r in rows]
    rows = [layout["A_within_host"][k] for k in keys]
    return [int(k) for k in keys], [r["A"] for r in rows], [r["ci95"][0] for r in rows], [r["ci95"][1] for r in rows]


def main() -> Path:
    recs = {kind: ns.results(name) for kind, name in f3.RECORDS}
    delta = recs["curve"]["delta_eV"]
    x_max = max(ip.x_of(recs["cv"]))
    extras = ns.payload(EXTRAS)
    assert abs(extras["delta_eV"] - delta) < 1e-12
    star = {kind: extras["figure_inputs"][name] for kind, name in LAYOUTS}

    def draw_dstar(ax, block: str, color: str) -> None:
        for kind, layout in star.items():
            x, m, lo, hi = dstar_series(layout, block)
            (f2s.dstar_curve if kind == "curve" else f2s.dstar_points)(ax, x, m, lo, hi, color)

    fig, (ax_a, ax_b) = sp.figure("single", height_mm=72.0, ncols=2)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.93, bottom=0.30, wspace=0.34)

    ns.zero_line(ax_a)
    for model in ("S", "D"):
        for kind, p in recs.items():
            x, m, lo, hi = ip.mae_series(p, model, "within_host_skill")
            if kind == "curve":
                ns.curve(ax_a, x, m, lo, hi, sp.COLORS[model])
            else:
                ns.points(ax_a, x, m, lo, hi, sp.COLORS[model], kind)
    draw_dstar(ax_a, "skill", f2s.DSTAR_COLOR)
    sp.budget_axis(ax_a, x_max=x_max)
    ax_a.set_ylabel("Within-host skill (eV)")

    ns.delta_lines(ax_b, delta)
    ns.zero_line(ax_b)
    black = sp.COLORS["A"]
    for kind, p in recs.items():
        x, m, lo, hi, _ = ip.a_series(p, "A_within_host")
        if kind == "curve":
            ns.curve(ax_b, x, m, lo, hi, black)
        else:
            ns.points(ax_b, x, m, lo, hi, black, kind)
    draw_dstar(ax_b, "a", f2s.DSTAR_COLOR)
    sp.budget_axis(ax_b, x_max=x_max)
    ax_b.set_ylabel("Within-host difference (eV)")

    f2s.column_legend(fig, [
        [sp.model_handle("S", "S (structure only)"), sp.model_handle("D", "D (structure + descriptors)"),
         f2s.dstar_handle(f2s.DSTAR_COLOR, f2s.DSTAR_LABEL)],
        [ns.line_handle(black, r"$A_\mathrm{within}$ (S $-$ D)"),
         f2s.dstar_handle(f2s.DSTAR_COLOR, r"$A^{*}_\mathrm{within}$ (S $-$ D$^{*}$)"), ns.delta_handle(), ns.zero_handle()],
        f2s.design_handles()])
    sp.panel_letters(fig, (ax_a, ax_b))
    pdf = sp.save(fig, ns.out_path(SUB, STEM))
    return pdf


if __name__ == "__main__":
    print(main())
