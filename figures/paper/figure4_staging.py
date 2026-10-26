"""Figure 4: the staged network P against S, D and D* across training hosts.

Inputs (read only): results/derived_staging/staging_fair.json (``budgets.<B>.mae.<S|D|P>``, ``P_minus_S``,
``P_minus_D``) and results/derived_dstar/dstar_gaps.json (D* MAE and ``staging.budgets.<B>.P_minus_Dstar``).

    (a) test MAE of S, D, D* and P
    (b) MAE_P - MAE_S, MAE_P - MAE_D and MAE_P - MAE_D* from 100 training hosts, with +-delta

    python figures/paper/figure4_staging.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import base_figure4 as f4
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure2_descriptor_advantage as f2s
import network_style as ns
import style_paper as sp

SUB, STEM = "main", "figure4_staging"
RECORD = ROOT / "results" / "derived_staging" / "staging_fair.json"
CONTRASTS = ("P_minus_S", "P_minus_D")
CONTRAST_COLORS = {"P_minus_S": "#332288", "P_minus_D": f4.CONTRAST_COLORS["D_minus_P"]}
CONTRAST_LABELS = {"P_minus_S": r"P $-$ S", "P_minus_D": r"P $-$ D"}
B_FROM = 100
DSTAR_MAE = lambda b: b["metrics"]["mae"]["MAE"]["Dstar"]  # noqa: E731


def main() -> Path:
    rec = ns.payload(RECORD)
    delta = rec["delta_eV"]
    dp = f2s.dstar_payload()
    assert abs(dp["delta_eV"] - delta) < 1e-12

    fig, (ax_a, ax_b) = sp.figure("single", height_mm=72.0, ncols=2)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.93, bottom=0.30, wspace=0.32)

    span_a = []
    for m in f4.MODELS:
        x, mean, lo, hi = f4.series(rec, lambda b, m=m: b["mae"][m])
        ns.curve(ax_a, x, mean, lo, hi, sp.COLORS[m])
        span_a += mean + hi
    x, mean, lo, hi, _ = f2s.sweep(dp, DSTAR_MAE)
    f2s.dstar_curve(ax_a, x, mean, lo, hi, f2s.DSTAR_COLOR)
    span_a += mean + hi
    ax_a.set_ylim(min(0.2, min(lo) - 0.04), f2s.Y_TOP)
    sp.budget_axis(ax_a)
    ax_a.set_ylabel("Test MAE (eV)")

    span_b = [delta, -delta, 0.0]
    ns.delta_lines(ax_b, delta)
    ns.zero_line(ax_b)
    for key in CONTRASTS:
        x, mean, lo, hi = f4.series(rec, lambda b, key=key: b[key])
        keep = [i for i, v in enumerate(x) if v >= B_FROM]
        x, mean, lo, hi = ([s[i] for i in keep] for s in (x, mean, lo, hi))
        ns.curve(ax_b, x, mean, lo, hi, CONTRAST_COLORS[key])
        span_b += mean + lo + hi
    x = sorted(int(b) for b in dp["staging"]["budgets"])
    rows = [dp["staging"]["budgets"][str(b)]["P_minus_Dstar"] for b in x]
    mean, lo, hi = [r["mean"] for r in rows], [r["ci95"][0] for r in rows], [r["ci95"][1] for r in rows]
    f2s.dstar_curve(ax_b, x, mean, lo, hi, f2s.DSTAR_COLOR)
    span_b += mean + lo + hi
    ax_b.set_ylim(*ns.limits(span_b))
    sp.budget_axis(ax_b, x_min=B_FROM / 1.15)
    ax_b.set_ylabel("P minus reference (eV)")

    model = {m: sp.model_handle(m, f4.MODEL_LABELS[m]) for m in f4.MODELS}
    f2s.column_legend(fig, [
        [model["S"], model["D"], f2s.dstar_handle(f2s.DSTAR_COLOR, f2s.DSTAR_LABEL), model["P"]],
        [*(ns.line_handle(CONTRAST_COLORS[k], CONTRAST_LABELS[k]) for k in CONTRASTS),
         f2s.dstar_handle(f2s.DSTAR_COLOR, r"P $-$ D$^{*}$")],
        [ns.delta_handle(r"$\pm\delta$"), ns.zero_handle()]])
    sp.panel_letters(fig, (ax_a, ax_b))
    pdf = sp.save(fig, ns.out_path(SUB, STEM))
    return pdf


if __name__ == "__main__":
    print(main())
