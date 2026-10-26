"""Figure 4 (W2b; the former Figure 4 on the network alone): staging. MAE of S, D and P, and the contrasts D - P and
P - S, at the same training sizes.

Input (read-only): results/staging_v2.json (``budgets.<B>.mae.<S|D|P>``, ``budgets.<B>.<D_minus_P|P_minus_S>``,
``delta_P_eV``).

    (a) seed-ensemble test MAE of S, D and P against training hosts, two-sided 95% bands
    (b) D - P (cost of predicting the descriptors) and P - S (staging against learning from structure directly),
        two-sided 95% bands, the margins +-delta_P and zero

Axes span the point estimates and the P - S intervals; wider bands (D and D - P at 25 and 50 training hosts) are
clipped and their full intervals are in the record.

    python figures/code/figure4_staging.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import network_style as ns
import style_paper as sp

SUB, STEM = "main", "figure4_staging"
# contrasts are not models: P - S in Okabe-Ito vermillion (used by no model), D - P in the paired-difference black
CONTRAST_COLORS = {"P_minus_S": "#D55E00", "D_minus_P": sp.COLORS["A"]}
CONTRAST_LABELS = {"P_minus_S": r"P $-$ S", "D_minus_P": r"D $-$ P"}
MODELS = ("S", "D", "P")
MODEL_LABELS = {"S": "S (structure only)", "D": "D (structure + descriptors)",
                "P": "P (predicted descriptors into D)"}


def series(payload: dict, getter):
    x = sorted(int(b) for b in payload["budgets"])
    rows = [getter(payload["budgets"][str(b)]) for b in x]
    return x, [r["mean"] for r in rows], [r["ci95"][0] for r in rows], [r["ci95"][1] for r in rows]


def main() -> Path:
    rec = ns.results(f"staging_{ns.NETWORK}")
    delta = rec["delta_P_eV"]

    fig, (ax_a, ax_b) = sp.figure("single", height_mm=78.0, ncols=2)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.93, bottom=0.30, wspace=0.32)

    span_a = []
    for m in MODELS:
        x, mean, lo, hi = series(rec, lambda b, m=m: b["mae"][m])
        ns.curve(ax_a, x, mean, lo, hi, sp.COLORS[m])
        span_a += mean
    ax_a.set_ylim(*ns.limits(span_a))
    sp.budget_axis(ax_a)
    ax_a.set_ylabel("Test MAE (eV)")

    span_b = [delta, -delta, 0.0]
    ns.delta_lines(ax_b, delta)
    ns.zero_line(ax_b)
    for key in ("D_minus_P", "P_minus_S"):
        x, mean, lo, hi = series(rec, lambda b, key=key: b[key])
        ns.curve(ax_b, x, mean, lo, hi, CONTRAST_COLORS[key])
        span_b += mean + (lo + hi if key == "P_minus_S" else [])
    ax_b.set_ylim(*ns.limits(span_b))
    sp.budget_axis(ax_b)
    ax_b.set_ylabel("Difference in test MAE (eV)")

    handles = [*(sp.model_handle(m, MODEL_LABELS[m]) for m in MODELS),
               *(ns.line_handle(CONTRAST_COLORS[k], CONTRAST_LABELS[k]) for k in ("P_minus_S", "D_minus_P")),
               ns.delta_handle(r"$\pm\delta_P$"), ns.zero_handle()]
    sp.legend_handles(fig, handles, ncol=3)
    sp.panel_letters(fig, (ax_a, ax_b))
    pdf = sp.save(fig, ns.out_path(SUB, STEM))
    ns.write_caption(SUB, STEM, CAPTION)
    return pdf


CAPTION = """
**Figure 4. Predicting the descriptors from structure against learning the energy from structure directly.** (**a**)
Test MAE of S, D and the staged model P, which passes descriptors predicted from the pristine structure by P1 to the
trained D of the same training set and seed. (**b**) D − P, the cost of predicting rather than knowing the
descriptors, and P − S, staging against learning from structure directly, with the margins ±δ_P. Bands are
two-sided bootstrap intervals, clipped at the axis limits, values in `results/staging_v2.json`.
"""


if __name__ == "__main__":
    print(main())
