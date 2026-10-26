"""The G1 learning-curve figure, read from results/g1_primary.json.

    (a) site-level test MAE against training hosts for B0, RF-Kumagai, S and D, with two-sided 95% bands;
        dashed reference lines at the published RF (0.34 eV) and CGCNN (0.29 eV) values
    (b) A = MAE_S - MAE_D with its two-sided 95% band; dashed lines at delta and at 0

    python scripts/analysis/fig_g1.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dftgnn.viz import style

ROOT = Path(__file__).resolve().parents[2]
KUMAGAI_RF_EV = 0.34      # Kumagai et al. 2021, q = 0, N = 700
KIYOHARA_CGCNN_EV = 0.29  # Kiyohara et al. 2025, q = 0


def main() -> None:
    p = json.loads((ROOT / "results" / "g1_primary.json").read_text())["payload"]
    B = [int(b) for b in p["budgets"]]
    per = p["per_budget"]
    desc = p["descriptive_curves"]["mae"]
    style.apply()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(style.WIDTH_1COL_IN, 2.75))
    fig.subplots_adjust(left=0.09, right=0.985, top=0.93, bottom=0.36, wspace=0.28)

    def series(get):
        return ([get(b)["mean"] for b in B], [get(b)["ci"][0] for b in B], [get(b)["ci"][1] for b in B])

    for key, get in (("B0", lambda b: desc["B0"][str(b)]), ("RF-Kumagai", lambda b: desc["RF-Kumagai"][str(b)]),
                     ("S", lambda b: per[str(b)]["MAE_S"]), ("D", lambda b: per[str(b)]["MAE_D"])):
        style.curve(ax1, B, *series(get), key=key)
    style.hline(ax1, KUMAGAI_RF_EV, "Kumagai et al. RF, 0.34 eV", dashes=(4, 2), color=style.COLORS["RF-Kumagai"])
    style.hline(ax1, KIYOHARA_CGCNN_EV, "Kiyohara et al. CGCNN, 0.29 eV", dashes=(1.5, 1.5))
    style.budget_axis(ax1)
    ax1.set_ylabel("Test MAE (eV)")
    ax1.set_ylim(0.0, 1.3)

    style.curve(ax2, B, *series(lambda b: per[str(b)]["A"]), key="A", label=r"$A$ = MAE$_S$ $-$ MAE$_D$")
    style.hline(ax2, p["delta_eV"], r"$\delta$ = 0.05 eV", dashes=(4, 2))
    style.hline(ax2, 0.0, "zero", dashes=(1, 2))
    style.budget_axis(ax2)
    ax2.set_ylabel(r"$A$ (eV)")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), handlelength=2.6,
               columnspacing=1.2)
    style.panel_letter(fig, ax1, "a")
    style.panel_letter(fig, ax2, "b")
    out = ROOT / "figures" / "main"
    out.mkdir(parents=True, exist_ok=True)
    style.save(fig, out / "fig_g1_learning_curves")


if __name__ == "__main__":
    main()
