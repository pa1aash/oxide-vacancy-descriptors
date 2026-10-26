"""SI figure: within-host residual MAE of S and D at B = 654 by chemistry and site.

Reads `results/within_host_breakdown.json` (written by analysis/within_host_breakdown.py) only. For each group:
the within-host residual MAE of S and D, both backbones, with two-sided 95% host-cluster bootstrap intervals,
and the zero-skill value (a predictor constant within each host) as a grey tick.

    (a) cation element   (b) d-electron class, with all tested hosts as the top row
    (c) perturbed-host-state-prone chemistry   (d) host stability tercile   (e) PBEsol(+U) band gap tercile
    (f) within-host rank of the vacancy-O Bader charge

    python figures/code/fig_si_withinhost_groups.py [--results-root DIR] [--out-dir DIR]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inputs_paper as ip
import style_paper as sp
from matplotlib.lines import Line2D

STEM = "fig_si_withinhost_groups"
RESULT = "within_host_breakdown.json"
METRIC = "within_host_mae"
SERIES = (("S", "v2"), ("S", "v1"), ("D", "v2"), ("D", "v1"))   # drawing order, top to bottom within a row
OFFSET = {k: o for k, o in zip(SERIES, (-0.27, -0.09, 0.09, 0.27))}
ZERO_SKILL_COLOR = "#7F7F7F"
TICKS = {
    "d_class": {"all": "all tested hosts", "d0": r"d$^{0}$", "d10": r"d$^{10}$", "other": "other",
                "unassigned": "unassigned"},
    "phs_prone": {"phs_prone": "Nb, W, Ti, V, Ta or Mo", "not_phs_prone": "none of these"},
    "stability_tercile": {"lower": "most stable third", "middle": "middle third", "upper": "least stable third"},
    "band_gap_tercile": {"lower": "smallest-gap third", "middle": "middle third", "upper": "largest-gap third"},
    "bader_rank": {"lowest": "most negative O", "middle": "intermediate O", "highest": "least negative O"},
}
RIGHT = ("d_class", "phs_prone", "stability_tercile", "band_gap_tercile", "bader_rank")


def rows_for(groups: list[dict], variable: str) -> list[dict]:
    rows = [g for g in groups if g["variable"] == variable]
    if variable == "d_class":
        rows = [g for g in groups if g["variable"] == "pooled"] + rows
    return rows


def draw_panel(ax, rows: list[dict], labels: list[str], x_max: float) -> None:
    for i, g in enumerate(rows):
        zs = g["v2"]["S"]["zero_skill_mae"]["point"]
        if zs is not None:
            ax.plot([zs, zs], [i - 0.4, i + 0.4], color=ZERO_SKILL_COLOR, lw=1.6, solid_capstyle="butt", zorder=1)
        for model, arm in SERIES:
            e = g[arm][model][METRIC]
            if e["point"] is None:
                continue
            y = i + OFFSET[(model, arm)]
            c = sp.COLORS[model]
            if "ci95" in e:
                lo, hi = e["ci95"]
                ax.plot([lo, hi], [y, y], color=c, lw=0.9, ls=sp.BACKBONE_LS[arm], zorder=2)
            ax.plot([e["point"]], [y], ls="none", marker=sp.MARKER, ms=sp.MARKER_SIZE, mec=c, mew=0.8,
                    mfc=c if sp.BACKBONE_FILLED[arm] else "white", zorder=3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(0.0, x_max)
    ax.tick_params(axis="y", length=0)


def x_upper(groups: list[dict]) -> float:
    hi = 0.0
    for g in groups:
        for model, arm in SERIES:
            e = g[arm][model][METRIC]
            hi = max(hi, *(e.get("ci95") or []), e["point"] or 0.0)
        hi = max(hi, g["v2"]["S"]["zero_skill_mae"]["point"] or 0.0)
    return 0.1 * int(hi / 0.1 + 1)


def main() -> Path:
    a = ip.cli(__doc__)
    root = Path(a.results_root) if a.results_root else sp.ROOT / "results"
    out_dir = a.out_dir if a.out_dir != sp.ROOT / "figures" / "main" else sp.ROOT / "figures" / "si"
    payload = json.loads((root / RESULT).read_text())["payload"]
    groups = payload["groups"]
    x_max = x_upper(groups)

    fig, _ = sp.figure("single", height_mm=205.0)
    fig.clf()
    right_rows = [rows_for(groups, v) for v in RIGHT]
    gs = fig.add_gridspec(len(RIGHT), 2, width_ratios=[1.0, 1.0], height_ratios=[len(r) + 1.2 for r in right_rows],
                          left=0.075, right=0.985, top=0.95, bottom=0.12, wspace=0.62, hspace=0.42)
    ax_cat = fig.add_subplot(gs[:, 0])
    cat = rows_for(groups, "cation")
    draw_panel(ax_cat, cat, [g["group"] for g in cat], x_max)
    ax_cat.set_xlabel("Within-host residual MAE (eV)")
    axes_right = []
    for j, (v, rows) in enumerate(zip(RIGHT, right_rows)):
        ax = fig.add_subplot(gs[j, 1])
        draw_panel(ax, rows, [TICKS[v][g["group"]] for g in rows], x_max)
        axes_right.append(ax)
    axes_right[-1].set_xlabel("Within-host residual MAE (eV)")

    handles = [sp.model_handle("S"), sp.model_handle("D"), *(sp.backbone_handle(v) for v in sp.BACKBONES),
               Line2D([], [], color=ZERO_SKILL_COLOR, lw=1.6, label="zero-skill (host-constant predictor)")]
    sp.legend_handles(fig, handles, ncol=2)
    sp.panel_letters(fig, [ax_cat, *axes_right])
    pdf = sp.save(fig, Path(out_dir) / STEM)
    ip.write_caption(out_dir, STEM, caption())
    return pdf


def caption() -> str:
    return """
**Figure S4. Site resolution within a host, by chemistry and site (EXTENSION).** Within-host residual MAE of S
(structure only, blue) and D (structure plus the DFT descriptors of the pristine host and site, green) at the
largest training budget, for groups of test hosts or sites; lower is better. The grey tick in each row is the
zero-skill value, the within-host residual MAE of any predictor that is constant within each host (the mean
absolute deviation of the true energies from their host mean); a model point at or right of it has no
within-host skill in that group. Circles: point values; horizontal bars: two-sided percentile intervals of
the host-cluster bootstrap, paired across models and backbones (`payload.bootstrap`); no bar where fewer
hosts contribute than `payload.definitions.small_n` requires. Filled circles and solid bars: primary
backbone (v2), adopted after diagnostics of the originally registered backbone; open circles and dashed
bars: originally registered backbone (v1). (**a**) Cation element, each host under every cation it contains,
elements as in `payload.definitions.cation`; (**b**) d-electron class of the host (`data/host_dbins_v1.csv`),
with all tested hosts as the top row; (**c**) hosts containing at least one of the cations with the
highest measured perturbed-host-state rates against the rest (`payload.definitions.phs_prone`); (**d**) host
stability terciles (formation energy per atom; edges in `payload.definitions.terciles`); (**e**) PBEsol(+U)
band gap terciles; (**f**) sites at the most negative, intermediate and least negative Bader charge of the
vacancy oxygen within their host (`payload.definitions.bader_rank`). Each site's prediction is the seed
ensemble averaged over the registered resamples in which its host was tested. Values from
`results/within_host_breakdown.json`, entries of `payload.groups` (`variable`, `group`), keys
`v2|v1.S|D.within_host_mae` (`point`, `ci95`) and `v2.S.zero_skill_mae.point`; group sizes in `n_hosts`,
`n_sites`, `n_hosts_by_metric` and small groups in `flag`. Analysis rule: the within-host breakdown row of
`docs/deviations.md`, logged before computing. Script:
`figures/code/fig_si_withinhost_groups.py`.
"""


if __name__ == "__main__":
    print(main())
