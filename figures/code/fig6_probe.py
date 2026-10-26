"""Inputs: results/probe_c4_v1.json (the §10 rule and the class-mean curves) and results/probe_descriptive.json
(the per-descriptor R^2), written by analysis/probe_stats.py under the rules of the row of 2026-10-14. No run
record or embedding is read here.

    (a) host-electronic class-mean test R^2 against training hosts: trained probe and the controls
    (b) the same for the site-electronic class
    (c) per-descriptor test R^2 at B = 654: trained probe, shuffled-label control, random-init control (init0)

The trained probe is drawn in the v1 style (dashed line, open markers, hatched band); the controls are
grey-toned series. If the rule fails the figure goes to figures/si/ and the caption says so.

    python figures/code/fig6_probe.py [--results-root DIR] [--out-dir DIR]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inputs_paper as ip
import style_paper as sp

STEM = "fig6_probe"
VERSION = "v1"
TRAINED = sp.COLORS["S"]
# controls: distinct grey tones and marker shapes; init1 and init2 (descriptive spread of init0) thin, no band
CONTROL_STYLE = {
    "shuffled": {"color": "#4D4D4D", "marker": "s", "band": True, "lw": sp.BASELINE_W, "ls": "-"},
    "init0": {"color": "#8C8C8C", "marker": "^", "band": True, "lw": sp.BASELINE_W, "ls": "-"},
    "init1": {"color": "#BDBDBD", "marker": "v", "band": False, "lw": 0.7, "ls": (0, (1.5, 1.5))},
    "init2": {"color": "#BDBDBD", "marker": "D", "band": False, "lw": 0.7, "ls": (0, (1.5, 1.5))},
}
LABELS = {
    "trained": "trained S encoder",
    "shuffled": "control: shuffled labels",
    "init0": "control: random-init encoder (init0)",
    "init1": "random-init spread (init1, init2; descriptive)",
}
CLASSES = (("host", "Host-electronic class-mean test $R^2$"), ("site", "Site-electronic class-mean test $R^2$"))


def curve(block: dict, budgets):
    rows = [block[str(b)] for b in budgets]
    return [r["mean"] for r in rows], [r["ci95"][0] for r in rows], [r["ci95"][1] for r in rows]


def control_series(ax, x, m, lo, hi, key: str) -> None:
    st = CONTROL_STYLE[key]
    if st["band"]:
        ax.fill_between(x, lo, hi, color=st["color"], alpha=sp.BAND_ALPHA, linewidth=0)
    ax.plot(x, m, color=st["color"], lw=st["lw"], ls=st["ls"], marker=st["marker"], ms=sp.MARKER_SIZE * 0.85,
            mfc=st["color"], mec=st["color"])


def control_handle(key: str, label: str):
    from matplotlib.lines import Line2D

    st = CONTROL_STYLE[key]
    return Line2D([], [], color=st["color"], lw=st["lw"], ls=st["ls"], marker=st["marker"],
                  ms=sp.MARKER_SIZE * 0.85, mfc=st["color"], mec=st["color"], label=label)


def trained_handle():
    from matplotlib.lines import Line2D

    return Line2D([], [], color=TRAINED, lw=sp.LINE_W, ls=sp.BACKBONE_LS[VERSION], marker=sp.MARKER,
                  ms=sp.MARKER_SIZE, mfc="white", mec=TRAINED, mew=0.8, label=LABELS["trained"])


def main() -> Path:
    import matplotlib.pyplot as plt

    a = ip.cli(__doc__)
    c4 = ip.require("probe_c4_v1", a.results_root)
    desc = ip.require("probe_descriptive", a.results_root)
    holds = bool(c4["C4_holds"])
    out_dir = Path(a.out_dir)
    if not holds and out_dir.name == "main":
        out_dir = out_dir.parent / "si"
    budgets = c4["budgets"]

    sp.apply()
    fig = plt.figure(figsize=(sp.WIDTHS_IN["single"], 172.0 * sp.MM))
    top = fig.add_gridspec(1, 2, left=0.1, right=0.985, top=0.94, bottom=0.66, wspace=0.34)
    low = fig.add_gridspec(1, 1, left=0.25, right=0.985, top=0.565, bottom=0.165)
    ax_a, ax_b = fig.add_subplot(top[0, 0]), fig.add_subplot(top[0, 1])
    ax_c = fig.add_subplot(low[0, 0])

    for ax, (key, ylabel) in zip((ax_a, ax_b), CLASSES):
        cur = c4["curves"][key]
        sp.reference_line(ax, 0.0, "", "zero")
        for ctrl in ("init2", "init1", "init0", "shuffled"):
            control_series(ax, budgets, *curve(cur[ctrl], budgets), ctrl)
        m, lo, hi = curve(cur["trained"], budgets)
        sp.band(ax, budgets, lo, hi, TRAINED, VERSION)
        ax.plot(budgets, m, color=TRAINED, lw=sp.LINE_W, ls=sp.BACKBONE_LS[VERSION], marker=sp.MARKER,
                ms=sp.MARKER_SIZE, mfc="white", mec=TRAINED, mew=0.8, zorder=4)
        sp.budget_axis(ax)
        ax.set_ylabel(ylabel)
    lims = [min(ax.get_ylim()[0] for ax in (ax_a, ax_b)), max(ax.get_ylim()[1] for ax in (ax_a, ax_b))]
    for ax in (ax_a, ax_b):
        ax.set_ylim(*lims)

    names, classes = desc["descriptors"], desc["classes"]
    per = desc["per_descriptor_r2"]
    y = list(range(len(names)))[::-1]                      # host-electronic first, from the top
    sep = len(names) - sum(1 for c in classes if c == "host") - 0.5
    ax_c.axhline(sep, color="#D9D9D9", lw=sp.REF_W)
    ax_c.axvline(0.0, color=sp.COLORS["reference"], lw=sp.REF_W, dashes=sp.REF_DASHES["zero"])
    offsets = {"shuffled": 0.22, "init0": 0.0, "trained": -0.22}
    for key in ("shuffled", "init0", "trained"):
        vals = [per[key][n]["654"] for n in names]
        mean = [v["mean"] for v in vals]
        err = [[v["mean"] - v["ci95"][0] for v in vals], [v["ci95"][1] - v["mean"] for v in vals]]
        yy = [v + offsets[key] for v in y]
        if key == "trained":
            eb = ax_c.errorbar(mean, yy, xerr=err, fmt=sp.MARKER, ms=sp.MARKER_SIZE, color=TRAINED, ecolor=TRAINED,
                               elinewidth=sp.REF_W, capsize=0, mfc="white", mec=TRAINED, mew=0.8, zorder=4)
            eb[2][0].set_linestyle(sp.BACKBONE_LS[VERSION])
        else:
            st = CONTROL_STYLE[key]
            ax_c.errorbar(mean, yy, xerr=err, fmt=st["marker"], ms=sp.MARKER_SIZE * 0.85, color=st["color"],
                          ecolor=st["color"], elinewidth=sp.REF_W, capsize=0, mfc=st["color"], mec=st["color"])
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(names)
    ax_c.tick_params(axis="y", labelsize=6.5)
    ax_c.set_ylim(-0.7, len(names) - 0.3)
    ax_c.set_xlabel(r"Test $R^2$ at 654 training hosts")

    handles = [trained_handle(), control_handle("shuffled", LABELS["shuffled"]),
               control_handle("init0", LABELS["init0"]), control_handle("init1", LABELS["init1"]),
               sp.backbone_handle(VERSION), sp.reference_handle("zero", "zero")]
    # explicit grids: the legend is placed below them directly (sp.save checks it clears every axes)
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    sp.panel_letters(fig, [ax_a, ax_b, ax_c])
    pdf = sp.save(fig, out_dir / STEM)
    ip.write_caption(out_dir, STEM, caption(holds))
    return pdf


def caption(holds: bool) -> str:
    where = ("The rule holds, so the figure is in the main text." if holds else
             "The rule does not hold (`results/probe_c4_v1.json`, `payload.C4_holds` false; `payload.outcome`), so "
             "C4 moves to the Supplementary Information and this figure is placed there (`figures/si/`).")
    return f"""
**Figure S7. Latent probe: linear recovery of the DFT electronic descriptors from the structure-only model's readout
input.** For every resample r and budget B, a ridge regression maps the readout input vector of the seed-0 S
checkpoint ([vacancy-node embedding, pooled node embeddings, global state]) to each descriptor given to D, with
the regularisation chosen by cross-validation grouped by host on the training sites, and is scored by
test R² on the test sites (implementation and rule: `docs/deviations.md`, the latent-probe row of the A4 step;
`docs/ANALYSIS_PLAN.md`, latent-probe section). Originally registered backbone (v1), the registered analysis:
dashed line, open markers and hatched band for the trained probe (dashed bars in (**c**)). Controls in grey: the same probe with descriptor
rows permuted among training hosts (shuffled labels; squares) and the probe on an untrained encoder of the same
architecture (init0, the registered random-init control; triangles); the encoders init1 and init2 (light grey,
dotted, no band) show the spread of the random-init control and are descriptive. (**a**)
Host-electronic and (**b**) site-electronic class-mean test R² (unweighted mean over the descriptors of the class)
against training hosts on a logarithmic axis: `results/probe_c4_v1.json`, keys
`payload.curves.<host|site>.<trained|shuffled|init0|init1|init2>.<B>` (`mean` over the resamples, two-sided
interval `ci95` of the resample bootstrap; per-resample values `per_resample`). The dotted line is zero. (**c**)
Test R² per descriptor at the largest budget, host-electronic descriptors above the grey rule and site-electronic
below: `results/probe_descriptive.json`, keys `payload.per_descriptor_r2.<trained|shuffled|init0>.<descriptor>.654`
(`mean`, `ci95`). The rule of the latent-probe section is evaluated in `payload.rule.<host|site>` of
`results/probe_c4_v1.json`: `trend` (Spearman ρ of log budget against the trained class mean over all resample-budget cells,
`rho`, `ci95`, `holds`) and `controls.<B>|<shuffled|init0>` (trained minus control, `mean`, `ci95`, `exceeds`) at
the budgets the rule names. {where} Panel letters (a) and (b) together form the class-mean panel of the analysis plan and (c) its
per-descriptor panel. The collaborator's probe on the primary backbone (v2) is POST HOC and is not drawn; its
quantities are in `results/probe_descriptive.json`, key `payload.s_v2`. Within-host R² of the site-electronic
descriptors: `payload.within_host_site`. Script: `figures/code/fig6_probe.py`.
"""


if __name__ == "__main__":
    print(main())
