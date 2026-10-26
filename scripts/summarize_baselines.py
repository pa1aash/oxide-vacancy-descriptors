"""Collect the baseline results into results/baselines_summary.json."""
from __future__ import annotations

import json

import pandas as pd

from dftgnn.baselines import host_stat_table
from dftgnn.config import load_config
from dftgnn.io.results import REPO_ROOT, write_result
from dftgnn.stats import metrics as M

K21_MAE = 0.34
K21_TOL = 0.05
K25_MAE = 0.29
MODELS = {"b0": "b0_physics_floor", "rf_kumagai": "rf_kumagai", "rf_electronic": "rf_electronic",
          "rf_structural": "rf_structural"}
LABEL = {"b0": "B0 physics floor", "rf_kumagai": "RF-Kumagai (70 descriptors)",
         "rf_electronic": "RF-electronic (22)", "rf_structural": "RF-structural (48)"}


def load(name: str) -> dict:
    return json.loads((REPO_ROOT / "results" / f"{name}.json").read_text())


def fmt(x: float, ci=None, nd: int = 3) -> str:
    return f"{x:.{nd}f}" + (f" [{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]" if ci else "")


def agg(a: dict, m: str) -> dict:
    return {"mean": a[m]["mean"], "sd": a[m]["sd"], "ci": a[m]["ci"]}


def pt(k: dict, m: str) -> dict:
    return {"point": k["metrics"][m]["point"], "ci": k["metrics"][m]["ci"]}


def stats_by_resample(frame: pd.DataFrame, budget: int) -> list[pd.DataFrame]:
    f = frame[(frame["resample"] >= 0) & (frame["budget"] == budget)]
    return [host_stat_table(g) for _, g in f.groupby("resample")]


def main() -> None:
    cfg = load_config()
    n_boot = cfg.stats.cluster_bootstrap_n
    full = str(max(cfg.budgets.hosts))
    R = {k: load(v)["payload"] for k, v in MODELS.items()}
    oracle = load("host_mean_oracle")["payload"]
    curve = load("rf_kumagai_curve")["payload"]
    sens = load("sens_high_moment_rf")["payload"]

    outer = {}
    for k, p in R.items():
        a = p["outer"][full]["aggregate"]
        outer[k] = {m: agg(a, m) for m in ("mae", "rmse", "r2", "within_host_mae", "host_mean_mae")}
    outer["host_mean_oracle"] = {m: agg(oracle["aggregate"], m)
                                 for m in ("mae", "rmse", "r2", "within_host_mae", "host_mean_mae")}
    kiy = {k: {m: pt(p["kiyohara"], m) for m in ("mae", "rmse", "r2", "within_host_mae", "host_mean_mae")}
           | {"n_train_hosts": p["kiyohara"]["n_train_hosts"], "n_test_hosts": p["kiyohara"]["n_test_hosts"]}
           for k, p in R.items()}
    kiy["host_mean_oracle"] = {m: pt(oracle["kiyohara"], m)
                               for m in ("mae", "rmse", "r2", "within_host_mae", "host_mean_mae")}
    by_budget = {k: {b: agg(v["aggregate"], "mae") for b, v in p["outer"].items()} for k, p in R.items()}

    # paired differences at full budget (same hosts, identical bootstrap draws)
    preds = {k: pd.read_parquet(REPO_ROOT / p["predictions"]["path"]) for k, p in R.items()}
    st = {k: stats_by_resample(f, int(full)) for k, f in preds.items()}
    pairs = {"rf_electronic_minus_rf_structural": ("rf_electronic", "rf_structural"),
             "rf_kumagai_minus_b0": ("rf_kumagai", "b0"),
             "rf_electronic_minus_rf_kumagai": ("rf_electronic", "rf_kumagai"),
             "rf_structural_minus_rf_kumagai": ("rf_structural", "rf_kumagai")}
    seed = cfg.split.seeds[0]
    deltas = {n: {m: v for m, v in M.aggregate_delta(st[a], st[b], n_boot=n_boot, ci=cfg.stats.ci,
                                                      seed=seed).items() if m in ("mae", "within_host_mae")}
              for n, (a, b) in pairs.items()}
    kst = {k: host_stat_table(f[f["resample"] == -1]) for k, f in preds.items()}
    kiy_delta = {"rf_electronic_minus_rf_structural": M.aggregate_delta(
        [kst["rf_electronic"]], [kst["rf_structural"]], n_boot=n_boot, ci=cfg.stats.ci, seed=seed)["mae"]}

    k21 = outer["rf_kumagai"]["mae"]["mean"]
    k21_diff = k21 - K21_MAE
    acceptance = {
        "criterion": f"mean q=0 MAE of RF-Kumagai over the 10 full-budget ({full}-host) resamples within "
                     f"{K21_TOL} eV of K21's {K21_MAE} eV",
        "ours_mean_mae_eV": k21, "ours_ci": outer["rf_kumagai"]["mae"]["ci"], "published_eV": K21_MAE,
        "difference_eV": k21_diff, "pass": bool(abs(k21_diff) <= K21_TOL),
        "differences_from_K21": [
            "universe v1 (1726 sites / 818 hosts) vs K21's 1745 / 824 before our D1-D3 filters",
            f"test set of {len(load_split_test())} hosts (20%) vs 48 hosts",
            "training pool of 654 hosts vs 700",
            "MAE over our test sites; K21 evaluates site errors of its 48 test oxides",
            "scikit-learn 1.9.1 vs 0.24.1",
        ],
        "curve_endpoint_check": {"curve_mae_at_654_eV": curve["by_size"][str(curve["sizes"][-1])]["aggregate"]["mae"]["mean"],
                                 "published_at_700_eV": K21_MAE}}
    payload = {
        "full_budget_hosts": int(full), "outer_full_budget": outer, "kiyohara_split": kiy,
        "mae_by_budget": by_budget, "paired_deltas_full_budget": deltas,
        "paired_delta_kiyohara_cluster_bootstrap": kiy_delta,
        "reference": {"kumagai_2021_rf_q0_mae_eV": K21_MAE, "kiyohara_2025_cgcnn_q0_mae_eV": K25_MAE},
        "rf_kumagai_vs_published": acceptance,
        "kiyohara_vs_published": {k: {"mae": kiy[k]["mae"]["point"], "minus_0.29": kiy[k]["mae"]["point"] - K25_MAE,
                                      "minus_0.34": kiy[k]["mae"]["point"] - K21_MAE} for k in kiy},
        "learning_curve_endpoints": {b: curve["by_size"][b]["aggregate"]["mae"] | {"n_resamples": 100}
                                     for b in (str(curve["sizes"][0]), str(curve["sizes"][-1]))},
        "high_moment_sensitivity_delta_mae": sens["delta_total_excluded_minus_baseline"]["mae"],
        "splits_manifest_sha256": R["rf_kumagai"]["splits_manifest_sha256"],
    }
    write_result("baselines_summary", payload, config=cfg)
    (REPO_ROOT / "docs" / "baselines.md").write_text(render(payload, R, oracle, curve, sens, deltas, kiy_delta))


def load_split_test() -> list:
    from dftgnn.split import load_split

    return load_split("outer_r0")["test"]


def render(p, R, oracle, curve, sens, deltas, kiy_delta) -> str:
    o, kiy, acc = p["outer_full_budget"], p["kiyohara_split"], p["rf_kumagai_vs_published"]
    full = p["full_budget_hosts"]
    order = ["b0", "rf_kumagai", "rf_electronic", "rf_structural", "host_mean_oracle"]
    names = {**LABEL, "host_mean_oracle": "Host-mean oracle (descriptive, not a model)"}
    L = ["# Non-GNN baselines (S04)", "",
         "Every number is read from `results/*.json` by `scripts/summarize_baselines.py`. Intervals are 95%:",
         "hierarchical bootstrap over outer resamples and test hosts for the 10-resample rows, test-host cluster",
         "bootstrap for the single Kiyohara split. Energies are in eV; targets are the neutral O-vacancy formation",
         "energies of universe v1 (1726 sites, 818 hosts).", "",
         f"## Full budget ({full} training hosts), 10 outer resamples", "",
         "| Predictor | MAE | within-host residual MAE | RMSE | R2 |", "|---|---|---|---|---|"]
    for k in order:
        L.append(f"| {names[k]} | {fmt(o[k]['mae']['mean'], o[k]['mae']['ci'])} | "
                 f"{fmt(o[k]['within_host_mae']['mean'], o[k]['within_host_mae']['ci'])} | "
                 f"{fmt(o[k]['rmse']['mean'])} | {fmt(o[k]['r2']['mean'])} |")
    L += ["", "The host-mean oracle predicts each test site with the true mean over its host's sites. No predictor that",
          "cannot tell the O sites of one host apart can beat its MAE.",
          "Within-host residual MAE is the zero-skill value for any predictor that is constant within a host: the oracle and B0 "
          "(both host-level) share it by construction, so it is the reference for within-host skill, not a lower bound.", "",
          f"Within-host spread of the site energies (mean sd over hosts with two or more sites): {oracle['within_host_spread_universe']['sd_eV']['mean']:.3f} eV; "
          f"mean range {oracle['within_host_spread_universe']['range_eV']['mean']:.3f} eV.", "",
          "## Kiyohara split (train on their 571 train hosts, test on their 126 test hosts)", "",
          "| Predictor | MAE | within-host residual MAE | RMSE | R2 |", "|---|---|---|---|---|"]
    for k in order:
        m = kiy[k]
        L.append(f"| {names[k]} | {fmt(m['mae']['point'], m['mae']['ci'])} | "
                 f"{fmt(m['within_host_mae']['point'], m['within_host_mae']['ci'])} | "
                 f"{fmt(m['rmse']['point'])} | {fmt(m['r2']['point'])} |")
    L += ["", "Baselines need no validation set, so the 121 validation hosts are left unused.", "",
          "## Comparison with published values", "",
          "| Reference | Value | Ours |", "|---|---|---|",
          f"| Kumagai 2021 RF, q = 0, 700 training oxides (K21 abstract, Fig. 5(a), Sec. III D) | {K21_MAE} eV | "
          f"RF-Kumagai at {full} hosts: {fmt(acc['ours_mean_mae_eV'], acc['ours_ci'])} eV; difference "
          f"{acc['difference_eV']:+.3f} eV, **{'PASS' if acc['pass'] else 'FAIL'}** (tolerance {K21_TOL} eV) |",
          f"| Kiyohara 2025 CGCNN, q = 0, their split (PRL 135, 246101) | {K25_MAE} eV | "
          f"RF-Kumagai on their split: {fmt(kiy['rf_kumagai']['mae']['point'], kiy['rf_kumagai']['mae']['ci'])} eV; "
          f"B0: {fmt(kiy['b0']['mae']['point'])} eV; host-mean oracle: {fmt(kiy['host_mean_oracle']['mae']['point'])} eV |",
          "", "Differences between our setting and K21's: " + "; ".join(acc["differences_from_K21"]) + ".", "",
          "## Learning curve (MAE by training hosts, 10 outer resamples)", "",
          "| Training hosts | " + " | ".join(LABEL[k] for k in LABEL) + " |", "|---|" + "---|" * len(LABEL)]
    for b in sorted(p["mae_by_budget"]["b0"], key=int):
        L.append(f"| {b} | " + " | ".join(fmt(p["mae_by_budget"][k][b]["mean"]) for k in LABEL) + " |")
    L += ["", "## Kumagai-style RF learning curve (100 resamples per size)", "",
          "| Training hosts | MAE (mean over 100 resamples) | sd across resamples | 95% interval |", "|---|---|---|---|"]
    for n in curve["sizes"]:
        a = curve["by_size"][str(n)]["aggregate"]["mae"]
        L.append(f"| {n} | {a['mean']:.3f} | {a['sd']:.3f} | [{a['ci'][0]:.3f}, {a['ci'][1]:.3f}] |")
    e = curve["extrapolation_a_N^-0.5_plus_b"]
    L += ["", f"Fit a N^-0.5 + b through N = {e['through_sizes'][0]} and {e['through_sizes'][1]}: "
          f"b = {e['b_N_to_infinity_eV']:.3f} eV (K21 reports {e['published_N_to_infinity_eV']} eV). "
          "Published values at N = 22, 70 and 220 appear only in Fig. 5(a) and are not used.", "",
          "## Descriptor-class characterisation (SI; no model is altered)", "",
          "| Contrast at full budget | MAE difference | 95% interval | within-host MAE difference |", "|---|---|---|---|"]
    for n, d in deltas.items():
        L.append(f"| {n.replace('_minus_', ' minus ').replace('_', '-')} | {d['mae']['mean']:+.3f} | "
                 f"[{d['mae']['ci'][0]:+.3f}, {d['mae']['ci'][1]:+.3f}] | {d['within_host_mae']['mean']:+.3f} |")
    d = sens["delta_total_excluded_minus_baseline"]["mae"]
    L += ["", "## Sensitivity: high-moment entries", "",
          f"Removing the {sens['n_excluded_entries']} entries with |defect moment| > 0.5 muB (pre-registered) from RF-Kumagai at "
          f"{full} hosts changes the MAE by {d['mean']:+.3f} eV, 95% interval [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}] "
          "(paired over the same test hosts).", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
