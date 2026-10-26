"""Baseline runs on the tracked splits (CPU). Each subcommand writes its result via write_result.

    oracle | b0 | rf --variant {kumagai,electronic,structural} | curve | sens

Run from a clean tree: ``write_result`` refuses to write otherwise.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from dftgnn.baselines import (
    Data,
    feature_sets,
    host_stat_table,
    metrics_block,
    physics,
    rf,
    run_kiyohara,
    run_outer,
    run_split,
    summarize_outer,
    write_predictions,
)
from dftgnn.config import load_config
from dftgnn.io.results import DirtyTreeError
from dftgnn.io.results import write_result as _write_result
from dftgnn.split import curve_resample, derived_seed, load_meta, load_split
from dftgnn.stats import metrics as M
from dftgnn.stats.variance import within_group_spread

warnings.filterwarnings("ignore")
KUMAGAI_MAE_ENDPOINT = 0.34   # K21 abstract, Fig. 5(a) label, Sec. III D (q = 0, N = 700)
KUMAGAI_MAE_INFINITY = 0.19   # K21 Sec. III D: extrapolated N -> infinity, q = 0


def write_result(name, payload, *, config):
    """write_result, waiting for a clean tree instead of discarding hours of compute."""
    import time

    while True:
        try:
            return _write_result(name, payload, config=config)
        except DirtyTreeError:
            print("working tree dirty; commit to let the result be written (retry in 60 s)", flush=True)
            time.sleep(60)


def _manifest_sha() -> str:
    import hashlib

    from dftgnn.split import SPLITS_DIR

    return hashlib.sha256((SPLITS_DIR / "MANIFEST.sha256").read_bytes()).hexdigest()


def _kiyohara_block(frame, info, cfg):
    return {**info, "metrics": metrics_block(frame, n_boot=cfg.stats.cluster_bootstrap_n,
                                             seed=cfg.split.seeds[0])}


def _common(cfg):
    return {"splits_manifest_sha256": _manifest_sha(), "budgets_hosts_requested": cfg.budgets.hosts,
            "pool_size": load_meta()["pool_size"], "bootstrap": {
                "cluster_n": cfg.stats.cluster_bootstrap_n, "ci": cfg.stats.ci,
                "seed": cfg.split.seeds[0]}}


# ------------------------------------------------------------------------------------ oracle
def oracle(cfg, data):
    n_boot = cfg.stats.cluster_bootstrap_n
    rows, stats, spreads = [], [], []
    splits = [(f"outer_r{r}", load_split(f"outer_r{r}")["test"]) for r in range(cfg.split.n_outer_resamples)]
    splits.append(("kiyohara", load_split("kiyohara")["test"]))
    for name, test in splits:
        te = data.uni.iloc[data.rows(test)]
        pred = te.groupby("host_id").target_Ef_eV.transform("mean")
        fr = pd.DataFrame({"host_id": te.host_id.to_numpy(), "y_true": te.target_Ef_eV.to_numpy(),
                           "y_pred": pred.to_numpy()})
        spread = within_group_spread(te.target_Ef_eV, te.host_id)
        rec = {"split": name, "n_test_hosts": te.host_id.nunique(), "n_test_sites": len(te),
               "n_test_hosts_ge2_sites": spread["n_groups_with_ge2"],
               "metrics": metrics_block(fr, n_boot=n_boot, seed=cfg.split.seeds[0]),
               "within_host_sd_mean_eV": spread["sd_eV"]["mean"],
               "within_host_range_mean_eV": spread["range_eV"]["mean"]}
        if name == "kiyohara":
            kio = rec
        else:
            rows.append(rec)
            stats.append(host_stat_table(fr))
        spreads.append(spread)
    u = data.uni
    payload = {
        "definition": ("Descriptive, not a model: every test site is predicted by its own host's TRUE "
                       "mean target over the host's sites (labels of the test hosts are used). The MAE is "
                       "the irreducible error of any predictor that cannot distinguish the O sites of a "
                       "host: it equals the mean absolute deviation of site energies about their host mean, "
                       "and singleton hosts contribute zero."),
        "per_resample": rows, "aggregate": M.aggregate_resamples(stats, n_boot=n_boot, ci=cfg.stats.ci,
                                                               seed=cfg.split.seeds[0]),
        "kiyohara": kio,
        "within_host_spread_universe": within_group_spread(u.target_Ef_eV, u.host_id),
        "within_host_spread_note": "sd uses ddof=1 over hosts with >= 2 sites; range = max - min of site energies",
        **_common(cfg),
    }
    write_result("host_mean_oracle", payload, config=cfg)


# ------------------------------------------------------------------------------------ B0
def b0(cfg, data):
    fit = physics.make_fit(data)
    frame, infos = run_outer(data, fit, cfg, seed_tag="b0")
    kframe, kinfo = run_kiyohara(data, fit, seed_tag="b0")
    full = np.arange(len(data.uni))
    allfit = fit(full, full, 0)["info"]["coef"]
    coef_by_budget = {}
    for b in sorted({i["budget_hosts"] for i in infos}):
        cs = pd.DataFrame([i["coef"] for i in infos if i["budget_hosts"] == b])
        coef_by_budget[str(b)] = {k: {"mean": float(cs[k].mean()), "sd": float(cs[k].std(ddof=1)),
                                      "n_resamples_positive": int((cs[k] > 0).sum())} for k in cs}
    s = coef_by_budget[str(max(int(k) for k in coef_by_budget))]
    payload = {
        "model": "B0: E_f = a + b * formation_energy + c * band_gap, ordinary least squares (3 coefficients: "
                 "intercept plus the two slopes)",
        "stability_descriptor": {"column": "formation_energy", "units": "eV/atom",
                                 "definition": "host formation energy per atom from the Materials Project, "
                                               "MP2020-corrected; more negative = more stable oxide",
                                 "locator": "K21 Sec. II A and II I; docs/descriptor_taxonomy.csv"},
        "gap_descriptor": {"column": "band_gap", "units": "eV",
                           "definition": "PBEsol(+U) band gap of the pristine host as released (not the nsc-dd "
                                         "hybrid gap)",
                           "locator": "K21 Sec. II I; docs/descriptor_taxonomy.csv"},
        "coefficients_by_budget": coef_by_budget,
        "coefficients_all_universe_fit_descriptive": allfit,
        "coefficients_kiyohara_train": kinfo["coef"],
        "interpretation_full_budget_mean": physics.interpret(
            s["stability_eV_per_eV_per_atom"]["mean"], s["gap_eV_per_eV"]["mean"]),
        "interpretation_caveat": "association in this sample, not a mechanism; the two descriptors are correlated.",
        "outer": summarize_outer(frame, infos, cfg), "kiyohara": _kiyohara_block(kframe, kinfo, cfg),
        "predictions": write_predictions("b0_physics_floor", pd.concat([frame, kframe], ignore_index=True)),
        **_common(cfg),
    }
    write_result("b0_physics_floor", payload, config=cfg)


# ------------------------------------------------------------------------------------ RF
def rf_variant(cfg, data, variant):
    feats = feature_sets(cfg, data)[variant]
    fit = rf.make_fit(data, feats)
    frame, infos = run_outer(data, fit, cfg, seed_tag=f"rf_{variant}")
    kframe, kinfo = run_kiyohara(data, fit, seed_tag=f"rf_{variant}")
    payload = {
        "model": f"RF-{variant}", "features": feats, "n_features": len(feats),
        "feature_classes": sorted({data.desc_class[f] for f in feats}),
        "hyperparameters": {"n_estimators": rf.N_TREES, "max_features_grid": rf.max_features_grid(len(feats)),
                            "cv": f"ShuffleSplit(n_splits={rf.CV_SPLITS}, test_size={rf.CV_TEST_SIZE}) on sites, "
                                  "as released by K21", "scoring": "neg_mean_squared_error",
                            "others": "scikit-learn defaults", "trees_source": "reported in K21 Sec. II I (400)"},
        "importance_note": "impurity-based from the refit forest; permutation on the TEST hosts, "
                           f"{rf.PERM_REPEATS} repeats, scoring = MAE (increase in eV when the column is shuffled)",
        "outer": summarize_outer(frame, infos, cfg), "kiyohara": _kiyohara_block(kframe, kinfo, cfg),
        "predictions": write_predictions(f"rf_{variant}", pd.concat([frame, kframe], ignore_index=True)),
        **_common(cfg),
    }
    write_result(f"rf_{variant}", payload, config=cfg)


# ------------------------------------------------------------------------------------ curve
def curve(cfg, data):
    cur = load_split("rf_curve")
    fit = rf.make_fit(data, feature_sets(cfg, data)["kumagai"], importances=False)
    frames, infos = [], []
    for i in range(cfg.curve.n_resamples):
        for n in cfg.curve.sizes:
            train, test = curve_resample(cur, i, n)
            seed = derived_seed(i, f"rf_curve_{n}") % 2**31
            fr, info = run_split(data, fit, train, test, seed, resample=i, budget=n)
            frames.append(fr)
            infos.append({"resample": i, "size": n, "seed": seed, **{k: v for k, v in info.items() if k != "importance"},
                          "impurity": {f: v["impurity"] for f, v in info["importance"].items()}})
            print(f"curve i={i} N={n}", flush=True)
    frame = pd.concat(frames, ignore_index=True)
    n_boot = cfg.stats.cluster_bootstrap_n
    out = {}
    for n, fn in frame.groupby("budget"):
        stats = [host_stat_table(g) for _, g in fn.groupby("resample")]
        agg = M.aggregate_resamples(stats, n_boot=n_boot, ci=cfg.stats.ci, seed=cfg.split.seeds[0])
        imp = pd.DataFrame([i["impurity"] for i in infos if i["size"] == n]).mean().sort_values(ascending=False)
        out[str(n)] = {"aggregate": agg, "mae_sd_across_resamples_kumagai_style": agg["mae"]["sd"],
                       "mean_impurity_importance_top10": {k: float(v) for k, v in imp.head(10).items()}}
    sizes = sorted(int(k) for k in out)
    n1, n2 = (220, sizes[-1])
    m1, m2 = out[str(n1)]["aggregate"]["mae"]["mean"], out[str(n2)]["aggregate"]["mae"]["mean"]
    a = (m1 - m2) / (n1**-0.5 - n2**-0.5)
    b = m1 - a * n1**-0.5
    payload = {
        "protocol": ("Kumagai-style: nominal training sizes of K21 Fig. 5(a) (22, 70, 220, 700 oxides; 700 capped "
                     f"at the {sizes[-1]}-host pool), {cfg.curve.n_resamples} grouped random resamples per "
                     f"size, {cfg.curve.n_test_hosts} test hosts per resample, unstratified; RF as in rf_kumagai"),
        "by_size": out, "sizes": sizes,
        "extrapolation_a_N^-0.5_plus_b": {"through_sizes": [n1, n2], "a": a, "b_N_to_infinity_eV": b,
                                           "published_N_to_infinity_eV": KUMAGAI_MAE_INFINITY},
        "published_overlay": {
            "tabulated_in_text": {"mae_at_700_eV": KUMAGAI_MAE_ENDPOINT, "mae_N_to_infinity_eV": KUMAGAI_MAE_INFINITY,
                                  "locator": "K21 abstract, Fig. 5(a) labels, Sec. III D"},
            "not_used": "values at N = 22, 70, 220 exist only in Fig. 5(a); the figure is not digitised"},
        "predictions": write_predictions("rf_kumagai_curve", frame), "infos": infos, **_common(cfg),
    }
    write_result("rf_kumagai_curve", payload, config=cfg)


# ------------------------------------------------------------------------------------ sensitivity
def sens(cfg, data):
    thr = cfg.sensitivity.exclude_high_moment.threshold_muB
    keep = (data.uni.defect_moment_muB.abs() <= thr).to_numpy()
    n_excl = int((~keep).sum())
    sub = data.masked(keep)
    pool = load_meta()["pool_size"]
    fit = rf.make_fit(sub, feature_sets(cfg, sub)["kumagai"], importances=False)
    frame, infos = run_outer(sub, fit, cfg, seed_tag="rf_kumagai_sens", budgets=[pool])
    base = pd.read_parquet("results/predictions/rf_kumagai.parquet")
    base = base[(base["resample"] >= 0) & (base["budget"] == pool)]
    excluded_ids = set(data.uni.site_id[~keep])
    hi_test = base[base.site_id.isin(excluded_ids)]
    n_boot = cfg.stats.cluster_bootstrap_n
    A, Bm, C = [], [], []
    for r in range(cfg.split.n_outer_resamples):
        test_hosts = sorted(base[base["resample"] == r].host_id.unique())
        b_full = base[base["resample"] == r]
        b_red = b_full[~b_full.site_id.isin(excluded_ids)]
        red = frame[frame["resample"] == r]
        A.append(host_stat_table(red).reindex(test_hosts, fill_value=0.0))
        Bm.append(host_stat_table(b_full).reindex(test_hosts, fill_value=0.0))
        C.append(host_stat_table(b_red).reindex(test_hosts, fill_value=0.0))
    payload = {
        "definition": (f"RF-Kumagai at the full budget ({pool} hosts) with the {n_excl} entries with "
                       f"|defect moment| > {thr} muB removed from training and test (pre-registered in config). "
                       "delta = excluded-run MAE minus baseline MAE, paired over the same test hosts with "
                       "identical bootstrap draws."),
        "n_excluded_entries": n_excl, "excluded_site_ids": sorted(excluded_ids),
        "excluded_entries_in_test_by_resample": [int(hi_test[hi_test["resample"] == r].shape[0])
                                                 for r in range(cfg.split.n_outer_resamples)],
        "baseline_full": M.aggregate_resamples(Bm, n_boot=n_boot, ci=cfg.stats.ci, seed=cfg.split.seeds[0]),
        "excluded_run": M.aggregate_resamples(A, n_boot=n_boot, ci=cfg.stats.ci, seed=cfg.split.seeds[0]),
        "baseline_on_reduced_test_sites": M.aggregate_resamples(C, n_boot=n_boot, ci=cfg.stats.ci,
                                                                seed=cfg.split.seeds[0]),
        "delta_total_excluded_minus_baseline": M.aggregate_delta(A, Bm, n_boot=n_boot, ci=cfg.stats.ci,
                                                                 seed=cfg.split.seeds[0]),
        "delta_evaluation_only_baseline_reduced_minus_baseline": M.aggregate_delta(
            C, Bm, n_boot=n_boot, ci=cfg.stats.ci, seed=cfg.split.seeds[0]),
        "delta_training_only_excluded_minus_baseline_reduced": M.aggregate_delta(
            A, C, n_boot=n_boot, ci=cfg.stats.ci, seed=cfg.split.seeds[0]),
        "runs": infos, "predictions": write_predictions("rf_kumagai_sens_high_moment", frame),
        **_common(cfg),
    }
    write_result("sens_high_moment_rf", payload, config=cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["oracle", "b0", "rf", "curve", "sens"])
    ap.add_argument("--variant", choices=["kumagai", "electronic", "structural"])
    a = ap.parse_args()
    cfg = load_config()
    data = Data.load(cfg)
    if a.what == "rf":
        rf_variant(cfg, data, a.variant)
    else:
        {"oracle": oracle, "b0": b0, "curve": curve, "sens": sens}[a.what](cfg, data)


if __name__ == "__main__":
    main()
