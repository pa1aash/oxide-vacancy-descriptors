"""Grouped K-fold CV at K = 10 and 20 and the extended-budget statement.

Rules:, rows of 2026-10-11 and 2026-10-14 (implementation, items b, c, e, with no test.

Quantities per backbone, K and model: the pooled out-of-fold prediction of every site (seed-ensemble mean over
seeds 0-2 of the fold that holds its host); site MAE, within-host residual MAE and host-mean MAE on those
predictions; A = MAE_S - MAE_D and A_within. Uncertainty: paired host-cluster bootstrap over the 818 hosts, 2000
draws of multinomial host weights (``rng.multinomial(818, uniform, size=2000)``, generator
``numpy.random.default_rng(20261008)``), one draw set shared by S and D, both K, both backbones and all metrics;
percentile intervals (numpy linear), one-sided 95% upper bound = 95th percentile, strict comparison with delta.
Fold-level SD: sample SD over the K folds of the per-fold A (seed ensemble on the fold's test sites).

Extended-budget statement, per backbone: holds only if the one-sided upper bound on A is below delta at B =
654, at K = 10 (about 736 training hosts) and at K = 20 (about 777).

Reads ``results/analysis/cv_eval_long.parquet`` and ``precision_cv_runs.parquet`` (sha256 checked) and
``results/precision_a_b400_b654.json``. Writes through ``write_result``:

    results/cv_extended_budget.json     v2 and v1 blocks, statement, GPU-class and capped tables
    results/cv_g1_v1.json, results/cv_g1_v2.json     figure inputs (schema of results/c1c2_<arm>.json)

    PYTHONPATH=$PWD/src python analysis/cv_analysis.py
"""
from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import a1_common as c

KS = (10, 20)
N_HOSTS = 818
N_SEEDS = 3
METRICS = ("mae", "within_host_mae", "host_mean_mae")


def pooled_oof(cv: pd.DataFrame, n_seeds: int = N_SEEDS) -> pd.DataFrame:
    """Per (model, K, site): seed-ensemble out-of-fold prediction; each site exactly once per (model, K)."""
    g = cv.groupby(["model", "K", "site_id"], sort=True)
    if (g.fold.nunique() != 1).any():
        raise AssertionError("a site is tested in more than one fold")
    if (g.y_true.nunique() != 1).any() or (g.host_id.nunique() != 1).any():
        raise AssertionError("a site carries different targets or hosts across seeds")
    out = g.agg(host_id=("host_id", "first"), y_true=("y_true", "first"), y_pred=("y_pred", "mean"),
                fold=("fold", "first"), n_seeds=("seed", "nunique")).reset_index()
    if (out.n_seeds != n_seeds).any():
        raise AssertionError("a site lacks seeds")
    return out


def host_matrix(frame: pd.DataFrame, hosts: list[str]) -> np.ndarray:
    """Per-host sufficient statistics in the order ``hosts`` (every host must be present)."""
    from dftgnn.stats.metrics import host_stats

    st = host_stats(frame.y_true.to_numpy(), frame.y_pred.to_numpy(), frame.host_id.to_numpy())
    if list(st.index) != sorted(hosts) or list(hosts) != sorted(hosts):
        raise AssertionError("host rows differ from the host list")
    return st.to_numpy()


def host_draws(n_hosts: int, n_boot: int = c.N_BOOT, seed: int = c.BOOT_SEED) -> np.ndarray:
    """(n_boot, n_hosts) multinomial host weights: hosts drawn with replacement, all their sites kept."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_hosts, np.full(n_hosts, 1.0 / n_hosts), size=n_boot).astype(float)


def metric_point(stats: np.ndarray, metric: str) -> float:
    from dftgnn.stats.metrics import _metrics_from_totals
    return float(_metrics_from_totals(stats.sum(axis=0))[metric])


def metric_reps(stats: np.ndarray, w: np.ndarray, metric: str) -> np.ndarray:
    from dftgnn.stats.metrics import _metrics_from_totals
    return np.asarray(_metrics_from_totals(w @ stats)[metric], float)


def summary(point: float, reps: np.ndarray) -> dict:
    return {"mean": point, "ci": c.quantiles(reps)["ci95"], "upper_one_sided": c.quantiles(reps)["upper_one_sided_95"],
            "boot_sd": float(reps.std(ddof=1))}


def paired(stats_a: np.ndarray, stats_b: np.ndarray, w: np.ndarray, metric: str) -> dict:
    """Marginals of a and b and their paired difference a - b on one draw set."""
    ra, rb = metric_reps(stats_a, w, metric), metric_reps(stats_b, w, metric)
    pa, pb = metric_point(stats_a, metric), metric_point(stats_b, metric)
    return {"a": summary(pa, ra), "b": summary(pb, rb), "diff": summary(pa - pb, ra - rb)}


def fold_level(cv: pd.DataFrame, k: int) -> dict:
    """Per-fold A and A_within (seed ensemble on each fold's test sites) and their sample SD over folds."""
    from dftgnn.stats.metrics import point_metrics

    per, per_w = [], []
    for f in range(k):
        m = {}
        for model in c.MODELS:
            g = cv[(cv.K == k) & (cv.fold == f) & (cv.model == model)]
            e = g.groupby("site_id").agg(host_id=("host_id", "first"), y_true=("y_true", "first"),
                                         y_pred=("y_pred", "mean"))
            m[model] = point_metrics(e.y_true, e.y_pred, e.host_id)
        per.append(m["S"]["mae"] - m["D"]["mae"])
        per_w.append(m["S"]["within_host_mae"] - m["D"]["within_host_mae"])
    return {"A_per_fold": per, "A_fold_sd": float(np.std(per, ddof=1)), "A_fold_mean": float(np.mean(per)),
            "A_within_per_fold": per_w, "A_within_fold_sd": float(np.std(per_w, ddof=1))}


def analyse_backbone(cv: pd.DataFrame, hosts: list[str], w: np.ndarray, x_train: Mapping[int, float]) -> dict:
    out = {}
    for k in KS:
        oof = pooled_oof(cv[cv.K == k])
        n_sites = oof.groupby("model").site_id.nunique().to_dict()
        st = {m: host_matrix(oof[oof.model == m], hosts) for m in c.MODELS}
        const = host_matrix(c.constant_within_host(oof[oof.model == "S"]), hosts)
        res = {met: paired(st["S"], st["D"], w, met) for met in METRICS}
        skill = {m: paired(const, st[m], w, "within_host_mae")["diff"] for m in c.MODELS}
        out[f"k{k}"] = {
            "K": k, "x_training_hosts": x_train[k], "n_hosts": len(hosts), "n_sites": n_sites,
            "MAE_S": res["mae"]["a"], "MAE_D": res["mae"]["b"], "A": res["mae"]["diff"],
            "within_host_S": res["within_host_mae"]["a"], "within_host_D": res["within_host_mae"]["b"],
            "A_within": res["within_host_mae"]["diff"],
            "host_mean_S": res["host_mean_mae"]["a"], "host_mean_D": res["host_mean_mae"]["b"],
            "A_host_mean": res["host_mean_mae"]["diff"],
            "within_host_skill_S": skill["S"], "within_host_skill_D": skill["D"],
            "fold_level": fold_level(cv, k),
            "below_delta": c.passes(res["mae"]["diff"]["upper_one_sided"]),
            "below_delta_within": c.passes(res["within_host_mae"]["diff"]["upper_one_sided"]),
        }
    return out


def statement(points: Mapping[str, float], delta: float = c.DELTA_EV) -> dict:
    """Extended-budget statement from the one-sided upper bounds at '654', 'k10', 'k20'."""
    passing = [p for p, u in points.items() if c.passes(u, delta)]
    failing = [p for p, u in points.items() if not c.passes(u, delta)]
    return {"upper_one_sided": dict(points), "passing_points": passing, "failing_points": failing,
            "holds": not failing}


def figure_record(bb: str, block: dict) -> dict:
    def ms(s):
        return {"mean": s["mean"], "ci": s["ci"]}
    keys = [f"k{k}" for k in KS]
    return {"arm": bb, "label": c.LABEL, "design": "grouped K-fold CV (N3), pooled out-of-fold, seeds 0-2",
            "delta_eV": c.DELTA_EV, "budgets": keys,
            "x_training_hosts": {kk: block[kk]["x_training_hosts"] for kk in keys},
            "mae": {kk: {"S": {"mae": ms(block[kk]["MAE_S"]), "within_host_mae": ms(block[kk]["within_host_S"]),
                               "within_host_skill": ms(block[kk]["within_host_skill_S"])},
                         "D-state": {"mae": ms(block[kk]["MAE_D"]), "within_host_mae": ms(block[kk]["within_host_D"]),
                                     "within_host_skill": ms(block[kk]["within_host_skill_D"])}} for kk in keys},
            "A_mae": {kk: {"A": block[kk]["A"]["mean"], "ci95": block[kk]["A"]["ci"],
                           "upper_one_sided_95": block[kk]["A"]["upper_one_sided"],
                           "fold_sd": block[kk]["fold_level"]["A_fold_sd"]} for kk in keys},
            "A_within_host": {kk: {"A": block[kk]["A_within"]["mean"], "ci95": block[kk]["A_within"]["ci"],
                                   "upper_one_sided_95": block[kk]["A_within"]["upper_one_sided"]} for kk in keys},
            "source_record": "results/cv_extended_budget.json"}


def main() -> None:
    from dftgnn.config import load_config
    from dftgnn.io.results import write_result
    from dftgnn.split import load_split

    cfg = load_config()
    assert cfg.stats.delta_eV == c.DELTA_EV
    cv, src = c.load_table("cv_eval_long")
    runs, src_runs = c.load_table("precision_cv_runs")
    split = load_split("cv_v1")
    hosts = sorted({h for f in split["by_k"]["10"]["folds"] for h in f["test"]})
    assert len(hosts) == N_HOSTS
    x_train = {k: float(np.mean([f["n_train"] for f in split["by_k"][str(k)]["folds"]])) for k in KS}
    w = host_draws(N_HOSTS)
    blocks = {bb: analyse_backbone(cv[cv.backbone == bb], hosts, w, x_train) for bb in c.BACKBONES}

    prec_rec = c.record("precision_a_b400_b654")
    prec = prec_rec["payload"]
    g1 = c.record("g1_primary")["payload"]["per_budget"]["654"]["A"]
    v2c = c.record("c1c2_v2")["payload"]["A_mae"]["654"]
    p654 = {"v1": {"A": g1["mean"], "ci95": g1["ci"], "upper_one_sided": g1["upper_one_sided"],
                   "source": "results/g1_primary.json per_budget.654.A (registered G1 result, 10 x 3)",
                   "pipeline_10x3_upper": prec["v1"]["10x3"]["654"]["A"]["upper_one_sided"]},
            "v2": {"A": prec["v2"]["10x3"]["654"]["A"]["mean"], "ci95": prec["v2"]["10x3"]["654"]["A"]["ci"],
                   "upper_one_sided": prec["v2"]["10x3"]["654"]["A"]["upper_one_sided"],
                   "source": "results/precision_a_b400_b654.json v2.10x3.654.A (G1 estimator on the v2 10 x 3 "
                             "runs; POST HOC backbone)",
                   "c1c2_v2_upper_one_sided_95": v2c["upper_one_sided_95"]}}
    stmt = {}
    for bb in c.BACKBONES:
        pts = {"654": p654[bb]["upper_one_sided"], "k10": blocks[bb]["k10"]["A"]["upper_one_sided"],
               "k20": blocks[bb]["k20"]["A"]["upper_one_sided"]}
        stmt[bb] = {**statement(pts), "point_654": p654[bb],
                    "x_training_hosts": {"654": 654.0, "k10": x_train[10], "k20": x_train[20]}}
    stmt_within = {bb: statement({"k10": blocks[bb]["k10"]["A_within"]["upper_one_sided"],
                                  "k20": blocks[bb]["k20"]["A_within"]["upper_one_sided"]}) for bb in c.BACKBONES}
    payload = c.to_jsonable({
        "label": c.LABEL,
        "rules": "docs/deviations.md rows 2026-10-11 (§7, §11) and 2026-10-14 (§7, §11)",
        "delta_eV": c.DELTA_EV, "Ks": list(KS), "x_training_hosts": {f"k{k}": x_train[k] for k in KS},
        "estimator": {"bootstrap": "paired host-cluster over the 818 hosts", "draws": c.N_BOOT,
                      "generator": f"numpy.random.default_rng({c.BOOT_SEED}).multinomial(818, uniform, size=2000)",
                      "shared": "S and D, K = 10 and 20, both backbones, all metrics",
                      "two_sided_ci": c.CI, "one_sided_upper_quantile": c.UPPER_Q,
                      "prediction": "pooled out-of-fold seed-ensemble mean (seeds 0-2)"},
        "v2": blocks["v2"], "v1": blocks["v1"],
        "extended_budget_statement": {
            "rule": "holds only if the one-sided upper bound on A is below delta at B = 654 (the registered G1 "
                    "result), at about 736 hosts (K = 10) and at about 777 hosts (K = 20); failing points named",
            **stmt},
        "within_host_bounds_descriptive": stmt_within,
        "fold_partition": {str(k): True for k in KS},
        "descriptive": {
            "gpu_class": {f"cv|{k}": c.gpu_table(runs, "cv", k) for k in KS},
            "capped": {f"cv|{k}": c.capped_table(runs, "cv", k) for k in KS},
            "note": "descriptive, no test (2026-10-14 row, item e)"},
        "inputs": {"cv_eval_long": src, "precision_cv_runs": src_runs,
                   "precision_a_b400_b654": {"git_sha": prec_rec["git_sha"]}},
    })
    print(json.dumps(payload["extended_budget_statement"], indent=1))
    print(write_result("cv_extended_budget", payload, config=cfg, repo_root=c.ROOT))
    for bb in c.BACKBONES:
        print(write_result(f"cv_g1_{bb}", figure_record(bb, payload[bb]), config=cfg, repo_root=c.ROOT))


if __name__ == "__main__":
    main()
