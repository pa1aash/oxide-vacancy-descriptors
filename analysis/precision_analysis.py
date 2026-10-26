"""Precision extension: A = MAE_S - MAE_D at B = 400 and 654 over 30 resamples x 5 seeds.

Rules:, rows of 2026-10-11 and 2026-10-14 (implementation, items d, e, with no test.

Estimator: the G1 estimator of ``dftgnn.stats.g1`` unchanged. Seed-ensemble mean per (model, r, B, site); per
resample host statistics; paired hierarchical bootstrap (``HierDraws.make``: 2000 replicates, seed 20261008,
resamples with replacement, then test hosts within each drawn resample); the replicate value is the mean over
the drawn resamples. One draw set over r = 0..29 is shared by both budgets, models, metrics and backbones; the
10-resample subsets use the draw set over r = 0..9 (the G1 primary draws). Pooled set 30 x 5; beside-series 10
x 3, 10 x 5 and 30 x 3.

Reads ``results/analysis/precision_eval_long.parquet`` and ``precision_cv_runs.parquet`` (sha256 checked against
``results/precision_cv_eval_table.json``). Writes through ``write_result``:

    results/precision_a_b400_b654.json     v2 and v1 blocks: every subset, both budgets, sanity checks,
                                           GPU-class and capped-versus-uncapped tables
    results/precision_g1_v1.json           figure input (schema of results/c1c2_<arm>.json), pooled 30 x 5
    results/precision_g1_v2.json

    PYTHONPATH=$PWD/src python analysis/precision_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import a1_common as c

BUDGETS = (400, 654)
N_HOSTS_PER_RESAMPLE = 164
METRICS = ("mae", "within_host_mae", "host_mean_mae")
SANITY_TOL = 1e-6


def subset(df: pd.DataFrame, n_resamples: int, n_seeds: int) -> pd.DataFrame:
    """Resamples 0..n_resamples-1 and seeds 0..n_seeds-1; every (model, r, B) must hold all n_seeds seeds."""
    sub = df[(df.r < n_resamples) & (df.seed < n_seeds)]
    per = sub.groupby(["model", "r", "B"]).seed.nunique()
    if (per != n_seeds).any() or per.index.get_level_values("r").nunique() != n_resamples:
        raise AssertionError(f"subset {n_resamples} x {n_seeds}: incomplete cells")
    return sub


def resample_stats_by_budget(ens: pd.DataFrame, model: str, n_resamples: int, budgets=BUDGETS,
                             constant: bool = False) -> dict[int, list[np.ndarray]]:
    from dftgnn.stats.g1 import resample_stats

    out = {}
    for B in budgets:
        frames = [ens[(ens.model == model) & (ens.r == r) & (ens.B == B)] for r in range(n_resamples)]
        out[B] = [resample_stats(c.constant_within_host(f) if constant else f) for f in frames]
    return out


def analyse(df: pd.DataFrame, n_resamples: int, n_seeds: int, draws, budgets=BUDGETS) -> dict:
    """Per budget: MAE_S, MAE_D, A and the within-host and host-mean analogues (G1 ``contrast`` summaries)."""
    from dftgnn.stats.g1 import contrast, seed_ensemble

    sub = subset(df, n_resamples, n_seeds)
    ens = seed_ensemble(sub[["model", "r", "B", "site_id", "host_id", "seed", "y_true", "y_pred"]])
    if (ens.n_seeds != n_seeds).any():
        raise AssertionError("seed ensemble over the wrong number of seeds")
    st = {m: resample_stats_by_budget(ens, m, n_resamples, budgets) for m in c.MODELS}
    const = resample_stats_by_budget(ens, "S", n_resamples, budgets, constant=True)
    out = {}
    for metric in METRICS:
        out[metric] = contrast(st["S"], st["D"], draws, metric, ci=c.CI, upper_quantile=c.UPPER_Q)
    skill = {m: contrast(const, st[m], draws, "within_host_mae", ci=c.CI, upper_quantile=c.UPPER_Q)
             for m in c.MODELS}
    res = {}
    for B in budgets:
        res[str(B)] = {
            "MAE_S": out["mae"][B]["a"], "MAE_D": out["mae"][B]["b"], "A": out["mae"][B]["diff"],
            "within_host_S": out["within_host_mae"][B]["a"], "within_host_D": out["within_host_mae"][B]["b"],
            "A_within": out["within_host_mae"][B]["diff"],
            "host_mean_S": out["host_mean_mae"][B]["a"], "host_mean_D": out["host_mean_mae"][B]["b"],
            "A_host_mean": out["host_mean_mae"][B]["diff"],
            "within_host_skill_S": skill["S"][B]["diff"], "within_host_skill_D": skill["D"][B]["diff"],
            "n_resamples": n_resamples, "n_seeds": n_seeds,
            "below_delta": c.passes(out["mae"][B]["diff"]["upper_one_sided"]),
            "below_delta_within": c.passes(out["within_host_mae"][B]["diff"]["upper_one_sided"]),
        }
    return res


def sanity(blocks: dict, root: Path) -> dict:
    """Sanity."""
    g1 = c.record("g1_primary", root)["payload"]["per_budget"]
    v2 = c.record("c1c2_v2", root)["payload"]
    v1c = c.record("c1c2_v1", root)["payload"]
    rows = {}
    for B in map(str, BUDGETS):
        for bb in c.BACKBONES:
            mine = blocks[bb]["10x3"][B]
            if bb == "v1":
                ref = {"A": g1[B]["A"]["mean"], "MAE_S": g1[B]["MAE_S"]["mean"], "MAE_D": g1[B]["MAE_D"]["mean"],
                       "A_ci_lo": g1[B]["A"]["ci"][0], "A_ci_hi": g1[B]["A"]["ci"][1],
                       "A_upper": g1[B]["A"]["upper_one_sided"], "A_within": v1c["A_within_host"][B]["A"]}
                src = "results/g1_primary.json per_budget (A_within: results/c1c2_v1.json A_within_host)"
            else:
                ref = {"A": v2["A_mae"][B]["A"], "MAE_S": v2["mae"][B]["S"]["mae"]["mean"],
                       "MAE_D": v2["mae"][B]["D-state"]["mae"]["mean"], "A_within": v2["A_within_host"][B]["A"]}
                src = "results/c1c2_v2.json A_mae, mae, A_within_host"
            got = {"A": mine["A"]["mean"], "MAE_S": mine["MAE_S"]["mean"], "MAE_D": mine["MAE_D"]["mean"],
                   "A_ci_lo": mine["A"]["ci"][0], "A_ci_hi": mine["A"]["ci"][1],
                   "A_upper": mine["A"]["upper_one_sided"], "A_within": mine["A_within"]["mean"]}
            diff = {k: float(got[k] - ref[k]) for k in ref}
            rows[f"{bb}|{B}"] = {"reference": src, "pipeline": {k: got[k] for k in ref}, "reference_values": ref,
                                 "difference": diff, "max_abs_difference": float(max(abs(v) for v in diff.values())),
                                 "within_1e-6": bool(all(abs(v) <= SANITY_TOL for v in diff.values()))}
    return rows


def sanity_eval_table(prec: pd.DataFrame, root: Path) -> dict:
    """v1 10 x 3 seed predictions against the per-seed columns of results/analysis/eval_table.parquet."""
    ev = pd.read_parquet(root / "results" / "analysis" / "eval_table.parquet")
    ev = ev[ev.model.isin(c.MODELS) & ev.B.isin(BUDGETS)]
    long = ev.melt(id_vars=["model", "r", "B", "site_id"], value_vars=[f"y_pred_seed{s}" for s in range(3)],
                   var_name="seed", value_name="y_ref")
    long["seed"] = long.seed.str[-1].astype(int)
    mine = prec[(prec.backbone == "v1") & (prec.r < 10) & (prec.seed < 3)]
    j = mine.merge(long, on=["model", "r", "B", "seed", "site_id"], how="outer", indicator=True)
    both = j[j._merge == "both"]
    return {"rows_compared": len(both), "rows_only_in_one": int((j._merge != "both").sum()),
            "max_abs_difference": float((both.y_pred - both.y_ref).abs().max()),
            "reference": "results/analysis/eval_table.parquet (y_pred_seed0..2 of S and D at B = 400, 654)"}


def figure_record(bb: str, pooled: dict, beside: dict) -> dict:
    """c1c2-schema payload; top-level = pooled 30 x 5."""
    def ms(s):
        return {"mean": s["mean"], "ci": s["ci"]}
    mae, a_mae, a_w = {}, {}, {}
    for B in map(str, BUDGETS):
        p = pooled[B]
        mae[B] = {"S": {"mae": ms(p["MAE_S"]), "within_host_mae": ms(p["within_host_S"]),
                        "within_host_skill": ms(p["within_host_skill_S"])},
                  "D-state": {"mae": ms(p["MAE_D"]), "within_host_mae": ms(p["within_host_D"]),
                              "within_host_skill": ms(p["within_host_skill_D"])}}
        a_mae[B] = {"A": p["A"]["mean"], "ci95": p["A"]["ci"], "upper_one_sided_95": p["A"]["upper_one_sided"],
                    "per_resample": p["A"]["per_resample"], "sd_across_resamples": p["A"]["sd_over_resamples"]}
        a_w[B] = {"A": p["A_within"]["mean"], "ci95": p["A_within"]["ci"],
                  "upper_one_sided_95": p["A_within"]["upper_one_sided"]}
    return {"arm": bb, "label": c.LABEL, "design": "precision extension (N4), pooled 30 resamples x 5 seeds",
            "delta_eV": c.DELTA_EV, "budgets": list(BUDGETS), "x_training_hosts": {str(B): B for B in BUDGETS},
            "mae": mae, "A_mae": a_mae, "A_within_host": a_w,
            "beside_subsets": {k: {B: {"A": v[B]["A"]["mean"], "ci95": v[B]["A"]["ci"],
                                       "upper_one_sided_95": v[B]["A"]["upper_one_sided"]} for B in v}
                               for k, v in beside.items()},
            "source_record": "results/precision_a_b400_b654.json"}


def main() -> None:
    from dftgnn.config import load_config
    from dftgnn.io.results import write_result
    from dftgnn.stats.g1 import HierDraws

    cfg = load_config()
    assert cfg.stats.delta_eV == c.DELTA_EV
    prec, src = c.load_table("precision_eval_long")
    runs, src_runs = c.load_table("precision_cv_runs")
    draws = {n: HierDraws.make([N_HOSTS_PER_RESAMPLE] * n, c.N_BOOT, seed=c.BOOT_SEED) for n in (10, 30)}
    blocks = {}
    for bb in c.BACKBONES:
        df = prec[prec.backbone == bb]
        blocks[bb] = {name: analyse(df, n_r, n_s, draws[n_r]) for name, n_r, n_s in c.subset_names()}
    checks = sanity(blocks, c.ROOT)
    ev_check = sanity_eval_table(prec, c.ROOT)
    statement = {bb: {"rule": "the non-inferiority bound holds at 654 only if the pooled 30 x 5 one-sided upper "
                              "bound on A at B = 654 is below delta (strict)",
                      "upper_one_sided_654": blocks[bb]["30x5"]["654"]["A"]["upper_one_sided"],
                      "holds_at_654": blocks[bb]["30x5"]["654"]["below_delta"],
                      "upper_one_sided_400": blocks[bb]["30x5"]["400"]["A"]["upper_one_sided"],
                      "below_delta_at_400": blocks[bb]["30x5"]["400"]["below_delta"]} for bb in c.BACKBONES}
    payload = c.to_jsonable({
        "label": c.LABEL,
        "rules": "docs/deviations.md rows 2026-10-11 (§7, §11) and 2026-10-14 (§7, §11)",
        "delta_eV": c.DELTA_EV, "budgets": list(BUDGETS),
        "estimator": {"module": "dftgnn.stats.g1", "draws": c.N_BOOT, "seed": c.BOOT_SEED,
                      "draw_sets": {"30": "r = 0..29, shared by budgets, models, metrics, backbones",
                                    "10": "r = 0..9 (the G1 primary draws)"},
                      "two_sided_ci": c.CI, "one_sided_upper_quantile": c.UPPER_Q,
                      "point": "mean over resamples of the per-resample metric of the seed-ensemble prediction"},
        "subsets": {name: {"n_resamples": n_r, "n_seeds": n_s} for name, n_r, n_s in c.subset_names()},
        "pooled": "30x5",
        "v2": blocks["v2"], "v1": blocks["v1"],
        "statement_654": statement,
        "sanity": {"registered_10x3": checks, "v1_seed_predictions_vs_eval_table": ev_check,
                   "note": "the 10 x 3 cells are the original sweep runs in both backbones; no precision-stage "
                           "spec repeats a sweep spec, so no GPU-class difference can enter this comparison"},
        "descriptive": {
            "gpu_class": {f"precision|{B}": c.gpu_table(runs, "precision", B) for B in BUDGETS},
            "capped": {f"precision|{B}": c.capped_table(runs, "precision", B) for B in BUDGETS},
            "note": "descriptive, no test (2026-10-14 row, item e)"},
        "inputs": {"precision_eval_long": src, "precision_cv_runs": src_runs},
    })
    print(json.dumps({"statement_654": payload["statement_654"], "sanity": payload["sanity"]}, indent=1))
    print(write_result("precision_a_b400_b654", payload, config=cfg, repo_root=c.ROOT))
    for bb in c.BACKBONES:
        beside = {k: payload[bb][k] for k in ("10x3", "10x5", "30x3")}
        print(write_result(f"precision_g1_{bb}", figure_record(bb, payload[bb]["30x5"], beside), config=cfg,
                           repo_root=c.ROOT))


if __name__ == "__main__":
    main()
