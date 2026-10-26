"""python scripts/analysis_c1c2.py --arm v2            v2 arm (results/v2_sweep/)
    add --no-write to print only

For each budget B: the seed-ensemble-mean prediction of S and D per outer resample on identical test sites,
the advantage A = MAE_S - MAE_D (positive: D better) with the paired hierarchical bootstrap of
``dftgnn.stats.metrics.aggregate_delta`` (``analysis.bootstrap.draws`` draws, resamples then test hosts). The
one-sided 95% upper bound is the 95th percentile of the bootstrap distribution, obtained as the upper end of
the two-sided 90% interval of the same draws (same seed). N* is the smallest B whose upper bound is below
delta and stays below delta at every larger budget; if the first qualifying budget is the largest, N* is
reported as reached only at the maximum budget; if none qualifies, N* > max budget. Is A at the largest budget
with its two-sided 95% interval. Everything is repeated for the within-host residual MAE (secondary), and each
model's within-host skill (the within-host residual MAE of a host-constant predictor minus the model's, paired
on the same hosts; 0 means no skill) is reported per budget. The LOCO advantage (results/loco for v1,
results/loco_v2 for v2) is reported next to N*, with a test-host bootstrap of the pooled folds. Only complete
cells enter: a (model, resample, budget) needs all its seeds.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import diagnostics_sweep as D

from dftgnn.config import load_config
from dftgnn.stats.metrics import _draw_weights, aggregate_delta, aggregate_resamples, host_stats

ARMS = {"v1": "", "v2": "v2_sweep"}
LOCO = {"v1": "loco", "v2": "loco_v2"}


def complete_cells(ix, models, n_seeds):
    ok = ix.groupby(["model", "B", "r"]).seed.nunique()
    ok = ok[ok == n_seeds].reset_index()
    cells = {}
    for (b, r), g in ok.groupby(["B", "r"]):
        if set(g.model) >= set(models):
            cells.setdefault(int(b), []).append(int(r))
    return cells


def a_block(st_s, st_d, metric, cfg) -> dict:
    n, seed = cfg.analysis.bootstrap.draws, 0
    two = aggregate_delta(st_s, st_d, n_boot=n, ci=cfg.analysis.display_ci, seed=seed)[metric]
    one = aggregate_delta(st_s, st_d, n_boot=n, ci=1 - 2 * (1 - cfg.analysis.n_star.upper_quantile), seed=seed)[metric]
    return {"A": two["mean"], "ci95": two["ci"], "upper_one_sided_95": one["ci"][1],
            "per_resample": two["per_resample"], "sd_across_resamples": two["sd"]}


def n_star(rows: dict, delta: float, key: str = "upper_one_sided_95") -> dict:
    budgets = sorted(rows)
    qual = [b for b in budgets if rows[b][key] < delta]
    for b in budgets:
        if all(rows[c][key] < delta for c in budgets if c >= b):
            label = "reached only at the maximum budget" if b == budgets[-1] else "N*"
            return {"N_star": b, "label": label, "qualifying_budgets": qual}
    return {"N_star": None, "label": f"N* > {budgets[-1]}", "qualifying_budgets": qual}


def loco_advantage(arm: str, d: str, cfg) -> dict | None:
    ix = D.run_index(LOCO[arm])
    if ix.empty:
        return None
    ix["k"] = ix.split.map(lambda s: int(s.split("_f")[1]))
    s_fr, d_fr = [], []
    for k in sorted(ix.k.unique()):
        gs, gd = ix[(ix.model == "S") & (ix.k == k)], ix[(ix.model == d) & (ix.k == k)]
        if gs.seed.nunique() < len(cfg.robustness.loco.seeds) or gd.seed.nunique() < len(cfg.robustness.loco.seeds):
            return None
        s = D.ensemble(gs)
        s_fr.append(s)
        d_fr.append(D._align(D.ensemble(gd), s))
    import pandas as pd

    s, dd = pd.concat(s_fr, ignore_index=True), pd.concat(d_fr, ignore_index=True)
    sa, sb = host_stats(s.y_true, s.y_pred, s.host_id).to_numpy(), host_stats(dd.y_true, dd.y_pred, dd.host_id).to_numpy()
    w = _draw_weights(np.random.default_rng(0), sa.shape[0], cfg.analysis.bootstrap.draws)
    boot = (w @ sa[:, 1]) / (w @ sa[:, 0]) - (w @ sb[:, 1]) / (w @ sb[:, 0])
    point = sa[:, 1].sum() / sa[:, 0].sum() - sb[:, 1].sum() / sb[:, 0].sum()
    return {"A": float(point), "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "upper_one_sided_95": float(np.quantile(boot, 0.95)), "n_hosts": int(sa.shape[0]),
            "mae_S": float(sa[:, 1].sum() / sa[:, 0].sum()), "mae_D": float(sb[:, 1].sum() / sb[:, 0].sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS), required=True)
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    d = cfg.models.injection_mode
    delta = cfg.stats.delta_eV
    ix = D.run_index(ARMS[a.arm])
    ix = ix[ix.split.str.startswith("outer")]
    cells = complete_cells(ix, ("S", d), len(cfg.training.seeds))
    budgets = [b for b in cfg.budgets.hosts if len(cells.get(b, [])) == cfg.split.n_outer_resamples]
    rows, rows_w, mae = {}, {}, {}
    for b in budgets:
        st = {}
        for m in ("S", d):
            fr = {r: D.ensemble(ix[(ix.model == m) & (ix.B == b) & (ix.r == r)]) for r in cells[b]}
            st[m] = fr
        ref = st["S"]
        st[d] = {r: D._align(f, ref[r]) for r, f in st[d].items()}
        hs = {m: [D.stats_of(st[m][r]) for r in cells[b]] for m in st}
        rows[b] = a_block(hs["S"], hs[d], "mae", cfg)
        rows_w[b] = a_block(hs["S"], hs[d], "within_host_mae", cfg)
        mae[b] = {m: {k: aggregate_resamples(hs[m], n_boot=cfg.analysis.bootstrap.draws, seed=0)[k]
                      for k in ("mae", "within_host_mae")} for m in hs}
        const = [D.stats_of(D.host_constant(st["S"][r])) for r in cells[b]]
        for m in hs:                          # within-host skill: host-constant value minus the model's, paired
            sk = aggregate_delta(const, hs[m], n_boot=cfg.analysis.bootstrap.draws, seed=0)["within_host_mae"]
            mae[b][m]["within_host_skill"] = sk
    res = {"arm": a.arm, "delta_eV": delta, "budgets": budgets,
           "incomplete_budgets": [b for b in cfg.budgets.hosts if b not in budgets],
           "A_mae": {str(b): v for b, v in rows.items()}, "A_within_host": {str(b): v for b, v in rows_w.items()},
           "mae": {str(b): {m: {k: {"mean": v[k]["mean"], "ci": v[k]["ci"]} for k in v} for m, v in mm.items()}
                   for b, mm in mae.items()},
           "loco": loco_advantage(a.arm, d, cfg)}
    if budgets:
        res["N_star"] = n_star(rows, delta)
        res["N_star_within_host"] = n_star(rows_w, delta)
        if max(budgets) == cfg.analysis.c2_ceiling_budget:
            res["C2"] = {"budget": max(budgets), "A": rows[max(budgets)]["A"], "ci95": rows[max(budgets)]["ci95"]}
    print(f"arm {a.arm}: complete budgets {budgets} (incomplete: {res['incomplete_budgets']})")
    print(f"{'B':>4} {'MAE_S':>7} {'MAE_D':>7} {'A':>7} {'95% CI':>18} {'1-sided UB':>10} | {'A_wh':>7} {'UB_wh':>7}")
    for b in budgets:
        r, w = rows[b], rows_w[b]
        print(f"{b:>4} {mae[b]['S']['mae']['mean']:7.3f} {mae[b][d]['mae']['mean']:7.3f} {r['A']:+7.3f} "
              f"[{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}] {r['upper_one_sided_95']:+10.3f} | {w['A']:+7.3f} "
              f"{w['upper_one_sided_95']:+7.3f}")
    for k in ("N_star", "N_star_within_host", "C2", "loco"):
        print(k, res.get(k))
    if not a.no_write:
        from dftgnn.io.results import write_result

        print(write_result(f"c1c2_{a.arm}", res, config=cfg))


if __name__ == "__main__":
    main()
