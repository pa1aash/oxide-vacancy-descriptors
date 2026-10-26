"""python scripts/diagnostics_sweep.py            print the tables and write results/diagnostics_sweep.json
    python scripts/diagnostics_sweep.py --no-write print only

Reads the committed S and D run records and prediction parquets, the baseline predictions (B0, RF-Kumagai)
and, for the training-mean predictor and the descriptor checks, the universe parquet. Every model quantity
uses the seed-ensemble-mean prediction of a (model, resample, budget). Intervals are the hierarchical
bootstrap of ``dftgnn.stats.metrics`` (resamples, then test hosts), 2000 draws, two-sided 95%.

Sections (numbered as in the review):
  1  within-host collapse: hosts whose site predictions are constant, the within-host prediction spread, the
     correlation of predicted and true within-host deviations, and within-host skill (zero-skill value minus
     the model's within-host residual MAE) with a paired interval;
  2  trivial references at every budget: the training-mean predictor, B0, RF-Kumagai, S and D, with the
     paired differences of S and D against B0, RF-Kumagai and the mean predictor;
  6  epoch cap: 600 minus 200 epochs for S and D at B = 654 (resamples 0-2), paired, with the change in A;
  8 on the Kiyohara test hosts that were in no tuning training or validation set.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dftgnn.config import load_config
from dftgnn.split import budget_train, load_split, val_split
from dftgnn.stats.metrics import (
    _draw_weights,
    _interval,
    aggregate_delta,
    aggregate_resamples,
    cluster_bootstrap_ci,
    host_stats,
)
from dftgnn.train import val_seed

CONST_TOL_EV = 1e-3          # a host's predictions count as constant if max - min < 1 meV
N_BOOT, SEED, CI = 2000, 0, 0.95


# ---------------------------------------------------------------- loading
def run_index(subdir: str = "") -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(ROOT / "results" / subdir / "*.json"))):
        p = json.loads(Path(f).read_text()).get("payload")
        if not isinstance(p, dict) or "spec" not in p or p["spec"]["model"] not in ("S", "D-state", "D-late"):
            continue
        s = p["spec"]
        rows.append({"run_id": p["run_id"], "model": s["model"], "split": s["split"], "r": s["r"],
                     "B": s["budget"], "seed": s["seed"], "pooling": s["hp"]["pooling"],
                     "epochs_run": p["epochs_run"], "best_epoch": p["best_epoch"], "max_epochs": p["max_epochs"],
                     "pred": ROOT / p["predictions"]["path"]})
    return pd.DataFrame(rows)


def ensemble(runs: pd.DataFrame) -> pd.DataFrame:
    """Seed-ensemble-mean prediction per site; every seed must cover the same sites."""
    frames = [pd.read_parquet(p).set_index("site_id") for p in runs.pred]
    idx = frames[0].index
    if any(not f.index.sort_values().equals(idx.sort_values()) for f in frames):
        raise ValueError("seeds of one (model, r, B) cover different test sites")
    out = frames[0][["host_id", "y_true"]].copy()
    out["y_pred"] = np.mean([f.loc[idx, "y_pred"].to_numpy() for f in frames], axis=0)
    out["n_seeds"] = len(frames)
    return out.reset_index()


def stats_of(fr: pd.DataFrame) -> pd.DataFrame:
    return host_stats(fr.y_true, fr.y_pred, fr.host_id)


def host_constant(fr: pd.DataFrame) -> pd.DataFrame:
    """The same sites predicted with their host's true mean (any host-constant predictor has its within-host
    residual MAE, the zero-skill value)."""
    return fr.assign(y_pred=fr.groupby("host_id").y_true.transform("mean"))


def within_stats(fr: pd.DataFrame) -> dict:
    n = fr.groupby("host_id").y_true.transform("size")
    m = fr[n >= 2]
    g = m.groupby("host_id")
    rng = g.y_pred.max() - g.y_pred.min()
    dp = m.y_pred - g.y_pred.transform("mean")
    dt = m.y_true - g.y_true.transform("mean")
    return {"n_multi_hosts": int(rng.size), "n_const_hosts": int((rng < CONST_TOL_EV).sum()),
            "pred_spread_eV": float(dp.abs().mean()), "true_spread_eV": float(dt.abs().mean()),
            "corr_within": float(np.corrcoef(dp, dt)[0, 1]) if dp.std() > 0 else float("nan")}


def summary(agg: dict, metric: str) -> dict:
    a = agg[metric]
    out = {"mean": a["mean"], "ci": a["ci"], "per_resample": a["per_resample"]}
    return out


# ---------------------------------------------------------------- section 1
def section_within(ix: pd.DataFrame, budgets: list[int], models: list[str]) -> dict:
    out = {}
    for m in models:
        for b in budgets:
            sub = ix[(ix.model == m) & (ix.B == b)]
            ens = {r: ensemble(g) for r, g in sub.groupby("r")}
            ws = [within_stats(fr) for fr in ens.values()]
            per_seed = [within_stats(pd.read_parquet(p)) for p in sub.pred]
            skill = aggregate_delta([stats_of(host_constant(fr)) for fr in ens.values()],
                                    [stats_of(fr) for fr in ens.values()], n_boot=N_BOOT, seed=SEED)
            agg = aggregate_resamples([stats_of(fr) for fr in ens.values()], n_boot=N_BOOT, seed=SEED)
            out[f"{m}|{b}"] = {
                "model": m, "B": b, "pooling": sorted(sub.pooling.unique()),
                "n_multi_hosts_mean": float(np.mean([w["n_multi_hosts"] for w in ws])),
                "ensemble_const_hosts_mean": float(np.mean([w["n_const_hosts"] for w in ws])),
                "per_seed_const_hosts_mean": float(np.mean([w["n_const_hosts"] for w in per_seed])),
                "pred_spread_eV": float(np.mean([w["pred_spread_eV"] for w in ws])),
                "true_spread_eV": float(np.mean([w["true_spread_eV"] for w in ws])),
                "corr_within": float(np.nanmean([w["corr_within"] for w in ws])),
                "within_host_mae": summary(agg, "within_host_mae"),
                "within_skill_eV": summary(skill, "within_host_mae"),
                "per_resample_within_host_mae": dict(zip([int(r) for r in ens], agg["within_host_mae"]
                                                         ["per_resample"], strict=True)),
            }
    return out


# ---------------------------------------------------------------- section 2
def baseline_frames(name: str) -> dict[tuple[int, int], pd.DataFrame]:
    d = pd.read_parquet(ROOT / "results" / "predictions" / f"{name}.parquet")
    return {(int(r), int(b)): g[["site_id", "host_id", "y_true", "y_pred"]].reset_index(drop=True)
            for (r, b), g in d.groupby(["resample", "budget"]) if r >= 0}


def mean_predictor(uni: pd.DataFrame, r: int, b: int, test: pd.DataFrame, how: str) -> pd.DataFrame:
    """Every test site predicted with the mean (or median) target of the training-budget sites: the B hosts,
    validation hosts included, which is what the GNNs consume."""
    tr = uni[uni.host_id.isin(budget_train(load_split(f"outer_r{r}"), b))].target_Ef_eV
    c = float(tr.mean() if how == "mean" else tr.median())
    return test.assign(y_pred=c)


def _align(fr: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    fr = fr.set_index("site_id").loc[ref.site_id].reset_index()
    if not np.allclose(fr.y_true.to_numpy(), ref.y_true.to_numpy()):
        raise ValueError("targets differ between aligned prediction frames")
    return fr


def section_references(ix: pd.DataFrame, uni: pd.DataFrame, budgets: list[int], d: str) -> dict:
    b0, rf = baseline_frames("b0_physics_floor"), baseline_frames("rf_kumagai")
    out = {}
    for b in budgets:
        fr = {}
        for m in ("S", d):
            sub = ix[(ix.model == m) & (ix.B == b)]
            fr[m] = {int(r): ensemble(g) for r, g in sub.groupby("r")}
        rs = sorted(fr["S"])
        ref = {r: fr["S"][r] for r in rs}
        fr["B0"] = {r: _align(b0[(r, b)], ref[r]) for r in rs}
        fr["RF-Kumagai"] = {r: _align(rf[(r, b)], ref[r]) for r in rs}
        fr["mean"] = {r: mean_predictor(uni, r, b, ref[r], "mean") for r in rs}
        fr["median"] = {r: mean_predictor(uni, r, b, ref[r], "median") for r in rs}
        fr[d] = {r: _align(fr[d][r], ref[r]) for r in rs}
        st = {k: [stats_of(v[r]) for r in rs] for k, v in fr.items()}
        row = {k: summary(aggregate_resamples(v, n_boot=N_BOOT, seed=SEED), "mae") for k, v in st.items()}
        for a in ("S", d):
            for c in ("B0", "RF-Kumagai", "mean"):
                row[f"{a}-{c}"] = summary(aggregate_delta(st[a], st[c], n_boot=N_BOOT, seed=SEED), "mae")
        out[str(b)] = row
    return out


# ---------------------------------------------------------------- section 3
def section_d_vs_s(ix: pd.DataFrame, budgets: list[int], d: str) -> dict:
    out = {}
    for b in budgets:
        per = []
        for r in sorted(ix.r.unique()):
            s = ensemble(ix[(ix.model == "S") & (ix.B == b) & (ix.r == r)])
            dd = _align(ensemble(ix[(ix.model == d) & (ix.B == b) & (ix.r == r)]), s)
            per.append((int(r), float((s.y_pred - s.y_true).abs().mean() - (dd.y_pred - dd.y_true).abs().mean())))
        a = np.array([p[1] for p in per])
        out[str(b)] = {"A_mean": float(a.mean()), "A_min": float(a.min()), "A_max": float(a.max()),
                       "n_positive": int((a > 0).sum()), "per_resample": dict(per)}
    return out


def section_kiyohara(d: str) -> dict:
    s = ensemble(run_index("c0_official"))
    kd = run_index("")
    dd = _align(ensemble(kd[(kd.model == d) & (kd.split == "kiyohara")]), s)
    res = {}
    for name, fr in (("S", s), (d, dd)):
        res[name] = {k: v["point"] for k, v in cluster_bootstrap_ci(fr.y_true, fr.y_pred, fr.host_id,
                                                                    n_boot=N_BOOT, seed=SEED).items()}
    return res


def section_descriptors(uni: pd.DataFrame, d_runs: pd.DataFrame) -> dict:
    """Non-finite and zero values, within-host variation of the site descriptors, and the standardisation
    the training loop applies (fit on the training sites of each run, validation excluded)."""
    from dftgnn.graphs import descriptor_names

    host, site = descriptor_names(uni.attrs["descriptor_class"])
    multi = uni[uni.groupby("host_id").host_id.transform("size") >= 2]
    rows = {}
    dev = multi.target_Ef_eV - multi.groupby("host_id").target_Ef_eV.transform("mean")
    for c in host + site:
        x = uni[c].astype(float)
        varies = multi.groupby("host_id")[c].nunique() > 1
        xd = multi[c] - multi.groupby("host_id")[c].transform("mean")
        rows[c] = {"class": "host" if c in host else "site", "n_nonfinite": int((~np.isfinite(x)).sum()),
                   "n_zero": int((x == 0).sum()), "n_distinct": int(x.nunique()),
                   "frac_multi_hosts_varying": float(varies.mean()),
                   "within_host_corr_with_Ef": float(np.corrcoef(xd, dev)[0, 1]) if xd.std() > 0 else None}
    # training-only standardisation, recomputed for each D run at the largest budget, resample 0
    std = []
    for _, run in d_runs.iterrows():
        sp = load_split(run.split)
        tr, _ = val_split(budget_train(sp, run.B), seed=val_seed(run.r, run.B, run.seed))
        trn = uni[uni.host_id.isin(tr)][host + site].to_numpy(float)
        tst = uni[uni.host_id.isin(sp["test"])][host + site].to_numpy(float)
        sd = trn.std(0)
        z = (tst - trn.mean(0)) / np.where(sd < 1e-12, 1.0, sd)
        std.append({"run_id": run.run_id, "seed": int(run.seed), "n_train_sites": int(trn.shape[0]),
                    "test_z_max_abs": float(np.abs(z).max()), "test_z_all_finite": bool(np.isfinite(z).all()),
                    "n_constant_train_columns": int((sd < 1e-12).sum())})
    return {"n_host": len(host), "n_site": len(site), "per_descriptor": rows, "standardisation": std}


# ---------------------------------------------------------------- section 6
def hier_contrast(stats: list[list[pd.DataFrame]], coefs: list[float]) -> dict:
    """Hierarchical bootstrap of sum_k c_k MAE_k over resamples and test hosts, with every model's host rows
    drawn identically (all models share the host rows of a resample)."""
    rng = np.random.default_rng(SEED)
    R = len(stats[0])
    arr = [[np.asarray(s, float) for s in st] for st in stats]
    point = np.array([sum(c * a[r][:, 1].sum() / a[r][:, 0].sum() for c, a in zip(coefs, arr, strict=True))
                      for r in range(R)])
    pick = rng.integers(0, R, size=(N_BOOT, R))
    vals = np.zeros((N_BOOT, R))
    for k in range(R):
        for j in range(R):
            rows = np.nonzero(pick[:, k] == j)[0]
            if rows.size == 0:
                continue
            w = _draw_weights(rng, arr[0][j].shape[0], rows.size)
            vals[rows, k] = sum(c * (w @ a[j][:, 1]) / (w @ a[j][:, 0]) for c, a in zip(coefs, arr, strict=True))
    boot = vals.mean(1)
    return {"mean": float(point.mean()), "ci": _interval(boot, CI), "per_resample": [float(x) for x in point]}


def section_capsens(ix: pd.DataFrame, d: str) -> dict:
    cap = run_index("capsens")
    rs = sorted(cap.r.unique())
    fr = {}
    for m in ("S", d):
        for tag, src in (("200", ix), ("600", cap)):
            fr[(m, tag)] = [ensemble(src[(src.model == m) & (src.B == 654) & (src.r == r)]) for r in rs]
    ref = fr[("S", "200")]
    fr = {k: [_align(f, ref[i]) for i, f in enumerate(v)] for k, v in fr.items()}
    st = {k: [stats_of(f) for f in v] for k, v in fr.items()}
    out = {"resamples": [int(r) for r in rs]}
    for m in ("S", d):
        out[f"{m}_600_minus_200"] = hier_contrast([st[(m, "600")], st[(m, "200")]], [1.0, -1.0])
        out[f"{m}_mae_200"] = hier_contrast([st[(m, "200")]], [1.0])
        out[f"{m}_mae_600"] = hier_contrast([st[(m, "600")]], [1.0])
        c = cap[cap.model == m]
        out[f"{m}_600_runs_reaching_200"] = int((c.best_epoch >= 199).sum())
        out[f"{m}_600_best_epoch"] = sorted(int(x) + 1 for x in c.best_epoch)
        s200 = ix[(ix.model == m) & (ix.B == 654)]
        out[f"{m}_200_runs_ran_all_epochs_all_resamples"] = int((s200.epochs_run == 200).sum())
        out[f"{m}_200_runs_total"] = len(s200)
    out["A_600_minus_A_200"] = hier_contrast([st[("S", "600")], st[(d, "600")], st[("S", "200")], st[(d, "200")]],
                                             [1.0, -1.0, -1.0, 1.0])
    pairs = []                            # each 600-epoch run against its 200-epoch twin (same r, seed)
    for _, c in cap.iterrows():
        tw = ix[(ix.model == c.model) & (ix.B == 654) & (ix.r == c.r) & (ix.seed == c.seed)].iloc[0]
        p6 = pd.read_parquet(c.pred).set_index("site_id")
        p2 = pd.read_parquet(tw.pred).set_index("site_id").loc[p6.index]
        pairs.append({"model": c.model, "r": int(c.r), "seed": int(c.seed), "run_200": tw.run_id, "run_600": c.run_id,
                      "epochs_run_200": int(tw.epochs_run), "epochs_run_600": int(c.epochs_run),
                      "best_epoch_200": int(tw.best_epoch) + 1, "best_epoch_600": int(c.best_epoch) + 1,
                      "max_abs_pred_diff_eV": float((p6.y_pred - p2.y_pred).abs().max())})
    out["pairs"] = pairs
    out["n_pairs_identical"] = sum(p["max_abs_pred_diff_eV"] == 0.0 for p in pairs)
    out["n_pairs_best_after_200"] = sum(p["best_epoch_600"] > 200 for p in pairs)
    full = ix[(ix.B == 654) & (ix.epochs_run == 200) & ix.model.isin(["S", d])]
    out["runs_200_that_ran_all_epochs"] = [
        {"model": f.model, "r": int(f.r), "seed": int(f.seed), "best_epoch": int(f.best_epoch) + 1,
         "covered_by_capsens": int(f.r) in out["resamples"]} for _, f in full.sort_values(["model", "r", "seed"]).iterrows()]
    return out


# ---------------------------------------------------------------- section 8
def section_c0_overlap(uni: pd.DataFrame, d: str) -> dict:
    kt = set(load_split("kiyohara")["test"])
    tune_sets = {b: set(budget_train(load_split("outer_r0"), b)) for b in (50, 200, 654)}
    any_tune = set().union(*tune_sets.values())
    s = ensemble(run_index("c0_official"))
    kd = run_index("")
    dd = _align(ensemble(kd[(kd.model == d) & (kd.split == "kiyohara")]), s)
    rf = pd.read_parquet(ROOT / "results" / "predictions" / "rf_kumagai.parquet")
    rf = _align(rf[rf["resample"] == -1][["site_id", "host_id", "y_true", "y_pred"]].reset_index(drop=True), s)
    out = {"n_test_hosts": len(kt), "n_in_any_tuning_set": len(kt & any_tune),
           "per_anchor": {str(b): len(kt & v) for b, v in tune_sets.items()},
           "nested": all(tune_sets[a] <= tune_sets[b] for a, b in ((50, 200), (200, 654)))}
    for subset, hosts in (("all", kt), ("non_overlap", kt - any_tune), ("overlap", kt & any_tune)):
        res = {"n_hosts": len(hosts)}
        for name, fr in (("S", s), (d, dd), ("RF-Kumagai", rf)):
            f = fr[fr.host_id.isin(hosts)]
            ci = cluster_bootstrap_ci(f.y_true, f.y_pred, f.host_id, n_boot=N_BOOT, seed=SEED, metrics=("mae",))
            res[name] = {"mae": ci["mae"]["point"], "ci": ci["mae"]["ci"], "n_sites": len(f)}
        out[subset] = res
    return out


# ---------------------------------------------------------------- output
def fmt_ci(x: dict) -> str:
    return f"{x['mean']:.3f} [{x['ci'][0]:.3f}, {x['ci'][1]:.3f}]"


def print_tables(res: dict, d: str) -> None:
    print("\n## 1. Within-host behaviour (seed-ensemble predictions; means over resamples)\n")
    print("| Model | B | Pooling | Multi-site test hosts | Constant hosts (ensemble) | Constant hosts (single seed) | "
          "Pred. within-host spread (eV) | Corr. of within-host deviations | Within-host MAE | Within-host skill |")
    print("|---|---:|---|---:|---:|---:|---:|---:|---|---|")
    for v in res["within"].values():
        print(f"| {v['model']} | {v['B']} | {', '.join(v['pooling'])} | {v['n_multi_hosts_mean']:.1f} | "
              f"{v['ensemble_const_hosts_mean']:.1f} | {v['per_seed_const_hosts_mean']:.1f} | "
              f"{v['pred_spread_eV']:.4f} | {v['corr_within']:.2f} | {fmt_ci(v['within_host_mae'])} | "
              f"{fmt_ci(v['within_skill_eV'])} |")
    print("\n## 2. Site-level test MAE against trivial references (eV)\n")
    cols = ["mean", "median", "B0", "RF-Kumagai", "S", d]
    print("| B | " + " | ".join(cols) + " |")
    print("|---:|" + "---|" * len(cols))
    for b, row in res["references"].items():
        print(f"| {b} | " + " | ".join(fmt_ci(row[c]) for c in cols) + " |")
    print("\n| B | " + " | ".join(f"{a} - {c}" for a in ("S", d) for c in ("mean", "B0", "RF-Kumagai")) + " |")
    print("|---:|" + "---|" * 6)
    for b, row in res["references"].items():
        print(f"| {b} | " + " | ".join(fmt_ci(row[f"{a}-{c}"]) for a in ("S", d)
                                       for c in ("mean", "B0", "RF-Kumagai")) + " |")
    print("\n## 3. A = MAE_S - MAE_D per budget (point values only)\n")
    print("| B | mean A | min | max | resamples with A > 0 |")
    print("|---:|---:|---:|---:|---:|")
    for b, v in res["d_vs_s"].items():
        print(f"| {b} | {v['A_mean']:+.3f} | {v['A_min']:+.3f} | {v['A_max']:+.3f} | {v['n_positive']}/10 |")
    print("\nKiyohara split:", json.dumps(res["kiyohara"], indent=1))
    print("\nDescriptors:")
    for k, v in res["descriptors"]["per_descriptor"].items():
        print(f"  {k:40s} {v}")
    print("  standardisation:", res["descriptors"]["standardisation"])
    print("\n## 6. Epoch cap\n", json.dumps(res["capsens"], indent=1))
    print("\n## 8. C0 overlap\n", json.dumps(res["c0_overlap"], indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    from dftgnn.data.universe import read_universe

    uni = read_universe(cfg)
    d = cfg.models.injection_mode
    ix = run_index("")
    ix = ix[ix.split.str.startswith("outer")]
    budgets = list(cfg.budgets.hosts)
    res = {
        "within": section_within(ix, budgets, ["S", d]),
        "references": section_references(ix, uni, budgets, d),
        "d_vs_s": section_d_vs_s(ix, budgets, d),
        "kiyohara": section_kiyohara(d),
        "descriptors": section_descriptors(uni, ix[(ix.model == d) & (ix.B == 654) & (ix.r == 0)]),
        "capsens": section_capsens(ix, d),
        "c0_overlap": section_c0_overlap(uni, d),
        "settings": {"const_tol_eV": CONST_TOL_EV, "n_boot": N_BOOT, "seed": SEED, "ci": CI, "d_variant": d},
    }
    print_tables(res, d)
    if not a.no_write:
        from dftgnn.io.results import write_result

        print(write_result("diagnostics_sweep", res, config=cfg))


if __name__ == "__main__":
    main()
