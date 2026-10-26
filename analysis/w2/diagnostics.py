"""Diagnostics of the final network (v2) from existing predictions, with the existing estimators.

Nothing is trained. Every interval comes from an estimator already used for the records it sits beside:

    sweep budgets   ``dftgnn.stats.g1`` (seed-ensemble mean over seeds 0-2, paired hierarchical bootstrap
                    ``HierDraws``, 2000 replicates, seed 20261008) on ``staging_eval_long.parquet`` (S, D) and
                    ``eval_table.parquet`` (RF-Kumagai), the draws of (``analysis/c1/final_network.py``)
    precision       ``analysis/precision_analysis.analyse`` (30 x 5 and 10 x 3 at B = 400, 654)
    screening       ``figures/code/screening_data.cluster_ci`` (host-cluster bootstrap, ``stats.bootstrap_n``
                    draws, seed 0)

Diagnostics (all descriptive; none replaces a result of the paper):

    a  descriptor network D at 25 and 50 hosts: single-run MAE of every resample x seed run, the outlying runs
       (Tukey fences, 1.5 IQR, over the 30 runs of a budget), their cause (non-finite values, training divergence,
       extreme predictions on a few test sites, whole resamples), median and 10% trimmed mean of the run MAE, and A
       with all runs and with the outlying runs removed (same estimator; resamples left without a D run dropped)
    b  training-limit counts of S, D, P1 and P (P is P1 then D; it has no epochs of its own) per budget; at 654 the
       paired comparison (same resample and seed) of single-run MAE between capped and uncapped runs and the sign of
       the effect on A = MAE_S - MAE_D
    c  crossover with the forest: MAE_S - MAE_RF and MAE_D - MAE_RF per budget, with the paired G1 interval
    d  the one-sided bound at 200, 400 and 654 hosts (10 x 3) with the resample-to-resample spread of A
    e  screening: per-site table of the final network and the pooled MAE with its interval

Every recomputed number that also sits in an earlier record is compared with it (tolerance 1e-9); the script stops
on a mismatch. Writes through ``write_result`` into ``results/derived_w2/``.

    PYTHONPATH=$PWD/src python analysis/w2/diagnostics.py [--dry-run [--dump DIR]] [--data-root ~/Desktop/dft-gnn]
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import trim_mean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis import a1_common as c
from analysis import precision_analysis, staging_analysis

OUT_DIR = ROOT / "results" / "derived_w2"
BUDGETS = (25, 50, 100, 200, 400, 654)
SMALL = (25, 50)
BOUND_BUDGETS = (200, 400, 654)
REPRO_TOL = 1e-9
DELTA = c.DELTA_EV
TUKEY_K = 1.5
TRIM = 0.10
EXTREME_EV = 10.0          # a site error above this is far outside the target range (1.6 to 8.3 eV)
BB = "v2"
ARM = "final network: He initialisation, validation-tuned (tuned_v2.yaml); called v2 in the repository"
LABEL = "DERIVED (W2); diagnosis of the final network from existing predictions; descriptive, no test"


class Mismatch(AssertionError):
    pass


def module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check(repro: dict, key: str, pairs: list[tuple[str, float, float]], source: str) -> None:
    """Record max |mine - stored| over the pairs; raise on any difference of 1e-9 or more."""
    d = [abs(float(x) - float(y)) for _, x, y in pairs]
    worst = max(range(len(d)), key=d.__getitem__)
    repro[key] = {"reference": source, "n_values_compared": len(d), "max_abs_difference": float(max(d)),
                  "worst": pairs[worst][0], "reproduced": bool(max(d) < REPRO_TOL)}
    if not repro[key]["reproduced"]:
        raise Mismatch(f"{key}: {pairs[worst][0]} differs by {max(d):.3e} from {source}")


def a_entry(s: dict) -> dict:
    lo, hi = float(s["ci"][0]), float(s["ci"][1])
    up = float(s["upper_one_sided"])
    return {"A": float(s["mean"]), "ci95": [lo, hi], "upper_one_sided_95": up,
            "two_sided_within_pm_delta": bool(lo > -DELTA and hi < DELTA), "upper_below_delta": bool(up < DELTA),
            "ci95_excludes_zero": bool(lo > 0 or hi < 0)}


def ms(s: dict) -> dict:
    return {"mean": float(s["mean"]), "ci95": [float(s["ci"][0]), float(s["ci"][1])]}


def diff_entry(s: dict) -> dict:
    lo, hi = float(s["ci"][0]), float(s["ci"][1])
    return {"difference": float(s["mean"]), "ci95": [lo, hi], "upper_one_sided_95": float(s["upper_one_sided"]),
            "below_with_95_interval": bool(hi < 0), "point_below": bool(s["mean"] < 0),
            "per_resample": [float(x) for x in s["per_resample"]]}


# ---------------------------------------------------------------- shared inputs

def load_inputs(root: Path) -> dict:
    from dftgnn.stats.g1 import HierDraws

    srec = c.record("staging_eval_table", root)
    long, f_long = staging_analysis.load_table("staging_eval_long", srec)
    runs, f_runs = staging_analysis.load_table("staging_runs", srec)
    ens = staging_analysis.ensemble(long)
    draws = staging_analysis.make_draws(ens)
    g1rec = c.record("g1_eval_table", root)["payload"]["files"]["eval_table"]
    if c.sha256(root / g1rec["path"]) != g1rec["sha256"]:
        raise AssertionError("eval_table.parquet differs from results/g1_eval_table.json")
    table = pd.read_parquet(root / g1rec["path"])
    n_hosts = [table[(table.model == "S") & (table.B == 25) & (table.r == r)].host_id.nunique() for r in range(10)]
    g1draws = HierDraws.make(n_hosts, c.N_BOOT, c.BOOT_SEED)
    if not (np.array_equal(g1draws.pick, draws.pick) and np.array_equal(g1draws.w, draws.w)):
        raise AssertionError("staging draws differ from the G1 draws")
    pruns, f_pruns = c.load_table("precision_cv_runs", root)
    plong, f_plong = c.load_table("precision_eval_long", root)
    return {"long": long, "runs": runs, "ens": ens, "draws": draws, "table": table, "pruns": pruns,
            "plong": plong, "inputs": {"staging_eval_long": f_long, "staging_runs": f_runs,
                                       "eval_table": {"path": g1rec["path"], "sha256": g1rec["sha256"]},
                                       "precision_cv_runs": f_pruns, "precision_eval_long": f_plong}}


def stats(ens: pd.DataFrame, model: str, resamples=range(10)) -> dict:
    from dftgnn.stats.g1 import resample_stats

    sub = ens[(ens.backbone == BB) & (ens.model == model)]
    return {B: [resample_stats(sub[(sub.B == B) & (sub.r == r)]) for r in resamples] for B in BUDGETS}


def sweep_contrast(inp: dict, a: dict, b: dict) -> dict:
    from dftgnn.stats.g1 import contrast

    return contrast(a, b, inp["draws"], "mae", ci=c.CI, upper_quantile=c.UPPER_Q)


def rf_stats(inp: dict) -> dict:
    from dftgnn.stats.g1 import resample_stats

    sub = inp["table"][inp["table"].model == "RF-Kumagai"]
    return {B: [resample_stats(sub[(sub.B == B) & (sub.r == r)]) for r in range(10)] for B in BUDGETS}


# ---------------------------------------------------------------- (a) descriptor network at 25 and 50 hosts

def run_record(run_id: str, root: Path) -> dict:
    hits = list((root / "results").glob(f"*/{run_id}.json"))
    if len(hits) != 1:
        raise AssertionError(f"{run_id}: {len(hits)} run records")
    return json.loads(hits[0].read_text())["payload"]


def descriptor_z(spec: dict, uni: pd.DataFrame, desc_cols: list[str], cfg) -> pd.DataFrame:
    """Per test site of a run: largest |z| of its descriptors under the run's own training standardisation.

    The training and test hosts come from ``dftgnn.train.resolve_hosts`` on the run's stored spec; mean and SD are
    taken over the training sites as in ``dftgnn.models.Standardiser.fit`` (population SD; SD below 1e-12 set to 1).
    """
    from dftgnn.train import RunSpec, resolve_hosts

    hosts = resolve_hosts(RunSpec(**spec), cfg)
    tr = uni[uni.host_id.isin(hosts["train"])][desc_cols].to_numpy(float)
    mu, sd = tr.mean(0), tr.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    te = uni[uni.host_id.isin(hosts["test"])]
    z = np.abs((te[desc_cols].to_numpy(float) - mu) / sd)
    k = z.argmax(1)
    return pd.DataFrame({"site_id": te.site_id.to_numpy(), "max_abs_z": z.max(1),
                         "max_z_descriptor": [desc_cols[i].removeprefix("desc_") for i in k],
                         "n_train_hosts": len(hosts["train"])})


def descriptor_small(inp: dict, root: Path, uni: pd.DataFrame, cfg, repro: dict) -> dict:
    from dftgnn.stats.g1 import HierDraws, contrast, resample_stats, seed_ensemble

    long, runs = inp["long"], inp["runs"]
    p1 = sorted(json.loads((root / "results" / "probe_descriptive.json").read_text())["payload"]["s_v2"]
                ["per_descriptor_654"]["trained"])
    desc_cols = [f"desc_{k}" for k in p1]
    assert all(col in uni.columns for col in desc_cols) and len(desc_cols) == 22
    out = {"rule": {"outlying_run": f"single-run MAE above Q3 + {TUKEY_K} x IQR of the 30 runs of the budget "
                                    "(Tukey fence)",
                    "extreme_site": f"absolute error above {EXTREME_EV} eV (targets lie between 1.6 and 8.3 eV)",
                    "trimmed_mean": f"scipy.stats.trim_mean, proportion {TRIM} cut from each end",
                    "descriptor_z": "largest |z| over the 22 descriptors of D, standardised with the training "
                                    "sites of the run (dftgnn.train.resolve_hosts on the stored spec)"},
           "budgets": {}}
    pairs = []
    for B in SMALL:
        rr = runs[(runs.backbone == BB) & (runs.model == "D") & (runs.B == B)].sort_values(["r", "seed"])
        rows = []
        for _, run in rr.iterrows():
            p = long[long.run_id == run.run_id]
            err = (p.y_pred - p.y_true).abs()
            rec = run_record(run.run_id, root)
            hist = pd.DataFrame(rec["history"])
            z = descriptor_z(rec["spec"], uni, desc_cols, cfg)
            pz = p.merge(z, on="site_id", how="left", validate="one_to_one")
            assert pz.max_abs_z.notna().all()
            ext = pz[err.to_numpy() > EXTREME_EV]
            rows.append({"r": int(run.r), "seed": int(run.seed), "run_id": run.run_id, "mae": float(err.mean()),
                         "n_sites": len(p), "n_nonfinite_predictions": int((~np.isfinite(p.y_pred)).sum()),
                         "min_prediction_eV": float(p.y_pred.min()), "max_prediction_eV": float(p.y_pred.max()),
                         "max_abs_error_eV": float(err.max()), "n_extreme_sites": len(ext),
                         "share_of_abs_error_from_extreme_sites": float(err[err > EXTREME_EV].sum() / err.sum()),
                         "mae_without_extreme_sites": float(err[err <= EXTREME_EV].mean()),
                         "extreme_hosts": sorted(set(ext.host_id)),
                         "extreme_formulae": sorted({s.split("_Va_")[0] for s in ext.site_id}),
                         "median_max_abs_z_extreme_sites": float(ext.max_abs_z.median()) if len(ext) else None,
                         "median_max_abs_z_other_sites": float(pz.max_abs_z[err.to_numpy() <= EXTREME_EV].median()),
                         "max_z_descriptors_extreme_sites": dict(ext.max_z_descriptor.value_counts().astype(int)),
                         "epochs_run": int(run.epochs_run), "best_epoch": int(rec["best_epoch"]),
                         "best_val_mae_eV": float(rec["best_val_metric"]),
                         "history_all_finite": bool(np.isfinite(hist[["train_loss", "val_metric"]].to_numpy()).all()),
                         "max_train_loss": float(hist.train_loss.max()),
                         "final_train_loss": float(hist.train_loss.iloc[-1])})
            pairs.append((f"{B}.r{run.r}.s{run.seed}.mae", rows[-1]["mae"], run.mae))
            pairs.append((f"{B}.r{run.r}.s{run.seed}.record_mae", rows[-1]["mae"], rec["metrics"]["mae"]))
        df = pd.DataFrame(rows)
        q1, q3 = np.quantile(df.mae, [0.25, 0.75])
        fence = float(q3 + TUKEY_K * (q3 - q1))
        df["outlying"] = df.mae > fence
        outl = df[df.outlying]
        # A with all runs and with the outlying D runs removed
        a_all = contrast(stats(inp["ens"], "S"), stats(inp["ens"], "D"), inp["draws"], "mae", ci=c.CI,
                         upper_quantile=c.UPPER_Q)[B]
        keep = long[(long.backbone == BB) & (long.B == B) & (long.model.isin(["S", "D"]))]
        drop = set(outl.run_id)
        keep = keep[~keep.run_id.isin(drop)]
        ens = seed_ensemble(keep, keys=("model", "r", "B", "site_id"))
        rs = sorted(set(ens[ens.model == "D"].r) & set(ens[ens.model == "S"].r))
        st = {m: {B: [resample_stats(ens[(ens.model == m) & (ens.r == r)]) for r in rs]} for m in ("S", "D")}
        dr = HierDraws.make([len(s) for s in st["S"][B]], c.N_BOOT, c.BOOT_SEED)
        a_clean = contrast(st["S"], st["D"], dr, "mae", ci=c.CI, upper_quantile=c.UPPER_Q)[B]
        s_runs = runs[(runs.backbone == BB) & (runs.model == "S") & (runs.B == B)].mae
        out["budgets"][str(B)] = {
            "runs": df.to_dict(orient="records"),
            "tukey_fence_eV": fence, "q1_eV": float(q1), "q3_eV": float(q3),
            "outlying_runs": [f"r{r} seed {s}" for r, s in zip(outl.r, outl.seed)],
            "n_outlying": len(outl), "outlying_resamples": sorted({int(x) for x in outl.r}),
            "D_run_mae": {"median": float(df.mae.median()), "trimmed_mean_10pc": float(trim_mean(df.mae, TRIM)),
                          "mean": float(df.mae.mean()), "min": float(df.mae.min()), "max": float(df.mae.max())},
            "S_run_mae_reference": {"median": float(s_runs.median()), "trimmed_mean_10pc": float(trim_mean(s_runs, TRIM)),
                                    "mean": float(s_runs.mean())},
            "n_runs_with_nonfinite": int((df.n_nonfinite_predictions > 0).sum()),
            "n_runs_with_nonfinite_history": int((~df.history_all_finite).sum()),
            "n_runs_with_extreme_sites": int((df.n_extreme_sites > 0).sum()),
            "n_extreme_sites_total": int(df.n_extreme_sites.sum()),
            "share_of_abs_error_from_extreme_sites": float(
                (df.mae * df.n_sites * df.share_of_abs_error_from_extreme_sites).sum() / (df.mae * df.n_sites).sum()),
            "extreme_formulae_all_runs": sorted({f for x in df.extreme_formulae for f in x}),
            "A_all_runs": {**a_entry(a_all["diff"]), "MAE_S": ms(a_all["a"]), "MAE_D": ms(a_all["b"]),
                           "per_resample": a_all["diff"]["per_resample"]},
            "A_outlying_runs_removed": {**a_entry(a_clean["diff"]), "MAE_S": ms(a_clean["a"]),
                                        "MAE_D": ms(a_clean["b"]), "resamples_kept": rs,
                                        "n_D_runs_kept": int(len(df) - len(outl)),
                                        "note": "seed ensemble over the remaining D seeds of each resample; new "
                                                "HierDraws over the kept resamples (2000, seed 20261008)"}}
    check(repro, "a.single_run_mae_vs_staging_runs_and_run_records", pairs,
          "results/analysis/staging_runs.parquet mae; results/v2_sweep/<run_id>.json payload.metrics.mae")
    return out


def c1_rows(root: Path) -> dict:
    return json.loads((root / "results" / "derived_c1" / "final_network_summary.json").read_text())["payload"][
        "table"]["rows"]


def repro_a_vs_c1(out: dict, root: Path, repro: dict) -> None:
    rows = c1_rows(root)
    pairs = []
    for B in SMALL:
        v, r = out["budgets"][str(B)]["A_all_runs"], rows[f"B{B}"]
        pairs += [(f"{B}.A", v["A"], r["A"]["A"]), (f"{B}.A.lo", v["ci95"][0], r["A"]["ci95"][0]),
                  (f"{B}.A.hi", v["ci95"][1], r["A"]["ci95"][1]),
                  (f"{B}.A.upper", v["upper_one_sided_95"], r["A"]["upper_one_sided_95"]),
                  (f"{B}.MAE_D", v["MAE_D"]["mean"], r["MAE_D"]["mean"]),
                  (f"{B}.MAE_D.lo", v["MAE_D"]["ci95"][0], r["MAE_D"]["ci95"][0]),
                  (f"{B}.MAE_D.hi", v["MAE_D"]["ci95"][1], r["MAE_D"]["ci95"][1])]
    check(repro, "a.A_all_runs_vs_final_network_summary", pairs,
          "results/derived_c1/final_network_summary.json payload.table.rows.B25, B50")


# ---------------------------------------------------------------- (b) training-limit counts

def training_limit(inp: dict, repro: dict, root: Path) -> dict:
    runs = inp["runs"]
    r = runs[runs.backbone == BB].copy()
    r["capped"] = (r.epochs_run >= r.max_epochs) & (r.max_epochs > 0)
    cap = r.set_index("run_id").capped
    pm = r.model == "P"
    r.loc[pm, "capped_P1"] = r.loc[pm, "p1_run"].map(cap).astype(bool)
    r.loc[pm, "capped_D"] = r.loc[pm, "d_run"].map(cap).astype(bool)
    assert r.loc[pm, ["capped_P1", "capped_D"]].notna().all().all()
    counts = {}
    for B in sorted(r.B.unique()):
        g = r[r.B == B]
        row = {}
        for m in ("S", "D", "P1"):
            x = g[g.model == m]
            row[m] = {"n_runs": len(x), "n_capped": int(x.capped.sum())}
        x = g[g.model == "P"]
        row["P"] = {"n_runs": len(x), "n_capped_component": int((x.capped_P1 | x.capped_D).sum()),
                    "n_capped_P1_component": int(x.capped_P1.sum()), "n_capped_D_component": int(x.capped_D.sum())}
        counts[str(B)] = row
    max_epochs = sorted({int(x) for x in r.max_epochs[r.max_epochs > 0]})

    # paired comparison at 654: precision 30 x 5 (contains the sweep 10 x 3 cells) and the sweep alone
    pr = inp["pruns"]
    p = pr[(pr.design == "precision") & (pr.backbone == BB) & (pr.B == 654)]
    wide = p.pivot_table(index=["r", "seed"], columns="model", values=["mae", "capped"], aggfunc="first")
    cells = pd.DataFrame({"mae_S": wide["mae"]["S"], "mae_D": wide["mae"]["D"],
                          "capped_S": wide["capped"]["S"].astype(bool), "capped_D": wide["capped"]["D"].astype(bool)})
    cells = cells.reset_index()
    cells["A_cell"] = cells.mae_S - cells.mae_D

    def paired(cl: pd.DataFrame) -> dict:
        def grp(mask):
            g = cl[mask]
            return {"n_cells": len(g), "mean_mae_S": float(g.mae_S.mean()) if len(g) else None,
                    "mean_mae_D": float(g.mae_D.mean()) if len(g) else None,
                    "mean_A_cell": float(g.A_cell.mean()) if len(g) else None}
        groups = {"S_capped_D_uncapped": grp(cl.capped_S & ~cl.capped_D),
                  "S_uncapped_D_capped": grp(~cl.capped_S & cl.capped_D),
                  "both_capped": grp(cl.capped_S & cl.capped_D),
                  "neither_capped": grp(~cl.capped_S & ~cl.capped_D),
                  "all": grp(np.ones(len(cl), bool))}
        any_cap = cl.capped_S | cl.capped_D
        a_all, a_unc = float(cl.A_cell.mean()), float(cl[~any_cap].A_cell.mean())
        return {"groups": groups, "mean_A_cell_all": a_all, "mean_A_cell_without_capped_cells": a_unc,
                "effect_of_capped_cells_on_mean_A_cell": a_all - a_unc,
                "sign_of_effect": "lowers A" if a_all < a_unc else "raises A" if a_all > a_unc else "none",
                "capped_minus_uncapped_mae_S": (groups["S_capped_D_uncapped"]["mean_mae_S"]
                                                - groups["neither_capped"]["mean_mae_S"])
                if groups["S_capped_D_uncapped"]["n_cells"] else None}

    sweep_mask = cells.r.lt(10) & cells.seed.lt(3)
    out654 = {"precision_30x5": paired(cells), "sweep_10x3": paired(cells[sweep_mask]),
              "cells": cells.to_dict(orient="records")}
    # seed-ensemble A at 654 (30 x 5) with the capped runs removed: point only
    pl = inp["plong"]
    pl = pl[(pl.backbone == BB) & (pl.B == 654) & pl.model.isin(["S", "D"])]
    capped_ids = set(p.run_id[p.capped])
    out654["ensemble_A_point_30x5"] = {"all_runs": ens_a(pl), "capped_runs_removed": ens_a(pl[~pl.run_id.isin(capped_ids)]),
                                       "n_capped_runs_removed": len(capped_ids),
                                       "note": "mean over resamples of MAE_S - MAE_D of the seed ensemble over the "
                                               "remaining seeds; point only, no interval"}
    # reproduction: counts and capped-vs-uncapped means of the descriptive table, and the sweep counts
    rec = c.record("precision_a_b400_b654", root)["payload"]["descriptive"]["capped"]
    pairs = []
    for point in (400, 654):
        mine = c.capped_table(pr, "precision", point)
        for a_row, b_row in zip(sorted(mine, key=lambda x: (x["backbone"], x["model"])),
                                sorted(rec[f"precision|{point}"], key=lambda x: (x["backbone"], x["model"])),
                                strict=True):
            for k in ("n_runs", "n_capped", "n_uncapped", "mean_single_run_mae_capped", "mean_single_run_mae_uncapped"):
                if b_row[k] is not None:
                    pairs.append((f"{point}.{b_row['backbone']}.{b_row['model']}.{k}", a_row[k], b_row[k]))
            for s, v in b_row["n_capped_by_source"].items():
                pairs.append((f"{point}.{b_row['backbone']}.{b_row['model']}.by_source.{s}",
                              a_row["n_capped_by_source"][s], v))
        g = r[(r.B == point) & r.model.isin(["S", "D"])]
        for m, mm in (("S", "S"), ("D", "D")):
            b_row = next(x for x in rec[f"precision|{point}"] if x["backbone"] == BB and x["model"] == mm)
            pairs.append((f"{point}.sweep.{m}.n_capped", int(g[g.model == m].capped.sum()),
                          b_row["n_capped_by_source"]["sweep"]))
    check(repro, "b.capped_counts_vs_precision_record", pairs,
          "results/precision_a_b400_b654.json payload.descriptive.capped (counts, by source, mean single-run MAE)")
    return {"definition": "capped: epochs_run >= max_epochs (the run stopped at the epoch limit, not by early "
                          "stopping); P is P1 then D and has no epochs of its own, so a P run counts as capped when its "
                          "P1 or its D component is capped", "max_epochs": max_epochs, "counts_sweep_10x3": counts,
            "at_654": out654}


def ens_a(pl: pd.DataFrame) -> float:
    e = pl.groupby(["model", "r", "site_id"]).agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean")).reset_index()
    e["ae"] = (e.y_pred - e.y_true).abs()
    m = e.groupby(["r", "model"]).ae.mean().unstack()
    return float((m["S"] - m["D"]).mean())


# ---------------------------------------------------------------- (c) crossover with the forest

def crossover(inp: dict, repro: dict, root: Path) -> dict:
    sS, sD, sRF = stats(inp["ens"], "S"), stats(inp["ens"], "D"), rf_stats(inp)
    out = {}
    rows = c1_rows(root)
    pairs = []
    for m, st in (("S", sS), ("D", sD)):
        con = sweep_contrast(inp, st, sRF)
        out[m] = {str(B): {**diff_entry(con[B]["diff"]), f"MAE_{m}": ms(con[B]["a"]), "MAE_RF": ms(con[B]["b"])}
                  for B in BUDGETS}
        below = [B for B in BUDGETS if out[m][str(B)]["below_with_95_interval"]]
        stays = [B for i, B in enumerate(BUDGETS) if all(out[m][str(x)]["below_with_95_interval"] for x in BUDGETS[i:])]
        out[m]["smallest_budget_below_with_95_interval"] = below[0] if below else None
        out[m]["smallest_budget_below_at_it_and_all_larger"] = stays[0] if stays else None
        pt = [B for B in BUDGETS if out[m][str(B)]["point_below"]]
        out[m]["smallest_budget_point_below"] = pt[0] if pt else None
        for B in BUDGETS:
            r = rows[f"B{B}"]
            for k, ref in ((f"MAE_{m}", r[f"MAE_{m}"]), ("MAE_RF", r["MAE_RF"])):
                v = out[m][str(B)][k]
                pairs += [(f"{m}.{B}.{k}", v["mean"], ref["mean"]), (f"{m}.{B}.{k}.lo", v["ci95"][0], ref["ci95"][0]),
                          (f"{m}.{B}.{k}.hi", v["ci95"][1], ref["ci95"][1])]
    check(repro, "c.S_D_RF_marginals_vs_final_network_summary", pairs,
          "results/derived_c1/final_network_summary.json payload.table.rows.B* (MAE_S, MAE_D, MAE_RF)")
    out["reported_budgets"] = [100, 200, 400]
    out["rule"] = ("difference MAE_model - MAE_RF-Kumagai of the seed ensemble, paired G1 hierarchical bootstrap; "
                   "below with 95% interval = the upper end of the two-sided 95% interval is below 0")
    return out


# ---------------------------------------------------------------- (d) bound at 200, 400, 654

def bound_spread(inp: dict, repro: dict, root: Path) -> dict:
    from dftgnn.stats.g1 import HierDraws

    con = sweep_contrast(inp, stats(inp["ens"], "S"), stats(inp["ens"], "D"))
    runs = inp["runs"]
    sr = runs[(runs.backbone == BB) & runs.model.isin(["S", "D"])]
    single = sr.pivot_table(index=["B", "r", "seed"], columns="model", values="mae").reset_index()
    single["A"] = single.S - single.D
    out = {}
    rows = c1_rows(root)
    c2 = c.record("c1c2_v2", root)["payload"]["A_mae"]
    pairs = []
    for B in BOUND_BUDGETS:
        d = con[B]["diff"]
        per = np.array(d["per_resample"])
        sg = single[single.B == B]
        out[str(B)] = {"design": "10 x 3 (sweep)", **a_entry(d), "boot_sd": float(d["boot_sd"]),
                       "sd_over_resamples": float(d["sd_over_resamples"]), "per_resample_A": per.tolist(),
                       "min_resample_A": float(per.min()), "max_resample_A": float(per.max()),
                       "range_resample_A": float(per.max() - per.min()),
                       "n_resamples_A_above_delta": int((per > DELTA).sum()), "n_resamples_A_below_0": int((per < 0).sum()),
                       "per_resample_MAE_S": con[B]["a"]["per_resample"], "per_resample_MAE_D": con[B]["b"]["per_resample"],
                       "sd_over_resamples_MAE_S": float(con[B]["a"]["sd_over_resamples"]),
                       "sd_over_resamples_MAE_D": float(con[B]["b"]["sd_over_resamples"]),
                       "single_run_A": {"sd_over_30_runs": float(sg.A.std(ddof=1)),
                                        "mean_within_resample_sd_over_seeds": float(sg.groupby("r").A.std(ddof=1).mean())}}
        r = rows[f"B{B}"]["A"]
        pairs += [(f"{B}.A", out[str(B)]["A"], r["A"]), (f"{B}.lo", out[str(B)]["ci95"][0], r["ci95"][0]),
                  (f"{B}.hi", out[str(B)]["ci95"][1], r["ci95"][1]),
                  (f"{B}.upper", out[str(B)]["upper_one_sided_95"], r["upper_one_sided_95"])]
        pairs += [(f"{B}.per_resample[{i}]", x, y) for i, (x, y) in enumerate(zip(per, c2[str(B)]["per_resample"], strict=True))]
        pairs += [(f"{B}.sd_over_resamples", out[str(B)]["sd_over_resamples"], c2[str(B)]["sd_across_resamples"])]
    check(repro, "d.A_10x3_vs_final_network_summary_and_c1c2_v2", pairs,
          "results/derived_c1/final_network_summary.json payload.table.rows.B*.A; results/c1c2_v2.json "
          "payload.A_mae.*.per_resample, sd_across_resamples")
    # the 30 x 5 design at 400 and 654 for the same spread statistics
    prec = inp["plong"][inp["plong"].backbone == BB]
    dr = HierDraws.make([precision_analysis.N_HOSTS_PER_RESAMPLE] * 30, c.N_BOOT, seed=c.BOOT_SEED)
    res = precision_analysis.analyse(prec, 30, 5, dr)
    ref = c.record("precision_a_b400_b654", root)["payload"][BB]["30x5"]
    pairs = []
    for B in ("400", "654"):
        a = res[B]["A"]
        per = np.array(a["per_resample"])
        out[f"{B}_30x5"] = {"design": "30 x 5 (precision extension)", "A": float(a["mean"]),
                            "ci95": [float(a["ci"][0]), float(a["ci"][1])],
                            "upper_one_sided_95": float(a["upper_one_sided"]),
                            "sd_over_resamples": float(per.std(ddof=1)), "min_resample_A": float(per.min()),
                            "max_resample_A": float(per.max()), "n_resamples_A_above_delta": int((per > DELTA).sum()),
                            "n_resamples_A_below_0": int((per < 0).sum())}
        pairs += [(f"{B}.A", a["mean"], ref[B]["A"]["mean"]), (f"{B}.upper", a["upper_one_sided"], ref[B]["A"]["upper_one_sided"])]
        pairs += [(f"{B}.per_resample[{i}]", x, y) for i, (x, y) in enumerate(zip(per, ref[B]["A"]["per_resample"], strict=True))]
    check(repro, "d.A_30x5_vs_precision_record", pairs, "results/precision_a_b400_b654.json payload.v2.30x5.*.A")
    out["delta_eV"] = DELTA
    return out


# ---------------------------------------------------------------- (e) screening

def cation_elements(formula: str) -> list[str]:
    return sorted({e for e in re.findall(r"[A-Z][a-z]?", formula) if e != "O"})


def screening(inp: dict, repro: dict, root: Path, cfg) -> dict:
    demo = json.loads((root / "figures" / "main" / "data" / "screening_demo.json").read_text())["payload"]
    c5 = c.record("screening_c5", root)["payload"]
    sd = module(root / "figures" / "code" / "screening_data.py", "screening_data")
    B = int(c5["budget"])
    e = inp["ens"]
    e = e[(e.backbone == BB) & (e.model == "S") & (e.B == B)]
    rows, pairs = [], []
    for s in c5["sites"]:
        g = e[e.site_id == s["site_id"]]
        info, d = demo["site_index"][s["site_id"]], demo["sites"][s["site_id"]]
        pred = float(g.y_pred.mean())
        row = {"site_id": s["site_id"], "oxide": s["candidate"], "site": info["label"], "host_id": s["host_id"],
               "cation_elements": cation_elements(s["candidate"]), "first_shell_cations": int(d["first_shell_cations"]),
               "dft_eV": float(g.y_true.iloc[0]), "prediction_eV": pred, "error_eV": pred - float(g.y_true.iloc[0]),
               "abs_error_eV": abs(pred - float(g.y_true.iloc[0])), "n_resamples": int(g.r.nunique()),
               "min_over_resamples_eV": float(g.y_pred.min()), "max_over_resamples_eV": float(g.y_pred.max())}
        rows.append(row)
        pairs += [(f"{row['site_id']}.pred", pred, s["S_v2_eV"]), (f"{row['site_id']}.pred_demo", pred, d["v2"]["pred_eV"]),
                  (f"{row['site_id']}.dft", row["dft_eV"], s["dft_eV"]),
                  (f"{row['site_id']}.n", row["n_resamples"], d["v2"]["n_resamples"]),
                  (f"{row['site_id']}.abs_err", row["abs_error_eV"], d["v2"]["abs_err_eV"])]
    err = np.array([x["error_eV"] for x in rows])
    host = np.array([x["host_id"] for x in rows])
    lo, hi = sd.cluster_ci(err, host, cfg.stats.bootstrap_n)
    summ = {"mae_eV": float(np.abs(err).mean()), "mae_ci95": [lo, hi], "bias_eV": float(err.mean()),
            "max_abs_err_eV": float(np.abs(err).max()), "n_sites": len(rows), "n_hosts": len(set(host)),
            "n_over_prediction": int((err > 0).sum()), "n_under_prediction": int((err < 0).sum()),
            "worst_site": rows[int(np.abs(err).argmax())]["site_id"], "budget": B,
            "estimator": "figures/code/screening_data.cluster_ci: host-cluster percentile bootstrap, "
                         f"{cfg.stats.bootstrap_n} draws, seed 0; prediction = seed-ensemble mean averaged over the "
                         "resamples that hold the host out"}
    ref = demo["summary"][BB]
    pairs += [("mae", summ["mae_eV"], ref["mae_eV"]), ("mae_c5", summ["mae_eV"], c5["mae_S_v2"]),
              ("ci.lo", lo, ref["mae_ci"][0]), ("ci.hi", hi, ref["mae_ci"][1]), ("bias", summ["bias_eV"], ref["bias_eV"]),
              ("max_abs", summ["max_abs_err_eV"], ref["max_abs_err_eV"])]
    check(repro, "e.screening_vs_screening_demo_and_c5", pairs,
          "figures/main/data/screening_demo.json payload.sites.*.v2, summary.v2; results/screening_c5.json")
    return {"sites": rows, "summary": summ}


# ---------------------------------------------------------------- main

def universe(data_root: Path) -> pd.DataFrame:
    from dftgnn.config import load_config
    from dftgnn.data.universe import read_universe

    cfg = load_config()
    data = cfg.data.model_copy(update={"processed_dir": str(data_root / cfg.data.processed_dir)})
    return read_universe(cfg.model_copy(update={"data": data}))


def main() -> None:
    from dftgnn.config import load_config
    from dftgnn.io.results import write_result

    cfg = load_config()
    assert cfg.stats.delta_eV == DELTA
    data_root = Path(sys.argv[sys.argv.index("--data-root") + 1]) if "--data-root" in sys.argv else ROOT
    uni = universe(data_root.expanduser().resolve())
    inp = load_inputs(ROOT)
    repro: dict = {}
    a = descriptor_small(inp, ROOT, uni, cfg, repro)
    repro_a_vs_c1(a, ROOT, repro)
    b = training_limit(inp, repro, ROOT)
    cc = crossover(inp, repro, ROOT)
    d = bound_spread(inp, repro, ROOT)
    e = screening(inp, repro, ROOT, cfg)
    summary = {"n_comparisons": len(repro), "n_values_compared": sum(v["n_values_compared"] for v in repro.values()),
               "max_abs_difference": max(v["max_abs_difference"] for v in repro.values()),
               "all_reproduced_below_1e-9": all(v["reproduced"] for v in repro.values())}
    print(json.dumps({"reproduction": summary, **{k: v["max_abs_difference"] for k, v in repro.items()}}, indent=1))
    common = {"label": LABEL, "arm": ARM, "delta_eV": DELTA, "inputs": inp["inputs"],
              "reproduction": repro, "reproduction_summary": summary}
    records = {
        "w2_descriptor_small_budget": {**common, **a, "scope": "diagnosis, not a reanalysis for the paper"},
        "w2_training_limit": {**common, **b},
        "w2_forest_crossover": {**common, **cc},
        "w2_bound_spread": {**common, "budgets": d},
        "w2_screening_sites": {**common, **e},
    }
    if "--dry-run" in sys.argv:
        dump = Path(sys.argv[sys.argv.index("--dump") + 1]) if "--dump" in sys.argv else None
        for k, v in records.items():
            print(k, len(json.dumps(c.to_jsonable(v))))
            if dump:
                (dump / f"{k}.json").write_text(json.dumps(c.to_jsonable(v), indent=1))
        return
    for name, payload in records.items():
        print(write_result(name, c.to_jsonable(payload), config=cfg, repo_root=ROOT, results_dir=OUT_DIR))


if __name__ == "__main__":
    main()
