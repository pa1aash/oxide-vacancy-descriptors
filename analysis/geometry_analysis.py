"""Geometry analysis C3b: ΔMAE under MLIP geometry, the null rule, the per-host regression on the three geometry
components, the non-converged sensitivity, and the descriptive pieces.

Physics. A model trained on DFT host geometry sees, under condition (ii), the same host relaxed by MACE-MP-0, and
under condition (iii) that relaxed host isotropically rescaled back to the DFT volume. ΔMAE(c) = MAE(c) - MAE(i)
measures how far the learned map depends on geometric detail the MLIP gets wrong; D keeps its DFT descriptors under
every condition (a fixed oracle), so its change isolates the graph input. The three components separate a volume
change (eps_v), a shape change at fixed volume (deviatoric strain) and internal relaxation of the atoms (internal
RMSD).

Rules:, row of 2026-10-14, items a-i.

Reads ``results/analysis/geometry_eval_long.parquet`` and ``geometry_hosts.parquet`` (sha256 checked against
``results/geometry_eval_table.json``), and for the descriptive pieces the stored predictions
(``results/analysis/eval_table.parquet``), the sweep run records, the collaborator's shards and record, and
the gitignored DFT and MLIP unit cells under ``--data-root`` (read only). Writes through ``write_result``:

    results/geometry_c3b.json           ΔMAE, null rule, regression, non-converged sensitivity, item i
    results/geometry_descriptive.json   component distributions, d-bins, GPU classes, 809 cross-check, sanity checks
    results/analysis/geometry_host_change.parquet   per (model, B, condition, host) mean |prediction change| (Fig 5)

    PYTHONPATH=$PWD/src python analysis/geometry_analysis.py --data-root ~/Desktop/dft-gnn
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODELS = ("S", "D", "P")
BUDGETS = (654, 200)
REGISTERED_B = 654
N_RESAMPLES = 10
N_BOOT = 2000
BOOT_SEED = 20261008
CHECK_SEED = 20261008
DELTA_EV = 0.05
RMSD_NULL_A = 0.01
MIN_STRATUM_HOSTS = 20
N_RECOMPUTE = 20
RECOMPUTE_TOL = 1e-6
CONDITIONS = ("dft", "mlip", "mlip_rescaled")
CHANGED = ("mlip", "mlip_rescaled")
COMPONENTS = ("eps_v", "deviatoric", "internal_rmsd_A")
DBINS = ("d0", "d10", "other", "unassigned")
GPU_ORDER = ("A10", "TITAN RTX", "RTX 4090")
NULL_STRING = "serial fragility not realised in this regime"
LABELS = {654: "registered (ANALYSIS_PLAN §9, v1 backbone, our 818 relaxations)",
          200: "DESCRIPTIVE EXTRA (not pre-registered, no test)"}
CROSS_LABEL = ("POST HOC descriptive cross-check: the collaborator's v2 evaluation on his 809 relaxations; never "
               "pre-registered")
TOST_LABEL = "POST HOC (ratified collaborator rule of 2026-10-09, ratified 2026-10-11)"
LATTICE_LEN_TOL = 0.10
LATTICE_ANGLE_TOL_DEG = 5.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- ΔMAE (items a, b)

def ensemble(long: pd.DataFrame) -> pd.DataFrame:
    """Seed-ensemble mean per (model, r, B, condition, site); every cell must hold the same seed count."""
    from dftgnn.stats.g1 import seed_ensemble

    ens = seed_ensemble(long, keys=("model", "r", "B", "condition", "site_id"))
    if ens.n_seeds.nunique() != 1:
        raise AssertionError("cells with different seed counts")
    return ens


def resample_stats(ens: pd.DataFrame, model: str, B: int, condition: str,
                   n_resamples: int = N_RESAMPLES) -> list[np.ndarray]:
    from dftgnn.stats.g1 import resample_stats as rs

    sub = ens[(ens.model == model) & (ens.B == B) & (ens.condition == condition)]
    return [rs(sub[sub.r == r]) for r in range(n_resamples)]


def make_draws(ens: pd.DataFrame, n_resamples: int = N_RESAMPLES, n_boot: int = N_BOOT, seed: int = BOOT_SEED):
    """One draw set over the evaluated test-host counts per resample (equal for every model, condition, budget)."""
    from dftgnn.stats.g1 import HierDraws

    counts = ens.groupby(["model", "B", "condition", "r"]).host_id.nunique().unstack("r")
    if (counts.nunique(axis=0) != 1).any():
        raise AssertionError("test-host counts differ between models, budgets or conditions")
    n_hosts = [int(counts[r].iloc[0]) for r in range(n_resamples)]
    return HierDraws.make(n_hosts, n_boot, seed), n_hosts


def bounds(reps: np.ndarray) -> dict:
    q = lambda p: float(np.quantile(reps, p))
    return {"ci95": [q(0.025), q(0.975)], "ci90": [q(0.05), q(0.95)], "boot_sd": float(reps.std(ddof=1))}


def marginal(stats: Sequence[np.ndarray], draws) -> tuple[dict, np.ndarray]:
    from dftgnn.stats.g1 import point_metric

    per = [point_metric(s, "mae") for s in stats]
    reps = draws.replicates(stats, "mae")
    return {"mean": float(np.mean(per)), **bounds(reps), "per_resample": per}, reps


def delta_block(ens: pd.DataFrame, draws, budgets: Sequence[int] = BUDGETS) -> tuple[dict, dict]:
    """Per budget and model: MAE of each condition and ΔMAE(c) = MAE(c) - MAE(i), paired on the same draws.
    Second value: the replicate and per-resample arrays of every ΔMAE (for item i)."""
    out, keep = {}, {}
    for B in budgets:
        out[str(B)] = {"label": LABELS.get(B, ""), "mae": {}, "delta_mae": {}}
        for m in MODELS:
            st = {c: resample_stats(ens, m, B, c, len(draws.n_hosts)) for c in CONDITIONS}
            mar = {c: marginal(st[c], draws) for c in CONDITIONS}
            out[str(B)]["mae"][m] = {c: mar[c][0] for c in CONDITIONS}
            out[str(B)]["delta_mae"][m] = {}
            for c in CHANGED:
                per = np.subtract(mar[c][0]["per_resample"], mar["dft"][0]["per_resample"])
                reps = mar[c][1] - mar["dft"][1]
                out[str(B)]["delta_mae"][m][c] = {"mean": float(per.mean()), **bounds(reps),
                                                  "includes_zero": bool(np.quantile(reps, 0.025) <= 0
                                                                        <= np.quantile(reps, 0.975)),
                                                  "per_resample": [float(x) for x in per]}
                keep[(B, m, c)] = (per, reps)
    return out, keep


def tost_outcome(ci95: Sequence[float], ci90: Sequence[float], delta: float = DELTA_EV) -> dict:
    """The four-outcome rule of 2026-10-09 (ratified 2026-10-11), in its order, for ΔMAE_P - ΔMAE_S (negative: P
    degrades less than S)."""
    inside = bool(ci90[0] > -delta and ci90[1] < delta)
    if ci95[1] < 0:
        base = "P better"
    elif ci95[0] > 0:
        base = "P worse"
    elif inside:
        base = "equivalent"
    else:
        base = "inconclusive"
    within = base in ("P better", "P worse") and inside
    return {"outcome": base, "within_equivalence_margin": within, "ci90_inside_margin": inside,
            "report": base + ("; different but within the equivalence margin" if within else "")}


def tost_block(keep: Mapping, budgets: Sequence[int] = BUDGETS, delta: float = DELTA_EV) -> dict:
    """ΔMAE_P - ΔMAE_S per budget and condition with the four-outcome TOST."""
    out = {"label": TOST_LABEL, "delta_eV": delta, "meaning": "negative: P degrades less than S", "by": {}}
    for B in budgets:
        for c in CHANGED:
            per = keep[(B, "P", c)][0] - keep[(B, "S", c)][0]
            reps = keep[(B, "P", c)][1] - keep[(B, "S", c)][1]
            b = bounds(reps)
            out["by"][f"{B}_{c}"] = {"B": B, "condition": c, "mean": float(per.mean()), **b,
                                     **tost_outcome(b["ci95"], b["ci90"], delta)}
    return out


# ---------------------------------------------------------------- per-host change and regression (item d)

def host_change(ens: pd.DataFrame, hosts: pd.DataFrame) -> pd.DataFrame:
    """Per (model, B, condition in (ii), (iii), host): mean over every (r, site) of |ens(c) - ens(i)|."""
    piv = ens.pivot_table(index=["model", "B", "r", "site_id", "host_id"], columns="condition",
                          values="y_pred").reset_index()
    if piv[list(CONDITIONS)].isna().any().any():
        raise AssertionError("a site lacks a condition")
    rows = []
    for c in CHANGED:
        d = piv.assign(change=(piv[c] - piv["dft"]).abs())
        g = d.groupby(["model", "B", "host_id"]).agg(mean_abs_change=("change", "mean"),
                                                     n_rows=("change", "size")).reset_index()
        rows.append(g.assign(condition=c))
    out = pd.concat(rows, ignore_index=True)
    h = hosts.set_index("host_id")
    for col in (*COMPONENTS, "d_bin", "converged"):
        out[col] = h[col].loc[out.host_id].to_numpy()
    return out.sort_values(["model", "B", "condition", "host_id"]).reset_index(drop=True)


def standardise(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu, sd = X.mean(0), X.std(0, ddof=1)
    if (sd <= 0).any():
        raise ValueError("a predictor is constant over the hosts of the fit")
    return (X - mu) / sd, mu, sd


def ols(y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Intercept and slopes of y on the columns of Z."""
    A = np.column_stack([np.ones(len(y)), Z])
    return np.linalg.lstsq(A, y, rcond=None)[0]


def r2(y: np.ndarray, Z: np.ndarray, coef: np.ndarray) -> float:
    res = y - np.column_stack([np.ones(len(y)), Z]) @ coef
    tot = ((y - y.mean()) ** 2).sum()
    return float(1 - (res ** 2).sum() / tot) if tot > 0 else float("nan")


def boot_index(n: int, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> np.ndarray:
    """Host bootstrap indices of one host set (shared by every model, budget and fit of that set)."""
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def fit_set(y: np.ndarray, X: np.ndarray, idx: np.ndarray, names: Sequence[str] = COMPONENTS) -> dict:
    """Joint and single-component OLS on standardised predictors with host-bootstrap intervals."""
    Z, mu, sd = standardise(X)
    out = {"n_hosts": len(y), "predictor_mean": dict(zip(names, map(float, mu), strict=True)),
           "predictor_sd": dict(zip(names, map(float, sd), strict=True))}
    specs = {"joint": list(range(len(names))), **{f"only_{n}": [i] for i, n in enumerate(names)}}
    reps_joint = None
    for key, cols in specs.items():
        coef = ols(y, Z[:, cols])
        reps = np.array([ols(y[i], Z[i][:, cols]) for i in idx])
        if key == "joint":
            reps_joint = reps
        out[key] = {"intercept": float(coef[0]), "r2": r2(y, Z[:, cols], coef),
                    "coef_per_sd": {names[j]: {"coef": float(coef[k + 1]),
                                               "ci95": [float(np.quantile(reps[:, k + 1], 0.025)),
                                                        float(np.quantile(reps[:, k + 1], 0.975))]}
                                    for k, j in enumerate(cols)}}
    return out, reps_joint, sd


def regression_block(change: pd.DataFrame, hosts_sets: Mapping[str, Sequence[str]], budgets: Sequence[int] = BUDGETS,
                     condition: str = "mlip", n_boot: int = N_BOOT) -> tuple[dict, dict]:
    """Per host set (pooled, d-bin strata) and (model, B): the item-d fits. Second value: joint-fit replicates."""
    out, keep = {}, {}
    for set_name, hs in hosts_sets.items():
        hs = sorted(hs)
        out[set_name] = {"n_hosts": len(hs)}
        if len(hs) < MIN_STRATUM_HOSTS:
            out[set_name]["fit"] = f"fewer than {MIN_STRATUM_HOSTS} hosts: counts only, no fit"
            continue
        idx = boot_index(len(hs), n_boot)
        for B in budgets:
            for m in MODELS:
                sub = change[(change.model == m) & (change.B == B) & (change.condition == condition)]
                sub = sub.set_index("host_id").loc[hs]
                X = sub[list(COMPONENTS)].to_numpy(float)
                res, reps, sd = fit_set(sub.mean_abs_change.to_numpy(float), X, idx)
                out[set_name][f"{m}_{B}"] = res
                iqr = np.subtract(*np.quantile(X, [0.75, 0.25], axis=0))
                keep[(set_name, m, B)] = (res, reps, sd, iqr)
    return out, keep


def slope_block(keep: Mapping, set_name: str = "pooled", budgets: Sequence[int] = BUDGETS,
                delta: float = DELTA_EV) -> dict:
    """Joint-fit slope of P minus that of S, per component, in eV per IQR, against +-delta."""
    out = {"label": TOST_LABEL, "delta_eV": delta, "units": "eV per IQR of the component over the fit's hosts",
           "by": {}}
    for B in budgets:
        if (set_name, "P", B) not in keep:
            continue
        rp, bp, sd, iqr = keep[(set_name, "P", B)]
        rs, bs, _, _ = keep[(set_name, "S", B)]
        for j, comp in enumerate(COMPONENTS):
            f = iqr[j] / sd[j]
            point = (rp["joint"]["coef_per_sd"][comp]["coef"] - rs["joint"]["coef_per_sd"][comp]["coef"]) * f
            reps = (bp[:, j + 1] - bs[:, j + 1]) * f
            b = bounds(reps)
            o = tost_outcome(b["ci95"], b["ci90"], delta)
            out["by"][f"{B}_{comp}"] = {"B": B, "component": comp, "iqr": float(iqr[j]), "mean": float(point), **b,
                                        **o}
    return out


# ---------------------------------------------------------------- distributions (items c, g)

def dist(x) -> dict:
    x = np.asarray(x, float)
    if x.size == 0:
        return {"n": 0}
    q = lambda p: float(np.quantile(x, p))
    return {"n": int(x.size), "median": q(0.5), "q25": q(0.25), "q75": q(0.75), "p95": q(0.95),
            "max": float(x.max()), "min": float(x.min()), "mean": float(x.mean())}


def component_dists(h: pd.DataFrame) -> dict:
    out = {c: dist(h[c]) for c in COMPONENTS}
    out["abs_eps_v"] = dist(h["eps_v"].abs())
    return out


def grouped_dists(h: pd.DataFrame, col: str, order: Sequence[str]) -> dict:
    return {g: component_dists(h[h[col] == g]) for g in order}


# ---------------------------------------------------------------- null rule (item e)

def null_rule(median_rmsd: float, delta: Mapping, B: int = REGISTERED_B, threshold: float = RMSD_NULL_A) -> dict:
    """Null rule."""
    cond1 = bool(median_rmsd < threshold)
    excl = [f"{m}_{c}" for m in MODELS for c in CHANGED if not delta[str(B)]["delta_mae"][m][c]["includes_zero"]]
    cond2 = not excl
    if cond1 and cond2:
        outcome = NULL_STRING
    else:
        held = [n for n, ok in (("(1) median internal RMSD below threshold", cond1),
                                ("(2) every |ΔMAE| interval includes 0", cond2)) if ok]
        outcome = "null rule not met; holds: " + ("; ".join(held) if held else "neither")
    return {"budget": B, "threshold_A": threshold, "median_internal_rmsd_A": float(median_rmsd),
            "median_minus_threshold_A": float(median_rmsd - threshold),
            "median_over_threshold": float(median_rmsd / threshold),
            "cond1_median_rmsd_below": cond1, "cond2_all_intervals_include_zero": cond2,
            "intervals_excluding_zero": excl, "n_intervals": len(MODELS) * len(CHANGED), "outcome": outcome}


# ---------------------------------------------------------------- 809 cross-check (item h)

def comparable(uc_dft, uc_other, len_tol: float = LATTICE_LEN_TOL, ang_tol: float = LATTICE_ANGLE_TOL_DEG) -> str:
    """'' if the two unit cells share species order and lattice basis within the tolerances, else the reason."""
    if [s.specie.symbol for s in uc_dft] != [s.specie.symbol for s in uc_other]:
        return "species order differs"
    a, b = np.array(uc_dft.lattice.abc), np.array(uc_other.lattice.abc)
    if (np.abs(b / a - 1) > len_tol).any():
        return "lattice length beyond tolerance"
    if (np.abs(np.array(uc_dft.lattice.angles) - np.array(uc_other.lattice.angles)) > ang_tol).any():
        return "lattice angle beyond tolerance"
    return ""


def agreement(ours: np.ndarray, his: np.ndarray) -> dict:
    from scipy.stats import spearmanr

    return {"spearman": float(spearmanr(ours, his).statistic),
            "median_abs_difference": float(np.median(np.abs(ours - his))),
            "median_ours": float(np.median(ours)), "median_his": float(np.median(his)),
            "max_abs_difference": float(np.max(np.abs(ours - his)))}


def cross_check(h: pd.DataFrame, his_cells: Mapping, dft_cells: Mapping, his_record: Mapping,
                our_delta: Mapping) -> dict:
    from dftgnn.mlip.geometry import host_metrics

    ours = h.set_index("host_id")
    common = sorted(set(his_cells) & set(ours.index))
    rows, reasons = [], {}
    for hid in common:
        why = comparable(dft_cells[hid], his_cells[hid]["structure"])
        if why:
            reasons[why] = reasons.get(why, 0) + 1
            continue
        m = host_metrics(dft_cells[hid], his_cells[hid]["structure"])
        rows.append({"host_id": hid, "his_converged": his_cells[hid]["converged"],
                     **{f"his_{c}": m[c] for c in COMPONENTS}, **{c: float(ours.loc[hid, c]) for c in COMPONENTS}})
    t = pd.DataFrame(rows)
    pay = his_record["payload"]
    his_max = max(((k, c, abs(v["mean"])) for k, d in pay["dMAE"].items() for c, v in d.items() if c != "mae"),
                  key=lambda x: x[2])
    our_max = {B: max(((m, c, abs(our_delta[B]["delta_mae"][m][c]["mean"])) for m in MODELS for c in CHANGED),
                      key=lambda x: x[2]) for B in our_delta}
    return {"label": CROSS_LABEL, "n_his_relaxed": len(his_cells), "n_common": len(common),
            "n_compared": len(t), "not_compared": reasons,
            "n_compared_both_converged": int((t.his_converged & ours.loc[t.host_id, "converged"].to_numpy()).sum()),
            "components": {c: agreement(t[c].to_numpy(float), t[f"his_{c}"].to_numpy(float)) for c in COMPONENTS},
            "his_recorded_geometry": pay["geometry"], "his_recorded_null_rule": pay["null_rule"],
            "his_max_abs_dmae": {"key": his_max[0], "condition": his_max[1], "value_eV": his_max[2]},
            "our_max_abs_delta_mae": {B: {"model": v[0], "condition": v[1], "value_eV": v[2]}
                                      for B, v in our_max.items()},
            "note": "his components are recomputed here with dftgnn.mlip.geometry.host_metrics from his relaxed unit "
                    "cells against our DFT unit cells; his evaluation uses the v2 backbone on his host set"}


# ---------------------------------------------------------------- sanity checks (TASK 3)

def sanity_sweep(long: pd.DataFrame, eval_table: pd.DataFrame, comp_ok: Mapping[str, bool]) -> dict:
    """Condition (i) predictions of S and D against the stored sweep predictions of the same checkpoints."""
    dft = long[(long.condition == "dft") & long.model.isin(["S", "D"])]
    et = eval_table[eval_table.model.isin(["S", "D"])].set_index(["model", "r", "B", "site_id"])
    out = {"same_checkpoint_runs": int(sum(comp_ok.values())), "runs_checked": len(comp_ok), "by": {}}
    diffs = []
    for (m, B, gpu), g in dft.groupby(["model", "B", "gpu"]):
        ref = np.array([et.loc[(m, r, b, s), f"y_pred_seed{sd}"]
                        for r, b, s, sd in zip(g.r, g.B, g.site_id, g.seed, strict=True)], float)
        d = np.abs(g.y_pred.to_numpy(float) - ref)
        diffs.append(d)
        out["by"][f"{m}_{B}_{gpu}"] = {"model": m, "B": int(B), "gpu": gpu, "n_runs": int(g.run_id.nunique()),
                                       "n_predictions": len(d), "max_abs_diff_eV": float(d.max()),
                                       "median_abs_diff_eV": float(np.median(d))}
    out["max_abs_diff_eV"] = float(np.concatenate(diffs).max())
    out["note"] = ("the sweep predictions were made on the original sweep GPU; condition (i) was re-run on NRP GPU "
                   "classes (A10, TITAN RTX, RTX 4090), so floating-point differences between GPU classes are "
                   "expected")
    return out


def sanity_recompute(h: pd.DataFrame, dft_cells: Mapping, mlip_dir: Path, n: int = N_RECOMPUTE,
                     seed: int = CHECK_SEED, tol: float = RECOMPUTE_TOL) -> dict:
    from pymatgen.core import Structure

    from dftgnn.mlip.geometry import host_metrics

    ids = sorted(h.host_id)
    pick = [ids[i] for i in np.random.default_rng(seed).choice(len(ids), n, replace=False)]
    rec = h.set_index("host_id")
    worst = {c: 0.0 for c in COMPONENTS}
    for hid in pick:
        uc_m = Structure.from_dict(json.loads((mlip_dir / f"{hid}_uc.json").read_text()))
        m = host_metrics(dft_cells[hid], uc_m)
        for c in COMPONENTS:
            worst[c] = max(worst[c], abs(m[c] - float(rec.loc[hid, c])))
    return {"hosts": pick, "max_abs_diff": worst, "tolerance": tol,
            "pass": bool(all(v <= tol for v in worst.values()))}


def check_sweep_components(recs: Sequence[Mapping], results: Path) -> dict[str, bool]:
    """Each S and D geomeval run's checkpoint is the sweep run of the same (model, r, B, seed), 200-epoch cap."""
    out = {}
    for p in recs:
        k = p["key"]
        if k["model"] not in ("S", "D"):
            continue
        rid = next(iter(k["components"].values()))
        f = results / f"{rid}.json"
        ok = False
        if f.is_file():
            q = json.loads(f.read_text())["payload"]
            sp = q["spec"]
            ok = (sp["model"] == {"S": "S", "D": "D-state"}[k["model"]] and sp["split"] == f"outer_r{k['r']}"
                  and int(sp["budget"]) == int(k["budget"]) and int(sp["seed"]) == int(k["seed"])
                  and sp.get("results_subdir") is None and int(q.get("max_epochs") or 0) == 200)
        out[p["run_id"]] = bool(ok)
    return out


# ---------------------------------------------------------------- main

def load_table(name: str, rec: Mapping) -> pd.DataFrame:
    meta = rec["payload"]["outputs"][name]
    path = ROOT / meta["path"]
    if sha256(path) != meta["sha256"]:
        raise AssertionError(f"{path}: sha256 differs from results/geometry_eval_table.json")
    return pd.read_parquet(path)


def analyse(ens: pd.DataFrame, hosts_all: pd.DataFrame, n_boot: int = N_BOOT) -> tuple[dict, pd.DataFrame, dict]:
    """ΔMAE, null rule and pooled plus stratified regression on one host set."""
    draws, n_hosts = make_draws(ens, n_boot=n_boot)
    delta, keep = delta_block(ens, draws)
    change = host_change(ens, hosts_all)
    evaluated = sorted(set(change.host_id))
    he = hosts_all.set_index("host_id").loc[evaluated].reset_index()
    sets = {"pooled": evaluated, **{b: list(he.host_id[he.d_bin == b]) for b in DBINS}}
    reg, rkeep = regression_block(change, sets, n_boot=n_boot)
    return {"n_test_hosts_per_resample": n_hosts, "delta": delta, "regression": reg,
            "tost": tost_block(keep), "slopes": slope_block(rkeep)}, change, keep


def main() -> None:
    from pymatgen.core import Structure

    from dftgnn.config import load_config
    from dftgnn.io.results import write_result

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, default=ROOT)
    a = ap.parse_args()
    dr = a.data_root.expanduser().resolve()
    mlip_dir = dr / "data" / "processed" / "mlip_v1"
    cells_path = dr / "data" / "processed" / "unit_cells_v1.json"
    cfg = load_config()
    if (cfg.mlip.null_rule.median_internal_rmsd_A_below != RMSD_NULL_A
            or cfg.analysis.equivalence_margin_eV != DELTA_EV or cfg.analysis.bootstrap.draws != N_BOOT):
        raise AssertionError("constants differ from configs/config.yaml")

    tab = json.loads((ROOT / "results" / "geometry_eval_table.json").read_text())
    long = load_table("long", tab)
    hosts = load_table("hosts", tab)
    ens = ensemble(long)

    main_res, change, _ = analyse(ens, hosts)
    rmsd_median = float(np.median(hosts.internal_rmsd_A))
    nr = null_rule(rmsd_median, main_res["delta"])

    nonconv = sorted(hosts.host_id[~hosts.converged])
    ens_c = ens[~ens.host_id.isin(nonconv)]
    sens_res, _, _ = analyse(ens_c, hosts)
    hc = hosts[hosts.converged]
    sens = {"excluded_hosts": nonconv, "n_excluded": len(nonconv),
            "n_excluded_in_test_sets": len(set(nonconv) & set(ens.host_id)),
            "n_test_hosts_per_resample": sens_res["n_test_hosts_per_resample"], "delta": sens_res["delta"],
            "regression_pooled": sens_res["regression"]["pooled"],
            "null_rule": null_rule(float(np.median(hc.internal_rmsd_A)), sens_res["delta"])}

    change_path = ROOT / "results" / "analysis" / "geometry_host_change.parquet"
    change.to_parquet(change_path, index=False)

    c3b = {
        "label": {"654": LABELS[654], "200": LABELS[200], "item_i": TOST_LABEL},
        "rules": "docs/deviations.md row of 2026-10-14 (§9, step A3); ANALYSIS_PLAN §9",
        "estimator": {"ensemble": "seed mean over seeds 0-2 per (model, r, B, condition, site)",
                      "bootstrap": "dftgnn.stats.g1.HierDraws, paired hierarchical over resamples and test hosts",
                      "draws": N_BOOT, "seed": BOOT_SEED, "interval": "two-sided 95%: 2.5th and 97.5th percentiles",
                      "regression": "OLS with intercept, predictors standardised over the fit's hosts (ddof 1), "
                                    "host bootstrap, one index matrix per host set",
                      "min_stratum_hosts": MIN_STRATUM_HOSTS},
        "input": {"eval_table": "results/geometry_eval_table.json",
                  "host_change": {"path": "results/analysis/geometry_host_change.parquet",
                                  "sha256": sha256(change_path), "rows": len(change)}},
        "n_test_hosts_per_resample": main_res["n_test_hosts_per_resample"],
        "n_hosts_evaluated": int(ens.host_id.nunique()), "n_sites_evaluated": int(ens.site_id.nunique()),
        "budgets": main_res["delta"],
        "null_rule": nr,
        "regression": {"response": "per-host mean over (r, site) of |ens(ii) - ens(i)| (eV)",
                       "coef_units": "eV per SD of the component over the fit's hosts",
                       "strata": main_res["regression"]},
        "nonconverged_sensitivity": sens,
        "post_hoc_tost": main_res["tost"],
        "post_hoc_slopes": main_res["slopes"],
    }
    print(write_result("geometry_c3b", c3b, config=cfg, repo_root=ROOT))

    # ---------------- descriptive record
    dft_raw = json.loads(cells_path.read_text())
    dft_cells = {hid: Structure.from_dict(d) for hid, d in dft_raw.items()}
    evaluated = sorted(set(ens.host_id))
    he = hosts[hosts.host_id.isin(evaluated)]

    shards = sorted((ROOT / "results" / "mlip").glob("relaxed_shard*.json.gz"))
    his_cells = {}
    for f in shards:
        with gzip.open(f) as fh:
            shard = json.loads(fh.read())
        for hid, d in shard["hosts"].items():
            if hid in his_cells:
                raise AssertionError(f"{hid} in two shards")
            his_cells[hid] = {"structure": Structure.from_dict(d["unit_cell"]), "converged": bool(d["converged"])}
    his_record = json.loads((ROOT / "results" / "mlip_eval_v2.json").read_text())

    eval_table = pd.read_parquet(ROOT / "results" / "analysis" / "eval_table.parquet")
    geo_recs = [json.loads(f.read_text())["payload"] for f in sorted((ROOT / "results" / "geomeval").glob("*.json"))]
    comp_ok = check_sweep_components(geo_recs, ROOT / "results")
    long_ok = long[long.run_id.isin([k for k, v in comp_ok.items() if v]) | long.model.eq("P")]

    desc = {
        "label": "descriptive (no test); geometry of our 818 MACE-MP-0 relaxations",
        "rules": "docs/deviations.md row of 2026-10-14 (§9, step A3), items c, g, h and the sanity checks",
        "inputs": {"unit_cells": {"path": "data/processed/unit_cells_v1.json", "sha256": sha256(cells_path)},
                   "his_shards": {f.name: sha256(f) for f in shards},
                   "eval_table": {"path": "results/analysis/eval_table.parquet",
                                  "sha256": sha256(ROOT / "results/analysis/eval_table.parquet")}},
        "components_all_hosts": component_dists(hosts),
        "components_evaluated_hosts": component_dists(he),
        "components_by_dbin": grouped_dists(hosts, "d_bin", DBINS),
        "dbin_counts": {"all": {b: int((hosts.d_bin == b).sum()) for b in DBINS},
                        "evaluated": {b: int((he.d_bin == b).sum()) for b in DBINS}},
        "components_by_relax_gpu": grouped_dists(hosts, "relax_gpu", GPU_ORDER),
        "relax_gpu_counts": {g: int((hosts.relax_gpu == g).sum()) for g in GPU_ORDER},
        "relax_gpu_note": "each host was relaxed once on one card; relaxations were not repeated on a second card, so "
                          "cross-card reproducibility of the relaxed geometry is not measured",
        "convergence": {"n_converged": int(hosts.converged.sum()), "n_not_converged": int((~hosts.converged).sum()),
                        "not_converged": sorted(hosts.host_id[~hosts.converged]),
                        "components_not_converged": component_dists(hosts[~hosts.converged])},
        "cross_check_809": cross_check(hosts, his_cells, dft_cells, his_record,
                                       {B: main_res["delta"][str(B)] for B in BUDGETS}),
        "sanity": {"sweep_condition_i": sanity_sweep(long_ok, eval_table, comp_ok),
                   "counts": tab["payload"]["counts"],
                   "recompute_components": sanity_recompute(hosts, dft_cells, mlip_dir)},
    }
    print(write_result("geometry_descriptive", desc, config=cfg, repo_root=ROOT))


if __name__ == "__main__":
    main()
