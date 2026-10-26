"""The §10 rule and the descriptive extras.

Rules:, row of 2026-10-14, fixed before any v1 probe number existed. Input: ``results/probe_fit_v1.json``
(``analysis/probe_fit.py``).

Rule, per class (host-electronic, site-electronic), on the class-mean test R^2 of the 60 (r, B) cells:

* TREND   Spearman rho between log B and the trained class mean over the 60 cells; 95% percentile interval from
          2000 bootstrap draws of the 10 resamples (all six budgets of each drawn resample); holds if rho > 0 and
          the interval excludes 0;
* CONTROLS at B = 200, 400, 654 and for each control (shuffled, init0): the resample mean of trained minus control
          and its bootstrap 95% interval; the probe exceeds the control if the lower bound is > 0.

Holds for a class if the trend holds and both controls are exceeded at all three budgets; holds if it holds
for both classes, otherwise moves to the SI. One (2000, 10) index matrix from ``default_rng(20261008)`` serves
every bootstrap, so all intervals are paired by resample.

Writes, via write_result, ``results/probe_c4_v1.json`` (rule and the class-mean curves) and
``results/probe_descriptive.json`` (per-descriptor R^2, within-host R^2, the 22-descriptor mean, alpha edges, and the
S-v2 quantities).

    PYTHONPATH=$PWD/src python analysis/probe_stats.py
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = (25, 50, 100, 200, 400, 654)
CONTROL_BUDGETS = (200, 400, 654)
CONTROLS = ("shuffled", "init0")
N_RESAMPLES = 10
N_BOOT = 2000
BOOT_SEED = 20261008
CI = 0.95
CLASS_KEYS = ("host", "site")


def boot_index(n_res: int = N_RESAMPLES, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> np.ndarray:
    """The single (n_boot, n_res) matrix of resample draws with replacement used for every interval."""
    return np.random.default_rng(seed).integers(0, n_res, size=(n_boot, n_res))


def interval(reps: np.ndarray) -> list[float]:
    lo, hi = (1 - CI) / 2, 1 - (1 - CI) / 2
    return [float(np.nanquantile(reps, lo)), float(np.nanquantile(reps, hi))]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(x, y).statistic)


def trend(values: np.ndarray, budgets: Sequence[int], idx: np.ndarray) -> dict:
    """``values`` (n_res, n_budgets). Spearman rho of log B against the value over all cells, and its interval from
    the resample bootstrap (each draw keeps all budgets of every drawn resample)."""
    v = np.asarray(values, float)
    lb = np.log(np.asarray(budgets, float))
    x = np.tile(lb, v.shape[0])
    rho = spearman(x, v.ravel())
    reps = np.array([spearman(np.tile(lb, len(row)), v[row].ravel()) for row in idx])
    ci = interval(reps)
    return {"rho": rho, "ci95": ci, "holds": bool(rho > 0 and ci[0] > 0), "n_cells": int(v.size)}


def paired_diff(a: np.ndarray, b: np.ndarray, idx: np.ndarray) -> dict:
    """Resample mean of a - b (per resample) and its bootstrap interval with the shared draws."""
    d = np.asarray(a, float) - np.asarray(b, float)
    reps = d[idx].mean(1)
    ci = interval(reps)
    return {"mean": float(d.mean()), "ci95": ci, "exceeds": bool(ci[0] > 0)}


def mean_ci(v: np.ndarray, idx: np.ndarray) -> dict:
    v = np.asarray(v, float)
    return {"mean": float(np.nanmean(v)), "ci95": interval(np.nanmean(v[idx], axis=1))}


def apply_rule(curves: Mapping[str, np.ndarray], idx: np.ndarray, budgets: Sequence[int] = BUDGETS,
               controls: Sequence[str] = CONTROLS, control_budgets: Sequence[int] = CONTROL_BUDGETS) -> dict:
    """``curves`` maps probe -> (n_res, n_budgets) class means. Returns trend, control comparisons and outcome."""
    t = trend(curves["trained"], budgets, idx)
    comp = {}
    for c in controls:
        for b in control_budgets:
            j = list(budgets).index(b)
            comp[f"{b}|{c}"] = paired_diff(curves["trained"][:, j], curves[c][:, j], idx)
    beats = all(v["exceeds"] for v in comp.values())
    return {"trend": t, "controls": comp, "exceeds_both_controls_all_budgets": beats,
            "holds": bool(t["holds"] and beats)}


def matrix(cells: Sequence[dict], probe: str, getter, budgets: Sequence[int] = BUDGETS,
           n_res: int = N_RESAMPLES) -> np.ndarray:
    """(n_res, n_budgets) array of getter(cell['probes'][probe]) over the cells."""
    m = np.full((n_res, len(budgets)), np.nan)
    for c in cells:
        m[int(c["r"]), list(budgets).index(int(c["B"]))] = getter(c["probes"][probe])
    if np.isnan(m).any():
        raise AssertionError(f"{probe}: missing cells")
    return m


def _f(x) -> float:
    return np.nan if x is None else float(x)


def curves_block(cells: Sequence[dict], probes: Sequence[str], idx: np.ndarray, key: str) -> dict:
    """Per probe and budget: resample mean of the class mean ``key`` with its interval, and the per-resample values."""
    out = {}
    for p in probes:
        m = matrix(cells, p, lambda q, k=key: _f(q["class_mean"][k]))
        out[p] = {str(b): {**mean_ci(m[:, j], idx), "per_resample": m[:, j].tolist()} for j, b in enumerate(BUDGETS)}
    return out


def class_mean_of(r2: Sequence, classes: Sequence[str], key: str) -> float:
    v = np.array([_f(x) for x in r2])
    m = (np.asarray(classes) == key) if key != "all" else np.ones(len(v), bool)
    m &= np.isfinite(v)
    return float(v[m].mean()) if m.any() else np.nan


# ---------------------------------------------------------------- the collaborator's S-v2 probe

def v2_block(rec: dict, names: Sequence[str], classes: Sequence[str], idx: np.ndarray) -> dict:
    pay = rec["payload"]
    if [d.removeprefix("desc_") for d in pay["descriptors"]] != list(names):
        raise AssertionError("S-v2 probe descriptors differ in order from the taxonomy order")
    probes = {"trained": "trained", "shuffled": "shuffled", "random": "init0"}
    cells = []
    max_diff = 0.0
    for run in pay["runs"]:
        pr = {}
        for his, ours in probes.items():
            r2 = run[his]["r2"]
            cm = {k: class_mean_of(r2, classes, k) for k in ("host", "site", "all")}
            for k, v in cm.items():
                s = run[his].get("class_mean", {}).get(k)
                if s is not None and np.isfinite(v):
                    max_diff = max(max_diff, abs(float(s) - v))
            pr[ours] = {"r2": r2, "class_mean": cm}
        cells.append({"r": run["r"], "B": run["B"], "probes": pr})
    if len(cells) != 60:
        raise AssertionError(f"S-v2 probe: {len(cells)} runs")
    out = {"label": "POST HOC (collaborator's probe on the S-v2 checkpoints; descriptive)",
           "source": "results/probe_v2_sweep.json", "source_git_sha": rec["git_sha"],
           "controls_map": {"random": "random-init encoder (his single control, mapped to init0)"},
           "max_abs_diff_recomputed_vs_stored_class_mean": max_diff,
           "within_host": "not computable: the S-v2 probe predictions are not stored"}
    for key in ("host", "site", "all"):
        curves = {p: matrix(cells, p, lambda q, k=key: q["class_mean"][k]) for p in ("trained", "shuffled", "init0")}
        out[key] = {"rule": apply_rule(curves, idx),
                    "curves": {p: {str(b): mean_ci(curves[p][:, j], idx) for j, b in enumerate(BUDGETS)}
                               for p in curves}}
    out["per_descriptor_654"] = {
        p: {n: mean_ci(matrix(cells, p, lambda q, i=i: _f(q["r2"][i]))[:, BUDGETS.index(654)], idx)
            for i, n in enumerate(names)}
        for p in ("trained", "shuffled", "init0")}
    out["C4_holds_under_A4_rule"] = bool(out["host"]["rule"]["holds"] and out["site"]["rule"]["holds"])
    return out


# ---------------------------------------------------------------- main

def main() -> None:
    from dftgnn.config import load_config
    from dftgnn.io.results import write_result

    cfg = load_config()
    fit_path = ROOT / "results" / "probe_fit_v1.json"
    fit = json.loads(fit_path.read_text())
    pay = fit["payload"]
    cells, names, classes = pay["cells"], pay["descriptors"], pay["classes"]
    if len(cells) != 60:
        raise AssertionError(f"{len(cells)} cells")
    idx = boot_index()
    probes = pay["probes"]

    rule = {}
    for key in CLASS_KEYS:
        curves = {p: matrix(cells, p, lambda q, k=key: _f(q["class_mean"][k])) for p in probes}
        rule[key] = apply_rule(curves, idx)
    holds = all(rule[k]["holds"] for k in CLASS_KEYS)
    c4 = {
        "label": "registered analysis (originally registered backbone v1); rule of docs/deviations.md 2026-10-14 (§10)",
        "source": {"path": "results/probe_fit_v1.json", "git_sha": fit["git_sha"]},
        "bootstrap": {"draws": N_BOOT, "seed": BOOT_SEED, "unit": "resample", "ci": CI},
        "rule": rule,
        "C4_holds": holds,
        "outcome": "C4 holds (main text)" if holds else "C4 moves to the SI",
        "curves": {k: curves_block(cells, probes, idx, k) for k in CLASS_KEYS},
        "budgets": list(BUDGETS),
        "probe_labels": pay["probe_labels"],
    }

    # descriptive extras
    per_desc = {}
    for p in probes:
        per_desc[p] = {}
        for i, n in enumerate(names):
            m = matrix(cells, p, lambda q, i=i: _f(q["r2"][i]))
            per_desc[p][n] = {str(b): mean_ci(m[:, j], idx) for j, b in enumerate(BUDGETS)}
    site_names = [n for n, c in zip(names, classes) if c == "site"]
    within, within_desc, within_diff = {}, {}, {}
    for p in probes:
        m = matrix(cells, p, lambda q: _f(q["within_class_mean_site"]))
        within[p] = {str(b): mean_ci(m[:, j], idx) for j, b in enumerate(BUDGETS)}
        within_desc[p] = {}
        for i, n in enumerate(site_names):
            md = matrix(cells, p, lambda q, i=i: _f(q["within_r2_site"][i]))
            within_desc[p][n] = {str(b): mean_ci(md[:, j], idx) for j, b in enumerate(BUDGETS)}
    mt = matrix(cells, "trained", lambda q: _f(q["within_class_mean_site"]))
    for c in ("shuffled", "init0"):
        mc = matrix(cells, c, lambda q: _f(q["within_class_mean_site"]))
        within_diff[c] = {str(b): paired_diff(mt[:, j], mc[:, j], idx) for j, b in enumerate(BUDGETS)}
    n_within = {str(b): {"hosts": [c["probes"]["trained"]["within_n_hosts"] for c in cells if c["B"] == b],
                         "sites": [c["probes"]["trained"]["within_n_sites"] for c in cells if c["B"] == b]}
                for b in BUDGETS}
    edges = {p: {"upper": int(sum(c["probes"][p]["alpha_at_upper_edge"] for c in cells)),
                 "lower": int(sum(c["probes"][p]["alpha_at_lower_edge"] for c in cells)),
                 "fits": len(cells) * len(names)} for p in probes}
    undefined = {p: {k: int(sum(c["probes"][p]["class_mean"].get(f"n_undefined_{k}", 0) for c in cells))
                     for k in CLASS_KEYS} for p in probes}
    v2_rec = json.loads((ROOT / "results" / "probe_v2_sweep.json").read_text())
    desc = {
        "label": "DESCRIPTIVE (not part of the C4 rule); v1 = originally registered backbone; s_v2 = POST HOC",
        "source": {"path": "results/probe_fit_v1.json", "git_sha": fit["git_sha"]},
        "bootstrap": {"draws": N_BOOT, "seed": BOOT_SEED, "unit": "resample", "ci": CI},
        "descriptors": names, "classes": classes,
        "per_descriptor_r2": per_desc,
        "all22_class_mean": curves_block(cells, probes, idx, "all"),
        "all22_rule_descriptive": apply_rule({p: matrix(cells, p, lambda q: _f(q["class_mean"]["all"]))
                                              for p in probes}, idx),
        "within_host_site": {"definition": "per cell over test hosts with >= 2 test sites, target and prediction "
                                           "mean-removed per host; class mean over the 10 site-electronic "
                                           "descriptors", "class_mean": within, "per_descriptor": within_desc,
                             "trained_minus_control": within_diff, "n_used": n_within},
        "alpha_at_grid_edge": edges,
        "undefined_r2": undefined,
        "s_v2": v2_block(v2_rec, names, classes, idx),
    }
    print(write_result("probe_c4_v1", c4, config=cfg, repo_root=ROOT))
    print(write_result("probe_descriptive", desc, config=cfg, repo_root=ROOT))


if __name__ == "__main__":
    main()
