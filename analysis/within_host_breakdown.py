"""Within-host breakdown of S and D at B = 654 by chemistry and site.

Analysis rules:, row of 2026-10-11 "Within-host breakdown", logged before this script ran. Pure reading of
committed files plus the universe table (read only, sha256 recorded). Writes
`results/within_host_breakdown.json` through `dftgnn.io.results.write_result`.

Quantities (per arm, model and group)
    site MAE           mean |p - y| over the group's sites; p is the seed-ensemble prediction averaged over
                       the resamples in which the site's host was tested
    within-host MAE    per host with >= 2 tested sites, |(p - mean_h p) - (y - mean_h y)|, averaged over
                       the group's sites in such hosts; host means over all tested sites of the host
    zero-skill MAE     |y - mean_h y| over the same sites: the within-host MAE of a host-constant predictor
    signed residual    mean of (p - mean_h p) - (y - mean_h y) (site-rank groups only)
    Spearman           per host with >= 3 tested sites, rank correlation of p (or of the vacancy-O Bader
                       charge) with y; mean over hosts

Uncertainty: host-cluster bootstrap within each group, 2000 draws, percentile 95% intervals, generator
``default_rng(SeedSequence([20261012, k]))`` with k the group's position in the fixed order; one draw set
per group shared by both models and both arms, so S - D and v2 - v1 are paired on the same hosts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

BUDGET = 654
N_BOOT = 2000
BOOT_SEED = 20261012
CI = 0.95
MIN_HOSTS_INTERVAL = 10
MIN_TEST_HOSTS_CATION = 15
PHS_PRONE = ("Nb", "W", "Ti", "V", "Ta", "Mo")
DCLASSES = ("d0", "d10", "other", "unassigned")
TERCILES = ("lower", "middle", "upper")
RANKS = ("lowest", "middle", "highest")
MODELS = ("S", "D")
ARMS = ("v2", "v1")
V2_MODEL = {"S": "S", "D-state": "D"}
SUM_COLS = ("n", "sabs", "wn", "wsum", "zsum", "ssum")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- site-level predictions

def seed_ensemble(df: pd.DataFrame) -> pd.DataFrame:
    """Mean over seeds per (model, r, site_id); ``df`` has model, r, seed, site_id, host_id, y_true, y_pred."""
    g = df.groupby(["model", "r", "site_id"], sort=True)
    if (g.y_true.nunique() != 1).any() or (g.host_id.nunique() != 1).any():
        raise ValueError("a site carries different targets or hosts across seeds")
    return g.agg(host_id=("host_id", "first"), y_true=("y_true", "first"),
                 y_pred=("y_pred", "mean"), n_seeds=("seed", "nunique")).reset_index()


def average_over_resamples(ens: pd.DataFrame) -> pd.DataFrame:
    """Per (model, site): mean of the seed-ensemble predictions over the resamples that tested the site."""
    g = ens.groupby(["model", "site_id"], sort=True)
    if (g.y_true.nunique() != 1).any() or (g.host_id.nunique() != 1).any():
        raise ValueError("a site carries different targets or hosts across resamples")
    return g.agg(host_id=("host_id", "first"), y_true=("y_true", "first"), y_pred=("y_pred", "mean"),
                 n_resamples=("r", "nunique")).reset_index()


def site_terms(sites: pd.DataFrame) -> pd.DataFrame:
    """Add per-site error terms for one model's averaged predictions (host_id, y_true, y_pred)."""
    d = sites.copy()
    d["n_host"] = d.groupby("host_id").y_true.transform("size")
    dy = d.y_true - d.groupby("host_id").y_true.transform("mean")
    dp = d.y_pred - d.groupby("host_id").y_pred.transform("mean")
    multi = d.n_host >= 2
    d["ae"] = (d.y_pred - d.y_true).abs()
    d["res"] = (dp - dy).where(multi, np.nan)
    d["zs"] = dy.abs().where(multi, np.nan)
    return d


def host_sums(terms: pd.DataFrame, mask: pd.Series | np.ndarray, hosts: Sequence[str]) -> np.ndarray:
    """(len(hosts), 6) per-host sums over the masked sites, columns ``SUM_COLS``."""
    t = terms[np.asarray(mask, bool)]
    multi = t.res.notna()
    df = pd.DataFrame({
        "host_id": t.host_id, "n": 1.0, "sabs": t.ae, "wn": multi.astype(float),
        "wsum": t.res.abs().fillna(0.0), "zsum": t.zs.fillna(0.0), "ssum": t.res.fillna(0.0)})
    s = df.groupby("host_id")[list(SUM_COLS)].sum()
    return s.reindex(list(hosts)).fillna(0.0).to_numpy()


def metrics_from_sums(tot: np.ndarray) -> dict[str, np.ndarray]:
    """Metrics from (..., 6) column sums."""
    n, sabs, wn, wsum, zsum, ssum = (tot[..., i] for i in range(6))
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"mae": sabs / n, "within_host_mae": wsum / wn, "zero_skill_mae": zsum / wn,
                "signed_residual": ssum / wn}


# ---------------------------------------------------------------- Spearman per host

def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (average ranks); NaN if either variable is constant."""
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def host_spearman(df: pd.DataFrame, x: str, y: str = "y_true", min_sites: int = 3) -> pd.Series:
    """Spearman(x, y) per host with >= ``min_sites`` sites (NaN where undefined), indexed by host."""
    out = {}
    for h, g in df.groupby("host_id", sort=True):
        if len(g) >= min_sites:
            out[h] = spearman(g[x].to_numpy(float), g[y].to_numpy(float))
    return pd.Series(out, dtype=float)


# ---------------------------------------------------------------- groups

def tercile_labels(values: pd.Series) -> pd.Series:
    """'lower' x <= q(1/3), 'middle', 'upper' x > q(2/3); quantiles over the given (one per host) values."""
    q1, q2 = np.quantile(values.to_numpy(float), [1 / 3, 2 / 3])
    lab = np.where(values <= q1, "lower", np.where(values > q2, "upper", "middle"))
    return pd.Series(lab, index=values.index)


def cation_elements(formula: str) -> list[str]:
    from pymatgen.core import Composition
    return sorted(str(e) for e in Composition(formula).elements if str(e) != "O")


def bader_rank_labels(sites: pd.DataFrame, col: str = "bader_charge") -> pd.Series:
    """Per site: 'lowest' / 'highest' (host min / max of ``col``), 'middle', or NaN (not eligible)."""
    n = sites.groupby("host_id")[col].transform("size")
    lo = sites.groupby("host_id")[col].transform("min")
    hi = sites.groupby("host_id")[col].transform("max")
    lab = pd.Series(np.nan, index=sites.index, dtype=object)
    ok = (n >= 2) & (hi > lo)
    lab[ok & (sites[col] == lo)] = "lowest"
    lab[ok & (sites[col] == hi)] = "highest"
    lab[ok & (sites[col] > lo) & (sites[col] < hi)] = "middle"
    return lab


def build_groups(sites: pd.DataFrame, hosts: pd.DataFrame) -> list[dict]:
    """Fixed-order group list. ``sites``: site_id, host_id, bader_charge (one row per tested site).
    ``hosts``: index host_id; columns dclass, cations (list), formation_energy, band_gap."""
    host_of = sites.host_id
    groups = [{"variable": "pooled", "group": "all", "mask": np.ones(len(sites), bool)}]
    for c in DCLASSES:
        groups.append({"variable": "d_class", "group": c,
                       "mask": host_of.map(hosts.dclass).eq(c).to_numpy()})
    counts: dict[str, int] = {}
    for cs in hosts.cations:
        for e in cs:
            counts[e] = counts.get(e, 0) + 1
    for e in sorted(x for x, n in counts.items() if n >= MIN_TEST_HOSTS_CATION):
        has = hosts.cations.map(lambda cs, e=e: e in cs)
        groups.append({"variable": "cation", "group": e, "mask": host_of.map(has).to_numpy(bool)})
    prone = hosts.cations.map(lambda cs: any(e in cs for e in PHS_PRONE))
    for lab, val in (("phs_prone", True), ("not_phs_prone", False)):
        groups.append({"variable": "phs_prone", "group": lab,
                       "mask": host_of.map(prone).eq(val).to_numpy()})
    for var, col in (("stability_tercile", "formation_energy"), ("band_gap_tercile", "band_gap")):
        lab = tercile_labels(hosts[col])
        for t in TERCILES:
            groups.append({"variable": var, "group": t, "mask": host_of.map(lab).eq(t).to_numpy()})
    rank = bader_rank_labels(sites)
    for t in RANKS:
        groups.append({"variable": "bader_rank", "group": t, "mask": rank.eq(t).to_numpy()})
    for k, g in enumerate(groups):
        g["k"] = k
    return groups


# ---------------------------------------------------------------- bootstrap

def draw_weights(n_hosts: int, k: int, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> np.ndarray:
    """(n_boot, n_hosts) multinomial host multiplicities for group position k."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, k]))
    return rng.multinomial(n_hosts, np.full(n_hosts, 1.0 / n_hosts), size=n_boot).astype(float)


def interval(x: np.ndarray, ci: float = CI) -> list[float] | None:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    lo, hi = (1 - ci) / 2, 1 - (1 - ci) / 2
    return [float(np.quantile(x, lo)), float(np.quantile(x, hi))]


def summarise_group(sums: Mapping[tuple[str, str], np.ndarray], w: np.ndarray | None,
                    metrics: Sequence[str]) -> dict:
    """Point values, intervals and paired differences for one group.

    ``sums[(arm, model)]``: (n_hosts, 6) per-host sums over the group's hosts (same host order).
    ``w``: (n_boot, n_hosts) weights, or None for no intervals.
    """
    any_sums = next(iter(sums.values()))
    n_hosts_metric = {"mae": int((any_sums[:, 0] > 0).sum()), "within_host_mae": int((any_sums[:, 2] > 0).sum())}
    n_hosts_metric["zero_skill_mae"] = n_hosts_metric["signed_residual"] = n_hosts_metric["within_host_mae"]
    pts, reps = {}, {}
    for key, s in sums.items():
        pts[key] = metrics_from_sums(s.sum(axis=0))
        reps[key] = metrics_from_sums(w @ s) if w is not None else None
    out: dict = {"n_sites": int(any_sums[:, 0].sum()), "n_sites_within": int(any_sums[:, 2].sum()),
                 "n_hosts": len(any_sums), "n_hosts_by_metric": n_hosts_metric}

    def entry(point: float, rep: np.ndarray | None, m: str) -> dict:
        e = {"point": float(point) if np.isfinite(point) else None}
        if rep is not None and n_hosts_metric[m] >= MIN_HOSTS_INTERVAL:
            e["ci95"] = interval(rep)
        return e

    for arm, model in sums:
        out.setdefault(arm, {})[model] = {
            m: entry(pts[(arm, model)][m], None if reps[(arm, model)] is None else reps[(arm, model)][m], m)
            for m in metrics}
    diffs: dict = {}
    for m in metrics:
        if m == "zero_skill_mae":
            continue
        for arm in ARMS:
            if (arm, "S") in sums and (arm, "D") in sums:
                p = pts[(arm, "S")][m] - pts[(arm, "D")][m]
                r = None if w is None else reps[(arm, "S")][m] - reps[(arm, "D")][m]
                diffs.setdefault(f"S_minus_D_{arm}", {})[m] = entry(p, r, m)
        for model in MODELS:
            if ("v2", model) in sums and ("v1", model) in sums:
                p = pts[("v2", model)][m] - pts[("v1", model)][m]
                r = None if w is None else reps[("v2", model)][m] - reps[("v1", model)][m]
                diffs.setdefault(f"v2_minus_v1_{model}", {})[m] = entry(p, r, m)
    out["differences"] = diffs
    return out


def summarise_spearman(rho: Mapping[tuple[str, str], pd.Series], ref: pd.Series, w: np.ndarray | None,
                       hosts: Sequence[str]) -> dict:
    """Mean over hosts of per-host Spearman, with the host bootstrap ``w`` over ``hosts`` (paired)."""
    out: dict = {}

    def one(s: pd.Series) -> dict:
        v = s.reindex(list(hosts)).to_numpy(float)
        ok = np.isfinite(v)
        e = {"mean": float(v[ok].mean()) if ok.any() else None, "median": float(np.median(v[ok])) if ok.any() else None,
             "n_hosts": int(ok.sum()), "n_hosts_undefined": int((~ok).sum())}
        return e, v, ok

    rep_cache = {}
    for key, s in list(rho.items()) + [(("reference", "bader_charge"), ref)]:
        e, v, ok = one(s)
        if w is not None and ok.sum() >= MIN_HOSTS_INTERVAL:
            with np.errstate(divide="ignore", invalid="ignore"):
                r = (w[:, ok] @ v[ok]) / w[:, ok].sum(axis=1)
            e["ci95"] = interval(r)
            rep_cache[key] = (r, ok)
        out.setdefault(key[0], {})[key[1]] = e
    diffs = {}
    for arm in ARMS:
        a, b = rep_cache.get((arm, "S")), rep_cache.get((arm, "D"))
        if a is not None and b is not None:
            both = a[1] & b[1]
            va = rho[(arm, "S")].reindex(list(hosts)).to_numpy(float)
            vb = rho[(arm, "D")].reindex(list(hosts)).to_numpy(float)
            d = va[both] - vb[both]
            with np.errstate(divide="ignore", invalid="ignore"):
                r = (w[:, both] @ d) / w[:, both].sum(axis=1)
            diffs[f"S_minus_D_{arm}"] = {"mean": float(d.mean()), "n_hosts": int(both.sum()), "ci95": interval(r)}
    for model in MODELS:
        a, b = rep_cache.get(("v2", model)), rep_cache.get(("v1", model))
        if a is not None and b is not None:
            both = a[1] & b[1]
            va = rho[("v2", model)].reindex(list(hosts)).to_numpy(float)
            vb = rho[("v1", model)].reindex(list(hosts)).to_numpy(float)
            d = va[both] - vb[both]
            with np.errstate(divide="ignore", invalid="ignore"):
                r = (w[:, both] @ d) / w[:, both].sum(axis=1)
            diffs[f"v2_minus_v1_{model}"] = {"mean": float(d.mean()), "n_hosts": int(both.sum()), "ci95": interval(r)}
    out["differences"] = diffs
    return out


# ---------------------------------------------------------------- inputs

def load_v1(root: Path) -> tuple[pd.DataFrame, dict]:
    path = root / "results/analysis/eval_table.parquet"
    e = pd.read_parquet(path)
    e = e[(e.B == BUDGET) & e.model.isin(MODELS)]
    seeds = ["y_pred_seed0", "y_pred_seed1", "y_pred_seed2"]
    if not np.allclose(e[seeds].mean(axis=1), e.y_pred, rtol=0, atol=1e-12):
        raise ValueError("eval_table y_pred is not the mean of the seed columns")
    long = e.drop(columns="y_pred").melt(id_vars=["model", "r", "site_id", "host_id", "y_true"], value_vars=seeds,
                  var_name="seed", value_name="y_pred")
    long["seed"] = long.seed.str[-1].astype(int)
    return long, {"path": str(path.relative_to(root)), "sha256": sha256(path), "rows_used": len(e)}


def load_v2(root: Path) -> tuple[pd.DataFrame, dict]:
    frames, files = [], []
    for rec in sorted((root / "results/v2_sweep").glob("*.json")):
        p = json.loads(rec.read_text())["payload"]
        spec = p["spec"]
        if spec["budget"] != BUDGET or spec["model"] not in V2_MODEL:
            continue
        pred = root / p["predictions"]["path"]
        h = sha256(pred)
        if h != p["predictions"]["sha256"]:
            raise ValueError(f"sha256 mismatch for {pred}")
        df = pd.read_parquet(pred)
        if len(df) != p["predictions"]["rows"]:
            raise ValueError(f"row count mismatch for {pred}")
        df = df.assign(model=V2_MODEL[spec["model"]], r=int(spec["r"]), seed=int(spec["seed"]))
        frames.append(df)
        files.append({"record": str(rec.relative_to(root)), "predictions": p["predictions"]["path"],
                      "sha256": h, "model": spec["model"], "r": int(spec["r"]), "seed": int(spec["seed"])})
    cells = {(f["model"], f["r"], f["seed"]) for f in files}
    if len(files) != 60 or len(cells) != 60:
        raise ValueError(f"expected 60 v2 B=654 S/D-state runs, found {len(files)}")
    return pd.concat(frames, ignore_index=True), {"n_files": len(files), "files": files}


def g1_replication(ens: pd.DataFrame) -> dict:
    """Resample-mean site MAE and within-host MAE (the G1 point estimator) from the seed ensemble."""
    from dftgnn.stats.g1 import point_metric, resample_stats
    out = {}
    for model in MODELS:
        per = {"mae": [], "within_host_mae": []}
        for r in sorted(ens.r.unique()):
            f = ens[(ens.model == model) & (ens.r == r)].sort_values("host_id")
            s = resample_stats(f)
            for m, vals in per.items():
                vals.append(point_metric(s, m))
        out[model] = {m: float(np.mean(v)) for m, v in per.items()}
    return out


# ---------------------------------------------------------------- main

def run(root: Path, data_root: Path) -> dict:
    v1_long, v1_meta = load_v1(root)
    v2_long, v2_meta = load_v2(root)
    ens = {"v1": seed_ensemble(v1_long), "v2": seed_ensemble(v2_long)}
    for arm, e in ens.items():
        if (e.n_seeds != 3).any():
            raise ValueError(f"{arm}: a (model, r, site) lacks one of the three seeds")
    keys = {arm: set(map(tuple, e[["model", "r", "site_id"]].to_numpy())) for arm, e in ens.items()}
    if keys["v1"] != keys["v2"]:
        raise ValueError("v1 and v2 test (model, r, site) sets differ")
    avg = {arm: average_over_resamples(e) for arm, e in ens.items()}

    upath = data_root / "data/processed/universe_v1.parquet"
    U = pd.read_parquet(upath)
    U = U.set_index("site_id")
    sites = avg["v1"][avg["v1"].model == "S"][["site_id", "host_id", "y_true", "n_resamples"]].reset_index(drop=True)
    if not np.allclose(sites.y_true, U.loc[sites.site_id, "target_Ef_eV"].to_numpy(), rtol=0, atol=1e-9):
        raise ValueError("targets differ from the universe table")
    if not (U.loc[sites.site_id, "host_id"].to_numpy() == sites.host_id.to_numpy()).all():
        raise ValueError("host ids differ from the universe table")
    sites["bader_charge"] = U.loc[sites.site_id, "desc_bader_charge"].to_numpy()

    tested = sorted(sites.host_id.unique())
    hu = U.reset_index().drop_duplicates("host_id").set_index("host_id")
    dbins = pd.read_csv(root / "data/host_dbins_v1.csv").set_index("host_id")
    hosts = pd.DataFrame(index=pd.Index(tested, name="host_id"))
    hosts["formula"] = dbins.loc[tested, "formula"]
    hosts["dclass"] = dbins.loc[tested, "bin"]
    hosts["cations"] = hosts.formula.map(cation_elements)
    hosts["formation_energy"] = hu.loc[tested, "desc_formation_energy"]
    hosts["band_gap"] = hu.loc[tested, "desc_band_gap"]
    if hosts[["formation_energy", "band_gap"]].isna().any().any():
        raise ValueError("missing host descriptor")
    n_all_hosts = int(U.host_id.nunique())

    terms = {}
    for arm in ARMS:
        for model in MODELS:
            a = avg[arm][avg[arm].model == model].set_index("site_id").loc[sites.site_id].reset_index()
            terms[(arm, model)] = site_terms(a)

    groups = build_groups(sites, hosts)
    metrics_site = ("mae", "within_host_mae", "zero_skill_mae")
    results = []
    for g in groups:
        mask = g["mask"]
        ghosts = sorted(sites.host_id[mask].unique())
        sums = {key: host_sums(t, mask, ghosts) for key, t in terms.items()}
        small = len(ghosts) < MIN_HOSTS_INTERVAL
        w = None if small else draw_weights(len(ghosts), g["k"])
        mets = metrics_site + (("signed_residual",) if g["variable"] == "bader_rank" else ())
        res = {"variable": g["variable"], "group": g["group"], "k": g["k"],
               "flag": "descriptive, small n" if small else None}
        res.update(summarise_group(sums, w, mets))
        results.append(res)

    # Spearman per host (>= 3 tested sites), pooled hosts, its own draw position after the groups
    rho = {}
    for key, t in terms.items():
        rho[key] = host_spearman(t, "y_pred")
    ref = host_spearman(sites, "bader_charge")
    sp_hosts = sorted(rho[("v1", "S")].index)
    k_sp = len(groups)
    w_sp = draw_weights(len(sp_hosts), k_sp)
    spear = summarise_spearman(rho, ref, w_sp, sp_hosts)
    spear.update({"k": k_sp, "hosts_with_ge3_sites": len(sp_hosts)})

    # sanity checks
    g1p = json.loads((root / "results/g1_primary.json").read_text())["payload"]["per_budget"][str(BUDGET)]
    g1w = json.loads((root / "results/g1_within_host.json").read_text())["payload"]["within_host_residual_mae"][
        "per_budget"][str(BUDGET)]
    c2 = json.loads((root / "results/c1c2_v2.json").read_text())["payload"]["mae"][str(BUDGET)]
    rep = {arm: g1_replication(ens[arm]) for arm in ARMS}
    pooled = results[0]
    sanity = {
        "i_site_mae_v1": {
            "pipeline_resample_mean": {m: rep["v1"][m]["mae"] for m in MODELS},
            "g1_primary": {"S": g1p["MAE_S"]["mean"], "D": g1p["MAE_D"]["mean"]},
            "pipeline_site_averaged": {m: pooled["v1"][m]["mae"]["point"] for m in MODELS},
        },
        "ii_within_host_mae_v1": {
            "pipeline_resample_mean": {m: rep["v1"][m]["within_host_mae"] for m in MODELS},
            "g1_within_host": {"S": g1w["S"]["mean"], "D": g1w["D"]["mean"]},
            "pipeline_site_averaged": {m: pooled["v1"][m]["within_host_mae"]["point"] for m in MODELS},
        },
        "v2_reference": {
            "pipeline_resample_mean": rep["v2"],
            "c1c2_v2": {"S": {"mae": c2["S"]["mae"]["mean"], "within_host_mae": c2["S"]["within_host_mae"]["mean"]},
                        "D": {"mae": c2["D-state"]["mae"]["mean"],
                              "within_host_mae": c2["D-state"]["within_host_mae"]["mean"]}},
        },
    }
    for k in ("i_site_mae_v1", "ii_within_host_mae_v1"):
        s = sanity[k]
        ref_key = "g1_primary" if k.startswith("i_") else "g1_within_host"
        s["abs_diff_resample_mean_vs_record"] = {m: abs(s["pipeline_resample_mean"][m] - s[ref_key][m]) for m in MODELS}
        s["match_1e-3"] = all(v <= 1e-3 for v in s["abs_diff_resample_mean_vs_record"].values())
        s["site_averaged_minus_record"] = {m: s["pipeline_site_averaged"][m] - s[ref_key][m] for m in MODELS}
    v2r = sanity["v2_reference"]
    v2r["abs_diff"] = {m: {q: abs(v2r["pipeline_resample_mean"][m][q] - v2r["c1c2_v2"][m][q])
                           for q in ("mae", "within_host_mae")} for m in MODELS}

    def sizes(var: str) -> dict:
        rows = [r for r in results if r["variable"] == var]
        return {"groups": {r["group"]: [r["n_hosts"], r["n_sites"]] for r in rows},
                "sum_hosts": sum(r["n_hosts"] for r in rows), "sum_sites": sum(r["n_sites"] for r in rows)}

    nh, ns = pooled["n_hosts"], pooled["n_sites"]
    add = {v: sizes(v) for v in ("d_class", "phs_prone", "stability_tercile", "band_gap_tercile", "bader_rank",
                                 "cation")}
    for v in ("d_class", "phs_prone", "stability_tercile", "band_gap_tercile"):
        add[v]["adds_up"] = add[v]["sum_hosts"] == nh and add[v]["sum_sites"] == ns
    elig = bader_rank_labels(sites).notna()
    add["bader_rank"]["eligible_sites"] = int(elig.sum())
    add["bader_rank"]["eligible_hosts"] = int(sites.host_id[elig].nunique())
    add["bader_rank"]["adds_up"] = add["bader_rank"]["sum_sites"] == int(elig.sum())
    kept = [r["group"] for r in results if r["variable"] == "cation"]
    no_kept = hosts.cations.map(lambda cs: not any(e in kept for e in cs))
    add["cation"]["hosts_without_a_listed_cation"] = int(no_kept.sum())
    add["cation"]["host_counts_all_elements"] = dict(sorted(
        pd.Series([e for cs in hosts.cations for e in cs]).value_counts().astype(int).to_dict().items()))
    sanity["iii_group_sizes"] = add
    sanity["hosts"] = {"universe": n_all_hosts, "tested": nh, "never_tested": n_all_hosts - nh}
    sanity["sites_tested"] = ns
    sanity["resamples_per_site"] = {str(k): int(v) for k, v in sites.n_resamples.value_counts().sort_index().items()}

    edges = {}
    for var, col in (("stability_tercile", "formation_energy"), ("band_gap_tercile", "band_gap")):
        q1, q2 = np.quantile(hosts[col].to_numpy(float), [1 / 3, 2 / 3])
        edges[var] = {"column": f"desc_{col}", "q1_3": float(q1), "q2_3": float(q2)}

    return {
        "label": "EXTENSION",
        "rule": "docs/deviations.md row of 2026-10-11, within-host breakdown (logged before computing)",
        "backbones": {"v2": "primary backbone (POST HOC, author decision 2026-10-11)",
                      "v1": "originally registered backbone"},
        "budget": BUDGET,
        "models": {"S": "S (v1 eval_table 'S'; v2 'S')", "D": "D-state (v1 eval_table 'D'; v2 'D-state')"},
        "definitions": {
            "prediction": "seed-ensemble mean (seeds 0-2) per (model, r, site), averaged over the resamples "
                          "in which the site's host was tested",
            "mae": "mean |p - y| over the group's sites",
            "within_host_mae": "per host with >= 2 tested sites: |(p - host mean p) - (y - host mean y)|, "
                               "host means over all tested sites; mean over the group's sites in such hosts",
            "zero_skill_mae": "|y - host mean y| over the same sites as within_host_mae",
            "signed_residual": "mean of (p - host mean p) - (y - host mean y) over the same sites",
            "spearman": "per host with >= 3 tested sites, Spearman(p, y) with average ranks; NaN (excluded, "
                        "counted) if p or y is constant; mean over hosts",
            "phs_prone": list(PHS_PRONE),
            "cation": f"every element other than O; elements in >= {MIN_TEST_HOSTS_CATION} tested hosts",
            "bader_rank": "hosts with >= 2 tested sites and non-constant desc_bader_charge: lowest = host "
                          "minimum (most negative), highest = host maximum, middle = strictly between",
            "terciles": edges,
            "small_n": f"groups with < {MIN_HOSTS_INTERVAL} hosts: flag, no interval; an interval only if >= "
                       f"{MIN_HOSTS_INTERVAL} hosts contribute to the metric",
        },
        "bootstrap": {"draws": N_BOOT, "type": "host-cluster within group, multinomial weights",
                      "generator": "numpy.random.default_rng(numpy.random.SeedSequence([20261012, k]))",
                      "ci": CI, "quantile": "numpy linear", "paired": "S, D, v1, v2 share each group's draws"},
        "inputs": {"v1": v1_meta, "v2": v2_meta,
                   "universe": {"path": "data/processed/universe_v1.parquet", "sha256": sha256(upath),
                                "read_only": True},
                   "dbins": {"path": "data/host_dbins_v1.csv", "sha256": sha256(root / "data/host_dbins_v1.csv")}},
        "groups": results,
        "spearman": spear,
        "sanity": sanity,
    }


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, default=None,
                    help="checkout holding data/processed/universe_v1.parquet (default: this repo)")
    a = ap.parse_args(argv)
    from dftgnn.io.results import write_result
    root = Path(__file__).resolve().parents[1]
    payload = run(root, a.data_root or root)
    out = write_result("within_host_breakdown", payload, repo_root=root, results_dir=root / "results")
    print(out)


if __name__ == "__main__":
    main()
