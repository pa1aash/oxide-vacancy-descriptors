"""Where the descriptor advantage lives: hosts used in tuning, concentration over hosts, chemical-system siblings,
high-moment entries and family-level clustering of the leave-family-out design.

Nothing is trained. Physically: A* = MAE_S - MAE_D* (and A = MAE_S - MAE_D) is the error of the neutral
oxygen-vacancy formation energy removed by the pristine-cell electronic descriptors (band gap, oxide formation
energy, dielectric response, band-edge character, Bader charge) beyond what the crystal structure gives. A gain
that is real descriptor physics should not hinge on the hosts that picked the hyperparameters, on a handful of
hard hosts, on test hosts whose element set already sits in training, or on the few vacancies that end with a
localised moment (Ce(IV) reduction, low-lying empty La 4f states in PBEsol without U) instead of a closed-shell
vacancy state. Each block restricts the headline estimator to a subset of test sites or re-weights it.

Headline design: 654 training hosts, 30 resamples x 5 seeds, 164 test hosts per resample; S and D from
results/analysis/precision_eval_long.parquet (v2, B = 654), D* from results/dretrain_v2 (loaders of
analysis/dstar/dstar_analysis.py, every check of that script applied). Estimator: seed-ensemble mean prediction per
(model, r, site); per resample the site MAE over the subset's test sites of that resample; point = mean over the
resamples whose test set holds at least one subset site. Intervals: HierDraws.make([164] * 30, 2000, 20261008) of
src/dftgnn/stats/g1.py, the draws of the paper. The subset is a site filter applied inside each drawn resample: a
drawn slot keeps its resample's bootstrap host multiplicities, its MAE is sum(w_h * sum_{subset sites of h} |e|) /
sum(w_h * n_subset_sites_h), slots with no subset site are left out of the replicate mean (none at the full set).
Every gap is paired on the same draws; ci95 = 2.5/97.5 percentiles, U = upper_one_sided_95 = 95th percentile. With
the full site set the estimator reproduces results/derived_dstar/dstar_gaps.json to 1e-9 eV.

Blocks
    tuning_hosts   The 654-host settings of S, D and D* were selected on resample 0 at 654 hosts: its training pool
                   (654 hosts) split by dftgnn.split.val_split(frac 0.1, min 3, seed val_seed(0, 654, 0)) into 589
                   training and 65 validation hosts (reproduced and checked against the n_train/n_val host and site
                   counts of every 654-anchor trial CSV in results/tuning). A*, A without the 65 validation hosts in
                   any test set; on those 65 hosts only (resample 0 holds none of them); a random-subset reference
                   (2000 random 65-host subsets of resample 0's training pool, seed 20261008, same point estimator);
                   the fraction of each resample's test hosts inside the tuning set; A*, A of resample 0 (no test
                   host in tuning) against the mean over resamples 1-29.
    concentration  Site appearances = (resample, test site) pairs of the 30 resamples. Paired slot difference
                   d = |e_S| - |e_D*| (and |e_S| - |e_D|) of the seed ensembles. Pooled value = mean of d over all
                   appearances; a host's contribution = sum of d over its appearances / total appearances, so the
                   contributions add up to the pooled value. Shares of the pooled value carried by the top
                   ceil(5 %) of hosts and by the top 10 hosts; median of d; fraction of appearances with d > 0 and of
                   hosts whose mean d over their appearances is > 0 (D* more accurate); 10 % trimmed mean (10 % of
                   appearances cut at each end, scipy.stats.trim_mean). Intervals: host-cluster bootstrap over the
                   distinct test hosts (HierDraws.make([n_hosts], 2000, 20261008): hosts drawn with replacement,
                   every appearance of a drawn host kept with its multiplicity).
    chemsys        Element set of each host from its formula (data/universe_v1_ids.csv, pymatgen Composition).
                   Per resample, a test site is "absent" when no host of that resample's training pool (all 654
                   non-test hosts) has the same element set, "present" otherwise. A*, A on each subset.
    high_moment    The 10 entries with |defect moment| > 0.5 muB listed in results/derived_n4/high_moment_final_
                   network.json (that record retrains without them on the 10 x 3 design and reports A only; its
                   primary_common_sites block is the test-side exclusion with the original runs, reproduced here).
                   Test-side exclusion on the headline design: A*, A without these sites; change excluded - all,
                   paired per replicate.
    loco_family    Leave-family-out design (splits/loco_v1.json: 131 host families in 5 GroupKFold folds, each fold
                   trained without its families; results/loco_v2 S and D-state runs, seeds 0-2). Pooled A over the
                   1726 out-of-fold sites, reproduced with its host bootstrap from results/derived_c1/loco_family.json;
                   then a family-cluster bootstrap: HierDraws.make([131], 2000, 20261008), families drawn with
                   replacement, every site of a drawn family kept.

Writes through ``write_result`` into ``results/derived_trv1_subsets/subset_sens.json``.

    PYTHONPATH=$PWD/src python analysis/trv1/subset_sens.py [--dry-run --out FILE]
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis import a1_common as c
from analysis.c1.final_network import compare, ingest_module
from analysis.dstar import dstar_analysis as da
from dftgnn.stats.g1 import HierDraws, seed_ensemble

NAME = "subset_sens"
OUT_DIR = ROOT / "results" / "derived_trv1_subsets"
B_HEAD = da.HEADLINE_B
N_R, N_S = da.GRID[B_HEAD]
N_HOSTS = 164
DELTA = c.DELTA_EV
MODELS = ("S", "D", "Dstar")
GAPS = (("S_minus_Dstar", "S", "Dstar"), ("S_minus_D", "S", "D"))
HM_REC = "derived_n4/high_moment_final_network"
LOCO_REC = "derived_c1/loco_family"
TRIM = 0.10
N_RANDOM_SUBSETS = 2000


# ---------------------------------------------------------------- subset estimator on shared draws

class Design:
    """Seed-ensemble site errors of one design in wide form, with per-resample host indices matching the
    host-sorted rows of dftgnn.stats.g1.resample_stats (and therefore the HierDraws weight columns)."""

    def __init__(self, ens: pd.DataFrame, models, n_r: int, draws: HierDraws):
        wide = None
        for m in models:
            e = ens[ens.model == m][["r", "site_id", "host_id", "y_true", "y_pred"]].rename(
                columns={"y_pred": f"p_{m}"})
            if wide is None:
                wide = e
            else:
                wide = wide.merge(e, on=["r", "site_id"], suffixes=("", "_x"), validate="one_to_one")
                if not ((wide.host_id == wide.host_id_x).all() and np.array_equal(wide.y_true, wide.y_true_x)):
                    raise AssertionError(f"{m}: hosts or targets differ from the first model")
                wide = wide.drop(columns=["host_id_x", "y_true_x"])
        wide = wide.sort_values(["r", "host_id", "site_id"]).reset_index(drop=True)
        if sorted(wide.r.unique()) != list(range(n_r)):
            raise AssertionError("resamples missing")
        hidx = np.empty(len(wide), int)
        self.hosts = []
        for r in range(n_r):
            m = (wide.r == r).to_numpy()
            hs = sorted(wide.host_id[m].unique())
            self.hosts.append(hs)
            pos = {h: i for i, h in enumerate(hs)}
            hidx[m] = wide.host_id[m].map(pos).to_numpy()
        if tuple(len(h) for h in self.hosts) != draws.n_hosts:
            raise AssertionError("host count per resample differs from the draws")
        wide["hidx"] = hidx
        self.wide, self.models, self.n_r, self.draws = wide, tuple(models), n_r, draws
        self.H = max(draws.n_hosts)
        self.ae = {m: np.abs(wide[f"p_{m}"].to_numpy() - wide.y_true.to_numpy()) for m in models}
        self.r = wide.r.to_numpy()

    def sums(self, mask: np.ndarray, m: str) -> tuple[np.ndarray, np.ndarray]:
        """(sum |e|, n sites) per (resample, host index) over the masked sites."""
        sab = np.zeros((self.n_r, self.H))
        n = np.zeros((self.n_r, self.H))
        np.add.at(sab, (self.r[mask], self.wide.hidx.to_numpy()[mask]), self.ae[m][mask])
        np.add.at(n, (self.r[mask], self.wide.hidx.to_numpy()[mask]), 1.0)
        return sab, n

    def estimate(self, mask: np.ndarray) -> dict:
        """model -> (per-resample MAE, NaN where the resample has no subset site; replicate values)."""
        d = self.draws
        out = {}
        for m in self.models:
            sab, n = self.sums(mask, m)
            with np.errstate(invalid="ignore", divide="ignore"):
                per = sab.sum(1) / n.sum(1)
            ts = np.zeros((d.n_boot, self.n_r))
            tn = np.zeros((d.n_boot, self.n_r))
            for j in range(self.n_r):
                sel = d.pick == j
                ts[sel] = d.w[sel] @ sab[j]
                tn[sel] = d.w[sel] @ n[j]
            with np.errstate(invalid="ignore", divide="ignore"):
                slot = ts / tn
            empty = tn == 0
            if empty.all(axis=1).any():
                raise AssertionError("a replicate without any subset site")
            rep = np.nanmean(np.where(empty, np.nan, slot), axis=1)
            out[m] = (per, rep)
        return out


def gap_stats(per: np.ndarray, rep: np.ndarray) -> dict:
    lo, hi = float(np.quantile(rep, 0.025)), float(np.quantile(rep, 0.975))
    up = float(np.quantile(rep, c.UPPER_Q))
    ok = per[np.isfinite(per)]
    return {"A": float(ok.mean()), "ci95": [lo, hi], "upper_one_sided_95": up,
            "upper_below_delta": bool(up < DELTA), "ci95_excludes_zero": bool(lo > 0 or hi < 0),
            "boot_sd": float(rep.std(ddof=1)), "n_resamples_used": int(ok.size),
            "per_resample": [float(x) if math.isfinite(x) else None for x in per]}


def marginal(per: np.ndarray, rep: np.ndarray) -> dict:
    ok = per[np.isfinite(per)]
    return {"mean": float(ok.mean()), "ci95": [float(np.quantile(rep, 0.025)), float(np.quantile(rep, 0.975))]}


def subset_block(des: Design, mask: np.ndarray, full: dict | None = None) -> dict:
    est = des.estimate(mask)
    out = {"MAE": {m: marginal(*est[m]) for m in des.models},
           "gaps": {k: gap_stats(est[a][0] - est[b][0], est[a][1] - est[b][1])
                    for k, a, b in GAPS if a in est and b in est}}
    w = des.wide[mask]
    per_r = w.groupby("r").size().reindex(range(des.n_r), fill_value=0)
    out["counts"] = {"n_site_appearances": len(w), "n_distinct_sites": int(w.site_id.nunique()),
                     "n_distinct_hosts": int(w.host_id.nunique()),
                     "n_resamples_with_sites": int((per_r > 0).sum()),
                     "sites_per_resample_min": int(per_r.min()), "sites_per_resample_max": int(per_r.max()),
                     "sites_per_resample_mean": float(per_r.mean()),
                     "fraction_of_all_site_appearances": float(mask.mean())}
    if full is not None:
        ch = {}
        for k, a, b in GAPS:
            if a not in est or b not in est:
                continue
            rep = (est[a][1] - est[b][1]) - (full[a][1] - full[b][1])
            pt = out["gaps"][k]["A"] - float(np.mean(full[a][0] - full[b][0]))
            lo, hi = float(np.quantile(rep, 0.025)), float(np.quantile(rep, 0.975))
            ch[k] = {"change": pt, "ci95": [lo, hi], "ci95_excludes_zero": bool(lo > 0 or hi < 0),
                     "boot_sd": float(rep.std(ddof=1))}
        out["change_from_all_sites"] = ch
    return out


def short(block: dict) -> dict:
    """The two headline gaps of a subset block without per-resample lists (for summaries)."""
    return {k: {x: v[x] for x in ("A", "ci95", "upper_one_sided_95")} for k, v in block["gaps"].items()}


# ---------------------------------------------------------------- headline data

def headline_design() -> tuple[Design, dict]:
    ref, f_ref = da.s_reference()
    dstar, info = da.load_dstar(ref)
    prec, _ = c.load_table("precision_eval_long")
    base = prec[(prec.backbone == "v2") & (prec.B == B_HEAD) & (prec.r < N_R) & (prec.seed < N_S)][da.COLS]
    long = pd.concat([base, dstar[dstar.B == B_HEAD][da.COLS]], ignore_index=True)
    ens = seed_ensemble(long)
    if (ens.n_seeds != N_S).any():
        raise AssertionError("headline: seed ensemble over the wrong number of seeds")
    draws = HierDraws.make([N_HOSTS] * N_R, c.N_BOOT, seed=c.BOOT_SEED)
    return Design(ens, MODELS, N_R, draws), {"inputs": f_ref, "dstar_spec_sha256": info["spec_sha256"]}


def reproduce_headline(full_block: dict) -> dict:
    st = c.record("derived_dstar/dstar_gaps")["payload"]["headline_654_30x5"][str(B_HEAD)]
    mine = {"MAE": {m: full_block["MAE"][m] for m in MODELS},
            "gaps": {k: {x: full_block["gaps"][k][x] for x in ("A", "ci95", "upper_one_sided_95", "per_resample")}
                     for k, _, _ in GAPS}}
    ref = {"MAE": {m: st["MAE"][m] for m in MODELS},
           "gaps": {k: {x: st["gaps"][k][x] for x in ("A", "ci95", "upper_one_sided_95", "per_resample")}
                    for k, _, _ in GAPS}}
    return compare(mine, ref, "results/derived_dstar/dstar_gaps.json payload.headline_654_30x5.654 (MAE of S, D, "
                              "D* with ci95; S - D* and S - D: A, ci95, upper_one_sided_95, per_resample)")


# ---------------------------------------------------------------- block 1: tuning hosts

def tuning_hosts_sets() -> dict:
    from dftgnn.config import load_config
    from dftgnn.split import budget_train, load_split, val_split
    from dftgnn.train import val_seed

    cfg = load_config()
    t, v = cfg.tuning, cfg.training.validation
    r0 = int(t.anchor_resample)
    sp = load_split(f"outer_r{r0}")
    pool = budget_train(sp, B_HEAD)
    tr, va = val_split(pool, frac=v.frac, min_hosts=v.min_hosts, seed=val_seed(r0, B_HEAD, int(t.trial_seed)))
    ids = pd.read_csv(ROOT / "data" / "universe_v1_ids.csv")
    nsites = ids.groupby("host_id").size()
    n_tr_sites, n_va_sites = int(nsites.loc[tr].sum()), int(nsites.loc[va].sum())
    checked = []
    for p in sorted((ROOT / "results" / "tuning").glob("trials_*_a654.csv")):
        df = pd.read_csv(p)
        df = df[df.state == "COMPLETE"]
        got = {(int(a), int(b), int(x), int(y)) for a, b, x, y in
               zip(df.n_train_hosts, df.n_val_hosts, df.n_train_sites, df.n_val_sites)}
        if got != {(len(tr), len(va), n_tr_sites, n_va_sites)}:
            raise AssertionError(f"{p.name}: tuning host/site counts {got} differ from the reproduced split")
        checked.append({"file": str(p.relative_to(ROOT)), "n_complete_trials": len(df)})
    if not set(va) <= set(pool) or set(sp["test"]) & set(pool) or len(pool) != B_HEAD:
        raise AssertionError("tuning pool")
    return {"r0": r0, "test_r0": sorted(sp["test"]), "pool": sorted(pool), "train": sorted(tr), "val": sorted(va),
            "n_train_sites": n_tr_sites, "n_val_sites": n_va_sites, "trial_csvs_checked": checked,
            "val_seed": int(val_seed(r0, B_HEAD, int(t.trial_seed))), "frac": float(v.frac),
            "min_hosts": int(v.min_hosts)}


def random_subset_reference(des: Design, cand: list[str], k: int, value: dict) -> dict:
    """Point A*, A on 2000 random k-host subsets of ``cand`` (same estimator as the 65-host value)."""
    allh = sorted(set(des.wide.host_id))
    col = {h: i for i, h in enumerate(allh)}
    hc = des.wide.host_id.map(col).to_numpy()
    sab = {m: np.zeros((des.n_r, len(allh))) for m in des.models}
    n = np.zeros((des.n_r, len(allh)))
    for m in des.models:
        np.add.at(sab[m], (des.r, hc), des.ae[m])
    np.add.at(n, (des.r, hc), 1.0)
    rng = np.random.default_rng(c.BOOT_SEED)
    ci = np.array([col[h] for h in cand if h in col])
    if len(ci) != len(cand):
        raise AssertionError("candidate host never tested")
    ind = np.zeros((N_RANDOM_SUBSETS, len(allh)))
    for b in range(N_RANDOM_SUBSETS):
        ind[b, rng.choice(ci, size=k, replace=False)] = 1.0
    nn = n @ ind.T
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        mae = {m: (sab[m] @ ind.T) / nn for m in des.models}
    for key, a, b in GAPS:
        vals = np.nanmean(mae[a] - mae[b], axis=0)
        obs = value[key]["A"]
        out[key] = {"observed_65_validation_hosts": obs, "random_mean": float(vals.mean()),
                    "random_median": float(np.median(vals)),
                    "random_q025_q975": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))],
                    "random_max": float(vals.max()),
                    "n_random_at_or_above_observed": int((vals >= obs).sum()),
                    "fraction_at_or_above_observed": float((vals >= obs).mean())}
    return {"n_subsets": N_RANDOM_SUBSETS, "subset_size": k, "rng_seed": c.BOOT_SEED,
            "population": "the 654 hosts of resample 0's training pool (the population the validation hosts were "
                          "drawn from); subsets drawn without replacement with numpy default_rng(20261008).choice",
            **out}


def tuning_block(des: Design, full: dict, full_block: dict) -> dict:
    s = tuning_hosts_sets()
    V, T = set(s["val"]), set(s["pool"])
    hosts = des.wide.host_id
    ex = subset_block(des, ~hosts.isin(V).to_numpy(), full)
    only = subset_block(des, hosts.isin(V).to_numpy(), full)
    frac = []
    for r in range(des.n_r):
        th = des.hosts[r]
        frac.append(sum(h in T for h in th) / len(th))
    frac = np.array(frac)
    nval_test = [sum(h in V for h in des.hosts[r]) for r in range(des.n_r)]
    r0 = {}
    for k, _, _ in GAPS:
        per = np.array(full_block["gaps"][k]["per_resample"], float)
        rest = per[1:]
        r0[k] = {"resample_0": float(per[0]), "mean_resamples_1_29": float(rest.mean()),
                 "sd_resamples_1_29": float(rest.std(ddof=1)),
                 "resample_0_minus_mean_1_29": float(per[0] - rest.mean()),
                 "rank_of_resample_0_among_30_ascending": int((per < per[0]).sum() + 1)}
    r0_draws = HierDraws.make([N_HOSTS], c.N_BOOT, seed=c.BOOT_SEED)
    sub = Design(des.wide[des.wide.r == 0].melt(
        id_vars=["r", "site_id", "host_id", "y_true"], value_vars=[f"p_{m}" for m in des.models],
        var_name="model", value_name="y_pred").assign(model=lambda x: x.model.str[2:]), des.models, 1, r0_draws)
    r0b = sub.estimate(np.ones(len(sub.wide), bool))
    for k, a, b in GAPS:
        g = gap_stats(r0b[a][0] - r0b[b][0], r0b[a][1] - r0b[b][1])
        r0[k]["resample_0_host_bootstrap_ci95"] = g["ci95"]
        if abs(g["A"] - r0[k]["resample_0"]) > 1e-12:
            raise AssertionError("resample-0 value differs between routes")
    return {
        "construction": "tuning resample r = 0, anchor 654 hosts: training pool = budget_train(outer_r0, 654) "
                        "(all 654 non-test hosts of resample 0); validation hosts = val_split(pool, frac 0.1, "
                        "min_hosts 3, seed val_seed(0, 654, 0)); tuning set = pool = training + validation hosts. "
                        "The 654-anchor studies of S, D-state, D-late, P1 (tuned_v2) and the D* study (v2_D_a654) "
                        "all record these host and site counts",
        "split": {"n_pool": len(s["pool"]), "n_train": len(s["train"]), "n_val": len(s["val"]),
                  "n_train_sites": s["n_train_sites"], "n_val_sites": s["n_val_sites"], "val_seed": s["val_seed"],
                  "validation_hosts": s["val"], "trial_csvs_checked": s["trial_csvs_checked"],
                  "reproduced_exactly": True},
        "excluding_validation_hosts": ex,
        "validation_hosts_only": only,
        "validation_hosts_only_random_reference": random_subset_reference(des, s["pool"], len(V), only["gaps"]),
        "overlap": {"definition": "fraction of a resample's 164 test hosts inside the 654-host tuning set",
                    "mean_resamples_1_29": float(frac[1:].mean()), "min_resamples_1_29": float(frac[1:].min()),
                    "max_resamples_1_29": float(frac[1:].max()), "mean_all_30": float(frac.mean()),
                    "resample_0": float(frac[0]), "per_resample": frac.tolist(),
                    "n_validation_hosts_in_test_per_resample": nval_test,
                    "n_validation_hosts_ever_tested_in_1_29": len(V & set().union(*map(set, des.hosts[1:])))},
        "resample_0_vs_rest": {**r0, "note": "resample 0 is the tuning resample: none of its test hosts entered "
                                             "tuning; the interval is a host bootstrap within resample 0 "
                                             "(HierDraws.make([164], 2000, 20261008))"},
    }


# ---------------------------------------------------------------- block 2: concentration

def wmedian(sorted_vals: np.ndarray, cw: np.ndarray) -> np.ndarray:
    """np.median of the expanded sample for integer weights; cw (n_boot, n) cumulative weights in sorted order."""
    tot = cw[:, -1]
    lo = np.array([np.searchsorted(cw[b], (tot[b] - 1) // 2, side="right") for b in range(len(tot))])
    hi = np.array([np.searchsorted(cw[b], tot[b] // 2, side="right") for b in range(len(tot))])
    return 0.5 * (sorted_vals[lo] + sorted_vals[hi])


def wtrim(sorted_vals: np.ndarray, w: np.ndarray, frac: float) -> np.ndarray:
    """scipy.stats.trim_mean(expanded sample, frac) for integer weights w (n_boot, n) in sorted order."""
    cw = np.cumsum(w, axis=1)
    tot = cw[:, -1:]
    g = np.floor(frac * tot)
    start, end = cw - w, cw
    keep = np.clip(np.minimum(end, tot - g) - np.maximum(start, g), 0, None)
    return (keep * sorted_vals).sum(1) / keep.sum(1)


def concentration_one(des: Design, model: str, draws: HierDraws) -> dict:
    from scipy.stats import trim_mean

    w = des.wide
    d = des.ae["S"] - des.ae[model]
    hosts = sorted(w.host_id.unique())
    hpos = {h: i for i, h in enumerate(hosts)}
    hi = w.host_id.map(hpos).to_numpy()
    nH, N = len(hosts), len(d)
    Dh = np.bincount(hi, weights=d, minlength=nH)
    Nh = np.bincount(hi, minlength=nH).astype(float)
    Wh = np.bincount(hi, weights=(d > 0).astype(float), minlength=nH)
    pooled = float(d.mean())
    contrib = Dh / N
    order = np.argsort(-contrib, kind="stable")
    n5 = math.ceil(0.05 * nH)
    cum = np.cumsum(contrib[order])
    n_half = int(np.searchsorted(cum, 0.5 * pooled) + 1)
    hmean = Dh / Nh
    point = {"pooled_mean": pooled, "median": float(np.median(d)), "trimmed_mean_10pct": float(trim_mean(d, TRIM)),
             "fraction_appearances_D_more_accurate": float((d > 0).mean()),
             "fraction_hosts_D_more_accurate": float((hmean > 0).mean())}
    cnt = draws.w[:, 0, :nH]
    srt = np.argsort(d, kind="stable")
    sv = d[srt]
    ws = cnt[:, hi[srt]]
    reps = {"pooled_mean": (cnt @ Dh) / (cnt @ Nh),
            "median": wmedian(sv, np.cumsum(ws, axis=1)),
            "trimmed_mean_10pct": wtrim(sv, ws, TRIM),
            "fraction_appearances_D_more_accurate": (cnt @ Wh) / (cnt @ Nh),
            "fraction_hosts_D_more_accurate": (cnt @ (hmean > 0).astype(float)) / nH}
    stats = {k: {"point": point[k], "ci95": [float(np.quantile(reps[k], 0.025)), float(np.quantile(reps[k], 0.975))],
                 "boot_sd": float(reps[k].std(ddof=1))} for k in point}
    ids = pd.read_csv(ROOT / "data" / "universe_v1_ids.csv").drop_duplicates("host_id").set_index("host_id").formula
    top = [{"host_id": hosts[i], "formula": str(ids[hosts[i]]), "contribution": float(contrib[i]),
            "n_appearances": int(Nh[i]), "host_mean_difference": float(hmean[i])} for i in order[:10]]
    return {"n_site_appearances": N, "n_hosts": nH, "n_top_5pct_hosts": n5,
            "share_top_5pct_hosts": float(contrib[order[:n5]].sum() / pooled),
            "share_top_10_hosts": float(contrib[order[:10]].sum() / pooled),
            "n_hosts_carrying_half_of_pooled": n_half,
            "contribution_check_sum": float(contrib.sum()),
            "n_hosts_negative_contribution": int((contrib < 0).sum()),
            **stats, "top_10_hosts": top}


def concentration_block(des: Design) -> dict:
    nH = des.wide.host_id.nunique()
    draws = HierDraws.make([nH], c.N_BOOT, seed=c.BOOT_SEED)
    return {"definition": "site appearance = (resample, test site) over the 30 resamples; d = |e_S| - |e_X| of the "
                          "seed ensembles (positive = X more accurate); pooled = mean of d over all appearances; "
                          "contribution of a host = sum of d over its appearances / number of all appearances; "
                          "shares = sum of the largest contributions / pooled; fraction of hosts uses the host mean "
                          "of d over its appearances; trimmed mean cuts 10 % of appearances at each end "
                          "(scipy.stats.trim_mean); intervals: host-cluster bootstrap over the distinct test hosts, "
                          "HierDraws.make([n_hosts], 2000, 20261008), multiplicity of a drawn host applied to all "
                          "its appearances (median and trimmed mean of the expanded sample)",
            "note": "the pooled mean weights resamples by their site count and so differs slightly from the "
                    "paper estimator (mean over resamples of the per-resample MAE)",
            "Dstar": concentration_one(des, "Dstar", draws), "D": concentration_one(des, "D", draws)}


# ---------------------------------------------------------------- block 3: chemical systems

def element_sets() -> pd.Series:
    from pymatgen.core import Composition

    ids = pd.read_csv(ROOT / "data" / "universe_v1_ids.csv").drop_duplicates("host_id").set_index("host_id")
    return ids.formula.map(lambda f: "-".join(sorted(str(e) for e in Composition(f).elements)))


def chemsys_block(des: Design, full: dict) -> dict:
    es = element_sets()
    allhosts = set(es.index)
    w = des.wide
    absent = np.zeros(len(w), bool)
    per_r = []
    for r in range(des.n_r):
        test = set(des.hosts[r])
        pool_sets = set(es.loc[sorted(allhosts - test)])
        m = (w.r == r).to_numpy()
        a = ~w.host_id[m].map(es).isin(pool_sets).to_numpy()
        absent[m] = a
        th_abs = sum(es[h] not in pool_sets for h in test)
        per_r.append({"r": r, "n_pool_hosts": len(allhosts - test), "n_test_sites_absent": int(a.sum()),
                      "n_test_sites_present": int((~a).sum()), "n_test_hosts_absent": int(th_abs),
                      "n_test_hosts_present": len(test) - int(th_abs)})
    vc = es.value_counts()
    shared = es[es.map(vc) > 1]
    return {"definition": "element set = sorted distinct elements of the host formula (pymatgen Composition of "
                          "data/universe_v1_ids.csv formula); a test site is absent when no host of that resample's "
                          "training pool (the 654 non-test hosts) has the same element set",
            "n_hosts": len(es), "n_element_sets": int(vc.size),
            "n_hosts_sharing_element_set_with_another_host": len(shared),
            "n_element_sets_shared_by_two_or_more_hosts": int((vc > 1).sum()),
            "fraction_site_appearances_absent": float(absent.mean()),
            "absent_from_training_pool": subset_block(des, absent, full),
            "present_in_training_pool": subset_block(des, ~absent, full),
            "per_resample_counts": per_r}


# ---------------------------------------------------------------- block 4: high-moment entries

def high_moment_block(des: Design, full: dict) -> dict:
    from analysis.dclass import dclass_analysis as dca

    rec = c.record(HM_REC)["payload"]
    excl = sorted(e["site_id"] for e in rec["excluded_entries"])
    w = des.wide
    keep = ~w.site_id.isin(excl).to_numpy()
    out_head = subset_block(des, keep, full)
    n_in = w[~keep].groupby("r").size().reindex(range(des.n_r), fill_value=0)

    # reproduction: test-side exclusion with the original runs on the 10 x 3 design (primary_common_sites)
    sweep, f_sweep = dca.load_sweep()
    s654 = sweep[sweep.B == B_HEAD]
    ens = seed_ensemble(s654)
    if (ens.n_seeds != 3).any():
        raise AssertionError("sweep seeds")
    sdraws = HierDraws.make([N_HOSTS] * 10, c.N_BOOT, seed=c.BOOT_SEED)
    sdes = Design(ens, ("S", "D"), 10, sdraws)
    sfull = sdes.estimate(np.ones(len(sdes.wide), bool))
    skeep = ~sdes.wide.site_id.isin(excl).to_numpy()
    scom = sdes.estimate(skeep)

    def view(est):
        g = gap_stats(est["S"][0] - est["D"][0], est["S"][1] - est["D"][1])
        return {"A": {"mean": g["A"], "ci": g["ci95"], "upper_one_sided": g["upper_one_sided_95"]},
                "MAE_S": {"mean": float(np.nanmean(est["S"][0])),
                          "ci": [float(np.quantile(est["S"][1], q)) for q in (0.025, 0.975)]},
                "MAE_D": {"mean": float(np.nanmean(est["D"][0])),
                          "ci": [float(np.quantile(est["D"][1], q)) for q in (0.025, 0.975)]}}
    fn = rec["final_network"]
    repro = {"primary_common_sites": compare(view(scom), fn["primary_common_sites"],
                                             f"results/{HM_REC}.json payload.final_network.primary_common_sites"),
             "primary_full_sites": compare(view(sfull), fn["primary_full_sites"],
                                           f"results/{HM_REC}.json payload.final_network.primary_full_sites")}
    nin_sweep = sdes.wide[~skeep].groupby("r").size().reindex(range(10), fill_value=0).tolist()
    if nin_sweep != [x["in_test"] for x in fn["excluded_per_resample"]]:
        raise AssertionError("excluded entries per resample differ from the record")
    return {"definition": "test-side exclusion: the listed sites are removed from every test set of the headline "
                          "design (no retraining); hosts keep their other sites; change = excluded - all, paired per "
                          "replicate on the headline draws",
            "source_record": f"results/{HM_REC}.json (threshold {rec['threshold_muB']} muB; that record retrains S "
                             "and D without the entries on the 10 x 3 design and reports A)",
            "excluded_site_ids": excl, "n_excluded_entries": len(excl),
            "n_excluded_in_test_per_resample": n_in.tolist(), "n_resamples_with_excluded_in_test": int((n_in > 0).sum()),
            "headline_654_30x5": out_head,
            "record_reproduction_10x3": repro,
            "record_values_10x3": {"primary_common_sites_A": fn["primary_common_sites"]["A"]["mean"],
                                   "retrained_high_moment_A": fn["high_moment"]["A"]["mean"]},
            "inputs": {"staging_eval_long": f_sweep}}


# ---------------------------------------------------------------- block 5: leave-family-out

def loco_block() -> dict:
    from dftgnn.split import load_split

    ing = ingest_module()
    br = ing.Branch(ROOT)
    split = load_split("loco_v1")
    fam = split["host_family"]
    ids = pd.read_csv(ROOT / "data" / "universe_v1_ids.csv")
    recs = br.records("loco_v2", ("S", "D-state"), "loco/")
    runs = br.predictions(recs, {"S": "S", "D-state": "D"})
    runs["r"] = runs.run_id.map({p["run_id"]: int(p["spec"]["split"].split("_f")[1]) for p in recs})
    for (m, k), g in runs.groupby(["model", "r"]):
        sites = set(ids.site_id[ids.host_id.isin(split["folds"][k]["test"])])
        if sorted(g.seed.unique()) != [0, 1, 2]:
            raise AssertionError(f"LOCO {m} fold {k}: seeds")
        for _, gs in g.groupby("seed"):
            if set(gs.site_id) != sites or gs.site_id.duplicated().any():
                raise AssertionError(f"LOCO {m} fold {k}: sites differ from the fold's test hosts")
    ens = ing.ensemble(runs)
    wide = ens[ens.model == "S"].merge(ens[ens.model == "D"], on=["site_id", "r"], suffixes=("_S", "_D"),
                                       validate="one_to_one")
    if not ((wide.host_id_S == wide.host_id_D).all() and np.allclose(wide.y_true_S, wide.y_true_D)):
        raise AssertionError("LOCO S and D differ in hosts or targets")
    wide = wide.rename(columns={"host_id_S": "host_id", "y_true_S": "y_true"})
    wide["family"] = wide.host_id.map(fam)
    eS = (wide.y_pred_S - wide.y_true).abs().to_numpy()
    eD = (wide.y_pred_D - wide.y_true).abs().to_numpy()

    def cluster(key: str) -> tuple[float, np.ndarray, int]:
        labs = sorted(wide[key].unique())
        pos = wide[key].map({x: i for i, x in enumerate(labs)}).to_numpy()
        sS = np.bincount(pos, weights=eS, minlength=len(labs))
        sD = np.bincount(pos, weights=eD, minlength=len(labs))
        n = np.bincount(pos, minlength=len(labs)).astype(float)
        dr = HierDraws.make([len(labs)], c.N_BOOT, seed=c.BOOT_SEED).w[:, 0, :len(labs)]
        rep = (dr @ sS) / (dr @ n) - (dr @ sD) / (dr @ n)
        return float(sS.sum() / n.sum() - sD.sum() / n.sum()), rep, len(labs)

    def entry(pt, rep, n):
        lo, hi = float(np.quantile(rep, 0.025)), float(np.quantile(rep, 0.975))
        up = float(np.quantile(rep, c.UPPER_Q))
        return {"A": pt, "ci95": [lo, hi], "upper_one_sided_95": up, "upper_below_delta": bool(up < DELTA),
                "ci95_excludes_zero": bool(lo > 0 or hi < 0), "boot_sd": float(rep.std(ddof=1)), "n_clusters": n}
    host = entry(*cluster("host_id"))
    famb = entry(*cluster("family"))
    st = c.record(LOCO_REC)["payload"]["v2"]["pooled"]
    repro = compare({k: host[k] for k in ("A", "ci95", "upper_one_sided_95")},
                    {k: st[k] for k in ("A", "ci95", "upper_one_sided_95")},
                    f"results/{LOCO_REC}.json payload.v2.pooled (A, ci95, upper_one_sided_95; host bootstrap)")
    if wide.family.nunique() != split["n_families"]:
        raise AssertionError("family count")
    return {"design": "splits/loco_v1.json: 131 host families (S02 definition ii-a) in 5 GroupKFold folds; each "
                      "fold's S and D-state trained without its families (results/loco_v2, seeds 0-2, seed-ensemble "
                      "mean); pooled out-of-fold site MAE over 1726 sites; A = MAE_S - MAE_D (D* was not run on "
                      "this design)",
            "n_sites": len(wide), "n_hosts": int(wide.host_id.nunique()), "n_families": int(wide.family.nunique()),
            "n_runs": int(runs.run_id.nunique()),
            "host_bootstrap": {**host, "construction": "HierDraws.make([818], 2000, 20261008): hosts with "
                                                       "replacement, all sites of a drawn host"},
            "family_bootstrap": {**famb, "construction": "HierDraws.make([131], 2000, 20261008): families with "
                                                         "replacement, all sites of a drawn family"},
            "family_over_host_interval_width": float((famb["ci95"][1] - famb["ci95"][0])
                                                     / (host["ci95"][1] - host["ci95"][0])),
            "reproduction": repro}


# ---------------------------------------------------------------- main

def main() -> None:
    from dftgnn.config import load_config
    from dftgnn.io.results import write_result

    t0 = time.time()
    cfg = load_config()
    if float(cfg.stats.delta_eV) != DELTA:
        raise AssertionError("stats.delta_eV differs from 0.05")
    des, meta = headline_design()
    allm = np.ones(len(des.wide), bool)
    full = des.estimate(allm)
    full_block = subset_block(des, allm)
    times = {"load": time.time() - t0}
    repro = {"headline_all_sites": reproduce_headline(full_block)}
    t = time.time()
    tun = tuning_block(des, full, full_block)
    times["tuning_hosts"] = time.time() - t
    t = time.time()
    conc = concentration_block(des)
    times["concentration"] = time.time() - t
    t = time.time()
    chem = chemsys_block(des, full)
    times["chemsys"] = time.time() - t
    t = time.time()
    hm = high_moment_block(des, full)
    times["high_moment"] = time.time() - t
    repro.update({f"high_moment.{k}": v for k, v in hm["record_reproduction_10x3"].items()})
    t = time.time()
    loco = loco_block()
    times["loco_family"] = time.time() - t
    repro["loco_family.pooled_host_bootstrap"] = loco["reproduction"]
    bad = [k for k, v in repro.items() if not v["reproduced"]]
    if bad:
        raise AssertionError(f"not reproduced: {bad} {repro}")
    summary = {
        "all_sites": short(full_block),
        "excluding_65_validation_hosts": short(tun["excluding_validation_hosts"]),
        "only_65_validation_hosts": short(tun["validation_hosts_only"]),
        "chemsys_absent": short(chem["absent_from_training_pool"]),
        "chemsys_present": short(chem["present_in_training_pool"]),
        "high_moment_excluded": short(hm["headline_654_30x5"]),
        "loco_family_bootstrap": {k: loco["family_bootstrap"][k] for k in ("A", "ci95", "upper_one_sided_95")},
    }
    payload = {
        "label": "subset sensitivity of A* = MAE_S - MAE_D* and A = MAE_S - MAE_D on the headline design (654 hosts, "
                 "30 x 5): tuning hosts, concentration over hosts, chemical-system siblings, high-moment entries; "
                 "family-cluster bootstrap of the leave-family-out A",
        "delta_eV": DELTA,
        "sign_convention": {"S_minus_Dstar": "A* = MAE_S - MAE_D* (positive = D* more accurate)",
                            "S_minus_D": "A = MAE_S - MAE_D (positive = D more accurate)",
                            "d": "|e_S| - |e_X| per site appearance (positive = X more accurate)",
                            "change": "subset value minus all-site value, paired on the same draws"},
        "estimator": "seed-ensemble mean prediction per (model, r, site); per resample the site MAE over the "
                     "subset's test sites; point = mean over resamples with at least one subset site; intervals on "
                     "HierDraws.make([164] * 30, 2000, 20261008) with the subset applied as a site filter inside "
                     "each drawn resample slot (bootstrap host multiplicities kept, slot MAE over the subset sites "
                     "of the drawn hosts, empty slots left out of the replicate mean); ci95 = 2.5/97.5 percentiles; "
                     "upper_one_sided_95 (U) = 95th percentile; all gaps paired on the same draws",
        "design": "654 training hosts, outer resamples r = 0-29 x seeds 0-4; S and D from "
                  "results/analysis/precision_eval_long.parquet (v2, B = 654); D* from results/dretrain_v2 (D-late, "
                  "configs/tuned_d_v2.yaml)",
        "summary": summary,
        "all_sites": full_block,
        "tuning_hosts": tun, "concentration": conc, "chemsys": chem, "high_moment": hm, "loco_family": loco,
        "reproduction": repro,
        "reproduction_summary": {"max_abs_difference": max(v["max_abs_difference"] for v in repro.values()),
                                 "n_values_compared": sum(v["n_values_compared"] for v in repro.values()),
                                 "all_reproduced_below_1e-9": not bad},
        "runtime_s": {k: round(v, 1) for k, v in times.items()},
        "inputs": {**meta["inputs"], "dretrain_v2_spec_sha256": meta["dstar_spec_sha256"],
                   "universe_ids": {"path": "data/universe_v1_ids.csv",
                                    "sha256": c.sha256(ROOT / "data" / "universe_v1_ids.csv")},
                   "dstar_gaps": {"path": "results/derived_dstar/dstar_gaps.json",
                                  "sha256": c.sha256(ROOT / "results" / "derived_dstar" / "dstar_gaps.json")},
                   "high_moment": {"path": f"results/{HM_REC}.json", "sha256": c.sha256(ROOT / f"results/{HM_REC}.json")},
                   "loco_family": {"path": f"results/{LOCO_REC}.json",
                                   "sha256": c.sha256(ROOT / f"results/{LOCO_REC}.json")},
                   "loco_split": {"path": "splits/loco_v1.json", "sha256": c.sha256(ROOT / "splits" / "loco_v1.json")}},
    }
    payload = c.to_jsonable(payload)
    print(f"runtime_s {time.time() - t0:.1f}", file=sys.stderr)
    if "--dry-run" in sys.argv:
        if "--out" in sys.argv:
            out = Path(sys.argv[sys.argv.index("--out") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=1))
            print(out)
        else:
            print(json.dumps(payload, indent=1))
        return
    print(write_result(NAME, payload, config=cfg, repo_root=ROOT, results_dir=OUT_DIR))


if __name__ == "__main__":
    main()
