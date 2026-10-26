"""Interval width of the descriptor advantage when the distinct test hosts are the sampling unit, multiplicity-adjusted
intervals, the two halves of the 654-host design and the within-host rank order on the full design.

Nothing is trained. Physically: A* = MAE_S - MAE_D* is the error of the neutral oxygen-vacancy formation energy that
the pristine-cell electronic descriptors remove beyond the crystal structure. At 654 training hosts the training pool
of a resample is the complement of its 164 test hosts, so the 30 outer resamples test the same 818 hosts about six
times each: they differ only in which hosts are held out, not in the hosts that exist. The paper's hierarchical
bootstrap (HierDraws) redraws a host independently in every resample where it appears, treating its six appearances
as independent oxides. Here the oxide host itself is the sampling unit: a host drawn into a bootstrap world brings
every one of its appearances with it, so a host whose vacancy energies the networks get unusually right or wrong
moves all thirty resamples together.

Data and estimator (as the paper): S and D from results/analysis/precision_eval_long.parquet (backbone v2, 654
hosts), D* from results/dretrain_v2 (loaded and checked by analysis/dstar/dstar_analysis.load_dstar); seed ensemble =
mean prediction over seeds per (model, r, site); every quantity is a ratio of per-host sums (site MAE: sum |e| / n;
within-host residual MAE: sum of host-centred |residual| / sites, hosts with >= 2 tested sites; lowest-site selection:
hits / eligible hosts with >= 2 tested sites, the definition of analysis/coord/site_selection.py; within-host Spearman:
sum of per-host rank correlations / eligible hosts with >= 3 tested sites); a slot value is the ratio of the weighted
sums of one resample; a replicate is the mean over the slots; point = mean over the 30 resamples of the per-resample
value; differences are taken per replicate (paired); two-sided 95% = 2.5/97.5 percentiles, U = 95th percentile,
SD = replicate SD (ddof 1). Relative reduction R = A*/MAE_S: point = ratio of the point MAEs, replicate = ratio of
the replicate MAEs (analysis/dsens/dsens.py).

Bootstrap schemes (2000 replicates, seed 20261008 unless stated)
    hier      the paper's HierDraws.make([164] * 30, 2000, 20261008): slot k draws a resample with replacement, then
              that resample's 164 test hosts with replacement, independently in every slot
    host      host-cluster: V = default_rng(seed).multinomial(818, uniform, size=n_boot) over the 818 distinct test
              hosts (sorted by host_id); every resample keeps its own 30 slots, and a host's sites in resample r
              carry weight V[b, host] in every resample that tested it; also 20000 replicates for A* and A
    crossed   rng = default_rng(seed); V as in host (the same first draw, so the host weights equal those of host),
              then pick = rng.integers(0, 30, size=(n_boot, 30)); slot k is resample pick[b, k] with the host
              weights V[b, .]: a site enters slot k with multiplicity V[b, host] when its resample was drawn into
              the slot and its host was drawn (a site is kept only when both are drawn)

Blocks
    overlap        distinct test hosts and appearances per host over the 30 resamples
    schemes        A*, A, D - D*, within-host A*, selection D* - S and D - S, R under the three schemes; the lower
                   end of the A* interval is the largest tolerance the interval stays wholly above
    bonferroni     two-sided 1 - 0.05/6 intervals (0.4167/99.5833 percentiles) for A*, A, within-host A*,
                   selection D* - S and R under hier and host
    design_split   resamples 0-9 and 10-29 (5 seeds) and resamples 0-9 with seeds 0-2, each with
                   HierDraws.make([164] * k, 2000, 20261008); the difference (10-29 minus 0-9) with two intervals
    spearman_30x5  within-host Spearman of S, D, D* and S - D*, D - D*, S - D on 30 x 5 with hier intervals, next to
                   the stored 10 x 3 values (recomputed with the definition of analysis/within_host_breakdown.py)

Every stored HierDraws value is recomputed and compared with results/derived_dstar/dstar_gaps.json,
results/derived_dstar/dstar_extras.json, results/derived_coord/site_selection.json and results/derived_dsens/dsens.json
(tolerance 1e-9). Writes through ``write_result`` into ``results/derived_trv1_boot/boot_design.json``.

    PYTHONPATH=$PWD/src python analysis/trv1/boot_design.py [--dry-run --out FILE]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis import a1_common as c
from analysis import within_host_breakdown as wb
from analysis.c1.final_network import compare
from analysis.coord import site_selection as ss
from analysis.dstar import dstar_analysis as da
from dftgnn.stats.g1 import HierDraws, seed_ensemble
from dftgnn.stats.metrics import host_stats

OUT_DIR = ROOT / "results" / "derived_trv1_boot"
B = 654
N_R, N_S, N_HOSTS = 30, 5, 164
N_BOOT, N_BOOT_BIG, SEED = c.N_BOOT, 20000, c.BOOT_SEED
DELTA = c.DELTA_EV
N_CLAIMS = 6
BONF = (100 * 0.05 / N_CLAIMS / 2, 100 * (1 - 0.05 / N_CLAIMS / 2))
MODELS = ("S", "D", "Dstar")
TOL = 1e-9
SIGNS = {"A_star": "MAE_S - MAE_D* (positive = D* more accurate)",
         "A": "MAE_S - MAE_D (positive = D more accurate)",
         "D_minus_Dstar": "MAE_D - MAE_D* (positive = D* more accurate)",
         "within_host_A_star": "within-host residual MAE of S minus that of D* (positive = D* ranks sites better)",
         "selection_Dstar_minus_S": "hit fraction of D* minus that of S (positive = D* picks the lowest site more often)",
         "selection_D_minus_S": "hit fraction of D minus that of S",
         "relative_reduction": "A*/MAE_S (fraction of the structure-only error removed by the descriptors)",
         "spearman": "S_minus_Dstar etc.: Spearman of the first minus the second (negative = D* orders sites better)",
         "design_split_difference": "value on resamples 10-29 minus value on resamples 0-9"}


# ---------------------------------------------------------------- per-host ratio statistics

def frame_of(ens: pd.DataFrame, m: str, r: int) -> pd.DataFrame:
    return ens[(ens.model == m) & (ens.r == r)].sort_values("site_id")


def ratio_stats(ens: pd.DataFrame, rs) -> tuple[dict, list[np.ndarray]]:
    """quantity -> model -> list over resamples of (164, 2) [numerator, denominator] per host (sorted by host_id);
    also the host ids per resample."""
    out = {q: {m: [] for m in MODELS} for q in ("mae", "within", "select", "rho")}
    out["rho_pair"] = {}
    hosts = []
    for r in rs:
        rho = {}
        for m in MODELS:
            f = frame_of(ens, m, r)
            hs = host_stats(f.y_true.to_numpy(), f.y_pred.to_numpy(), f.host_id.to_numpy())
            out["mae"][m].append(hs[["sabs", "n"]].to_numpy())
            out["within"][m].append(hs[["wsum", "wn"]].to_numpy())
            t = ss.host_table(f)
            if not (t.host_id.to_numpy() == hs.index.to_numpy()).all():
                raise AssertionError("host order differs between host_stats and host_table")
            out["select"][m].append(ss.stats(t, 2)[:, :2])
            sp = wb.host_spearman(f, "y_pred").reindex(hs.index)
            rho[m] = sp.to_numpy(float)
            e = np.isfinite(rho[m]).astype(float)
            out["rho"][m].append(np.column_stack([np.nan_to_num(rho[m]) * e, e]))
            if m == "S":
                hosts.append(hs.index.to_numpy())
            elif not (hs.index.to_numpy() == hosts[-1]).all():
                raise AssertionError(f"{m}, r = {r}: test hosts differ from S")
        for a, b in (("S", "Dstar"), ("D", "Dstar"), ("S", "D")):
            d = rho[a] - rho[b]
            e = np.isfinite(d).astype(float)
            out["rho_pair"].setdefault(f"{a}_minus_{b}", []).append(np.column_stack([np.nan_to_num(d) * e, e]))
    return out, hosts


def per_resample(st: list[np.ndarray]) -> np.ndarray:
    t = np.array([s.sum(axis=0) for s in st])
    return t[:, 0] / t[:, 1]


# ---------------------------------------------------------------- the three schemes: slot totals

def tot_hier(draws: HierDraws, st: list[np.ndarray]) -> np.ndarray:
    tot = np.zeros((draws.n_boot, len(st), st[0].shape[1]))
    for j, s in enumerate(st):
        m = draws.pick == j
        tot[m] = draws.w[m][:, : len(s)] @ s
    return tot


def tot_host(V: np.ndarray, idx: list[np.ndarray], st: list[np.ndarray]) -> np.ndarray:
    return np.stack([V[:, idx[r]] @ st[r] for r in range(len(st))], axis=1)


def slot_values(tot: np.ndarray) -> np.ndarray:
    if (tot[..., 1] <= 0).any():
        raise AssertionError("a bootstrap slot has an empty denominator")
    return tot[..., 0] / tot[..., 1]


class Scheme:
    """Replicate values (n_boot,) of the resample-mean of a ratio quantity under one bootstrap scheme."""

    def __init__(self, name: str, *, draws: HierDraws | None = None, V=None, idx=None, pick=None):
        self.name, self.draws, self.V, self.idx, self.pick = name, draws, V, idx, pick

    def slots(self, st: list[np.ndarray]) -> np.ndarray:
        if self.name == "hier":
            return slot_values(tot_hier(self.draws, st))
        t = tot_host(self.V, self.idx, st)
        if self.name == "crossed":
            t = t[np.arange(t.shape[0])[:, None], self.pick]
        return slot_values(t)

    def reps(self, st: list[np.ndarray]) -> np.ndarray:
        return self.slots(st).mean(axis=1)


def host_draws(n_hosts: int, n_boot: int, seed: int, n_r: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    V = rng.multinomial(n_hosts, np.full(n_hosts, 1.0 / n_hosts), size=n_boot).astype(float)
    pick = rng.integers(0, n_r, size=(n_boot, n_r))
    return V, pick


# ---------------------------------------------------------------- summaries

def summ(point: float, reps: np.ndarray, bonferroni: bool = False) -> dict:
    out = {"point": float(point), "ci95": [float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))],
           "upper_one_sided_95": float(np.quantile(reps, 0.95)), "boot_sd": float(reps.std(ddof=1)),
           "n_boot": int(reps.size)}
    if bonferroni:
        out["ci_bonferroni_6"] = [float(np.percentile(reps, BONF[0])), float(np.percentile(reps, BONF[1]))]
    return out


def quantities(st: dict, sch: Scheme) -> dict:
    """Points and replicates of every block-2 quantity on one scheme."""
    rep = {(q, m): sch.reps(st[q][m]) for q in ("mae", "within", "select") for m in MODELS}
    per = {(q, m): per_resample(st[q][m]) for q in ("mae", "within", "select") for m in MODELS}

    def gap(q, a, b):
        return float((per[(q, a)] - per[(q, b)]).mean()), rep[(q, a)] - rep[(q, b)]

    out = {"A_star": gap("mae", "S", "Dstar"), "A": gap("mae", "S", "D"), "D_minus_Dstar": gap("mae", "D", "Dstar"),
           "within_host_A_star": gap("within", "S", "Dstar"), "within_host_A": gap("within", "S", "D"),
           "selection_Dstar_minus_S": gap("select", "Dstar", "S"), "selection_D_minus_S": gap("select", "D", "S")}
    ps, px = per[("mae", "S")].mean(), per[("mae", "Dstar")].mean()
    out["relative_reduction"] = (float((ps - px) / ps), (rep[("mae", "S")] - rep[("mae", "Dstar")]) / rep[("mae", "S")])
    for m in MODELS:
        out[f"MAE_{m}"] = (float(per[("mae", m)].mean()), rep[("mae", m)])
    return out


# ---------------------------------------------------------------- loading

def load() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    ref, _ = da.s_reference()
    dstar, info = da.load_dstar(ref)
    prec, f_prec = c.load_table("precision_eval_long")
    base = prec[(prec.backbone == "v2") & (prec.B == B) & (prec.r < N_R) & (prec.seed < N_S)][da.COLS]
    long = pd.concat([base, dstar[dstar.B == B][da.COLS]], ignore_index=True)
    return long, dstar, {"precision_eval_long": f_prec, "dretrain_v2_spec": {"path": "specs/dretrain_v2.jsonl",
                                                                            "sha256": info["spec_sha256"]}}


def ensemble(long: pd.DataFrame, rs, n_s: int) -> pd.DataFrame:
    sub = long[long.r.isin(list(rs)) & (long.seed < n_s)]
    ens = seed_ensemble(sub)
    if (ens.n_seeds != n_s).any() or sorted(ens.r.unique()) != sorted(rs):
        raise AssertionError("seed ensemble over the wrong seeds or resamples")
    return ens


# ---------------------------------------------------------------- blocks

def overlap_block(hosts: list[np.ndarray]) -> dict:
    allh = np.concatenate(hosts)
    u, cnt = np.unique(allh, return_counts=True)
    return {"n_resamples": len(hosts), "test_hosts_per_resample": sorted({len(h) for h in hosts}),
            "host_appearances": int(allh.size), "distinct_test_hosts": int(u.size),
            "appearances_per_host": {"mean": float(cnt.mean()), "min": int(cnt.min()), "max": int(cnt.max()),
                                     "median": float(np.median(cnt)),
                                     "histogram": {str(k): int(v) for k, v in zip(*np.unique(cnt, return_counts=True))}},
            "training_hosts": B, "note": "654 training hosts + 164 test hosts = 818 distinct hosts: the training pool "
                                         "of a resample is the complement of its test hosts"}


def schemes_block(st: dict, hosts: list[np.ndarray], all_hosts: np.ndarray) -> tuple[dict, dict]:
    idx = [np.searchsorted(all_hosts, h) for h in hosts]
    for h, i in zip(hosts, idx, strict=True):
        if not (all_hosts[i] == h).all():
            raise AssertionError("host index mapping")
    draws = HierDraws.make([N_HOSTS] * N_R, N_BOOT, seed=SEED)
    V, pick = host_draws(len(all_hosts), N_BOOT, SEED, N_R)
    schemes = {"hier": Scheme("hier", draws=draws), "host": Scheme("host", V=V, idx=idx),
               "crossed": Scheme("crossed", V=V, idx=idx, pick=pick)}
    raw = {k: quantities(st, s) for k, s in schemes.items()}
    out = {}
    for q in raw["hier"]:
        out[q] = {k: summ(raw[k][q][0], raw[k][q][1], bonferroni=k in ("hier", "host")) for k in schemes}
    # 20000 host-cluster replicates for A* and A
    Vb, _ = host_draws(len(all_hosts), N_BOOT_BIG, SEED, N_R)
    big = Scheme("host", V=Vb, idx=idx)
    rb = {m: big.reps(st["mae"][m]) for m in MODELS}
    for q, a, b2 in (("A_star", "S", "Dstar"), ("A", "S", "D")):
        out[q]["host_20000"] = summ(out[q]["hier"]["point"], rb[a] - rb[b2], bonferroni=True)
    for k in ("hier", "host", "crossed", "host_20000"):
        e = out["A_star"][k]
        e["largest_tolerance_wholly_below_interval_eV"] = e["ci95"][0]
        e["interval_wholly_above_delta"] = bool(e["ci95"][0] > DELTA)
        e["bonferroni_lower_above_delta"] = bool(e["ci_bonferroni_6"][0] > DELTA) if "ci_bonferroni_6" in e else None
    for k in ("hier", "host", "crossed", "host_20000"):
        e = out["A"][k]
        e["upper_below_delta"] = bool(e["upper_one_sided_95"] < DELTA)
        e["two_sided_within_pm_delta"] = bool(e["ci95"][0] > -DELTA and e["ci95"][1] < DELTA)
    width = {k: out["A_star"][k]["ci95"][1] - out["A_star"][k]["ci95"][0] for k in schemes}
    out["A_star_interval_width_ratio_to_hier"] = {k: float(width[k] / width["hier"]) for k in schemes}
    out["boot_sd_ratio_to_hier"] = {q: {k: float(out[q][k]["boot_sd"] / out[q]["hier"]["boot_sd"]) for k in schemes}
                                    for q in raw["hier"]}
    return out, {"draws": draws, "V": V, "idx": idx}


def bonferroni_block(sch: dict) -> dict:
    keys = ("A_star", "A", "within_host_A_star", "selection_Dstar_minus_S", "relative_reduction")
    out = {"level": f"two-sided 1 - 0.05/{N_CLAIMS}", "percentiles": [BONF[0], BONF[1]],
           "family": "six Abstract claims (Bonferroni over six)"}
    for q in keys:
        out[q] = {k: {"point": sch[q][k]["point"], "ci_bonferroni_6": sch[q][k]["ci_bonferroni_6"]}
                  for k in ("hier", "host") if k in sch[q]}
        if q in ("A_star", "A"):
            out[q]["host_20000"] = {"point": sch[q]["host_20000"]["point"],
                                    "ci_bonferroni_6": sch[q]["host_20000"]["ci_bonferroni_6"]}
        if q == "A_star":
            for k in out[q]:
                out[q][k]["lower_above_delta"] = bool(out[q][k]["ci_bonferroni_6"][0] > DELTA)
        else:
            for k in out[q]:
                lo, hi = out[q][k]["ci_bonferroni_6"]
                out[q][k]["excludes_zero"] = bool(lo > 0 or hi < 0)
    return out


def split_quantities(st_mae: dict, rs: list[int], sch: Scheme | None, *, slots: dict | None = None) -> dict:
    per = {m: per_resample([st_mae[m][r] for r in rs]) for m in MODELS}
    pts = {f"MAE_{m}": float(per[m].mean()) for m in MODELS}
    pts["A"] = float((per["S"] - per["D"]).mean())
    pts["A_star"] = float((per["S"] - per["Dstar"]).mean())
    if slots is not None:
        rep = {m: slots[m][:, rs].mean(axis=1) for m in MODELS}
    else:
        rep = {m: sch.reps([st_mae[m][r] for r in rs]) for m in MODELS}
    reps = {f"MAE_{m}": rep[m] for m in MODELS}
    reps["A"], reps["A_star"] = rep["S"] - rep["D"], rep["S"] - rep["Dstar"]
    return {k: (pts[k], reps[k]) for k in pts}


def design_split(long: pd.DataFrame, st5: dict, idx5: list[np.ndarray],
                 V: np.ndarray) -> dict:
    lo, hi = list(range(10)), list(range(10, N_R))
    h10 = HierDraws.make([N_HOSTS] * 10, N_BOOT, seed=SEED)
    h20 = HierDraws.make([N_HOSTS] * 20, N_BOOT, seed=SEED)
    h20b = HierDraws.make([N_HOSTS] * 20, N_BOOT, seed=SEED + 1)
    q_lo = split_quantities(st5["mae"], lo, Scheme("hier", draws=h10))
    q_hi = split_quantities(st5["mae"], hi, Scheme("hier", draws=h20))
    q_hi_ind = split_quantities(st5["mae"], hi, Scheme("hier", draws=h20b))
    ens3 = ensemble(long, lo, 3)
    st3, _ = ratio_stats(ens3, lo)
    q_lc = split_quantities(st3["mae"], lo, Scheme("hier", draws=h10))
    hs = Scheme("host", V=V, idx=idx5)
    slots = {m: hs.slots(st5["mae"][m]) for m in MODELS}
    q_lo_h = split_quantities(st5["mae"], lo, None, slots=slots)
    q_hi_h = split_quantities(st5["mae"], hi, None, slots=slots)
    out = {"r0_9_seeds0_4": {k: summ(*v) for k, v in q_lo.items()},
           "r10_29_seeds0_4": {k: summ(*v) for k, v in q_hi.items()},
           "r0_9_seeds0_2": {k: summ(*v) for k, v in q_lc.items()}}
    diff = {}
    for k in q_lo:
        d_pt = q_hi[k][0] - q_lo[k][0]
        a = summ(d_pt, q_hi_ind[k][1] - q_lo[k][1])
        b = summ(d_pt, q_hi_h[k][1] - q_lo_h[k][1])
        for e in (a, b):
            e["ci95_excludes_zero"] = bool(e["ci95"][0] > 0 or e["ci95"][1] < 0)
        diff[k] = {"independent_hier": a, "host_cluster_paired": b}
    out["difference_r10_29_minus_r0_9_seeds0_4"] = diff
    out["draws"] = {"r0_9": "HierDraws.make([164] * 10, 2000, 20261008)",
                    "r10_29": "HierDraws.make([164] * 20, 2000, 20261008) (slot j = resample 10 + j)",
                    "difference_independent_hier": "replicate (10-29, HierDraws.make([164] * 20, 2000, 20261009)) "
                                                   "minus replicate (0-9, HierDraws.make([164] * 10, 2000, 20261008)); "
                                                   "the two halves treated as independent designs, a separate seed so "
                                                   "the two generator streams are not aligned",
                    "difference_host_cluster_paired": "host weights V of the host-cluster scheme (818 hosts, 2000, "
                                                      "20261008) shared by both halves, replicate = mean over 10-29 of "
                                                      "the slot values minus mean over 0-9 (hosts tested in both halves "
                                                      "move together)"}
    out["note"] = ("r0_9_seeds0_2 is the learning-curve subset; its S and D come from precision_eval_long (the "
                   "learning-curve records use staging_eval_long, compared in reproduction)")
    return out


def spearman_entry(st_q: list[np.ndarray], draws: HierDraws) -> dict:
    per = per_resample(st_q)
    reps = Scheme("hier", draws=draws).reps(st_q)
    e = summ(per.mean(), reps)
    e["n_host_cases"] = int(sum(s[:, 1].sum() for s in st_q))
    e["eligible_hosts_per_resample"] = {"mean": float(np.mean([s[:, 1].sum() for s in st_q])),
                                        "min": int(min(s[:, 1].sum() for s in st_q))}
    return e


def stored_def_spearman(ens: pd.DataFrame, k: int) -> dict:
    """The definition of analysis/within_host_breakdown.py: per site the mean of the seed-ensemble predictions over the
    resamples that tested it; per host (>= 3 sites) Spearman of that mean and the target; mean over hosts; host-cluster
    bootstrap with draw_weights(n_hosts, k) (SeedSequence([20261012, k])), differences paired on hosts."""
    avg = wb.average_over_resamples(ens)
    rho = {m: wb.host_spearman(avg[avg.model == m], "y_pred") for m in MODELS}
    hosts = sorted(rho["S"].index)
    w = wb.draw_weights(len(hosts), k)
    vals = {m: rho[m].reindex(hosts).to_numpy(float) for m in MODELS}
    out = {}

    def one(v):
        ok = np.isfinite(v)
        r = (w[:, ok] @ v[ok]) / w[:, ok].sum(axis=1)
        return {"mean": float(v[ok].mean()), "ci95": wb.interval(r), "n_hosts": int(ok.sum())}

    for m in MODELS:
        out[m] = one(vals[m])
    for a, b in (("S", "Dstar"), ("D", "Dstar"), ("S", "D")):
        out[f"{a}_minus_{b}"] = one(vals[a] - vals[b])
    out["hosts_with_ge3_sites"] = len(hosts)
    return out


def spearman_block(st: dict, dstar: pd.DataFrame) -> tuple[dict, dict]:
    draws = HierDraws.make([N_HOSTS] * N_R, N_BOOT, seed=SEED)
    res = {m: spearman_entry(st["rho"][m], draws) for m in MODELS}
    res.update({k: spearman_entry(v, draws) for k, v in st["rho_pair"].items()})
    # stored 10 x 3 values, recomputed from the v2_sweep runs and D* (r 0-9, seeds 0-2)
    rec = c.record("derived_dstar/dstar_extras")["payload"]["spearman"]
    k = int(rec["bootstrap_position_k"])
    v2, _ = wb.load_v2(ROOT)
    ds = dstar[(dstar.B == B) & (dstar.r < 10) & (dstar.seed < 3)]
    ds = ds[["site_id", "host_id", "y_true", "y_pred", "r", "seed"]].assign(model="Dstar")
    e10 = wb.seed_ensemble(pd.concat([v2[["model", "r", "seed", "site_id", "host_id", "y_true", "y_pred"]], ds],
                                     ignore_index=True))
    if (e10.n_seeds != 3).any():
        raise AssertionError("10 x 3 Spearman: seed ensemble")
    got10 = stored_def_spearman(e10, k)
    stored = {m: {q: rec[m][q] for q in ("mean", "ci95", "n_hosts")} for m in MODELS}
    stored.update({key: {q: rec[key][q] for q in ("mean", "ci95", "n_hosts")}
                   for key in ("S_minus_Dstar", "D_minus_Dstar", "S_minus_D")})
    repro = compare(got10, stored, "results/derived_dstar/dstar_extras.json payload.spearman (S, D, D*, S - D*, "
                                   "D - D*, S - D; mean, ci95, n_hosts)")
    # the same stored definition on the 30 x 5 predictions (like-for-like design comparison)
    return {"design_30x5_hier": res,
            "design_30x5_stored_definition": None,
            "stored_10x3": {**stored, "hosts_with_ge3_sites": rec["hosts_with_ge3_sites"],
                            "source": "results/derived_dstar/dstar_extras.json payload.spearman"},
            "definitions": {
                "design_30x5_hier": "per resample, per test host with >= 3 tested sites, Spearman (average ranks; "
                                    "undefined when constant, excluded) of the 5-seed-ensemble prediction and the DFT "
                                    "target; per-resample value = mean over eligible hosts; point = mean over 30 "
                                    "resamples; HierDraws.make([164] * 30, 2000, 20261008), slot value = weighted mean "
                                    "of host correlations; differences per host (both defined) then the same averaging",
                "stored_10x3": "analysis/within_host_breakdown.py: prediction per site averaged over the resamples "
                               "that tested it (3-seed ensembles, r 0-9), one Spearman per host over all its sites, "
                               "mean over hosts, host-cluster bootstrap SeedSequence([20261012, k = "
                               f"{k}]); recomputed here from results/v2_sweep and D* r 0-9 seeds 0-2"},
            "recomputed_10x3": got10, "k": k}, {"spearman_10x3": repro}


# ---------------------------------------------------------------- reproduction

def reproduction(sch: dict) -> dict:
    g = c.record("derived_dstar/dstar_gaps")["payload"]["headline_654_30x5"]["654"]
    mine = {"MAE": {m: {"mean": sch[f"MAE_{m}"]["hier"]["point"], "ci95": sch[f"MAE_{m}"]["hier"]["ci95"]}
                    for m in MODELS},
            "gaps": {k: {"A": sch[q]["hier"]["point"], "ci95": sch[q]["hier"]["ci95"],
                         "upper_one_sided_95": sch[q]["hier"]["upper_one_sided_95"],
                         "boot_sd": sch[q]["hier"]["boot_sd"]}
                     for k, q in (("S_minus_Dstar", "A_star"), ("S_minus_D", "A"), ("D_minus_Dstar", "D_minus_Dstar"))}}
    ref = {"MAE": {m: g["MAE"][m] for m in MODELS},
           "gaps": {k: {q: g["gaps"][k][q] for q in ("A", "ci95", "upper_one_sided_95", "boot_sd")}
                    for k in ("S_minus_Dstar", "S_minus_D", "D_minus_Dstar")}}
    out = {"dstar_gaps": compare(mine, ref, "results/derived_dstar/dstar_gaps.json payload.headline_654_30x5.654 "
                                            "(MAE of S, D, D*; S - D*, S - D, D - D*: A, ci95, U, boot_sd)")}
    w = c.record("derived_dstar/dstar_extras")["payload"]["a_within"]["headline_654_30x5"]
    h = {"S_minus_Dstar": sch["within_host_A_star"]["hier"], "S_minus_D": sch["within_host_A"]["hier"]}
    out["dstar_extras_a_within"] = compare(
        {k: {"A": v["point"], "ci95": v["ci95"], "upper_one_sided_95": v["upper_one_sided_95"]} for k, v in h.items()},
        {k: {q: w[k][q] for q in ("A", "ci95", "upper_one_sided_95")} for k in h},
        "results/derived_dstar/dstar_extras.json payload.a_within.headline_654_30x5.{S_minus_Dstar, S_minus_D}")
    s = c.record("derived_coord/site_selection")["payload"]["results"]["hosts_ge2_sites"]
    sel = {"Dstar_minus_S": sch["selection_Dstar_minus_S"]["hier"], "D_minus_S": sch["selection_D_minus_S"]["hier"]}
    out["site_selection"] = compare(
        {k: {"mean": v["point"], "ci95": v["ci95"], "upper_one_sided_95": v["upper_one_sided_95"],
             "boot_sd": v["boot_sd"]} for k, v in sel.items()},
        {k: {q: s[k][q] for q in ("mean", "ci95", "upper_one_sided_95", "boot_sd")} for k in sel},
        "results/derived_coord/site_selection.json payload.results.hosts_ge2_sites.{Dstar_minus_S, D_minus_S}")
    d = c.record("derived_dsens/dsens")["payload"]["relative_reduction"]["rows"]["654_30x5"]["Dstar"]
    rr = sch["relative_reduction"]["hier"]
    out["dsens_relative"] = compare({"R": rr["point"], "ci95": rr["ci95"], "boot_sd": rr["boot_sd"]},
                                    {q: d[q] for q in ("R", "ci95", "boot_sd")},
                                    "results/derived_dsens/dsens.json payload.relative_reduction.rows.654_30x5.Dstar")
    return out


def split_reproduction(split: dict) -> dict:
    """r 0-9, seeds 0-2 against the learning-curve record (S, D from staging_eval_long there)."""
    w = c.record("derived_dstar/dstar_gaps")["payload"]["sweep_10x3"]["654"]
    mine = split["r0_9_seeds0_2"]
    got = {"MAE": {m: {"mean": mine[f"MAE_{m}"]["point"], "ci95": mine[f"MAE_{m}"]["ci95"]} for m in MODELS},
           "gaps": {k: {"A": mine[q]["point"], "ci95": mine[q]["ci95"],
                        "upper_one_sided_95": mine[q]["upper_one_sided_95"]}
                    for k, q in (("S_minus_D", "A"), ("S_minus_Dstar", "A_star"))}}
    ref = {"MAE": {m: w["MAE"][m] for m in MODELS},
           "gaps": {k: {q: w["gaps"][k][q] for q in ("A", "ci95", "upper_one_sided_95")}
                    for k in ("S_minus_D", "S_minus_Dstar")}}
    return compare(got, ref, "results/derived_dstar/dstar_gaps.json payload.sweep_10x3.654 (MAE of S, D, D*; S - D, "
                             "S - D*: A, ci95, U); draws identical (analysis/staging_analysis.make_draws)")


# ---------------------------------------------------------------- main

def run() -> dict:
    t0 = time.time()
    long, dstar, inputs = load()
    ens = ensemble(long, range(N_R), N_S)
    st, hosts = ratio_stats(ens, range(N_R))
    all_hosts = np.unique(np.concatenate(hosts))
    ov = overlap_block(hosts)
    sch, aux = schemes_block(st, hosts, all_hosts)
    bon = bonferroni_block(sch)
    split = design_split(long, st, aux["idx"], aux["V"])
    sp, sp_repro = spearman_block(st, dstar)
    sp["design_30x5_stored_definition"] = stored_def_spearman(ens, sp.pop("k"))
    repro = {**reproduction(sch), **sp_repro, "design_split_r0_9_seeds0_2": split_reproduction(split)}
    payload = {
        "label": "intervals of the descriptor advantage with the 818 distinct test hosts as the sampling unit "
                 "(host-cluster and crossed bootstraps) next to the paper's HierDraws; Bonferroni intervals over six "
                 "claims; resample halves of the 654-host design; within-host Spearman on 30 x 5",
        "design": "654 training hosts, outer resamples r = 0-29 x seeds 0-4 (5-seed ensembles); S and D from "
                  "results/analysis/precision_eval_long.parquet (v2), D* from results/dretrain_v2",
        "delta_eV": DELTA, "sign_convention": SIGNS,
        "estimator": {
            "common": "seed-ensemble mean per (model, r, site); each quantity is a ratio of per-host sums; slot value "
                      "= ratio of the weighted sums of one resample; replicate = mean over 30 slots; point = mean over "
                      "resamples of the per-resample value (R: ratio of point MAEs); paired differences per replicate; "
                      "ci95 = 2.5/97.5 percentiles, U (upper_one_sided_95) = 95th percentile, boot_sd ddof 1",
            "hier": "HierDraws.make([164] * 30, 2000, 20261008): per slot a resample with replacement, then its 164 "
                    "test hosts with replacement, independently per slot (a host tested in six resamples is redrawn "
                    "six times independently)",
            "host": "host-cluster: V = default_rng(20261008).multinomial(818, uniform, size=n_boot) over the 818 "
                    "distinct test hosts sorted by host_id; all 30 resamples kept, every site of host h in resample r "
                    "weighted V[b, h], so a drawn host carries all its appearances; 2000 replicates (20000 for A* and "
                    "A, same generator seed)",
            "crossed": "rng = default_rng(20261008); V = rng.multinomial(818, uniform, size=2000) (identical to the "
                       "host scheme's V), then pick = rng.integers(0, 30, size=(2000, 30)); slot k = resample "
                       "pick[b, k] with host weights V[b, .]; a site enters slot k with multiplicity V[b, host] when "
                       "its resample is drawn into the slot and its host is drawn, and is absent otherwise",
            "bonferroni": "two-sided 1 - 0.05/6: percentiles 0.4167 and 99.5833 of the same replicates",
        },
        "overlap": ov,
        "schemes": sch,
        "bonferroni": bon,
        "design_split": split,
        "spearman_30x5": sp,
        "reproduction": repro,
        "reproduction_summary": {"max_abs_difference_hier_records": max(
            repro[k]["max_abs_difference"] for k in ("dstar_gaps", "dstar_extras_a_within", "site_selection",
                                                     "dsens_relative")),
            "all_hier_records_below_1e-9": all(repro[k]["reproduced"] for k in (
                "dstar_gaps", "dstar_extras_a_within", "site_selection", "dsens_relative")),
            "spearman_10x3_reproduced": repro["spearman_10x3"]["reproduced"],
            "design_split_r0_9_seeds0_2_reproduced": repro["design_split_r0_9_seeds0_2"]["reproduced"]},
        "inputs": inputs,
    }
    bad = [k for k in ("dstar_gaps", "dstar_extras_a_within", "site_selection", "dsens_relative")
           if not repro[k]["reproduced"]]
    if bad:
        raise AssertionError(f"HierDraws records not reproduced: {bad} {repro}")
    print(f"runtime_s {time.time() - t0:.1f}", file=sys.stderr)
    return c.to_jsonable(payload)


def main() -> None:
    from dftgnn.config import load_config
    from dftgnn.io.results import write_result

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    cfg = load_config()
    if float(cfg.stats.delta_eV) != DELTA:
        raise AssertionError("stats.delta_eV differs from 0.05")
    payload = run()
    if a.dry_run:
        text = json.dumps(payload, indent=1, allow_nan=False)
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(text)
            print(a.out)
        else:
            print(text)
        return
    print(write_result("boot_design", payload, config=cfg, repo_root=ROOT, results_dir=OUT_DIR))


if __name__ == "__main__":
    main()
