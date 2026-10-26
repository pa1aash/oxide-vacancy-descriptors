"""Descriptor gain by the cation coordination of the removed oxygen (coordination spec (a) gains).

Nothing is trained. Physically, the number of M-O bonds broken when the neutral O leaves (its cation coordination
number CN in the pristine host) sets the local part of the formation energy; the pristine-cell electronic
descriptors carry the part set by the host's electronic structure (where the two left-behind electrons go). This
module asks whether the gain of the descriptor networks over the structure-only network S depends on CN. The bins
come from the committed record results/derived_coord/coord_bins.json (sha256 asserted; analysis/coord/bins.py);
bins and estimator were fixed before any gain was computed.

Estimator (654 hosts, 30 resamples x 5 seeds; S, D from precision_eval_long v2, D* from dretrain_v2)
    ensemble    seed-ensemble mean per (model, r, site)
    site terms  |p - y| (site MAE); within-host residual |(p - mean_h p) - (y - mean_h y)| with host means over ALL
                tested sites of the host, only for hosts with >= 2 tested sites (else no within-host term); this
                is dftgnn.stats.metrics.host_stats site by site
    bin         selects sites; per host and bin: sums of the terms of the host's sites in the bin and their counts
    pooled      the class-restricted estimator: replicate = (sum over slots of the weighted bin numerator) / (sum
                over slots of the weighted bin denominator) under the headline draws HierDraws.make([164] * 30,
                2000, 20261008) (hosts sorted by host_id); point = the same ratio over the 30 resamples with unit
                weights; 95% = 2.5 / 97.5 percentiles, one-sided upper = 95th percentile
    gains       S - D* and S - D per replicate (paired); positive = the descriptor network more accurate
    pattern     adjacent-bin differences of the gain (3 - 2, 4 - 3, 5+ - 4) per replicate; linear trend = least-
                squares slope of the gain over bin index 0..3 (bins 2, 3, 4, 5+) per replicate (eV per bin)
    reference   the pooled estimator on all sites next to the headline mean over resamples (the latter
                reproduces results/derived_dstar/dstar_extras.json S_minus_Dstar, the +0.031 eV)
The unassigned bin (CN < 2, rule fixed in advance) is reported, labelled, and kept out of the pattern checks.

    PYTHONPATH=$PWD/src python analysis/coord/coord_gain.py [--dry-run] [--out <path>]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis import a1_common as c
from analysis.coord import site_selection as ss

OUT_DIR = ROOT / "results" / "derived_coord"
DRY_OUT = Path("coord_gain_dryrun.json")
BINS_RECORD = OUT_DIR / "coord_bins.json"
BINS_SHA256 = "6ffd869fc905dadd92347cffed52e696daff9713e9eb99e5cd5e8fc1f7cc33ef"
ORDERED = ("2", "3", "4", "5+")
GROUPS = (*ORDERED, "unassigned", "all")
METRICS = ("within_host_mae", "mae")             # primary first
MODELS = ("S", "D", "Dstar")
GAPS = (("S_minus_Dstar", "S", "Dstar"), ("S_minus_D", "S", "D"))
N_R, N_SEEDS, N_HOSTS, HEADLINE_B = 30, 5, 164, 654
TOL = 1e-9
LABEL = ("descriptor gain by cation coordination of the vacancy oxygen (bins 2, 3, 4, 5+, unassigned = CN < 2); "
         "S, D, D* at 654 hosts, 30 x 5, site-level pooled class-restricted estimator, headline draws")


def site_terms(frame: pd.DataFrame) -> pd.DataFrame:
    """Per site: ae = |p - y|; wr = within-host residual (host means over all tested sites), 0 if n_host < 2;
    wn = 1 if the host has >= 2 tested sites else 0."""
    d = frame[["site_id", "host_id", "y_true", "y_pred"]].copy()
    g = d.groupby("host_id")
    n = g.y_true.transform("size")
    d["ae"] = (d.y_pred - d.y_true).abs()
    wr = ((d.y_true - g.y_true.transform("mean")) - (d.y_pred - g.y_pred.transform("mean"))).abs()
    d["wn"] = (n >= 2).astype(float)
    d["wr"] = wr.where(n >= 2, 0.0)
    return d


def host_group_stats(terms: pd.DataFrame, site_bin: pd.Series, hosts: list, groups=GROUPS) -> np.ndarray:
    """(n_hosts, n_groups, 4) per host and group: [sum wr, sum wn, sum ae, n sites]; 'all' takes every site."""
    b = terms.site_id.map(site_bin)
    if b.isna().any():
        raise AssertionError("site without a bin")
    out = np.zeros((len(hosts), len(groups), 4))
    pos = {h: i for i, h in enumerate(hosts)}
    hi = terms.host_id.map(pos).to_numpy()
    vals = terms[["wr", "wn", "ae"]].to_numpy(float)
    for k, grp in enumerate(groups):
        m = np.ones(len(terms), bool) if grp == "all" else (b == grp).to_numpy()
        for col in range(3):
            np.add.at(out[:, k, col], hi[m], vals[m, col])
        np.add.at(out[:, k, 3], hi[m], 1.0)
    return out


def pooled(st: list[np.ndarray], draws) -> tuple[np.ndarray, np.ndarray]:
    """Pooled ratios: point (n_groups, 2) over resamples with unit weights, replicates (n_boot, n_groups, 2);
    the last axis is [within_host_mae, mae]."""
    shape = st[0].shape[1:]
    slots = np.zeros((draws.n_boot, len(st), int(np.prod(shape))))
    for j, s in enumerate(st):
        m = draws.pick == j
        slots[m] = draws.w[m][:, : len(s)] @ s.reshape(len(s), -1)
    tot = slots.sum(axis=1).reshape((draws.n_boot,) + shape)
    pt = np.sum(st, axis=0).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rep = np.stack([tot[..., 0] / tot[..., 1], tot[..., 2] / tot[..., 3]], axis=-1)
        point = np.stack([pt[..., 0] / pt[..., 1], pt[..., 2] / pt[..., 3]], axis=-1)
    if not (np.isfinite(rep).all() and np.isfinite(point).all()):
        raise AssertionError("empty bin in a replicate")
    return point, rep


def per_resample_ratio(s: np.ndarray) -> np.ndarray:
    """(n_groups, 2) ratios of one resample (unit weights)."""
    t = s.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.stack([t[:, 0] / t[:, 1], t[:, 2] / t[:, 3]], axis=-1)


def slope(g: np.ndarray) -> np.ndarray:
    """Least-squares slope over bin index 0..K-1 along the last axis."""
    x = np.arange(g.shape[-1], dtype=float)
    x -= x.mean()
    return (g * x).sum(axis=-1) / (x**2).sum()


def entry(point: float, reps: np.ndarray, per=None) -> dict:
    lo, hi = (1 - c.CI) / 2, 1 - (1 - c.CI) / 2
    e = {"point": float(point), "ci95": [float(np.quantile(reps, lo)), float(np.quantile(reps, hi))],
         "upper_one_sided_95": float(np.quantile(reps, c.UPPER_Q)), "boot_sd": float(reps.std(ddof=1))}
    e["ci95_excludes_zero"] = bool(e["ci95"][0] > 0 or e["ci95"][1] < 0)
    if per is not None:
        e["per_resample"] = [float(v) for v in per]
    return e


def bins_record() -> tuple[pd.Series, dict]:
    got = c.sha256(BINS_RECORD)
    if got != BINS_SHA256:
        raise AssertionError(f"coord_bins.json sha256 {got} differs from the committed record")
    rec = json.loads(BINS_RECORD.read_text())
    s = pd.DataFrame(rec["payload"]["sites"]).set_index("site_id").bin
    return s, {"path": "results/derived_coord/coord_bins.json", "sha256": got, "record_git_sha": rec["git_sha"]}


def run() -> dict:
    from dftgnn.stats.g1 import HierDraws, point_metric, resample_stats, seed_ensemble, summarize

    site_bin, f_bins = bins_record()
    long, inputs = ss.headline_frame()
    ens = seed_ensemble(long)
    if (ens.n_seeds != N_SEEDS).any() or sorted(ens.r.unique()) != list(range(N_R)):
        raise AssertionError("654-host ensemble: wrong seeds or resamples")
    draws = HierDraws.make([N_HOSTS] * N_R, c.N_BOOT, seed=c.BOOT_SEED)
    repro = ss.reproduce(ens, draws)
    if not repro["reproduced"]:
        raise AssertionError(f"within-host gap not reproduced: {repro}")
    st, worst, hosts_r, counts = {m: [] for m in MODELS}, 0.0, [], []
    for r in range(N_R):
        ref = ens[(ens.model == "S") & (ens.r == r)]
        hosts = sorted(ref.host_id.unique())
        if len(hosts) != N_HOSTS:
            raise AssertionError(f"r = {r}: {len(hosts)} test hosts")
        hosts_r.append(hosts)
        for m in MODELS:
            f = ens[(ens.model == m) & (ens.r == r)].sort_values("site_id")
            if not (f.site_id.to_numpy() == ref.sort_values("site_id").site_id.to_numpy()).all():
                raise AssertionError(f"{m}, r = {r}: test sites differ from S")
            t = site_terms(f)
            s = host_group_stats(t, site_bin, hosts)
            st[m].append(s)
            # the 'all' group must equal the headline host statistics
            h0 = resample_stats(f)
            got = per_resample_ratio(s)[GROUPS.index("all")]
            worst = max(worst, abs(got[0] - point_metric(h0, "within_host_mae")), abs(got[1] - point_metric(h0, "mae")))
        s = st["S"][-1]
        counts.append({g: {"sites": int(s[:, k, 3].sum()), "hosts": int((s[:, k, 3] > 0).sum()),
                           "within_sites": int(s[:, k, 1].sum()), "within_hosts": int((s[:, k, 1] > 0).sum())}
                       for k, g in enumerate(GROUPS)})
    if worst > TOL:
        raise AssertionError(f"site terms differ from host_stats by {worst}")
    res = {m: pooled(st[m], draws) for m in MODELS}
    per = {m: np.array([per_resample_ratio(s) for s in st[m]]) for m in MODELS}   # (R, groups, 2)
    out = {}
    for k, g in enumerate(GROUPS):
        blk = {}
        for mi, met in enumerate(METRICS):
            e = {m: entry(res[m][0][k, mi], res[m][1][:, k, mi]) for m in MODELS}
            for name, a, b in GAPS:
                e[name] = entry(res[a][0][k, mi] - res[b][0][k, mi], res[a][1][:, k, mi] - res[b][1][:, k, mi],
                                per[a][:, k, mi] - per[b][:, k, mi])
            blk[met] = e
        cs = [x[g] for x in counts]
        blk["counts"] = {
            "distinct_sites_tested": int(site_bin.index.isin(ens.site_id).sum()) if g == "all" else
            int(((site_bin == g) & site_bin.index.isin(ens.site_id)).sum()),
            **{f"per_resample_{q}": {"mean": float(np.mean([x[q] for x in cs])), "min": int(min(x[q] for x in cs)),
                                     "max": int(max(x[q] for x in cs)), "total": int(sum(x[q] for x in cs))}
               for q in ("sites", "hosts", "within_sites", "within_hosts")}}
        if g == "unassigned":
            blk["note"] = "CN < 2 (rule fixed in advance); reported for completeness, not in the pattern checks"
        out[g] = blk
    idx = [GROUPS.index(g) for g in ORDERED]
    pattern = {}
    for name, a, b in GAPS:
        pattern[name] = {}
        for mi, met in enumerate(METRICS):
            gp = res[a][0][idx, mi] - res[b][0][idx, mi]
            gr = res[a][1][:, idx, mi] - res[b][1][:, idx, mi]
            pattern[name][met] = {
                "adjacent": {f"{ORDERED[i + 1]}_minus_{ORDERED[i]}": entry(gp[i + 1] - gp[i], gr[:, i + 1] - gr[:, i])
                             for i in range(len(ORDERED) - 1)},
                "slope_per_bin": entry(slope(gp), slope(gr)),
                "extreme_5plus_minus_2": entry(gp[-1] - gp[0], gr[:, -1] - gr[:, 0])}
    # reference: pooled on all sites against the headline mean over resamples
    k = GROUPS.index("all")
    hs = {m: [resample_stats(ens[(ens.model == m) & (ens.r == r)].sort_values("site_id")) for r in range(N_R)]
          for m in MODELS}
    head = {}
    for name, a, b in GAPS:
        head[name] = {}
        for mi, met in enumerate(METRICS):
            d = per[a][:, k, mi] - per[b][:, k, mi]
            sm = summarize(d, draws.replicates(hs[a], met) - draws.replicates(hs[b], met), ci=c.CI,
                           upper_quantile=c.UPPER_Q)
            head[name][met] = {"headline_mean_over_resamples": {"point": sm["mean"], "ci95": sm["ci"],
                                                                "upper_one_sided_95": sm["upper_one_sided"]},
                               "pooled_all_sites": {k2: out["all"][met][name][k2]
                                                    for k2 in ("point", "ci95", "upper_one_sided_95")}}
    return {"label": LABEL,
            "definition": {"bins": "results/derived_coord/coord_bins.json payload.sites[*].bin (CN of CrystalNN "
                                   "defaults; 2, 3, 4, 5+; unassigned = CN < 2)",
                           "within_host_mae": "site-weighted mean of |(p - mean_h p) - (y - mean_h y)| over the bin's "
                                              "sites in hosts with >= 2 tested sites; host means over all tested "
                                              "sites of the host (primary)",
                           "mae": "site-weighted mean |p - y| over the bin's sites",
                           "estimator": "pooled over slots: sum(w num) / sum(w den), headline draws "
                                        "HierDraws.make([164] * 30, 2000, 20261008); point = pooled over 30 "
                                        "resamples, unit weights",
                           "sign": "S_minus_Dstar, S_minus_D: positive = descriptor network more accurate",
                           "pattern": "adjacent = gain(bin i+1) - gain(bin i); slope = least squares over bin "
                                      "index 0..3 (2, 3, 4, 5+), eV per bin; both per replicate"},
            "design": "654 hosts, outer resamples r = 0-29 x seeds 0-4",
            "groups": out, "pattern": pattern, "reference_all_sites": head,
            "checks": {"site_terms_vs_host_stats_max_abs": float(worst), "within_gap_reproduction": repro},
            "inputs": {**inputs, "coord_bins": f_bins}}


def main() -> None:
    from dftgnn.config import load_config
    from dftgnn.io.results import write_result

    cfg = load_config()
    payload = c.to_jsonable(run())
    if "--dry-run" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else DRY_OUT
        out.write_text(json.dumps(payload, indent=1, allow_nan=False))
        print(out)
        return
    print(write_result("coord_gain", payload, config=cfg, repo_root=ROOT, results_dir=OUT_DIR))


if __name__ == "__main__":
    main()
