"""Held-out metrics with cluster-bootstrap confidence intervals over test hosts.

Metrics (all in the target's units, eV, except R^2)
    mae, rmse, r2            site level
    within_host_mae          ``variance.within_host_residual_mae``: MAE of host-centred residuals
                             over hosts with at least two sites
    host_mean_mae            mean over hosts of |mean_j y_hj - mean_j yhat_hj|, each host weighted once

Every metric is a smooth function of per-host sufficient statistics (``host_stats``), so a bootstrap draw is a
weighted sum of host rows. The point values come from direct formulas and the tests assert that the two routes
agree.

Cluster bootstrap (``cluster_bootstrap_ci``). The cluster is the test host: hosts are drawn with
replacement, all sites of a drawn host are kept, and a host drawn twice counts twice. Percentile
interval over ``n_boot`` draws (default ``config.stats.cluster_bootstrap_n`` = 2000). Predictions
are held fixed; the interval quantifies test-set sampling variability only.

Aggregation across outer resamples (``aggregate_resamples``). Point estimate: mean over the R
resamples of the per-resample metric, with the sample sd (ddof=1) across resamples. Interval:
two-stage (hierarchical) bootstrap. In each of ``n_boot`` draws (1) R resamples are drawn with
replacement from the R available, and (2) within each drawn resample the test hosts are redrawn with
replacement, independently for each pick; the draw's value is the mean of the R bootstrapped
metrics. The 95% interval is the 2.5th-97.5th percentile of the ``n_boot`` values. This resamples
both outer-split variability (which hosts are held out and which hosts train) and within-test host
sampling, but not model retraining noise beyond what differs between resamples. Resamples overlap
in test hosts, so the interval treats them as exchangeable replicates, not as independent data.
Paired differences between two models use identical draws (``aggregate_delta``).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from dftgnn.stats.variance import within_host_residual_mae

METRICS = ("mae", "rmse", "r2", "within_host_mae", "host_mean_mae")
_COLS = ("n", "sabs", "sse", "sy", "syy", "h", "dsum", "wsum", "wn")


def point_metrics(y_true, y_pred, groups) -> dict[str, float]:
    """All five metrics from direct formulas."""
    y = np.asarray(y_true, float)
    p = np.asarray(y_pred, float)
    e = p - y
    sst = float(((y - y.mean()) ** 2).sum())
    df = pd.DataFrame({"g": groups, "y": y, "p": p}).groupby("g")
    hm = (df.y.mean() - df.p.mean()).abs()
    return {
        "mae": float(np.abs(e).mean()),
        "rmse": float(np.sqrt((e**2).mean())),
        "r2": float(1.0 - (e**2).sum() / sst),
        "within_host_mae": within_host_residual_mae(y, p, groups),
        "host_mean_mae": float(hm.mean()),
    }


def host_stats(y_true, y_pred, groups) -> pd.DataFrame:
    """Per-host sufficient statistics (rows sorted by host)."""
    d = pd.DataFrame({"g": groups, "y": np.asarray(y_true, float), "p": np.asarray(y_pred, float)})
    d["e"] = d.p - d.y
    d["ae"] = d.e.abs()
    d["e2"] = d.e**2
    d["y2"] = d.y**2
    d["n"] = d.groupby("g").y.transform("size")
    d["r"] = ((d.y - d.groupby("g").y.transform("mean"))
              - (d.p - d.groupby("g").p.transform("mean"))).abs()
    d["r"] = d.r.where(d.n >= 2, 0.0)
    g = d.groupby("g", sort=True)
    out = pd.DataFrame({
        "n": g.size().astype(float), "sabs": g.ae.sum(), "sse": g.e2.sum(),
        "sy": g.y.sum(), "syy": g.y2.sum(), "h": 1.0,
        "dsum": (g.y.mean() - g.p.mean()).abs(), "wsum": g.r.sum(),
        "wn": g.size().where(g.size() >= 2, 0).astype(float),
    })
    return out[list(_COLS)]


def _metrics_from_totals(t: np.ndarray) -> dict[str, np.ndarray]:
    """``t``: (..., 9) weighted column sums of host_stats."""
    n, sabs, sse, sy, syy, h, dsum, wsum, wn = (t[..., i] for i in range(9))
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "mae": sabs / n,
            "rmse": np.sqrt(sse / n),
            "r2": 1.0 - sse / (syy - sy**2 / n),
            "within_host_mae": wsum / wn,
            "host_mean_mae": dsum / h,
        }


def _draw_weights(rng: np.random.Generator, n_hosts: int, size: int) -> np.ndarray:
    """(size, n_hosts) host multiplicities of ``size`` independent with-replacement draws."""
    return rng.multinomial(n_hosts, np.full(n_hosts, 1.0 / n_hosts), size=size).astype(float)


def _interval(x: np.ndarray, ci: float) -> list[float]:
    lo, hi = (1 - ci) / 2, 1 - (1 - ci) / 2
    return [float(np.quantile(x, lo)), float(np.quantile(x, hi))]


def cluster_bootstrap_ci(y_true, y_pred, groups, *, n_boot: int = 2000, ci: float = 0.95,
                         seed: int = 0, metrics: Sequence[str] = METRICS) -> dict[str, dict]:
    """Point value, percentile CI and bootstrap sd of each metric over test hosts."""
    st = host_stats(y_true, y_pred, groups)
    pt = point_metrics(y_true, y_pred, groups)
    w = _draw_weights(np.random.default_rng(seed), len(st), n_boot)
    boot = _metrics_from_totals(w @ st.to_numpy())
    return {m: {"point": pt[m], "ci": _interval(boot[m], ci), "boot_sd": float(boot[m].std(ddof=1)),
                "n_hosts": len(st), "n_boot": n_boot, "ci_level": ci} for m in metrics}


def _hier_values(stats: Sequence[pd.DataFrame] | Sequence[np.ndarray], *, other=None, n_boot: int,
                 seed: int) -> dict[str, np.ndarray]:
    """Hierarchical-bootstrap draws of the resample-mean metric (or of paired differences)."""
    rng = np.random.default_rng(seed)
    a = [np.asarray(s, float) for s in stats]
    b = [np.asarray(s, float) for s in other] if other is not None else None
    R = len(a)
    pick = rng.integers(0, R, size=(n_boot, R))
    vals = {m: np.zeros((n_boot, R)) for m in METRICS}
    for k in range(R):
        for j in range(R):
            rows = np.nonzero(pick[:, k] == j)[0]
            if rows.size == 0:
                continue
            w = _draw_weights(rng, a[j].shape[0], rows.size)
            ma = _metrics_from_totals(w @ a[j])
            if b is not None:
                mb = _metrics_from_totals(w @ b[j])
                for m in METRICS:
                    vals[m][rows, k] = ma[m] - mb[m]
            else:
                for m in METRICS:
                    vals[m][rows, k] = ma[m]
    return {m: v.mean(axis=1) for m, v in vals.items()}


def _per_resample(stats: Sequence[pd.DataFrame]) -> dict[str, np.ndarray]:
    per = [_metrics_from_totals(np.asarray(s, float).sum(axis=0)) for s in stats]
    return {m: np.array([p[m] for p in per]) for m in METRICS}


def aggregate_resamples(stats: Sequence[pd.DataFrame], *, n_boot: int = 2000, ci: float = 0.95,
                        seed: int = 0) -> dict[str, dict]:
    """Mean, sd and hierarchical-bootstrap interval across outer resamples (see module docstring)."""
    per = _per_resample(stats)
    boot = _hier_values(stats, n_boot=n_boot, seed=seed)
    return {m: {"mean": float(per[m].mean()), "sd": float(per[m].std(ddof=1)) if len(stats) > 1 else 0.0,
                "ci": _interval(boot[m], ci), "boot_sd": float(boot[m].std(ddof=1)),
                "per_resample": [float(x) for x in per[m]], "n_resamples": len(stats),
                "n_boot": n_boot, "ci_level": ci} for m in METRICS}


def aggregate_delta(stats_a: Sequence[pd.DataFrame], stats_b: Sequence[pd.DataFrame], *,
                    n_boot: int = 2000, ci: float = 0.95, seed: int = 0) -> dict[str, dict]:
    """Paired difference A minus B (same hosts per resample, identical bootstrap draws)."""
    if [len(s) for s in stats_a] != [len(s) for s in stats_b]:
        raise ValueError("paired runs must share the host rows of every resample")
    pa, pb = _per_resample(stats_a), _per_resample(stats_b)
    boot = _hier_values(stats_a, other=stats_b, n_boot=n_boot, seed=seed)
    out = {}
    for m in METRICS:
        d = pa[m] - pb[m]
        out[m] = {"mean": float(d.mean()), "sd": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
                  "ci": _interval(boot[m], ci), "boot_sd": float(boot[m].std(ddof=1)),
                  "per_resample": [float(x) for x in d], "n_boot": n_boot, "ci_level": ci}
    return out
