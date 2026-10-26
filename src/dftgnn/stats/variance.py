"""One-way random-effects variance decomposition of a target over groups (hosts or formulas).

Estimator: ANOVA method of moments for unbalanced groups (Searle, Casella and McCulloch,
Variance Components, Sec. 3.6):

    MSB = sum_i n_i (ybar_i - ybar)^2 / (k - 1)
    MSW = sum_i sum_j (y_ij - ybar_i)^2 / (N - k)
    n0  = (N - sum_i n_i^2 / N) / (k - 1)
    sigma2_between = (MSB - MSW) / n0,  sigma2_within = MSW
    ICC = sigma2_between / (sigma2_between + sigma2_within)

Singleton groups enter the between-group sum and contribute no within-group degrees of
freedom. Estimates are not truncated at zero. Confidence intervals: percentile cluster
bootstrap, resampling whole groups with replacement; a group drawn twice counts as two groups.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _components(n: np.ndarray, s: np.ndarray, ss: np.ndarray) -> dict[str, float]:
    """Variance components from per-group count, sum and sum of squares."""
    k = n.size
    N = n.sum()
    ybar = s.sum() / N
    ssb = float((s**2 / n).sum() - N * ybar**2)
    ssw = float((ss - s**2 / n).sum())
    msb = ssb / (k - 1)
    msw = ssw / (N - k)
    n0 = (N - (n**2).sum() / N) / (k - 1)
    sb2 = (msb - msw) / n0
    return {"sigma2_between": sb2, "sigma2_within": msw, "icc": sb2 / (sb2 + msw),
            "msb": msb, "msw": msw, "n0": float(n0)}


def _group_stats(y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.DataFrame({"g": groups, "y": y, "y2": y**2}).groupby("g", sort=True)
    return (df.size().to_numpy(float), df.y.sum().to_numpy(float), df.y2.sum().to_numpy(float))


def oneway_icc(y, groups, *, n_boot: int = 2000, ci: float = 0.95, seed: int = 0) -> dict:
    """Point estimates and cluster-bootstrap percentile CIs of the variance components."""
    y = np.asarray(y, dtype=float)
    n, s, ss = _group_stats(y, np.asarray(groups))
    point = _components(n, s, ss)
    rng = np.random.default_rng(seed)
    k = n.size
    keys = ("sigma2_between", "sigma2_within", "icc")
    boot = {key: np.empty(n_boot) for key in keys}
    for b in range(n_boot):
        i = rng.integers(0, k, size=k)
        c = _components(n[i], s[i], ss[i])
        for key in keys:
            boot[key][b] = c[key]
    lo, hi = (1 - ci) / 2, 1 - (1 - ci) / 2
    out = {
        "estimator": "ANOVA method of moments, unbalanced one-way random effects",
        "n_obs": int(n.sum()), "n_groups": int(k),
        "n_groups_with_ge2": int((n >= 2).sum()),
        "df_between": int(k - 1), "df_within": int(n.sum() - k),
        **{key: float(v) for key, v in point.items()},
        "total_variance_sample": float(y.var(ddof=1)),
        "bootstrap": {"n_resamples": n_boot, "seed": seed, "unit": "group (cluster)",
                      "ci_level": ci, "method": "percentile"},
    }
    for key in keys:
        out[f"{key}_ci"] = [float(np.quantile(boot[key], lo)), float(np.quantile(boot[key], hi))]
        out[f"{key}_boot_sd"] = float(boot[key].std(ddof=1))
    return out


def within_group_spread(y, groups, q=(0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)) -> dict:
    """Per-group range and sd (ddof=1) over groups with at least two observations."""
    df = pd.DataFrame({"g": groups, "y": np.asarray(y, dtype=float)})
    g = df.groupby("g").y
    stats = pd.DataFrame({"n": g.size(), "range": g.max() - g.min(), "sd": g.std(ddof=1)})
    stats = stats[stats.n >= 2]
    qs = {f"q{round(x * 100):02d}": x for x in q}
    return {
        "n_groups_with_ge2": len(stats), "n_obs_in_those_groups": int(stats.n.sum()),
        "sites_per_group_counts": {int(k): int(v) for k, v in
                                   stats.n.value_counts().sort_index().items()},
        "range_eV": {**{k: float(stats.range.quantile(v)) for k, v in qs.items()},
                     "mean": float(stats.range.mean())},
        "sd_eV": {**{k: float(stats.sd.quantile(v)) for k, v in qs.items()},
                  "mean": float(stats.sd.mean())},
    }


def within_host_residual_mae(y_true, y_pred, groups) -> float:
    """MAE of host-centred residuals over hosts with at least two sites.

    For site j of host i: r_ij = (y_ij - mean_i y) - (yhat_ij - mean_i yhat). Returns
    mean |r_ij| over all sites of hosts with n_i >= 2. Removes each host's mean offset so
    only the ranking and spacing of sites within a host is scored.
    """
    df = pd.DataFrame({"g": groups, "y": np.asarray(y_true, float), "p": np.asarray(y_pred, float)})
    n = df.groupby("g").y.transform("size")
    df = df[n >= 2]
    r = (df.y - df.groupby("g").y.transform("mean")) - (df.p - df.groupby("g").p.transform("mean"))
    return float(r.abs().mean())
