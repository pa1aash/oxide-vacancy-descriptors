"""G1 primary analysis as clarified in (2026-10-08).

Quantities
    seed ensemble     per (model, r, B, site): mean of the seed predictions
    A_rB              MAE_S(r, B) - MAE_D(r, B) on the same test sites; positive = descriptors help
    secondary         within-host residual MAE (hosts with >= 2 sites), host-mean MAE

Hierarchical bootstrap (``HierDraws``). ``n_boot`` replicates; in each, R slots are filled by drawing a
resample index with replacement and then that resample's test hosts with replacement (multiplicities from
a multinomial, every site of a drawn host kept). Within a replicate the RNG is consumed slot by slot:
``integers`` for the resample, then ``multinomial`` for its hosts. A replicate's value is the mean over the
R slots of the metric recomputed on the slot's host draw. One set of draws is applied to every model,
budget and metric, so all differences are paired. One-sided upper bound: ``upper_quantile`` percentile;
two-sided interval: (1 - ci) / 2 and 1 - (1 - ci) / 2 percentiles (numpy linear interpolation).

N* (``n_star``): the smallest tested budget whose upper bound is below delta and stays below delta at every
larger tested budget; ``">654"`` style lower bound if none qualifies.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from dftgnn.stats.metrics import _metrics_from_totals, host_stats

BOOT_SEED = 20261008
METRICS = ("mae", "within_host_mae", "host_mean_mae")


def seed_ensemble(df: pd.DataFrame, keys: Sequence[str] = ("model", "r", "B", "site_id")) -> pd.DataFrame:
    """Mean prediction over seeds per ``keys``; ``df`` has columns keys + host_id, seed, y_true, y_pred."""
    g = df.groupby(list(keys), sort=True)
    if (g.y_true.nunique() != 1).any() or (g.host_id.nunique() != 1).any():
        raise ValueError("a site carries different targets or hosts across seeds")
    out = g.agg(host_id=("host_id", "first"), y_true=("y_true", "first"), y_pred=("y_pred", "mean"),
                n_seeds=("seed", "nunique")).reset_index()
    return out


def resample_stats(frame: pd.DataFrame) -> np.ndarray:
    """Per-host sufficient statistics (rows sorted by host_id) of one (model, r, B) prediction set."""
    return host_stats(frame.y_true.to_numpy(), frame.y_pred.to_numpy(), frame.host_id.to_numpy()).to_numpy()


def point_metric(stats: np.ndarray, metric: str) -> float:
    return float(_metrics_from_totals(np.asarray(stats, float).sum(axis=0))[metric])


@dataclass(frozen=True)
class HierDraws:
    """Shared hierarchical-bootstrap draws: ``pick[b, k]`` resample of slot k, ``w[b, k, :n]`` host weights."""

    pick: np.ndarray
    w: np.ndarray
    n_hosts: tuple[int, ...]

    @classmethod
    def make(cls, n_hosts: Sequence[int], n_boot: int, seed: int = BOOT_SEED) -> HierDraws:
        rng = np.random.default_rng(seed)
        R, H = len(n_hosts), max(n_hosts)
        pick = np.empty((n_boot, R), dtype=int)
        w = np.zeros((n_boot, R, H))
        for b in range(n_boot):
            for k in range(R):
                j = int(rng.integers(0, R))
                n = n_hosts[j]
                pick[b, k] = j
                w[b, k, :n] = rng.multinomial(n, np.full(n, 1.0 / n))
        return cls(pick, w, tuple(int(n) for n in n_hosts))

    @property
    def n_boot(self) -> int:
        return self.pick.shape[0]

    def replicates(self, stats: Sequence[np.ndarray], metric: str) -> np.ndarray:
        """(n_boot,) replicate values of the resample-mean ``metric`` for one model's per-resample stats."""
        if tuple(len(s) for s in stats) != self.n_hosts:
            raise ValueError("host rows per resample differ from the draws")
        tot = np.zeros((self.n_boot, len(stats), 9))
        for j, s in enumerate(stats):
            m = self.pick == j
            tot[m] = self.w[m][:, : len(s)] @ np.asarray(s, float)
        return _metrics_from_totals(tot)[metric].mean(axis=1)


def summarize(per_resample: Sequence[float], reps: np.ndarray, *, ci: float = 0.95,
              upper_quantile: float = 0.95) -> dict:
    """Point (mean over resamples), two-sided interval, one-sided upper bound, per-resample values."""
    per = np.asarray(per_resample, float)
    lo, hi = (1 - ci) / 2, 1 - (1 - ci) / 2
    return {"mean": float(per.mean()), "ci": [float(np.quantile(reps, lo)), float(np.quantile(reps, hi))],
            "upper_one_sided": float(np.quantile(reps, upper_quantile)),
            "boot_sd": float(reps.std(ddof=1)),
            "sd_over_resamples": float(per.std(ddof=1)) if per.size > 1 else 0.0,
            "per_resample": [float(x) for x in per]}


def n_star(upper: Mapping[int, float], delta: float) -> dict:
    """N star."""
    budgets = sorted(upper)
    below = {b: bool(upper[b] < delta) for b in budgets}
    qualifying = [b for i, b in enumerate(budgets) if all(below[x] for x in budgets[i:])]
    ns = qualifying[0] if qualifying else None
    return {"n_star": ns, "report": str(ns) if ns is not None else f">{budgets[-1]}",
            "below_delta": {str(b): below[b] for b in budgets}, "delta_eV": delta,
            "budgets_tested": budgets}


def contrast(stats_a: Mapping[int, Sequence[np.ndarray]], stats_b: Mapping[int, Sequence[np.ndarray]],
             draws: HierDraws, metric: str, *, ci: float = 0.95, upper_quantile: float = 0.95) -> dict:
    """Per budget: marginal summaries of a and b and of the paired difference a - b."""
    out = {}
    for B in sorted(stats_a):
        ra, rb = draws.replicates(stats_a[B], metric), draws.replicates(stats_b[B], metric)
        pa = [point_metric(s, metric) for s in stats_a[B]]
        pb = [point_metric(s, metric) for s in stats_b[B]]
        out[B] = {"a": summarize(pa, ra, ci=ci, upper_quantile=upper_quantile),
                  "b": summarize(pb, rb, ci=ci, upper_quantile=upper_quantile),
                  "diff": summarize(np.subtract(pa, pb), ra - rb, ci=ci, upper_quantile=upper_quantile)}
    return out
