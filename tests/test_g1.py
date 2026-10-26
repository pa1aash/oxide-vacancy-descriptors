"""G1 primary-analysis machinery on synthetic prediction tables with known answers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dftgnn.stats.g1 import (
    HierDraws,
    contrast,
    n_star,
    point_metric,
    resample_stats,
    seed_ensemble,
    summarize,
)
from dftgnn.stats.variance import within_host_residual_mae

BUDGETS = [25, 50, 100, 200, 400, 654]


def _seeds(model, r, B, sites, hosts, y, preds):
    rows = []
    for s, p in enumerate(preds):
        rows += [{"model": model, "r": r, "B": B, "site_id": a, "host_id": h, "seed": s, "y_true": t,
                  "y_pred": q} for a, h, t, q in zip(sites, hosts, y, p, strict=True)]
    return rows


def test_seed_ensemble_is_the_mean_of_the_seeds():
    df = pd.DataFrame(_seeds("S", 0, 25, ["a", "b"], ["h1", "h1"], [1.0, 2.0],
                             [[1.0, 4.0], [2.0, 5.0], [6.0, 0.0]]))
    ens = seed_ensemble(df).set_index("site_id")
    assert ens.loc["a", "y_pred"] == pytest.approx(3.0)
    assert ens.loc["b", "y_pred"] == pytest.approx(3.0)
    assert (ens.n_seeds == 3).all()


def test_seed_ensemble_rejects_inconsistent_targets():
    df = pd.DataFrame(_seeds("S", 0, 25, ["a"], ["h1"], [1.0], [[1.0], [2.0]]))
    df.loc[1, "y_true"] = 9.0
    with pytest.raises(ValueError):
        seed_ensemble(df)


def _frame(y, p, hosts):
    return pd.DataFrame({"y_true": y, "y_pred": p, "host_id": hosts})


def test_advantage_known_answer():
    # resample 0: S errors 1, 1, 2 -> MAE 4/3; D errors 0.5, 0.5, 0.5 -> 0.5; A = 0.8333
    # resample 1: S errors 0.2, 0.4 -> 0.3; D errors 0.5, 0.7 -> 0.6; A = -0.3 (negative, kept)
    y0, y1 = np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0])
    h0, h1 = ["a", "a", "b"], ["c", "d"]
    S = [resample_stats(_frame(y0, y0 + [1, -1, 2], h0)), resample_stats(_frame(y1, y1 + [0.2, -0.4], h1))]
    D = [resample_stats(_frame(y0, y0 + [0.5, 0.5, -0.5], h0)), resample_stats(_frame(y1, y1 - [0.5, 0.7], h1))]
    a = [point_metric(s, "mae") - point_metric(d, "mae") for s, d in zip(S, D, strict=True)]
    assert a == pytest.approx([4 / 3 - 0.5, -0.3])
    draws = HierDraws.make([2, 2], n_boot=200, seed=1)
    out = contrast({25: S}, {25: D}, draws, "mae")[25]["diff"]
    assert out["mean"] == pytest.approx((4 / 3 - 0.5 - 0.3) / 2)
    assert out["per_resample"] == pytest.approx(a)
    assert out["ci"][0] <= out["mean"] <= out["ci"][1]


def test_identical_models_give_zero_difference_in_every_draw():
    rng = np.random.default_rng(0)
    stats = [resample_stats(_frame(rng.normal(size=12), rng.normal(size=12), np.repeat(list("abcdef"), 2)))
             for _ in range(3)]
    draws = HierDraws.make([6, 6, 6], n_boot=100, seed=3)
    assert np.all(draws.replicates(stats, "mae") - draws.replicates(stats, "mae") == 0)


def test_draws_are_reproducible_and_seeded():
    a, b = HierDraws.make([5, 7], 50, seed=20261008), HierDraws.make([5, 7], 50, seed=20261008)
    assert np.array_equal(a.pick, b.pick) and np.array_equal(a.w, b.w)
    # host multiplicities of each slot sum to that resample's host count, padding stays zero
    n = np.array([5, 7])[a.pick]
    assert np.array_equal(a.w.sum(axis=2), n)
    assert np.all(a.w[a.pick == 0][:, 5:] == 0)


def test_bootstrap_covers_a_known_true_advantage():
    """Independent test sets from a host population with true A = 0.1 eV (one site per host)."""
    true_a, R, H, reps = 0.1, 10, 40, 120
    rng = np.random.default_rng(7)
    lower_miss = upper_miss = covered = 0
    for i in range(reps):
        S, D = [], []
        for _ in range(R):
            y = rng.normal(size=H)
            hosts = [f"h{k}" for k in range(H)]
            S.append(resample_stats(_frame(y, y + rng.exponential(0.5, H) * rng.choice([-1, 1], H), hosts)))
            D.append(resample_stats(_frame(y, y + rng.exponential(0.4, H) * rng.choice([-1, 1], H), hosts)))
        draws = HierDraws.make([H] * R, n_boot=300, seed=1000 + i)
        d = contrast({654: S}, {654: D}, draws, "mae")[654]["diff"]
        covered += d["ci"][0] <= true_a <= d["ci"][1]
        lower_miss += d["upper_one_sided"] < true_a
        upper_miss += d["ci"][0] > true_a
    assert covered / reps >= 0.90
    assert lower_miss / reps <= 0.10


def test_summarize_keeps_negative_values():
    reps = np.linspace(-0.3, -0.1, 101)
    s = summarize([-0.2, -0.2], reps)
    assert s["mean"] == pytest.approx(-0.2)
    assert s["upper_one_sided"] == pytest.approx(np.quantile(reps, 0.95))
    assert s["ci"] == pytest.approx([np.quantile(reps, 0.025), np.quantile(reps, 0.975)])


def test_n_star_first_budget_below_but_later_above_does_not_qualify():
    upper = dict(zip(BUDGETS, [0.01, 0.08, 0.04, 0.03, 0.02, 0.01], strict=True))
    out = n_star(upper, 0.05)
    assert out["n_star"] == 100 and out["report"] == "100"
    assert out["below_delta"]["25"] is True


def test_n_star_none_qualifies_is_reported_as_lower_bound():
    upper = dict(zip(BUDGETS, [0.3, 0.2, 0.04, 0.03, 0.02, 0.06], strict=True))
    out = n_star(upper, 0.05)
    assert out["n_star"] is None and out["report"] == ">654"


def test_n_star_all_qualify():
    out = n_star(dict(zip(BUDGETS, [0.04] * 6, strict=True)), 0.05)
    assert out["n_star"] == 25 and out["report"] == "25"


def test_n_star_with_negative_advantage_and_strict_comparison():
    upper = dict(zip(BUDGETS, [0.2, 0.05, -0.01, -0.2, -0.05, -0.1], strict=True))
    out = n_star(upper, 0.05)
    assert out["n_star"] == 100          # 0.05 is not below 0.05
    assert out["below_delta"]["50"] is False


def test_within_host_residual_known_answer_and_bootstrap_route():
    # host a: y = [0, 2], p = [1, 1] -> residuals (1 - 1) - (0 - 1) = 1 and (1 - 1) - (2 - 1) = -1
    # host b: y = [0, 1, 2], p = [0, 0, 3] -> p - pbar = [-1, -1, 2], y - ybar = [-1, 0, 1] -> [0, -1, 1]
    # host c: single site, excluded
    y = np.array([0.0, 2.0, 0.0, 1.0, 2.0, 5.0])
    p = np.array([1.0, 1.0, 0.0, 0.0, 3.0, 9.0])
    h = ["a", "a", "b", "b", "b", "c"]
    expect = (1 + 1 + 0 + 1 + 1) / 5
    assert within_host_residual_mae(y, p, h) == pytest.approx(expect)
    assert point_metric(resample_stats(_frame(y, p, h)), "within_host_mae") == pytest.approx(expect)
