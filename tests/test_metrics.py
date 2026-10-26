import numpy as np
import pytest

from dftgnn.stats import metrics as M
from dftgnn.stats.variance import within_host_residual_mae


def toy():
    g = np.array(["a", "a", "b", "b", "b", "c"])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    return y, g


def test_constant_offset_known_answers():
    y, g = toy()
    m = M.point_metrics(y, y + 0.5, g)
    sst = ((y - y.mean()) ** 2).sum()
    assert m["mae"] == pytest.approx(0.5) and m["rmse"] == pytest.approx(0.5)
    assert m["r2"] == pytest.approx(1 - 6 * 0.25 / sst)
    assert m["within_host_mae"] == pytest.approx(0.0)  # offsets cancel within a host
    assert m["host_mean_mae"] == pytest.approx(0.5)


def test_perfect_and_mean_predictor():
    y, g = toy()
    m = M.point_metrics(y, y, g)
    assert m == {"mae": 0, "rmse": 0, "r2": 1, "within_host_mae": 0, "host_mean_mae": 0}
    m = M.point_metrics(y, np.full_like(y, y.mean()), g)
    assert m["r2"] == pytest.approx(0.0)


def test_within_host_known_value():
    # host a: y 1,2 predicted 1,3 -> residuals centred: (-0.5)-(-1)=0.5 and 0.5-1=-0.5 -> |r|=0.5
    y = np.array([1.0, 2.0, 7.0])
    p = np.array([1.0, 3.0, 0.0])
    g = np.array(["a", "a", "b"])  # singleton b excluded
    assert M.point_metrics(y, p, g)["within_host_mae"] == pytest.approx(0.5)
    assert within_host_residual_mae(y, p, g) == pytest.approx(0.5)


def test_host_mean_mae_unweighted_by_size():
    y = np.array([0.0, 0.0, 0.0, 10.0])
    p = np.array([1.0, 1.0, 1.0, 10.0])
    g = np.array(["a", "a", "a", "b"])
    assert M.point_metrics(y, p, g)["host_mean_mae"] == pytest.approx(0.5)  # (1 + 0) / 2


def test_sufficient_statistics_match_direct():
    rng = np.random.default_rng(1)
    g = rng.integers(0, 40, 300).astype(str)
    y = rng.normal(size=300)
    p = y + rng.normal(scale=0.3, size=300) + 0.1
    pt = M.point_metrics(y, p, g)
    via = M._metrics_from_totals(M.host_stats(y, p, g).to_numpy().sum(axis=0))
    for k in M.METRICS:
        assert via[k] == pytest.approx(pt[k], rel=1e-12)


def test_bootstrap_degenerate_has_zero_width():
    y, g = toy()
    r = M.cluster_bootstrap_ci(y, y + 0.5, g, n_boot=300, seed=0)
    assert r["mae"]["ci"] == pytest.approx([0.5, 0.5])


def test_bootstrap_sd_matches_analytic_se():
    rng = np.random.default_rng(2)
    H, k = 400, 2
    g = np.repeat(np.arange(H), k)
    y = rng.normal(size=H * k)
    e = rng.normal(scale=0.5, size=H * k)
    r = M.cluster_bootstrap_ci(y, y + e, g, n_boot=2000, seed=3)["mae"]
    host_mae = np.abs(e).reshape(H, k).mean(axis=1)
    se = host_mae.std(ddof=1) / np.sqrt(H)
    assert r["boot_sd"] == pytest.approx(se, rel=0.12)
    assert r["ci"][0] < r["point"] < r["ci"][1]


def test_bootstrap_reproducible():
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(30), 2)
    y = rng.normal(size=60)
    p = y + rng.normal(scale=0.2, size=60)
    a = M.cluster_bootstrap_ci(y, p, g, n_boot=200, seed=5)
    b = M.cluster_bootstrap_ci(y, p, g, n_boot=200, seed=5)
    assert a == b


def _runs(R, H=60, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    out = []
    for r in range(R):
        g = np.repeat(np.arange(H), 2)
        y = rng.normal(size=2 * H)
        out.append(M.host_stats(y, y + shift + rng.normal(scale=0.3, size=2 * H), g))
    return out


def test_aggregate_mean_sd_and_interval():
    runs = _runs(10)
    agg = M.aggregate_resamples(runs, n_boot=500, seed=1)["mae"]
    per = np.array(agg["per_resample"])
    assert agg["mean"] == pytest.approx(per.mean()) and agg["sd"] == pytest.approx(per.std(ddof=1))
    assert agg["ci"][0] < agg["mean"] < agg["ci"][1]


def test_aggregate_identical_resamples_reduce_to_cluster_ci():
    one = _runs(1)[0]
    agg = M.aggregate_resamples([one] * 5, n_boot=2000, seed=2)["mae"]
    w = M._draw_weights(np.random.default_rng(9), len(one), 2000)
    ref = M._metrics_from_totals(w @ one.to_numpy())["mae"]
    assert agg["sd"] == pytest.approx(0.0, abs=1e-12)
    assert agg["boot_sd"] == pytest.approx(ref.std(ddof=1) / np.sqrt(5), rel=0.15)


def test_delta_known_shift_and_pairing():
    a = _runs(6, shift=0.0, seed=4)
    b = [s.copy() for s in a]
    for s in b:
        s["sabs"] += s["n"] * 0.1  # error 0.1 larger everywhere (adds to |e|)
    d = M.aggregate_delta(b, a, n_boot=500, seed=1)["mae"]
    assert d["mean"] == pytest.approx(0.1)
    assert d["ci"] == pytest.approx([0.1, 0.1])  # pairing removes all sampling noise
    with pytest.raises(ValueError):
        M.aggregate_delta(a[:2], a[:3])
