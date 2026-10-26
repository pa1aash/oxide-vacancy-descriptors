import numpy as np
import pytest

from dftgnn.stats.variance import oneway_icc, within_group_spread, within_host_residual_mae


def test_balanced_moments_match_textbook():
    # balanced: n0 = n, sigma_b^2 = (MSB - MSW) / n
    y = np.array([1.0, 3.0, 5.0, 7.0, 4.0, 6.0])
    g = np.array([0, 0, 1, 1, 2, 2])
    r = oneway_icc(y, g, n_boot=50)
    msw = (2 + 2 + 2) / 3
    means = np.array([2.0, 6.0, 5.0])
    msb = 2 * ((means - y.mean()) ** 2).sum() / 2
    assert r["msw"] == pytest.approx(msw)
    assert r["msb"] == pytest.approx(msb)
    assert r["n0"] == pytest.approx(2)
    assert r["icc"] == pytest.approx(((msb - msw) / 2) / ((msb - msw) / 2 + msw))


def test_recovers_known_icc_unbalanced():
    rng = np.random.default_rng(1)
    sizes = rng.integers(1, 6, size=1500)
    g = np.repeat(np.arange(sizes.size), sizes)
    y = rng.normal(0, 2.0, sizes.size)[g] + rng.normal(0, 1.0, g.size)
    r = oneway_icc(y, g, n_boot=200)
    assert r["icc"] == pytest.approx(0.8, abs=0.03)
    assert r["icc_ci"][0] < 0.8 < r["icc_ci"][1]


def test_spread_and_within_host_mae():
    y = np.array([1.0, 2.0, 4.0, 9.0])
    g = np.array(["a", "a", "a", "b"])
    s = within_group_spread(y, g)
    assert s["n_groups_with_ge2"] == 1
    assert s["range_eV"]["q50"] == pytest.approx(3.0)
    # constant offset per host gives zero within-host error
    assert within_host_residual_mae(y, y + 5.0, g) == pytest.approx(0.0)
    p = np.array([1.0, 2.0, 1.0, 0.0])  # host a: residual centred (0,0,3)-(... )
    r = (y[:3] - y[:3].mean()) - (p[:3] - p[:3].mean())
    assert within_host_residual_mae(y, p, g) == pytest.approx(np.abs(r).mean())
