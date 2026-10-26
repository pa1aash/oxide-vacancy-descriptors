"""Synthetic checks with known answers for the latent probe (fit mathematics, controls, rule, bootstrap)."""
import numpy as np
import pytest

from analysis import probe_fit as pf
from analysis import probe_stats as ps


def _sites(n_hosts=60, per_host=3, seed=0):
    rng = np.random.default_rng(seed)
    hosts = np.repeat([f"h{h:03d}" for h in range(n_hosts)], per_host)
    sids = np.array([f"{h}_s{k}" for h, k in zip(hosts, np.tile(np.arange(per_host), n_hosts))])
    return rng, hosts, sids


def test_ridge_matches_sklearn():
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(1)
    x, y, xe = rng.normal(size=(80, 7)), rng.normal(size=(80, 3)), rng.normal(size=(20, 7))
    p = pf.ridge_predict(x, y, xe, [0.1, 10.0])
    for i, a in enumerate([0.1, 10.0]):
        np.testing.assert_allclose(p[i], Ridge(alpha=a).fit(x, y).predict(xe), atol=1e-10)


def test_planted_linear_map_recovered():
    rng, hosts, _ = _sites(n_hosts=120)
    x = rng.normal(size=(len(hosts), 20))
    w = rng.normal(size=(20, 4))
    y = x @ w + 0.05 * rng.normal(size=(len(hosts), 4))
    tr = np.arange(len(hosts)) < 270
    f = pf.fit_probe(x[tr], y[tr], hosts[tr], x[~tr])
    r2 = pf.r2_scores(y[~tr], f["pred"])
    assert (r2 > 0.99).all()


def test_shuffled_control_near_zero():
    rng, hosts, sids = _sites(n_hosts=300, per_host=2)
    hx = {h: rng.normal(size=10) for h in np.unique(hosts)}
    x = np.array([hx[h] for h in hosts]) + 0.01 * rng.normal(size=(len(hosts), 10))
    y = x[:, :3].copy()
    tr = np.arange(len(hosts)) < 480
    ysh = pf.shuffle_by_host(y[tr], hosts[tr], sids[tr], seed=pf.SHUFFLE_SEED)
    true_r2 = pf.r2_scores(y[~tr], pf.fit_probe(x[tr], y[tr], hosts[tr], x[~tr])["pred"])
    sh_r2 = pf.r2_scores(y[~tr], pf.fit_probe(x[tr], ysh, hosts[tr], x[~tr])["pred"])
    assert (true_r2 > 0.99).all()
    assert (np.abs(sh_r2) < 0.1).all()


def test_shuffle_permutes_whole_hosts_deterministically():
    _, hosts, sids = _sites(n_hosts=30, per_host=3)
    y = np.array([[int(h[1:]) * 10 + k] for h, k in zip(hosts, np.tile(np.arange(3), 30))], float)
    a = pf.shuffle_by_host(y, hosts, sids, seed=5)
    b = pf.shuffle_by_host(y, hosts, sids, seed=5)
    np.testing.assert_array_equal(a, b)
    for h in np.unique(hosts):
        src = set((a[hosts == h, 0] // 10).astype(int))
        assert len(src) == 1                                   # all rows of a host come from one host
    np.testing.assert_array_equal(np.sort(a[:, 0]), np.sort(y[:, 0]))   # equal counts: a permutation of rows
    assert not np.array_equal(a, y)


def test_shuffle_cycles_unequal_counts():
    hosts = np.array(["a", "b", "b", "b"])
    sids = np.array(["a1", "b1", "b2", "b3"])
    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    out = pf.shuffle_by_host(y, hosts, sids, seed=0)
    assert out.shape == y.shape and np.isfinite(out).all()


def test_group_cv_never_splits_a_host():
    rng, hosts, _ = _sites(n_hosts=37, per_host=4)
    hosts = rng.permutation(hosts)
    folds = pf.group_folds(hosts)
    assert len(folds) == 5
    seen = np.zeros(len(hosts), int)
    for tr, va in folds:
        assert not set(hosts[tr]) & set(hosts[va])
        seen[va] += 1
    assert (seen == 1).all()


def test_alpha_ties_go_to_smaller():
    sse = np.array([[1.0, 2.0], [1.0, 1.0], [3.0, 1.0]])
    np.testing.assert_array_equal(pf.choose_alpha(sse), [0, 1])


def test_r2_undefined_for_constant_target_and_class_mean_skips_it():
    y = np.column_stack([np.zeros(5), np.arange(5.0)])
    r2 = pf.r2_scores(y, y + 0.0)
    assert np.isnan(r2[0]) and r2[1] == 1.0
    cm = pf.class_means(r2, ["host", "host"])
    assert cm["host"] == 1.0 and cm["n_undefined_host"] == 1


def test_within_host_r2():
    hosts = np.array(["a", "a", "b", "b", "b", "c"])
    y = np.array([1.0, 2.0, 5.0, 7.0, 6.0, 9.0])
    r2, nh, ns = pf.within_host_r2(y, y + np.array([3, 3, -1, -1, -1, 4.0]), hosts)
    assert r2[0] == pytest.approx(1.0) and nh == 2 and ns == 5           # host offsets do not count
    flat = np.array([1.5, 1.5, 6.0, 6.0, 6.0, 9.0])                      # host means only
    assert pf.within_host_r2(y, flat, hosts)[0][0] == pytest.approx(0.0)


def test_probe_cell_runs_all_probes():
    rng, hosts, sids = _sites(n_hosts=40, per_host=3)
    x = rng.normal(size=(len(hosts), 8))
    y = np.column_stack([x[:, 0], x[:, 1] + 0.1 * rng.normal(size=len(hosts))])
    split = np.where(np.arange(len(hosts)) < 90, "train", "test")
    out = pf.probe_cell({"trained": x, "init0": rng.normal(size=x.shape)}, y, split, hosts, sids, ["host", "site"])
    assert set(out) == {"trained", "shuffled", "init0"}
    assert out["trained"]["class_mean"]["host"] > 0.9
    assert out["trained"]["class_mean"]["host"] > out["shuffled"]["class_mean"]["host"]


def test_trend_rule_logic():
    idx = ps.boot_index(n_res=10, n_boot=300)
    rng = np.random.default_rng(3)
    up = np.tile(np.linspace(0.0, 0.5, 6), (10, 1)) + 0.02 * rng.normal(size=(10, 6))
    flat = 0.02 * rng.normal(size=(10, 6))
    t = ps.trend(up, ps.BUDGETS, idx)
    assert t["rho"] > 0.8 and t["holds"]
    assert not ps.trend(flat, ps.BUDGETS, idx)["holds"]
    assert not ps.trend(-up, ps.BUDGETS, idx)["holds"]


def test_apply_rule_needs_both_controls():
    idx = ps.boot_index(n_res=10, n_boot=300)
    rng = np.random.default_rng(4)
    trained = np.tile(np.linspace(0.1, 0.6, 6), (10, 1)) + 0.01 * rng.normal(size=(10, 6))
    low = trained - 0.3
    ok = ps.apply_rule({"trained": trained, "shuffled": low, "init0": low}, idx)
    assert ok["holds"] and ok["exceeds_both_controls_all_budgets"]
    bad = ps.apply_rule({"trained": trained, "shuffled": low, "init0": trained + 0.01}, idx)
    assert bad["trend"]["holds"] and not bad["holds"]
    assert not bad["controls"]["654|init0"]["exceeds"] and bad["controls"]["654|shuffled"]["exceeds"]


def test_bootstrap_is_paired_by_resample():
    idx = ps.boot_index(n_res=10, n_boot=500)
    np.testing.assert_array_equal(idx, ps.boot_index(n_res=10, n_boot=500))
    a = np.random.default_rng(5).normal(size=10)
    d = ps.paired_diff(a + 0.2, a, idx)                 # a common resample effect cancels exactly
    assert d["ci95"] == pytest.approx([0.2, 0.2])
    unpaired = np.random.default_rng(6).normal(size=10)
    assert np.diff(ps.paired_diff(a + 0.2, unpaired, idx)["ci95"])[0] > 0.1


def test_matrix_requires_every_cell():
    cells = [{"r": r, "B": b, "probes": {"trained": {"class_mean": {"host": 0.1}}}}
             for r in range(10) for b in ps.BUDGETS if (r, b) != (3, 200)]
    with pytest.raises(AssertionError):
        ps.matrix(cells, "trained", lambda q: q["class_mean"]["host"])
