"""Probe mechanics on synthetic features (no data files)."""
from __future__ import annotations

import numpy as np

from dftgnn import probe


def _data(n_hosts=60, per=3, d=20, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_hosts), per)
    x = rng.normal(size=(len(groups), d))
    w = rng.normal(size=(d, 2))
    y = x @ w + 0.1 * rng.normal(size=(len(groups), 2))
    return x, y, groups


def test_probe_recovers_a_linear_target_and_shuffled_labels_do_not():
    x, y, g = _data()
    tr, te = g < 45, g >= 45
    ok = probe.probe(x[tr], y[tr], g[tr], x[te], y[te])
    assert min(ok["r2"]) > 0.9
    ys = probe.shuffle_by_host(y[tr], g[tr], seed=0)
    bad = probe.probe(x[tr], ys, g[tr], x[te], y[te])
    assert max(bad["r2"]) < 0.3


def test_shuffle_by_host_moves_whole_hosts():
    _x, y, g = _data(n_hosts=10, per=2)
    ys = probe.shuffle_by_host(y, g, seed=1)
    for h in np.unique(g):
        rows = ys[g == h]
        src = [k for k in np.unique(g) if np.allclose(y[g == k], rows)]
        assert len(src) == 1
