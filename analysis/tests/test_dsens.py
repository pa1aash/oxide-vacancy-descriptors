"""Synthetic checks with known answers for the cell dropping and seed re-ensembling of analysis/dsens/dsens.py."""
import numpy as np
import pandas as pd
import pytest

from analysis.dsens import dsens as ds


def _long(n_r=3, n_s=4):
    """Two models, two hosts x two sites; seed s of model m predicts y + offset(m) + s (so the ensemble mean is known)."""
    rows = []
    for m, off in (("S", 0.5), ("Dstar", 0.1)):
        for r in range(n_r):
            for s in range(n_s):
                for h in range(2):
                    for k in range(2):
                        y = float(h + 0.1 * k + r)
                        rows.append({"model": m, "r": r, "B": 654, "seed": s, "site_id": f"r{r}h{h}k{k}",
                                     "host_id": f"h{h}", "y_true": y, "y_pred": y + off + s})
    return pd.DataFrame(rows)


def test_drop_cells_reensembles_remaining_seeds():
    long = _long()
    ens, n_left = ds.drop_cells(long, [(1, 0), (1, 3), (2, 2)], 3, 4)
    assert n_left == {0: 4, 1: 2, 2: 3}
    err = (ens.y_pred - ens.y_true).round(12)
    mean_seed = {0: 1.5, 1: 1.5, 2: (0 + 1 + 3) / 3}
    for m, off in (("S", 0.5), ("Dstar", 0.1)):
        for r in range(3):
            e = err[(ens.model == m) & (ens.r == r)]
            assert np.allclose(e, off + mean_seed[r])
            assert (ens[(ens.model == m) & (ens.r == r)].n_seeds == n_left[r]).all()


def test_drop_cells_none_is_full_ensemble():
    long = _long()
    ens, n_left = ds.drop_cells(long, [], 3, 4)
    assert set(n_left.values()) == {4}
    assert np.allclose(ens.y_pred - ens.y_true, np.where(ens.model == "S", 0.5, 0.1) + 1.5)


def test_drop_cells_rejects_emptied_resample():
    with pytest.raises(AssertionError):
        ds.drop_cells(_long(n_s=2), [(0, 0), (0, 1)], 3, 2)


def test_relative_reduction_known_values():
    rp = {"S": (np.array([0.4, 0.2]), np.array([0.3, 0.3, 0.6])),
          "Dstar": (np.array([0.3, 0.15]), np.array([0.15, 0.24, 0.3]))}
    out = ds.relative(rp, "Dstar")
    assert out["R"] == pytest.approx(0.25)
    assert out["ci95"][0] == pytest.approx(np.quantile([0.5, 0.2, 0.5], 0.025))
    assert out["ci95"][1] == pytest.approx(0.5)
