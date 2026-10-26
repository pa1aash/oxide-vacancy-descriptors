"""Synthetic checks with known answers for the site-level pooled estimator of analysis/coord/coord_gain.py."""
import numpy as np
import pandas as pd

from analysis.coord import coord_gain as cg

# host A: three sites, y 1, 2, 3, p 1, 1, 4 (host means y 2, p 2); host B: one site, error 1
FRAME = pd.DataFrame({"site_id": ["a1", "a2", "a3", "b1"], "host_id": ["A", "A", "A", "B"],
                      "y_true": [1.0, 2.0, 3.0, 5.0], "y_pred": [1.0, 1.0, 4.0, 6.0]})
BIN = pd.Series({"a1": "2", "a2": "3", "a3": "3", "b1": "2"})


def test_site_terms_use_host_means_over_all_sites():
    t = cg.site_terms(FRAME).set_index("site_id")
    assert t.ae.tolist() == [0.0, 1.0, 1.0, 1.0]
    # residuals (y - 2) - (p - 2): 0, 1, -1 in host A; host B has one site, no within-host term
    assert t.wr.tolist() == [0.0, 1.0, 1.0, 0.0] and t.wn.tolist() == [1.0, 1.0, 1.0, 0.0]


def test_group_stats_and_pooled_ratio():
    s = cg.host_group_stats(cg.site_terms(FRAME), BIN, ["A", "B"], groups=("2", "3", "all"))
    assert s[0, 0].tolist() == [0.0, 1.0, 0.0, 1.0] and s[1, 0].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert s[0, 1].tolist() == [2.0, 2.0, 2.0, 2.0]
    assert cg.per_resample_ratio(s)[:, 1].tolist() == [0.5, 1.0, 0.75]

    class Draws:     # one replicate; slot 0 on resample 0 with host weights (0, 2), slot 1 on resample 1 with (1, 0)
        n_boot = 1
        pick = np.array([[0, 1]])
        w = np.array([[[0.0, 2.0], [1.0, 0.0]]])

    point, rep = cg.pooled([s, s], Draws)
    assert point[:, 1].tolist() == [0.5, 1.0, 0.75]
    # bin 2 site MAE pooled over slots: (2 * 1 + 1 * 0) / (2 * 1 + 1 * 1); within-host: 0 / 1
    assert np.isclose(rep[0, 0, 1], 2 / 3) and rep[0, 0, 0] == 0.0


def test_slope():
    assert np.isclose(cg.slope(np.array([1.0, 3.0, 5.0, 7.0])), 2.0)
    assert np.allclose(cg.slope(np.array([[0.0, 0.0, 0.0, 1.0], [2.0, 2.0, 2.0, 2.0]])), [0.3, 0.0])
