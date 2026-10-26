"""Synthetic checks with known answers for the hit rule of analysis/coord/site_selection.py and the bins."""
import numpy as np
import pandas as pd

from analysis.coord import bins
from analysis.coord import site_selection as ss


def test_hit_unique_minimum():
    assert ss.host_hit([1.0, 2.0, 3.0], [0.5, 0.9, 1.2])[:2] == (1, 1)
    assert ss.host_hit([1.0, 2.0, 3.0], [0.9, 0.5, 1.2])[:2] == (0, 1)


def test_hit_ties_in_target():
    # two sites share the lowest target: either one picked is a hit, the third is not
    assert ss.host_hit([1.0, 1.0, 3.0], [0.7, 0.5, 2.0]) == (1, 2, False)
    assert ss.host_hit([1.0, 1.0, 3.0], [0.5, 0.7, 2.0]) == (1, 2, False)
    assert ss.host_hit([1.0, 1.0, 3.0], [0.9, 0.8, 0.1]) == (0, 2, False)


def test_tie_in_prediction_takes_first_site_and_is_flagged():
    assert ss.host_hit([2.0, 1.0], [0.3, 0.3]) == (0, 1, True)
    assert ss.host_hit([1.0, 2.0], [0.3, 0.3]) == (1, 1, True)


def test_host_table_eligibility_and_fractions():
    f = pd.DataFrame({"site_id": ["a1", "a2", "b1", "c1", "c2", "c3"], "host_id": ["A", "A", "B", "C", "C", "C"],
                      "y_true": [1.0, 2.0, 5.0, 3.0, 3.0, 4.0], "y_pred": [1.1, 1.5, 9.0, 2.0, 1.0, 0.5]})
    t = ss.host_table(f)
    assert t.host_id.tolist() == ["A", "B", "C"] and t.hit.tolist() == [1, 1, 0]
    s2, s3 = ss.stats(t, 2), ss.stats(t, 3)
    assert s2[:, 1].tolist() == [1, 0, 1] and s3[:, 1].tolist() == [0, 0, 1]
    per = ss.per_resample([s2])
    assert per[0, 0] == 0.5 and np.isclose(per[0, 1], (1 / 2 + 2 / 3) / 2)

    class Draws:     # one replicate, two slots (two copies of the resample), host weights (2, 1, 0) and (0, 0, 3)
        n_boot = 1
        pick = np.array([[0, 1]])
        w = np.array([[[2.0, 1.0, 0.0], [0.0, 0.0, 3.0]]])

    rep = ss.replicates([s2, s2], Draws)
    assert rep[0, 0] == (1.0 + 0.0) / 2 and np.isclose(rep[0, 1], (1 / 2 + 2 / 3) / 2)


def test_bins():
    assert bins.bin_of(None) == ("unassigned", "crystalnn_failed")
    assert bins.bin_of(1) == ("unassigned", "cn<2") and bins.bin_of(0)[1] == "cn<2"
    assert [bins.bin_of(n)[0] for n in (2, 3, 4, 5, 8)] == ["2", "3", "4", "5+", "5+"]
    assert bins.release_cn("{'Ag': [2.18], 'Bi': [2.13, 2.29], 'O': [2.5]}") == 3
