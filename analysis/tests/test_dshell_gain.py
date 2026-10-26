"""Toy arrays with known answers for the pooled class-restricted estimator of analysis/dshell/dshell_gain.py."""
import numpy as np

from analysis.dshell import dshell_gain as dg
from dftgnn.stats.g1 import HierDraws

# two resamples, two hosts each; columns (numerator, denominator)
STATS = [np.array([[2.0, 1.0], [9.0, 3.0]]), np.array([[4.0, 2.0], [5.0, 5.0]])]
MASKS = [np.array([True, False]), np.array([True, True])]


def test_pooled_point_is_ratio_of_sums():
    pick = np.zeros((1, 2), dtype=int)
    w = np.zeros((1, 2, 2))
    pt, _ = dg.pooled(STATS, MASKS, pick, w, 0, 1)
    assert pt == (2.0 + 4.0 + 5.0) / (1.0 + 2.0 + 5.0)


def test_pooled_replicate_by_hand():
    # replicate 0: slot 0 -> resample 1 with weights (2, 0); slot 1 -> resample 0 with weights (1, 1)
    # replicate 1: both slots resample 0, weights (0, 2) and (2, 0)
    pick = np.array([[1, 0], [0, 0]])
    w = np.array([[[2.0, 0.0], [1.0, 1.0]], [[0.0, 2.0], [2.0, 0.0]]])
    _, reps = dg.pooled(STATS, MASKS, pick, w, 0, 1)
    # replicate 0: class rows: resample 1 host 0 twice (8 / 4), resample 0 host 0 once (2 / 1); host 1 of r0 masked
    assert reps[0] == (8.0 + 2.0) / (4.0 + 1.0)
    # replicate 1: slot 0 draws only the masked-out host -> contributes nothing
    assert reps[1] == 4.0 / 2.0


def test_empty_class_replicate_is_undefined():
    pick = np.array([[0, 0], [1, 1]])
    w = np.array([[[0.0, 2.0], [0.0, 2.0]], [[1.0, 1.0], [1.0, 1.0]]])
    _, reps = dg.pooled(STATS, MASKS, pick, w, 0, 1)
    s = dg.summary(1.0, reps)
    assert np.isnan(reps[0]) and s["n_replicates_undefined"] == 1


def test_mean_over_slots_matches_hier_draws():
    rng = np.random.default_rng(1)
    stats = [np.abs(rng.normal(size=(6, 9))) + 0.1 for _ in range(4)]
    draws = HierDraws.make([6] * 4, 50, seed=3)
    tot = dg.slot_totals(stats, [np.ones(6)] * 4, draws.pick, draws.w)
    mine = (tot[..., 1] / tot[..., 0]).mean(axis=1)
    assert np.allclose(mine, draws.replicates(stats, "mae"), atol=1e-12)
