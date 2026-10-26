import numpy as np
import pytest

from dftgnn.baselines import Data, feature_sets, physics, rf, run_kiyohara, run_outer
from dftgnn.config import load_config
from dftgnn.data import universe as U

CFG = load_config()
needs_universe = pytest.mark.skipif(not (U.REPO_ROOT / "data/processed/universe_v1.parquet").is_file(),
                                    reason="universe parquet not built")


def test_ols_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = 1.5 - 2.0 * X[:, 0] + 0.7 * X[:, 1]
    assert physics.ols(X, y) == pytest.approx([1.5, -2.0, 0.7])


def test_max_features_grid():
    assert rf.max_features_grid(70) == list(range(20, 45))
    for n in (22, 48):
        g = rf.max_features_grid(n)
        assert g == sorted(set(g)) and 1 <= g[0] and g[-1] <= n
        assert g[0] == round(20 / 70 * n) and g[-1] == round(44 / 70 * n)


def test_interpretation_follows_sign():
    s = physics.interpret(-1.0, 0.3)
    assert "more stable oxide" in s["stability"] and "wider-gap oxide goes with a costlier" in s["gap"]
    assert "opposite" in physics.interpret(1.0, -0.3)["stability"]


@needs_universe
def test_feature_sets_partition_taxonomy():
    d = Data.load(CFG)
    fs = feature_sets(CFG, d)
    assert len(fs["kumagai"]) == 70 and len(fs["electronic"]) == 22 and len(fs["structural"]) == 48
    assert not set(fs["electronic"]) & set(fs["structural"])
    assert set(fs["electronic"]) | set(fs["structural"]) == set(fs["kumagai"])
    assert {d.desc_class[f] for f in fs["electronic"]} == set(CFG.models.D.descriptor_classes)
    assert {physics.STABILITY, physics.GAP} <= set(fs["electronic"])


@needs_universe
def test_run_loops_never_leak_hosts_and_cover_all_cells():
    d = Data.load(CFG)
    seen = []

    def fit(tr, te, seed):
        a, b = set(d.uni.host_id.iloc[tr]), set(d.uni.host_id.iloc[te])
        assert not a & b
        seen.append(len(a))
        return {"pred": np.zeros(len(te)), "info": {}}

    frame, infos = run_outer(d, fit, CFG, seed_tag="t", budgets=[25, 654], resamples=range(2), log=lambda *_: None)
    assert seen == [25, 654, 25, 654] and len(infos) == 4
    assert set(zip(frame["resample"], frame["budget"], strict=True)) == {(0, 25), (0, 654), (1, 25), (1, 654)}
    assert frame.groupby(["resample", "budget"]).host_id.nunique().eq(164).all()
    kf, ki = run_kiyohara(d, fit, seed_tag="t")
    assert ki["n_train_hosts"] == 571 and ki["n_test_hosts"] == 126 and (kf["resample"] == -1).all()


@needs_universe
def test_b0_runs_and_has_expected_signs():
    d = Data.load(CFG)
    fit = physics.make_fit(d)
    full = np.arange(len(d.uni))
    c = fit(full, full, 0)["info"]["coef"]
    assert c["stability_eV_per_eV_per_atom"] < 0 < c["gap_eV_per_eV"]  # descriptive check on the sample
