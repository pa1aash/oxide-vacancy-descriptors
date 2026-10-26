"""Synthetic checks with known answers for the precision and grouped-CV analyses."""
import numpy as np
import pandas as pd
import pytest

from analysis import a1_common as c
from analysis import cv_analysis as cva
from analysis import precision_analysis as pa
from analysis import precision_cv_eval_table as et
from dftgnn.stats.g1 import HierDraws


def _hosts(n_hosts, sites_per_host=2, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for h in range(n_hosts):
        for s in range(sites_per_host):
            rows.append((f"h{h:03d}_s{s}", f"h{h:03d}", float(rng.normal(4.0, 1.0))))
    return pd.DataFrame(rows, columns=["site_id", "host_id", "y_true"])


def _precision_long(n_r=3, n_seeds=2, n_hosts=6, err_s=0.3, err_d=0.1, budgets=(400, 654)):
    """Every resample tests the same hosts; S errs by +err_s on every site, D by -err_d (seed-independent)."""
    base = _hosts(n_hosts)
    rows = []
    for r in range(n_r):
        for B in budgets:
            for seed in range(n_seeds):
                for model, e in (("S", err_s), ("D", -err_d)):
                    rows.append(base.assign(model=model, r=r, B=B, seed=seed, y_pred=base.y_true + e))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- known A

def test_precision_known_A():
    df = _precision_long()
    draws = HierDraws.make([6] * 3, 200, seed=1)
    res = pa.analyse(df, 3, 2, draws)
    for B in ("400", "654"):
        assert res[B]["MAE_S"]["mean"] == pytest.approx(0.3)
        assert res[B]["MAE_D"]["mean"] == pytest.approx(0.1)
        assert res[B]["A"]["mean"] == pytest.approx(0.2)
        assert res[B]["A"]["ci"] == pytest.approx([0.2, 0.2])
        assert res[B]["A"]["upper_one_sided"] == pytest.approx(0.2)
        assert not res[B]["below_delta"]
        # a constant shift has no within-host error
        assert res[B]["A_within"]["mean"] == pytest.approx(0.0)


def test_precision_seed_ensemble_cancels_opposite_seed_errors():
    df = _precision_long(n_seeds=2, err_s=0.0, err_d=0.0)
    sign = np.where(df.seed == 0, 1.0, -1.0)
    df = df.assign(y_pred=df.y_pred + 0.4 * sign * (df.model == "S"))
    res = pa.analyse(df, 3, 2, HierDraws.make([6] * 3, 50, seed=2))
    assert res["654"]["MAE_S"]["mean"] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------- subset selection

def test_subset_selection():
    df = _precision_long(n_r=4, n_seeds=5)
    sub = pa.subset(df, 2, 3)
    assert sorted(sub.r.unique()) == [0, 1] and sorted(sub.seed.unique()) == [0, 1, 2]
    assert len(sub) == 2 * 2 * 3 * 2 * 12


def test_subset_rejects_incomplete_cells():
    df = _precision_long(n_r=2, n_seeds=3)
    df = df[~((df.r == 1) & (df.seed == 2) & (df.model == "D") & (df.B == 400))]
    with pytest.raises(AssertionError):
        pa.subset(df, 2, 3)


def test_subset_names_fixed():
    assert [s[0] for s in c.subset_names()] == ["30x5", "10x3", "10x5", "30x3"]


# ---------------------------------------------------------------- CV: fold coverage and pooling

def test_check_partition():
    et.check_partition([["a", "b"], ["c"]], ["a", "b", "c"])
    with pytest.raises(AssertionError):
        et.check_partition([["a", "b"], ["b", "c"]], ["a", "b", "c"])
    with pytest.raises(AssertionError):
        et.check_partition([["a"], ["c"]], ["a", "b", "c"])


def _cv_long(k=3, n_hosts=9, seeds=3, err_s=0.25, err_d=0.05):
    base = _hosts(n_hosts)
    fold_of = {h: i % k for i, h in enumerate(sorted(base.host_id.unique()))}
    rows = []
    for model, e in (("S", err_s), ("D", -err_d)):
        for seed in range(seeds):
            rows.append(base.assign(model=model, K=k, seed=seed, fold=base.host_id.map(fold_of),
                                    y_pred=base.y_true + e))
    return pd.concat(rows, ignore_index=True), sorted(base.host_id.unique())


def test_pooled_oof_each_site_once_and_known_A():
    cv, hosts = _cv_long()
    oof = cva.pooled_oof(cv, n_seeds=3)
    assert oof.groupby("model").site_id.nunique().tolist() == [18, 18]
    w = cva.host_draws(len(hosts), n_boot=100, seed=3)
    st = {m: cva.host_matrix(oof[oof.model == m], hosts) for m in ("S", "D")}
    res = cva.paired(st["S"], st["D"], w, "mae")
    assert res["diff"]["mean"] == pytest.approx(0.2)
    assert res["diff"]["ci"] == pytest.approx([0.2, 0.2])


def test_pooled_oof_rejects_site_in_two_folds():
    cv, _ = _cv_long()
    bad = cv.copy()
    bad.loc[(bad.seed == 1) & (bad.site_id == "h000_s0"), "fold"] = 2
    with pytest.raises(AssertionError):
        cva.pooled_oof(bad, n_seeds=3)


def test_pooled_oof_rejects_missing_seed():
    cv, _ = _cv_long()
    with pytest.raises(AssertionError):
        cva.pooled_oof(cv[~((cv.seed == 2) & (cv.site_id == "h001_s1") & (cv.model == "S"))], n_seeds=3)


# ---------------------------------------------------------------- bootstrap pairing

def test_host_bootstrap_is_paired():
    cv, hosts = _cv_long(err_s=0.0, err_d=0.0)
    rng = np.random.default_rng(5)
    cv = cv.assign(y_pred=cv.y_pred + rng.normal(0, 0.3, len(cv)))
    oof = cva.pooled_oof(cv)
    w = cva.host_draws(len(hosts), n_boot=300, seed=7)
    s = cva.host_matrix(oof[oof.model == "S"], hosts)
    d = cva.host_matrix(oof[oof.model == "D"], hosts)
    res = cva.paired(s, d, w, "mae")
    ra, rb = cva.metric_reps(s, w, "mae"), cva.metric_reps(d, w, "mae")
    assert np.quantile(ra - rb, 0.95) == pytest.approx(res["diff"]["upper_one_sided"])
    same = cva.paired(s, s, w, "mae")      # identical predictions: every paired replicate is exactly zero
    assert same["diff"]["ci"] == [0.0, 0.0] and same["diff"]["upper_one_sided"] == 0.0


def test_host_draws_keep_all_hosts_weight():
    w = cva.host_draws(818, n_boot=20, seed=c.BOOT_SEED)
    assert w.shape == (20, 818) and (w.sum(axis=1) == 818).all()
    assert np.array_equal(w, cva.host_draws(818, n_boot=20, seed=c.BOOT_SEED))


def test_hierarchical_draws_pair_S_and_D():
    df = _precision_long(err_s=0.0, err_d=0.0)
    rng = np.random.default_rng(9)
    df = df.assign(y_pred=df.y_pred + rng.normal(0, 0.2, len(df)) * (df.model == "S"))
    df.loc[df.model == "D", "y_pred"] = df.loc[df.model == "S", "y_pred"].to_numpy()
    res = pa.analyse(df, 3, 2, HierDraws.make([6] * 3, 100, seed=4))
    assert res["654"]["A"]["ci"] == [0.0, 0.0]


# ---------------------------------------------------------------- integrity and statement

def test_check_run_flags():
    pred = pd.DataFrame({"site_id": ["a", "b"], "host_id": ["H", "H"], "y_true": [1.0, 2.0], "y_pred": [1.0, 2.0]})
    et.check_run(pred, {"a", "b"}, {"H"}, ["X"], "ok")
    with pytest.raises(AssertionError):
        et.check_run(pred, {"a", "b"}, {"H"}, ["H"], "leak")
    with pytest.raises(AssertionError):
        et.check_run(pred.assign(y_pred=[1.0, np.nan]), {"a", "b"}, {"H"}, ["X"], "nan")
    with pytest.raises(AssertionError):
        et.check_run(pred.iloc[:1], {"a", "b"}, {"H"}, ["X"], "missing")


def test_statement_names_failing_points():
    s = cva.statement({"654": 0.01, "k10": 0.06, "k20": 0.04})
    assert not s["holds"] and s["failing_points"] == ["k10"] and s["passing_points"] == ["654", "k20"]
    assert cva.statement({"654": 0.01, "k10": 0.02, "k20": 0.049})["holds"]
    assert not cva.statement({"654": 0.05, "k10": 0.0, "k20": 0.0})["holds"]   # strict


def test_gpu_table_pairs_only_same_class_cells():
    rows = []
    for seed, (gs, gd) in enumerate((("A10", "A10"), ("A10", "L4"), ("L4", "L4"))):
        common = {"backbone": "v2", "design": "cv", "K": 10, "fold": 0, "seed": seed, "B": 736, "r": -1,
                  "source": "cv"}
        rows.append({**common, "model": "S", "mae": 0.30, "gpu_class": gs, "capped": False})
        rows.append({**common, "model": "D", "mae": 0.20, "gpu_class": gd, "capped": seed == 0})
    runs = pd.DataFrame(rows)
    tab = {r["gpu_class"]: r for r in c.gpu_table(runs, "cv", 10, n_boot=50)}
    assert tab["A10"]["n_paired_cells"] == 1 and tab["L4"]["n_paired_cells"] == 1
    assert tab["A10"]["A_paired_cells"]["A"] == pytest.approx(0.1)
    assert tab["A10"]["n_runs"] == {"S": 2, "D": 1}
    cap = {(r["backbone"], r["model"]): r for r in c.capped_table(runs, "cv", 10)}
    assert cap[("v2", "D")]["n_capped"] == 1 and cap[("v2", "S")]["n_capped"] == 0


def test_desc_index_unique():
    idx = {c.desc_index(bb, d, p, g) for bb in c.BACKBONES for d, p in c.DESIGN_POINTS for g in c.GPU_ORDER}
    assert len(idx) == len(c.BACKBONES) * len(c.DESIGN_POINTS) * len(c.GPU_ORDER)
