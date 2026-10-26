"""Synthetic checks with known answers for the sensitivity analyses (high moment, CGCNN, capped twins)."""
import numpy as np
import pandas as pd
import pytest

from analysis import sensitivity_analysis as sa
from analysis import sensitivity_eval_tables as et
from dftgnn.stats.g1 import HierDraws, point_metric


def _sites(n_hosts=6, per_host=2, seed=0):
    rng = np.random.default_rng(seed)
    rows = [(f"h{h}_s{s}", f"h{h}", float(rng.normal(4.0, 1.0))) for h in range(n_hosts) for s in range(per_host)]
    return pd.DataFrame(rows, columns=["site_id", "host_id", "y_true"])


def _wide(n_r=3, budgets=(400, 654), err=None):
    """G1-style eval table: per (model, r, B, site) three seed columns; S errs +0.3, D -0.1 on every seed."""
    err = err or {"S": 0.3, "D": -0.1}
    base = _sites()
    rows = []
    for r in range(n_r):
        for B in budgets:
            for m, e in err.items():
                f = base.assign(model=m, r=r, B=B)
                for s in range(3):
                    f[f"y_pred_seed{s}"] = f.y_true + e
                rows.append(f)
    return sa.ensemble_from_seeds(pd.concat(rows, ignore_index=True))


# ---------------------------------------------------------------- common-site comparison

def test_common_sites_drop_excluded_and_check_runs():
    prim = _wide(n_r=2, budgets=(654,))
    excluded = {"h0_s0", "h0_s1", "h3_s1"}            # h0 loses every site
    sens = pd.concat([prim[(prim.model == "S") & ~prim.site_id.isin(excluded)].assign(run_id=f"run{k}")
                      for k in range(2)])
    common = sa.common_sites(prim[prim.model == "S"], sens, excluded)
    assert common[0] == set(prim.site_id[(prim.r == 0) & (prim.model == "S")]) - excluded
    bad = pd.concat([sens, sens.iloc[:1].assign(site_id="h0_s0")])
    with pytest.raises(AssertionError):
        sa.common_sites(prim[prim.model == "S"], bad, excluded)


def test_padded_host_rows_change_nothing_but_shape():
    f = _sites().assign(y_pred=lambda d: d.y_true + 0.2)
    hosts = [f"h{h}" for h in range(6)] + ["h_missing"]
    st = sa.padded_stats(f, sorted(hosts))
    assert st.shape == (7, 9)
    assert not st[sorted(hosts).index("h_missing")].any()
    assert point_metric(st, "mae") == pytest.approx(0.2)
    with pytest.raises(AssertionError):
        sa.padded_stats(f, ["h0"])


def test_common_site_metric_ignores_removed_sites():
    """Removing sites with a large error lowers the MAE to that of the remaining sites, on both versions."""
    f = _sites().assign(y_pred=lambda d: d.y_true + 0.1)
    f.loc[f.site_id == "h2_s0", "y_pred"] += 5.0
    hosts = sorted(f.host_id.unique())
    keep = {0: set(f.site_id) - {"h2_s0"}}
    st = sa.stats_by_r(f.assign(model="S", r=0), "S", {0: hosts}, keep)
    assert point_metric(st[0], "mae") == pytest.approx(0.1)


# ---------------------------------------------------------------- twin replacement

def test_replace_twins_changes_only_its_seed_column():
    wide = _wide(n_r=2)
    cell = wide[(wide.model == "S") & (wide.r == 1) & (wide.B == 654)]
    twin = cell[["site_id", "host_id", "y_true"]].assign(model="S", r=1, B=654, seed=2, run_id="t",
                                                         y_pred=cell.y_true + 0.6)
    out, done = sa.replace_twins(wide, twin)
    got = out[(out.model == "S") & (out.r == 1) & (out.B == 654)]
    assert np.allclose(got.y_pred_seed2 - got.y_true, 0.6)
    assert np.allclose(got.y_pred - got.y_true, (0.3 + 0.3 + 0.6) / 3)
    rest = out.drop(got.index)
    assert np.allclose(rest.y_pred, wide.drop(got.index).y_pred)
    assert done == [{"model": "S", "r": 1, "B": 654, "seed": 2, "run_id": "t"}]
    with pytest.raises(AssertionError):
        sa.replace_twins(wide, twin.iloc[1:])


def test_twin_spec_check():
    o = {"run_id": "o", "max_epochs": 200, "spec": {"model": "S", "split": "outer_r0", "r": 0, "budget": 654,
                                                     "seed": 0, "hp": {"a": 1}, "lr": 1e-3, "weight_decay": 0.0,
                                                     "batch_size": 16}}
    t = {"run_id": "t", "max_epochs": 600, "patience": 30, "spec": dict(o["spec"])}
    et.check_twin_spec(o, t)
    with pytest.raises(AssertionError):
        et.check_twin_spec(o, {**t, "spec": {**t["spec"], "seed": 1}})
    with pytest.raises(AssertionError):
        et.check_twin_spec(o, {**t, "max_epochs": 200})


def test_twin_map_needs_every_original():
    queue = {"jobs": [{"original_run_id": "a", "stopped_by_cap": True},
                      {"original_run_id": "0b2035de2e44fade", "stopped_by_cap": True},
                      {"original_run_id": "373c897db2d3b822", "stopped_by_cap": True},
                      {"original_run_id": "d06182b6b418fb01", "stopped_by_cap": True}]}
    twins = [{"run_id": "ta", "spec": {"tags": {"original_run_id": "a"}}}]
    m = et.twin_map(queue, twins)
    assert m["a"] == "ta" and m["0b2035de2e44fade"] == "70471916fc229b83"
    with pytest.raises(AssertionError):
        et.twin_map({"jobs": queue["jobs"] + [{"original_run_id": "b", "stopped_by_cap": True}]}, twins)


# ---------------------------------------------------------------- paired difference and bootstrap pairing

def test_paired_difference_known_values():
    wide = _wide(n_r=3, budgets=(654,))
    alt = _wide(n_r=3, budgets=(654,), err={"S": 0.25, "D": -0.1})
    hosts = {r: [f"h{h}" for h in range(6)] for r in range(3)}
    draws = HierDraws.make([6] * 3, 300, seed=3)
    _, base = sa.hier_pair(sa.stats_by_r(wide, "S", hosts), sa.stats_by_r(wide, "D", hosts), draws)
    out, new = sa.hier_pair(sa.stats_by_r(alt, "S", hosts), sa.stats_by_r(alt, "D", hosts), draws)
    ch = sa.changes(new, base)
    assert out["A"]["mean"] == pytest.approx(0.15)
    assert ch["A"]["mean"] == pytest.approx(-0.05)
    assert ch["A"]["ci"] == pytest.approx([-0.05, -0.05])      # constant shifts: paired draws cancel exactly
    assert ch["MAE_D"]["ci"] == pytest.approx([0.0, 0.0])


def test_bootstrap_pairing_identical_versions_give_zero_width():
    rng = np.random.default_rng(0)
    wide = _wide(n_r=3, budgets=(654,))
    for s in range(3):
        wide[f"y_pred_seed{s}"] += rng.normal(0, 0.2, len(wide))
    wide = sa.ensemble_from_seeds(wide)
    hosts = {r: [f"h{h}" for h in range(6)] for r in range(3)}
    draws = HierDraws.make([6] * 3, 300, seed=5)
    a, ra = sa.hier_pair(sa.stats_by_r(wide, "S", hosts), sa.stats_by_r(wide, "D", hosts), draws)
    _, rb = sa.hier_pair(sa.stats_by_r(wide, "S", hosts), sa.stats_by_r(wide, "D", hosts), draws)
    ch = sa.changes(rb, ra)
    assert ch["A"]["ci"] == [0.0, 0.0] and ch["A"]["mean"] == 0.0
    assert a["A"]["ci"][0] < a["A"]["ci"][1]                    # the marginal interval is not degenerate


def test_cluster_bootstrap_pairs_models():
    f = _sites(n_hosts=20).assign(y_pred=lambda d: d.y_true + np.random.default_rng(1).normal(0, 0.3, len(d)))
    hosts = sorted(f.host_id.unique())
    w = sa.cluster_weights(len(hosts), 500, seed=9)
    s1 = sa.padded_stats(f, hosts)
    s2 = sa.padded_stats(f.assign(y_pred=f.y_pred + 0.0), hosts)
    a, ra = sa.cluster_summary(s1, w, "mae")
    b, rb = sa.cluster_summary(s2, w, "mae")
    d = sa.diff_summary(a["point"], b["point"], ra, rb)
    assert d["point"] == 0.0 and d["ci95"] == [0.0, 0.0]
    assert a["ci95"][0] < a["point"] < a["ci95"][1]
    assert w.sum(axis=1).tolist() == [len(hosts)] * 500


def test_excluded_sites_threshold():
    uni = pd.DataFrame({"site_id": [f"s{i}" for i in range(12)], "host_id": "h",
                        "defect_moment_muB": [2.0] * 9 + [-1.9, 0.5, 0.0]})
    assert et.excluded_sites(uni, 0.5) == sorted(f"s{i}" for i in range(10))
    with pytest.raises(AssertionError):
        et.excluded_sites(uni.iloc[1:], 0.5)


def test_moment_counts_exclude_entries():
    sites_of = {"h0": {"a", "b"}, "h1": {"c"}, "h2": {"d", "e"}}
    row = {"run_id": "x", "n_train_sites": 2, "n_val_sites": 1, "n_train_hosts": 1, "n_val_hosts": 1}
    et.check_moment_counts(row, ["h0", "h1", "h2"], sites_of, {"c", "d"})    # h1 dropped, h0 2 + h2 1
    with pytest.raises(AssertionError):
        et.check_moment_counts({**row, "n_val_sites": 2}, ["h0", "h1", "h2"], sites_of, {"c", "d"})
