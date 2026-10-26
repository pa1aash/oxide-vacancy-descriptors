"""Synthetic checks with known answers for analysis/within_host_breakdown.py."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from analysis import within_host_breakdown as wb


def _sites(rows):
    return pd.DataFrame(rows, columns=["site_id", "host_id", "y_true", "y_pred"])


# host A: y 1, 2, 3 with constant prediction 1; host B: one site, error 1; host C: perfect, two sites
SITES = _sites([("a1", "A", 1.0, 1.0), ("a2", "A", 2.0, 1.0), ("a3", "A", 3.0, 1.0),
                ("b1", "B", 5.0, 6.0),
                ("c1", "C", 4.0, 4.0), ("c2", "C", 6.0, 6.0)])


def test_seed_ensemble_and_resample_average():
    rows = []
    for r, preds in ((0, (1.0, 2.0, 3.0)), (3, (5.0, 5.0, 5.0))):
        for seed, p in enumerate(preds):
            rows.append(("S", r, seed, "x1", "X", 7.0, p))
    df = pd.DataFrame(rows, columns=["model", "r", "seed", "site_id", "host_id", "y_true", "y_pred"])
    ens = wb.seed_ensemble(df)
    assert ens.y_pred.tolist() == [2.0, 5.0]
    assert (ens.n_seeds == 3).all()
    avg = wb.average_over_resamples(ens)
    assert avg.y_pred.item() == 3.5 and avg.n_resamples.item() == 2


def test_seed_ensemble_rejects_inconsistent_targets():
    df = pd.DataFrame([("S", 0, 0, "x", "X", 1.0, 0.0), ("S", 0, 1, "x", "X", 2.0, 0.0)],
                      columns=["model", "r", "seed", "site_id", "host_id", "y_true", "y_pred"])
    with pytest.raises(ValueError):
        wb.seed_ensemble(df)


def test_group_mae_within_host_and_zero_skill():
    t = wb.site_terms(SITES)
    s = wb.host_sums(t, np.ones(len(t), bool), ["A", "B", "C"])
    m = wb.metrics_from_sums(s.sum(axis=0))
    assert m["mae"] == pytest.approx((0 + 1 + 2 + 1 + 0 + 0) / 6)
    # within-host residuals: A |0-(-1)|, |0-0|, |0-1| -> 1, 0, 1; C -> 0, 0; B excluded
    assert m["within_host_mae"] == pytest.approx(2 / 5)
    # zero skill: |y - host mean| over A and C: 1, 0, 1, 1, 1
    assert m["zero_skill_mae"] == pytest.approx(4 / 5)
    assert m["signed_residual"] == pytest.approx(0.0)


def test_host_means_use_all_tested_sites_when_group_is_partial():
    t = wb.site_terms(SITES)
    mask = SITES.site_id.isin(["a3"]).to_numpy()
    m = wb.metrics_from_sums(wb.host_sums(t, mask, ["A"]).sum(axis=0))
    # a3: dp = 0, dy = 3 - 2 = 1 -> residual -1, zero skill 1
    assert m["within_host_mae"] == pytest.approx(1.0)
    assert m["zero_skill_mae"] == pytest.approx(1.0)
    assert m["signed_residual"] == pytest.approx(-1.0)


def test_constant_within_host_predictor_equals_zero_skill():
    rng = np.random.default_rng(1)
    d = pd.DataFrame({"site_id": [f"s{i}" for i in range(40)], "host_id": np.repeat(list("ABCDEFGHIJ"), 4),
                      "y_true": rng.normal(size=40)})
    d["y_pred"] = d.groupby("host_id").y_true.transform("mean") + np.repeat(rng.normal(size=10), 4)
    m = wb.metrics_from_sums(wb.host_sums(wb.site_terms(d), np.ones(40, bool), sorted(set(d.host_id))).sum(0))
    assert m["within_host_mae"] == pytest.approx(m["zero_skill_mae"])


def test_spearman_known_values_and_undefined():
    assert wb.spearman(np.array([1.0, 2, 3]), np.array([3.0, 2, 1])) == pytest.approx(-1.0)
    assert np.isnan(wb.spearman(np.array([1.0, 1, 1]), np.array([1.0, 2, 3])))
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=9), rng.normal(size=9)
    x[3] = x[5]  # a tie
    assert wb.spearman(x, y) == pytest.approx(spearmanr(x, y).statistic)


def test_host_spearman_needs_three_sites():
    rho = wb.host_spearman(SITES, "y_pred")
    assert list(rho.index) == ["A"] and np.isnan(rho["A"])  # constant prediction in A
    d = SITES.copy()
    d.loc[d.host_id == "A", "y_pred"] = [3.0, 2.0, 1.0]
    assert wb.host_spearman(d, "y_pred")["A"] == pytest.approx(-1.0)


def test_tercile_and_bader_rank_labels():
    v = pd.Series(np.arange(1.0, 10.0), index=[f"h{i}" for i in range(9)])
    lab = wb.tercile_labels(v)
    assert lab.value_counts().to_dict() == {"lower": 3, "middle": 3, "upper": 3}
    s = pd.DataFrame({"host_id": ["A", "A", "A", "B", "B", "C", "D", "D"],
                      "bader_charge": [-1.2, -1.0, -1.1, -1.0, -1.0, -1.3, -1.4, -1.2]})
    r = wb.bader_rank_labels(s)
    assert r.tolist()[:3] == ["lowest", "highest", "middle"]
    assert r.iloc[3:6].isna().all()  # B constant, C single site
    assert r.iloc[6:].tolist() == ["lowest", "highest"]


def test_bootstrap_draws_reproducible_and_group_specific():
    a, b, c = wb.draw_weights(12, 3, n_boot=50), wb.draw_weights(12, 3, n_boot=50), wb.draw_weights(12, 4, n_boot=50)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert (a.sum(axis=1) == 12).all()


def test_bootstrap_pairing_cancels_host_constant_offsets():
    # D = S + a per-host constant: identical within-host residuals, so the paired S - D interval is [0, 0]
    rng = np.random.default_rng(2)
    hosts = [f"h{i:02d}" for i in range(15)]
    d = pd.DataFrame({"site_id": [f"s{i}" for i in range(45)], "host_id": np.repeat(hosts, 3),
                      "y_true": rng.normal(size=45)})
    s_model = d.assign(y_pred=d.y_true + rng.normal(scale=0.3, size=45))
    d_model = s_model.assign(y_pred=s_model.y_pred + np.repeat(rng.normal(size=15), 3))
    mask = np.ones(45, bool)
    sums = {("v2", "S"): wb.host_sums(wb.site_terms(s_model), mask, hosts),
            ("v2", "D"): wb.host_sums(wb.site_terms(d_model), mask, hosts)}
    w = wb.draw_weights(len(hosts), 0, n_boot=200)
    out = wb.summarise_group(sums, w, ("mae", "within_host_mae", "zero_skill_mae"))
    diff = out["differences"]["S_minus_D_v2"]["within_host_mae"]
    assert diff["point"] == pytest.approx(0.0, abs=1e-12)
    assert diff["ci95"] == pytest.approx([0.0, 0.0], abs=1e-12)
    # unpaired draws would not cancel: the marginal intervals are wide
    lo, hi = out["v2"]["S"]["within_host_mae"]["ci95"]
    assert hi - lo > 0.01
    # the site-MAE difference is not zero and its interval is reported
    assert "ci95" in out["differences"]["S_minus_D_v2"]["mae"]


def test_small_groups_get_no_interval():
    t = wb.site_terms(SITES)
    hosts = ["A", "B", "C"]
    sums = {("v1", "S"): wb.host_sums(t, np.ones(6, bool), hosts),
            ("v1", "D"): wb.host_sums(t, np.ones(6, bool), hosts)}
    out = wb.summarise_group(sums, wb.draw_weights(3, 0, n_boot=20), ("mae", "within_host_mae"))
    assert "ci95" not in out["v1"]["S"]["mae"]
    assert out["n_hosts"] == 3 and out["n_sites"] == 6 and out["n_sites_within"] == 5
    assert out["n_hosts_by_metric"]["within_host_mae"] == 2


def test_spearman_summary_paired_difference():
    hosts = [f"h{i:02d}" for i in range(12)]
    rng = np.random.default_rng(3)
    rs = pd.Series(rng.uniform(-1, 1, 12), index=hosts)
    rho = {("v2", "S"): rs, ("v2", "D"): rs - 0.1, ("v1", "S"): rs, ("v1", "D"): rs}
    out = wb.summarise_spearman(rho, rs, wb.draw_weights(12, 0, n_boot=100), hosts)
    assert out["differences"]["S_minus_D_v2"]["mean"] == pytest.approx(0.1)
    assert out["differences"]["S_minus_D_v2"]["ci95"] == pytest.approx([0.1, 0.1])
    assert out["v2"]["S"]["n_hosts"] == 12


def test_build_groups_partition():
    sites = pd.DataFrame({"site_id": [f"s{i}" for i in range(9)],
                          "host_id": ["A", "A", "B", "C", "C", "C", "D", "E", "F"],
                          "bader_charge": [-1.0, -1.2, -1.1, -1.0, -1.1, -1.3, -1.0, -1.0, -1.0]})
    hosts = pd.DataFrame({"dclass": ["d0", "d10", "other", "d0", "unassigned", "d10"],
                          "cations": [["Ti"], ["Zn"], ["Mg"], ["Nb", "Na"], ["Au"], ["Zn", "Ti"]],
                          "formation_energy": [-3.0, -2.0, -1.0, -2.5, -1.5, -0.5],
                          "band_gap": [3.0, 2.0, 5.0, 1.0, 0.5, 4.0]},
                         index=pd.Index(list("ABCDEF"), name="host_id"))
    groups = wb.build_groups(sites, hosts)
    by = {}
    for g in groups:
        by.setdefault(g["variable"], 0)
        by[g["variable"]] += int(g["mask"].sum())
    for v in ("d_class", "phs_prone", "stability_tercile", "band_gap_tercile"):
        assert by[v] == len(sites)
    assert by["bader_rank"] == 5  # A (2) and C (3); B, D, E, F single-site
    phs = {g["group"]: g["mask"] for g in groups if g["variable"] == "phs_prone"}
    assert sites.host_id[phs["phs_prone"]].unique().tolist() == ["A", "D", "F"]
    assert [g["k"] for g in groups] == list(range(len(groups)))
    assert not any(g["variable"] == "cation" for g in groups)  # no element in >= 15 hosts
