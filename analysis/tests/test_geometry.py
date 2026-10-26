"""Synthetic checks with known answers for the geometry analysis (eval table, ΔMAE, pairing, regression, null
rule)."""
import numpy as np
import pandas as pd
import pytest

from analysis import geometry_analysis as ga
from analysis import geometry_eval_table as et


def _sites(n_hosts, sites_per_host=2, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for h in range(n_hosts):
        for s in range(sites_per_host):
            rows.append((f"h{h:03d}_s{s}", f"h{h:03d}", float(rng.normal(4.0, 1.0))))
    return pd.DataFrame(rows, columns=["site_id", "host_id", "y_true"])


def _long(shift, host_err=0.0, n_r=4, n_hosts=10, budgets=(654, 200), seeds=(0, 1, 2), models=ga.MODELS):
    """Condition (i) errs by +0.1 + a host-specific term; condition c adds shift[c] on every site (same sign), so
    ΔMAE(c) = shift[c] exactly and the host term cancels in the paired difference."""
    base = _sites(n_hosts)
    rng = np.random.default_rng(3)
    herr = {h: float(rng.uniform(0, host_err)) for h in base.host_id.unique()}
    he = base.host_id.map(herr).to_numpy()
    rows = []
    for r in range(n_r):
        for B in budgets:
            for seed in seeds:
                for m in models:
                    for c in ga.CONDITIONS:
                        e = 0.1 + he + shift.get(c, 0.0)
                        rows.append(base.assign(model=m, r=r, B=B, seed=seed, condition=c, y_pred=base.y_true + e,
                                                run_id=f"{m}{r}{B}{seed}", gpu="A10"))
    return pd.concat(rows, ignore_index=True)


def _hosts(long, seed=0):
    rng = np.random.default_rng(seed)
    hs = sorted(long.host_id.unique())
    return pd.DataFrame({"host_id": hs, "eps_v": rng.normal(0.01, 0.005, len(hs)),
                         "deviatoric": rng.uniform(0, 0.02, len(hs)), "internal_rmsd_A": rng.uniform(0, 0.05, len(hs)),
                         "d_bin": ["d0" if i % 2 else "d10" for i in range(len(hs))], "converged": True})


# ---------------------------------------------------------------- ΔMAE

def test_known_delta_and_pairing():
    long = _long({"mlip": 0.03, "mlip_rescaled": 0.01}, host_err=0.5)
    ens = ga.ensemble(long)
    draws, n_hosts = ga.make_draws(ens, n_resamples=4, n_boot=300)
    assert n_hosts == [10, 10, 10, 10]
    delta, _ = ga.delta_block(ens, draws)
    for B in ("654", "200"):
        for m in ga.MODELS:
            d = delta[B]["delta_mae"][m]
            assert d["mlip"]["mean"] == pytest.approx(0.03, abs=1e-12)
            assert d["mlip_rescaled"]["mean"] == pytest.approx(0.01, abs=1e-12)
            # paired: the host term cancels, so the interval collapses onto the shift
            assert d["mlip"]["ci95"] == pytest.approx([0.03, 0.03], abs=1e-12)
            assert not d["mlip"]["includes_zero"]
            # the marginal MAE carries the host term and has a wide interval
            mae = delta[B]["mae"][m]["dft"]
            assert mae["ci95"][1] - mae["ci95"][0] > 0.05


def test_zero_shift_includes_zero_and_tost():
    long = _long({}, host_err=0.2)
    ens = ga.ensemble(long)
    draws, _ = ga.make_draws(ens, n_resamples=4, n_boot=200)
    delta, keep = ga.delta_block(ens, draws)
    assert all(delta["654"]["delta_mae"][m][c]["includes_zero"] for m in ga.MODELS for c in ga.CHANGED)
    t = ga.tost_block(keep)
    assert t["by"]["654_mlip"]["outcome"] == "equivalent"


def test_seed_ensemble_is_the_mean():
    long = _long({"mlip": 0.0}, n_r=1, budgets=(654,))
    long.loc[long.seed == 0, "y_pred"] += 0.3
    long.loc[long.seed == 1, "y_pred"] -= 0.3
    ens = ga.ensemble(long)
    first = long[(long.seed == 2)].set_index(["model", "r", "B", "condition", "site_id"]).y_pred
    got = ens.set_index(["model", "r", "B", "condition", "site_id"]).y_pred
    assert np.allclose(got.loc[first.index].to_numpy(), first.to_numpy())


def test_draws_require_equal_host_counts():
    long = _long({}, n_r=2)
    long = long[~((long.model == "P") & (long.r == 0) & (long.host_id == "h000"))]
    with pytest.raises(AssertionError):
        ga.make_draws(ga.ensemble(long), n_resamples=2, n_boot=10)


# ---------------------------------------------------------------- per-host change and regression

def test_host_change_known_values():
    long = _long({"mlip": 0.04, "mlip_rescaled": -0.02}, n_r=2)
    ch = ga.host_change(ga.ensemble(long), _hosts(long))
    assert np.allclose(ch[ch.condition == "mlip"].mean_abs_change, 0.04)
    assert np.allclose(ch[ch.condition == "mlip_rescaled"].mean_abs_change, 0.02)
    assert (ch.n_rows == 2 * 2).all()        # 2 resamples x 2 sites per host


def test_regression_recovers_planted_coefficient():
    rng = np.random.default_rng(5)
    n = 400
    X = np.column_stack([rng.normal(0.01, 0.005, n), rng.uniform(0, 0.02, n), rng.uniform(0, 0.05, n)])
    beta_raw = np.array([0.0, 0.0, 0.8])        # eV per Å of internal RMSD
    y = 0.002 + X @ beta_raw + rng.normal(0, 0.001, n)
    idx = ga.boot_index(n, 300)
    res, _, _ = ga.fit_set(y, X, idx)
    want = 0.8 * X[:, 2].std(ddof=1)            # eV per SD
    got = res["joint"]["coef_per_sd"]["internal_rmsd_A"]
    assert got["coef"] == pytest.approx(want, rel=0.03)
    assert got["ci95"][0] < want < got["ci95"][1]
    for c in ("eps_v", "deviatoric"):
        ci = res["joint"]["coef_per_sd"][c]["ci95"]
        assert ci[0] < 0 < ci[1]
    assert res["joint"]["r2"] > 0.9
    assert res["only_internal_rmsd_A"]["coef_per_sd"]["internal_rmsd_A"]["coef"] == pytest.approx(want, rel=0.05)


def test_boot_index_shared_and_seeded():
    assert np.array_equal(ga.boot_index(50, 20), ga.boot_index(50, 20))
    assert ga.boot_index(50, 20).max() < 50


def test_small_stratum_gets_no_fit():
    long = _long({"mlip": 0.01}, n_r=2, n_hosts=30)
    ens = ga.ensemble(long)
    ch = ga.host_change(ens, _hosts(long))
    hs = sorted(ch.host_id.unique())
    out, _ = ga.regression_block(ch, {"pooled": hs, "tiny": hs[:5]}, n_boot=50)
    assert "fit" in out["tiny"] and out["tiny"]["n_hosts"] == 5
    assert "S_654" in out["pooled"]


def test_slope_difference_zero_when_models_equal():
    long = _long({"mlip": 0.01}, n_r=2, n_hosts=30, host_err=0.0)
    ens = ga.ensemble(long)
    hosts = _hosts(long)
    ch = ga.host_change(ens, hosts)
    # plant a host-dependent change identical for every model
    ch["mean_abs_change"] = 0.3 * ch.internal_rmsd_A + 10.0 * ch.deviatoric ** 2
    _, keep = ga.regression_block(ch, {"pooled": sorted(ch.host_id.unique())}, n_boot=100)
    s = ga.slope_block(keep)
    for v in s["by"].values():
        assert v["mean"] == pytest.approx(0.0, abs=1e-12)
        assert v["outcome"] == "equivalent"


# ---------------------------------------------------------------- null rule and TOST outcome

def _delta(includes):
    return {"654": {"delta_mae": {m: {c: {"includes_zero": includes.get((m, c), True)} for c in ga.CHANGED}
                                  for m in ga.MODELS}}}


def test_null_rule_both_hold():
    nr = ga.null_rule(0.005, _delta({}))
    assert nr["outcome"] == ga.NULL_STRING
    assert nr["cond1_median_rmsd_below"] and nr["cond2_all_intervals_include_zero"]


def test_null_rule_only_intervals():
    nr = ga.null_rule(0.02, _delta({}))
    assert nr["outcome"].startswith("null rule not met")
    assert not nr["cond1_median_rmsd_below"] and nr["cond2_all_intervals_include_zero"]
    assert nr["median_minus_threshold_A"] == pytest.approx(0.01)
    assert "(2)" in nr["outcome"] and "(1)" not in nr["outcome"]


def test_null_rule_only_rmsd_and_neither():
    nr = ga.null_rule(0.001, _delta({("P", "mlip"): False}))
    assert nr["intervals_excluding_zero"] == ["P_mlip"]
    assert "(1)" in nr["outcome"] and "(2)" not in nr["outcome"]
    nr = ga.null_rule(0.01, _delta({("S", "mlip_rescaled"): False}))     # strict: 0.01 is not below 0.01
    assert nr["outcome"].endswith("neither")


@pytest.mark.parametrize("ci95,ci90,want", [
    ([-0.03, -0.01], [-0.028, -0.012], "P better"),
    ([0.01, 0.2], [0.02, 0.15], "P worse"),
    ([-0.01, 0.01], [-0.008, 0.008], "equivalent"),
    ([-0.1, 0.1], [-0.08, 0.08], "inconclusive"),
])
def test_tost_outcome(ci95, ci90, want):
    assert ga.tost_outcome(ci95, ci90)["outcome"] == want


def test_tost_within_margin_wording():
    o = ga.tost_outcome([0.001, 0.004], [0.0015, 0.0035])
    assert o["outcome"] == "P worse" and o["within_equivalence_margin"]


# ---------------------------------------------------------------- distributions and comparability

def test_dist_values():
    d = ga.dist(np.arange(1, 101, dtype=float))
    assert d["median"] == pytest.approx(50.5) and d["max"] == 100 and d["p95"] == pytest.approx(95.05)


def test_comparable_cells():
    from pymatgen.core import Lattice, Structure

    s = Structure(Lattice.cubic(4.0), ["Zn", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    assert ga.comparable(s, Structure(Lattice.cubic(4.1), ["Zn", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])) == ""
    assert ga.comparable(s, Structure(Lattice.cubic(4.0), ["O", "Zn"], [[0, 0, 0], [0.5, 0.5, 0.5]])) != ""
    assert ga.comparable(s, Structure(Lattice.cubic(5.0), ["Zn", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])) != ""


# ---------------------------------------------------------------- eval-table integrity

def _pred(sites, conds=et.CONDITIONS):
    return pd.concat([sites.assign(condition=c, prediction=1.0) for c in conds], ignore_index=True)


def test_check_run_passes_and_fails():
    s = _sites(4)
    test_sites = set(s.site_id)
    et.check_run(_pred(s), test_sites, set(), "ok")
    with pytest.raises(AssertionError):
        et.check_run(_pred(s, conds=("dft", "mlip")), test_sites, set(), "missing condition")
    bad = _pred(s)
    bad.loc[0, "prediction"] = np.nan
    with pytest.raises(AssertionError):
        et.check_run(bad, test_sites, set(), "nan")
    with pytest.raises(AssertionError):
        et.check_run(_pred(s), test_sites, {"h001"}, "excluded present")
    with pytest.raises(AssertionError):
        et.check_run(_pred(s.iloc[1:]), test_sites, set(), "site missing")
    dup = pd.concat([_pred(s), _pred(s.iloc[:1])], ignore_index=True)
    with pytest.raises(AssertionError):
        et.check_run(dup, test_sites, set(), "duplicate")


def test_vacancy_species():
    sp = {"h": ["Zn", "O", "O"]}
    assert et.check_vacancy_species([("h", 1), ("h", 2)], sp, "x") == 2
    with pytest.raises(AssertionError):
        et.check_vacancy_species([("h", 0)], sp, "x")


def test_check_cells_and_specs():
    recs = []
    for m in et.MODELS:
        for r in range(et.N_RESAMPLES):
            for B in et.BUDGETS:
                for s in et.SEEDS:
                    comps = {k: "x" for k in et.EXPECTED_COMPONENTS[m]}
                    recs.append({"run_id": f"{m}{r}{B}{s}", "key": {"model": m, "r": r, "budget": B, "seed": s,
                                                                   "components": comps},
                                 "spec": {"r": r}})
    counts = et.check_cells(recs)
    assert set(counts.values()) == {30}
    specs = [{"run_id": p["run_id"], "key": p["key"], "spec": p["spec"]} for p in recs]
    assert et.check_specs(recs, specs) == 180
    with pytest.raises(AssertionError):
        et.check_cells(recs[1:])
    specs[0] = {**specs[0], "spec": {"r": 99}}
    with pytest.raises(AssertionError):
        et.check_specs(recs, specs)


def test_tiling_split():
    t = pd.DataFrame({"point_op": [None, "", "[[-1,0,0],[0,-1,0],[0,0,-1]]"], "status": "ok",
                      "excluded_from_C3b": False})
    assert et.tiling_split(t) == {"n_hosts": 3, "translation_only": 2, "point_operation": 1, "status_ok": 3,
                                  "excluded_from_C3b": 0}
