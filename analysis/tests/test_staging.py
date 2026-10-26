"""Synthetic checks with known answers for the staging analysis (eval table, contrasts, TOST, pairing)."""
import numpy as np
import pandas as pd
import pytest

from analysis import staging_analysis as sa
from analysis import staging_eval_table as et
from analysis import staging_swapin_check as sw


def _hosts(n_hosts, sites_per_host=2, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for h in range(n_hosts):
        for s in range(sites_per_host):
            rows.append((f"h{h:03d}_s{s}", f"h{h:03d}", float(rng.normal(4.0, 1.0))))
    return pd.DataFrame(rows, columns=["site_id", "host_id", "y_true"])


def _long(errors, n_r=4, n_seeds=3, n_hosts=8, budgets=(200, 654), noise=0.0, backbone="v1"):
    """Every resample tests the same hosts; model m errs by +errors[m] on every site (|error| = MAE)."""
    base = _hosts(n_hosts)
    rng = np.random.default_rng(1)
    rows = []
    for r in range(n_r):
        for B in budgets:
            for seed in range(n_seeds):
                for m, e in errors.items():
                    jit = rng.normal(0.0, noise, len(base)) if noise else 0.0
                    rows.append(base.assign(backbone=backbone, model=m, r=r, B=B, seed=seed,
                                            y_pred=base.y_true + e + jit, run_id=f"{m}{r}{B}{seed}"))
    return pd.concat(rows, ignore_index=True)


def _block(errors, budgets=(200, 654), noise=0.0, **kw):
    ens = sa.ensemble(_long(errors, budgets=budgets, noise=noise, **kw))
    draws = sa.make_draws(ens, n_resamples=kw.get("n_r", 4), n_boot=200)
    block, a = sa.staging_block(ens, draws, budgets=budgets)
    return block, a


# ---------------------------------------------------------------- known contrasts

def test_known_contrasts_exact():
    block, a = _block({"S": 0.30, "D": 0.10, "P": 0.25})
    for B in ("200", "654"):
        v = block[B]
        assert v["mae"]["S"]["mean"] == pytest.approx(0.30)
        assert v["mae"]["D"]["mean"] == pytest.approx(0.10)
        assert v["mae"]["P"]["mean"] == pytest.approx(0.25)
        assert v["D_minus_P"]["mean"] == pytest.approx(-0.15)       # D - P: cost of predicting the descriptors
        assert v["P_minus_S"]["mean"] == pytest.approx(-0.05)       # P - S: negative = P better
        # constant errors: every bootstrap replicate equals the point value
        assert v["P_minus_S"]["ci95"] == pytest.approx([-0.05, -0.05])
        assert a[B]["mean"] == pytest.approx(0.20)                  # S - D for the G1 sanity check


def test_contrast_sign_and_bounds_ordered():
    block, _ = _block({"S": 0.30, "D": 0.20, "P": 0.40}, noise=0.05)
    ps = block["654"]["P_minus_S"]
    assert ps["mean"] > 0
    assert ps["ci95"][0] <= ps["ci90"][0] <= ps["mean"] <= ps["ci90"][1] <= ps["ci95"][1]
    assert ps["lower_one_sided_95"] == ps["ci90"][0] and ps["upper_one_sided_95"] == ps["ci90"][1]


# ---------------------------------------------------------------- TOST direction and order

@pytest.mark.parametrize("ci95, ci90, outcome, within", [
    ((-0.20, -0.10), (-0.18, -0.12), "P better", False),
    ((0.10, 0.20), (0.12, 0.18), "P worse", False),
    ((0.005, 0.045), (0.01, 0.04), "P worse", True),          # worse, but the 90% interval is inside the margin
    ((-0.045, -0.005), (-0.04, -0.01), "P better", True),
    ((-0.03, 0.04), (-0.02, 0.03), "equivalent", False),
    ((-0.03, 0.08), (-0.02, 0.06), "inconclusive", False),
    ((-0.03, 0.06), (-0.02, 0.05), "inconclusive", False),    # strict: an endpoint on the margin is not inside
])
def test_tost_outcome(ci95, ci90, outcome, within):
    o = sa.tost_outcome(ci95, ci90, 0.05)
    assert o["outcome"] == outcome
    assert o["within_equivalence_margin"] is within
    assert o["report"].startswith(outcome)
    assert ("different but within the equivalence margin" in o["report"]) is within


def test_noninferior_strict():
    assert sa.noninferior(0.049, 0.05) and not sa.noninferior(0.05, 0.05) and not sa.noninferior(0.07, 0.05)


def test_tost_direction_from_data():
    better, _ = _block({"S": 0.40, "D": 0.10, "P": 0.20}, noise=0.02)
    worse, _ = _block({"S": 0.20, "D": 0.10, "P": 0.40}, noise=0.02)
    equiv, _ = _block({"S": 0.30, "D": 0.10, "P": 0.30}, noise=0.02)
    assert better["654"]["P_minus_S"]["outcome"] == "P better"
    assert worse["654"]["P_minus_S"]["outcome"] == "P worse"
    assert equiv["654"]["P_minus_S"]["outcome"] == "equivalent"
    assert equiv["654"]["P_minus_S"]["noninferior"] and not worse["654"]["P_minus_S"]["noninferior"]
    s = sa.summary(better)
    assert s["any_budget_P_better"] and s["n_budgets_P_better"] == 2


def test_draws_require_same_test_hosts():
    long = _long({"S": 0.3, "D": 0.1, "P": 0.2})
    long = long[~((long.model == "P") & (long.r == 1) & (long.host_id == "h000"))]
    with pytest.raises(AssertionError):
        sa.make_draws(sa.ensemble(long), n_resamples=4, n_boot=10)


# ---------------------------------------------------------------- pairing and integrity

def _runs(backbone="v1"):
    rows = []
    for B in (25, 654):
        for s in (0, 1):
            cell = {"backbone": backbone, "split": "outer_r0", "r": 0, "B": B, "seed": s}
            rows += [{**cell, "model": "D", "run_id": f"d{B}{s}", "d_run": "", "p1_run": ""},
                     {**cell, "model": "P1", "run_id": f"q{B}{s}", "d_run": "", "p1_run": ""},
                     {**cell, "model": "P", "run_id": f"p{B}{s}", "d_run": f"d{B}{s}", "p1_run": f"q{B}{s}"}]
    return pd.DataFrame(rows)


def test_pairing_accepts_matched_cells():
    assert et.check_pairing(_runs()) == 4


def test_pairing_rejects_mismatched_seed():
    runs = _runs()
    runs.loc[runs.run_id == "p6540", "d_run"] = "d6541"      # D of seed 1 under the P of seed 0
    with pytest.raises(AssertionError, match="cell"):
        et.check_pairing(runs)


def test_pairing_rejects_wrong_budget_and_model():
    runs = _runs()
    runs.loc[runs.run_id == "p250", "p1_run"] = "q6540"
    with pytest.raises(AssertionError):
        et.check_pairing(runs)
    runs = _runs()
    runs.loc[runs.run_id == "p250", "p1_run"] = "d250"        # a D run in the P1 slot
    with pytest.raises(AssertionError, match="expected P1"):
        et.check_pairing(runs)


def test_pairing_rejects_missing_component():
    runs = _runs()
    runs.loc[runs.run_id == "p250", "d_run"] = "nope"
    with pytest.raises(AssertionError, match="not found"):
        et.check_pairing(runs)


def test_kiyohara_d_component():
    runs = pd.DataFrame([{"backbone": "v1", "model": "P", "split": "kiyohara", "r": -1, "B": 571, "seed": 1,
                          "run_id": "pk1", "d_run": "dk1", "p1_run": "qk1"}])
    assert et.check_kiyohara_d(runs, {"v1": {"dk1": 1}}) == 1
    with pytest.raises(AssertionError):
        et.check_kiyohara_d(runs, {"v1": {"dk1": 0}})


def test_check_run_and_specs():
    base = _hosts(3).assign(y_pred=1.0)
    sites, hosts = set(base.site_id), set(base.host_id)
    et.check_run(base, sites, hosts, ["x"], "ok")
    with pytest.raises(AssertionError, match="non-finite"):
        et.check_run(base.assign(y_pred=np.nan), sites, hosts, ["x"], "nan")
    with pytest.raises(AssertionError, match="duplicate"):
        et.check_run(pd.concat([base, base.iloc[:1]]), sites, hosts, ["x"], "dup")
    with pytest.raises(AssertionError, match="test set"):
        et.check_run(base.iloc[1:], sites, hosts, ["x"], "missing")
    with pytest.raises(AssertionError, match="training"):
        et.check_run(base, sites, hosts, ["h000"], "leak")
    lines = [{"run_id": "a", "spec": {"r": 0}}, {"run_id": "b", "spec": {"r": 1}}]
    assert et.check_specs([{"run_id": "a", "spec": {"r": 0}}, {"run_id": "b", "spec": {"r": 1}}], lines) == 2
    with pytest.raises(AssertionError):
        et.check_specs([{"run_id": "a", "spec": {"r": 0}}], lines)
    with pytest.raises(AssertionError):
        et.check_specs([{"run_id": "a", "spec": {"r": 0}}, {"run_id": "b", "spec": {"r": 2}}], lines)


def test_per_descriptor_metrics_known():
    df = pd.DataFrame({"true_desc_a": [0.0, 1.0, 2.0, 3.0], "pred_desc_a": [0.0, 1.0, 2.0, 3.0],
                       "true_desc_b": [0.0, 2.0, 0.0, 2.0], "pred_desc_b": [1.0, 1.0, 1.0, 1.0]})
    per = et.per_descriptor(df, ["desc_a", "desc_b"])
    assert per["desc_a"]["r2"] == pytest.approx(1.0) and per["desc_a"]["mae"] == 0.0
    assert per["desc_b"]["r2"] == pytest.approx(0.0) and per["desc_b"]["mae"] == pytest.approx(1.0)


def test_p1_summary_classes():
    rows = []
    for r in range(10):
        for s in range(3):
            rows += [{"backbone": "v1", "r": r, "B": 654, "seed": s, "run_id": f"{r}{s}", "descriptor": "desc_h",
                      "class": "host-electronic DFT", "n": 10, "mae": 0.1, "r2": 0.8},
                     {"backbone": "v1", "r": r, "B": 654, "seed": s, "run_id": f"{r}{s}", "descriptor": "desc_s",
                      "class": "site-electronic DFT", "n": 10, "mae": 0.2, "r2": 0.2 + 0.01 * s}]
    out = sa.p1_summary(pd.DataFrame(rows), budgets=(654,))
    assert out["654"]["by_class"]["host_electronic"]["class_mean_r2"] == pytest.approx(0.8)
    assert out["654"]["by_class"]["site_electronic"]["class_mean_r2"] == pytest.approx(0.21)
    assert out["654"]["per_descriptor"]["desc_s"]["n_runs"] == 30


def test_swapin_cells_deterministic():
    p = pd.DataFrame([{"r": r, "B": B, "seed": s, "run_id": f"{r}_{B}_{s}"} for B in (25, 50, 100, 200, 400, 654)
                      for r in range(10) for s in range(3)])
    a, b = sw.pick_cells(p), sw.pick_cells(p.sample(frac=1.0, random_state=3))
    assert list(a.run_id) == list(b.run_id) and len(a) == 3 and a.run_id.nunique() == 3
    with pytest.raises(AssertionError):
        sw.pick_cells(p.iloc[:-1])
