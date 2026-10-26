import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from dftgnn.config import load_config
from dftgnn.data import universe as U
from dftgnn.split import (
    SPLITS_DIR,
    budget_train,
    coverage,
    curve_resample,
    derived_seed,
    dumps,
    host_table,
    kiyohara_hosts,
    load_meta,
    load_split,
    make_all,
    val_split,
)

CFG = load_config()
pytestmark = pytest.mark.skipif(not (U.REPO_ROOT / "data/processed/universe_v1.parquet").is_file(),
                                reason="universe parquet not built")


@pytest.fixture(scope="module")
def uni():
    return U.read_universe(CFG)


@pytest.fixture(scope="module")
def splits(uni):
    return make_all(uni, CFG)


def outer(splits):
    return [splits[f"outer_r{r}"] for r in range(CFG.split.n_outer_resamples)]


def test_no_test_host_in_train(splits):
    for s in outer(splits):
        assert not set(s["test"]) & set(s["budget_order"])
        assert len(set(s["budget_order"])) == len(s["budget_order"])


def test_pool_and_test_sizes(splits, uni):
    n = uni.host_id.nunique()
    for s in outer(splits):
        assert len(s["test"]) == round(CFG.split.test_fraction * n)
        assert len(s["test"]) + len(s["budget_order"]) == n
        assert len(s["budget_order"]) == CFG.budgets.max  # 654


def test_nesting(splits):
    for s in outer(splits):
        prev: list[str] = []
        for b in CFG.budgets.hosts:
            cur = budget_train(s, b)
            assert len(cur) == b
            assert cur[: len(prev)] == prev and set(prev) <= set(cur)
            prev = cur


def test_test_set_is_stratified(splits, uni):
    t = host_table(uni).set_index("host_id")
    for s in outer(splits):
        assert max(s["stratum_test_counts"]) - min(s["stratum_test_counts"]) <= 1
        q = pd.qcut(t.mean_Ef, 5, labels=False)
        share = q.loc[s["test"]].value_counts(normalize=True)
        assert share.min() > 0.15  # no quintile starved


def test_resamples_differ(splits):
    tests = [tuple(s["test"]) for s in outer(splits)]
    assert len(set(tests)) == len(tests)


def test_determinism_bytes(uni, splits):
    again = make_all(uni, CFG)
    for k in splits:
        assert dumps(again[k]) == dumps(splits[k])


def test_tracked_files_match_regeneration(splits):
    for k, v in splits.items():
        assert (SPLITS_DIR / f"{k}.json").read_bytes() == dumps(v)


def test_manifest_hashes():
    for line in (SPLITS_DIR / "MANIFEST.sha256").read_text().splitlines():
        sha, name = line.split("  ")
        assert hashlib.sha256((SPLITS_DIR / name).read_bytes()).hexdigest() == sha


def test_derived_seed_stable():
    assert derived_seed(3, "budget_order") == derived_seed(3, "budget_order")
    assert derived_seed(3, "budget_order") != derived_seed(4, "budget_order")


def test_coverage_reported(splits, uni):
    """Independent hold-outs cannot test every host (0.8**10 of hosts are never tested).

    The counts are recorded in splits/meta.json; here we check they are reported correctly and
    agree with the independence expectation within a loose binomial band.
    """
    hosts = list(host_table(uni).host_id)
    cov = coverage(outer(splits), hosts)
    assert json.loads(json.dumps(cov)) == load_meta()["coverage"]
    assert cov["never_in_test"] + cov["in_test_at_least_once"] == len(hosts)
    exp = cov["expected_never_in_test_if_independent"]
    assert abs(cov["never_in_test"] - exp) < 4 * np.sqrt(exp)


def test_kiyohara_covers_universe(uni):
    k = kiyohara_hosts(uni)
    allk = k["train"] + k["val"] + k["test"]
    assert len(allk) == len(set(allk)) == uni.host_id.nunique() == 818
    assert load_split("kiyohara") == k
    assert {kk: len(v) for kk, v in k.items()} == {"train": 571, "val": 121, "test": 126}


def test_val_split():
    hosts = [f"h{i}" for i in range(100)]
    tr, va = val_split(hosts, frac=0.1, min_hosts=3, seed=1)
    assert len(va) == 10 and not set(tr) & set(va) and sorted(tr + va) == sorted(hosts)
    assert val_split(hosts, 0.1, 3, 1) == (tr, va)
    tr, va = val_split(hosts[:25], frac=0.1, min_hosts=3, seed=1)
    assert len(va) == 3  # floor of min_hosts at small budgets
    with pytest.raises(ValueError):
        val_split(hosts[:3], 0.1, 3, 0)


def test_curve_resamples(splits):
    cur = splits["rf_curve"]
    assert len(cur["resamples"]) == CFG.curve.n_resamples
    for i in (0, 57, 99):
        for n in CFG.curve.sizes:
            train, test = curve_resample(cur, i, n)
            assert len(train) == n and len(test) == CFG.curve.n_test_hosts
            assert not set(train) & set(test)
        small, _ = curve_resample(cur, i, 22)
        big, _ = curve_resample(cur, i, 220)
        assert big[:22] == small  # nested
    assert len({tuple(r["test"]) for r in cur["resamples"]}) == CFG.curve.n_resamples


def test_loco_splits_are_family_disjoint_and_cover_every_host():
    import importlib.util
    import json
    from pathlib import Path

    d = Path(__file__).resolve().parents[1] / "splits" / "loco"
    folds = [json.loads((d / f"loco_f{k}.json").read_text()) for k in range(5)]
    tests = [set(f["test"]) for f in folds]
    assert sum(len(t) for t in tests) == len(set().union(*tests)) == 818
    for f in folds:
        assert not set(f["test"]) & set(f["budget_order"])
        assert len(f["test"]) + len(f["budget_order"]) == 818
    fams = [set(f["test_families"]) for f in folds]
    assert sum(len(x) for x in fams) == len(set().union(*fams)) == folds[0]["n_families_total"] == 131
    spec = importlib.util.spec_from_file_location("mls", d.parents[1] / "scripts" / "make_loco_splits_nrp.py")
    mls = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mls)
    if (mls.G.STORE_DIR / "meta.json").is_file():
        assert all((d / n).read_bytes() == b for n, b in mls.make().items())
