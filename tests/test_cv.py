"""EXTENSION grouped K-fold CV."""
import numpy as np
import pytest

from dftgnn.config import load_config
from dftgnn.data import universe as U
from dftgnn.split import SPLITS_DIR, dumps, host_table, load_split
from dftgnn.split.cv import CV_FILE, CV_KS, assign_folds, make_cv, parse_name, split_name, val_r
from dftgnn.train import RunSpec, resolve_hosts, val_seed

CFG = load_config()
pytestmark = pytest.mark.skipif(not (U.REPO_ROOT / "data/processed/universe_v1.parquet").is_file(),
                                reason="universe parquet not built")


@pytest.fixture(scope="module")
def uni():
    return U.read_universe(CFG)


@pytest.fixture(scope="module")
def cv(uni):
    return make_cv(uni)


def test_byte_identical_regeneration(uni, cv):
    assert (SPLITS_DIR / CV_FILE).read_bytes() == dumps(cv)
    assert dumps(make_cv(uni)) == dumps(cv)
    assert load_split("cv_v1") == cv


@pytest.mark.parametrize("k", CV_KS)
def test_folds_partition_hosts(cv, k):
    by = cv["by_k"][str(k)]
    tests = [set(f["test"]) for f in by["folds"]]
    assert len(tests) == k
    allh = set().union(*tests)
    assert len(allh) == sum(len(t) for t in tests) == 818          # disjoint and covering
    for f in by["folds"]:
        assert not set(f["train"]) & set(f["test"])
        assert len(f["train"]) + len(f["test"]) == 818 == f["n_train"] + f["n_test"]
    sizes = [len(t) for t in tests]
    assert max(sizes) - min(sizes) <= 1


@pytest.mark.parametrize("k", CV_KS)
def test_every_host_in_exactly_one_test_fold(cv, k):
    by = cv["by_k"][str(k)]
    count = {}
    for f in by["folds"]:
        for h in f["test"]:
            count[h] = count.get(h, 0) + 1
            assert by["fold_of_host"][h] == f["fold"]
    assert len(count) == 818 and set(count.values()) == {1}


def test_training_sizes(cv):
    assert {f["n_train"] for f in cv["by_k"]["10"]["folds"]} == {736, 737}
    assert {f["n_train"] for f in cv["by_k"]["20"]["folds"]} == {777, 778}


def test_stratified_round_robin(uni):
    """Every fold takes hosts from every E_f quintile (the round-robin deals within quintile groups)."""
    hosts = host_table(uni)
    ranked = hosts.sort_values(["mean_Ef", "host_id"], kind="stable")
    groups = np.array_split(ranked.host_id.to_numpy(), 5)
    for k in CV_KS:
        f_of = assign_folds(hosts, k)
        for g in groups:
            per = np.bincount([f_of[h] for h in g], minlength=k)
            assert per.max() - per.min() <= 1


def test_names_and_resolve_hosts(cv):
    assert parse_name(split_name(20, 7)) == (20, 7)
    with pytest.raises(ValueError):
        parse_name("loco_v1_f1")
    f = cv["by_k"]["10"]["folds"][3]
    hp = {"hidden_width": 64}
    spec = RunSpec(model="S", hp=hp, lr=1e-3, weight_decay=1e-5, batch_size=16, split=f["split"], r=3,
                   budget=f["n_train"], seed=1)
    h = resolve_hosts(spec, CFG)
    assert h["test"] == f["test"]
    assert sorted(h["train"] + h["val"]) == f["train"]
    assert not set(h["test"]) & (set(h["train"]) | set(h["val"]))
    assert len(h["val"]) == round(0.1 * f["n_train"])
    from dftgnn.split import val_split

    assert val_split(f["train"], seed=val_seed(val_r(10, 3), f["n_train"], 1))[1] == h["val"]
    bad = RunSpec(model="S", hp=hp, lr=1e-3, weight_decay=1e-5, batch_size=16, split=f["split"], r=3,
                  budget=654, seed=1)
    with pytest.raises(ValueError):
        resolve_hosts(bad, CFG)
