import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from dftgnn.split import SPLITS_DIR, dumps, load_split
from dftgnn.split.loco import N_FAMILIES_V1, families, make_loco

ROOT = Path(__file__).resolve().parents[1]
UNI = ROOT / "data/processed/universe_v1.parquet"
pytestmark = pytest.mark.skipif(not UNI.is_file(), reason="universe parquet not built")


@pytest.fixture(scope="module")
def uni():
    return pd.read_parquet(UNI)


@pytest.fixture(scope="module")
def loco(uni):
    return make_loco(uni)


def test_partition_and_sizes(loco):
    tests = [h for f in loco["folds"] for h in f["test"]]
    assert len(tests) == len(set(tests)) == 818
    for f in loco["folds"]:
        assert len(f["train"]) + len(f["test"]) == 818 == loco["n_hosts"]
        assert not set(f["train"]) & set(f["test"])
        assert f["n_train"] == len(f["train"])


def test_families_disjoint_across_folds(loco):
    seen = {}
    for f in loco["folds"]:
        for h in f["test"]:
            fam = loco["host_family"][h]
            assert seen.setdefault(fam, f["fold"]) == f["fold"]
    assert seen == loco["family_to_fold"]
    assert loco["n_families"] == N_FAMILIES_V1 == len(seen)


def test_tracked_file_byte_identical(loco):
    assert (SPLITS_DIR / "loco_v1.json").read_bytes() == dumps(loco)
    assert load_split("loco_v1")["folds"][0]["test"] == loco["folds"][0]["test"]


def test_family_key_matches_s02(uni):
    spec = importlib.util.spec_from_file_location("audit_misc", ROOT / "scripts/audit_misc.py")
    am = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(am)
    h = families(uni)
    from pymatgen.core import Composition

    elements = h.formula.map(lambda f: "|".join(e.symbol for e in Composition(f).elements))
    s02 = elements.str.split("|").map(lambda e: tuple(sorted({am.group_of(x) for x in e if x != "O"})))
    assert list(s02.map(lambda t: "-".join(map(str, t)))) == list(h.family)
    assert am.families(pd.DataFrame({"elements": elements}))["group_pattern_distinct_groups"] == N_FAMILIES_V1
