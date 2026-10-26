"""EXTENSION outer resamples r = 10..29 (splits/resamples_10_29.json) and the append-only manifest writer."""
import hashlib
import importlib.util
import json

import pytest

from dftgnn.config import load_config
from dftgnn.data import universe as U
from dftgnn.split import SPLITS_DIR, budget_train, dumps, load_split, make_all
from dftgnn.split.ext import EXT_FILE, EXT_RESAMPLES, make_ext

CFG = load_config()
pytestmark = pytest.mark.skipif(not (U.REPO_ROOT / "data/processed/universe_v1.parquet").is_file(),
                                reason="universe parquet not built")
_spec = importlib.util.spec_from_file_location("make_splits_ext", U.REPO_ROOT / "scripts" / "make_splits_ext.py")
mse = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mse)


@pytest.fixture(scope="module")
def uni():
    return U.read_universe(CFG)


@pytest.fixture(scope="module")
def ext(uni):
    return make_ext(uni, CFG)


def all30():
    return [load_split(f"outer_r{r}") for r in range(30)]


def test_registered_resamples_regenerate_byte_identical(uni):
    reg = make_all(uni, CFG)
    for r in range(10):
        assert (SPLITS_DIR / f"outer_r{r}.json").read_bytes() == dumps(reg[f"outer_r{r}"])


def test_tracked_file_byte_identical_and_deterministic(uni, ext):
    assert (SPLITS_DIR / EXT_FILE).read_bytes() == dumps(ext)
    assert dumps(make_ext(uni, CFG)) == dumps(ext)


def test_manifest_lists_extension_and_every_entry_verifies():
    lines = (SPLITS_DIR / "MANIFEST.sha256").read_text().splitlines()
    names = [ln.split("  ")[1] for ln in lines]
    assert EXT_FILE in names and len(names) == len(set(names))
    for ln in lines:
        s, n = ln.split("  ")
        assert hashlib.sha256((SPLITS_DIR / n).read_bytes()).hexdigest() == s


def test_no_test_host_in_train_and_sizes():
    for s in all30():
        assert not set(s["test"]) & set(s["budget_order"])
        assert len(s["test"]) == 164 and len(s["budget_order"]) == 654
        assert len(set(s["test"]) | set(s["budget_order"])) == 818


def test_nesting():
    for r in EXT_RESAMPLES:
        s = load_split(f"outer_r{r}")
        prev: set[str] = set()
        for b in CFG.budgets.hosts:
            cur = set(budget_train(s, b))
            assert len(cur) == b and prev <= cur
            prev = cur


def test_seed_scheme_and_distinct_from_registered():
    tests = {r: frozenset(load_split(f"outer_r{r}")["test"]) for r in range(30)}
    assert len(set(tests.values())) == 30
    for r in EXT_RESAMPLES:
        assert load_split(f"outer_r{r}")["seed"] == r and load_split(f"outer_r{r}")["resample"] == r


def test_load_split_range():
    with pytest.raises(FileNotFoundError):
        load_split("outer_r30")


def test_coverage_over_30_resamples(ext):
    cnt = {}
    for s in all30():
        for h in s["test"]:
            cnt[h] = cnt.get(h, 0) + 1
    cov = ext["coverage_registered_plus_extension"]
    assert cov["n_resamples"] == 30 and cov["in_test_at_least_once"] == len(cnt)
    assert cov["never_in_test"] == 818 - len(cnt)
    print(f"hosts tested at least once over 30 resamples: {len(cnt)} of 818")


def test_append_only_refuses_changes(tmp_path):
    (tmp_path / "a.json").write_bytes(b"A\n")
    sha_a = hashlib.sha256(b"A\n").hexdigest()
    (tmp_path / "MANIFEST.sha256").write_text(f"{sha_a}  a.json\n")
    before = (tmp_path / "MANIFEST.sha256").read_text()
    added = mse.append_only(tmp_path, {"b.json": b"B\n"}, check=False)
    assert len(added) == 1 and (tmp_path / "MANIFEST.sha256").read_text().startswith(before)
    assert mse.append_only(tmp_path, {"b.json": b"B\n"}, check=False) == []       # idempotent
    with pytest.raises(SystemExit):
        mse.append_only(tmp_path, {"b.json": b"changed\n"}, check=False)        # existing file differs
    with pytest.raises(SystemExit):
        mse.append_only(tmp_path, {"a.json": b"other\n"}, check=False)          # listed hash differs
    assert json.dumps(mse.verify_manifest(tmp_path)) == "[]"
