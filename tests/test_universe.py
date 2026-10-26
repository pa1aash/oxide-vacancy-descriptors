import ast
import hashlib
import json
import warnings
from pathlib import Path

import pytest

from dftgnn.config import load_config
from dftgnn.data import universe as U

warnings.filterwarnings("ignore")
CFG = load_config()
PATHS = U.release_paths(CFG)
pytestmark = pytest.mark.skipif(not PATHS.ml_csv.is_file(), reason="Kumagai release not unpacked")


@pytest.fixture(scope="session")
def built():
    return U.build_universe(CFG, return_cascade=True)


@pytest.fixture(scope="session")
def uni(built):
    return built[0]


@pytest.fixture(scope="session")
def release():
    return U.release_neutral(CFG)[0]


def _structure(path):
    from pymatgen.core import Structure

    return Structure.from_file(U.REPO_ROOT / path)


def test_zno_maps_to_atom_150_with_four_zn(uni):
    row = uni.set_index("site_id").loc["ZnO_Va_O1_0"]
    assert row.vacancy_atom_index == 150
    assert row.supercell_cif_path == "data/raw/unpacked/oxygen_vacancies_db/site_info/ZnO/supercell.cif"
    st = _structure(row.supercell_cif_path)
    assert st[150].specie.symbol == "O"
    shell = [n for n in st.get_neighbors(st[150], 2.5)]
    assert len(shell) == 4
    assert all(n.specie.symbol == "Zn" for n in shell)
    # Zn-O bond lengths of the supercell: 1.934 A (x3) and 1.942 A
    assert all(1.92 <= n.nn_distance <= 1.95 for n in shell)


def test_no_excluded_entry_present(built, release):
    uni, cas = built
    ids = set(uni.site_id)
    removed = {r["site_id"] for s in cas.steps for r in s["removed"]}
    assert not ids & removed
    steps = {s["step"]: s for s in cas.steps}
    assert steps["D1"]["n_removed"] == 15
    # independent of the cascade bookkeeping: no D1 flag, no Sr2Zr7O16, no PHS
    rel = release.assign(site_id=release.full_name + "_0").set_index("site_id").loc[sorted(ids)]
    flagged = rel.release_flags.str.split("|").map(lambda x: bool(set(x) & set(U.D1_FLAGS)))
    assert not flagged.any()
    assert not (rel.is_shallow == True).any()
    assert "Sr2Zr7O16" not in set(uni.formula)
    for h, d3 in cas.d3.items():
        if d3.get("in_universe") and not d3["outcome"].startswith("pass"):
            assert h not in set(uni.formula)


def test_row_count_matches_cascade_final(built):
    uni, cas = built
    final = cas.steps[-1]
    assert final["step"] == "final"
    assert len(uni) == final["sites"]
    assert uni.formula.nunique() == final["hosts"]
    rec = U.REPO_ROOT / "results" / "filter_cascade.json"
    if rec.is_file():
        steps = json.loads(rec.read_text())["payload"]["steps"]
        assert steps[-1]["sites"] == len(uni)
        assert steps[-1]["hosts"] == uni.formula.nunique()


def test_every_vacancy_index_is_oxygen(uni):
    for path, grp in uni.groupby("supercell_cif_path"):
        st = _structure(path)
        assert len(st) == grp.n_atoms.iloc[0]
        for i in grp.vacancy_atom_index:
            assert st[int(i)].specie.symbol == "O", (path, i)


def test_no_path_from_glob(uni, monkeypatch):
    # static: the module never calls a glob
    tree = ast.parse(Path(U.__file__).read_text())
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names |= {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
              for a in n.names}
    assert not names & {"glob", "rglob", "iglob", "fnmatch"}
    # dynamic: resolution works with globbing disabled and gives the per-host JSON path
    def boom(*a, **k):
        raise AssertionError("glob called")
    monkeypatch.setattr(Path, "glob", boom)
    monkeypatch.setattr(Path, "rglob", boom)
    for f, path in uni.groupby("formula").supercell_cif_path.first().items():
        bulk = json.loads(PATHS.host_json(f, "bulk").read_text())
        assert bulk["formula"] == f
        expected = U._rel(PATHS.site_info / bulk["formula"] / "supercell.cif")
        assert path == expected == U._rel(U.resolve_supercell_cif(PATHS, f, bulk))


def test_ids_file_reproduces_exactly(uni):
    tracked = U.IDS_FILE.read_bytes()
    assert U.ids_csv_bytes(uni) == tracked
    sha_line = U.IDS_FILE.with_name(U.IDS_FILE.name + ".sha256").read_text().split()
    assert sha_line == [hashlib.sha256(tracked).hexdigest(), U.IDS_FILE.name]


def test_schema(uni):
    desc = [c for c in uni.columns if c.startswith("desc_")]
    assert len(desc) == 70
    assert set(uni.attrs["descriptor_class"]) == set(desc)
    assert uni.site_id.is_unique
    assert set(uni.kiyohara_split) <= {"train", "val", "test", "absent"}
    assert uni.spin_polarised.dtype == bool
    assert (uni.defect_moment_muB[~uni.spin_polarised] == 0).all()
    for c in ("site_id", "host_id", "formula", "site_label", "supercell_cif_path",
              "vacancy_atom_index", "n_atoms", "target_Ef_eV", "defect_moment_muB"):
        assert c in uni.columns
