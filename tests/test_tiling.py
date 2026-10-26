import numpy as np
import pytest

pytest.importorskip("pymatgen")
from pymatgen.core import Lattice, Structure

from dftgnn.mlip import tiling


def _cell():
    return Structure(Lattice.from_parameters(3.1, 3.4, 5.2, 88, 95, 101), ["Zn", "Zn", "O", "O", "O"],
                     [[0, 0, 0], [0.5, 0.4, 0.5], [0.3, 0.2, 0.1], [0.7, 0.6, 0.45], [0.1, 0.8, 0.75]])


@pytest.mark.parametrize("M", [np.diag([2, 3, 1]), np.array([[1, 1, 0], [-1, 1, 0], [0, 0, 2]])])
def test_tile_matches_pymatgen_supercell_shuffled_and_translated(M):
    uc = _cell()
    sc = uc.copy()
    sc.make_supercell(M)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(sc))
    sc = Structure(sc.lattice, [sc[i].specie for i in order], [sc[i].frac_coords + [0.13, -0.07, 0.31] for i in order])
    rec = tiling.map_host(uc, sc)
    assert rec["status"] == "ok" and rec["max_residual_A"] < 1e-8 and rec["M"] == M.tolist()
    rebuilt = tiling.tiled_supercell(uc, rec["M"], rec["perm"], np.array(sc.lattice.matrix))
    assert [s.specie.symbol for s in rebuilt] == [s.specie.symbol for s in sc]
    d = np.array(sc.frac_coords) - (np.array(rebuilt.frac_coords) + rec["translation"])
    assert np.abs(d - np.round(d)).max() < 1e-9


def test_non_integer_matrix_fails():
    uc = _cell()
    sc = uc.copy()
    sc.scale_lattice(uc.volume * 8 * 1.01)
    sc.make_supercell(2)
    assert tiling.map_host(uc, sc)["reason"] == "non_integer_matrix"


def test_parse_transformation():
    T = tiling.parse_transformation("Space group: P\nTransformation matrix: [5, 0, 0]  [0, 5, 0]  [0, 0, 3]\n")
    assert T.tolist() == [[5, 0, 0], [0, 5, 0], [0, 0, 3]]
