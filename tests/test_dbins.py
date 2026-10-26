import pytest

pytest.importorskip("pymatgen")

from pymatgen.core import Lattice, Structure

from dftgnn.mlip.dbins import d_count, host_bin


@pytest.mark.parametrize("el,q,d", [("Zn", 2, 10), ("Cd", 2, 10), ("Ga", 3, 10), ("In", 3, 10), ("Sn", 4, 10),
                                    ("Pb", 2, 10), ("Bi", 3, 10), ("Ag", 1, 10), ("Ti", 4, 0), ("Ti", 3, 1),
                                    ("Mg", 2, 0), ("Al", 3, 0), ("La", 3, 0), ("Ce", 4, 0), ("Y", 3, 0),
                                    ("Au", 3, 8), ("W", 6, 0), ("Cu", 2, 9), ("Si", 4, 0), ("Na", 1, 0),
                                    ("Hg", 1, 10), ("Hg", 2, 10)])
def test_d_count(el, q, d):
    assert d_count(el, q) == d


def _rocksalt(cation):
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.2), [cation, "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def test_host_bins():
    assert host_bin(_rocksalt("Mg"))["bin"] == "d0"
    assert host_bin(_rocksalt("Cd"))["bin"] == "d10"
    assert host_bin(_rocksalt("Ni"))["bin"] == "other"
