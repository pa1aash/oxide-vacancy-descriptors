"""MLIP geometry helpers on synthetic hosts (no release files needed)."""
from __future__ import annotations

import numpy as np
from pymatgen.core import Lattice, Structure

from dftgnn.mlip import nrp as mlip


def _unit():
    return Structure(Lattice.from_parameters(3.0, 3.2, 4.1, 90, 95, 90),
                     ["Mg", "O", "O"], [[0, 0, 0], [0.5, 0.5, 0.3], [0.2, 0.7, 0.6]])


def _release_like(unit, matrix, seed=0, shift=(0.11, -0.07, 0.23)):
    """A supercell as the release might store it: rotated frame, shuffled atoms, shifted origin."""
    t = unit.copy()
    t.make_supercell(matrix)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(t))
    c, s_ = np.cos(0.4), np.sin(0.4)
    rot = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
    lat = Lattice(t.lattice.matrix @ rot)
    return Structure(lat, [t[i].specie for i in order], (t.frac_coords[order] + shift) % 1.0)


def test_mapping_reproduces_the_release_supercell_exactly():
    unit, m = _unit(), [[2, 0, 0], [0, 2, 0], [0, 0, 1]]
    sc = _release_like(unit, m)
    perm, q, shift = mlip.tile_permutation(unit, m, sc)
    rebuilt = mlip.supercell_from_unit(unit, m, perm, q, shift)
    assert [s.specie for s in rebuilt] == [s.specie for s in sc]
    g = mlip.geometry(sc, rebuilt)
    assert abs(g["isotropic_strain"]) < 1e-10 and g["deviatoric_strain"] < 1e-8 and g["internal_rmsd_A"] < 1e-6


def test_geometry_components_separate_volume_shape_and_internal_motion():
    unit, m = _unit(), [[2, 0, 0], [0, 2, 0], [0, 0, 1]]
    sc = _release_like(unit, m)
    perm, q, shift = mlip.tile_permutation(unit, m, sc)
    iso = unit.copy()
    iso.scale_lattice(unit.volume * 1.03 ** 3)                  # +3% isotropic, no shape change, no internal motion
    g = mlip.geometry(sc, mlip.supercell_from_unit(iso, m, perm, q, shift))
    assert abs(g["isotropic_strain"] - 0.03) < 1e-9 and g["deviatoric_strain"] < 1e-8 and g["internal_rmsd_A"] < 1e-8
    moved = unit.copy()
    moved.translate_sites([1], [0.0, 0.0, 0.02], frac_coords=True)  # one O moves; lattice unchanged
    g = mlip.geometry(sc, mlip.supercell_from_unit(moved, m, perm, q, shift))
    assert abs(g["isotropic_strain"]) < 1e-12 and g["internal_rmsd_A"] > 0
    back = mlip.rescale_to_volume(mlip.supercell_from_unit(iso, m, perm, q, shift), sc)
    assert abs(back.volume - sc.volume) < 1e-6


def test_mapping_rejects_a_wrong_tiling():
    unit = _unit()
    sc = _release_like(unit, [[2, 0, 0], [0, 2, 0], [0, 0, 1]])
    assert mlip.tile_permutation(unit, [[1, 0, 0], [0, 2, 0], [0, 0, 2]], sc) is None


def test_read_matrix():
    txt = "Space group: P6_3mc\nTransformation matrix: [5, 0, 0]  [0, 5, 0]  [0, 0, 3]\nCell multiplicity: 75\n"
    assert mlip.read_matrix(txt) == [[5, 0, 0], [0, 5, 0], [0, 0, 3]]
