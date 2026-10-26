"""MLIP host geometry.

Inputs (built on the Mac, where the release is unpacked; ``prepare_inputs``). For every host: the PBEsol(+U)
unit cell from the release's per-host JSON (``structure_graph.structure``), the supercell transformation
matrix from ``site_info/<formula>/cell_info.txt`` and the permutation that maps the tiled unit cell onto the
release supercell (``tile_permutation``): tiling the DFT unit cell with the matrix and permuting by it
reproduces the release supercell atom for atom (same element, same position up to ``MATCH_TOL_A``), so vacancy
atom indices carry over one to one. Hosts whose tiling does not reproduce the supercell are excluded and
counted.

Relaxation (on a GPU pod with MACE installed; ``relax_unit_cell``): MACE-MP-0 (size and dtype from
``config.mlip``), cell and positions relaxed with FIRE on a ``FrechetCellFilter`` to ``fmax`` or ``max_steps``.

Geometry (``geometry``). The MLIP supercell is the relaxed unit cell tiled with the same matrix and permuted
with the same permutation. Components, relative to the DFT supercell (lattices as row vectors, L_mlip = L_dft F^T):
    isotropic strain   eps_v = (V_mlip / V_dft)^(1/3) - 1
    deviatoric strain  || F / det(F)^(1/3) - I ||_F
    internal RMSD      RMSD between the DFT Cartesian positions and the MLIP fractional positions placed on
                       the DFT lattice (minimum image, fixed atom correspondence)
The control condition rescales the MLIP supercell isotropically to the DFT volume.
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np

MATCH_TOL_A = 0.05


def read_matrix(cell_info_text: str) -> list[list[int]]:
    m = re.search(r"Transformation matrix:\s*\[([^\]]*)\]\s*\[([^\]]*)\]\s*\[([^\]]*)\]", cell_info_text)
    if m is None:
        raise ValueError("no transformation matrix in cell_info.txt")
    return [[round(float(x)) for x in row.split(",")] for row in m.groups()]


def _tile(unit, matrix):
    s = unit.copy()
    s.make_supercell(matrix)
    return s


def tile_permutation(unit, matrix, supercell, tol: float = MATCH_TOL_A) -> tuple[list[int], list, list] | None:
    """``(perm, Q, shift)`` with ``tiled[perm[i]] + shift`` = ``supercell[i]`` and ``S = T @ Q`` (row-vector
    lattices of the supercell S and the tiled cell T). Q is the rotation between their Cartesian frames (the
    identity when the release used the unit cell's frame); ``shift`` is the fractional origin shift of the
    symmetrised supercell. None if the tiling does not reproduce the supercell: different cell lengths or angles,
    or, for every candidate shift, an atom farther than ``tol`` from its partner."""
    tiled = _tile(unit, matrix)
    if len(tiled) != len(supercell):
        return None
    if not (np.allclose(tiled.lattice.abc, supercell.lattice.abc, atol=tol)
            and np.allclose(tiled.lattice.angles, supercell.lattice.angles, atol=0.1)):
        return None
    q = np.linalg.solve(tiled.lattice.matrix, supercell.lattice.matrix)
    if not np.allclose(q @ q.T, np.eye(3), atol=1e-3):
        return None
    lat = supercell.lattice
    fs = supercell.frac_coords
    zt = np.array([site.specie.Z for site in tiled])
    zs = np.array([site.specie.Z for site in supercell])
    # the symmetrised supercell may sit at a shifted origin: try no shift first, then every shift that puts a
    # tiled atom of the rarest element on the supercell's first atom of that element
    rare = min(set(zs.tolist()), key=lambda z: (zs == z).sum())
    i0 = int(np.nonzero(zs == rare)[0][0])
    shifts = [np.zeros(3)] + [fs[i0] - tiled.frac_coords[j] for j in np.nonzero(zt == rare)[0]]
    for shift in shifts:
        perm = _match(tiled.frac_coords + shift, fs, zt, zs, lat.matrix, tol)
        if perm is not None:
            return perm, q.tolist(), (shift - np.round(shift)).tolist()
    return None


def _match(ft, fs, zt, zs, lat, tol):
    perm = np.full(len(fs), -1)
    used = np.zeros(len(ft), bool)
    for i in range(len(fs)):
        cand = np.nonzero((zt == zs[i]) & ~used)[0]
        d = ft[cand] - fs[i]
        d -= np.round(d)
        dist = np.linalg.norm(d @ lat, axis=1)
        k = int(np.argmin(dist))
        if dist[k] > tol:
            return None
        perm[i] = cand[k]
        used[cand[k]] = True
    return perm.tolist()


def prepare_inputs(universe, release_root: Path) -> dict:
    """Per host: unit cell (pymatgen dict), matrix, permutation; plus the excluded hosts with the reason."""
    from pymatgen.core import Structure

    hosts = universe.groupby("host_id").agg(formula=("formula", "first"), cif=("supercell_cif_path", "first"))
    out, excluded = {}, {}
    root = Path(release_root)
    repo = root.parents[3]
    for hid, row in hosts.sort_index().iterrows():
        bulk = json.loads((root / "oxygen_vacancies_db_data" / row.formula / "bulk_visual_data.json").read_text())
        unit = Structure.from_dict(bulk["structure_graph"]["structure"])
        matrix = read_matrix((root / "site_info" / row.formula / "cell_info.txt").read_text())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sc = Structure.from_file(repo / row.cif)
        tp = tile_permutation(unit, matrix, sc)
        if tp is None:
            excluded[hid] = "tiling does not reproduce the release supercell"
            continue
        out[hid] = {"formula": row.formula, "unit_cell": unit.as_dict(), "matrix": matrix, "perm": tp[0], "rotation": tp[1],
                    "shift": tp[2]}
    return {"hosts": out, "excluded": excluded, "match_tol_A": MATCH_TOL_A}


def supercell_from_unit(unit, matrix, perm, rotation=None, shift=None):
    """Tile ``unit`` with ``matrix``, order the atoms like the release supercell and express the lattice in the
    release supercell's Cartesian frame (``rotation`` from ``tile_permutation``)."""
    from pymatgen.core import Lattice, Structure

    t = _tile(unit, matrix)
    lat = t.lattice.matrix if rotation is None else t.lattice.matrix @ np.asarray(rotation)
    frac = t.frac_coords[perm] + (np.zeros(3) if shift is None else np.asarray(shift))
    return Structure(Lattice(lat), [t[i].specie for i in perm], frac)


def geometry(dft_sc, mlip_sc) -> dict:
    """Isotropic strain, deviatoric strain and internal RMSD of ``mlip_sc`` against ``dft_sc`` (same atom order)."""
    ld, lm = dft_sc.lattice.matrix, mlip_sc.lattice.matrix
    eps_v = (abs(np.linalg.det(lm)) / abs(np.linalg.det(ld))) ** (1 / 3) - 1
    f_t = np.linalg.solve(ld, lm)                 # L_mlip = L_dft @ F^T
    f = f_t.T
    dev = f / np.cbrt(np.linalg.det(f)) - np.eye(3)
    d = mlip_sc.frac_coords - dft_sc.frac_coords
    d -= np.round(d)
    rmsd = float(np.sqrt(((d @ ld) ** 2).sum(1).mean()))
    return {"isotropic_strain": float(eps_v), "deviatoric_strain": float(np.linalg.norm(dev)),
            "internal_rmsd_A": rmsd}


def rescale_to_volume(mlip_sc, dft_sc):
    """``mlip_sc`` scaled isotropically to the volume of ``dft_sc`` (fractional coordinates unchanged)."""
    s = mlip_sc.copy()
    s.scale_lattice(dft_sc.volume)
    return s


def relax_unit_cell(unit, calc, fmax: float, max_steps: int) -> tuple[object, dict]:
    """Relax cell and positions with FIRE on a FrechetCellFilter. Returns (structure, info)."""
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE
    from pymatgen.io.ase import AseAtomsAdaptor

    atoms = AseAtomsAdaptor.get_atoms(unit)
    atoms.calc = calc
    opt = FIRE(FrechetCellFilter(atoms), logfile=None)
    converged = bool(opt.run(fmax=fmax, steps=max_steps))
    forces = atoms.get_forces()
    info = {"converged": converged, "steps": int(opt.get_number_of_steps()),
            "final_fmax_eV_per_A": float(np.linalg.norm(forces, axis=1).max()),
            "energy_eV": float(atoms.get_potential_energy())}
    return AseAtomsAdaptor.get_structure(atoms), info
