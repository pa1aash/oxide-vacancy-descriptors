"""Geometry change between DFT and MLIP hosts, and the three inference conditions.

Physics. The MLIP geometry differs from the DFT geometry in three separable ways: a volume change (isotropic
strain), a shape change at fixed volume (deviatoric strain) and internal relaxation of the atoms within the
cell (internal RMSD). Evaluating models under (i) DFT geometry, (ii) MLIP geometry and (iii) MLIP geometry
rescaled to the DFT volume separates the volume part from the rest.

Definitions (row-vector lattices L, rows are the cell vectors):
    F^T = L_dft^-1 L_mlip; polar decomposition F = R U removes the rigid rotation R
    eps_v = (V_mlip / V_dft)^(1/3) - 1
    deviatoric strain = || U - (tr U / 3) I ||_F
    internal RMSD (A) = RMS over atoms of the minimum-image displacement between the MLIP fractional coordinates
        placed on the DFT lattice and the DFT positions, fixed atom correspondence, mean displacement removed
The metrics are computed on the unit cells (the relaxation preserves atom order).
"""
from __future__ import annotations

import numpy as np

CONDITIONS = ("dft", "mlip", "mlip_rescaled")
VOLUME_TOL = 1e-10


def deformation(L_dft: np.ndarray, L_mlip: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(F, R, U) with F^T = L_dft^-1 L_mlip and F = R U."""
    from scipy.linalg import polar

    F = (np.linalg.solve(L_dft, L_mlip)).T
    R, U = polar(F, side="right")
    return F, R, U


def metrics(L_dft, frac_dft, L_mlip, frac_mlip) -> dict:
    L_dft, L_mlip = np.asarray(L_dft, float), np.asarray(L_mlip, float)
    _, R, U = deformation(L_dft, L_mlip)
    v_d, v_m = abs(np.linalg.det(L_dft)), abs(np.linalg.det(L_mlip))
    dev = U - np.trace(U) / 3 * np.eye(3)
    df = np.asarray(frac_mlip, float) - np.asarray(frac_dft, float)
    df -= np.round(df)
    df -= df.mean(0)
    d = df @ L_dft
    return {"eps_v": float((v_m / v_d) ** (1 / 3) - 1), "deviatoric": float(np.linalg.norm(dev)),
            "internal_rmsd_A": float(np.sqrt((d ** 2).sum(1).mean())), "volume_dft_A3": float(v_d),
            "volume_mlip_A3": float(v_m), "rotation_angle_deg": float(np.degrees(np.arccos(
                np.clip((np.trace(R) - 1) / 2, -1, 1))))}


def host_metrics(uc_dft, uc_mlip) -> dict:
    if [s.specie.symbol for s in uc_dft] != [s.specie.symbol for s in uc_mlip]:
        raise ValueError("unit cells differ in species order")
    return metrics(uc_dft.lattice.matrix, uc_dft.frac_coords, uc_mlip.lattice.matrix, uc_mlip.frac_coords)


def rescaled_to_volume(structure, volume: float):
    """``structure`` with its lattice scaled isotropically to ``volume`` (fractional coordinates kept)."""
    from pymatgen.core import Lattice, Structure

    f = (volume / structure.volume) ** (1 / 3)
    s = Structure(Lattice(np.array(structure.lattice.matrix) * f), structure.species, structure.frac_coords)
    if abs(s.volume - volume) / volume > VOLUME_TOL:
        raise AssertionError(f"rescaled volume {s.volume} != {volume}")
    return s


def condition_structures(dft_sc, mlip_sc) -> dict:
    """{"mlip": MLIP tiled supercell, "mlip_rescaled": the same at the DFT supercell volume}."""
    if [s.specie.symbol for s in dft_sc] != [s.specie.symbol for s in mlip_sc]:
        raise ValueError("MLIP supercell is not in release atom order")
    return {"mlip": mlip_sc, "mlip_rescaled": rescaled_to_volume(mlip_sc, dft_sc.volume)}


def condition_graphs(dft_sc, mlip_sc, vacancy_indices, cutoff: float = 5.0) -> dict:
    """Host graphs of conditions (ii) and (iii); the vacancy atoms of the host are asserted to be O in both."""
    from dftgnn.graphs import host_graph

    out = {}
    for name, s in condition_structures(dft_sc, mlip_sc).items():
        g = host_graph(s, cutoff)
        for v in vacancy_indices:
            if int(g["z"][int(v)]) != 8 or dft_sc[int(v)].specie.symbol != "O":
                raise AssertionError(f"vacancy atom {v} is not O under condition {name}")
        out[name] = g
    return out
