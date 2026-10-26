"""MLIP relaxation of one host.

Physics. MACE-MP-0 medium (float64) relaxes the DFT unit cell, cell and positions together, with no
symmetry constraint: ASE ``FrechetCellFilter`` (``ExpCellFilter`` if unavailable) driven by FIRE to
fmax 0.01 eV/A (the filter's generalised forces, which include the stress term) within 500 steps. A host
that does not converge is kept and flagged. The relaxed unit cell is tiled with the host's integer supercell
matrix and its atoms put in release supercell order (``dftgnn.mlip.tiling``); hosts flagged
``excluded_from_C3b`` get no tiled supercell.

Outputs in ``data/processed/mlip_v1/`` (gitignored), per host:
    <host_id>.json      record: converged, steps, final fmax, energies, volumes, wall seconds, peak RSS,
                        paths and sha256 of the structures below, model sha256
    <host_id>_uc.json   relaxed unit cell (pymatgen dict, full precision)
    <host_id>_sc.json   relaxed tiled supercell in release atom order (lattice M L_mlip)
Idempotent: a host whose record exists and whose structure files match their hashes is skipped.
"""
from __future__ import annotations

import json
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np

from dftgnn.graphs.store import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "data" / "processed" / "mlip_v1"
FMAX = 0.01
MAX_STEPS = 500


def _peak_rss_bytes() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == "darwin" else r * 1024)


def cell_filter(atoms):
    try:
        from ase.filters import FrechetCellFilter

        return FrechetCellFilter(atoms), "FrechetCellFilter"
    except ImportError:                                   # pragma: no cover - older ASE
        from ase.filters import ExpCellFilter

        return ExpCellFilter(atoms), "ExpCellFilter"


def record_ok(host_id: str, out_dir: Path = OUT_DIR) -> bool:
    f = out_dir / f"{host_id}.json"
    if not f.is_file():
        return False
    try:
        rec = json.loads(f.read_text())
        return all((out_dir / a["path"]).is_file() and sha256_file(out_dir / a["path"]) == a["sha256"]
                   for a in rec["files"].values())
    except (json.JSONDecodeError, KeyError):
        return False


def point_op(tiling_row: dict):
    """The host's recorded lattice point operation R, or None (identity: a translation-only host).

    A pandas row read from the tiling map carries NaN (not None) for a translation-only host; NaN, None, an empty
    string and JSON null all mean no operation.
    """
    op = tiling_row.get("point_op")
    if isinstance(op, str):
        op = json.loads(op) if op.strip() else None
    if op is None or (isinstance(op, float) and math.isnan(op)):
        return None
    return op


def relax_structure(uc, calc, fmax: float = FMAX, steps: int = MAX_STEPS) -> tuple:
    """(relaxed pymatgen Structure, info dict) for one unit cell."""
    from ase.optimize import FIRE
    from pymatgen.io.ase import AseAtomsAdaptor

    atoms = AseAtomsAdaptor.get_atoms(uc)
    atoms.calc = calc
    e0, v0 = float(atoms.get_potential_energy()), float(atoms.get_volume())
    filt, fname = cell_filter(atoms)
    opt = FIRE(filt, logfile=None)
    t0 = time.time()
    converged = bool(opt.run(fmax=fmax, steps=steps))
    wall = time.time() - t0
    f_final = float(np.sqrt((filt.get_forces() ** 2).sum(1)).max())
    info = {"converged": converged, "steps": int(opt.nsteps), "final_fmax_eV_per_A": f_final,
            "initial_energy_eV": e0, "final_energy_eV": float(atoms.get_potential_energy()),
            "initial_volume_A3": v0, "final_volume_A3": float(atoms.get_volume()), "n_atoms": len(atoms),
            "relax_wall_s": wall, "s_per_step": wall / max(1, opt.nsteps), "filter": fname, "optimiser": "FIRE",
            "fmax_eV_per_A": fmax, "max_steps": steps}
    return AseAtomsAdaptor.get_structure(atoms), info


def relax_host(host_id: str, uc, tiling_row: dict | None, calc, *, model_sha256: str, out_dir: Path = OUT_DIR,
               force: bool = False) -> dict:
    """Relax, tile and write one host; returns its record (``skipped`` True if already done)."""
    from dftgnn.mlip.tiling import tiled_supercell

    out_dir.mkdir(parents=True, exist_ok=True)
    if not force and record_ok(host_id, out_dir):
        return {**json.loads((out_dir / f"{host_id}.json").read_text()), "skipped": True}
    t0 = time.time()
    relaxed, info = relax_structure(uc, calc)
    files = {}
    p = out_dir / f"{host_id}_uc.json"
    p.write_text(json.dumps(relaxed.as_dict()))
    files["unit_cell"] = {"path": p.name, "sha256": sha256_file(p)}
    tiled = None
    if tiling_row is not None and tiling_row.get("status") == "ok":
        M = json.loads(tiling_row["M"]) if isinstance(tiling_row["M"], str) else tiling_row["M"]
        perm = json.loads(tiling_row["perm"]) if isinstance(tiling_row["perm"], str) else tiling_row["perm"]
        tiled = tiled_supercell(relaxed, M, perm, point_op=point_op(tiling_row))
        p = out_dir / f"{host_id}_sc.json"
        p.write_text(json.dumps(tiled.as_dict()))
        files["supercell"] = {"path": p.name, "sha256": sha256_file(p)}
    rec = {"host_id": host_id, **info, "tiled": tiled is not None,
           "excluded_from_C3b": tiled is None, "model_sha256": model_sha256, "files": files,
           "wall_s": time.time() - t0, "peak_rss_bytes": _peak_rss_bytes()}
    tmp = out_dir / f".{host_id}.json.tmp"
    tmp.write_text(json.dumps(rec, indent=1, sort_keys=True))
    tmp.replace(out_dir / f"{host_id}.json")
    return {**rec, "skipped": False}
