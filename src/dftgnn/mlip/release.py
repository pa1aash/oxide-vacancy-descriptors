"""Host structures from the K21 release (no MLIP involved).

The pristine DFT unit cell of a host is the structure in ``structure_graph.structure`` of
``oxygen_vacancies_db_data/<formula>/bulk_visual_data.json``; the pristine supercell is
``site_info/<formula>/supercell.cif`` (the universe's ``supercell_cif_path``). The release directory name is
the host's formula (checked against the supercell path for every host of universe v1).

``unit_cells`` caches every unit cell of the universe as pymatgen dicts in
``data/processed/unit_cells_v1.json`` (gitignored), the file shipped to the pod.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DB = REPO_ROOT / "data" / "raw" / "unpacked" / "oxygen_vacancies_db"
UNIVERSE = REPO_ROOT / "data" / "processed" / "universe_v1.parquet"
UNIT_CELLS = REPO_ROOT / "data" / "processed" / "unit_cells_v1.json"


def host_table(universe: pd.DataFrame | None = None) -> pd.DataFrame:
    """host_id -> formula, release directory, supercell CIF (sorted by host_id)."""
    u = universe if universe is not None else pd.read_parquet(UNIVERSE)
    h = (u.groupby("host_id").agg(formula=("formula", "first"), cif=("supercell_cif_path", "first"),
                                  n_atoms=("n_atoms", "first"))
         .reset_index().sort_values("host_id").reset_index(drop=True))
    h["release_dir"] = h.cif.str.split("/").str[-2]
    if (h.release_dir != h.formula).any():
        raise ValueError("release directory differs from the formula for some hosts")
    return h


def read_unit_cell(formula: str, db: Path = DB):
    """The DFT unit cell (pymatgen Structure) of the release host ``formula``."""
    from pymatgen.core import Structure

    d = json.loads((db / "oxygen_vacancies_db_data" / formula / "bulk_visual_data.json").read_text())
    return Structure.from_dict(d["structure_graph"]["structure"])


def read_supercell(cif: str, root: Path = REPO_ROOT):
    from pymatgen.core import Structure

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Structure.from_file(str(root / cif))


def build_unit_cells(hosts: pd.DataFrame, out: Path = UNIT_CELLS) -> dict:
    cells = {r.host_id: read_unit_cell(r.formula).as_dict() for r in hosts.itertuples()}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cells, sort_keys=True))
    return cells


def unit_cells(path: Path = UNIT_CELLS) -> dict:
    """host_id -> pymatgen Structure, from the cache (``build_unit_cells`` writes it)."""
    from pymatgen.core import Structure

    return {h: Structure.from_dict(d) for h, d in json.loads(path.read_text()).items()}
