"""One MLIP host job (queue stage ``relax``): relax and tile (``relax.relax_host``), then the geometry metrics of
the unit cells and the host graphs of conditions (ii) and (iii) (``geometry``).

Adds to ``data/processed/mlip_v1/<host_id>.json``: ``geometry`` (eps_v, deviatoric, internal RMSD) and, for
tiled hosts, ``files.graphs`` = ``<host_id>_graphs.pt`` ({"mlip": graph, "mlip_rescaled": graph}).
Idempotent: a host whose record already carries the metrics and verified files is skipped.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from dftgnn.graphs.store import sha256_file
from dftgnn.mlip import geometry
from dftgnn.mlip.relax import OUT_DIR, record_ok, relax_host


def host_job(host_id: str, uc_dft, sc_dft, vacancy_indices: list[int], tiling_row: dict, calc, *,
             model_sha256: str, out_dir: Path = OUT_DIR, cutoff: float = 5.0) -> dict:
    from pymatgen.core import Structure

    f = out_dir / f"{host_id}.json"
    if record_ok(host_id, out_dir):
        rec = json.loads(f.read_text())
        if "geometry" in rec and (rec["excluded_from_C3b"] or "graphs" in rec["files"]):
            return {**rec, "skipped": True}
    rec = relax_host(host_id, uc_dft, tiling_row, calc, model_sha256=model_sha256, out_dir=out_dir)
    rec.pop("skipped", None)
    uc_m = Structure.from_dict(json.loads((out_dir / rec["files"]["unit_cell"]["path"]).read_text()))
    rec["geometry"] = geometry.host_metrics(uc_dft, uc_m)
    if rec["tiled"]:
        sc_m = Structure.from_dict(json.loads((out_dir / rec["files"]["supercell"]["path"]).read_text()))
        gs = geometry.condition_graphs(sc_dft, sc_m, vacancy_indices, cutoff)
        rec["rescale_factor"] = float((sc_dft.volume / sc_m.volume) ** (1 / 3))
        p = out_dir / f"{host_id}_graphs.pt"
        torch.save(gs, p)
        rec["files"]["graphs"] = {"path": p.name, "sha256": sha256_file(p)}
    tmp = out_dir / f".{host_id}.json.tmp"
    tmp.write_text(json.dumps(rec, indent=1, sort_keys=True))
    tmp.replace(f)
    return {**rec, "skipped": False}


def load_condition_graphs(host_id: str, out_dir: Path = OUT_DIR) -> dict:
    return torch.load(out_dir / f"{host_id}_graphs.pt", weights_only=True)
