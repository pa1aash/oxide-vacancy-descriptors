"""Relax every mapped host unit cell with MACE-MP-0.

    python scripts/mlip_relax.py --shard 0 --n-shards 4 --device cuda \
        --model medium --dtype float64 --fmax 0.01 --max-steps 500

Runs in the MACE environment (``/workspace/venv-mlip`` on NRP); the values come from ``config.mlip`` and are passed
on the command line so this script needs only mace, ase and pymatgen. Hosts are taken in sorted order and host i
belongs to shard ``i % n_shards``. Writes ``results/mlip/relaxed_shard<k>.json.gz``: for each host the relaxed
unit cell (pymatgen dict) and the relaxation record (converged, steps, final fmax, energy), plus the MACE model
name, version, dtype and the sha256 of the inputs file. Resumable: hosts already in the shard file are skipped.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

INPUTS = ROOT / "data" / "processed" / "mlip_inputs.json.gz"
OUT_DIR = ROOT / "results" / "mlip"


def _save(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(json.dumps(obj, sort_keys=True).encode(), mtime=0))
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default="medium")
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--fmax", type=float, default=0.01)
    ap.add_argument("--max-steps", type=int, default=500)
    a = ap.parse_args()

    import mace
    from mace.calculators import mace_mp
    from pymatgen.core import Structure

    from dftgnn.mlip.nrp import relax_unit_cell

    blob = INPUTS.read_bytes()
    inputs = json.loads(gzip.decompress(blob))["hosts"]
    hosts = sorted(inputs)[a.shard::a.n_shards]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"relaxed_shard{a.shard}.json.gz"
    out = json.loads(gzip.decompress(out_path.read_bytes())) if out_path.exists() else {
        "meta": {"model": f"mace-mp-0 {a.model}", "mace_version": mace.__version__, "dtype": a.dtype,
                 "optimiser": "FIRE on FrechetCellFilter", "fmax_eV_per_A": a.fmax, "max_steps": a.max_steps,
                 "inputs_sha256": hashlib.sha256(blob).hexdigest(), "shard": a.shard, "n_shards": a.n_shards},
        "hosts": {}}
    calc = mace_mp(model=a.model, default_dtype=a.dtype, device=a.device)
    t0 = time.time()
    for n, hid in enumerate(hosts):
        if hid in out["hosts"]:
            continue
        unit = Structure.from_dict(inputs[hid]["unit_cell"])
        t = time.time()
        try:
            relaxed, info = relax_unit_cell(unit, calc, a.fmax, a.max_steps)
            out["hosts"][hid] = {"unit_cell": relaxed.as_dict(), **info, "wall_s": round(time.time() - t, 2)}
        except Exception as exc:  # noqa: BLE001 - a host that cannot be relaxed is recorded, not fatal
            out["hosts"][hid] = {"error": repr(exc)[:500]}
        if n % 10 == 0 or n == len(hosts) - 1:
            _save(out_path, out)
            print(f"shard {a.shard}: {len(out['hosts'])}/{len(hosts)} hosts, {time.time() - t0:.0f}s", flush=True)
    _save(out_path, out)
    print(f"shard {a.shard}: done, {len(out['hosts'])} hosts", flush=True)


if __name__ == "__main__":
    main()
