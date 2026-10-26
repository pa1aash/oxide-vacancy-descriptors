"""Tiling map for every host of universe v1 (no MACE involved).

    nice -n 15 python scripts/mlip/build_tiling.py

Writes data/processed/tiling_map_v1.parquet (gitignored; permutation and translation per host),
data/tiling_summary_v1.csv (tracked) and results/tiling_v1.json (counts and reasons) via write_result.
Procedure and tolerances: src/dftgnn/mlip/tiling.py (translation only, then one lattice point operation).
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd

from dftgnn.graphs.store import sha256_file
from dftgnn.io.results import write_result
from dftgnn.mlip import release, tiling

ROOT = Path(__file__).resolve().parents[2]
MAP = ROOT / "data" / "processed" / "tiling_map_v1.parquet"
SUMMARY = ROOT / "data" / "tiling_summary_v1.csv"


def main() -> None:
    hosts = release.host_table()
    cells = release.unit_cells()
    rows = []
    for k, h in enumerate(hosts.itertuples()):
        ci = (release.DB / "site_info" / h.formula / "cell_info.txt").read_text()
        rec = tiling.map_host(cells[h.host_id], release.read_supercell(h.cif), ci)
        rows.append({"host_id": h.host_id, "formula": h.formula, **rec})
        if k % 100 == 0:
            print(f"{k}/{len(hosts)}", flush=True)
    t = pd.DataFrame(rows)
    t["excluded_from_C3b"] = t.status != "ok"
    if "point_op" not in t:
        t["point_op"] = None
    t["point_op"] = t.point_op.map(lambda v: None if v is None or (isinstance(v, float) and math.isnan(v)) else v)
    for c in ("M", "T_release", "perm", "translation", "point_op"):
        t[c] = t[c].map(lambda v: None if v is None else json.dumps(v))
    MAP.parent.mkdir(parents=True, exist_ok=True)
    t.to_parquet(MAP, index=False)
    cols = ["host_id", "formula", "status", "reason", "frame", "T_agrees", "M", "T_release", "n_uc", "n_sc",
            "max_residual_A", "M_direct_max_dev", "excluded_from_C3b", "point_op"]
    csv_bytes = t[cols].to_csv(index=False, float_format="%.6g").encode()   # written after write_result (clean tree)
    counts = dict(Counter(t.status))
    reasons = dict(Counter(t.reason[t.status != "ok"]))
    print(f"tiling: {counts}; failure reasons {reasons}; frames {dict(Counter(t.frame.fillna('none')))}; "
          f"T disagrees for {int((t.T_agrees == False).sum())} hosts; max residual over ok hosts "
          f"{t.max_residual_A[t.status == 'ok'].max():.2e} A")
    for r in t[t.status != "ok"].itertuples():
        print(f"  failed {r.host_id} {r.formula}: {r.reason} (n_uc {r.n_uc}, n_sc {r.n_sc}, dev {r.M_direct_max_dev:.3g})")
    write_result("tiling_v1", {
        "n_hosts": len(t), "status_counts": counts, "failure_reasons": reasons,
        "frames": dict(Counter(t.frame.fillna("none"))),
        "n_release_T_disagrees": int((t.T_agrees == False).sum()),
        "hosts_release_T_disagrees": sorted(t.formula[t.T_agrees == False]),
        "max_residual_ok_A": float(t.max_residual_A[t.status == "ok"].max()),
        "excluded_from_C3b": sorted(t.host_id[t.excluded_from_C3b]),
        "point_op_hosts": {r.formula: r.point_op for r in t[t.point_op.notna()].itertuples()},
        "rule": "translation only, then one lattice point operation (docs/deviations.md 2026-10-11)",
        "tolerances": {"matrix_relative": tiling.REL_TOL, "residual_A": tiling.RESID_TOL_A},
        "summary": {"path": str(SUMMARY.relative_to(ROOT)), "sha256": hashlib.sha256(csv_bytes).hexdigest()},
        "map": {"path": str(MAP.relative_to(ROOT)), "sha256": sha256_file(MAP), "tracked": False}})
    SUMMARY.write_bytes(csv_bytes)


if __name__ == "__main__":
    main()
