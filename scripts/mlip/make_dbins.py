"""Cache the DFT unit cells of universe v1 and build the host d-electron bin table.

    nice -n 15 python scripts/mlip/make_dbins.py

Writes data/processed/unit_cells_v1.json (gitignored), data/host_dbins_v1.csv (tracked) and
results/host_dbins_v1.json (counts per bin) via write_result. Rule: src/dftgnn/mlip/dbins.py and the
clarification in (2026-10-10).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from dftgnn.graphs.store import sha256_file
from dftgnn.io.results import write_result
from dftgnn.mlip import dbins, release

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "host_dbins_v1.csv"


def main() -> None:
    hosts = release.host_table()
    cells = release.unit_cells() if release.UNIT_CELLS.is_file() else None
    if cells is None or sorted(cells) != list(hosts.host_id):
        release.build_unit_cells(hosts)
        cells = release.unit_cells()
    rows = []
    for h in hosts.itertuples():
        rows.append({"host_id": h.host_id, "formula": h.formula, **dbins.host_bin(cells[h.host_id])})
    t = pd.DataFrame(rows)[["host_id", "formula", "oxidation_method", "cation_states", "cation_d_counts", "bin",
                            "reason"]]
    t.to_csv(OUT, index=False)
    counts = dict(sorted(Counter(t.bin).items()))
    methods = dict(sorted(Counter(t.oxidation_method).items()))
    print(f"{len(t)} hosts; bins {counts}; oxidation methods {methods}")
    for r in t[t.bin == "unassigned"].itertuples():
        print(f"  unassigned {r.host_id} {r.formula}: {r.reason}")
    write_result("host_dbins_v1", {"n_hosts": len(t), "bin_counts": counts, "oxidation_methods": methods,
                                   "n_unassigned": int((t.bin == "unassigned").sum()),
                                   "table": {"path": str(OUT.relative_to(ROOT)), "sha256": sha256_file(OUT)}},
                 allow_dirty=False)


if __name__ == "__main__":
    main()
