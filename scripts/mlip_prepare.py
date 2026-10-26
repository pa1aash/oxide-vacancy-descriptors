"""MLIP inputs for every host. Runs where the release is unpacked. Writes data/processed/mlip_inputs.json.gz
(gitignored) and results/mlip_inputs.json (counts, excluded hosts with reasons, sha256 of the inputs file).

    python scripts/mlip_prepare.py
"""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dftgnn.config import load_config
from dftgnn.data.universe import read_universe, release_paths
from dftgnn.io.results import write_result
from dftgnn.mlip import nrp as mlip

OUT = ROOT / "data" / "processed" / "mlip_inputs.json.gz"


def main() -> None:
    warnings.filterwarnings("ignore")
    cfg = load_config()
    uni = read_universe(cfg)
    res = mlip.prepare_inputs(uni, release_paths(cfg).root)
    blob = gzip.compress(json.dumps(res, sort_keys=True).encode(), mtime=0)
    OUT.write_bytes(blob)
    payload = {"n_hosts": uni.host_id.nunique(), "n_mapped": len(res["hosts"]), "excluded": res["excluded"],
               "match_tol_A": res["match_tol_A"], "inputs_file": str(OUT.relative_to(ROOT)),
               "inputs_sha256": hashlib.sha256(blob).hexdigest(),
               "n_nonidentity_frame": sum(1 for h in res["hosts"].values()
                                          if max(abs(a - b) for ra, rb in zip(h["rotation"], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], strict=True)
                                                 for a, b in zip(ra, rb, strict=True)) > 1e-6)}
    print({k: v for k, v in payload.items() if k != "excluded"}, "excluded:", payload["excluded"])
    print(write_result("mlip_inputs", payload, config=cfg))


if __name__ == "__main__":
    main()
