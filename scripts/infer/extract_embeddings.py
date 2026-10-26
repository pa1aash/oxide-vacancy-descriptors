"""Extract readout-input embeddings of S (src/dftgnn/embeddings.py); idempotent per (r, B, kind).

    python scripts/infer/extract_embeddings.py --r 0 --budget 654 --kind trained
    python scripts/infer/extract_embeddings.py --manifest      # aggregate the sidecars into the manifest result
    python scripts/infer/extract_embeddings.py --estimate      # storage estimate, no extraction
"""
from __future__ import annotations

import argparse
import json

import torch

from dftgnn import embeddings as E
from dftgnn import infer
from dftgnn.config import load_config
from dftgnn.io.results import write_result
from dftgnn.split import budget_train, load_split
from dftgnn.train import REPO_ROOT, Store, runindex


def name(r: int, b: int, kind: str) -> str:
    return f"r{r}_B{b}_{kind}"


def estimate() -> dict:
    """Bytes of all npz files: sites x readout width x 4 bytes x kinds (site ids and flags excluded)."""
    import pandas as pd

    cfg = load_config()
    prim = runindex.primary(runindex.load())
    u = pd.read_parquet(REPO_ROOT / "data/processed/universe_v1.parquet")
    n_sites = u.groupby("host_id").size()
    total, rows = 0, 0
    for r in range(cfg.split.n_outer_resamples):
        sp = load_split(f"outer_r{r}")
        for b in cfg.budgets.hosts:
            hp = prim[("S", f"outer_r{r}", b, 0)]["spec"]["hp"]
            h2 = hp["hidden_width"] // 2
            dim = h2 + (2 * h2 if hp["pooling"] == "set2set" else h2) + h2
            n = int(n_sites[budget_train(sp, b)].sum() + n_sites[sp["test"]].sum())
            rows += n
            total += n * dim * 4 * len(E.KINDS)
    return {"bytes_embeddings": total, "site_rows_per_kind": rows, "n_files": 60 * len(E.KINDS)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int)
    ap.add_argument("--budget", type=int)
    ap.add_argument("--kind", choices=E.KINDS)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--manifest", action="store_true")
    ap.add_argument("--estimate", action="store_true")
    a = ap.parse_args(argv)
    if a.estimate:
        est = estimate()
        print(json.dumps(est), f"= {est['bytes_embeddings'] / 2**20:.1f} MiB")
        return
    if a.manifest:
        sides = [json.loads(f.read_text()) for f in sorted(E.OUT_DIR.glob("r*_B*_*.json"))]
        write_result("embeddings_v1_manifest", {"n_files": len(sides), "files": sides,
                                                "bytes": sum((E.OUT_DIR / s["file"]).stat().st_size for s in sides)})
        print(f"manifest: {len(sides)} files")
        return
    torch.set_num_threads(a.threads)
    nm = name(a.r, a.budget, a.kind)
    if E.done(nm):
        print(f"{nm}: already done")
        return
    rec = runindex.primary(runindex.load())[("S", f"outer_r{a.r}", a.budget, 0)]
    arrays = E.extract(a.kind, rec, infer.Sites(Store()), a.r, a.budget, device=torch.device(a.device))
    side = E.write(arrays, nm, {"r": a.r, "budget": a.budget, "kind": a.kind, "checkpoint_run_id": rec["run_id"],
                                "registered_control": a.kind in ("trained", "init0"),
                                "label": "DESCRIPTIVE variance of the random-init control"
                                if a.kind in ("init1", "init2") else None})
    print(f"{nm}: {side['shape']} {side['n_by_split']}")


if __name__ == "__main__":
    main()
