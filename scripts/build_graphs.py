"""Build the host-graph store from universe v1, then record it.

    python scripts/build_graphs.py           build shards + tracked manifest (clean tree required)
    python scripts/build_graphs.py --result  verify the store and write results/graphs_v1.json

The build writes the tracked manifest data/graphs_v1_manifest.sha256, which must be committed before
``--result`` can write from a clean tree.
"""
from __future__ import annotations

import argparse
import subprocess

from dftgnn.config import load_config
from dftgnn.data.universe import read_universe
from dftgnn.graphs import store
from dftgnn.io.results import write_result


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=store.REPO_ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def build() -> None:
    cfg = load_config()
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("working tree is dirty; the manifest records the code SHA, commit first")
    uni = read_universe(cfg)
    graphs, sites, host_ids = store.build_from_universe(uni, cfg.graph.cutoff_A)
    print(store.write_store(graphs, sites, host_ids, cfg.graph.cutoff_A, code_sha=_git("rev-parse", "HEAD")))


def result() -> None:
    cfg = load_config()
    graphs, sites, meta = store.load_store()
    header, files = store.read_manifest()
    payload = {
        "manifest_header": header, "manifest_sha256": store.manifest_sha(), "files": files,
        "store_bytes": sum((store.STORE_DIR / f).stat().st_size for f in files),
        "n_hosts": len(graphs), "n_sites": len(sites["site_id"]),
        "stats": store.graph_stats(graphs), "state_dim": meta["state_dim"],
        "descriptors": {"host": sites["desc_host_names"], "site": sites["desc_site_names"]},
        "converter": "matgl.ext.pymatgen.Structure2Graph (PyG backend)",
    }
    print(write_result("graphs_v1", payload, config=cfg))
    print(payload["store_bytes"], payload["stats"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", action="store_true")
    result() if ap.parse_args().result else build()
