"""Sharded on-disk graph store and its tracked sha256 manifest.

Layout of ``data/processed/graphs_v1/`` (gitignored):
    hosts_<k>.pt   list of host-graph dicts for hosts ``k*SHARD .. (k+1)*SHARD - 1`` (sorted host_id)
    sites.pt       column-wise site records (``dftgnn.graphs.site_records``)
    meta.json      host ids in store order, cutoff, state dimension, shard list

On disk, edge indices are int32 and image offsets int8. Both conversions are lossless (checked on
write), and loading restores int64. The tracked manifest ``data/graphs_v1_manifest.sha256`` lists one
``<sha256>  <file>`` line per file, under ``# key: value`` header lines: the cutoff, the code git SHA
that built the store, the universe IDs sha256 and the matgl version.
"""
from __future__ import annotations

import hashlib
import json
from importlib import metadata
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

from dftgnn.graphs import STATE_DIM, host_graph, load_structure, site_records

REPO_ROOT = Path(__file__).resolve().parents[3]
STORE_DIR = REPO_ROOT / "data" / "processed" / "graphs_v1"
MANIFEST = REPO_ROOT / "data" / "graphs_v1_manifest.sha256"
UNIVERSE_IDS_SHA = REPO_ROOT / "data" / "universe_v1_ids.csv.sha256"
SHARD = 100


def _build_one(args) -> dict:
    cif, cutoff = args
    return host_graph(load_structure(cif), cutoff)


def build_graphs(cifs: list[str], cutoff: float, processes: int | None = None) -> list[dict]:
    paths = [str(REPO_ROOT / c) for c in cifs]
    with get_context("spawn").Pool(processes) as pool:
        return pool.map(_build_one, [(p, cutoff) for p in paths], chunksize=8)


def _pack(g: dict) -> dict:
    ei, off = g["edge_index"], g["pbc_offset"]
    if ei.max() >= 2**31 or off.abs().max() > 127:
        raise ValueError("edge index or offset outside the packed range")
    return {**g, "edge_index": ei.to(torch.int32), "pbc_offset": off.to(torch.int8)}


def _unpack(g: dict) -> dict:
    return {**g, "edge_index": g["edge_index"].long(), "pbc_offset": g["pbc_offset"].long()}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_store(graphs: list[dict], sites: dict, host_ids: list[str], cutoff: float, *,
                code_sha: str, out_dir: Path = STORE_DIR, manifest: Path = MANIFEST) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("hosts_*.pt"):
        old.unlink()
    files = []
    for k in range(0, len(graphs), SHARD):
        name = f"hosts_{k // SHARD:03d}.pt"
        torch.save([_pack(g) for g in graphs[k:k + SHARD]], out_dir / name)
        files.append(name)
    torch.save(sites, out_dir / "sites.pt")
    meta = {"host_ids": host_ids, "cutoff_A": cutoff, "state_dim": STATE_DIM, "shard_size": SHARD,
            "host_shards": files, "n_hosts": len(graphs), "n_sites": len(sites["site_id"])}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    header = {
        "cutoff_A": cutoff,
        "code_git_sha": code_sha,
        "universe_ids_sha256": UNIVERSE_IDS_SHA.read_text().split()[0],
        "matgl": metadata.version("matgl"),
        "n_hosts": len(graphs),
        "n_sites": len(sites["site_id"]),
    }
    lines = [f"# {k}: {v}" for k, v in header.items()]
    lines += [f"{sha256_file(out_dir / f)}  {f}" for f in [*files, "sites.pt", "meta.json"]]
    manifest.write_text("\n".join(lines) + "\n")
    return header


def read_manifest(manifest: Path = MANIFEST) -> tuple[dict, dict[str, str]]:
    header, files = {}, {}
    for line in manifest.read_text().splitlines():
        if line.startswith("# "):
            k, v = line[2:].split(": ", 1)
            header[k] = v
        elif line.strip():
            sha, name = line.split("  ", 1)
            files[name] = sha
    return header, files


def manifest_sha(manifest: Path = MANIFEST) -> str:
    return sha256_file(manifest)


def verify_store(out_dir: Path = STORE_DIR, manifest: Path = MANIFEST) -> list[str]:
    """Names of files whose sha256 differs from the manifest (missing files included)."""
    _, files = read_manifest(manifest)
    bad = [n for n, sha in files.items() if not (out_dir / n).is_file() or sha256_file(out_dir / n) != sha]
    return bad


def load_store(out_dir: Path = STORE_DIR, *, verify: bool = True,
               manifest: Path = MANIFEST) -> tuple[list[dict], dict, dict]:
    """(graphs in store order, site records, meta). Raises if verification fails."""
    if verify:
        bad = verify_store(out_dir, manifest)
        if bad:
            raise ValueError(f"graph store does not match {manifest.name}: {bad}")
    meta = json.loads((out_dir / "meta.json").read_text())
    graphs = []
    for f in meta["host_shards"]:
        graphs += [_unpack(g) for g in torch.load(out_dir / f, weights_only=True)]
    sites = torch.load(out_dir / "sites.pt", weights_only=True)
    return graphs, sites, meta


def graph_stats(graphs: list[dict]) -> dict:
    atoms = np.array([g["z"].shape[0] for g in graphs])
    edges = np.array([g["edge_index"].shape[1] for g in graphs])
    deg = [np.bincount(g["edge_index"][0].numpy(), minlength=g["z"].shape[0]) for g in graphs]

    def mmm(x):
        return {"min": int(x.min()), "median": float(np.median(x)), "max": int(x.max())}

    return {"n_graphs": len(graphs), "atoms": mmm(atoms), "edges": mmm(edges),
            "max_degree": int(max(d.max() for d in deg)),
            "min_degree": int(min(d.min() for d in deg)),
            "isolated_atoms": int(sum((d == 0).sum() for d in deg)),
            "max_edge_dist_A": float(max(g["edge_dist"].max() for g in graphs)),
            "min_edge_dist_A": float(min(g["edge_dist"].min() for g in graphs))}


def build_from_universe(universe, cutoff: float, processes: int | None = None):
    """(graphs, sites, host_ids) for every host of ``universe`` (host_id order)."""
    hosts = universe.groupby("host_id").supercell_cif_path.agg(["first", "nunique"]).sort_index()
    if (hosts["nunique"] != 1).any():
        raise ValueError("a host maps to more than one supercell CIF")
    host_ids = list(hosts.index)
    graphs = build_graphs(list(hosts["first"]), cutoff, processes)
    return graphs, site_records(universe, host_ids), host_ids
