"""Host graphs, per-site records and site batches.

One graph per host, built from the pristine release supercell (the CIF path comes from the universe
table; nothing is globbed). The periodic radius graph (cutoff ``config.graph.cutoff_A`` = 5.0 A) comes
from matgl 4.1.0's PyG converter ``Structure2Graph``, which returns directed edges in both directions
with integer image offsets of the destination atom. Edge distances are computed here in float64:
``|pos[dst] + offset @ lattice - pos[src]|``.

Global state: the converter's default MEGNet state, ``[0.0, 0.0]`` (``STATE_DIM`` = 2).

A *site record* references its host by ``host_idx`` and carries the vacancy atom index, the target and
the 22 D descriptors (host-electronic then site-electronic, in ``descriptor_names`` order). The P1
target vector is the same 22 descriptors: P1 predicts every D descriptor. Standardisation happens at
training time with training-set statistics only.

``collate_sites`` turns a list of site records into a PyG ``Batch``: each site gets its own copy of
the host graph, with a binary vacancy flag on the vacancy node, so one host may appear several times
in a batch.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch, Data

STATE_DIM = 2
STATE_INIT = (0.0, 0.0)
HOST_CLASS = "host-electronic DFT"
SITE_CLASS = "site-electronic DFT"


def load_structure(cif_path):
    from pymatgen.core import Structure

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Structure.from_file(cif_path)


def edges_matgl(structure, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    """(edge_index (2, E) int64, image offsets (E, 3) int64) from matgl's ``Structure2Graph``."""
    from matgl.ext.pymatgen import Structure2Graph

    els = tuple(sorted({e.symbol for e in structure.composition}))
    g, _, _ = Structure2Graph(element_types=els, cutoff=cutoff).get_graph(structure)
    off = g.pbc_offset.numpy()
    if not np.allclose(off, np.round(off)):
        raise ValueError("non-integer image offsets")
    return g.edge_index.numpy().astype(np.int64), np.round(off).astype(np.int64)


def edge_distances(pos: np.ndarray, lattice: np.ndarray, edge_index: np.ndarray,
                   offsets: np.ndarray) -> np.ndarray:
    src, dst = edge_index
    return np.linalg.norm(pos[dst] + offsets @ lattice - pos[src], axis=1)


def host_graph(structure, cutoff: float) -> dict[str, torch.Tensor]:
    """Graph tensors of one pristine host supercell (see module docstring)."""
    lat = np.array(structure.lattice.matrix, dtype=np.float64)
    pos = np.array(structure.cart_coords, dtype=np.float64)
    ei, off = edges_matgl(structure, cutoff)
    order = np.lexsort((off[:, 2], off[:, 1], off[:, 0], ei[1], ei[0]))   # canonical edge order
    ei, off = ei[:, order], off[order]
    dist = edge_distances(pos, lat, ei, off)
    return {
        "z": torch.tensor([s.specie.Z for s in structure], dtype=torch.int64),
        "pos": torch.from_numpy(pos),
        "lattice": torch.from_numpy(lat),
        "edge_index": torch.from_numpy(ei),
        "edge_dist": torch.from_numpy(dist),
        "pbc_offset": torch.from_numpy(off),
        "state": torch.tensor([STATE_INIT], dtype=torch.float64),
    }


def descriptor_names(desc_class: dict[str, str]) -> tuple[list[str], list[str]]:
    """(host-electronic, site-electronic) ``desc_*`` column names, each sorted."""
    host = sorted(c for c, k in desc_class.items() if k == HOST_CLASS)
    site = sorted(c for c, k in desc_class.items() if k == SITE_CLASS)
    return host, site


def site_records(universe: pd.DataFrame, host_ids: list[str]) -> dict:
    """Column-wise site records in universe (site_id) order, referencing hosts by position."""
    hidx = {h: i for i, h in enumerate(host_ids)}
    host, site = descriptor_names(universe.attrs["descriptor_class"])
    desc = np.array(universe[host + site].to_numpy(np.float64), copy=True)
    return {
        "site_id": list(universe.site_id),
        "host_idx": torch.tensor([hidx[h] for h in universe.host_id], dtype=torch.int64),
        "vacancy_atom_index": torch.tensor(np.array(universe.vacancy_atom_index), dtype=torch.int64),
        "target": torch.tensor(np.array(universe.target_Ef_eV), dtype=torch.float64),
        "desc": torch.from_numpy(desc),
        "p1_target": torch.from_numpy(desc.copy()),
        "desc_host_names": host,
        "desc_site_names": site,
    }


class SiteData(Data):
    """PyG Data of one (host, vacancy site) example; ``vacancy_index`` is offset like ``edge_index``."""

    def __inc__(self, key, value, *args, **kwargs):
        if key == "vacancy_index":
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)


def site_data(g: dict[str, torch.Tensor], vac: int, *, y: float | torch.Tensor = 0.0,
              desc_host: torch.Tensor | None = None, desc_site: torch.Tensor | None = None,
              site_pos: int = -1) -> SiteData:
    """One example: host graph tensors plus the vacancy flag and the site's own vectors."""
    n = g["z"].shape[0]
    flag = torch.zeros(n, 1, dtype=torch.float32)
    flag[vac, 0] = 1.0
    d = SiteData(
        z=g["z"], pos=g["pos"].float(), edge_index=g["edge_index"],
        edge_dist=g["edge_dist"].float(), pbc_offset=g["pbc_offset"].float(),
        vac_flag=flag, vacancy_index=torch.tensor([vac]), state=g["state"].float(),
        y=torch.as_tensor([y], dtype=torch.float32), site_pos=torch.tensor([site_pos]),
        num_nodes=n,
    )
    if desc_host is not None:
        d.desc_host = desc_host.float().view(1, -1)
    if desc_site is not None:
        d.desc_site = desc_site.float().view(1, -1)
    return d


def collate_sites(graphs: list[dict], sites: dict, positions, *, y=None, desc=None) -> Batch:
    """Batch of the site records at ``positions``.

    ``y`` (n_sites,) and ``desc`` (n_sites, 22) override the stored target and descriptors (the
    training loop passes standardised values); by default the raw stored values are used.
    """
    y = sites["target"] if y is None else y
    desc = sites["desc"] if desc is None else desc
    nh = len(sites["desc_host_names"])
    items = []
    for p in positions:
        p = int(p)
        g = graphs[int(sites["host_idx"][p])]
        items.append(site_data(g, int(sites["vacancy_atom_index"][p]), y=y[p],
                               desc_host=desc[p, :nh], desc_site=desc[p, nh:], site_pos=p))
    return Batch.from_data_list(items)
