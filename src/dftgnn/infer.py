"""Inference with stored checkpoints on site sets whose host graphs may be swapped (no training).

Used by the P evaluation, the geometry conditions (section 9) and the embedding extraction (section 10). Every
model sees exactly what it saw in training: the vacancy flag on the site's atom, and for D the site's DFT
descriptors standardised with that checkpoint's training statistics. A prediction is mapped back to eV with
the checkpoint's target statistics.

``graphs`` maps host_id to the host-graph dict of the condition being evaluated (``dftgnn.graphs.host_graph``
format); condition (i) uses the tracked graph store.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

from dftgnn.graphs import site_data
from dftgnn.models import StagedP, Standardiser
from dftgnn.train import REPO_ROOT, load_checkpoint


def checkpoint_path(rec: dict, root: Path = REPO_ROOT) -> Path:
    return root / rec["checkpoint"]["path"]


def load(rec: dict, device=None, verify: bool = True):
    """(model in eval mode, checkpoint dict) of a run-index record; the sha256 is checked."""
    from dftgnn.graphs.store import sha256_file

    p = checkpoint_path(rec)
    if verify and sha256_file(p) != rec["checkpoint"]["sha256"]:
        raise ValueError(f"checkpoint {p} does not match its recorded sha256")
    return load_checkpoint(p, device)


def staged(p1_rec: dict, d_rec: dict, device=None) -> tuple[StagedP, dict]:
    p1, ck1 = load(p1_rec, device)
    d, ckd = load(d_rec, device)
    p = StagedP(p1, d, Standardiser(ck1["desc_mean"], ck1["desc_sd"]), Standardiser(ckd["desc_mean"], ckd["desc_sd"]))
    return p.eval(), ckd


class Sites:
    """Site table of the graph store: site_id, host_id, vacancy index, target, raw descriptors."""

    def __init__(self, store):
        self.store = store
        s = store.sites
        self.site_id = list(s["site_id"])
        self.host_id = store.site_host
        self.vac = s["vacancy_atom_index"]
        self.target = s["target"]
        self.desc = s["desc"]
        self.n_host = len(s["desc_host_names"])

    def positions(self, hosts) -> np.ndarray:
        return self.store.positions(hosts)

    def dft_graphs(self, hosts) -> dict:
        idx = {h: i for i, h in enumerate(self.store.meta["host_ids"])}
        return {h: self.store.graphs[idx[h]] for h in hosts}


def _batch(sites: Sites, graphs: dict, pos, desc_std: torch.Tensor) -> Batch:
    items = []
    for p in pos:
        p = int(p)
        g = graphs[sites.host_id[p]]
        d = desc_std[p]
        items.append(site_data(g, int(sites.vac[p]), y=float(sites.target[p]), desc_host=d[: sites.n_host],
                               desc_site=d[sites.n_host:], site_pos=p))
    return Batch.from_data_list(items)


@torch.no_grad()
def predict(net, ck: dict, sites: Sites, graphs: dict, pos, *, batch_size: int = 64, device=None,
            what: str = "output") -> np.ndarray:
    """Predictions at site positions ``pos`` with host graphs ``graphs``.

    ``what``: "output" (E_f in eV for S / D / P, raw descriptors for P1), or "readout" (the readout input
    vector, ``net.readout_vector``). Descriptors are standardised with ``ck``'s statistics (for P, the D
    checkpoint's; P replaces them with P1's predictions internally).
    """
    device = device or torch.device("cpu")
    desc_std = Standardiser(ck["desc_mean"].cpu(), ck["desc_sd"].cpu())(sites.desc).float()
    out = []
    for k in range(0, len(pos), batch_size):
        b = _batch(sites, graphs, pos[k:k + batch_size], desc_std).to(device)
        out.append((net.readout_vector(b) if what == "readout" else net(b)).cpu().double())
    z = torch.cat(out)
    if what == "readout":
        return z.numpy()
    if z.dim() == 2:                                   # P1: standardised descriptors -> raw
        return Standardiser(ck["desc_mean"].cpu(), ck["desc_sd"].cpu()).inverse(z).numpy()
    return Standardiser(ck["target_mean"].cpu(), ck["target_sd"].cpu()).inverse(z.view(-1, 1)).view(-1).numpy()


def task_id(key: dict) -> str:
    """Deterministic 16-hex id of an inference task (its key includes the code SHA)."""
    return hashlib.sha256(json.dumps(key, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
