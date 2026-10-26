"""Physics. The check asks whether the advantage of DFT descriptors over a structure-only graph model is a
property of the MEGNet backbone or of the problem. CGCNN replaces MEGNet's edge, so the host-level information
reaches the readout only through the pooled atom embeddings.

Architecture defaults are those of the original repository (txie-93/cgcnn, commit
f42ab233c4ee0c416879d6bc2d22a264418413ad, ``main.py`` and ``cgcnn/model.py``): atom feature length 64, 3
convolutions, hidden feature length 128 after pooling, 1 hidden layer, Gaussian distance filter with centres
from 0 to the cutoff in steps of 0.2 A and width equal to the step, softplus activations. The graphs are the
same 5.0 A radius graphs as the MEGNet models (the original uses the 12 nearest neighbours within 8 A).
Convolutions are ``torch_geometric.nn.CGConv`` (sum aggregation, batch norm on the aggregated message,
residual), each followed by a softplus as in the original ``ConvLayer``.

Inputs: a learned embedding of the atomic number plus a learned vector times the vacancy flag, which equals
the original's linear embedding of [one-hot(Z) | flag]. Readout: [vacancy-node embedding | mean-pooled
atom embeddings] -> Linear(h_fea_len) -> softplus -> Linear(1). cgcnn-D appends the 22 standardised D
descriptors (host-electronic then site-electronic) to the readout vector before the MLP.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch_geometric.utils import scatter

KINDS = ("cgcnn-S", "cgcnn-D")
N_ELEMENTS = 100
SOURCE = {"repository": "https://github.com/txie-93/cgcnn",
          "commit": "f42ab233c4ee0c416879d6bc2d22a264418413ad",
          "files": ["https://raw.githubusercontent.com/txie-93/cgcnn/master/main.py",
                    "https://raw.githubusercontent.com/txie-93/cgcnn/master/cgcnn/model.py",
                    "https://raw.githubusercontent.com/txie-93/cgcnn/master/cgcnn/data.py"]}


@dataclass(frozen=True)
class CGCNNParams:
    atom_fea_len: int = 64        # main.py --atom-fea-len
    n_conv: int = 3               # main.py --n-conv
    h_fea_len: int = 128          # main.py --h-fea-len
    n_h: int = 1                  # main.py --n-h
    gdf_step: float = 0.2         # data.py CIFData step (GaussianDistance var defaults to step)

    def to_dict(self) -> dict:
        return asdict(self)


class GaussianDistance(nn.Module):
    """exp(-(d - mu_k)^2 / var^2), mu_k = 0, step, ..., cutoff (original ``GaussianDistance``)."""

    def __init__(self, dmax: float, step: float):
        super().__init__()
        self.register_buffer("mu", torch.arange(0.0, dmax + step, step))
        self.var = step

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        return torch.exp(-((d.view(-1, 1) - self.mu) ** 2) / self.var ** 2)


class CGCNN(nn.Module):
    def __init__(self, kind: str, hp: CGCNNParams | None = None, *, n_host: int = 12, n_site: int = 10,
                 cutoff: float = 5.0):
        super().__init__()
        from torch_geometric.nn import CGConv

        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}")
        hp = hp if hp is not None else CGCNNParams()
        self.kind, self.hp, self.n_host, self.n_site = kind, hp, n_host, n_site
        self.config = {"kind": kind, "hp": hp.to_dict(), "n_host": n_host, "n_site": n_site, "cutoff": cutoff}
        a = hp.atom_fea_len
        self.gdf = GaussianDistance(cutoff, hp.gdf_step)
        nbr = self.gdf.mu.numel()
        self.embedding = nn.Embedding(N_ELEMENTS, a)
        self.flag_embedding = nn.Linear(1, a, bias=False)
        self.convs = nn.ModuleList([CGConv(a, dim=nbr, aggr="add", batch_norm=True) for _ in range(hp.n_conv)])
        self.act = nn.Softplus()
        self.readout_dim = 2 * a
        late = n_host + n_site if kind == "cgcnn-D" else 0
        layers: list[nn.Module] = [nn.Linear(self.readout_dim + late, hp.h_fea_len), nn.Softplus()]
        for _ in range(hp.n_h - 1):
            layers += [nn.Linear(hp.h_fea_len, hp.h_fea_len), nn.Softplus()]
        layers.append(nn.Linear(hp.h_fea_len, 1))
        self.head = nn.Sequential(*layers)

    def readout_vector(self, batch, desc_host=None, desc_site=None) -> torch.Tensor:
        """[vacancy node | mean-pooled atoms] after the last convolution."""
        x = self.embedding(batch.z) + self.flag_embedding(batch.vac_flag.float())
        e = self.gdf(batch.edge_dist.float())
        for conv in self.convs:
            x = self.act(conv(x, batch.edge_index, e))
        pooled = scatter(x, batch.batch, dim=0, dim_size=batch.num_graphs, reduce="mean")
        return torch.cat([x[batch.vacancy_index], pooled], dim=-1)

    def forward(self, batch, desc_host=None, desc_site=None) -> torch.Tensor:
        vec = self.readout_vector(batch)
        if self.kind == "cgcnn-D":
            dh = batch.desc_host if desc_host is None else desc_host
            ds = batch.desc_site if desc_site is None else desc_site
            vec = torch.cat([vec, dh.float(), ds.float()], dim=-1)
        return self.head(vec).squeeze(-1)
