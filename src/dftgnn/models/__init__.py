"""Graph networks S, D-state, D-late, P1 and the staged wrapper P.

Backbone: matgl 4.1.0 MEGNet building blocks, PyG backend, composed here without modification:
``BondExpansion`` (Gaussian, MEGNet defaults of 100 centres and width 0.5 over [0, cutoff + 1]),
``EmbeddingBlock`` (learned element embedding of dimension 16, global state passed through),
the ``MLP`` encoders, ``MEGNetBlock`` with residual connections, and ``Set2SetReadOut``. The wiring
mirrors ``matgl.models.MEGNet``. What changes is the input, because the node features also carry the
vacancy flag, and the readout.

Physics of the readout. The formation energy combines a host-level thermodynamic term with the local
environment of the removed oxygen. The readout vector therefore concatenates the final embedding of the
vacancy node (local), a pooling of all node embeddings and the global state (host). An MLP maps this
vector to the output.

Width mapping (an implementation convention, not a tuned value). With hidden width h, the encoders are
[in, h, h/2] and every MEGNet block has convolution MLPs [., h, h, h/2]. This gives MEGNet's
(64, 32) / (64, 64, 32) layout at h = 64. With readout width w the head is [in, w, w/2, n_out].

Variants (``kind``):
    S        node input = embedding(Z) | vacancy flag; readout -> 1
    D-state  node input also carries the site-electronic descriptors on the vacancy node (zeros
             elsewhere); the initial global state is [0, 0] | host-electronic descriptors
    D-late   S backbone; all 22 descriptors are concatenated to the readout vector
    P1       S backbone and readout; head -> all 22 standardised descriptors

Descriptor encoder. In D-state and D-late each descriptor class passes through its own MLP of ``desc_layers``
layers of width ``desc_width`` before it is injected where the variant injects it. With ``desc_layers`` = 0
(the default) no module is created and the descriptors enter as before, so the default network, its parameter
count and its initialisation are unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch_geometric.utils import scatter

from dftgnn.graphs import STATE_DIM

KINDS = ("S", "D-state", "D-late", "P1")
N_ELEMENTS = 100          # embedding rows indexed by atomic number Z (Z <= 94 in the release)
NODE_EMBED = 16           # matgl MEGNet default dim_node_embedding
RBF_CENTRES = 100         # matgl MEGNet default dim_edge_embedding
RBF_WIDTH = 0.5           # matgl MEGNet default gauss_width
S2S_ITERS, S2S_LAYERS = 2, 1
DESC_OPTIONS = ("desc_layers", "desc_width")


@dataclass(frozen=True)
class HParams:
    """Architecture hyperparameters of the tuned search space.

    The defaults are matgl's MEGNet defaults and serve only for tests, smoke runs and timing.
    They are not tuned values.
    """
    hidden_width: int = 64
    megnet_blocks: int = 3
    dropout: float = 0.0
    readout_mlp_width: int = 64
    pooling: str = "set2set"
    # v2 options.
    init: str = "default"          # "default": PyTorch / matgl init; "kaiming": He-uniform weights, zero biases
    local_readout: bool = False    # add the vacancy node's one-hop environment, computed from raw inputs
    # descriptor encoder of D-state / D-late; 0 layers: none
    desc_layers: int = 0
    desc_width: int = 0

    def to_dict(self) -> dict:
        """The fields, without the descriptor-encoder options at their defaults, so every spec, run id and checkpoint
        config written before they existed is unchanged."""
        d = asdict(self)
        for k in DESC_OPTIONS:
            if d[k] == 0:
                del d[k]
        return d


class Standardiser(nn.Module):
    """Column-wise (x - mean) / sd from training rows; sd below 1e-12 is replaced by 1."""

    def __init__(self, mean: torch.Tensor, sd: torch.Tensor):
        super().__init__()
        sd = torch.where(sd < 1e-12, torch.ones_like(sd), sd)
        self.register_buffer("mean", mean.double().clone())
        self.register_buffer("sd", sd.double().clone())

    @classmethod
    def fit(cls, x: torch.Tensor) -> Standardiser:
        x = x.double()
        return cls(x.mean(0), x.std(0, unbiased=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x.double() - self.mean) / self.sd).to(x.dtype)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        return (z.double() * self.sd + self.mean).to(z.dtype)


class VacancyNet(nn.Module):
    """MEGNet backbone with the vacancy readout; ``kind`` selects S, D-state, D-late or P1."""

    def __init__(self, kind: str, hp: HParams | None = None, *, n_host: int = 12, n_site: int = 10,
                 cutoff: float = 5.0):
        super().__init__()
        hp = hp if hp is not None else HParams()
        from matgl.layers import (
            MLP,
            ActivationFunction,
            BondExpansion,
            EmbeddingBlock,
            MEGNetBlock,
            Set2SetReadOut,
        )

        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}")
        if hp.pooling not in ("set2set", "mean"):
            raise ValueError(f"unknown pooling {hp.pooling!r}")
        if hp.init not in ("default", "kaiming"):
            raise ValueError(f"unknown init {hp.init!r}")
        if hp.desc_layers < 0 or (hp.desc_layers and (hp.desc_width <= 0 or kind not in ("D-state", "D-late"))):
            raise ValueError(f"desc_layers {hp.desc_layers}, desc_width {hp.desc_width} need a D model and a width > 0")
        self.kind, self.hp, self.n_host, self.n_site = kind, hp, n_host, n_site
        self.config = {"kind": kind, "hp": hp.to_dict(), "n_host": n_host, "n_site": n_site,
                       "cutoff": cutoff}
        h, w = hp.hidden_width, hp.readout_mlp_width
        h2 = h // 2
        act = ActivationFunction["softplus2"].value()
        dh_dim, ds_dim = (hp.desc_width, hp.desc_width) if hp.desc_layers else (n_host, n_site)
        node_extra = 1 + (ds_dim if kind == "D-state" else 0)
        state_in = STATE_DIM + (dh_dim if kind == "D-state" else 0)
        late = dh_dim + ds_dim if kind == "D-late" else 0
        n_out = n_host + n_site if kind == "P1" else 1

        self.bond_expansion = BondExpansion(rbf_type="Gaussian", initial=0.0, final=cutoff + 1.0,
                                            num_centers=RBF_CENTRES, width=RBF_WIDTH)
        self.embedding = EmbeddingBlock(degree_rbf=RBF_CENTRES, activation=act,
                                        dim_node_embedding=NODE_EMBED, ntypes_node=N_ELEMENTS,
                                        include_state=True, dim_state_embedding=state_in)
        self.node_encoder = MLP([NODE_EMBED + node_extra, h, h2], act, activate_last=True)
        self.edge_encoder = MLP([RBF_CENTRES, h, h2], act, activate_last=True)
        self.state_encoder = MLP([state_in, h, h2], act, activate_last=True)
        blk = {"conv_hiddens": [h, h, h2], "dropout": hp.dropout, "act": act, "skip": True}
        self.blocks = nn.ModuleList(
            [MEGNetBlock(dims=[h2], **blk)]
            + [MEGNetBlock(dims=[h2, h, h2], **blk) for _ in range(hp.megnet_blocks - 1)])
        self.pool = Set2SetReadOut(h2, n_iters=S2S_ITERS, n_layers=S2S_LAYERS) \
            if hp.pooling == "set2set" else None
        pooled = 2 * h2 if hp.pooling == "set2set" else h2
        self.readout_dim = h2 + pooled + h2
        # Local-environment branch: one message per edge leaving the vacancy node, from the raw distance
        # expansion and the neighbour's element embedding, averaged per graph. It reaches the readout after
        # two layers, so the site's geometry is not attenuated by the message-passing stack.
        self.local = MLP([RBF_CENTRES + NODE_EMBED, h, h2], act, activate_last=True) if hp.local_readout else None
        if self.local is not None:
            self.readout_dim += h2
        self.dropout = nn.Dropout(hp.dropout) if hp.dropout else None
        self.head = MLP([self.readout_dim + late, w, w // 2, n_out], act, activate_last=False)
        # created last, so that with the option off the modules above draw the same initial weights
        self.desc_host_enc = self.desc_site_enc = None
        if hp.desc_layers:
            self.desc_host_enc = MLP([n_host] + [hp.desc_width] * hp.desc_layers, act, activate_last=True)
            self.desc_site_enc = MLP([n_site] + [hp.desc_width] * hp.desc_layers, act, activate_last=True)
        if hp.init == "kaiming":
            for mod in self.modules():
                if isinstance(mod, nn.Linear):
                    nn.init.kaiming_uniform_(mod.weight, nonlinearity="relu")
                    if mod.bias is not None:
                        nn.init.zeros_(mod.bias)

    # ------------------------------------------------------------------ helpers
    def _desc(self, batch, desc_host, desc_site):
        dh = batch.desc_host if desc_host is None else desc_host
        ds = batch.desc_site if desc_site is None else desc_site
        dh, ds = dh.float(), ds.float()
        if self.desc_host_enc is not None:
            dh, ds = self.desc_host_enc(dh), self.desc_site_enc(ds)
        return dh, ds

    def readout_vector(self, batch, desc_host=None, desc_site=None) -> torch.Tensor:
        """[vacancy node | pooled nodes | global state] after the last block (the probe features)."""
        num_graphs = batch.num_graphs
        node_batch = batch.batch
        edge_batch = node_batch[batch.edge_index[0]]
        edge_attr = self.bond_expansion(batch.edge_dist.float())
        node_extra = [batch.vac_flag.float()]
        state = batch.state.float()
        if self.kind == "D-state":
            dh, ds = self._desc(batch, desc_host, desc_site)
            node_extra.append(batch.vac_flag.float() * ds[node_batch])
            state = torch.cat([state, dh], dim=-1)
        node_feat, edge_feat, state_feat = self.embedding(batch.z, edge_attr, state)
        node_feat = self.node_encoder(torch.cat([node_feat, *node_extra], dim=-1))
        edge_feat = self.edge_encoder(edge_feat)
        state_feat = self.state_encoder(state_feat)
        for block in self.blocks:
            edge_feat, node_feat, state_feat = block(batch.edge_index, edge_feat, node_feat, state_feat,
                                                     node_batch, edge_batch, batch.num_nodes, num_graphs)
        vac = node_feat[batch.vacancy_index]
        if self.pool is not None:
            pooled = self.pool(node_feat, node_batch, dim_size=num_graphs)
        else:
            pooled = scatter(node_feat, node_batch, dim=0, dim_size=num_graphs, reduce="mean")
        parts = [vac, pooled.view(num_graphs, -1), state_feat.view(num_graphs, -1)]
        if self.local is not None:
            src, dst = batch.edge_index
            out_vac = batch.vac_flag[src, 0] > 0                      # edges leaving the vacancy node
            emb = self.embedding.layer_node_embedding(batch.z)
            msg = self.local(torch.cat([edge_attr[out_vac], emb[dst[out_vac]]], dim=-1))
            parts.append(scatter(msg, node_batch[src[out_vac]], dim=0, dim_size=num_graphs, reduce="mean"))
        return torch.cat(parts, dim=-1)

    def forward(self, batch, desc_host=None, desc_site=None) -> torch.Tensor:
        vec = self.readout_vector(batch, desc_host, desc_site)
        if self.kind == "D-late":
            dh, ds = self._desc(batch, desc_host, desc_site)
            vec = torch.cat([vec, dh, ds], dim=-1)
        if self.dropout is not None:
            vec = self.dropout(vec)
        out = self.head(vec)
        return out.squeeze(-1) if self.kind != "P1" else out


class StagedP(nn.Module):
    """P = P1 then D. P1's standardised predictions are mapped to D's descriptor scaling.

    The map is ``z_D = (z_P1 * sd_P1 + mean_P1 - mean_D) / sd_D``, applied as an affine map with
    coefficients computed in float64. With identical statistics it is the identity, so P fed with the
    true descriptors (the true-descriptor swap-in) reproduces D exactly.
    """

    def __init__(self, p1: VacancyNet, d: VacancyNet, p1_desc: Standardiser, d_desc: Standardiser):
        super().__init__()
        if p1.kind != "P1" or d.kind not in ("D-state", "D-late"):
            raise ValueError("StagedP needs a P1 and a D model")
        self.p1, self.d, self.n_host = p1, d, d.n_host
        scale = p1_desc.sd / d_desc.sd
        shift = (p1_desc.mean - d_desc.mean) / d_desc.sd
        self.register_buffer("scale", scale.float())
        self.register_buffer("shift", shift.float())
        self.identity = bool(torch.equal(scale, torch.ones_like(scale)) and
                             torch.equal(shift, torch.zeros_like(shift)))

    def descriptors(self, batch) -> torch.Tensor:
        z = self.p1(batch)
        return z if self.identity else z * self.scale + self.shift

    def forward(self, batch) -> torch.Tensor:
        z = self.descriptors(batch)
        return self.d(batch, desc_host=z[:, : self.n_host], desc_site=z[:, self.n_host:])


def hparams_for(kind: str, hp: dict):
    """The hyperparameter dataclass of ``kind`` built from a spec's ``hp`` dict."""
    if kind.startswith("cgcnn-"):
        from dftgnn.models.cgcnn import CGCNNParams

        return CGCNNParams(**hp)
    return HParams(**hp)


def build_model(kind: str, hp=None, **kw) -> nn.Module:
    """VacancyNet for S / D-state / D-late / P1; the CGCNN cross-check for cgcnn-S / cgcnn-D."""
    if kind.startswith("cgcnn-"):
        from dftgnn.models.cgcnn import CGCNN

        return CGCNN(kind, hp, **kw)
    return VacancyNet(kind, hp, **kw)


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
