"""Test cgcnn module."""
from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch

from dftgnn.graphs import host_graph, site_data
from dftgnn.models import build_model, hparams_for, n_parameters
from dftgnn.models.cgcnn import KINDS, CGCNNParams

N_HOST, N_SITE = 12, 10


def _mgo(n=2, rattle=0.0):
    import numpy as np
    from pymatgen.core import Lattice, Structure

    s = Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.21), ["Mg", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    s.make_supercell([n, n, n])
    if rattle:
        rng = np.random.default_rng(0)
        for i in range(len(s)):
            s.translate_sites([i], rng.normal(0, rattle, 3), frac_coords=False)
    return s


@pytest.fixture(scope="module")
def graph():
    return host_graph(_mgo(), 5.0)


@pytest.fixture(scope="module")
def rattled():
    return host_graph(_mgo(rattle=0.08), 5.0)


def _ox(g):
    return torch.nonzero(g["z"] == 8).flatten().tolist()


def _batch(g, vacs, desc=None):
    torch.manual_seed(0)
    desc = torch.randn(len(vacs), N_HOST + N_SITE) if desc is None else desc
    return Batch.from_data_list([site_data(g, v, y=1.0, desc_host=desc[i, :N_HOST], desc_site=desc[i, N_HOST:])
                                 for i, v in enumerate(vacs)])


@pytest.mark.parametrize("kind", KINDS)
def test_forward_shapes(graph, kind):
    torch.manual_seed(0)
    net = build_model(kind, CGCNNParams()).eval()
    b = _batch(graph, _ox(graph)[:3])
    assert net(b).shape == (3,)
    assert net.readout_vector(b).shape == (3, net.readout_dim) == (3, 128)
    assert torch.isfinite(net(b)).all()


@pytest.mark.parametrize("kind", KINDS)
def test_permutation_invariant(rattled, kind):
    torch.manual_seed(1)
    net = build_model(kind, CGCNNParams()).eval()
    vac = _ox(rattled)[1]
    perm = torch.randperm(rattled["z"].shape[0], generator=torch.Generator().manual_seed(3))
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(len(perm))
    g2 = {**rattled, "z": rattled["z"][perm], "pos": rattled["pos"][perm], "edge_index": inv[rattled["edge_index"]]}
    a = net(_batch(rattled, [vac]))
    b = net(_batch(g2, [int(inv[vac])]))
    assert torch.allclose(a, b, atol=1e-5)


def test_descriptor_injection_only_in_d(graph):
    vacs = _ox(graph)[:2]
    d1, d2 = torch.zeros(2, 22), torch.ones(2, 22)
    for kind, changes in (("cgcnn-S", False), ("cgcnn-D", True)):
        torch.manual_seed(0)
        net = build_model(kind, CGCNNParams()).eval()
        diff = (net(_batch(graph, vacs, d1)) - net(_batch(graph, vacs, d2))).abs().max().item()
        assert (diff > 1e-6) == changes
        # explicit descriptors override the batch's (the staged-P path)
        if kind == "cgcnn-D":
            b = _batch(graph, vacs, d1)
            assert torch.allclose(net(b, desc_host=d2[:, :N_HOST], desc_site=d2[:, N_HOST:]),
                                  net(_batch(graph, vacs, d2)))


def test_flag_changes_output_at_init(rattled, capsys):
    torch.manual_seed(0)
    net = build_model("cgcnn-S", CGCNNParams()).eval()
    ox = _ox(rattled)
    out = net(_batch(rattled, ox[:8]))
    b = _batch(rattled, ox[:8])
    b.vac_flag.zero_()
    off = net(b)
    site_spread = (out.max() - out.min()).item()
    flag_effect = (out - off).abs().max().item()
    with capsys.disabled():
        print(f"\ncgcnn-S at init: max |flag on - flag off| = {flag_effect:.3e}; "
              f"spread over 8 O sites = {site_spread:.3e}")
    assert flag_effect > 1e-4 and site_spread > 1e-6


def test_parameter_counts(capsys):
    counts = {k: n_parameters(build_model(k, CGCNNParams())) for k in KINDS}
    with capsys.disabled():
        print(f"\nCGCNN parameter counts: {counts}")
    assert counts["cgcnn-D"] - counts["cgcnn-S"] == 22 * CGCNNParams().h_fea_len
    assert hparams_for("cgcnn-S", CGCNNParams().to_dict()) == CGCNNParams()
