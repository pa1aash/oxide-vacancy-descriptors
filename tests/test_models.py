"""Model tests on synthetic rock-salt MgO graphs; no data files needed."""
from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch

from dftgnn.graphs import host_graph, site_data
from dftgnn.models import KINDS, HParams, StagedP, Standardiser, build_model, n_parameters

N_HOST, N_SITE = 12, 10


def _mgo(n=2):
    from pymatgen.core import Lattice, Structure

    s = Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.21), ["Mg", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    s.make_supercell([n, n, n])
    return s


@pytest.fixture(scope="module")
def graph():
    return host_graph(_mgo(), 5.0)


def _oxygens(g):
    return torch.nonzero(g["z"] == 8).flatten().tolist()


def _batch(g, vacs, desc=None):
    torch.manual_seed(0)
    desc = torch.randn(len(vacs), N_HOST + N_SITE) if desc is None else desc
    return Batch.from_data_list([site_data(g, v, y=1.0, desc_host=desc[i, :N_HOST],
                                           desc_site=desc[i, N_HOST:]) for i, v in enumerate(vacs)])


@pytest.mark.parametrize("pooling", ["set2set", "mean"])
@pytest.mark.parametrize("kind", KINDS)
def test_forward_shapes(graph, kind, pooling):
    torch.manual_seed(0)
    net = build_model(kind, HParams(pooling=pooling)).eval()
    b = _batch(graph, _oxygens(graph)[:3])
    out = net(b)
    assert out.shape == ((3, N_HOST + N_SITE) if kind == "P1" else (3,))
    assert net.readout_vector(b).shape == (3, net.readout_dim)
    assert torch.isfinite(out).all()


def _permuted(g, perm):
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(len(perm))
    return {**g, "z": g["z"][perm], "pos": g["pos"][perm], "edge_index": inv[g["edge_index"]]}, inv


@pytest.mark.parametrize("pooling", ["set2set", "mean"])
def test_s_permutation_invariant(graph, pooling):
    torch.manual_seed(1)
    net = build_model("S", HParams(pooling=pooling)).eval()
    vacs = _oxygens(graph)[:2]
    perm = torch.randperm(graph["z"].shape[0], generator=torch.Generator().manual_seed(3))
    g2, inv = _permuted(graph, perm)
    with torch.no_grad():
        a = net(_batch(graph, vacs))
        b = net(_batch(g2, [int(inv[v]) for v in vacs]))
    assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)


@pytest.fixture(scope="module")
def distorted():
    """Rock-salt MgO with random displacements: every O site is symmetry-inequivalent."""
    return host_graph(_mgo().perturb(0.15, min_distance=0.05, seed=0), 3.0)


def test_site_graphs_of_one_host_differ_only_in_the_flag(distorted):
    o = _oxygens(distorted)
    a, b = site_data(distorted, o[0]), site_data(distorted, o[1])
    for k in ("z", "pos", "edge_index", "edge_dist", "pbc_offset", "state"):
        assert torch.equal(a[k], b[k]), k
    assert torch.nonzero(a.vac_flag.flatten()).flatten().tolist() == [o[0]]
    assert torch.nonzero(b.vac_flag.flatten()).flatten().tolist() == [o[1]]
    assert a.vacancy_index.tolist() == [o[0]] and b.vacancy_index.tolist() == [o[1]]


def test_readout_gathers_the_flagged_node_after_batching(graph, distorted):
    """Two hosts of different size, one of them twice: the vacancy part of the readout vector is the
    embedding of the flagged node of each graph, not node 0 or a node of another graph."""
    torch.manual_seed(8)
    net = build_model("S").eval()
    net.blocks = torch.nn.ModuleList()           # readout then sees the node-encoder output directly
    o1, o2 = _oxygens(graph), _oxygens(distorted)
    items = [site_data(graph, o1[1]), site_data(distorted, o2[3]), site_data(graph, o1[2])]
    b = Batch.from_data_list(items)
    flagged = torch.nonzero(b.vac_flag.flatten()).flatten()
    assert torch.equal(flagged, b.vacancy_index)
    assert (b.batch[b.vacancy_index] == torch.arange(3)).all()
    with torch.no_grad():
        nf, _, _ = net.embedding(b.z, net.bond_expansion(b.edge_dist), b.state)
        nf = net.node_encoder(torch.cat([nf, b.vac_flag], -1))
        vec = net.readout_vector(b)
    h2 = net.hp.hidden_width // 2
    assert torch.allclose(vec[:, :h2], nf[flagged])


@pytest.mark.xfail(strict=True, reason=(
    "Known property of the pre-registered backbone (docs/diagnostics_sweep.md, item 1): with matgl's "
    "default initialisation the distance information reaching a node is attenuated to float noise, so an "
    "untrained S gives inequivalent O sites of one host the same output (relative spread ~1e-7). Remove "
    "this marker if the backbone or its initialisation is changed."))
def test_untrained_s_distinguishes_inequivalent_o_sites(distorted):
    o = _oxygens(distorted)
    rel = []
    for seed in range(3):
        torch.manual_seed(seed)
        net = build_model("S").eval()
        with torch.no_grad():
            out = net(_batch(distorted, o))
        rel.append(float(out.std() / out.abs().mean()))
    assert min(rel) > 1e-4


def test_untrained_s_site_blindness_is_geometric(distorted):
    """The flag itself reaches the readout: the flagged node differs from the unflagged O nodes, but the
    unflagged O nodes, whose environments differ, are indistinguishable (the measured attenuation)."""
    torch.manual_seed(9)
    net = build_model("S").eval()
    o = _oxygens(distorted)
    b = _batch(distorted, [o[0]])
    with torch.no_grad():
        ea = net.bond_expansion(b.edge_dist)
        nf, ef, sf = net.embedding(b.z, ea, b.state)
        nf = net.node_encoder(torch.cat([nf, b.vac_flag], -1))
        ef, sf = net.edge_encoder(ef), net.state_encoder(sf)
        for blk in net.blocks:
            ef, nf, sf = blk(b.edge_index, ef, nf, sf, b.batch, b.batch[b.edge_index[0]], b.num_nodes, 1)
    others = nf[o[1:]]
    assert (nf[o[0]] - others[0]).abs().max() > 1e-3          # flag survives
    assert (others - others[0]).abs().max() < 1e-5            # geometry does not


def test_vacancy_flag_matters(graph):
    torch.manual_seed(2)
    net = build_model("S").eval()
    o = _oxygens(graph)
    with torch.no_grad():
        a = net(_batch(graph, [o[0]]))
        mg = int(torch.nonzero(graph["z"] == 12)[0])
        b = net(_batch(graph, [mg]))
    assert not torch.allclose(a, b)


@pytest.mark.parametrize("kind", ["D-state", "D-late"])
def test_descriptor_injection_wired(graph, kind):
    torch.manual_seed(4)
    net = build_model(kind).eval()
    vacs = _oxygens(graph)[:2]
    desc = torch.randn(2, N_HOST + N_SITE)
    with torch.no_grad():
        real = net(_batch(graph, vacs, desc))
        zero = net(_batch(graph, vacs, torch.zeros_like(desc)))
        host_only = net(_batch(graph, vacs, torch.cat([desc[:, :N_HOST], torch.zeros(2, N_SITE)], 1)))
    assert not torch.allclose(real, zero)
    assert not torch.allclose(real, host_only)       # site descriptors reach the output too


def test_s_ignores_descriptors(graph):
    torch.manual_seed(5)
    net = build_model("S").eval()
    vacs = _oxygens(graph)[:2]
    with torch.no_grad():
        a = net(_batch(graph, vacs, torch.randn(2, 22)))
        b = net(_batch(graph, vacs, torch.zeros(2, 22)))
    assert torch.equal(a, b)


class _Oracle(torch.nn.Module):
    """A stand-in P1 that returns the batch's own (true) standardised descriptors."""

    kind = "P1"

    def forward(self, batch):
        return torch.cat([batch.desc_host, batch.desc_site], dim=-1)


@pytest.mark.parametrize("kind", ["D-state", "D-late"])
def test_p_true_descriptor_swap_in_equals_d(graph, kind):
    torch.manual_seed(6)
    d = build_model(kind).eval()
    raw = torch.randn(40, 22, dtype=torch.float64) * 3 + 1
    st = Standardiser.fit(raw)
    p = StagedP(build_model("P1"), d, st, st).eval()
    p.p1 = _Oracle()
    vacs = _oxygens(graph)[:4]
    b = _batch(graph, vacs, st(raw[:4]).float())
    with torch.no_grad():
        assert torch.equal(p(b), d(b))


def test_p_maps_between_scalings(graph):
    torch.manual_seed(7)
    d = build_model("D-late").eval()
    raw = torch.randn(40, 22, dtype=torch.float64) * 2 + 5
    st_p1, st_d = Standardiser.fit(raw[:30]), Standardiser.fit(raw[10:])
    p = StagedP(build_model("P1"), d, st_p1, st_d).eval()
    p.p1 = _Oracle()
    vacs = _oxygens(graph)[:4]
    with torch.no_grad():
        out_p = p(_batch(graph, vacs, st_p1(raw[:4]).float()))
        out_d = d(_batch(graph, vacs, st_d(raw[:4]).float()))
    assert torch.allclose(out_p, out_d, atol=1e-5)


def test_standardiser_round_trip():
    x = torch.randn(50, 5, dtype=torch.float64)
    x[:, 2] = 3.0                                 # constant column: sd replaced by 1
    st = Standardiser.fit(x)
    assert torch.allclose(st.inverse(st(x)), x)
    assert torch.allclose(st(x)[:, 2], torch.zeros(50, dtype=torch.float64))


def test_parameter_counts_default():
    counts = {k: n_parameters(build_model(k)) for k in KINDS}
    print("parameter counts (default HParams):", counts)
    assert counts["D-state"] > counts["S"] and counts["P1"] > counts["S"]
    assert all(1e5 < c < 1e6 for c in counts.values())


def test_v2_defaults_reproduce_the_preregistered_backbone():
    torch.manual_seed(0)
    a = build_model("S")
    torch.manual_seed(0)
    b = build_model("S", HParams(init="default", local_readout=False))
    assert a.state_dict().keys() == b.state_dict().keys()
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())


@pytest.mark.parametrize("kind", KINDS)
def test_v2_options_forward(graph, kind):
    torch.manual_seed(0)
    net = build_model(kind, HParams(init="kaiming", local_readout=True)).eval()
    b = _batch(graph, _oxygens(graph)[:3])
    out = net(b)
    assert out.shape == ((3, N_HOST + N_SITE) if kind == "P1" else (3,)) and torch.isfinite(out).all()
    assert net.readout_vector(b).shape == (3, net.readout_dim)


def test_v2_backbone_sees_inequivalent_o_sites(distorted):
    """The v2 options fix the measured site-blindness at initialisation by orders of magnitude."""
    o = _oxygens(distorted)
    rel = []
    for seed in range(3):
        torch.manual_seed(seed)
        net = build_model("S", HParams(init="kaiming", local_readout=True)).eval()
        with torch.no_grad():
            out = net(_batch(distorted, o))
        rel.append(float(out.std() / out.abs().mean()))
    assert min(rel) > 1e-4


def test_local_readout_uses_only_the_vacancy_edges(graph, distorted):
    """Batching the vacancy example with other graphs leaves its local-environment vector unchanged."""
    torch.manual_seed(3)
    net = build_model("S", HParams(local_readout=True)).eval()
    o1, o2 = _oxygens(graph), _oxygens(distorted)
    with torch.no_grad():
        alone = net.readout_vector(Batch.from_data_list([site_data(distorted, o2[1])]))
        mixed = net.readout_vector(Batch.from_data_list([site_data(graph, o1[0]), site_data(distorted, o2[1])]))
    assert torch.allclose(alone[0], mixed[1], atol=1e-5)
