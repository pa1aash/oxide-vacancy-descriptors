"""S10b: the vacancy flag reaches the output of S at random initialisation (both poolings).

Host: a 2x2x2 rock-salt MgO supercell with one Mg replaced by Ca and the O nearest to it displaced by 0.2 A,
so that the two compared O sites differ in neighbour species and in bond lengths. Two examples of the same
host differ only in which O atom carries the vacancy (flag and readout index move together, as in training); a
third check zeroes the flag and keeps the readout index; a fourth moves only the flag and keeps the readout
index.
"""
from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch

from dftgnn.graphs import host_graph, site_data
from dftgnn.models import HParams, build_model

TOL = 1e-6


@pytest.fixture(scope="module")
def host():
    from pymatgen.core import Lattice, Structure

    s = Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.21), ["Mg", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    s.make_supercell([2, 2, 2])
    s.replace(0, "Ca")
    o = [i for i, site in enumerate(s) if site.specie.symbol == "O"]
    d = [s.lattice.get_distance_and_image(s[i].frac_coords, s[0].frac_coords)[0] for i in o]
    near, far = o[min(range(len(o)), key=d.__getitem__)], o[max(range(len(o)), key=d.__getitem__)]
    assert abs(min(d) - max(d)) > 0.5
    s.translate_sites([near], [0.2, 0.0, 0.0], frac_coords=False)
    return host_graph(s, 5.0), near, far


def _batch(g, vacs, flags=None):
    items = []
    for k, v in enumerate(vacs):
        d = site_data(g, v, y=0.0)
        if flags is not None:
            d.vac_flag = flags[k]
        items.append(d)
    return Batch.from_data_list(items)


CONFIGS = [(p, b, h) for p in ("set2set", "mean") for b in (2, 4) for h in (64, 128)]


@pytest.mark.xfail(strict=True, reason=(
    "S10b finding: at default initialisation the outputs of two inequivalent O sites of one host differ by "
    "about 1e-8 to 1e-7 (below 1e-6) for every configuration of both poolings, although the flag itself "
    "reaches the output (the two tests below pass); see docs/s_site_resolution.md"))
@pytest.mark.parametrize(("pooling", "blocks", "hidden"), CONFIGS)
def test_moving_the_vacancy_changes_the_output_at_init(host, pooling, blocks, hidden):
    g, near, far = host
    torch.manual_seed(0)
    net = build_model("S", HParams(hidden_width=hidden, megnet_blocks=blocks, pooling=pooling)).eval()
    with torch.no_grad():
        out = net(_batch(g, [near, far]))
    assert abs(float(out[0] - out[1])) > TOL


@pytest.mark.parametrize(("pooling", "blocks", "hidden"), CONFIGS)
def test_zeroing_the_flag_changes_the_output_at_init(host, pooling, blocks, hidden):
    g, near, far = host
    torch.manual_seed(0)
    net = build_model("S", HParams(hidden_width=hidden, megnet_blocks=blocks, pooling=pooling)).eval()
    with torch.no_grad():
        on = net(_batch(g, [near, far]))
        b = _batch(g, [near, far])
        b.vac_flag.zero_()
        off = net(b)
    assert torch.all((on - off).abs() > TOL)


@pytest.mark.parametrize(("pooling", "blocks", "hidden"), CONFIGS)
def test_moving_only_the_flag_changes_the_output_at_init(host, pooling, blocks, hidden):
    """Readout index fixed on ``near``; the flag sits on ``near`` or on ``far``."""
    g, near, far = host
    n = g["z"].shape[0]
    f_near, f_far = torch.zeros(n, 1), torch.zeros(n, 1)
    f_near[near, 0], f_far[far, 0] = 1.0, 1.0
    torch.manual_seed(0)
    net = build_model("S", HParams(hidden_width=hidden, megnet_blocks=blocks, pooling=pooling)).eval()
    with torch.no_grad():
        out = net(_batch(g, [near, near], flags=[f_near, f_far]))
    assert abs(float(out[0] - out[1])) > TOL
