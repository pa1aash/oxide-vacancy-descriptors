"""Graph store tests. Skipped when the gitignored store or universe is absent."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from dftgnn.graphs import collate_sites, edges_matgl, host_graph, load_structure
from dftgnn.graphs import store as S

pytestmark = pytest.mark.skipif(not (S.STORE_DIR / "meta.json").is_file(),
                                reason="graph store not built")
CUTOFF = 5.0


@pytest.fixture(scope="module")
def built():
    return S.load_store()


@pytest.fixture(scope="module")
def universe():
    from dftgnn.data.universe import read_universe

    return read_universe()


def _cif(universe, host_id):
    path = S.REPO_ROOT / universe.loc[universe.host_id == host_id, "supercell_cif_path"].iloc[0]
    if not path.is_file():
        pytest.skip("Kumagai release CIFs not unpacked (Mac only, not pushed to the pod)")
    return path


def test_counts_and_vacancy_on_oxygen(built):
    graphs, sites, meta = built
    assert len(graphs) == 818 and meta["n_hosts"] == 818
    assert len(sites["site_id"]) == 1726
    for h, v in zip(sites["host_idx"].tolist(), sites["vacancy_atom_index"].tolist(), strict=True):
        assert int(graphs[h]["z"][v]) == 8


def test_zno_flag_and_first_shell(built):
    graphs, sites, _ = built
    p = sites["site_id"].index("ZnO_Va_O1_0")
    g = graphs[int(sites["host_idx"][p])]
    b = collate_sites(graphs, sites, [p])
    assert torch.nonzero(b.vac_flag[:, 0]).flatten().tolist() == [150]
    assert int(b.vacancy_index[0]) == 150
    src, dst = g["edge_index"]
    m = (src == 150) & (g["edge_dist"] <= 1.95)
    assert int(m.sum()) == 4
    assert set(g["z"][dst[m]].tolist()) == {30}
    d = g["edge_dist"][m]
    assert bool(((d >= 1.92) & (d <= 1.95)).all())


def test_cutoff_and_symmetry(built):
    graphs, _, _ = built
    for g in graphs:
        assert float(g["edge_dist"].max()) <= CUTOFF
        ei, off, d = g["edge_index"].numpy(), g["pbc_offset"].numpy(), g["edge_dist"].numpy()
        fwd = {(a, b, *o): x for a, b, o, x in zip(ei[0], ei[1], map(tuple, off), d, strict=True)}
        for (a, b, o0, o1, o2), x in fwd.items():
            assert abs(fwd[(b, a, -o0, -o1, -o2)] - x) < 1e-9


def _pymatgen_counts(structure) -> np.ndarray:
    centre, _, _, dist = structure.get_neighbor_list(CUTOFF)
    keep = dist > 1e-8
    return np.bincount(centre[keep], minlength=len(structure))


def test_neighbour_counts_match_pymatgen(built, universe):
    graphs, _, meta = built
    rng = np.random.default_rng(6)
    idx = [*rng.choice(len(graphs), size=5, replace=False).tolist(),
           int(np.argmax([g["z"].shape[0] for g in graphs]))]
    for i in idx:
        st = load_structure(_cif(universe, meta["host_ids"][i]))
        mine = np.bincount(graphs[i]["edge_index"][0].numpy(), minlength=len(st))
        assert np.array_equal(mine, _pymatgen_counts(st)), meta["host_ids"][i]


def test_translation_invariance(built, universe):
    from pymatgen.core import Structure

    _, _, meta = built
    rng = np.random.default_rng(7)
    for i in rng.choice(len(meta["host_ids"]), size=3, replace=False):
        st = load_structure(_cif(universe, meta["host_ids"][i]))
        shift = rng.random(3)
        moved = Structure(st.lattice, st.species, (st.frac_coords + shift) % 1.0)
        a = np.sort(host_graph(st, CUTOFF)["edge_dist"].numpy())
        b = np.sort(host_graph(moved, CUTOFF)["edge_dist"].numpy())
        assert a.shape == b.shape
        assert np.abs(a - b).max() < 1e-6


def test_edges_come_from_matgl(built, universe):
    graphs, _, meta = built
    st = load_structure(_cif(universe, meta["host_ids"][0]))
    ei, _ = edges_matgl(st, CUTOFF)
    assert ei.shape[1] == graphs[0]["edge_index"].shape[1]


def test_round_trip_and_manifest(built, tmp_path):
    graphs, sites, meta = built
    assert S.verify_store() == []
    header, _ = S.read_manifest()
    assert float(header["cutoff_A"]) == CUTOFF
    assert header["universe_ids_sha256"] == S.UNIVERSE_IDS_SHA.read_text().split()[0]
    man = tmp_path / "m.sha256"
    S.write_store(graphs[:150], {**sites}, meta["host_ids"][:150], CUTOFF, code_sha="test",
                  out_dir=tmp_path / "store", manifest=man)
    g2, s2, _ = S.load_store(tmp_path / "store", manifest=man)
    for a, b in zip(graphs[:150], g2, strict=True):
        assert a.keys() == b.keys()
        for k in a:
            assert a[k].dtype == b[k].dtype and torch.equal(a[k], b[k])
    for k in ("host_idx", "vacancy_atom_index", "target", "desc", "p1_target"):
        assert torch.equal(sites[k], s2[k])
    (tmp_path / "store" / "hosts_000.pt").write_bytes(b"corrupt")
    assert S.verify_store(tmp_path / "store", man) == ["hosts_000.pt"]


def test_collate_repeats_host(built):
    graphs, sites, _ = built
    h = sites["host_idx"]
    p = int(torch.nonzero(torch.bincount(h) >= 2)[0])
    pos = torch.nonzero(h == p).flatten()[:2].tolist()
    b = collate_sites(graphs, sites, [pos[0], pos[1], pos[0]])
    n = graphs[p]["z"].shape[0]
    assert b.num_graphs == 3 and b.num_nodes == 3 * n
    assert b.vac_flag.sum() == 3
    assert b.vacancy_index.tolist() == [int(sites["vacancy_atom_index"][pos[0]]),
                                        n + int(sites["vacancy_atom_index"][pos[1]]),
                                        2 * n + int(sites["vacancy_atom_index"][pos[0]])]


def test_sites_of_one_host_share_the_graph_and_differ_only_in_the_flag(built):
    """Every host with several sites: distinct O vacancy atoms, and collated examples that are identical
    except for the flag (checked on the first 40 such hosts)."""
    graphs, sites, _ = built
    h = sites["host_idx"]
    multi = torch.nonzero(torch.bincount(h) >= 2).flatten()
    for p in multi.tolist():
        v = sites["vacancy_atom_index"][h == p]
        assert len(set(v.tolist())) == len(v)
    for p in multi[:40].tolist():
        pos = torch.nonzero(h == p).flatten().tolist()
        b = collate_sites(graphs, sites, pos)
        n = graphs[p]["z"].shape[0]
        flags = b.vac_flag.view(len(pos), n)
        assert (flags.sum(1) == 1).all()
        assert flags.argmax(1).tolist() == sites["vacancy_atom_index"][pos].tolist()
        for k in ("z", "pos", "edge_dist"):
            x = b[k].view(len(pos), n, -1) if k != "edge_dist" else b[k].view(len(pos), -1)
            assert (x == x[0]).all(), k
