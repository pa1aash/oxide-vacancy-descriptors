"""Inference on stored checkpoints (skipped when the checkpoints or the graph store are absent).

Condition (i) must reproduce the stored test predictions; geomeval plumbing is checked with condition graphs
equal to the DFT graphs (so (ii) and (iii) must equal (i)).
"""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from dftgnn import infer
from dftgnn.graphs.store import STORE_DIR
from dftgnn.train import REPO_ROOT, runindex

CK = REPO_ROOT / "checkpoints"
pytestmark = pytest.mark.skipif(not (CK.is_dir() and (STORE_DIR / "meta.json").is_file()),
                                reason="checkpoints or graph store not present")
torch.set_num_threads(3)


@pytest.fixture(scope="module")
def sites():
    from dftgnn.train import Store

    return infer.Sites(Store())


@pytest.fixture(scope="module")
def prim():
    return runindex.primary(runindex.load())


def _first_sites(rec, n=20):
    return pd.read_parquet(REPO_ROOT / rec["predictions"]["path"]).sort_values("site_id").head(n)


@pytest.mark.parametrize("model", ["S", "D-state", "P1"])
def test_condition_i_reproduces_s09(sites, prim, model):
    rec = prim[(model, "outer_r0", 654, 0)]
    stored = _first_sites(rec)
    net, ck = infer.load(rec)
    pos = np.array([sites.site_id.index(s) for s in stored.site_id])
    pred = infer.predict(net, ck, sites, sites.dft_graphs(sorted(set(stored.host_id))), pos)
    if model == "P1":
        cols = [f"pred_{n}" for n in ck["desc_names"]]
        assert np.abs(pred - stored[cols].to_numpy()).max() < 1e-3
    else:
        assert np.abs(pred - stored.y_pred.to_numpy()).max() < 1e-3


def test_geomeval_conditions_equal_when_graphs_equal(sites, prim, tmp_path):
    from dftgnn.mlip import geomeval as GE

    rec = prim[("S", "outer_r0", 654, 0)]
    hosts = sorted(set(_first_sites(rec, 6).host_id))
    tiling = pd.read_csv(GE.TILING_SUMMARY).set_index("host_id")
    hosts = [h for h in hosts if not tiling.loc[h, "excluded_from_C3b"]]
    g = sites.dft_graphs(hosts)
    for h in hosts:
        torch.save({"mlip": g[h], "mlip_rescaled": g[h]}, tmp_path / f"{h}_graphs.pt")
        (tmp_path / f"{h}.json").write_text(json.dumps(
            {"converged": True, "geometry": {"eps_v": 0.0, "deviatoric": 0.0, "internal_rmsd_A": 0.0}}))
    for model in ("S", "D", "P"):
        comps = GE.components(prim, model, 0, 654, 0)
        frame, info = GE.run(model, 0, 654, 0, comps, sites, hosts=hosts, mlip_dir=tmp_path)
        wide = frame.pivot(index="site_id", columns="condition", values="prediction")
        assert np.allclose(wide["dft"], wide["mlip"], atol=1e-6) and np.allclose(wide["dft"], wide["mlip_rescaled"], atol=1e-6)
        assert info["registered"] and info["n_hosts_evaluated"] == len(hosts)
