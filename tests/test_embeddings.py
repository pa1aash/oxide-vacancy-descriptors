"""Embedding extraction on stored S checkpoints (skipped without checkpoints / graph store)."""
import numpy as np
import pandas as pd
import pytest
import torch

from dftgnn import embeddings as E
from dftgnn import infer
from dftgnn.graphs.store import STORE_DIR
from dftgnn.train import REPO_ROOT, runindex

pytestmark = pytest.mark.skipif(not ((REPO_ROOT / "checkpoints").is_dir() and (STORE_DIR / "meta.json").is_file()),
                                reason="checkpoints or graph store not present")
torch.set_num_threads(3)


@pytest.fixture(scope="module")
def sites():
    from dftgnn.train import Store

    return infer.Sites(Store())


@pytest.fixture(scope="module")
def prim():
    return runindex.primary(runindex.load())


@pytest.mark.parametrize("budget", [654, 200, 25])
def test_readout_mlp_reproduces_stored_predictions(sites, prim, budget):
    rec = prim[("S", "outer_r0", budget, 0)]
    net, ck = infer.load(rec)
    stored = pd.read_parquet(REPO_ROOT / rec["predictions"]["path"]).sort_values("site_id").head(25)
    pos = np.array([sites.site_id.index(s) for s in stored.site_id])
    vec = infer.predict(net, ck, sites, sites.dft_graphs(sorted(set(stored.host_id))), pos, what="readout")
    with torch.no_grad():
        z = net.head(torch.from_numpy(vec.astype(np.float32))).squeeze(-1).double()
    pred = (z * ck["target_sd"].cpu() + ck["target_mean"].cpu()).numpy()
    assert np.abs(pred - stored.y_pred.to_numpy()).max() < 1e-3
    assert vec.shape[1] == E.block_bounds(net)[-1]


def test_init_extraction_deterministic_and_differs_from_trained(sites, prim, monkeypatch):
    rec = prim[("S", "outer_r0", 25, 0)]
    a = E.extract("init0", rec, sites, 0, 25)
    b = E.extract("init0", rec, sites, 0, 25)
    c = E.extract("init1", rec, sites, 0, 25)
    t = E.extract("trained", rec, sites, 0, 25)
    assert np.array_equal(a["embeddings"], b["embeddings"]) and list(a["site_ids"]) == list(t["site_ids"])
    assert not np.allclose(a["embeddings"], c["embeddings"]) and not np.allclose(a["embeddings"], t["embeddings"])
    assert set(a["split"]) == {"train", "val", "test"} and a["embeddings"].dtype == np.float32
    groups = E.site_split(0, 25)
    assert len(groups["train"]) + len(groups["val"]) == 25


# ---- device of the loaded model (nrp-code-v2 fix; deviations row of 2026-10-14)
KINDS_654 = ("S", "D-state", "P1")


def _stored_batch(sites, rec, ck, n=25):
    from dftgnn.models import Standardiser

    stored = pd.read_parquet(REPO_ROOT / rec["predictions"]["path"]).sort_values("site_id").head(n)
    pos = np.array([sites.site_id.index(s) for s in stored.site_id])
    desc_std = Standardiser(ck["desc_mean"].cpu(), ck["desc_sd"].cpu())(sites.desc).float()
    return stored, pos, infer._batch(sites, sites.dft_graphs(sorted(set(stored.host_id))), pos, desc_std)


@pytest.mark.parametrize("kind", KINDS_654)
def test_readout_runs_with_model_and_batch_on_another_device(sites, prim, kind):
    """The model follows the requested device (meta stands in for CUDA): nrp-code-v1 left the loaded weights, and the
    bond-expansion centres the first readout operation uses, on the CPU, so a CUDA batch failed."""
    meta = torch.device("meta")
    rec = prim[(kind, "outer_r0", 654, 0)]
    for net, ck in (infer.load(rec, meta), E.model_for("init0", rec, meta)):
        assert {t.device.type for t in (*net.parameters(), *net.buffers())} == {"meta"}
        assert net.bond_expansion.rbf.centers.device.type == "meta"
        _, pos, batch = _stored_batch(sites, rec, infer.load(rec)[1], n=6)
        vec = net.readout_vector(batch.to(meta))
        assert vec.device.type == "meta" and tuple(vec.shape) == (len(pos), net.readout_dim)


@pytest.mark.parametrize("kind", KINDS_654)
def test_cpu_readout_unchanged_and_reproduces_stored_predictions(sites, prim, kind):
    """On the CPU the fixed load gives byte-identical readout vectors to the nrp-code-v1 construction (build, then
    load the weights, never moved), and the head maps them to the stored predictions to 1e-3."""
    from dftgnn.models import Standardiser, build_model, hparams_for

    rec = prim[(kind, "outer_r0", 654, 0)]
    net, ck = infer.load(rec)
    mc = ck["model_config"]
    old = build_model(mc["kind"], hparams_for(mc["kind"], mc["hp"]), n_host=mc["n_host"], n_site=mc["n_site"],
                      cutoff=mc["cutoff"])
    old.load_state_dict(ck["state_dict"])
    old.eval()
    stored, pos, _ = _stored_batch(sites, rec, ck)
    graphs = sites.dft_graphs(sorted(set(stored.host_id)))
    vec = infer.predict(net, ck, sites, graphs, pos, what="readout")
    ref = infer.predict(old, ck, sites, graphs, pos, what="readout")
    assert vec.dtype == ref.dtype and vec.shape == ref.shape and vec.tobytes() == ref.tobytes()
    with torch.no_grad():
        z = net.head(torch.from_numpy(vec.astype(np.float32))).double()
    if kind == "P1":
        pred = Standardiser(ck["desc_mean"].cpu(), ck["desc_sd"].cpu()).inverse(z).numpy()
        want = stored[[c for c in stored.columns if c.startswith("pred_desc_")]].to_numpy()
    else:
        pred = (z.squeeze(-1) * ck["target_sd"].cpu() + ck["target_mean"].cpu()).numpy()
        want = stored.y_pred.to_numpy()
    assert pred.shape == want.shape and np.abs(pred - want).max() < 1e-3
