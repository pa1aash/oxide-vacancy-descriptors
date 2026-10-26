"""Training-loop tests: tiny CPU runs on the graph store, written to tmp dirs."""
from __future__ import annotations

import hashlib
import json

import pytest
import torch

from dftgnn.config import load_config
from dftgnn.graphs import store as G
from dftgnn.split import load_split
from dftgnn.train import RunSpec, Store, epoch_limits, pick_device, resolve_hosts, run_id, val_seed

HAVE_STORE = (G.STORE_DIR / "meta.json").is_file()
needs_store = pytest.mark.skipif(not HAVE_STORE, reason="graph store not built")
HP = {"hidden_width": 64, "megnet_blocks": 2, "dropout": 0.0, "readout_mlp_width": 64, "pooling": "mean"}


def _spec(model="S", **kw):
    sp = load_split("outer_r0")
    hosts = {"train": sp["budget_order"][:6], "val": sp["budget_order"][6:8], "test": sp["test"][:3]}
    base = {"model": model, "hp": HP, "lr": 1e-3, "weight_decay": 1e-5, "batch_size": 8,
            "split": "outer_r0", "r": 0, "budget": 25, "seed": 0, "hosts": hosts, "max_epochs": 2,
            "patience": 5, "smoke": True}
    return RunSpec(**{**base, **kw})


def test_val_seed_matches_clarified_rule():
    r, b, s = 3, 200, 1
    want = int.from_bytes(hashlib.sha256(f"val|{r}|{b}|{s}".encode()).digest()[:4], "big")
    assert val_seed(r, b, s) == want
    assert load_config().training.validation.seed_rule == "sha256_val_r_B_seed_first4_bigendian"


def test_outer_val_split_is_carved_from_training_hosts():
    cfg = load_config()
    spec = _spec(hosts=None, budget=50)
    h = resolve_hosts(spec, cfg)
    assert len(h["val"]) == 5 and len(h["train"]) == 45
    assert set(h["val"]) | set(h["train"]) == set(load_split("outer_r0")["budget_order"][:50])
    assert not set(h["test"]) & (set(h["train"]) | set(h["val"]))


def test_epoch_limits_come_from_the_config_unless_overridden():
    cfg = load_config()
    assert epoch_limits(_spec(max_epochs=None, patience=None), cfg) == (200, 30)
    assert epoch_limits(_spec(max_epochs=7, patience=2), cfg) == (7, 2)
    cfg.training.max_epochs = "TBD-S07"
    with pytest.raises(ValueError):
        epoch_limits(_spec(max_epochs=None, patience=None), cfg)


def test_run_id_sensitivity():
    a = run_id(_spec(), "c", "g")
    assert a == run_id(_spec(), "c", "g") and len(a) == 16
    assert a != run_id(_spec(seed=1), "c", "g")
    assert a != run_id(_spec(), "c2", "g")
    assert a != run_id(_spec(), "c", "g2")
    assert a != run_id(_spec(hp={**HP, "pooling": "set2set"}), "c", "g")


def test_device_never_mps():
    assert pick_device().type in ("cuda", "cpu")


@pytest.fixture(scope="module")
def data():
    return Store()


@needs_store
@pytest.mark.parametrize("model", ["S", "D-state", "D-late", "P1"])
def test_train_and_idempotent(data, model, tmp_path):
    from dftgnn.train.runner import execute

    spec = _spec(model)
    kw = {"results_dir": tmp_path / "res", "ckpt_dir": tmp_path / "ck", "allow_dirty": True,
          "log": lambda *_: None}
    out = execute(spec, data, **kw)
    assert out["status"] == "done"
    pay = json.loads(out["result"].read_text())["payload"]
    assert pay["epochs_run"] == 2 and pay["smoke"] is True
    assert pay["n_hosts"] == {"train": 6, "val": 2, "test": 3}
    if model == "P1":
        assert len(pay["metrics"]["per_descriptor"]) == 22
    else:
        assert set(pay["metrics"]) == {"mae", "rmse", "r2", "within_host_mae", "host_mean_mae"}
    ck = torch.load(tmp_path / "ck" / f"{out['run_id']}.pt", weights_only=False)
    assert {"target_mean", "target_sd", "desc_mean", "desc_sd"} <= set(ck)
    assert execute(spec, data, **kw)["status"] == "skipped"


@needs_store
def test_early_stopping_restores_best(data, tmp_path):
    from dftgnn.train import train_run

    res = train_run(_spec(max_epochs=6, patience=1), data, device=torch.device("cpu"),
                    log=lambda *_: None)
    vals = [h["val_metric"] for h in res["history"]]
    assert res["best_val_metric"] == min(vals)
    assert res["epochs_run"] <= 6
    if res["epochs_run"] < 6:
        assert res["epochs_run"] - 1 - res["best_epoch"] >= 1


@needs_store
def test_cpu_runs_are_reproducible(data):
    from dftgnn.train import train_run

    a = train_run(_spec(max_epochs=1), data, device=torch.device("cpu"), log=lambda *_: None)
    b = train_run(_spec(max_epochs=1), data, device=torch.device("cpu"), log=lambda *_: None)
    assert a["predictions"].y_pred.equals(b["predictions"].y_pred)


@needs_store
def test_staged_p_evaluates(data, tmp_path):
    from dftgnn.train.runner import execute

    kw = {"results_dir": tmp_path / "res", "ckpt_dir": tmp_path / "ck", "allow_dirty": True,
          "log": lambda *_: None}
    p1 = execute(_spec("P1"), data, **kw)["run_id"]
    d = execute(_spec("D-late"), data, **kw)["run_id"]
    out = execute(_spec("P", p1_run=p1, d_run=d), data, **kw)
    pay = json.loads(out["result"].read_text())["payload"]
    assert pay["components"] == {"P1": p1, "D": d} and pay["checkpoint"] is None


def test_ablations_zero_their_class_only():
    from torch_geometric.data import Batch

    from dftgnn.train import apply_ablation

    b = Batch(vac_flag=torch.ones(4, 1), desc_host=torch.ones(2, 12), desc_site=torch.ones(2, 10))
    apply_ablation(b, "desc_site")
    assert b.desc_site.abs().sum() == 0 and b.desc_host.sum() == 24 and b.vac_flag.sum() == 4
    apply_ablation(b, "desc_host")
    assert b.desc_host.abs().sum() == 0 and b.vac_flag.sum() == 4
    with pytest.raises(ValueError):
        apply_ablation(b, "other")


@needs_store
def test_descriptor_ablation_only_for_d(data):
    from dftgnn.train import train_run

    with pytest.raises(ValueError, match="D models only"):
        train_run(_spec("S", ablate="desc_site"), data, device=torch.device("cpu"), log=lambda *_: None)


@needs_store
def test_site_ablation_makes_d_ignore_site_descriptors(data):
    """Training runs with the site class ablated, and prediction under the ablation does not depend on the
    site descriptors (it does without it)."""
    from dftgnn.models import HParams, build_model
    from dftgnn.train import _predict, train_run

    res = train_run(_spec("D-state", ablate="desc_site", max_epochs=1), data, device=torch.device("cpu"),
                    log=lambda *_: None)
    assert res["epochs_run"] == 1
    torch.manual_seed(0)
    m = build_model("D-state", HParams(**HP), n_host=data.n_host, n_site=22 - data.n_host).eval()
    pos = data.positions(load_split("outer_r0")["test"][:3])
    d1 = data.sites["desc"].float()
    d2 = d1.clone()
    d2[:, data.n_host:] = torch.randn_like(d2[:, data.n_host:])
    a = _predict(m, data, pos, None, d1, 8, torch.device("cpu"), "desc_site")
    b = _predict(m, data, pos, None, d2, 8, torch.device("cpu"), "desc_site")
    c = _predict(m, data, pos, None, d2, 8, torch.device("cpu"), None)
    assert torch.equal(a, b) and not torch.equal(b, c)
