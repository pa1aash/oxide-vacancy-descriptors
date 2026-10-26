"""Tuning tests: trial-count and resume logic, search space, test-host exclusion, shared validation hosts,
non-finite objectives and VRAM admission. CPU only, tiny epochs or a stub trainer."""
from __future__ import annotations

import math
import os

import optuna
import pytest
import torch

from dftgnn.config import load_config
from dftgnn.graphs import store as G
from dftgnn.split import load_split
from dftgnn.tune import (
    MODELS,
    TuningStore,
    VramLedger,
    anchor_hosts,
    boundary_hits,
    n_complete,
    open_study,
    run_study,
    suggest,
    summarise,
    trial_spec,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
HAVE_STORE = (G.STORE_DIR / "meta.json").is_file()
needs_store = pytest.mark.skipif(not HAVE_STORE, reason="graph store not built")


class _Stub:
    """Stands in for the graph store: run_study only reads ``manifest_sha`` from it."""
    manifest_sha = "stub"


def _fake_train(values=None, kill_at=None):
    """A trainer returning ``values[i]`` as the best validation metric of call i."""
    calls = {"n": 0}

    def train(spec, data, cfg, *, device=None, ckpt_path=None, log=print):
        i = calls["n"]
        calls["n"] += 1
        if kill_at is not None and i == kill_at:
            raise KeyboardInterrupt          # a killed process: the trial stays RUNNING in storage
        assert spec.eval_test is False and ckpt_path is None
        v = 0.5 + 0.01 * i if values is None else values[i % len(values)]
        return {"best_val_metric": v, "best_epoch": 3, "epochs_run": 10, "peak_memory_bytes": 0,
                "n_hosts": {"train": 45, "val": 5, "test": 0}, "n_sites": {"train": 90, "val": 9, "test": 0},
                "device": "cpu"}
    return train


def _run(tmp_path, train, n=5, model="S", anchor=50):
    return run_study(model, anchor, _Stub(), cfg=load_config(), storage_dir=tmp_path / "db",
                     csv_dir=tmp_path / "csv", n_trials=n, train_fn=train, pod_commit="c", log=lambda *_: None)


def test_config_trial_count_is_30():
    assert load_config().tuning.optuna_trials_per_model_per_anchor == 30


def test_study_stops_at_n_complete_trials(tmp_path):
    st = _run(tmp_path, _fake_train(), n=6)
    assert n_complete(st) == 6 and len(st.trials) == 6
    rows = (tmp_path / "csv" / "trials_S_a50.csv").read_text().splitlines()
    assert len(rows) == 7                       # header + 6 trials
    st2 = _run(tmp_path, _fake_train(), n=6)    # rerun of a finished study adds nothing
    assert len(st2.trials) == 6


def test_resume_after_kill_counts_only_complete_trials(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, _fake_train(kill_at=2), n=5)
    st = open_study("S_a50", tmp_path / "db", load_config())
    states = [t.state for t in st.trials]
    assert states.count(optuna.trial.TrialState.COMPLETE) == 2
    assert states.count(optuna.trial.TrialState.RUNNING) == 1
    st = _run(tmp_path, _fake_train(), n=5)
    S = optuna.trial.TrialState
    assert n_complete(st) == 5
    assert [t.state for t in st.trials].count(S.FAIL) == 1
    assert not [t for t in st.trials if t.state == S.RUNNING]


def test_search_space_equals_config(tmp_path):
    cfg = load_config()
    st = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    tr = st.ask()
    suggest(tr, cfg.tuning.search_space)
    d = tr.distributions
    sp = cfg.tuning.search_space
    assert set(d) == set(sp) == {"learning_rate", "weight_decay", "hidden_width", "megnet_blocks", "dropout",
                                 "batch_size", "readout_mlp_width", "pooling"}
    for name, p in sp.items():
        if p.dist == "categorical":
            assert isinstance(d[name], optuna.distributions.CategoricalDistribution)
            assert list(d[name].choices) == list(p.choices)
        else:
            assert isinstance(d[name], optuna.distributions.FloatDistribution)
            assert (d[name].low, d[name].high, d[name].log) == (p.low, p.high, p.dist == "log_uniform")
    assert (sp["learning_rate"].low, sp["learning_rate"].high) == (1e-4, 3e-3)
    assert (sp["weight_decay"].low, sp["weight_decay"].high) == (1e-6, 1e-3)
    assert (sp["dropout"].low, sp["dropout"].high) == (0.0, 0.3)
    assert sp["pooling"].choices == ["set2set", "mean"]


def test_identical_validation_hosts_across_models():
    cfg = load_config()
    params = {"learning_rate": 1e-3, "weight_decay": 1e-5, "hidden_width": 64, "megnet_blocks": 3,
              "dropout": 0.1, "batch_size": 32, "readout_mlp_width": 64, "pooling": "mean"}
    from dftgnn.train import resolve_hosts

    test = set(load_split("outer_r0")["test"])
    for a in cfg.tuning.anchors:
        hs = [resolve_hosts(trial_spec(m, a, params, cfg), cfg) for m in MODELS]
        assert all(h["val"] == hs[0]["val"] and h["train"] == hs[0]["train"] for h in hs)
        assert all(h["test"] == [] for h in hs)
        assert hs[0] == {**anchor_hosts(a, cfg), "test": []}
        assert not (set(hs[0]["train"]) | set(hs[0]["val"])) & test
        assert len(hs[0]["train"]) + len(hs[0]["val"]) == a


def test_nonfinite_objective_is_complete_with_1000(tmp_path):
    st = _run(tmp_path, _fake_train(values=[float("nan"), 0.4, float("inf")]), n=3)
    vals = [t.value for t in st.trials]
    assert vals == [1000.0, 0.4, 1000.0]
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in st.trials)
    s = summarise(st, "S", 50, load_config(), top=5)
    assert s["n_nonfinite"] == 2 and s["best_value"] == 0.4 and s["n_complete"] == 3


def test_boundary_hits():
    sp = load_config().tuning.search_space
    p = {"learning_rate": 1.05e-4, "weight_decay": 1e-4, "hidden_width": 128, "megnet_blocks": 4,
         "dropout": 0.29, "batch_size": 32, "readout_mlp_width": 64, "pooling": "mean"}
    assert boundary_hits(p, sp) == {"learning_rate": "low", "megnet_blocks": "high", "dropout": "high"}


def test_vram_ledger_admission(tmp_path):
    a = VramLedger(tmp_path, vram_gb=10.0, poll=0.01)       # budget 8.5 GiB
    assert a.try_acquire(5.0)
    other = tmp_path / f"{os.getppid()}.json"               # a live process holding 5 GiB
    other.write_text('{"gb": 5.0, "solo": false}')
    a.release()
    assert not a.try_acquire(4.0)                           # 5 + 4 > 8.5
    assert a.try_acquire(3.0)
    a.release()
    assert not a.try_acquire(1.0, solo=True)                # solo waits for an empty ledger
    assert (tmp_path / f"{os.getpid()}.solo_wait").exists()
    other.unlink()
    assert a.try_acquire(1.0, solo=True)
    a.release()
    (tmp_path / "999999999.json").write_text('{"gb": 9.0, "solo": true}')   # a dead process is dropped
    assert a.try_acquire(8.0)
    a.release()


@pytest.fixture(scope="module")
def tstore():
    return TuningStore(resample=0)


@needs_store
def test_test_hosts_removed_from_memory(tstore):
    test = set(load_split("outer_r0")["test"])
    assert tstore.excluded_hosts == test
    with pytest.raises(AssertionError):
        tstore.positions([min(test)])
    hid = list(tstore.meta["host_ids"])
    assert all(tstore.graphs[hid.index(h)] is None for h in test)
    pos = [i for i, h in enumerate(tstore.site_host) if h in test]
    assert torch.isnan(tstore.sites["target"][pos]).all() and torch.isnan(tstore.sites["desc"][pos]).all()
    assert len(pos) == tstore.n_excluded_sites > 0


@needs_store
@pytest.mark.parametrize("model", ["S", "P1"])
def test_real_trial_on_cpu(tstore, tmp_path, model):
    """Two one-epoch trials through the real trainer at anchor 50; the test hosts are never touched."""
    st = run_study(model, 50, tstore, cfg=load_config(), storage_dir=tmp_path / "db", csv_dir=tmp_path / "csv",
                   n_trials=2, epoch_override=(1, 1), pod_commit="c", log=lambda *_: None)
    assert n_complete(st) == 2
    for t in st.trials:
        assert math.isfinite(t.value)
        assert t.user_attrs["n_val_hosts"] == 5 and t.user_attrs["n_train_hosts"] == 45


@needs_store
def test_never_finite_validation_does_not_crash(tstore, monkeypatch):
    import dftgnn.train as T

    real = T._predict
    monkeypatch.setattr(T, "_predict", lambda *a, **k: torch.full_like(real(*a, **k), float("nan")))
    cfg = load_config()
    params = {"learning_rate": 1e-3, "weight_decay": 1e-5, "hidden_width": 64, "megnet_blocks": 2,
              "dropout": 0.0, "batch_size": 32, "readout_mlp_width": 64, "pooling": "mean"}
    spec = trial_spec("S", 50, params, cfg)
    spec.max_epochs, spec.patience = 2, 2
    res = T.train_run(spec, tstore, cfg, device=torch.device("cpu"), log=lambda *_: None)
    assert not math.isfinite(res["best_val_metric"]) and res["n_sites"]["test"] == 0


def test_v2_trials_carry_the_fixed_backbone_options():
    from dftgnn.config import load_config
    from dftgnn.tune import trial_spec

    cfg = load_config()
    params = {"learning_rate": 1e-3, "weight_decay": 1e-5, "hidden_width": 64, "megnet_blocks": 3, "dropout": 0.1,
              "batch_size": 32, "readout_mlp_width": 64, "pooling": "mean"}
    arch = cfg.v2.variants[cfg.v2.selected]
    sp = trial_spec("S", 654, params, cfg, "v2_S_a654", arch)
    assert {k: sp.hp[k] for k in arch} == arch and sp.eval_test is False
    assert trial_spec("S", 654, params, cfg).hp == {k: params[k] for k in
                                                    ("hidden_width", "megnet_blocks", "dropout", "readout_mlp_width", "pooling")}
