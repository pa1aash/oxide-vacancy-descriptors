"""Cluster stages (dftgnn.train.nrp): p1tune_v2 and dclass (specs in specs/), p1v2b and pcompv2b (code only until
configs/tuned_p1_v2.yaml exists), the tuning task, the pinned D checkpoint of pcompv2b and the dclass name
check. CPU only; the stages that read the data or the graph store skip without them."""
from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from dftgnn.config import load_config
from dftgnn.graphs import store as G
from dftgnn.split import load_split
from dftgnn.train import RunSpec, component_checkpoint, run_id
from dftgnn.train import nrp as NRP

ROOT = Path(__file__).resolve().parents[1]
needs_data = pytest.mark.skipif(not (ROOT / "data/processed/universe_v1.parquet").is_file(),
                                reason="universe parquet not built")
needs_store = pytest.mark.skipif(not (G.STORE_DIR / "meta.json").is_file(), reason="graph store not built")
sys.path.insert(0, str(ROOT / "scripts" / "queue"))
_spec = importlib.util.spec_from_file_location("make_specs_c5", ROOT / "scripts" / "queue" / "make_specs.py")
MS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(MS)
NEW = ("p1tune_v2", "dclass", "p1v2b", "pcompv2b")
CELL = ("model", "hp", "lr", "weight_decay", "batch_size", "split", "r", "budget", "seed")


def _gen(stage):
    return MS.generate(stage, NRP.CODE_REF, "v2", None, str(NRP.TUNED_V2), Path("x") / f"{stage}.jsonl")


@pytest.fixture()
def tuned_p1(tmp_path, monkeypatch):
    """A stand-in configs/tuned_p1_v2.yaml (the P1 values of tuned_v1, the v2 arch, 400 epochs)."""
    v1 = yaml.safe_load((ROOT / "configs" / "tuned_v1.yaml").read_text())
    f = tmp_path / "tuned_p1_v2.yaml"
    f.write_text(yaml.safe_dump({"version": "tuned-p1-v2", "arch": {"init": "kaiming"}, "max_epochs": 400,
                                 "patience": 30, "models": {"P1": v1["models"]["P1"]}}))
    monkeypatch.setitem(NRP.NEEDS_FILE, "p1v2b", f)
    monkeypatch.setitem(NRP.NEEDS_FILE, "pcompv2b", f)
    return f


def test_stage_tables():
    assert NRP.STAGES[10:14] == NEW and NRP.STAGES[:10] == ("relax", "peval", "embed", "geomeval", "moment", "cgcnn",
                                                         "capped", "cv", "precision", "moment_v2")
    for s in NEW:
        assert s in NRP.V2_STAGES and s in NRP.V2_ONLY and NRP.spec_name(s, "v2") == f"{s}.jsonl"
        assert s in NRP.GROUP and s in NRP.LABEL
        with pytest.raises(ValueError, match="v2 version only"):
            NRP.build_stage(s, backbone="v1")
    assert {s: NRP.EXPECTED[s] for s in NEW} == {"p1tune_v2": 3, "dclass": 180, "p1v2b": 180, "pcompv2b": 180}


def test_n6_stages_wait_for_the_tuned_p1_file(monkeypatch, tmp_path, capsys):
    missing = tmp_path / "absent.yaml"
    monkeypatch.setitem(NRP.NEEDS_FILE, "p1v2b", missing)
    monkeypatch.setitem(NRP.NEEDS_FILE, "pcompv2b", missing)
    assert "p1v2b" not in NRP.available_stages() and "pcompv2b" not in NRP.available_stages()
    assert {s for s, _ in NRP.spec_targets()} == set(NRP.available_stages()) == set(NRP.STAGES) - {"p1v2b", "pcompv2b"} - (
        set() if NRP.NEEDS_FILE["dretrain_v2"].is_file() else {"dretrain_v2"})
    for s in ("p1v2b", "pcompv2b"):
        with pytest.raises(FileNotFoundError, match="absent.yaml does not exist"):
            NRP.build_stage(s, backbone="v2")
    MS.all_targets(False, NRP.CODE_REF)
    out = capsys.readouterr().out
    assert "p1tune_v2" in out and "dclass" in out and "p1v2b" not in out and "pcompv2b" not in out


@needs_data
def test_p1tune_v2_jobs():
    cfg = load_config()
    jobs, body, man = _gen("p1tune_v2")
    assert [j["spec"]["anchor"] for j in jobs] == [50, 200, 654] == list(cfg.tuning.anchors)
    space = {k: v.model_dump(mode="json") for k, v in cfg.tuning.search_space.items()}
    for j in jobs:
        sp, k = j["spec"], j["key"]
        assert sp["study"] == f"v2_P1_a{sp['anchor']}" and sp["split"] == "outer_r0" and sp["seed"] == 0
        assert (sp["max_epochs"], sp["patience"], sp["n_trials"]) == (400, 30, 30)
        assert sp["arch"] == yaml.safe_load(NRP.TUNED_V2.read_text())["arch"] == {"init": "kaiming"}
        assert k["search_space"] == space and k["code_sha"] == NRP.CODE_REF and k["stage"] == "p1tune_v2"
        for test, seen in NRP.leakage_hosts(j):
            assert test == set(load_split("outer_r0")["test"]) and not test & seen
    assert json.loads(man)["sha256"] == hashlib.sha256(body).hexdigest()


@needs_data
def test_dclass_jobs_match_the_v2_d_sweep():
    jobs = _gen("dclass")[0]
    cells = sorted((j["spec"]["tags"]["dclass"], j["spec"]["r"], j["spec"]["budget"], j["spec"]["seed"]) for j in jobs)
    assert cells == sorted((v, r, b, s) for v in ("D-host", "D-site") for r in range(10) for b in (200, 400, 654)
                           for s in range(3))
    recs = NRP.v2_d_records("D-state")
    kept = NRP.descriptor_classes()
    for j in jobs:
        sp = j["spec"]
        rec = recs[(sp["split"], sp["budget"], sp["seed"])]["spec"]
        assert {k: sp[k] for k in CELL} == {k: rec[k] for k in CELL}            # same settings and split as D
        assert sp["ablate"] == NRP.DCLASS[sp["tags"]["dclass"]] and sp["results_subdir"] == "dclass"
        assert sp["tags"]["descriptors_kept"] == kept["host" if sp["tags"]["dclass"] == "D-host" else "site"]
    assert len(kept["host"]) == 12 and len(kept["site"]) == 10


@needs_data
def test_p1v2b_and_pcompv2b(tuned_p1):
    assert {"p1v2b", "pcompv2b"} <= set(NRP.available_stages())
    p1 = _gen("p1v2b")[0]
    pc, _, man = _gen("pcompv2b")
    old_p1 = [json.loads(f.read_text())["payload"]["spec"] for f in (ROOT / "results" / "p1_v2").glob("*.json")]
    want = sorted((s["split"], s["r"], s["budget"], s["seed"]) for s in old_p1 if s["split"] != "kiyohara")
    assert sorted((j["spec"]["split"], j["spec"]["r"], j["spec"]["budget"], j["spec"]["seed"]) for j in p1) == want
    assert all((j["spec"]["max_epochs"], j["spec"]["patience"]) == (400, 30) for j in p1)
    assert all(j["spec"]["hp"]["init"] == "kaiming" and j["spec"]["results_subdir"] == "p1v2b" for j in p1)
    p1_ids = {(j["spec"]["split"], j["spec"]["budget"], j["spec"]["seed"]): j["run_id"] for j in p1}
    old_p = {(s["split"], s["budget"], s["seed"]): s for s in
             (json.loads(f.read_text())["payload"]["spec"] for f in (ROOT / "results" / "p_v2").glob("*.json"))}
    gsha = json.loads(man)["graphs_manifest_sha256"]
    for j in pc:
        sp = j["spec"]
        k = (sp["split"], sp["budget"], sp["seed"])
        assert sp["model"] == "P" and sp["p1_run"] == p1_ids[k] and j["after"] == [p1_ids[k]]
        assert sp["d_run"] == old_p[k]["d_run"] and {c: sp[c] for c in CELL[1:]} == {c: old_p[k][c] for c in CELL[1:]}
        pin = sp["tags"]["checkpoints"][sp["d_run"]]
        assert pin["dirs"] == ["/workspace/dft-gnn-v2/checkpoints"] and len(pin["sha256"]) == 64
        assert run_id(RunSpec.from_dict(sp), NRP.CODE_REF, gsha) == j["run_id"]
    assert json.loads(man)["tuned_p1"]["sha256"] == hashlib.sha256(tuned_p1.read_bytes()).hexdigest()
    ids = [j["run_id"] for j in p1 + pc]
    assert len(set(ids)) == 360


def test_p1v2b_refuses_other_epochs(tuned_p1):
    t = yaml.safe_load(tuned_p1.read_text())
    tuned_p1.write_text(yaml.safe_dump({**t, "max_epochs": 200}))
    with pytest.raises(ValueError, match="max_epochs 200"):
        NRP.build_p1v2b("c", "g", NRP.load_tuned(NRP.TUNED_V2), load_config(), None)


def test_component_checkpoint(tmp_path, monkeypatch):
    local, vol = tmp_path / "ck", tmp_path / "vol"
    local.mkdir()
    vol.mkdir()
    (vol / "d1.pt").write_bytes(b"weights")
    good = hashlib.sha256(b"weights").hexdigest()
    base = {"model": "P", "hp": {}, "lr": 0.0, "weight_decay": 0.0, "batch_size": 1, "split": "outer_r0", "r": 0,
            "budget": 25, "seed": 0, "p1_run": "p1", "d_run": "d1"}
    assert component_checkpoint("d1", RunSpec(**base), local) == local / "d1.pt"          # unpinned: unchanged rule
    pinned = RunSpec(**base, tags={"checkpoints": {"d1": {"sha256": good, "record": "r", "dirs": [str(vol)]}}})
    assert component_checkpoint("d1", pinned, local) == vol / "d1.pt"
    assert component_checkpoint("p1", pinned, local) == local / "p1.pt"
    (vol / "d1.pt").write_bytes(b"other")
    with pytest.raises(RuntimeError, match="sha256 differs"):
        component_checkpoint("d1", pinned, local)
    (vol / "d1.pt").unlink()
    with pytest.raises(FileNotFoundError, match="DFTGNN_CKPT_DIRS"):
        component_checkpoint("d1", pinned, local)
    env = tmp_path / "env"
    env.mkdir()
    (env / "d1.pt").write_bytes(b"weights")
    monkeypatch.setenv("DFTGNN_CKPT_DIRS", str(env))
    assert component_checkpoint("d1", pinned, local) == env / "d1.pt"


def test_tuning_task_writes_the_s_and_d_record_format(tmp_path, monkeypatch):
    """run_p1tune with a stub trainer: the tuning record has the keys of the v2 S and D records, the epoch limit of the
    spec, and the task record lists the record, the trial CSV and the SQLite copy with their hashes."""
    import optuna

    from dftgnn import tasks
    from dftgnn import tune as T
    from dftgnn.io.results import write_result

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    job = next(j for j in json.loads("[" + ",".join((ROOT / "specs" / "p1tune_v2.jsonl").read_text().split("\n")[:-1])
                                     + "]") if j["spec"]["anchor"] == 50)
    calls = {"n": 0}

    def fake_train(spec, data, cfg, *, device=None, ckpt_path=None, log=print):
        assert (spec.max_epochs, spec.patience, spec.hp["init"], spec.model) == (400, 30, "kaiming", "P1")
        calls["n"] += 1
        return {"best_val_metric": 0.5 - 0.001 * calls["n"], "best_epoch": 9, "epochs_run": 400 if calls["n"] == 2
                else 40, "peak_memory_bytes": 0, "n_hosts": {"train": 45, "val": 5, "test": 0},
                "n_sites": {"train": 93, "val": 9, "test": 0}, "device": "cpu"}

    res = tmp_path / "results"
    monkeypatch.setattr(T, "RESULTS_DIR", res / "tuning")
    monkeypatch.setattr(tasks, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(tasks, "write_result", functools.partial(write_result, allow_dirty=True))
    real = T.run_study
    monkeypatch.setattr(T, "run_study", lambda *a, **k: real(*a, **{**k, "n_trials": 3, "train_fn": fake_train}))

    class _Stub:
        manifest_sha = "stub"

    out = tasks.run_p1tune(job["run_id"], job, _Stub(), __import__("torch").device("cpu"),
                           storage=tmp_path / "db", log=lambda *_: None)
    assert out == tmp_path / "results" / "p1tune_v2" / f"{job['run_id']}.json"
    assert tasks.verified(out, job["run_id"])
    rec = json.loads((res / "tuning" / "tuning_v2_P1_a50.json").read_text())["payload"]
    ref = json.loads((ROOT / "results" / "tuning" / "tuning_v2_S_a50.json").read_text())["payload"]
    assert set(rec) == set(ref)
    assert (rec["max_epochs"], rec["patience"], rec["n_complete"], rec["arch"]) == (400, 30, 3, {"init": "kaiming"})
    assert rec["trials_reaching_cap"] == [1] and rec["study"] == "v2_P1_a50"
    assert rec["objective"] == "val_mae_standardised_descriptors"
    pay = json.loads(out.read_text())["payload"]
    assert [Path(a["path"]).name for a in pay["artifacts"]] == ["tuning_v2_P1_a50.json", "trials_v2_P1_a50.csv",
                                                                "v2_P1_a50.db"]
    assert pay["code_ref"] == NRP.CODE_REF and pay["run_id"] == job["run_id"]


@needs_store
def test_dclass_name_check_in_training():
    from dftgnn.train import Store, train_run

    data = Store()
    sp = load_split("outer_r0")
    hosts = {"train": sp["budget_order"][:6], "val": sp["budget_order"][6:8], "test": sp["test"][:3]}
    base = {"model": "D-state", "hp": {"hidden_width": 64, "megnet_blocks": 2, "dropout": 0.0, "readout_mlp_width": 64,
                                       "pooling": "mean"}, "lr": 1e-3, "weight_decay": 1e-5, "batch_size": 8,
            "split": "outer_r0", "r": 0, "budget": 25, "seed": 0, "hosts": hosts, "max_epochs": 1, "patience": 5,
            "smoke": True}
    kept = NRP.descriptor_classes()
    with pytest.raises(ValueError, match="descriptors_kept"):
        train_run(RunSpec(**base, ablate="desc_site", tags={"descriptors_kept": kept["site"]}), data,
                  log=lambda *_: None)
    res = train_run(RunSpec(**base, ablate="desc_site", tags={"descriptors_kept": kept["host"]}), data,
                    log=lambda *_: None)
    assert res["epochs_run"] == 1
