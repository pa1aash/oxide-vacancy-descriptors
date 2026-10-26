"""The descriptor-encoder option of D, the extended search space of dtune_v2 (opt-in options of dftgnn.tune), the
dtune_v2 task, the freeze script on a synthetic study and the dretrain_v2 builder with its missing-config
refusal. CPU only; the stages that read the data skip without them."""
from __future__ import annotations

import functools
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import optuna
import pytest
import torch
import yaml
from torch_geometric.data import Batch

from dftgnn import tasks
from dftgnn import tune as T
from dftgnn.config import load_config
from dftgnn.graphs import host_graph, site_data
from dftgnn.io.results import write_result
from dftgnn.models import KINDS, HParams, build_model
from dftgnn.train import CODE_REF_SUCCESSORS
from dftgnn.train import nrp as NRP

ROOT = Path(__file__).resolve().parents[1]
needs_data = pytest.mark.skipif(not (ROOT / "data/processed/universe_v1.parquet").is_file(),
                                reason="universe parquet not built")
sys.path.insert(0, str(ROOT / "scripts" / "queue"))
_spec = importlib.util.spec_from_file_location("make_specs_c7", ROOT / "scripts" / "queue" / "make_specs.py")
MS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(MS)
_fspec = importlib.util.spec_from_file_location("freeze_d_v2", ROOT / "scripts" / "tune" / "freeze_d_v2.py")
FZ = importlib.util.module_from_spec(_fspec)
_fspec.loader.exec_module(FZ)
optuna.logging.set_verbosity(optuna.logging.WARNING)
N_HOST, N_SITE = 12, 10


@pytest.fixture(scope="module")
def graph():
    from pymatgen.core import Lattice, Structure

    s = Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.21), ["Mg", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    s.make_supercell([2, 2, 2])
    return host_graph(s, 5.0)


def _batch(g, desc):
    vacs = torch.nonzero(g["z"] == 8).flatten().tolist()[: len(desc)]
    return Batch.from_data_list([site_data(g, v, y=1.0, desc_host=desc[i, :N_HOST], desc_site=desc[i, N_HOST:])
                                 for i, v in enumerate(vacs)])


@functools.cache
def _v7_models():
    """``dftgnn.models`` as tagged in nrp-code-v7 (before the descriptor encoder), imported from a temporary copy."""
    src = subprocess.run(["git", "show", "nrp-code-v7:src/dftgnn/models/__init__.py"], cwd=ROOT, capture_output=True,
                         text=True, check=False)
    if src.returncode:
        pytest.skip("tag nrp-code-v7 not in this clone")
    path = Path(tempfile.mkdtemp()) / "models_v7.py"
    path.write_text(src.stdout)
    spec = importlib.util.spec_from_file_location("models_v7", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["models_v7"] = mod          # dataclasses resolve their module by name
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------------- model option
@pytest.mark.parametrize("init", ["default", "kaiming"])
@pytest.mark.parametrize("kind", KINDS)
def test_d_v2_default_network_equals_the_tagged_v7_network(graph, kind, init):
    """Default options: the same modules, the same initial weights from the same seed, the same outputs and the same
    hyperparameter dict as the code of nrp-code-v7."""
    old = _v7_models()
    torch.manual_seed(3)
    a = build_model(kind, HParams(init=init)).eval()
    torch.manual_seed(3)
    b = old.build_model(kind, old.HParams(init=init)).eval()
    assert a.state_dict().keys() == b.state_dict().keys()
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())
    assert a.config == b.config and HParams().to_dict() == old.HParams().to_dict()
    desc = torch.randn(2, N_HOST + N_SITE)
    with torch.no_grad():
        assert torch.equal(a(_batch(graph, desc)), b(_batch(graph, desc)))


@pytest.mark.parametrize("kind", ["D-state", "D-late"])
def test_d_v2_descriptor_encoder(graph, kind):
    torch.manual_seed(0)
    hp = HParams(init="kaiming", desc_layers=2, desc_width=16)
    net = build_model(kind, hp).eval()
    assert net.desc_host_enc is not None and net.config["hp"]["desc_layers"] == 2
    assert HParams(**net.config["hp"]) == hp
    desc = torch.randn(2, N_HOST + N_SITE)
    with torch.no_grad():
        real = net(_batch(graph, desc))
        site0 = net(_batch(graph, torch.cat([desc[:, :N_HOST], torch.zeros(2, N_SITE)], 1)))
    assert real.shape == (2,) and torch.isfinite(real).all() and not torch.allclose(real, site0)


def test_d_v2_descriptor_encoder_refused_outside_d():
    for kind in ("S", "P1"):
        with pytest.raises(ValueError, match="desc_layers"):
            build_model(kind, HParams(desc_layers=1, desc_width=16))
    with pytest.raises(ValueError, match="desc_layers"):
        build_model("D-state", HParams(desc_layers=1, desc_width=0))
    assert build_model("D-state", HParams(desc_layers=0, desc_width=32)).desc_host_enc is None


# ------------------------------------------------------------------------------------------------- tuning options
def test_tuning_defaults_unchanged():
    cfg = load_config()
    p = {"learning_rate": 1e-3, "weight_decay": 1e-5, "batch_size": 16, "hidden_width": 64, "megnet_blocks": 3,
         "dropout": 0.1, "readout_mlp_width": 64, "pooling": "mean"}
    assert T.split_params(p) == {"hp": {k: p[k] for k in T.ARCH_KEYS}, "lr": 1e-3, "weight_decay": 1e-5,
                                 "batch_size": 16}
    assert T.csv_fields(None) == T.CSV_FIELDS and T.trial_model("S", p) == "S"
    with pytest.raises(ValueError, match="not a tuned model"):
        T.run_study("D", 50, None, cfg=cfg)


def test_dtune_space_is_a_superset():
    cfg = load_config()
    space = NRP.dtune_space(cfg)
    old = {k: v.model_dump(mode="json") for k, v in cfg.tuning.search_space.items()}
    assert set(old) < set(space) and set(space) - set(old) == {"injection", "desc_layers", "desc_width"}
    for k, o in old.items():
        n = space[k]
        if o["choices"] is None:
            assert n["low"] <= o["low"] and n["high"] >= o["high"]
        else:
            assert set(o["choices"]) <= set(n["choices"])
    assert "D-state" in space["injection"]["choices"] and 0 in space["desc_layers"]["choices"]
    assert (space["dropout"]["high"], space["weight_decay"]["low"], space["weight_decay"]["high"]) == (0.5, 1e-7, 1e-2)


def _fake_train(seen=None):
    calls = {"n": 0}

    def fake(spec, data, cfg, *, device=None, ckpt_path=None, log=print):
        calls["n"] += 1
        assert spec.eval_test is False and spec.model in ("D-state", "D-late")
        assert "injection" not in spec.hp and {"desc_layers", "desc_width"} <= set(spec.hp)
        if seen is not None:
            seen.append(spec)
        return {"best_val_metric": 0.5 - 0.001 * calls["n"], "best_epoch": 9, "epochs_run": 400 if calls["n"] == 2
                else 40, "peak_memory_bytes": 0, "n_hosts": {"train": 45, "val": 5, "test": 0},
                "n_sites": {"train": 93, "val": 9, "test": 0}, "device": "cpu"}
    return fake


def _jobs():
    return [json.loads(ln) for ln in (ROOT / "specs" / "dtune_v2.jsonl").read_text().splitlines() if ln]


def _run_dtune(tmp_path, monkeypatch, anchors=(50,), n_trials=4, seen=None):
    res = tmp_path / "results"
    monkeypatch.setattr(T, "RESULTS_DIR", res / "tuning")
    monkeypatch.setattr(tasks, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(tasks, "write_result", functools.partial(write_result, allow_dirty=True))
    real = T.run_study
    monkeypatch.setattr(T, "run_study", lambda *a, **k: real(*a, **{**k, "n_trials": n_trials,
                                                                     "train_fn": _fake_train(seen)}))

    class _Stub:
        manifest_sha = "stub"

    out = {}
    for j in _jobs():
        if j["spec"]["anchor"] in anchors:
            out[j["spec"]["anchor"]] = (j, tasks.run_tuning(j["run_id"], j, _Stub(), torch.device("cpu"),
                                                            storage=tmp_path / "db", log=lambda *_: None))
    return res, out


def test_dtune_v2_task_writes_the_tuning_record(tmp_path, monkeypatch):
    """run_tuning on the dtune_v2 job of anchor 50 with a stub trainer: trials train the sampled variant with the
    encoder options in hp, the record has the keys of the v2 S and D records plus the space and the best variant,
    and the task record lists the record, the trial CSV and the SQLite copy."""
    seen = []
    res, out = _run_dtune(tmp_path, monkeypatch, seen=seen)
    job, path = out[50]
    assert path == tmp_path / "results" / "dtune_v2" / f"{job['run_id']}.json" and tasks.verified(path, job["run_id"])
    assert all((s.max_epochs, s.patience, s.hp["init"], s.budget, s.r) == (400, 30, "kaiming", 50, 0) for s in seen)
    rec = json.loads((res / "tuning" / "tuning_v2_D_a50.json").read_text())["payload"]
    ref = json.loads((ROOT / "results" / "tuning" / "tuning_v2_S_a50.json").read_text())["payload"]
    assert set(rec) == set(ref) | {"search_space", "best_model"}
    assert rec["search_space"] == NRP.dtune_space(load_config()) and rec["objective"] == "val_mae_Ef_eV"
    assert rec["best_model"] == rec["best_params"]["injection"] and "injection" not in rec["best_hparams"]["hp"]
    assert (rec["max_epochs"], rec["patience"], rec["n_complete"], rec["trials_reaching_cap"]) == (400, 30, 4, [1])
    header = (res / "tuning" / "trials_v2_D_a50.csv").read_text().splitlines()[0].split(",")
    assert {"injection", "desc_layers", "desc_width"} <= set(header)
    pay = json.loads(path.read_text())["payload"]
    assert [Path(a["path"]).name for a in pay["artifacts"]] == ["tuning_v2_D_a50.json", "trials_v2_D_a50.csv",
                                                                "v2_D_a50.db"]


def test_dtune_v2_jobs_and_test_hosts():
    """Three studies at the configured anchors on resample 0's inner hosts; the leakage rule of p1tune_v2 applies."""
    cfg = load_config()
    jobs = _jobs()
    assert [j["spec"]["anchor"] for j in jobs] == list(cfg.tuning.anchors)
    for j in jobs:
        sp = j["spec"]
        assert (sp["model"], sp["max_epochs"], sp["patience"], sp["n_trials"], sp["r"], sp["seed"]) == \
            ("D", 400, 30, 30, 0, 0)
        assert sp["search_space"] == NRP.dtune_space(cfg) and j["stage"] == "dtune_v2" and j["group"] == "N7"
        (test, seen), = NRP.leakage_hosts(j, cfg)
        assert test and not test & seen
    assert "dtune_v2" in tasks.TASK_STAGES and tasks.SUBDIR["dtune_v2"] == "dtune_v2"


# ------------------------------------------------------------------------------------------------- freeze, dretrain
def test_freeze_d_v2_on_a_synthetic_study(tmp_path, monkeypatch):
    """Three synthetic 30-trial studies -> configs/tuned_d_v2.yaml that round-trips and that the dretrain_v2 helper
    reads; a record with a different epoch limit is refused."""
    res, _ = _run_dtune(tmp_path, monkeypatch, anchors=(50, 200, 654), n_trials=30)
    cfg = load_config()
    out, summaries = tmp_path / "tuned_d_v2.yaml", []
    studies = FZ.freeze(res / "tuning", out, cfg, lambda n, p: summaries.append((n, p)) or n, "abc123")
    back = yaml.safe_load(out.read_text())
    assert (back["version"], back["max_epochs"], back["patience"], back["arch"]) == ("tuned-d-v2", 400, 30,
                                                                                     {"init": "kaiming"})
    assert summaries[0][0] == "tuning_d_v2_summary" and set(summaries[0][1]["best"]) == {"D|50", "D|200", "D|654"}
    td = NRP.load_tuned_d(out)
    for a, pay in studies.items():
        kind, hp = NRP.tuned_d_hparams(td, a)
        assert kind == pay["best_model"] and hp["hp"]["init"] == "kaiming"
        assert {k: v for k, v in hp["hp"].items() if k != "init"} == pay["best_hparams"]["hp"]
    rec = res / "tuning" / "tuning_v2_D_a200.json"
    d = json.loads(rec.read_text())
    d["payload"]["max_epochs"] = 200
    rec.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="max_epochs"):
        FZ.freeze(res / "tuning", out, cfg, lambda n, p: n, "abc123")


def test_dretrain_v2_missing_config_is_refused(monkeypatch, tmp_path, capsys):
    missing = tmp_path / "absent_d.yaml"
    monkeypatch.setitem(NRP.NEEDS_FILE, "dretrain_v2", missing)
    assert "dretrain_v2" not in NRP.available_stages() and "dtune_v2" in NRP.available_stages()
    with pytest.raises(FileNotFoundError, match="absent_d.yaml does not exist: the stage dretrain_v2 needs"):
        NRP.build_stage("dretrain_v2", backbone="v2")
    monkeypatch.setattr(sys, "argv", ["make_specs.py", "--stage", "dretrain_v2", "--check"])
    with pytest.raises(SystemExit, match="freeze_d_v2.py"):
        MS.main()
    MS.all_targets(False, NRP.CODE_REF)
    assert "dretrain_v2" not in capsys.readouterr().out


@needs_data
def test_dretrain_v2_spec_pairs_with_the_existing_d_runs(tmp_path, monkeypatch):
    """Test dretrain v2 spec pairs with the existing d runs."""
    import glob

    v1 = yaml.safe_load((ROOT / "configs" / "tuned_v1.yaml").read_text())["models"]["D"]
    f = tmp_path / "tuned_d_v2.yaml"
    models = {a: {**p, "injection": "D-late" if a == 50 else "D-state", "desc_layers": 1, "desc_width": 32}
              for a, p in v1.items()}
    f.write_text(yaml.safe_dump({"version": "tuned-d-v2", "arch": {"init": "kaiming"}, "max_epochs": 400,
                                 "patience": 30, "models": {"D": models}}))
    monkeypatch.setitem(NRP.NEEDS_FILE, "dretrain_v2", f)
    jobs, _body, man = MS.generate("dretrain_v2", NRP.CODE_REF, "v2", None, str(NRP.TUNED_V2), Path("x/d.jsonl"))
    assert len(jobs) == 210 == NRP.EXPECTED["dretrain_v2"] and len({j["run_id"] for j in jobs}) == 210
    assert json.loads(man)["tuned_d"]["path"] == str(f) and "tuned_p1" not in json.loads(man)
    cells = {}
    for sub in ("v2_sweep", "precision_v2"):
        for p in glob.glob(str(ROOT / "results" / sub / "*.json")):
            sp = json.loads(Path(p).read_text())["payload"]["spec"]
            cells.setdefault(sp["model"], set()).add((sp["split"], sp["budget"], sp["seed"]))
    got = {(j["spec"]["split"], j["spec"]["budget"], j["spec"]["seed"]) for j in jobs}
    assert got <= cells["D-state"] and got <= cells["S"]
    assert {b: sum(c[1] == b for c in got) for b in (654, 400, 200)} == {654: 150, 400: 30, 200: 30}
    for j in jobs:
        sp = j["spec"]
        assert (sp["model"], sp["max_epochs"], sp["patience"], sp["results_subdir"]) == ("D-state", 400, 30,
                                                                                         "dretrain_v2")
        assert sp["hp"]["desc_layers"] == 1 and sp["hp"]["init"] == "kaiming" and j["group"] == "N8"
    monkeypatch.setitem(NRP.NEEDS_FILE, "dretrain_v2", tmp_path / "bad.yaml")
    (tmp_path / "bad.yaml").write_text(f.read_text().replace("max_epochs: 400", "max_epochs: 200"))
    with pytest.raises(ValueError, match="max_epochs 200"):
        NRP.build_stage("dretrain_v2", backbone="v2")


def test_code_ref_successors_include_v9():
    assert CODE_REF_SUCCESSORS["nrp-code-v1"] == ("nrp-code-v2", "nrp-code-v4", "nrp-code-v5", "nrp-code-v6",
                                                  "nrp-code-v7", "nrp-code-v8", "nrp-code-v9")
    assert NRP.STAGES[:16] == ("relax", "peval", "embed", "geomeval", "moment", "cgcnn", "capped", "cv", "precision",
                               "moment_v2", "p1tune_v2", "dclass", "p1v2b", "pcompv2b", "dtune_v2", "dretrain_v2")
