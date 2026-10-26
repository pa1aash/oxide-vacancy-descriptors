"""Sweep stage builder tests: job counts, budget-to-anchor hyperparameters, P dependencies, run-id uniqueness."""
from __future__ import annotations

from collections import Counter

import pytest
import yaml

from dftgnn.config import load_config
from dftgnn.split import budget_train, load_split, val_split
from dftgnn.train import RunSpec, admission, resolve_hosts, val_seed
from dftgnn.train import stages as ST

CODE, GSHA = "c" * 40, "g" * 64
LR = {(m, a): 1e-4 * (1 + i) for i, (m, a) in enumerate((m, a) for m in ("S", "D-state", "D-late", "P1")
                                                          for a in (50, 200, 654))}


def _tuned(d="D-state"):
    def p(m, a):
        return {"learning_rate": LR[(m, a)], "weight_decay": 1e-5, "hidden_width": 64 if a < 654 else 128,
                "megnet_blocks": 3, "dropout": 0.1, "batch_size": 32, "readout_mlp_width": 64, "pooling": "mean"}
    models = {m: {a: p(m, a) for a in (50, 200, 654)} for m in ("S", "D-state", "D-late", "P1")}
    models["D"] = models[d]
    return {"d_variant": d, "models": models}


def _cfg(d="D-state"):
    cfg = load_config()
    cfg.models.injection_mode = d
    return cfg


def test_sweep_counts_and_unique_ids():
    jobs = ST.build_sweep(CODE, GSHA, _tuned(), _cfg())
    c = Counter(j["spec"]["model"] for j in jobs)
    assert c == {"S": 180, "D-state": 180, "P1": 180, "P": 180}
    assert sum(v for k, v in c.items() if k != "P") == 540
    assert len({j["run_id"] for j in jobs}) == len(jobs)
    assert {j["stage"] for j in jobs} == {"sweep"}
    cells = Counter((j["spec"]["r"], j["spec"]["budget"], j["spec"]["seed"]) for j in jobs)
    assert len(cells) == 10 * 6 * 3 and set(cells.values()) == {4}


def test_sweep_uses_the_budget_anchor_hyperparameters():
    cfg = _cfg()
    for j in ST.build_sweep(CODE, GSHA, _tuned(), cfg):
        s = j["spec"]
        m = "D-state" if s["model"] == "P" else s["model"]
        a = cfg.tuning.budget_to_anchor[s["budget"]]
        assert s["lr"] == LR[(m, a)]
        assert s["hp"]["hidden_width"] == (128 if a == 654 else 64)
        assert set(s["hp"]) == {"hidden_width", "megnet_blocks", "dropout", "readout_mlp_width", "pooling"}
        assert s["split"] == f"outer_r{s['r']}" and s["eval_test"] and not s["smoke"]


def test_p_jobs_depend_on_their_p1_and_d_runs():
    jobs = ST.build_sweep(CODE, GSHA, _tuned("D-late"), _cfg("D-late"))
    by_cell = {}
    for j in jobs:
        s = j["spec"]
        by_cell.setdefault((s["r"], s["budget"], s["seed"]), {})[s["model"]] = j
    for cell in by_cell.values():
        p = cell["P"]
        assert sorted(p["after"]) == sorted([cell["P1"]["run_id"], cell["D-late"]["run_id"]])
        assert p["spec"]["p1_run"] == cell["P1"]["run_id"] and p["spec"]["d_run"] == cell["D-late"]["run_id"]
        assert "D-state" not in cell


def test_sweep_refuses_a_variant_that_differs_from_the_config():
    with pytest.raises(ValueError):
        ST.build_sweep(CODE, GSHA, _tuned("D-state"), _cfg("D-late"))


def test_sweep_validation_hosts_follow_the_seed_rule():
    cfg = _cfg()
    j = next(j for j in ST.build_sweep(CODE, GSHA, _tuned(), cfg)
             if (j["spec"]["r"], j["spec"]["budget"], j["spec"]["seed"]) == (3, 200, 2) and j["spec"]["model"] == "S")
    h = resolve_hosts(RunSpec.from_dict(j["spec"]), cfg)
    sp = load_split("outer_r3")
    tr, va = val_split(budget_train(sp, 200), frac=0.1, min_hosts=3, seed=val_seed(3, 200, 2))
    assert (h["train"], h["val"], h["test"]) == (tr, va, sp["test"])


def test_kiyohara_jobs_use_the_654_anchor():
    t = _tuned()
    jobs = ST.build_kiyohara(CODE, GSHA, ST.kiyohara_hparams_from_tuned(t), _cfg(), d_variant="D-state")
    c = Counter(j["spec"]["model"] for j in jobs)
    assert c == {"S": 3, "D-state": 3, "P1": 3, "P": 3}
    for j in jobs:
        m = "D-state" if j["spec"]["model"] == "P" else j["spec"]["model"]
        assert j["spec"]["split"] == "kiyohara" and j["spec"]["lr"] == LR[(m, 654)]


def test_c0_official_jobs():
    jobs = ST.build_c0_official(CODE, GSHA, _tuned(), _cfg())
    assert [j["spec"]["seed"] for j in jobs] == [0, 1, 2]
    assert all(j["spec"]["model"] == "S" and j["spec"]["split"] == "kiyohara"
               and j["spec"]["results_subdir"] == "c0_official" and j["spec"]["lr"] == LR[("S", 654)] for j in jobs)
    assert len({j["run_id"] for j in jobs}) == 3


def test_jobs_carry_est_peak_gb_and_load_tuned_roundtrip(tmp_path):
    table = admission.load_benchmark()
    jobs = ST.build_sweep(CODE, GSHA, _tuned(), _cfg(), table=table)
    assert all(j["est_peak_gb"] > 0 for j in jobs)
    raw = _tuned()
    raw["models"] = {m: {str(a): p for a, p in by.items()} for m, by in raw["models"].items()}
    f = tmp_path / "tuned.yaml"
    f.write_text(yaml.safe_dump(raw))
    assert ST.load_tuned(f)["models"]["S"][654]["learning_rate"] == LR[("S", 654)]


def test_diag_cross_cells_take_the_anchor_hyperparameters():
    cfg = _cfg()
    jobs = ST.build_diag_cross(CODE, GSHA, _tuned(), cfg)
    hc = cfg.diagnostics.hparam_cross
    assert len(jobs) == len(hc.cells) * len(hc.resamples) * len(hc.seeds)
    assert len({j["run_id"] for j in jobs}) == len(jobs)
    for j in jobs:
        sp = j["spec"]
        assert sp["model"] == "S" and sp["results_subdir"] == "diag_cross"
        assert sp["lr"] == LR[("S", sp["tags"]["anchor"])]
    assert {(j["spec"]["budget"], j["spec"]["tags"]["anchor"]) for j in jobs} == {tuple(c) for c in hc.cells}


def test_d_ablation_jobs():
    cfg = _cfg()
    jobs = ST.build_d_ablation(CODE, GSHA, _tuned(), cfg)
    da = cfg.diagnostics.descriptor_ablation
    assert len(jobs) == len(da.ablations) * len(da.resamples) * len(da.seeds)
    assert Counter(j["spec"]["ablate"] for j in jobs) == {None: 9, "desc_host": 9, "desc_site": 9}
    assert all(j["spec"]["model"] == "D-state" and j["spec"]["budget"] == 654
               and j["spec"]["lr"] == LR[("D-state", 654)] for j in jobs)
    assert len({j["run_id"] for j in jobs}) == len(jobs)


def test_dlate_jobs_use_the_unselected_variant_at_its_own_anchor():
    cfg = _cfg()
    jobs = ST.build_dlate(CODE, GSHA, _tuned(), cfg)
    assert len(jobs) == 3 * 10 * 3
    b2a = cfg.tuning.budget_to_anchor
    assert all(j["spec"]["model"] == "D-late" and j["spec"]["lr"] == LR[("D-late", b2a[j["spec"]["budget"]])]
               for j in jobs)
    with pytest.raises(ValueError, match="selected D"):
        ST.build_dlate(CODE, GSHA, _tuned("D-late"), _cfg("D-late"))


def test_review_stage_is_prioritised_and_disjoint_from_the_sweep():
    cfg = _cfg()
    jobs = ST.build_review(CODE, GSHA, _tuned(), cfg)
    assert [j["priority"] for j in jobs] == list(range(len(jobs)))
    assert {j["stage"] for j in jobs[:63]} == {"diag_cross", "d_ablation"} and jobs[-1]["stage"] == "dlate"
    sweep = {j["run_id"] for j in ST.build_sweep(CODE, GSHA, _tuned(), cfg)}
    assert not sweep & {j["run_id"] for j in jobs}


def test_loco_jobs():
    cfg = _cfg()
    jobs = ST.build_loco(CODE, GSHA, _tuned(), cfg)
    assert len(jobs) == 5 * 3 * 2 and len({j["run_id"] for j in jobs}) == 30
    for j in jobs:
        sp = j["spec"]
        k = sp["r"]
        assert sp["split"] == f"loco/loco_f{k}" and sp["budget"] == len(load_split(f"loco/loco_f{k}")["budget_order"])
        assert sp["lr"] == LR[(sp["model"], 654)]
        tr = resolve_hosts(RunSpec.from_dict(sp), cfg)
        assert not set(tr["test"]) & (set(tr["train"]) | set(tr["val"]))
    v2 = ST.build_loco(CODE, GSHA, _tuned(), cfg, arch={"init": "kaiming"}, subdir="loco_v2")
    assert not {j["run_id"] for j in v2} & {j["run_id"] for j in jobs}


def test_v2_screen_never_evaluates_test_hosts():
    cfg = _cfg()
    jobs = ST.build_v2_screen(CODE, GSHA, _tuned(), cfg)
    assert len(jobs) == len(cfg.v2.variants) * 3 * 3 and len({j["run_id"] for j in jobs}) == len(jobs)
    assert all(j["spec"]["eval_test"] is False and j["spec"]["split"] == "outer_r0" for j in jobs)
    for j in jobs:
        sp = j["spec"]
        assert {k: sp["hp"][k] for k in cfg.v2.variants[sp["tags"]["v2_variant"]]} == cfg.v2.variants[sp["tags"]["v2_variant"]]
        assert resolve_hosts(RunSpec.from_dict(sp), cfg)["test"] == []


def test_v2_stage_carries_the_frozen_backbone_everywhere():
    cfg = _cfg()
    t = _tuned()
    t["arch"] = {"init": "kaiming"}
    t["models"] = {m: t["models"][m] for m in ("S", "D-state", "D")}
    jobs = ST.build_v2(CODE, GSHA, t, cfg)
    stages = Counter(j["stage"] for j in jobs)
    assert stages == {"v2_sweep": 10 * 6 * 3 * 2, "v2_kiyohara": 6, "loco_v2": 30}
    assert all(j["spec"]["hp"]["init"] == "kaiming" for j in jobs)
    assert len({j["run_id"] for j in jobs}) == len(jobs)
    assert [j["priority"] for j in jobs] == list(range(len(jobs)))
    v1 = {j["run_id"] for j in ST.build_sweep(CODE, GSHA, _tuned(), cfg)}
    assert not v1 & {j["run_id"] for j in jobs}
    with pytest.raises(ValueError, match="not a v2"):
        ST.build_v2(CODE, GSHA, _tuned(), cfg)


def test_p_v2_composes_p1_v2_with_the_queued_d_v2_runs():
    cfg = _cfg()
    t2 = _tuned()
    t2["arch"] = {"init": "kaiming"}
    v2 = {j["run_id"]: j for j in ST.build_v2(CODE, GSHA, t2, cfg)}
    jobs = ST.build_p_v2(CODE, GSHA, t2, _tuned(), cfg)
    p1 = [j for j in jobs if j["stage"] == "p1_v2"]
    p = [j for j in jobs if j["stage"] == "p_v2"]
    assert len(p1) == len(p) == 183
    assert all(j["spec"]["hp"]["init"] == "kaiming" and j["spec"]["model"] == "P1" for j in p1)
    p1_ids = {j["run_id"] for j in p1}
    for j in p:
        p1_run, d_run = j["after"]
        assert p1_run in p1_ids and d_run in v2 and v2[d_run]["spec"]["model"] == "D-state"
        assert v2[d_run]["spec"]["budget"] == j["spec"]["budget"] and v2[d_run]["spec"]["seed"] == j["spec"]["seed"]
    assert min(j["priority"] for j in jobs) > max(j["priority"] for j in v2.values())
