"""Unique run ids, scheduling order, epoch-cap twins, test-host separation, reuse of the official runs,
priority-ordered claims, tranche packing and the HALT stop."""
from __future__ import annotations

import json
import random
from collections import Counter

from dftgnn import jobqueue as Q
from dftgnn.config import load_config
from dftgnn.train import RunSpec, resolve_hosts
from dftgnn.train import stages as ST
from test_sweep_stage import CODE, GSHA, _cfg, _tuned

C0_CODE = "d" * 40


def _session():
    return ST.build_session(CODE, GSHA, _tuned(), _cfg(), c0_code=C0_CODE)


def test_counts_unique_ids_and_no_p_jobs():
    jobs, reused = _session()
    assert len(jobs) == 564 and len({j["run_id"] for j in jobs}) == 564
    assert "P" not in {j["spec"]["model"] for j in jobs}
    c = Counter((j["stage"], j["spec"]["split"] == "kiyohara", j["spec"]["model"]) for j in jobs)
    assert c[("sweep", False, "S")] == c[("sweep", False, "D-state")] == c[("sweep", False, "P1")] == 180
    assert c[("kiyohara", True, "D-state")] == c[("kiyohara", True, "P1")] == 3 and ("kiyohara", True, "S") not in c
    assert c[("capsens", False, "S")] == c[("capsens", False, "D-state")] == 9
    assert len(reused) == 3 and all(j["spec"]["results_subdir"] == "c0_official" for j in reused)
    assert not {j["run_id"] for j in reused} & {j["run_id"] for j in jobs}


def test_priority_order_and_tranches():
    jobs, _ = _session()
    assert [j["priority"] for j in jobs] == list(range(len(jobs)))
    cls = []
    for j in jobs:
        sp = j["spec"]
        cls.append(1 if sp["split"] == "kiyohara" and j["spec"]["model"] != "P1" else
                   3 if j["stage"] == "capsens" else 4 if sp["model"] == "P1" else 2)
    assert cls == sorted(cls)                                        # Kiyohara, sweep, capsens, P1
    sweep = [j["spec"] for j in jobs if j["stage"] == "sweep" and j["spec"]["model"] != "P1"]
    keys = [(-s["budget"], s["r"], s["seed"]) for s in sweep]
    assert keys == sorted(keys)                                      # budget descending, resample, seed
    assert Counter(j["tranche"] for j in jobs) == {"T1": 123, "T2": 240, "T3": 18, "T4": 183}
    for j in jobs:
        sp = j["spec"]
        if j["tranche"] == "T1":
            assert sp["split"] == "kiyohara" or sp["budget"] in (654, 400)
        if j["tranche"] == "T2":
            assert sp["budget"] in (200, 100, 50, 25) and sp["model"] != "P1"
        assert (j["tranche"] == "T4") == (sp["model"] == "P1")
    last_t1 = max(j["priority"] for j in jobs if j["tranche"] == "T1")
    assert last_t1 < min(j["priority"] for j in jobs if j["tranche"] == "T2")


def test_capsens_jobs_use_600_epochs_and_distinct_ids():
    jobs, _ = _session()
    caps = [j for j in jobs if j["stage"] == "capsens"]
    twins = {(j["spec"]["model"], j["spec"]["r"], j["spec"]["seed"]): j for j in jobs
             if j["stage"] == "sweep" and j["spec"]["budget"] == 654}
    assert len(caps) == 18
    for j in caps:
        s = j["spec"]
        assert (s["max_epochs"], s["patience"], s["budget"], s["r"] in (0, 1, 2)) == (600, 30, 654, True)
        assert s["results_subdir"] == "capsens"
        t = twins[(s["model"], s["r"], s["seed"])]
        assert t["spec"]["max_epochs"] is None and t["run_id"] != j["run_id"]
        assert {k: v for k, v in s.items() if k not in ("max_epochs", "patience", "results_subdir", "tags")} == \
               {k: v for k, v in t["spec"].items() if k not in ("max_epochs", "patience", "results_subdir", "tags")}


def test_no_test_host_in_training_or_validation_for_20_jobs():
    cfg = load_config()
    jobs, _ = _session()
    for j in random.Random(0).sample(jobs, 20):
        h = resolve_hosts(RunSpec.from_dict(j["spec"]), cfg)
        assert h["test"] and not set(h["test"]) & (set(h["train"]) | set(h["val"]))
        assert not set(h["train"]) & set(h["val"])


def test_reused_c0_ids_are_the_published_c0_runs():
    files = sorted((ST.TUNED.parents[1] / "results" / "c0_official").glob("*.json"))
    pay = json.loads(files[0].read_text())["payload"]
    jobs = ST.build_c0_official(pay["code_sha"], pay["graphs_manifest_sha256"], ST.load_tuned())
    assert {j["run_id"] for j in jobs} == {f.stem for f in files} and len(files) == 3


def test_claims_follow_priority_then_solo(tmp_path):
    root = Q.init(tmp_path / "jobs")
    for name, pr in (("a", 5), ("b", 2), ("c", 9), ("d", 0)):
        Q.enqueue(root, {"run_id": name, "stage": "t", "spec": {"model": "S"}, "after": [], "priority": pr})
    assert [Q.claim(root, "w")["run_id"] for _ in range(4)] == ["d", "b", "a", "c"]


def test_worker_stops_on_halt(tmp_path):
    root = Q.init(tmp_path / "jobs")
    for i in range(3):
        Q.enqueue(root, {"run_id": f"j{i}", "stage": "t", "spec": {"model": "S"}, "after": []})
    flag = tmp_path / "HALT"
    ran = []

    def ex(job):
        ran.append(job["run_id"])
        flag.write_text("guard")
        return {}

    n = Q.run_worker(root, "w", ex, poll=0.01, log=lambda *_: None, should_stop=flag.exists)
    assert n == 1 and ran == ["j0"] and len(list((root / "pending").glob("*.json"))) == 2
