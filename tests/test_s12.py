"""Stage builders, queue integration and supervisor behaviour for the new job types (CPU, mocks only)."""
from __future__ import annotations

import importlib.util
import itertools
import json
from collections import Counter
from pathlib import Path

import pytest

from dftgnn import jobqueue as Q
from dftgnn.train import LOCO_PREFIX, RunSpec, resolve_hosts, run_id
from dftgnn.train import s12 as S12
from dftgnn.train import stages as ST

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "data/processed/universe_v1.parquet").is_file(),
                                reason="universe parquet not built")
_spec = importlib.util.spec_from_file_location("supervisor", ROOT / "scripts" / "queue" / "supervisor.py")
sup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sup)
COUNTS = {"relax": 818, "peval": 183, "embed": 240, "geomeval": 180, "loco": 30, "moment": 60, "cgcnn": 18,
          "capped": 33}


@pytest.fixture(scope="module")
def jobs():
    return S12.build_s12("c0ffee" * 6 + "abcd", "g" * 64, ST.load_tuned())


def test_counts_unique_and_deterministic(jobs):
    assert Counter(j["stage"] for j in jobs) == COUNTS
    ids = [j["run_id"] for j in jobs]
    assert len(set(ids)) == len(ids)
    again = S12.build_s12("c0ffee" * 6 + "abcd", "g" * 64, ST.load_tuned())
    assert [j["run_id"] for j in again] == ids
    other = S12.build_s12("beef" * 10, "g" * 64, ST.load_tuned(), stages=("relax", "cgcnn"))
    assert not {j["run_id"] for j in other} & set(ids)          # the code SHA enters every id


def test_priority_order_and_tranches(jobs):
    pr = {s: [j["priority"] for j in jobs if j["stage"] == s] for s in COUNTS}
    order = S12.STAGE_ORDER
    for a, b in itertools.pairwise(order):
        assert max(pr[a]) < min(pr[b])
    assert {j["tranche"] for j in jobs if j["stage"] in ("relax", "peval", "embed", "geomeval")} == {"I"}
    assert [j["tranche"] for j in jobs][-1] == "X"


def test_dependencies(jobs):
    relax = {j["run_id"]: j for j in jobs if j["stage"] == "relax"}
    import pandas as pd

    excl = set(pd.read_csv(ROOT / "data/tiling_summary_v1.csv").query("excluded_from_C3b").host_id)
    for j in jobs:
        if j["stage"] == "geomeval":
            hosts = {relax[a]["spec"]["host_id"] for a in j["after"]}
            assert hosts and not hosts & excl and all(a in relax for a in j["after"])
        else:
            assert j["after"] == []


def test_peval_pairs_existing_checkpoints(jobs):
    from dftgnn.train import runindex

    prim = runindex.primary(runindex.load())
    by_id = {r["run_id"]: r for r in prim.values()}
    for j in (j for j in jobs if j["stage"] == "peval"):
        sp = j["spec"]
        p1, d = by_id[sp["p1_run"]], by_id[sp["d_run"]]
        for rec, m in ((p1, "P1"), (d, "D-state")):
            assert (rec["spec"]["model"], rec["spec"]["split"], rec["spec"]["budget"], rec["spec"]["seed"]) == (
                m, sp["split"], sp["budget"], sp["seed"])


def test_training_stage_specs(jobs):
    loco = [j for j in jobs if j["stage"] == "loco"]
    folds = json.loads((ROOT / "splits/loco_v1.json").read_text())["folds"]
    assert {(j["spec"]["split"], j["spec"]["budget"]) for j in loco} == {
        (f"{LOCO_PREFIX}{f['fold']}", f["n_train"]) for f in folds}
    mom = [j for j in jobs if j["stage"] == "moment"]
    assert all(len(j["spec"]["exclude_sites"]) == 10 and j["spec"]["budget"] == 654 for j in mom)
    cg = [j for j in jobs if j["stage"] == "cgcnn"]
    assert {(j["spec"]["lr"], j["spec"]["weight_decay"], j["spec"]["batch_size"]) for j in cg} == {(1e-3, 1e-5, 32)}
    cap = [j for j in jobs if j["stage"] == "capped"]
    _, skipped = S12.capped_twins()
    assert {s["original_run_id"] for s in skipped} == {"0b2035de2e44fade", "373c897db2d3b822", "d06182b6b418fb01"}
    assert not {j["spec"]["tags"]["original_run_id"] for j in cap} & {s["original_run_id"] for s in skipped}
    assert all(j["spec"]["max_epochs"] == 600 and j["spec"]["patience"] == 30 for j in cap)


def test_loco_hosts_and_exclusion_in_run_id():
    from dftgnn.config import load_config

    f = json.loads((ROOT / "splits/loco_v1.json").read_text())["folds"][2]
    tuned = ST.tuned_hparams(ST.load_tuned(), "S", 654)
    spec = RunSpec(model="S", **tuned, split=f"{LOCO_PREFIX}2", r=2, budget=f["n_train"], seed=1)
    h = resolve_hosts(spec, load_config())
    assert sorted(h["train"] + h["val"]) == f["train"] and h["test"] == f["test"]
    with pytest.raises(ValueError):
        resolve_hosts(RunSpec(model="S", **tuned, split=f"{LOCO_PREFIX}2", r=2, budget=654 + 7, seed=1), load_config())
    a = run_id(spec, "x", "y")
    spec.exclude_sites = ["b", "a"]
    assert run_id(spec, "x", "y") != a


def test_idempotent_enqueue(jobs, tmp_path):
    sub = [j for j in jobs if j["stage"] in ("cgcnn", "capped")]
    assert sum(Q.enqueue(tmp_path, j) for j in sub) == len(sub)
    assert sum(Q.enqueue(tmp_path, j) for j in sub) == 0


# ------------------------------------------------------------------------------- supervisor, mocked
def tjob(i, stage, after=(), tranche="I", **spec):
    return {"run_id": f"t{i:03d}", "stage": stage, "tranche": tranche, "priority": i, "after": list(after),
            "spec": {"model": stage, "r": 0, "budget": 654, "seed": 0, **spec}}


class Mock(sup.Ops):
    def __init__(self, a, jobs):
        super().__init__(a)
        self.jobs, self.t, self.alive, self.launched, self.packed = jobs, 1000.0, 0, 0, []

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s

    def workers_alive(self):
        return self.alive

    def launch_worker(self, i):
        self.alive += 1
        self.launched += 1

    def enqueue(self):
        for j in self.jobs:
            Q.enqueue(Path(self.a.queue), j)
        return [j["run_id"] for j in self.jobs]

    def pack(self, tranche):
        self.packed.append(tranche)
        return f"outbox_{tranche}.tar.gz"


def make(tmp_path, jobs, *extra):
    a = sup.parse(["--chain", str(tmp_path / "chain"), "--queue", str(tmp_path / "jobs"), "--workers", "4",
                   "--stage", "s12", "--tranches", "I,L,M,C,X", *extra])
    a.chain = Path(a.chain)
    return sup.Supervisor(a, Mock(a, jobs)), a


def _run(sv, ok=True, wall=50.0):
    j = sv.Q.claim(sv.queue, "w")
    sv.Q.finish(sv.queue, j, ok=ok, info={"wall_time_s": wall} if ok else {"wall_time_s": wall, "traceback": "x"})
    return j


def test_dependency_order_and_tranches(tmp_path):
    js = [tjob(0, "relax", n_atoms_uc=4, host_id="h0"), tjob(1, "relax", n_atoms_uc=8, host_id="h1"),
          tjob(2, "geomeval", after=["t000", "t001"], geo_model="S"), tjob(3, "embed", kind="init0"),
          tjob(4, "loco", tranche="L", split="loco_v1_f0")]
    sv, _ = make(tmp_path, js)
    sv.start()
    assert sup.Supervisor.result_file(js[0]).parts[-2] == "mlip_v1"
    order = [_run(sv)["run_id"] for _ in range(2)]
    assert order == ["t000", "t001"]
    assert _run(sv)["run_id"] == "t002"                     # geomeval becomes ready after both relax jobs
    _run(sv)
    sv.cycle()
    assert sv.ops.packed == ["I"]
    _run(sv)
    sv.cycle()
    assert sv.ops.packed == ["I", "L"] and sv.finished() == "ALL_DONE"


def test_failed_dependency_drains(tmp_path):
    js = [tjob(0, "relax", n_atoms_uc=4, host_id="h0"), tjob(1, "geomeval", after=["t000"], geo_model="S")]
    sv, _ = make(tmp_path, js)
    sv.start()
    _run(sv, ok=False)
    sv.retry_failed()
    assert sv.finished() is None                             # retried once: relax is pending again
    _run(sv, ok=False)
    sv.retry_failed()
    assert sv.blocked_by_failures() and sv.finished() == "DRAINED_WITH_FAILURES"


def test_guard_halts_with_task_priors(tmp_path):
    js = [tjob(i, "relax", n_atoms_uc=30, host_id=f"h{i}") for i in range(50)]
    sv, _ = make(tmp_path, js, "--guard", "0.01")
    sv.start()
    sv.top_up_workers()
    assert sv.check_guard("test") and (sv.chain / "HALT").exists()
    assert sv.top_up_workers() == 0                         # nothing new starts after HALT
    prior = sup.prior_s(sup.job_class(js[0]))
    assert prior == pytest.approx(30 * (0.068 + 0.0111 * 30))


def test_resume_after_kill_keeps_state(tmp_path):
    js = [tjob(i, "embed", kind="trained") for i in range(6)]
    sv, a = make(tmp_path, js)
    sv.start()
    sv.top_up_workers()
    _run(sv)
    t0 = sv.state["first_launch"]
    sv2 = sup.Supervisor(a, sv.ops)
    sv2.start()
    sv2.top_up_workers()
    assert sv2.state["first_launch"] == t0 and sv.ops.launched == 4
    assert len(list((sv2.queue / "pending").glob("*.json"))) == 5


def test_verified_task_result_moves_to_done(tmp_path, monkeypatch):
    monkeypatch.setattr(sup, "CHECKOUT", tmp_path)
    art = tmp_path / "data" / "x.npz"
    art.parent.mkdir(parents=True)
    art.write_bytes(b"abc")
    js = [tjob(0, "embed", kind="trained")]
    res = tmp_path / "results" / "embeddings_v1" / "t000.json"
    res.parent.mkdir(parents=True)
    res.write_text(json.dumps({"payload": {"run_id": "t000", "artifacts": [
        {"path": "data/x.npz", "sha256": sup.sha256(art)}]}}))
    sv, _ = make(tmp_path / "q", js)
    sv.start()
    assert (sv.queue / "done" / "t000.json").exists()
    art.write_bytes(b"changed")
    assert not sv.verified(js[0])
