"""Tests of the pod-side sweep supervisor (scripts/queue/supervisor.py) with mocked processes and a real
stub-worker drill on the CPU: worker restart, no double launch, resume after a kill, guard halt, idempotent
enqueue, retry-once, tranche packing, ALL_DONE and skipping verified results."""
from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from dftgnn import jobqueue as Q

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("supervisor", ROOT / "scripts" / "queue" / "supervisor.py")
sup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sup)


def job(i, tranche="T1", model="S", budget=654, r=0, seed=0, **extra):
    return {"run_id": f"j{i:03d}", "stage": "sweep", "tranche": tranche, "priority": i, "after": [],
            "spec": {"model": model, "budget": budget, "r": r, "seed": seed, "split": f"outer_r{r}"}, **extra}


class Mock(sup.Ops):
    """No processes: ``alive`` is a counter the test manipulates; enqueue uses the real queue."""

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


def make(tmp_path, jobs, **kw):
    a = sup.parse(["--chain", str(tmp_path / "chain"), "--queue", str(tmp_path / "jobs"), "--workers", "4"])
    a.chain = Path(a.chain)
    for k, v in kw.items():
        setattr(a, k, v)
    ops = Mock(a, jobs)
    return sup.Supervisor(a, ops), ops


def finish(sv, rid, ok=True, wall=100.0, tb="boom"):
    """Move a job through running/ to done/ or failed/ the way a worker does."""
    j = sv.Q.claim(sv.queue, "w")
    assert j["run_id"] == rid, (j["run_id"], rid)
    sv.Q.finish(sv.queue, j, ok=ok, info={"wall_time_s": wall} if ok else {"wall_time_s": wall, "traceback": tb})


def test_idempotent_enqueue_and_four_workers_never_more(tmp_path):
    sv, ops = make(tmp_path, [job(i) for i in range(10)])
    sv.start()
    sv.start()
    assert len(list((sv.queue / "pending").glob("*.json"))) == 10
    for _ in range(5):
        sv.top_up_workers()
    assert ops.launched == 4 and ops.alive == 4          # repeated cycles do not double-launch


def test_dead_worker_is_replaced_and_fewer_jobs_than_workers(tmp_path):
    sv, ops = make(tmp_path, [job(i) for i in range(10)])
    sv.start()
    sv.top_up_workers()
    ops.alive -= 1                                         # a worker dies
    sv.top_up_workers()
    assert ops.alive == 4 and ops.launched == 5
    sv2, ops2 = make(tmp_path / "x", [job(0), job(1)])
    sv2.start()
    sv2.top_up_workers()
    assert ops2.launched == 2                              # never more workers than jobs


def test_resume_after_supervisor_kill_keeps_clock_and_workers(tmp_path):
    sv, ops = make(tmp_path, [job(i) for i in range(10)])
    sv.start()
    sv.top_up_workers()
    t0 = sv.state["first_launch"]
    sv2 = sup.Supervisor(sv.a, ops)                        # a fresh supervisor process, same chain dir
    sv2.start()
    sv2.top_up_workers()
    assert ops.launched == 4 and sv2.state["first_launch"] == t0


def test_orphaned_job_is_requeued_after_stale_heartbeat(tmp_path):
    sv, ops = make(tmp_path, [job(0)])
    sv.start()
    j = sv.Q.claim(sv.queue, "dead-worker")
    ops.t = time.time() + 1000
    assert sv.requeue_orphans() == [j["run_id"]]
    assert (sv.queue / "pending" / f"{j['run_id']}.json").exists()


def test_failed_job_is_retried_once_then_stays_failed(tmp_path):
    sv, _ops = make(tmp_path, [job(0)])
    sv.start()
    finish(sv, "j000", ok=False)
    sv.retry_failed()
    assert (sv.queue / "pending" / "j000.json").exists() and not list((sv.queue / "failed").glob("*.json"))
    finish(sv, "j000", ok=False)
    sv.retry_failed()
    assert (sv.queue / "failed" / "j000.json").exists() and not list((sv.queue / "pending").glob("*.json"))
    assert sv.finished() == "DRAINED_WITH_FAILURES"


def test_tranche_pack_once_and_all_done(tmp_path):
    sv, ops = make(tmp_path, [job(0, "T1"), job(1, "T1"), job(2, "T2")])
    sv.start()
    finish(sv, "j000")
    sv.cycle()
    assert ops.packed == []                                # T1 not finished
    finish(sv, "j001")
    sv.cycle()
    sv.cycle()
    assert ops.packed == ["T1"] and (sv.chain / "PACKED_T1").exists()
    finish(sv, "j002")
    assert sv.cycle() == "ALL_DONE" and (sv.chain / "ALL_DONE").exists()
    assert ops.packed == ["T1", "T2"]


def test_guard_halts_and_workers_are_not_restarted(tmp_path):
    jobs = [job(i, "T1") for i in range(2)] + [job(i, "T2", budget=200) for i in range(2, 400)]
    sv, ops = make(tmp_path, jobs, guard=5.0)
    sv.start()
    sv.top_up_workers()
    finish(sv, "j000", wall=3600.0)                        # measured: an hour per T1 run
    finish(sv, "j001", wall=3600.0)
    sv.cycle()
    assert (sv.chain / "HALT").exists() and "budget guard" in (sv.chain / "HALT").read_text()
    n = ops.launched
    ops.alive = 0
    sv.cycle()
    assert ops.launched == n                               # nothing restarts after HALT


def test_guard_does_not_halt_when_projection_is_inside(tmp_path):
    jobs = [job(i, "T1") for i in range(2)] + [job(2, "T2", budget=200)]
    sv, _ops = make(tmp_path, jobs, guard=50.0)
    sv.start()
    finish(sv, "j000", wall=900.0)
    finish(sv, "j001", wall=900.0)
    sv.cycle()
    assert not (sv.chain / "HALT").exists() and sv.state["projections"][-1]["projected_h"] < 50


def test_projection_uses_measured_class_means_and_prior_for_the_rest():
    done = [{"spec": {"model": "S", "budget": 654, "split": "outer_r0"}, "wall_time_s": 3600.0}]
    rem = [{"spec": {"model": "S", "budget": 654, "split": "outer_r1"}}] * 4
    p = sup.project(done, rem, 1.0, 4)
    assert abs(p["remaining_h"] - 1.0) < 1e-9 and abs(p["projected_h"] - 2.0) < 1e-9
    other = [{"spec": {"model": "D-state", "budget": 25, "split": "outer_r1"}}]
    p2 = sup.project(done, other, 0.0, 4)
    assert p2["remaining_h"] > 0 and p2["calibration"] > 0


def test_verified_existing_results_are_skipped(tmp_path, monkeypatch):
    sv, _ops = make(tmp_path, [job(0), job(1)])
    ck = tmp_path / "x.pt"
    ck.write_bytes(b"ckpt")
    pred = tmp_path / "x.parquet"
    pred.write_bytes(b"pred")
    res = tmp_path / "results" / "j000.json"
    res.parent.mkdir()
    res.write_text(json.dumps({"payload": {"run_id": "j000", "predictions": {"path": str(pred), "sha256": sup.sha256(pred)},
                                           "checkpoint": {"path": str(ck), "sha256": sup.sha256(ck)}}}))
    monkeypatch.setattr(sup, "CHECKOUT", tmp_path)
    sv.start()
    assert (sv.queue / "done" / "j000.json").exists() and (sv.queue / "pending" / "j001.json").exists()
    ck.write_bytes(b"corrupt")
    assert not sv.verified(job(0))


STUB = textwrap.dedent('''
    import sys, time, os
    sys.path.insert(0, {src!r})
    from pathlib import Path
    from dftgnn import jobqueue as Q
    root = Path({queue!r}); halt = Path({halt!r})
    Q.run_worker(root, f"stub-{{os.getpid()}}-w0", lambda j: (time.sleep({dur}), {{}})[1], poll=0.2,
                 log=lambda *_: None, should_stop=halt.exists)
''')


def test_real_process_drill_kill_a_worker_and_finish(tmp_path):
    """Real subprocess workers: kill one, the supervisor starts another, every job ends in done/."""
    queue, chain = tmp_path / "jobs", tmp_path / "chain"
    stub = tmp_path / "stub_worker.py"
    stub.write_text(STUB.format(src=str(ROOT / "src"), queue=str(queue), halt=str(chain / "HALT"), dur=1.5))
    jobs = [job(i, "T1" if i < 12 else "T2") for i in range(24)]
    a = sup.parse(["--chain", str(chain), "--queue", str(queue), "--workers", "4", "--poll", "0.3",
                   "--orphan-s", "2", "--python", sys.executable, "--marker", str(stub)])
    a.chain = Path(a.chain)

    class Real(sup.Ops):
        def launch_worker(self, i):
            with open(chain / "logs" / f"w{time.time_ns()}.log", "ab") as lf:
                subprocess.Popen([sys.executable, str(stub)], stdout=lf, stderr=subprocess.STDOUT,
                                 start_new_session=True)

        def enqueue(self):
            for j in jobs:
                Q.enqueue(queue, j)
            return [j["run_id"] for j in jobs]

        def pack(self, tranche):
            return f"outbox_{tranche}"

    ops = Real(a)
    s = sup.Supervisor(a, ops)
    s.start()
    killed = False
    end = None
    t0 = time.time()
    while time.time() - t0 < 90:
        end = s.cycle()
        if end:
            break
        if not killed and len(list((queue / "running").glob("*.json"))) >= 2:
            out = subprocess.run(["ps", "-ewwo", "pid,args"], capture_output=True, text=True,
                                 check=False).stdout
            pid = int(next(ln for ln in out.splitlines() if str(stub) in ln).split()[0])
            os.kill(pid, signal.SIGKILL)
            killed = True
        time.sleep(0.3)
    assert killed and end == "ALL_DONE"
    assert len(list((queue / "done").glob("*.json"))) == 24 and (chain / "PACKED_T1").exists()
    for _ in range(50):                                   # the surviving workers exit once the queue is empty
        if ops.workers_alive() == 0:
            break
        time.sleep(0.2)
    assert ops.workers_alive() == 0
