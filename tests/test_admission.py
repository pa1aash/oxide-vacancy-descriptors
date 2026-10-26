"""VRAM-aware admission, OOM requeue and solo exclusivity of the job queue.

CPU only: the device is simulated by passing ``vram_gb`` to the queue, and out-of-memory errors are raised
by the fake ``execute`` function.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import time

import pytest

from dftgnn import jobqueue as Q
from dftgnn.train import admission

VRAM = 10.0               # simulated device; the admissible budget is 0.85 x 10 = 8.5 GiB


class SimulatedOOM(RuntimeError):
    pass


def _job(rid, est=3.0, **kw):
    j = {"run_id": rid, "stage": "t", "spec": {"model": "S"}, "after": [], "est_peak_gb": est}
    j.update(kw)
    return j


def _running(root):
    return sorted(p.stem for p in (root / "running").glob("*.json"))


def _finish(root, job):
    Q.finish(root, job, ok=True, info={})


def test_admits_up_to_85_percent_of_vram(tmp_path):
    root = Q.init(tmp_path)
    for i in range(4):
        Q.enqueue(root, _job(f"j{i}", est=3.0))
    a = Q.claim(root, "w1", vram_gb=VRAM)
    b = Q.claim(root, "w2", vram_gb=VRAM)
    assert a and b                                        # 3 + 3 = 6 <= 8.5
    assert Q.claim(root, "w3", vram_gb=VRAM) is None      # 9 > 8.5
    _finish(root, a)
    assert Q.claim(root, "w3", vram_gb=VRAM) is not None  # 3 + 3 = 6 again
    assert len(_running(root)) == 2


def test_boundary_is_inclusive_and_sizes_mix(tmp_path):
    root = Q.init(tmp_path)
    Q.enqueue(root, _job("a", est=4.25))
    Q.enqueue(root, _job("b", est=4.25))                  # 8.5 exactly: admitted
    Q.enqueue(root, _job("c", est=0.1))                   # 8.6 > 8.5: refused
    assert Q.claim(root, "w1", vram_gb=VRAM) and Q.claim(root, "w2", vram_gb=VRAM)
    assert Q.claim(root, "w3", vram_gb=VRAM) is None


def test_a_small_job_is_admitted_past_a_large_one_that_does_not_fit(tmp_path):
    root = Q.init(tmp_path)
    Q.enqueue(root, _job("a", est=6.0))
    Q.enqueue(root, _job("b", est=6.0))                   # does not fit next to a
    Q.enqueue(root, _job("c", est=2.0))                   # fits
    assert Q.claim(root, "w1", vram_gb=VRAM)["run_id"] == "a"
    assert Q.claim(root, "w2", vram_gb=VRAM)["run_id"] == "c"


def test_missing_estimate_counts_as_the_whole_budget(tmp_path):
    root = Q.init(tmp_path)
    j = _job("a")
    del j["est_peak_gb"]
    Q.enqueue(root, j)
    Q.enqueue(root, _job("b", est=0.5))
    assert Q.claim(root, "w1", vram_gb=VRAM)["run_id"] == "a"
    assert Q.claim(root, "w2", vram_gb=VRAM) is None


def test_a_job_larger_than_the_budget_runs_only_on_an_idle_device(tmp_path):
    root = Q.init(tmp_path)
    Q.enqueue(root, _job("big", est=9.0))
    Q.enqueue(root, _job("small", est=1.0))
    assert Q.claim(root, "w1", vram_gb=VRAM)["run_id"] == "big"   # idle device: it can never fit, run it
    assert Q.claim(root, "w2", vram_gb=VRAM) is None


def test_without_vram_there_is_no_admission_control(tmp_path):
    root = Q.init(tmp_path)
    for i in range(5):
        Q.enqueue(root, _job(f"j{i}", est=99.0))
    assert all(Q.claim(root, f"w{i}") for i in range(5))


def _racer(root, name, vram, out):
    peak = 0
    while (job := Q.claim(root, name, vram_gb=vram)) is not None:
        peak = max(peak, len(list((root / "running").glob("*.json"))))
        time.sleep(0.05)
        Q.finish(root, job, ok=True, info={})
    out.put(peak)


def test_concurrent_workers_never_exceed_the_budget(tmp_path):
    root = Q.init(tmp_path)
    for i in range(24):
        Q.enqueue(root, _job(f"j{i:02d}", est=3.0))
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    ps = [ctx.Process(target=_racer, args=(root, f"w{k}", VRAM, out)) for k in range(6)]
    for p in ps:
        p.start()
    peaks = [out.get(timeout=60) for _ in ps]
    for p in ps:
        p.join()
    assert max(peaks) <= 2                                # floor(8.5 / 3)
    assert len(list((root / "done").glob("*.json"))) == 24


def test_oom_requeues_the_job_as_solo(tmp_path):
    root = Q.init(tmp_path)
    Q.enqueue(root, _job("a"))
    calls = []

    def execute(job):
        calls.append((job["run_id"], bool(job.get("needs_solo"))))
        if len(calls) == 1:
            raise SimulatedOOM("CUDA out of memory. Tried to allocate 2.00 GiB")
        return {"status": "done"}

    freed = []
    n = Q.run_worker(root, "w", execute, poll=0.01, log=lambda *_: None, vram_gb=VRAM,
                     on_oom=lambda: freed.append(1))
    assert calls == [("a", False), ("a", True)] and freed == [1] and n == 1
    done = json.loads((root / "done" / "a.json").read_text())
    assert done["needs_solo"] is True and done["oom_requeues"] == 1
    assert not list((root / "failed").glob("*.json"))


def test_oom_of_a_solo_job_is_a_failure(tmp_path):
    root = Q.init(tmp_path)
    Q.enqueue(root, _job("a", needs_solo=True))

    def execute(job):
        raise SimulatedOOM("CUDA out of memory")

    Q.run_worker(root, "w", execute, poll=0.01, log=lambda *_: None, vram_gb=VRAM)
    assert (root / "failed" / "a.json").exists() and not list((root / "pending").glob("*.json"))


def test_a_genuine_error_is_not_requeued(tmp_path):
    root = Q.init(tmp_path)
    Q.enqueue(root, _job("a"))

    def execute(job):
        raise ValueError("bug")

    Q.run_worker(root, "w", execute, poll=0.01, log=lambda *_: None, vram_gb=VRAM)
    assert (root / "failed" / "a.json").exists()


def test_requeue_clears_the_running_state(tmp_path):
    root = Q.init(tmp_path)
    Q.enqueue(root, _job("a"))
    job = Q.claim(root, "w", vram_gb=VRAM)
    assert Q.requeue_solo(root, job)
    assert not _running(root) and not (root / "running" / "a.hb").exists()
    assert json.loads((root / "pending" / "a.json").read_text())["needs_solo"] is True


def test_solo_job_waits_for_an_idle_device_and_blocks_others(tmp_path):
    root = Q.init(tmp_path)
    Q.enqueue(root, _job("a", est=2.0))
    a = Q.claim(root, "w1", vram_gb=VRAM)
    assert a["run_id"] == "a"
    Q.enqueue(root, _job("s", est=2.0, needs_solo=True))      # ready solo job, e.g. requeued after an OOM
    Q.enqueue(root, _job("b", est=2.0))
    assert Q.claim(root, "w2", vram_gb=VRAM) is None      # s needs an idle device; b is held back (drain)
    _finish(root, a)
    s = Q.claim(root, "w2", vram_gb=VRAM)
    assert s["run_id"] == "s"
    assert Q.claim(root, "w3", vram_gb=VRAM) is None      # nothing runs next to a solo job
    assert _running(root) == ["s"]
    _finish(root, s)
    assert Q.claim(root, "w3", vram_gb=VRAM)["run_id"] == "b"


def test_solo_exclusivity_with_racing_workers(tmp_path):
    root = Q.init(tmp_path)
    for i in range(12):
        Q.enqueue(root, _job(f"j{i:02d}", est=2.0, needs_solo=(i % 4 == 0)))
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    ps = [ctx.Process(target=_solo_racer, args=(root, f"w{k}", VRAM, out)) for k in range(5)]
    for p in ps:
        p.start()
    seen = [x for _ in ps for x in out.get(timeout=60)]
    for p in ps:
        p.join()
    solos = [s for s in seen if s[1]]
    assert len(solos) == 3 and all(n == 1 for _, _, n in solos)
    assert len(list((root / "done").glob("*.json"))) == 12


def _solo_racer(root, name, vram, out):
    seen = []
    while (job := Q.claim(root, name, vram_gb=vram)) is not None:
        n = len(list((root / "running").glob("*.json")))
        seen.append((job["run_id"], bool(job.get("needs_solo")), n))
        time.sleep(0.05)
        Q.finish(root, job, ok=True, info={})
    out.put(seen)


def test_estimate_lookup_and_ram_cap():
    gib = admission.GIB
    table = {"peak_bytes": {(64, 32, 3): 4 * gib, (128, 64, 4): 13 * gib}, "host_peak_rss_bytes": 3 * gib,
             "total_vram_bytes": 44 * gib}
    assert admission.est_peak_gb(table, 64, 32, 3) == pytest.approx(5.0)       # 4 x 1.25
    assert admission.est_peak_for_spec(table, {"hidden_width": 128, "megnet_blocks": 4}, 64) == pytest.approx(16.25)
    with pytest.raises(KeyError):
        admission.est_peak_gb(table, 64, 16, 3)
    assert admission.est_host_gb(table) == pytest.approx(3.75)
    assert admission.max_workers_by_ram(3.75) == 13        # floor(0.8 x 62 / 3.75)
    assert admission.max_workers_by_ram(100.0) == 1


def test_stage_jobs_carry_est_peak_gb():
    from dftgnn.models import HParams
    from dftgnn.train import RunSpec
    from dftgnn.train import stages as ST

    gib = admission.GIB
    table = {"peak_bytes": {(64, 32, 3): 4 * gib}, "host_peak_rss_bytes": gib, "total_vram_bytes": 44 * gib}
    spec = RunSpec(model="S", hp=HParams().to_dict(), lr=1e-3, weight_decay=1e-5, batch_size=32,
                   split="outer_r0", r=0, budget=25, seed=0)
    job = ST._job(spec, "t", "c" * 40, "g" * 64, table=table)
    assert job["est_peak_gb"] == pytest.approx(5.0)
    assert "est_peak_gb" not in ST._job(spec, "t", "c" * 40, "g" * 64)
