"""File-based, resumable job queue for training runs (no database, no git needed on the pod).

    jobs/pending/<run_id>.json   waiting
    jobs/running/<run_id>.json   claimed; jobs/running/<run_id>.hb is its heartbeat (mtime)
    jobs/done/<run_id>.json      finished (adds status, result path, wall time)
    jobs/failed/<run_id>.json    raised (adds the traceback)

A claim is ``os.rename`` from pending/ to running/. On one filesystem it is atomic, so exactly one
worker wins each job. A running job whose heartbeat is older than ``STALE_S`` (10 min) returns to
pending. A job may carry an integer ``priority`` (lower runs first; absent means 0, i.e. file-name order). A job may list ``after`` run ids (e.g. P after its P1 and D runs) and is claimable only when
all of them are in done/.

VRAM-aware admission. A job may carry ``est_peak_gb`` (the benchmark's realistic peak times a 1.25 safety
factor, ``dftgnn.train.admission``). With ``vram_gb`` given, ``claim`` admits a job only if the sum of
``est_peak_gb`` over the running jobs plus the new one is at most ``ADMIT_FRACTION`` (0.85) of ``vram_gb``.
The check and the claim run under an exclusive ``fcntl`` lock on ``<root>/.admit.lock``, so concurrent workers
cannot over-admit. A job whose estimate is missing is treated as needing the whole admissible budget. An
out-of-memory error requeues the job with ``needs_solo = true``; a solo job is admitted only when nothing else
is running, no other job is admitted while a solo job runs, and while a solo job is ready and pending no other
job is admitted either (so it cannot starve).
"""
from __future__ import annotations

import fcntl
import json
import os
import threading
import time
import traceback
from pathlib import Path

STATES = ("pending", "running", "done", "failed")
ADMIT_FRACTION = 0.85
STALE_S = 600
HEARTBEAT_S = 30


def init(root: Path) -> Path:
    for s in STATES:
        (root / s).mkdir(parents=True, exist_ok=True)
    return root


def enqueue(root: Path, job: dict) -> bool:
    """Add a job unless its run id is already known in any state. Returns True if added."""
    init(root)
    name = f"{job['run_id']}.json"
    if any((root / s / name).exists() for s in STATES):
        return False
    tmp = root / "pending" / f".{name}.tmp"
    tmp.write_text(json.dumps(job, indent=1, sort_keys=True))
    os.replace(tmp, root / "pending" / name)
    return True


def _ready(root: Path, job: dict) -> bool:
    return all((root / "done" / f"{a}.json").exists() for a in job.get("after", []))


def _age(hb: Path, job: Path, now: float) -> float | None:
    """Seconds since the heartbeat (or, without one, the claim) was last touched; None if the job is gone.

    A worker that finishes a job unlinks its running/ files while other workers and the supervisor scan them, so
    either file may vanish between the listing and the ``stat``.
    """
    for p in (hb, job):
        try:
            return now - p.stat().st_mtime
        except FileNotFoundError:
            continue
    return None


def requeue_stale(root: Path, max_age: float = STALE_S, now: float | None = None) -> list[str]:
    now = time.time() if now is None else now
    back = []
    for f in sorted((root / "running").glob("*.json")):
        hb = f.with_suffix(".hb")
        age = _age(hb, f, now)
        if age is not None and age > max_age:
            try:
                os.rename(f, root / "pending" / f.name)
            except FileNotFoundError:
                continue                       # finished or requeued by someone else meanwhile
            hb.unlink(missing_ok=True)
            back.append(f.stem)
    return back


def is_oom(exc: BaseException) -> bool:
    """CUDA out-of-memory (``torch.cuda.OutOfMemoryError`` or a RuntimeError saying so)."""
    return type(exc).__name__ in ("OutOfMemoryError", "SimulatedOOM") or "out of memory" in str(exc).lower()


def _running_jobs(root: Path) -> list[dict]:
    out = []
    for f in (root / "running").glob("*.json"):
        try:
            out.append(json.loads(f.read_text()))
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    return out


def _need_gb(job: dict, budget: float) -> float:
    est = job.get("est_peak_gb")
    return budget if est is None else float(est)


def claim(root: Path, worker: str, *, vram_gb: float | None = None) -> dict | None:
    """Atomically move one ready, admissible pending job to running/ and start its heartbeat file.

    ``vram_gb`` (total device memory) switches on admission control; see the module docstring.
    """
    if vram_gb is None:
        return _claim(root, worker, None)
    init(root)
    with open(root / ".admit.lock", "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            return _claim(root, worker, vram_gb)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _claim(root: Path, worker: str, vram_gb: float | None) -> dict | None:
    budget = None if vram_gb is None else ADMIT_FRACTION * vram_gb
    running = [] if vram_gb is None else _running_jobs(root)
    if vram_gb is not None and any(j.get("needs_solo") for j in running):
        return None
    used = sum(_need_gb(j, budget) for j in running) if budget is not None else 0.0
    ready = []
    for f in sorted((root / "pending").glob("*.json")):
        try:
            job = json.loads(f.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if _ready(root, job):
            ready.append((f, job))
    if budget is not None and running and any(j.get("needs_solo") for _, j in ready):
        return None                            # drain: let the running jobs finish, admit nothing else
    # a ready solo job goes first, then the lowest ``priority`` (default 0: file-name order, stable)
    ready.sort(key=lambda fj: (not fj[1].get("needs_solo") if budget is not None else False,
                               fj[1].get("priority", 0)))
    for f, job in ready:
        # (a job that alone exceeds the budget runs only on an idle device)
        if (budget is not None and not job.get("needs_solo") and running
                and used + _need_gb(job, budget) > budget + 1e-9):
            continue
        dst = root / "running" / f.name
        try:
            os.rename(f, dst)
        except FileNotFoundError:
            continue                           # another worker won this one
        dst.with_suffix(".hb").write_text(worker)
        job["_claimed_by"] = worker
        return job
    return None


def requeue_solo(root: Path, job: dict) -> bool:
    """Return a job that hit an out-of-memory error to pending/ with ``needs_solo = true``.

    False if it already was a solo job (it failed with the whole device to itself): the caller records it
    as failed.
    """
    if job.get("needs_solo"):
        return False
    name = f"{job['run_id']}.json"
    rec = {k: v for k, v in job.items() if not k.startswith("_")}
    rec["needs_solo"] = True
    rec["oom_requeues"] = int(rec.get("oom_requeues", 0)) + 1
    tmp = root / "pending" / f".{name}.tmp"
    tmp.write_text(json.dumps(rec, indent=1, sort_keys=True))
    os.replace(tmp, root / "pending" / name)
    (root / "running" / name).unlink(missing_ok=True)
    (root / "running" / name).with_suffix(".hb").unlink(missing_ok=True)
    return True


def finish(root: Path, job: dict, *, ok: bool, info: dict) -> None:
    name = f"{job['run_id']}.json"
    src = root / "running" / name
    rec = {**job, **info, "finished_at": time.time()}
    dst = root / ("done" if ok else "failed") / name
    tmp = dst.with_name(f".{name}.tmp")
    tmp.write_text(json.dumps(rec, indent=1, sort_keys=True, default=str))
    os.replace(tmp, dst)
    src.unlink(missing_ok=True)
    src.with_suffix(".hb").unlink(missing_ok=True)
    if ok:
        (root / "pending" / name).unlink(missing_ok=True)   # a stale requeue of a job that did finish


class Heartbeat:
    """Touch ``running/<run_id>.hb`` every ``every`` seconds while a job runs."""

    def __init__(self, root: Path, run_id: str, every: float = HEARTBEAT_S):
        self.path = root / "running" / f"{run_id}.hb"
        self.every = every
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.wait(self.every):
            try:
                os.utime(self.path)
            except FileNotFoundError:
                return

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


def run_worker(root: Path, worker: str, execute, *, idle_exit: bool = True, poll: float = 5.0,
               log=print, vram_gb: float | None = None, on_oom=None, should_stop=None) -> int:
    """Claim and execute jobs until none is left. ``execute(job) -> info dict``. Returns jobs run.

    ``vram_gb`` turns on admission control. A CUDA out-of-memory error requeues the job as a solo job
    (``on_oom`` is called first, e.g. to empty the CUDA cache). ``should_stop()`` is checked before every
    claim (a HALT file): a True ends the worker once its current job is done.
    """
    n = 0
    while True:
        if should_stop is not None and should_stop():
            log(f"{worker}: stop requested; exiting")
            return n
        requeue_stale(root)
        job = claim(root, worker, vram_gb=vram_gb)
        if job is None:
            pend = list((root / "pending").glob("*.json"))
            run = list((root / "running").glob("*.json"))
            if idle_exit and not run and not pend:
                return n
            if idle_exit and pend and not run:
                log(f"{worker}: {len(pend)} pending jobs have unmet dependencies; exiting")
                return n
            time.sleep(poll)
            continue
        t0 = time.time()
        log(f"{worker}: claimed {job['run_id']} ({job['spec']['model']})")
        try:
            with Heartbeat(root, job["run_id"]):
                info = execute(job)
            finish(root, job, ok=True, info={"wall_time_s": time.time() - t0, **info})
        except Exception as exc:  # noqa: BLE001 - every failure is recorded with its traceback
            if is_oom(exc):
                if on_oom is not None:
                    on_oom()
                if requeue_solo(root, job):
                    log(f"{worker}: {job['run_id']} ran out of memory; requeued as needs_solo")
                    continue
            finish(root, job, ok=False, info={"wall_time_s": time.time() - t0,
                                              "traceback": traceback.format_exc()})
        n += 1


def status(root: Path) -> dict:
    init(root)
    counts = {s: len(list((root / s).glob("*.json"))) for s in STATES}
    done = [json.loads(f.read_text()) for f in (root / "done").glob("*.json")]
    failed = [json.loads(f.read_text()) for f in (root / "failed").glob("*.json")]
    walls = [d["wall_time_s"] for d in done if "wall_time_s" in d]
    mean = sum(walls) / len(walls) if walls else None
    conc = max(counts["running"], 1)
    eta = None if mean is None else mean * (counts["pending"] + counts["running"]) / conc
    return {"counts": counts, "mean_wall_time_s": mean, "eta_s": eta, "concurrency_assumed": conc,
            "failures": [{"run_id": f["run_id"], "model": f["spec"]["model"],
                          "traceback": f.get("traceback", "")} for f in failed]}
