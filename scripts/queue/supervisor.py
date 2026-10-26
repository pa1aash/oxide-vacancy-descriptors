"""nohup /workspace/miniforge3/envs/dftgnn-gpu/bin/python /workspace/chain/supervisor.py \
        >> /workspace/chain/supervisor.log 2>&1 &

Operational only: it changes when jobs start, never how they run.

* Workers. Keeps ``--workers`` (4) worker processes alive: each is ``scripts/queue/worker.py --max-concurrent 1``
  (VRAM-aware admission is in the queue), found by scanning ``ps``, so a restarted supervisor never launches a
  second set. A dead worker is replaced; a running job whose heartbeat is older than ``--orphan-s`` is returned
  to pending. A failed job is requeued once (reported in the log); a second failure stays in failed/.
* Tranches. Jobs carry ``tranche`` T1-T4. When no job of a tranche is pending or running its done runs are
  packed once into the checkout's outbox with ``outbox.py pack --tranche`` (a snapshot of finished runs) and
  ``<chain>/PACKED_<tranche>`` records the tarball.
* Budget guard. At every tranche boundary (and whenever the elapsed time alone exceeds it) the total GPU-hours
  are projected: elapsed wall time since the first launch (one GPU, billed while it runs) plus the remaining
  work divided by the worker count. Remaining work uses, per job class (model, budget, split, epoch cap), the
  mean measured wall time, or for a class with no finished job the cost-table prior (seconds per epoch x the
  budget's epoch-time factor x epoch cap x 4 / 1.23 contention) scaled by the measured-to-prior ratio of the
  classes that have finished. If the projection exceeds ``--guard`` the supervisor writes ``<chain>/HALT`` with
  the reason; workers finish their current job and exit, and nothing new starts.
* ALL_DONE is written when every job in the manifest is in done/. The supervisor never stops, kills or
  deletes anything on the pod.
* (``--stage s12 --tranches I,L,M,C,X``). The non-training task stages (relax, embed, geomeval) are queue
  jobs like any other: one process each, admitted by ``est_peak_gb`` (an out-of-memory error requeues them as
  hosts). Their results verify through the artefact hashes in the payload. If nothing is running and every
  pending job waits on a dependency that is in failed/ after its single retry, DRAINED_WITH_FAILURES is written.
  The budget guard's priors for these stages come from the CPU smoke timings (``TASK_PRIOR_S``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

CHECKOUT = Path("/workspace/dft-gnn")
CHAIN = Path("/workspace/chain")
PY = "/workspace/miniforge3/envs/dftgnn-gpu/bin/python"
# cost-table epoch-time factors per budget and the measured baseline
EPOCH_FACTOR = {25: 0.049, 50: 0.079, 100: 0.213, 200: 0.361, 400: 0.665, 654: 1.073, "kiyohara": 1.130}
SEC_PER_EPOCH_UPPER = 2.44
CONTENTION = 4 / 1.23
GUARD_GPU_H = 59.4      # sweep + Kiyohara upper bound (55.4) + epoch-cap estimate (4)
TRANCHES = ("T1", "T2", "T3", "T4")
TASK_STAGES = ("relax", "embed", "geomeval", "p1tune_v2", "dtune_v2")
TASK_SUBDIR = {"relax": "mlip_v1", "embed": "embeddings_v1", "geomeval": "geomeval",
               "p1tune_v2": "p1tune_v2", "dtune_v2": "dtune_v2"}                          # dftgnn.tasks.SUBDIR
# CPU smoke (3 threads), seconds per task; relax is 30 FIRE steps x (0.068 + 0.0111 n_atoms) s; p1tune_v2 is
# one 30-trial study, twice the summed trial time of the 200-epoch P1 study at anchor 654 (results/tuning);
# dtune_v2 likewise from the v2 D-state study at anchor 654 (46,152 s)
TASK_PRIOR_S = {"embed": 60.0, "geomeval": 180.0, "peval": 45.0, "p1tune_v2": 75000.0, "dtune_v2": 92000.0}


def log(msg: str) -> None:
    print(f"{datetime.now(UTC).isoformat(timespec='seconds')} {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def job_class(job: dict) -> tuple:
    sp = job["spec"]
    if job.get("stage") in TASK_STAGES:
        return (job["stage"], sp.get("n_atoms_uc", 0) if job["stage"] == "relax" else sp.get("budget"), 0)
    if sp["model"] == "P":
        return ("peval", sp["budget"], 0)
    b = "kiyohara" if sp["split"] == "kiyohara" else sp["budget"]
    kind = sp["model"] if sp["model"] in ("S", "P1", "cgcnn-S", "cgcnn-D") else "D"
    return (kind, b, sp.get("max_epochs") or 200)


def _factor(b) -> float:
    if b in EPOCH_FACTOR:
        return EPOCH_FACTOR[b]
    keys = sorted(k for k in EPOCH_FACTOR if isinstance(k, int))
    return EPOCH_FACTOR[min(keys, key=lambda k: abs(k - b))]     # LOCO folds train on 654-655 hosts


def prior_s(cls: tuple) -> float:
    """Prior wall time of one job of a class: cost-table upper-style for training runs under four-way contention,
    the smoke timings for the inference tasks."""
    kind, b, epochs = cls
    if kind == "relax":
        return 30 * (0.068 + 0.0111 * b)
    if kind in TASK_PRIOR_S:
        return TASK_PRIOR_S[kind]
    return SEC_PER_EPOCH_UPPER * _factor(b) * epochs * CONTENTION


def project(done: list[dict], remaining: list[dict], elapsed_h: float, workers: int) -> dict:
    """Projected total GPU-hours (elapsed + remaining worker-seconds / workers) from measured wall times."""
    walls = defaultdict(list)
    for j in done:
        if "wall_time_s" in j and j.get("status") != "verified_existing":
            walls[job_class(j)].append(j["wall_time_s"])
    ratios = [sum(w) / len(w) / prior_s(c) for c, w in walls.items()]
    calib = sum(ratios) / len(ratios) if ratios else 1.0
    rem_s = 0.0
    for j in remaining:
        c = job_class(j)
        rem_s += sum(walls[c]) / len(walls[c]) if walls.get(c) else prior_s(c) * calib
    return {"elapsed_h": elapsed_h, "remaining_h": rem_s / workers / 3600, "calibration": calib,
            "projected_h": elapsed_h + rem_s / workers / 3600, "measured_classes": len(walls)}


class Ops:
    """Everything that touches processes, the checkout or the clock; replaced by mocks in the tests."""

    def __init__(self, a):
        self.a = a

    def now(self) -> float:
        return time.time()

    def sleep(self, s: float) -> None:
        time.sleep(s)

    def workers_alive(self) -> int:
        out = subprocess.run(["ps", "-ewwo", "args"], capture_output=True, text=True, check=False).stdout
        n = 0
        for line in out.splitlines():
            parts = line.split()
            if len(parts) > 1 and os.path.basename(parts[0]).startswith("python") and self.a.marker in parts:
                n += 1
        return n

    def launch_worker(self, i: int) -> None:
        cmd = [self.a.python, str(CHECKOUT / "scripts/queue/worker.py"), "--max-concurrent", "1", "--device", "cuda",
               "--threads", str(self.a.threads), "--queue", str(self.a.queue), "--halt-file",
               str(self.a.chain / "HALT")]
        logf = open(self.a.chain / "logs" / f"worker_{int(time.time())}_{i}.log", "ab")
        subprocess.Popen(cmd, cwd=CHECKOUT, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)

    def enqueue(self) -> list[str]:
        manifest = self.a.chain / f"{self.a.stage}_jobs.json"
        extra = (["--specs", *self.a.specs] + (["--shard", self.a.shard] if self.a.shard else [])
                 if self.a.specs else [])
        subprocess.run([self.a.python, str(CHECKOUT / "scripts/queue/enqueue.py"),
                        "specs" if self.a.specs else self.a.stage, *extra, "--manifest", str(manifest),
                        "--queue", str(self.a.queue)], cwd=CHECKOUT, check=True)
        return json.loads(manifest.read_text())

    def pack(self, tranche: str) -> str:
        r = subprocess.run([self.a.python, str(CHECKOUT / "scripts/queue/outbox.py"), "pack", "--tranche", tranche,
                            "--queue", str(self.a.queue)], cwd=CHECKOUT, capture_output=True, text=True,
                           check=False)
        if r.returncode:
            raise RuntimeError(f"pack {tranche} failed: {r.stdout}{r.stderr}")
        return r.stdout.strip().splitlines()[-1]


class Supervisor:
    def __init__(self, a, ops: Ops):
        self.a, self.ops = a, ops
        self.queue, self.chain = Path(a.queue), Path(a.chain)
        for d in ("logs",):
            (self.chain / d).mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(CHECKOUT / "src"))
        from dftgnn import jobqueue as Q       # stdlib-only module

        self.Q = Q
        self.ids: list[str] = []
        self.state = self._load_state()

    # ---------------------------------------------------------------- state
    def _load_state(self) -> dict:
        f = self.chain / "state.json"
        if f.exists():
            return json.loads(f.read_text())
        return {"first_launch": None, "retried": [], "tranche_checked": [], "projections": []}

    def _save(self) -> None:
        tmp = self.chain / ".state.json.tmp"
        tmp.write_text(json.dumps(self.state, indent=1, sort_keys=True))
        os.replace(tmp, self.chain / "state.json")

    def _jobs(self, state: str) -> list[dict]:
        out = []
        for f in sorted((self.queue / state).glob("*.json")):
            try:
                out.append(json.loads(f.read_text()))
            except (FileNotFoundError, json.JSONDecodeError):
                continue
        return out

    # ---------------------------------------------------------------- steps
    def start(self) -> None:
        self.ids = self.ops.enqueue()
        log(f"enqueued/known {len(self.ids)} jobs in the manifest")
        n = self.skip_verified()
        if n:
            log(f"{n} pending jobs already had verified results and were moved to done")

    @staticmethod
    def result_file(job: dict) -> Path:
        sub = TASK_SUBDIR.get(job.get("stage")) or job["spec"].get("results_subdir")
        return CHECKOUT / "results" / (sub or "") / f"{job['run_id']}.json"

    def verified(self, job: dict) -> bool:
        res = self.result_file(job)
        try:
            pay = json.loads(res.read_text())["payload"]
            arts = [pay[k] for k in ("predictions", "checkpoint") if pay.get(k)] + list(pay.get("artifacts", []))
            return bool(arts) and pay["run_id"] == job["run_id"] and all(
                sha256(CHECKOUT / a["path"]) == a["sha256"] for a in arts)
        except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError):
            return False

    def skip_verified(self) -> int:
        n = 0
        for job in self._jobs("pending"):
            if self.verified(job):
                res = self.result_file(job)
                src = self.queue / "pending" / f"{job['run_id']}.json"
                rec = {**job, "status": "verified_existing", "result": str(res), "wall_time_s": 0.0,
                       "finished_at": self.ops.now()}
                dst = self.queue / "done" / src.name
                dst.write_text(json.dumps(rec, indent=1, sort_keys=True))
                src.unlink(missing_ok=True)
                n += 1
        return n

    def requeue_orphans(self) -> list[str]:
        back = self.Q.requeue_stale(self.queue, max_age=self.a.orphan_s, now=self.ops.now())
        if back:
            log(f"requeued orphaned jobs (heartbeat older than {self.a.orphan_s:.0f} s): {back}")
        return back

    def retry_failed(self) -> None:
        for job in self._jobs("failed"):
            rid = job["run_id"]
            if rid in self.state["retried"]:
                continue
            tb = (job.get("traceback") or "").strip().splitlines()
            log(f"FAILED {rid} ({job['spec']['model']}, B={job['spec']['budget']}, r={job['spec']['r']}, "
                f"seed={job['spec']['seed']}): {tb[-1] if tb else '?'}; requeueing once")
            rec = {k: v for k, v in job.items() if k not in ("traceback", "finished_at", "wall_time_s", "status")}
            rec["first_failure"] = tb[-5:]
            (self.queue / "pending" / f"{rid}.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
            (self.queue / "failed" / f"{rid}.json").unlink(missing_ok=True)
            self.state["retried"].append(rid)
            self._save()

    def top_up_workers(self) -> int:
        if (self.chain / "HALT").exists() or getattr(self.a, "no_launch", False):
            return 0
        pend, run = self._jobs("pending"), self._jobs("running")
        alive = self.ops.workers_alive()
        launched = 0
        # never more workers than jobs that could use them
        want = min(self.a.workers, len(pend) + len(run))
        while alive + launched < want:
            if self.state["first_launch"] is None:
                self.state["first_launch"] = self.ops.now()
                self._save()
            self.ops.launch_worker(alive + launched)
            launched += 1
        if launched:
            log(f"launched {launched} worker(s); {alive} were alive")
        return launched

    def elapsed_h(self) -> float:
        t0 = self.state["first_launch"]
        return 0.0 if t0 is None else (self.ops.now() - t0) / 3600

    def tranche_state(self) -> dict:
        pend, run, done, failed = (self._jobs(s) for s in ("pending", "running", "done", "failed"))
        out = {}
        for t in self.a.tranches:
            n = lambda js: sum(j.get("tranche") == t for j in js)      # noqa: E731
            out[t] = {"pending": n(pend), "running": n(run), "done": n(done), "failed": n(failed)}
        return out

    def check_guard(self, why: str) -> bool:
        pend, run, done = self._jobs("pending"), self._jobs("running"), self._jobs("done")
        proj = project(done, pend + run, self.elapsed_h(), self.a.workers)
        proj.update({"why": why, "t": self.ops.now()})
        self.state["projections"].append(proj)
        self._save()
        log(f"projection ({why}): elapsed {proj['elapsed_h']:.2f} GPU-h, remaining {proj['remaining_h']:.2f}, total "
            f"{proj['projected_h']:.2f} vs guard {self.a.guard:.1f} (calibration {proj['calibration']:.2f}, "
            f"{proj['measured_classes']} measured classes)")
        if proj["projected_h"] > self.a.guard:
            (self.chain / "HALT").write_text(
                f"budget guard: projected {proj['projected_h']:.2f} GPU-h > {self.a.guard:.1f} ({why})\n")
            log("HALT written: running jobs will finish, nothing new starts")
            return True
        return False

    def handle_tranches(self) -> None:
        ts = self.tranche_state()
        for t in self.a.tranches:
            c = ts[t]
            if t in self.state["tranche_checked"] or c["pending"] or c["running"] or not (c["done"] or c["failed"]):
                continue
            tar = self.ops.pack(t)
            (self.chain / f"PACKED_{t}").write_text(f"{tar}\ndone {c['done']} failed {c['failed']}\n")
            log(f"tranche {t} finished (done {c['done']}, failed {c['failed']}); packed {tar}")
            self.state["tranche_checked"].append(t)
            self._save()
            self.check_guard(f"after {t}")

    def finished(self) -> str | None:
        if not self.ids:
            return None
        done = {f.stem for f in (self.queue / "done").glob("*.json")}
        if all(i in done for i in self.ids):
            (self.chain / "ALL_DONE").write_text(f"{len(done)} jobs done at {datetime.now(UTC).isoformat()}\n")
            return "ALL_DONE"
        c = {s: len(list((self.queue / s).glob("*.json"))) for s in ("pending", "running", "failed")}
        if (self.chain / "HALT").exists() and not c["running"]:
            return "HALTED"
        if not c["pending"] and not c["running"] and c["failed"]:
            (self.chain / "DRAINED_WITH_FAILURES").write_text(f"failed {c['failed']}\n")
            return "DRAINED_WITH_FAILURES"
        if c["pending"] and not c["running"] and self.blocked_by_failures():
            (self.chain / "DRAINED_WITH_FAILURES").write_text(
                f"failed {c['failed']}; {c['pending']} pending jobs wait on failed dependencies\n")
            return "DRAINED_WITH_FAILURES"
        return None

    def blocked_by_failures(self) -> bool:
        """True if no pending job is ready and every pending job waits on a dependency that failed after its
        retry (in failed/ and already retried once)."""
        done = {f.stem for f in (self.queue / "done").glob("*.json")}
        failed = {f.stem for f in (self.queue / "failed").glob("*.json")} & set(self.state["retried"])
        pend = self._jobs("pending")
        if not pend:
            return False
        for j in pend:
            missing = [a for a in j.get("after", []) if a not in done]
            if not missing or not any(a in failed for a in missing):
                return False
        return True

    def cycle(self) -> str | None:
        self.requeue_orphans()
        self.retry_failed()
        self.top_up_workers()
        self.handle_tranches()
        if self.elapsed_h() > self.a.guard and not (self.chain / "HALT").exists():
            self.check_guard("elapsed alone exceeds the guard")
        return self.finished()

    def run(self) -> str:
        self.start()
        while True:
            end = self.cycle()
            if end:
                log(f"supervisor ending: {end}")
                return end
            self.ops.sleep(self.a.poll)


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", default=str(CHAIN))
    ap.add_argument("--queue", default=None, help="default <checkout>/jobs")
    ap.add_argument("--python", default=PY)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--poll", type=float, default=60.0)
    ap.add_argument("--orphan-s", type=float, default=180.0)
    ap.add_argument("--guard", type=float, default=GUARD_GPU_H)
    ap.add_argument("--marker", default=None, help="default <checkout>/scripts/queue/worker.py")
    ap.add_argument("--stage", default="session", help="enqueue.py stage to build (S09: session, S12: s12)")
    ap.add_argument("--tranches", default=",".join(TRANCHES), help="tranche labels to pack, in order")
    ap.add_argument("--checkout", default=str(CHECKOUT), help="the git checkout the workers run from")
    ap.add_argument("--specs", nargs="+", help="cluster spec files (enqueue.py specs mode) instead of --stage")
    ap.add_argument("--shard", help="with --specs: I/N")
    ap.add_argument("--no-launch", action="store_true",
                    help="launch no workers (they run elsewhere on the shared queue); --workers sizes the projection")
    a = ap.parse_args(argv)
    a.tranches = tuple(t for t in a.tranches.split(",") if t)
    a.queue = a.queue or str(Path(a.checkout) / "jobs")
    a.marker = a.marker or str(Path(a.checkout) / "scripts/queue/worker.py")
    return a


def main() -> None:
    global CHECKOUT
    a = parse()
    CHECKOUT = Path(a.checkout)
    a.chain = Path(a.chain)
    a.chain.mkdir(parents=True, exist_ok=True)
    pidf = a.chain / "supervisor.pid"
    if pidf.exists():
        try:
            os.kill(int(pidf.read_text()), 0)
            raise SystemExit(f"supervisor already running (pid {pidf.read_text().strip()})")
        except (ProcessLookupError, ValueError):
            pass
    pidf.write_text(str(os.getpid()))
    Supervisor(a, Ops(a)).run()


if __name__ == "__main__":
    main()
