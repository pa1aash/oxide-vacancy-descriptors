"""Queue worker: N processes, each claiming jobs atomically until the queue is drained.

Training and P-evaluation jobs run through ``dftgnn.train.runner.execute``; the task stages (relax, embed,
geomeval) through ``dftgnn.tasks.TaskRunner`` in the same process (one task at a time).

    python scripts/queue/worker.py --max-concurrent 8 --device cuda

Launch one invocation per GPU (CUDA_VISIBLE_DEVICES selects it). A job carrying ``code_ref`` (cluster specs from
``scripts/queue/make_specs.py``) runs only if that reference, or its bug-fix tag (nrp-code-v2 for nrp-code-v1), matches the checked-out code. Torch threads are split evenly between the
processes over the CPUs the container may use (cgroup quota, not the host core count).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import socket
from pathlib import Path

from _common import QUEUE


def child(queue: str, name: str, device: str, threads: int, vram_gb: float | None,
          halt_file: str | None = None) -> None:
    import torch

    from dftgnn import jobqueue as Q
    from dftgnn.tasks import TASK_STAGES, TaskRunner
    from dftgnn.train import RunSpec, Store, check_code_ref
    from dftgnn.train.runner import execute

    torch.set_num_threads(threads)
    dev = torch.device(device)
    if dev.type == "mps":
        raise SystemExit("MPS is not used for training")
    data = Store()
    tasks = TaskRunner(data, dev)          # relax / embed / geomeval; state built on first use

    def free() -> None:
        if dev.type == "cuda":
            import gc

            gc.collect()
            torch.cuda.empty_cache()

    def run(job: dict) -> dict:
        if job.get("code_ref"):
            check_code_ref(job["code_ref"])       # pinned cluster specs run only on the code they name
        if job.get("stage") in TASK_STAGES:
            try:
                info = tasks(job)
            finally:
                free()
            return {**info, "device": str(dev), "torch_threads": threads}
        try:
            out = execute(RunSpec.from_dict(job["spec"]), data, device=dev, log=lambda *_: None)
        finally:
            free()      # the caching allocator otherwise keeps ~2x the live peak between jobs
        if out["run_id"] != job["run_id"]:
            raise RuntimeError(f"run id mismatch {out['run_id']} != {job['run_id']} (code or graphs differ)")
        return {"status": out["status"], "result": str(out["result"]), "device": str(dev),
                "torch_threads": threads}

    stop = (lambda: Path(halt_file).exists()) if halt_file else None
    n = Q.run_worker(Path(queue), name, run, log=lambda m: print(m, flush=True), vram_gb=vram_gb,
                     on_oom=free, should_stop=stop)
    print(f"{name}: ran {n} jobs", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-concurrent", type=int, default=1)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--queue", default=str(QUEUE))
    ap.add_argument("--threads", type=int, default=0, help="torch threads per process (0: auto)")
    ap.add_argument("--halt-file", default=None, help="stop claiming jobs once this file exists")
    a = ap.parse_args()
    if a.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("--device cuda but CUDA is unavailable")
    from dftgnn.train import available_cpus

    n_workers, vram_gb = a.max_concurrent, None
    if a.device == "cuda":
        import torch

        from dftgnn.train import admission

        vram_gb = torch.cuda.get_device_properties(0).total_memory / admission.GIB
        try:
            host_gb = admission.est_host_gb(admission.load_benchmark())
        except (OSError, KeyError):
            host_gb = None
        if host_gb:
            cap = admission.max_workers_by_ram(host_gb)
            if cap < n_workers:
                print(f"host-RAM cap: {n_workers} -> {cap} workers (est {host_gb:.1f} GiB each)", flush=True)
            n_workers = min(n_workers, cap)
    threads = a.threads or max(1, int(available_cpus() // n_workers))
    host = socket.gethostname()
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=child, args=(a.queue, f"{host}-{os.getpid()}-w{i}", a.device, threads, vram_gb, a.halt_file))
             for i in range(n_workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    bad = [p.exitcode for p in procs if p.exitcode]
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
