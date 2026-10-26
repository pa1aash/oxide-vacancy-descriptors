"""Run one geometry-condition inference task (src/dftgnn/mlip/geomeval.py, src/dftgnn/tasks.py); idempotent.

    python scripts/infer/geomeval.py --model S --r 0 --seed 0 --budget 654 [--device cuda]

On the pod the same task runs as queue stage ``geomeval``; this CLI computes the same run id.
"""
from __future__ import annotations

import argparse

import torch

from dftgnn import infer, tasks
from dftgnn.mlip import geomeval as GE
from dftgnn.train import Store, code_sha, runindex


def main(argv=None) -> str:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["S", "D", "P"])
    ap.add_argument("--r", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    prim = runindex.primary(runindex.load())
    comps = GE.components(prim, a.model, a.r, a.budget, a.seed)
    key = tasks.geomeval_key(a.model, a.r, a.budget, a.seed, comps, code_sha())
    rid = infer.task_id(key)
    spec = {"model": "geomeval", "geo_model": a.model, "r": a.r, "budget": a.budget, "seed": a.seed}
    if tasks.verified(tasks.result_path("geomeval", rid), rid):
        print(f"{rid}: already done")
        return rid
    path = tasks.run_geomeval(rid, spec, infer.Sites(Store()), prim, key, torch.device(a.device))
    print(f"{rid}: {path}")
    return rid


if __name__ == "__main__":
    main()
