"""Geometry-condition inference task: one model x resample x seed x budget.

Models: S, D (D-state; keeps its true DFT descriptors under every condition, a fixed oracle) and P (P1 on the
condition graph, then D on the same graph with P1's predicted descriptors). The test hosts of the resample are
taken, hosts flagged ``excluded_from_C3b`` (``data/tiling_summary_v1.csv``) are dropped and counted, and every
site is predicted under (i) DFT geometry (the tracked graph store), (ii) MLIP geometry and (iii) MLIP geometry
rescaled to the DFT volume (``data/processed/mlip_v1/<host>_graphs.pt``).

Output: ``results/geomeval/predictions/<task_id>.parquet`` (condition, site_id, host_id, prediction, target,
host flags) and ``results/geomeval/<task_id>.json`` via write_result. No metric is computed here.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from dftgnn import infer
from dftgnn.mlip import geometry
from dftgnn.mlip.pipeline import load_condition_graphs
from dftgnn.mlip.relax import OUT_DIR
from dftgnn.split import load_split

REPO_ROOT = Path(__file__).resolve().parents[3]
TILING_SUMMARY = REPO_ROOT / "data" / "tiling_summary_v1.csv"
DBINS = REPO_ROOT / "data" / "host_dbins_v1.csv"
RESULTS = REPO_ROOT / "results" / "geomeval"
REGISTERED_BUDGETS = (654,)


def components(index_primary: dict, model: str, r: int, budget: int, seed: int) -> dict:
    key = lambda m: index_primary[(m, f"outer_r{r}", budget, seed)]
    if model == "S":
        return {"S": key("S")}
    if model == "D":
        return {"D": key("D-state")}
    if model == "P":
        return {"P1": key("P1"), "D": key("D-state")}
    raise ValueError(model)


def run(model: str, r: int, budget: int, seed: int, comps: dict, sites: infer.Sites, *, hosts=None,
        mlip_dir: Path = OUT_DIR, device=None, batch_size: int = 64) -> tuple[pd.DataFrame, dict]:
    t0 = time.time()
    tiling = pd.read_csv(TILING_SUMMARY).set_index("host_id")
    dbin = pd.read_csv(DBINS).set_index("host_id")["bin"]
    test = sorted(load_split(f"outer_r{r}")["test"]) if hosts is None else sorted(hosts)
    excluded = [h for h in test if bool(tiling.loc[h, "excluded_from_C3b"])]
    keep = [h for h in test if h not in excluded]
    recs = {h: json.loads((mlip_dir / f"{h}.json").read_text()) for h in keep}
    if model == "P":
        net, ck = infer.staged(comps["P1"], comps["D"], device)
    else:
        net, ck = infer.load(next(iter(comps.values())), device)
    pos = sites.positions(keep)
    graphs = {"dft": sites.dft_graphs(keep)}
    cond = {h: load_condition_graphs(h, mlip_dir) for h in keep}
    for c in ("mlip", "mlip_rescaled"):
        graphs[c] = {h: cond[h][c] for h in keep}
    frames = []
    for c in geometry.CONDITIONS:
        pred = infer.predict(net, ck, sites, graphs[c], pos, batch_size=batch_size, device=device)
        hid = sites.host_id[pos]
        frames.append(pd.DataFrame({
            "condition": c, "site_id": [sites.site_id[p] for p in pos], "host_id": hid, "prediction": pred,
            "target": sites.target[pos].numpy(), "converged": [recs[h]["converged"] for h in hid],
            "d_bin": [dbin[h] for h in hid], "eps_v": [recs[h]["geometry"]["eps_v"] for h in hid],
            "deviatoric": [recs[h]["geometry"]["deviatoric"] for h in hid],
            "internal_rmsd_A": [recs[h]["geometry"]["internal_rmsd_A"] for h in hid]}))
    info = {"n_test_hosts": len(test), "n_excluded_from_C3b": len(excluded), "excluded_hosts": excluded,
            "n_hosts_evaluated": len(keep), "n_sites": len(pos),
            "n_nonconverged_hosts": int(sum(not recs[h]["converged"] for h in keep)),
            "registered": budget in REGISTERED_BUDGETS,
            "label": None if budget in REGISTERED_BUDGETS else "DESCRIPTIVE EXTRA (not pre-registered)",
            "wall_time_s": time.time() - t0}
    return pd.concat(frames, ignore_index=True), info
