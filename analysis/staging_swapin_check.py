"""True-descriptor swap-in check of the staged model P, both backbones.

Rule:, row of 2026-10-14, item g. Three (r, B, s) cells per backbone are drawn from the 180 outer cells sorted
by (B, r, s) with ``numpy.random.default_rng(20261014).choice(180, 3, replace=False)``. For each cell the P1
and D checkpoints of its P record are loaded on the CPU (sha256 checked); StagedP with P1's output replaced by
the true descriptors (in P1's standardisation, through StagedP's affine map) is evaluated on the cell's test
sites and compared with D on the same CPU. Pass: maximum absolute difference < 1e-5 eV. The CPU-versus-stored
differences of D and P are reported descriptively. A backbone whose checkpoints are not on this machine is
reported as not run, with the reason; nothing is substituted.

    PYTHONPATH=$PWD/src nice -n 15 python analysis/staging_swapin_check.py [--data-root DIR]

``--data-root`` is the checkout holding the gitignored checkpoints/ and data/processed/graphs_v1/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CELL_SEED = 20261014
N_CELLS = 3
TOL_EV = 1e-5
BACKBONES = ("v2", "v1")


def pick_cells(p_runs: pd.DataFrame, n: int = N_CELLS, seed: int = CELL_SEED) -> pd.DataFrame:
    """The ``n`` P runs drawn from the 180 outer cells sorted by (B, r, seed)."""
    outer = p_runs[p_runs.r >= 0].sort_values(["B", "r", "seed"]).reset_index(drop=True)
    if len(outer) != 180:
        raise AssertionError(f"{len(outer)} outer P cells, expected 180")
    idx = np.random.default_rng(seed).choice(len(outer), n, replace=False)
    return outer.iloc[idx].reset_index(drop=True)


def record_payload(run_id: str, subdir: str | None, root: Path) -> dict:
    path = root / "results" / (subdir or "") / f"{run_id}.json"
    return json.loads(path.read_text())["payload"]


def stored_predictions(rec: dict, root: Path) -> pd.Series:
    from analysis.staging_eval_table import sha256

    path = root / rec["predictions"]["path"]
    if sha256(path) != rec["predictions"]["sha256"]:
        raise AssertionError(f"{rec['run_id']}: predictions hash mismatch")
    return pd.read_parquet(path).set_index("site_id").y_pred


def run_cell(cell, sources: dict, store, data_root: Path) -> dict:
    import torch

    from analysis.staging_eval_table import sha256
    from dftgnn import infer
    from dftgnn.models import StagedP, Standardiser
    from dftgnn.split import load_split
    from dftgnn.train import load_checkpoint

    class SwapIn(torch.nn.Module):
        """StagedP with P1's output replaced by the true descriptors (given in P1's standardisation)."""

        def __init__(self, staged, true_p1):
            super().__init__()
            self.p, self.true_p1 = staged, true_p1

        def forward(self, batch):
            p = self.p
            z = self.true_p1[batch.site_pos.view(-1)]
            z = z if p.identity else z * p.scale + p.shift
            return p.d(batch, desc_host=z[:, : p.n_host], desc_site=z[:, p.n_host:])

    bb = cell.backbone
    p_rec = record_payload(cell.run_id, sources[bb]["P"], ROOT)
    p1_rec = record_payload(cell.p1_run, sources[bb]["P1"], ROOT)
    d_rec = record_payload(cell.d_run, sources[bb]["D-state"], ROOT)
    out = {"backbone": bb, "r": int(cell.r), "B": int(cell.B), "seed": int(cell.seed), "P_run": cell.run_id,
           "P1_run": cell.p1_run, "D_run": cell.d_run}
    paths = {k: data_root / rec["checkpoint"]["path"] for k, rec in (("P1", p1_rec), ("D", d_rec))
             if rec.get("checkpoint")}
    missing = [k for k in ("P1", "D") if k not in paths or not paths[k].is_file()]
    if missing:
        out.update(status="not run", reason=f"checkpoint(s) {missing} not on this machine (recorded sha256 only; "
                                            "the checkpoints of this backbone are on the cluster storage)")
        return out
    for k, rec in (("P1", p1_rec), ("D", d_rec)):
        if sha256(paths[k]) != rec["checkpoint"]["sha256"]:
            raise AssertionError(f"{k} checkpoint {paths[k]} does not match its recorded sha256")
    cpu = torch.device("cpu")
    p1, ck1 = load_checkpoint(paths["P1"], cpu)
    d, ckd = load_checkpoint(paths["D"], cpu)
    p = StagedP(p1, d, Standardiser(ck1["desc_mean"], ck1["desc_sd"]),
                Standardiser(ckd["desc_mean"], ckd["desc_sd"])).eval()
    sites = infer.Sites(store)
    test = sorted(load_split(f"outer_r{int(cell.r)}")["test"])
    pos = sites.positions(test)
    graphs = sites.dft_graphs(test)
    y_d = infer.predict(d, ckd, sites, graphs, pos)
    y_p = infer.predict(p, ckd, sites, graphs, pos)
    true_p1 = Standardiser(ck1["desc_mean"], ck1["desc_sd"])(sites.desc).float()
    y_swap = infer.predict(SwapIn(p, true_p1).eval(), ckd, sites, graphs, pos)
    site_ids = [sites.site_id[int(i)] for i in pos]
    st_d = stored_predictions(d_rec, ROOT).loc[site_ids].to_numpy(float)
    st_p = stored_predictions(p_rec, ROOT).loc[site_ids].to_numpy(float)
    dev = float(np.abs(y_swap - y_d).max())
    out.update(status="run", n_test_sites=len(pos), affine_identity=bool(p.identity),
               swap_in_max_abs_dev_eV=dev, passed=bool(dev < TOL_EV),
               cpu_vs_stored_D_max_abs_eV=float(np.abs(y_d - st_d).max()),
               cpu_vs_stored_P_max_abs_eV=float(np.abs(y_p - st_p).max()),
               stored_device={"D": d_rec.get("device"), "P": p_rec.get("device")})
    return out


def main() -> None:
    import torch

    from analysis.staging_eval_table import SOURCES, sha256
    from dftgnn.config import load_config
    from dftgnn.io.results import write_result
    from dftgnn.train import Store

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, default=ROOT)
    a = ap.parse_args()
    torch.set_num_threads(3)
    cfg = load_config()
    rec = json.loads((ROOT / "results" / "staging_eval_table.json").read_text())
    ref = rec["payload"]["files"]["staging_runs"]
    if sha256(ROOT / ref["path"]) != ref["sha256"]:
        raise AssertionError("staging_runs.parquet differs from its eval-table record")
    runs = pd.read_parquet(ROOT / ref["path"])
    store = Store(store_dir=a.data_root / "data" / "processed" / "graphs_v1",
                  manifest=ROOT / "data" / "graphs_v1_manifest.sha256")
    rows = []
    for bb in BACKBONES:
        cells = pick_cells(runs[(runs.backbone == bb) & (runs.model == "P")])
        for cell in cells.itertuples(index=False):
            row = run_cell(cell, SOURCES, store, a.data_root)
            print(json.dumps(row), flush=True)
            rows.append(row)
    summary = {}
    for bb in BACKBONES:
        mine = [r for r in rows if r["backbone"] == bb]
        ran = [r for r in mine if r["status"] == "run"]
        summary[bb] = {"n_cells": len(mine), "n_run": len(ran), "n_passed": sum(r["passed"] for r in ran),
                       "max_abs_dev_eV": max((r["swap_in_max_abs_dev_eV"] for r in ran), default=None),
                       "status": ("passed" if ran and len(ran) == len(mine) and all(r["passed"] for r in ran)
                                  else "failed" if any(not r["passed"] for r in ran) else "not run")}
        if any(not r["passed"] for r in ran):
            raise AssertionError(f"{bb}: swap-in check failed: {summary[bb]}")
    payload = {"label": "integrity check (row of 2026-10-14, §8, item g); not a result",
               "rule": {"cells": f"numpy.random.default_rng({CELL_SEED}).choice(180, {N_CELLS}, replace=False) over "
                                 "the outer P cells sorted by (B, r, seed)", "tolerance_eV": TOL_EV,
                        "device": "cpu"},
               "eval_table": {"path": ref["path"], "sha256": ref["sha256"], "record_git_sha": rec["git_sha"]},
               "summary": summary, "cells": rows}
    print(write_result("staging_swapin_check", payload, config=cfg, repo_root=ROOT))


if __name__ == "__main__":
    main()
