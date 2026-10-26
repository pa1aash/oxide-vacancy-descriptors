"""Shared pieces of the precision and grouped-CV analyses (EXTENSION).

Rules:, rows of 2026-10-11 (§7, §11) and 2026-10-14 (§7, §11). Holds the input loading with hash checks, the
per-host statistics shared by both designs, the within-host skill used by the figure records, and the
descriptive extras of item e of the 2026-10-14 row (GPU class, capped versus uncapped; no test).
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DELTA_EV = 0.05
N_BOOT = 2000
BOOT_SEED = 20261008          # fixed by the 2026-10-11 rule (precision and CV)
DESC_SEED = 20261014          # descriptive GPU-class draws only (2026-10-14 row, item e1)
CI = 0.95
UPPER_Q = 0.95
BACKBONES = ("v2", "v1")
MODELS = ("S", "D")
GPU_ORDER = ("A10", "RTX 3090", "L4", "RTX 4090", "original sweep")
DESIGN_POINTS = (("precision", 400), ("precision", 654), ("cv", 10), ("cv", 20))
LABEL = "EXTENSION (v2: primary backbone, POST HOC; v1: originally registered backbone)"
GPU_NOTE = {"RTX 4090": "not characterised in N1 (no RTX 4090 in the N1 repeatability runs)",
            "original sweep": "r = 0..9, seeds 0-2 of the pre-registered (v1) or v2 sweep, not run on the NRP pool"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record(name: str, root: Path = ROOT) -> dict:
    return json.loads((root / "results" / f"{name}.json").read_text())


def load_table(name: str, root: Path = ROOT) -> tuple[pd.DataFrame, dict]:
    """A parquet listed in ``results/precision_cv_eval_table.json``, after its sha256 check."""
    rec = record("precision_cv_eval_table", root)
    ref = rec["payload"]["files"][name]
    path = root / ref["path"]
    got = sha256(path)
    if got != ref["sha256"]:
        raise AssertionError(f"{name}: sha256 {got} differs from the eval-table record")
    return pd.read_parquet(path), {"path": ref["path"], "sha256": got, "record_git_sha": rec["git_sha"]}


def quantiles(reps: np.ndarray) -> dict:
    lo, hi = (1 - CI) / 2, 1 - (1 - CI) / 2
    return {"ci95": [float(np.quantile(reps, lo)), float(np.quantile(reps, hi))],
            "upper_one_sided_95": float(np.quantile(reps, UPPER_Q))}


def constant_within_host(frame: pd.DataFrame) -> pd.DataFrame:
    """The same sites with a prediction constant within every host (its within-host MAE is the zero-skill value)."""
    return frame.assign(y_pred=0.0)


# ---------------------------------------------------------------- descriptive extras (item e)

def gpu_cells(runs: pd.DataFrame, design: str, point: int) -> pd.DataFrame:
    """Per cell (r, B, seed) or (K, fold, seed) of one design point: S and D single-run MAE and GPU class."""
    sel = runs[runs.design == design]
    sel = sel[sel.B == point] if design == "precision" else sel[sel.K == point]
    keys = ["backbone", "r", "B", "seed"] if design == "precision" else ["backbone", "K", "fold", "seed"]
    wide = sel.pivot_table(index=keys, columns="model", values=["mae", "gpu_class"], aggfunc="first")
    out = pd.DataFrame({"mae_S": wide["mae"]["S"], "mae_D": wide["mae"]["D"],
                        "gpu_S": wide["gpu_class"]["S"], "gpu_D": wide["gpu_class"]["D"]}).reset_index()
    return out


def gpu_table(runs: pd.DataFrame, design: str, point: int, n_boot: int = N_BOOT) -> list[dict]:
    """Rows of the GPU-class table for one design point, both backbones (2026-10-14 row, item e1)."""
    cells = gpu_cells(runs, design, point)
    sel = runs[runs.design == design]
    sel = sel[sel.B == point] if design == "precision" else sel[sel.K == point]
    rows = []
    for bb in BACKBONES:
        for cls in GPU_ORDER:
            r_bb = sel[(sel.backbone == bb) & (sel.gpu_class == cls)]
            paired = cells[(cells.backbone == bb) & (cells.gpu_S == cls) & (cells.gpu_D == cls)]
            if r_bb.empty:
                continue
            row = {"backbone": bb, "design": design, "point": point, "gpu_class": cls,
                   "n_runs": {m: int((r_bb.model == m).sum()) for m in MODELS},
                   "mean_single_run_mae": {m: (float(r_bb.mae[r_bb.model == m].mean())
                                               if (r_bb.model == m).any() else None) for m in MODELS},
                   "n_paired_cells": len(paired)}
            if len(paired):
                d = (paired.mae_S - paired.mae_D).to_numpy(float)
                k = desc_index(bb, design, point, cls)
                rng = np.random.default_rng(np.random.SeedSequence([DESC_SEED, k]))
                idx = rng.integers(0, len(d), size=(n_boot, len(d)))
                reps = d[idx].mean(axis=1)
                lo, hi = (1 - CI) / 2, 1 - (1 - CI) / 2
                row["A_paired_cells"] = {"A": float(d.mean()), "sd_over_cells": float(d.std(ddof=1)) if len(d) > 1
                                         else None, "ci95_descriptive": [float(np.quantile(reps, lo)),
                                                                         float(np.quantile(reps, hi))],
                                         "bootstrap_seed": [DESC_SEED, k]}
            if cls in GPU_NOTE:
                row["note"] = GPU_NOTE[cls]
            rows.append(row)
    return rows


def desc_index(backbone: str, design: str, point: int, cls: str) -> int:
    """Position of a GPU-table row in the fixed order backbone, design point, class (item e1)."""
    n_c, n_p = len(GPU_ORDER), len(DESIGN_POINTS)
    return (BACKBONES.index(backbone) * n_p + DESIGN_POINTS.index((design, point))) * n_c + GPU_ORDER.index(cls)


def capped_table(runs: pd.DataFrame, design: str, point: int) -> list[dict]:
    """Capped (epochs_run >= max_epochs) versus uncapped runs for one design point (item e2); no test."""
    sel = runs[runs.design == design]
    sel = sel[sel.B == point] if design == "precision" else sel[sel.K == point]
    rows = []
    for bb in BACKBONES:
        for m in MODELS:
            g = sel[(sel.backbone == bb) & (sel.model == m)]
            row = {"backbone": bb, "design": design, "point": point, "model": m, "n_runs": len(g),
                   "n_capped": int(g.capped.sum()), "n_uncapped": int((~g.capped).sum()),
                   "mean_single_run_mae_capped": float(g.mae[g.capped].mean()) if g.capped.any() else None,
                   "mean_single_run_mae_uncapped": float(g.mae[~g.capped].mean()) if (~g.capped).any() else None}
            if design == "precision":
                row["n_capped_by_source"] = {s: int(g.capped[g.source == s].sum()) for s in sorted(g.source.unique())}
            rows.append(row)
    return rows


def to_jsonable(x):
    if isinstance(x, Mapping):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def passes(upper: float, delta: float = DELTA_EV) -> bool:
    """Passes."""
    return bool(upper < delta)


def subset_names() -> Sequence[tuple[str, int, int]]:
    """(name, n_resamples, n_seeds) of the pooled set and the beside-series (2026-10-11 rule)."""
    return (("30x5", 30, 5), ("10x3", 10, 3), ("10x5", 10, 5), ("30x3", 30, 3))
