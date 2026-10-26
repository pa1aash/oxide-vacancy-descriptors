"""B0 and RF-Kumagai on the leave-chemistry-out folds. Same fitting code and seed rule as
``scripts/run_baselines.py``, with the fold index in place of the resample. Writes
results/predictions/loco_b0_physics_floor.parquet, loco_rf_kumagai.parquet and results/loco_baselines.json.

    python scripts/loco_baselines.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from dftgnn.baselines import Data, feature_sets, metrics_block, run_split, write_predictions
from dftgnn.baselines import physics, rf
from dftgnn.config import load_config
from dftgnn.io.results import write_result
from dftgnn.split import derived_seed, load_split


def main() -> None:
    cfg = load_config()
    data = Data.load(cfg)
    fits = {"b0_physics_floor": ("b0", physics.make_fit(data)),
            "rf_kumagai": ("rf_kumagai", rf.make_fit(data, feature_sets(cfg, data)["kumagai"], importances=False))}
    payload = {"folds": cfg.robustness.loco.k, "models": {}}
    for name, (tag, fit) in fits.items():
        frames, per_fold = [], []
        for k in range(cfg.robustness.loco.k):
            sp = load_split(f"loco/loco_f{k}")
            seed = derived_seed(k, f"{tag}_loco") % 2**31
            frame, info = run_split(data, fit, sp["budget_order"], sp["test"], seed, fold=k)
            frames.append(frame)
            per_fold.append({"fold": k, "seed": seed, **info,
                             "metrics": metrics_block(frame, n_boot=cfg.stats.cluster_bootstrap_n, seed=0)})
            print(name, k, round(per_fold[-1]["metrics"]["mae"]["point"], 3))
        allf = pd.concat(frames, ignore_index=True)
        payload["models"][name] = {"per_fold": per_fold, "predictions": write_predictions(f"loco_{name}", allf),
                                   "pooled": metrics_block(allf, n_boot=cfg.stats.cluster_bootstrap_n, seed=0)}
    print(write_result("loco_baselines", payload, config=cfg))


if __name__ == "__main__":
    main()
