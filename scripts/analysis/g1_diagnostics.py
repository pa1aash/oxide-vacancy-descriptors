"""Two observations made while checking the G1 results.

1. Within-host spread of the predictions: mean over (resample, test host with >= 2 sites) of the sample SD of a
   model's predictions across the host's sites, per model and budget (seed ensemble and seed 0).
2. Epoch-cap pairs: whether each 600-epoch run reproduces its 200-epoch twin (prediction sha256, best epoch).

    python scripts/analysis/g1_diagnostics.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dftgnn.config import load_config
from dftgnn.io.results import write_result

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    cfg = load_config()
    t = pd.read_parquet(ROOT / "results" / "analysis" / "eval_table.parquet")
    spread = {}
    for (m, B), g in t[t.model.isin(["S", "D"])].groupby(["model", "B"]):
        g = g[g.groupby(["r", "host_id"]).y_pred.transform("size") >= 2]
        k = g.groupby(["r", "host_id"])
        spread[f"{m}|{B}"] = {"ensemble_eV": float(k.y_pred.std().mean()),
                              "seed0_eV": float(k.y_pred_seed0.std().mean()),
                              "target_eV": float(k.y_true.std().mean())}
    recs = {}
    for sub in ("", "capsens"):
        for f in sorted((ROOT / "results" / sub).glob("*.json")):
            try:
                p = json.loads(f.read_text()).get("payload")
            except ValueError:
                continue
            if not isinstance(p, dict) or "spec" not in p:
                continue
            sp = p["spec"]
            if sp.get("model") not in ("S", "D-state") or sp.get("budget") != 654 or sp.get("r") not in (0, 1, 2):
                continue
            if not str(sp.get("split", "")).startswith("outer_r") or sp.get("results_subdir") != (sub or None):
                continue
            recs[(sub or "sweep", sp["model"], sp["r"], sp["seed"])] = p
    pairs = []
    for (stage, m, r, s), p in sorted(recs.items()):
        if stage != "capsens":
            continue
        q = recs[("sweep", m, r, s)]
        pairs.append({"model": m, "r": r, "seed": s, "run_600": p["run_id"], "run_200": q["run_id"],
                      "best_epoch_600": p["best_epoch"], "best_epoch_200": q["best_epoch"],
                      "epochs_run_600": p["epochs_run"], "epochs_run_200": q["epochs_run"],
                      "predictions_identical_sha256": p["predictions"]["sha256"] == q["predictions"]["sha256"]})
    payload = {
        "within_host_prediction_sd": {"definition": "mean over (r, test host with >= 2 sites) of the sample SD "
                                                    "of predictions across the host's sites", "values": spread},
        "epoch_cap_pairs": {"pairs": pairs, "n_identical": sum(x["predictions_identical_sha256"] for x in pairs),
                            "n_pairs": len(pairs)},
        "descriptive": True,
    }
    print(json.dumps(payload["within_host_prediction_sd"], indent=1))
    print(payload["epoch_cap_pairs"]["n_identical"], "/", len(pairs))
    print(write_result("g1_diagnostics", payload, config=cfg))


if __name__ == "__main__":
    main()
