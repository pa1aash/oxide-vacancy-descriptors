"""The descriptor-class ablation of the final network (stage dclass), with the G1 estimator unchanged.

Nothing is trained. The 180 runs of stage dclass (final network D, He initialisation, outer resamples 0-9, seeds 0-2,
B = 200, 400, 654) read one class of pristine-cell electronic descriptors only: "D-host" keeps the 12 host-level
descriptors (band gap, oxide formation energy, dielectric response, band-edge orbital character) and ablates the
site-level ones; "D-site" keeps the 10 site-level descriptors (Bader charge, local band-edge character) and ablates
the host-level ones. Physically: the class that carries the descriptor advantage A = MAE_S - MAE_D is the class that
recovers it when given to the network alone.

Estimator (imported unchanged from analysis/c1/final_network.py and analysis/staging_analysis.py): seed-ensemble mean
per (model, r, B, site), per-host sufficient statistics, the G1 hierarchical bootstrap ``HierDraws`` (2000 replicates,
seed 20261008, one draw set for every model and budget, so every contrast is paired), ``contrast`` for paired
differences, ``a_entry`` for the interval and the one-sided 95% upper bound against delta = 0.05 eV. S and D are the
runs of results/analysis/staging_eval_long.parquet (backbone v2, the 10 x 3 sweep); D-host and D-site use the same
resamples, seeds and test sites. The 30 x 5 design of the headline (results/precision_a_b400_b654.json) has no
dclass runs, so every number here is on the 10 x 3 design (A at 654 hosts is the sweep value, not the headline).

Recovery of A by a class c: R_c = (MAE_S - MAE_c) / (MAE_S - MAE_D), the point ratio of the mean gaps; its interval is
the percentile interval of the replicate-wise ratio on the shared draws and is reported only where the two-sided
interval of A excludes zero (otherwise the denominator changes sign within the draws and the ratio has no interval).

Health: 8 runs reached the 200-epoch limit. The sensitivity drops the (B, r, seed) cells of those runs from every model
(S, D, D-host, D-site), so the pairing is kept, and recomputes everything with the seed ensemble of the remaining seeds.

Checks before a record is written (tolerance 1e-9 eV unless stated): run ids equal specs/dclass.jsonl; git_dirty false;
prediction sha256 equals the record; the recorded MAE equals that of the stored predictions (1e-6); finite
predictions; every run covers exactly the test sites of the S run of its cell, with equal targets; S, D MAE and A
reproduce rows B200, B400, B654 of results/derived_c1/final_network_summary.json.

Writes through ``write_result`` into ``results/derived_dclass/dclass_gaps.json``.

    PYTHONPATH=$PWD/src python analysis/dclass/dclass_analysis.py [--dry-run]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis import a1_common as c
from analysis import staging_analysis
from analysis.c1.final_network import a_entry, compare, ms
from dftgnn.stats.g1 import point_metric, resample_stats, seed_ensemble

OUT_DIR = ROOT / "results" / "derived_dclass"
STAGE_DIR = ROOT / "results" / "dclass"
BUDGETS = (200, 400, 654)
N_RESAMPLES, SEEDS = 10, (0, 1, 2)
DELTA = c.DELTA_EV
CAPPED = ("74200f613b1259ab", "91575dd604bd0184", "e484edc1c1b2a996", "3012f8b407994a7b", "09626ea67790038e",
          "1556e1f2d0de532a", "7812ffc2b2dd5967", "3a2a0181ed0182f3")
CLASS = {"D-host": "host-level descriptors only (12 kept, site-level ablated)",
         "D-site": "site-level descriptors only (10 kept, host-level ablated)"}
GAPS = (("S_minus_D", "S", "D"), ("S_minus_Dhost", "S", "D-host"), ("S_minus_Dsite", "S", "D-site"),
        ("D_minus_Dhost", "D", "D-host"), ("D_minus_Dsite", "D", "D-site"))
LABEL = ("EXTENSION on the final network (v2, POST HOC backbone); stage dclass of the N5 tranche, "
         "spec sha256 f0de2b14ba50d659258e31e5c143329a2f2807e28045cd72cc2b7c953d6a549c")


def load_sweep() -> tuple[pd.DataFrame, dict]:
    long, f = staging_analysis.load_table("staging_eval_long", c.record("staging_eval_table"))
    s = long[(long.backbone == "v2") & long.model.isin(["S", "D"]) & long.B.isin(BUDGETS)]
    return s.drop(columns=["backbone", "run_id"]).reset_index(drop=True), f


def load_dclass(sweep: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    spec = {}
    for line in (ROOT / "specs" / "dclass.jsonl").read_text().splitlines():
        row = json.loads(line)
        spec[row["run_id"]] = row
    files = sorted(STAGE_DIR.glob("*.json"))
    if len(files) != 180 or {p.stem for p in files} != set(spec):
        raise AssertionError("dclass records differ from specs/dclass.jsonl")
    ref = {(m, B, r): g.set_index("site_id") for (m, B, r), g in sweep[sweep.seed == 0].groupby(["model", "B", "r"])}
    frames, worst_mae, cells, capped_found = [], 0.0, set(), []
    for p in files:
        rec = json.loads(p.read_text())
        pl = rec["payload"]
        s = spec[p.stem]["spec"]
        if rec["git_dirty"] or pl["run_id"] != p.stem or pl["code_ref"] != "nrp-code-v1" or pl["smoke"]:
            raise AssertionError(f"{p.stem}: dirty, misnamed, wrong code ref or smoke")
        if pl["spec"] != s:
            raise AssertionError(f"{p.stem}: recorded spec differs from specs/dclass.jsonl")
        cls = s["tags"]["dclass"]
        if (cls, s["ablate"]) not in (("D-host", "desc_site"), ("D-site", "desc_host")):
            raise AssertionError(f"{p.stem}: class {cls} with ablate {s['ablate']}")
        B, r, seed = int(s["budget"]), int(s["r"]), int(s["seed"])
        pred_path = ROOT / pl["predictions"]["path"]
        if c.sha256(pred_path) != pl["predictions"]["sha256"]:
            raise AssertionError(f"{p.stem}: prediction sha256")
        d = pd.read_parquet(pred_path)
        if not np.isfinite(d[["y_true", "y_pred"]].to_numpy()).all():
            raise AssertionError(f"{p.stem}: non-finite predictions")
        mae = float(np.abs(d.y_true - d.y_pred).mean())
        worst_mae = max(worst_mae, abs(mae - pl["metrics"]["mae"]))
        base = ref[("S", B, r)]
        if len(d) != len(base) or set(d.site_id) != set(base.index) or d.site_id.duplicated().any():
            raise AssertionError(f"{p.stem}: test sites differ from the S run of the cell")
        j = d.set_index("site_id").loc[base.index]
        if not (np.array_equal(j.y_true.to_numpy(), base.y_true.to_numpy()) and (j.host_id == base.host_id).all()):
            raise AssertionError(f"{p.stem}: targets or hosts differ from the S run of the cell")
        if (cls, B, r, seed) in cells:
            raise AssertionError(f"{p.stem}: duplicate cell")
        cells.add((cls, B, r, seed))
        if pl["epochs_run"] >= pl["max_epochs"]:
            capped_found.append(p.stem)
        frames.append(d.assign(model=cls, r=r, B=B, seed=seed)[["model", "r", "B", "seed", "site_id", "host_id",
                                                                "y_true", "y_pred"]])
    if worst_mae > 1e-6:
        raise AssertionError(f"recorded MAE differs from the stored predictions by {worst_mae}")
    if sorted(capped_found) != sorted(CAPPED):
        raise AssertionError(f"capped runs {sorted(capped_found)} differ from the listed eight")
    if len(cells) != 2 * len(BUDGETS) * N_RESAMPLES * len(SEEDS):
        raise AssertionError("dclass grid incomplete")
    info = {"n_runs": len(files), "n_prediction_rows": int(sum(map(len, frames))),
            "max_abs_diff_recorded_mae_vs_predictions": worst_mae, "spec_sha256": c.sha256(ROOT / "specs/dclass.jsonl"),
            "n_runs_at_epoch_limit": len(capped_found), "runs_at_epoch_limit": sorted(capped_found)}
    return pd.concat(frames, ignore_index=True), info


def table(long: pd.DataFrame, draws=None) -> tuple[dict, object]:
    """Per budget: MAE of the four models, the five paired gaps, the recovery ratios. Seed-ensemble per cell."""
    ens = seed_ensemble(long, keys=("model", "r", "B", "site_id"))
    ens["backbone"] = "v2"
    if draws is None:
        draws = staging_analysis.make_draws(ens)
    models = ("S", "D", "D-host", "D-site")
    st = {m: staging_analysis.stats_by_budget(ens[ens.model == m].assign(backbone="v2"), m, BUDGETS) for m in models}
    reps = {m: {B: draws.replicates(st[m][B], "mae") for B in BUDGETS} for m in models}
    per = {m: {B: np.array([point_metric(s, "mae") for s in st[m][B]]) for B in BUDGETS} for m in models}
    out = {}
    from dftgnn.stats.g1 import contrast
    ct = {(a, b): contrast(st[a], st[b], draws, "mae", ci=c.CI, upper_quantile=c.UPPER_Q)
          for _, a, b in GAPS}
    n_seeds = ens.groupby(["model", "B", "r"]).n_seeds.first()
    for B in BUDGETS:
        row = {"x_training_hosts": B, "MAE": {m: ms(ct[("S", "D")][B]["a"] if m == "S" else
                                                  ct[("S", "D")][B]["b"] if m == "D" else
                                                  ct[("S", m)][B]["b"]) for m in models},
               "gaps": {k: a_entry(ct[(a, b)][B]["diff"]) for k, a, b in GAPS}}
        a_rep = reps["S"][B] - reps["D"][B]
        a_pt = float((per["S"][B] - per["D"][B]).mean())
        stable = bool(row["gaps"]["S_minus_D"]["ci95_excludes_zero"])
        rec = {}
        for cl in ("D-host", "D-site"):
            g_rep, g_pt = reps["S"][B] - reps[cl][B], float((per["S"][B] - per[cl][B]).mean())
            ratio_rep = g_rep / a_rep
            lo, hi = (1 - c.CI) / 2, 1 - (1 - c.CI) / 2
            rec[cl] = {"ratio": g_pt / a_pt, "ratio_ci95": [float(np.quantile(ratio_rep, lo)),
                                                           float(np.quantile(ratio_rep, hi))] if stable else None,
                       "ratio_ci95_reported": stable, "A_ci95_excludes_zero": stable,
                       "gap_S_minus_class": g_pt, "A": a_pt}
        row["recovery_of_A"] = rec
        row["n_seeds_per_cell_min"] = int(n_seeds.xs(B, level="B").min())
        out[str(B)] = row
    return out, draws


def reproduce(main: dict) -> dict:
    rows = c.record("derived_c1/final_network_summary")["payload"]["table"]["rows"]
    mine = {f"B{B}": {"MAE_S": main[str(B)]["MAE"]["S"], "MAE_D": main[str(B)]["MAE"]["D"],
                      "A": {"A": main[str(B)]["gaps"]["S_minus_D"]["A"],
                            "ci95": main[str(B)]["gaps"]["S_minus_D"]["ci95"],
                            "upper_one_sided_95": main[str(B)]["gaps"]["S_minus_D"]["upper_one_sided_95"]}}
            for B in BUDGETS}
    ref = {k: {"MAE_S": rows[k]["MAE_S"], "MAE_D": rows[k]["MAE_D"],
               "A": {x: rows[k]["A"][x] for x in ("A", "ci95", "upper_one_sided_95")}} for k in mine}
    r = compare(mine, ref, "results/derived_c1/final_network_summary.json payload.table.rows.B200/B400/B654 "
                           "(S, D MAE with ci95; A with ci95 and one-sided bound)")
    if not r["reproduced"]:
        raise AssertionError(f"S and D do not reproduce the final-network table: {r}")
    return r


def main() -> None:
    from dftgnn.config import load_config
    from dftgnn.io.results import write_result

    cfg = load_config()
    if float(cfg.stats.delta_eV) != DELTA:
        raise AssertionError("stats.delta_eV differs from 0.05")
    sweep, f_long = load_sweep()
    dc, info = load_dclass(sweep)
    long = pd.concat([sweep, dc], ignore_index=True)
    main_t, draws = table(long)
    repro = reproduce(main_t)

    drop = {(int(json.loads((STAGE_DIR / f"{i}.json").read_text())["payload"]["spec"]["budget"]),
             int(json.loads((STAGE_DIR / f"{i}.json").read_text())["payload"]["spec"]["r"]),
             int(json.loads((STAGE_DIR / f"{i}.json").read_text())["payload"]["spec"]["seed"])) for i in CAPPED}
    key = list(zip(long.B, long.r, long.seed))
    kept = long[[k not in drop for k in key]]
    sens_t, _ = table(kept, draws)

    payload = {"label": LABEL, "delta_eV": DELTA, "design": "10 resamples x 3 seeds (the sweep design), outer resamples "
               "r = 0-9, seed-ensemble mean over seeds 0-2, 2000 hierarchical bootstrap replicates, seed 20261008",
               "classes": CLASS, "budgets": main_t,
               "health": {**info, "cells_dropped_in_sensitivity": sorted([list(k) for k in drop]),
                          "n_cells_dropped": len(drop), "sensitivity": sens_t},
               "reproduction": repro,
               "inputs": {"staging_eval_long": f_long, "dclass_manifest": "results/nrp/artefacts/dclass.sha256"}}
    payload = c.to_jsonable(payload)
    if "--dry-run" in sys.argv:
        print(json.dumps(payload, indent=1))
        return
    print(write_result("dclass_gaps", payload, config=cfg, repo_root=ROOT, results_dir=OUT_DIR))


if __name__ == "__main__":
    main()
