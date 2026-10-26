"""Evaluation tables of the three merged sensitivity stages: high moment, CGCNN and the capped twins.

Rules:, rows of 2026-10-10 (§5, §12; the three designs) and 2026-10-14. The v2 sweep runs of the CGCNN
comparison and of the v2 primary point at B = 654 are read here as well, with the same checks. Pure reading of
committed run records and prediction parquets plus the universe table (read only, sha256 recorded). Writes

    results/analysis/sens_eval_long.parquet   stage, model, r, B, seed, site_id, host_id, y_true, y_pred, run_id
    results/analysis/sens_runs.parquet        one row per run (twins also carry their original's metric)
    results/sensitivity_eval_tables.json      hashes, counts, the excluded entries and the checks (write_result)

Integrity assertions (any failure raises): run ids equal the spec manifests (moment, cgcnn, equal the
`exclude_sites` of every moment spec, appear in no moment prediction, and the recorded training and validation
site counts of every moment run equal those of its hosts without them; each twin's spec equals its original's
apart from the epoch settings; one run per (stage, model, r, B, seed).

    PYTHONPATH=$PWD/src python analysis/sensitivity_eval_tables.py [--data-root DIR]
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.precision_cv_eval_table import check_run, check_targets, gpu_class, save, sha256

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = {"S": "S", "D-state": "D", "cgcnn-S": "cgcnn-S", "cgcnn-D": "cgcnn-D", "P1": "P1"}
STAGE_DIR = {"moment": "moment", "cgcnn": "cgcnn", "capped": "capped_rerun_v1"}
EXPECTED = {"moment": 60, "cgcnn": 18, "capped": 33, "capsens_twin": 3}
CAPSENS_TWINS = {"0b2035de2e44fade": "70471916fc229b83", "373c897db2d3b822": "5b9acd665ad1245d",
                 "d06182b6b418fb01": "6e7a8315a050ee37"}     # 2026-10-10 row: original -> capsens twin
TWIN_SAME = ("model", "split", "r", "budget", "seed", "hp", "lr", "weight_decay", "batch_size")
CGCNN_BUDGETS = (50, 200, 654)
N_EXCLUDED = 10


def v2_cell(spec: Mapping) -> bool:
    """v2 sweep cells read here: resample 0 at the CGCNN budgets, and every resample at B = 654."""
    r, B = int(spec["r"]), int(spec["budget"])
    return B == 654 or (r == 0 and B in CGCNN_BUDGETS)


# ---------------------------------------------------------------- records

def payload(path: Path) -> dict:
    return json.loads(path.read_text())["payload"]


def stage_records(directory: Path, subdir: str) -> list[dict]:
    """Run payloads of one results directory whose spec writes to ``subdir``."""
    out = []
    for f in sorted(directory.glob("*.json")):
        p = payload(f)
        if isinstance(p, dict) and p.get("spec", {}).get("results_subdir") == subdir and not p.get("smoke"):
            out.append(p)
    return out


def spec_run_ids(root: Path, stage: str) -> set[str]:
    lines = (root / "specs" / f"{stage}.jsonl").read_text().splitlines()
    ids = {json.loads(x)["run_id"] for x in lines if x.strip()}
    man = json.loads((root / "specs" / f"{stage}.manifest.json").read_text())
    if len(ids) != man["count"]:
        raise AssertionError(f"{stage}: {len(ids)} spec lines, manifest count {man['count']}")
    return ids


def check_manifest_ids(recs: Iterable[Mapping], want: set[str], stage: str) -> None:
    got = [p["run_id"] for p in recs]
    if len(got) != len(set(got)):
        raise AssertionError(f"{stage}: duplicate run id")
    if set(got) != want:
        raise AssertionError(f"{stage}: run ids differ from the spec manifest "
                             f"(missing {sorted(want - set(got))[:3]}, extra {sorted(set(got) - want)[:3]})")


def run_row(p: Mapping, stage: str) -> dict:
    sp, m = p["spec"], p["metrics"]
    hw = p.get("hardware")
    row = {"stage": stage, "run_id": p["run_id"], "model": MODEL_NAME[sp["model"]], "split": sp["split"],
           "r": int(sp["r"]), "B": int(sp["budget"]), "seed": int(sp["seed"]), "epochs_run": int(p["epochs_run"]),
           "best_epoch": int(p["best_epoch"]), "max_epochs": int(p["max_epochs"]),
           "gpu_class": gpu_class(p, "nrp") if isinstance(hw, Mapping) and hw.get("gpu_model") else "sweep pod",
           "metric_name": "mae" if "mae" in m else "mean_standardised_mae",
           "pred_path": p["predictions"]["path"], "pred_sha256": p["predictions"]["sha256"],
           "n_train_sites": int(p["n_sites"]["train"]), "n_val_sites": int(p["n_sites"]["val"]),
           "n_train_hosts": int(p["n_hosts"]["train"]), "n_val_hosts": int(p["n_hosts"]["val"])}
    row["metric"] = float(m[row["metric_name"]])
    return row


# ---------------------------------------------------------------- high moment

def excluded_sites(uni: pd.DataFrame, threshold: float) -> list[str]:
    ex = sorted(uni.site_id[uni.defect_moment_muB.abs() > threshold])
    if len(ex) != N_EXCLUDED:
        raise AssertionError(f"{len(ex)} entries above {threshold} muB, expected {N_EXCLUDED}")
    return ex


def check_moment_specs(recs: Iterable[Mapping], excluded: list[str]) -> None:
    for p in recs:
        if sorted(p["spec"]["exclude_sites"] or []) != excluded:
            raise AssertionError(f"{p['run_id']}: exclude_sites differ from the universe threshold list")


def check_moment_counts(row: Mapping, train_hosts: Iterable[str], sites_of: Mapping[str, set[str]],
                        excluded: set[str]) -> None:
    """Recorded training + validation sites equal those of the run's hosts minus the excluded entries."""
    kept = [h for h in train_hosts if sites_of[h] - excluded]
    n_sites = sum(len(sites_of[h] - excluded) for h in kept)
    if row["n_train_sites"] + row["n_val_sites"] != n_sites:
        raise AssertionError(f"{row['run_id']}: train+val sites {row['n_train_sites'] + row['n_val_sites']}, "
                             f"expected {n_sites} without the excluded entries")
    if row["n_train_hosts"] + row["n_val_hosts"] != len(kept):
        raise AssertionError(f"{row['run_id']}: train+val hosts differ from the budget hosts with a site left")


# ---------------------------------------------------------------- capped twins

def twin_map(queue: Mapping, twins: Iterable[Mapping]) -> dict[str, str]:
    """original run id -> twin run id for the 36 capped originals of the job specification."""
    by_orig = {}
    for p in twins:
        o = p["spec"]["tags"]["original_run_id"]
        if o in by_orig:
            raise AssertionError(f"two twins for {o}")
        by_orig[o] = p["run_id"]
    by_orig.update(CAPSENS_TWINS)
    jobs = {j["original_run_id"] for j in queue["jobs"]}
    if not all(j["stopped_by_cap"] for j in queue["jobs"]):
        raise AssertionError("a listed original did not stop at the cap")
    if set(by_orig) != jobs:
        raise AssertionError(f"twins do not match the capped originals: {sorted(set(by_orig) ^ jobs)[:4]}")
    return by_orig


def check_twin_spec(orig: Mapping, twin: Mapping) -> None:
    for k in TWIN_SAME:
        if orig["spec"].get(k) != twin["spec"].get(k):
            raise AssertionError(f"{twin['run_id']}: {k} differs from original {orig['run_id']}")
    if int(twin["max_epochs"]) != 600 or int(twin["patience"]) != 30 or int(orig["max_epochs"]) != 200:
        raise AssertionError(f"{twin['run_id']}: epoch settings are not 600 / 30 against an original at 200")


def original_payload(root: Path, run_id: str) -> dict:
    for d in ("results", "results/c0_official"):
        f = root / d / f"{run_id}.json"
        if f.exists():
            return payload(f)
    raise FileNotFoundError(run_id)


# ---------------------------------------------------------------- predictions

def load_preds(row: Mapping, root: Path) -> pd.DataFrame:
    path = root / row["pred_path"]
    if sha256(path) != row["pred_sha256"]:
        raise AssertionError(f"{row['run_id']}: predictions hash mismatch")
    df = pd.read_parquet(path)
    if row["metric_name"] != "mae":
        return df
    df = df[["site_id", "host_id", "y_true", "y_pred"]]
    if abs(float(np.abs(df.y_pred - df.y_true).mean()) - row["metric"]) > 1e-9:
        raise AssertionError(f"{row['run_id']}: recorded MAE differs from the stored predictions")
    return df


def split_hosts(split: str, B: int) -> tuple[set[str], list[str]]:
    from dftgnn.split import budget_train, load_split

    sp = load_split(split)
    if split == "kiyohara":
        return set(sp["test"]), list(sp["train"]) + list(sp["val"])
    return set(sp["test"]), budget_train(sp, B)


def check_unique(runs: pd.DataFrame) -> None:
    dup = runs.duplicated(subset=["stage", "model", "split", "B", "seed"], keep=False)
    if dup.any():
        raise AssertionError(f"more than one run per cell: {runs.run_id[dup].tolist()[:6]}")


def check_eval_table_seed(pred: pd.DataFrame, eval_table: pd.DataFrame, row: Mapping) -> None:
    """A sweep original's predictions equal its seed column of the G1 eval table."""
    sel = eval_table[(eval_table.model == row["model"]) & (eval_table.r == row["r"]) & (eval_table.B == row["B"])]
    got = pred.set_index("site_id").y_pred.sort_index()
    ref = sel.set_index("site_id")[f"y_pred_seed{row['seed']}"].sort_index()
    if not got.index.equals(ref.index) or not np.allclose(got.to_numpy(), ref.to_numpy(), rtol=0, atol=1e-12):
        raise AssertionError(f"{row['run_id']}: predictions differ from the eval table seed column")


# ---------------------------------------------------------------- build

def build(uni: pd.DataFrame, root: Path, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    excluded = excluded_sites(uni, threshold)
    ex_set = set(excluded)
    sites_of = uni.groupby("host_id").site_id.apply(set).to_dict()

    recs = {s: stage_records(root / "results" / d, d) for s, d in STAGE_DIR.items()}
    for s in STAGE_DIR:
        check_manifest_ids(recs[s], spec_run_ids(root, s), s)
    check_moment_specs(recs["moment"], excluded)
    queue = json.loads((root / "scripts" / "queue" / "capped_rerun_v1.json").read_text())
    tmap = twin_map(queue, recs["capped"])
    capsens = [payload(root / "results" / "capsens" / f"{t}.json") for t in CAPSENS_TWINS.values()]
    twins = {p["run_id"]: p for p in recs["capped"] + capsens}

    rows = [run_row(p, "moment") for p in recs["moment"]] + [run_row(p, "cgcnn") for p in recs["cgcnn"]]
    for orig_id, twin_id in sorted(tmap.items()):
        orig, twin = original_payload(root, orig_id), twins[twin_id]
        check_twin_spec(orig, twin)
        t = run_row(twin, "capped_twin")
        o = run_row(orig, "capped_original")
        t.update(original_run_id=orig_id, twin_source="capsens" if twin_id in CAPSENS_TWINS.values() else "capped")
        o.update(original_run_id=orig_id, twin_source=t["twin_source"])
        rows += [t, o]
    v2 = [p for p in stage_records(root / "results" / "v2_sweep", "v2_sweep")
          if p["spec"]["model"] in ("S", "D-state") and int(p["spec"]["seed"]) < 3 and v2_cell(p["spec"])]
    rows += [run_row(p, "v2_sweep") for p in v2]
    runs = pd.DataFrame(rows)
    check_unique(runs)

    counts = runs.stage.value_counts().to_dict()
    n_capsens = int((runs[runs.stage == "capped_twin"].twin_source == "capsens").sum())
    want = {"moment": EXPECTED["moment"], "cgcnn": EXPECTED["cgcnn"],
            "capped_twin": EXPECTED["capped"] + EXPECTED["capsens_twin"],
            "capped_original": EXPECTED["capped"] + EXPECTED["capsens_twin"],
            "v2_sweep": 2 * 3 * 3 + 2 * 9 * 3}
    if counts != want or n_capsens != EXPECTED["capsens_twin"]:
        raise AssertionError(f"run counts {counts} (capsens twins {n_capsens}), expected {want}")

    eval_table = pd.read_parquet(root / "results" / "analysis" / "eval_table.parquet")
    frames, n_checked, ex_hits = [], 0, 0
    for row in runs.to_dict("records"):
        pred = load_preds(row, root)
        test_hosts, train = split_hosts(row["split"], row["B"])
        if row["metric_name"] != "mae":              # P1 predicts descriptors: coverage of sites only
            if pred.site_id.duplicated().any() or set(pred.site_id) != set().union(*(sites_of[h] for h in test_hosts)):
                raise AssertionError(f"{row['run_id']}: P1 sites differ from the test set")
            n_checked += 1
            continue
        sites = set().union(*(sites_of[h] for h in test_hosts))
        if row["stage"] == "moment":
            ex_hits += len(set(pred.site_id) & ex_set)
            sites -= ex_set
            test_hosts = {h for h in test_hosts if sites_of[h] - ex_set}
            check_moment_counts(row, train, sites_of, ex_set)
        check_run(pred, sites, test_hosts, train, f"{row['stage']} {row['run_id']}")
        check_targets(pred, uni)
        if row["stage"] == "capped_original" and row["split"] != "kiyohara" and row["B"] in (400, 654) \
                and row["seed"] < 3:
            check_eval_table_seed(pred, eval_table, row)
        if row["model"] != "P1":
            frames.append(pred.assign(stage=row["stage"], model=row["model"], r=row["r"], B=row["B"],
                                      seed=row["seed"], split=row["split"], run_id=row["run_id"]))
        n_checked += 1
    if ex_hits:
        raise AssertionError(f"{ex_hits} excluded entries appear in moment predictions")
    long = pd.concat(frames, ignore_index=True)
    if not np.isfinite(long.y_pred.to_numpy(float)).all():
        raise AssertionError("non-finite predictions")
    long = long[["stage", "model", "split", "r", "B", "seed", "site_id", "host_id", "y_true", "y_pred", "run_id"]]
    long = long.sort_values(["stage", "model", "split", "B", "r", "seed", "site_id"]).reset_index(drop=True)
    runs = runs.drop(columns=["pred_path"]).sort_values(["stage", "model", "split", "B", "r", "seed"])
    exr = uni[uni.site_id.isin(ex_set)].sort_values("site_id")
    info = {"counts": counts, "capsens_twins": n_capsens, "runs_checked": n_checked,
            "excluded_entries": [{"site_id": s, "host_id": h, "defect_moment_muB": float(m)}
                                 for s, h, m in zip(exr.site_id, exr.host_id, exr.defect_moment_muB, strict=True)],
            "excluded_in_moment_predictions": ex_hits,
            "twin_map": dict(sorted(tmap.items()))}
    return long, runs.reset_index(drop=True), info


def main() -> None:
    from dftgnn.config import load_config
    from dftgnn.io.results import write_result

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, default=ROOT,
                    help="checkout holding the gitignored data/processed/universe_v1.parquet")
    a = ap.parse_args()
    cfg = load_config()
    threshold = float(cfg.sensitivity.exclude_high_moment.threshold_muB)
    uni_path = a.data_root / "data" / "processed" / "universe_v1.parquet"
    uni = pd.read_parquet(uni_path, columns=["site_id", "host_id", "target_Ef_eV", "defect_moment_muB"])
    long, runs, info = build(uni, ROOT, threshold)
    files = {"sens_eval_long": save(long, "sens_eval_long", ROOT), "sens_runs": save(runs, "sens_runs", ROOT)}
    g1_in = json.loads((ROOT / "results" / "g1_primary.json").read_text())["payload"]["input"]["eval_table"]
    payload_ = {
        "label": {"moment": "EXTENSION", "cgcnn": "EXTENSION (untuned)", "capped": "POST HOC",
                  "v2_sweep": "POST HOC (primary backbone; comparator only)"},
        "backbone_of_stages": {s: json.loads((ROOT / "specs" / f"{s}.manifest.json").read_text())["backbone"]
                               for s in STAGE_DIR},
        "rules": "docs/deviations.md rows 2026-10-10 (§5, §12) and 2026-10-14 (§5, §12)",
        "files": files,
        "universe": {"path": "data/processed/universe_v1.parquet", "sha256": sha256(uni_path)},
        "eval_table": {"path": g1_in["path"], "sha256": sha256(ROOT / g1_in["path"]),
                       "matches_g1_primary_input": sha256(ROOT / g1_in["path"]) == g1_in["sha256"]},
        "threshold_muB": threshold,
        "integrity": {**info, "assertions": [
            "run ids equal the spec manifests (moment, cgcnn, capped)",
            "every run covers the test sites of its split exactly once (moment: minus the excluded entries)",
            "all predictions finite", "no test host among the training hosts",
            "targets and hosts equal the universe", "prediction sha256 equals the record",
            "recorded MAE equals the MAE of the stored predictions",
            "the excluded entries are the universe rows above the threshold and equal every moment spec's list",
            "no excluded entry in any moment prediction; moment train+val site counts exclude them",
            "each twin spec equals its original apart from the epoch settings (600 / 30 against 200)",
            "sweep originals at B = 400, 654 equal their seed column of results/analysis/eval_table.parquet",
            "one run per (stage, model, split, B, seed)"], "passed": True},
        "run_ids": {s: sorted(runs.run_id[runs.stage == s].tolist()) for s in sorted(runs.stage.unique())},
    }
    if not payload_["eval_table"]["matches_g1_primary_input"]:
        raise AssertionError("eval_table.parquet differs from the G1 primary input")
    print(json.dumps({k: v for k, v in payload_.items() if k != "run_ids"}, indent=1))
    print(write_result("sensitivity_eval_tables", payload_, config=cfg, repo_root=ROOT))


if __name__ == "__main__":
    main()
