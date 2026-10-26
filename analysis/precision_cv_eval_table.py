"""Long evaluation tables of the precision and grouped-CV extensions, both backbones.

Analysis rules:, rows of 2026-10-11 and of 2026-10-14 (the implementation of those rules. Pure reading of
committed run records and prediction parquets plus the universe table (read only, sha256 recorded). Writes

    results/analysis/precision_eval_long.parquet   backbone, model, r, B, seed, site_id, host_id, y_true, y_pred
    results/analysis/cv_eval_long.parquet          backbone, model, K, fold, seed, site_id, host_id, y_true, y_pred
    results/analysis/precision_cv_runs.parquet     one row per run: backbone, design, source, model, cell, seed,
                                                   run_id, gpu_class, epochs_run, max_epochs, capped, mae
    results/precision_cv_eval_table.json           hashes, counts and the integrity checks (via write_result)

Precision pool per backbone: S and D-state at B = 400 and 654, r = 0..29, seeds 0-4. The cells r = 0..9, seeds 0-2
are the original sweep runs (v1: the S and D-state records directly under `results/`; v2: `results/v2_sweep/`);
every other cell comes from `results/precision/` (v1) or `results/precision_v2/` (v2).

Integrity assertions (any failure raises): every run covers the test sites of its split exactly once; all
predictions finite; no test host among the training hosts; targets and hosts equal the universe; prediction sha256
equals the record; recorded MAE equals the MAE of the stored predictions; one run per (backbone, model, cell,
seed); the folds of each K partition the 818 hosts; run counts precision 480 + 480 and cv 180 + 180.

    PYTHONPATH=$PWD/src python analysis/precision_cv_eval_table.py [--data-root DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "analysis"
MODEL_NAME = {"S": "S", "D-state": "D"}
BUDGETS = (400, 654)
KS = (10, 20)
N_HOSTS = 818
BACKBONES = ("v2", "v1")
SOURCES = {"v1": {"sweep": "", "precision": "precision", "cv": "cv"},
           "v2": {"sweep": "v2_sweep", "precision": "precision_v2", "cv": "cv_v2"}}
EXPECTED_NEW = {"precision": 480, "cv": 180}
EXPECTED_SWEEP = 2 * len(BUDGETS) * 10 * 3
GPU_CLASS = {"NVIDIA A10": "A10", "NVIDIA GeForce RTX 3090": "RTX 3090", "NVIDIA GeForce RTX 4090": "RTX 4090",
             "NVIDIA L4": "L4"}
SWEEP_CLASS = "original sweep"
TRACK_LIMIT = 20 * 1024**2


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- run records

def gpu_class(payload: Mapping, source: str) -> str:
    if source == "sweep":
        return SWEEP_CLASS
    model = (payload.get("hardware") or {}).get("gpu_model")
    if model not in GPU_CLASS:
        raise ValueError(f"{payload.get('run_id')}: unknown GPU model {model!r}")
    return GPU_CLASS[model]


def read_records(directory: Path, subdir: str | None) -> list[dict]:
    """S and D-state run payloads of one results directory; ``subdir`` is the expected ``results_subdir``."""
    out = []
    for f in sorted(directory.glob("*.json")):
        try:
            pay = json.loads(f.read_text()).get("payload")
        except ValueError:
            continue
        if not isinstance(pay, dict) or "spec" not in pay or pay.get("smoke"):
            continue
        sp = pay["spec"]
        if sp.get("model") not in MODEL_NAME or sp.get("tags", {}).get("pilot"):
            continue
        if sp.get("results_subdir") != subdir:
            continue
        out.append(pay)
    return out


def select_sweep(recs: Iterable[dict]) -> list[dict]:
    """Sweep records of the precision cells: outer splits r = 0..9, B in BUDGETS, seeds 0-2, 200-epoch cap."""
    keep = []
    for p in recs:
        sp = p["spec"]
        if not str(sp.get("split", "")).startswith("outer_r"):
            continue
        if int(sp["budget"]) in BUDGETS and int(sp["r"]) < 10 and int(sp["seed"]) < 3 and int(p["max_epochs"]) == 200:
            keep.append(p)
    return keep


def run_row(p: Mapping, backbone: str, design: str, source: str) -> dict:
    sp = p["spec"]
    row = {"backbone": backbone, "design": design, "source": source, "model": MODEL_NAME[sp["model"]],
           "seed": int(sp["seed"]), "run_id": p["run_id"], "split": sp["split"], "B": int(sp["budget"]),
           "gpu_class": gpu_class(p, source), "epochs_run": int(p["epochs_run"]),
           "max_epochs": int(p["max_epochs"]), "mae": float(p["metrics"]["mae"]),
           "within_host_mae": float(p["metrics"]["within_host_mae"]),
           "pred_path": p["predictions"]["path"], "pred_sha256": p["predictions"]["sha256"]}
    row["capped"] = row["epochs_run"] >= row["max_epochs"]
    if design == "precision":
        row.update(r=int(sp["r"]), K=-1, fold=-1)
    else:
        k, f = (int(x) for x in sp["split"].removeprefix("cv_v1_k").split("_f"))
        row.update(r=-1, K=k, fold=f)
    return row


def load_predictions(runs: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Stack the prediction parquets of ``runs``; checks sha256 and the recorded MAE."""
    frames = []
    for row in runs.itertuples(index=False):
        path = root / row.pred_path
        if sha256(path) != row.pred_sha256:
            raise AssertionError(f"{row.run_id}: predictions hash mismatch")
        df = pd.read_parquet(path)[["site_id", "host_id", "y_true", "y_pred"]]
        if abs(float(np.abs(df.y_pred - df.y_true).mean()) - row.mae) > 1e-9:
            raise AssertionError(f"{row.run_id}: recorded MAE differs from the stored predictions")
        frames.append(df.assign(run_id=row.run_id))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- integrity

def check_partition(folds: Sequence[Sequence[str]], all_hosts: Iterable[str]) -> None:
    """The test hosts of the folds are disjoint and cover ``all_hosts`` exactly."""
    seen: set[str] = set()
    for i, f in enumerate(folds):
        fs = set(f)
        if len(fs) != len(f):
            raise AssertionError(f"fold {i}: duplicate host")
        if seen & fs:
            raise AssertionError(f"fold {i}: host in more than one fold")
        seen |= fs
    if seen != set(all_hosts):
        raise AssertionError("folds do not cover the host set")


def check_run(pred: pd.DataFrame, test_sites: set[str], test_hosts: set[str], train_hosts: Iterable[str],
              label: str) -> None:
    """One run: finite, every test site exactly once, no test host in training."""
    if not np.isfinite(pred.y_pred.to_numpy(float)).all():
        raise AssertionError(f"{label}: non-finite predictions")
    if pred.site_id.duplicated().any():
        raise AssertionError(f"{label}: duplicate site")
    if set(pred.site_id) != test_sites:
        raise AssertionError(f"{label}: sites differ from the test set")
    if test_hosts & set(train_hosts):
        raise AssertionError(f"{label}: test host in training")


def check_targets(pred: pd.DataFrame, uni: pd.DataFrame) -> None:
    u = uni.set_index("site_id")
    if not np.allclose(pred.y_true.to_numpy(), u.target_Ef_eV.loc[pred.site_id].to_numpy(), atol=1e-9):
        raise AssertionError("targets differ from the universe")
    if not (pred.host_id.to_numpy() == u.host_id.loc[pred.site_id].to_numpy()).all():
        raise AssertionError("hosts differ from the universe")


def check_unique_cells(runs: pd.DataFrame) -> None:
    keys = ["backbone", "design", "model", "r", "K", "fold", "B", "seed"]
    keys = [k for k in keys if k in runs]
    dup = runs.duplicated(subset=keys, keep=False)
    if dup.any():
        raise AssertionError(f"more than one run per cell: {runs.loc[dup, 'run_id'].tolist()[:6]}")


def check_counts(runs: pd.DataFrame) -> dict:
    out = {}
    for bb in BACKBONES:
        for design, n_new in EXPECTED_NEW.items():
            sel = runs[(runs.backbone == bb) & (runs.design == design)]
            got = sel.source.value_counts().to_dict()
            want = {design: n_new, **({"sweep": EXPECTED_SWEEP} if design == "precision" else {})}
            if got != want:
                raise AssertionError(f"{bb} {design}: run counts {got}, expected {want}")
            out[f"{bb}|{design}"] = got
    return out


def precision_cells_complete(runs: pd.DataFrame) -> None:
    for bb in BACKBONES:
        sel = runs[(runs.backbone == bb) & (runs.design == "precision")]
        cells = set(zip(sel.model, sel.B, sel.r, sel.seed, strict=True))
        want = {(m, B, r, s) for m in ("S", "D") for B in BUDGETS for r in range(30) for s in range(5)}
        if cells != want:
            raise AssertionError(f"{bb} precision: cells missing {sorted(want - cells)[:5]} extra "
                                 f"{sorted(cells - want)[:5]}")


def cv_cells_complete(runs: pd.DataFrame, cv: Mapping) -> None:
    for bb in BACKBONES:
        sel = runs[(runs.backbone == bb) & (runs.design == "cv")]
        cells = set(zip(sel.model, sel.K, sel.fold, sel.seed, strict=True))
        want = {(m, k, f, s) for m in ("S", "D") for k in KS for f in range(k) for s in range(3)}
        if cells != want:
            raise AssertionError(f"{bb} cv: cells differ from the design")
        for row in sel.itertuples(index=False):
            fd = cv["by_k"][str(row.K)]["folds"][row.fold]
            if row.B != fd["n_train"]:
                raise AssertionError(f"{row.run_id}: budget {row.B} != fold training size {fd['n_train']}")


# ---------------------------------------------------------------- build

def build(uni: pd.DataFrame, root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    from dftgnn.split import budget_train, load_split

    rows = []
    for bb in BACKBONES:
        src = SOURCES[bb]
        sweep_dir = root / "results" / src["sweep"] if src["sweep"] else root / "results"
        sweep = select_sweep(read_records(sweep_dir, src["sweep"] or None))
        rows += [run_row(p, bb, "precision", "sweep") for p in sweep]
        for design in ("precision", "cv"):
            recs = read_records(root / "results" / src[design], src[design])
            rows += [run_row(p, bb, design, design) for p in recs]
    runs = pd.DataFrame(rows)
    counts = check_counts(runs)
    check_unique_cells(runs)
    precision_cells_complete(runs)
    cv = load_split("cv_v1")
    cv_cells_complete(runs, cv)

    all_hosts = sorted(uni.host_id.unique())
    if len(all_hosts) != N_HOSTS:
        raise AssertionError(f"universe holds {len(all_hosts)} hosts")
    for k in KS:
        check_partition([f["test"] for f in cv["by_k"][str(k)]["folds"]], all_hosts)

    sites_of = uni.groupby("host_id").site_id.apply(set).to_dict()

    def test_sites(hosts):
        return set().union(*(sites_of[h] for h in hosts))

    preds = load_predictions(runs, root)
    check_targets(preds, uni)
    by_run = dict(tuple(preds.groupby("run_id", sort=False)))
    outer = {r: load_split(f"outer_r{r}") for r in range(30)}
    n_checked = 0
    for row in runs.itertuples(index=False):
        if row.design == "precision":
            sp = outer[row.r]
            th, train = set(sp["test"]), budget_train(sp, row.B)
        else:
            fd = cv["by_k"][str(row.K)]["folds"][row.fold]
            th, train = set(fd["test"]), fd["train"]
        check_run(by_run[row.run_id], test_sites(th), th, train, f"{row.backbone} {row.run_id}")
        n_checked += 1

    meta = runs.set_index("run_id")
    long = preds.join(meta[["backbone", "design", "model", "r", "K", "fold", "B", "seed"]], on="run_id")
    prec = long[long.design == "precision"][["backbone", "model", "r", "B", "seed", "site_id", "host_id",
                                             "y_true", "y_pred", "run_id"]]
    cvl = long[long.design == "cv"][["backbone", "model", "K", "fold", "B", "seed", "site_id", "host_id",
                                     "y_true", "y_pred", "run_id"]]
    prec = prec.sort_values(["backbone", "model", "B", "r", "seed", "site_id"]).reset_index(drop=True)
    cvl = cvl.sort_values(["backbone", "model", "K", "fold", "seed", "site_id"]).reset_index(drop=True)
    runs = runs.drop(columns=["pred_path"]).sort_values(
        ["backbone", "design", "model", "B", "r", "K", "fold", "seed"]).reset_index(drop=True)
    info = {"counts": counts, "runs_checked": n_checked,
            "fold_partition": {str(k): {"n_folds": k, "partition_of_818_hosts": True,
                                        "n_test_hosts": [len(f["test"]) for f in cv["by_k"][str(k)]["folds"]],
                                        "n_train_hosts": [f["n_train"] for f in cv["by_k"][str(k)]["folds"]]}
                               for k in KS}}
    return prec, cvl, runs, info


def save(df: pd.DataFrame, name: str, root: Path) -> dict:
    out = root / "results" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.parquet"
    df.to_parquet(path, index=False)
    size = path.stat().st_size
    return {"path": str(path.relative_to(root)), "sha256": sha256(path), "bytes": size, "rows": len(df),
            "tracked": size < TRACK_LIMIT}


def main() -> None:
    from dftgnn.config import load_config
    from dftgnn.io.results import write_result

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, default=ROOT,
                    help="checkout holding the gitignored data/processed/universe_v1.parquet")
    a = ap.parse_args()
    cfg = load_config()
    uni_path = a.data_root / "data" / "processed" / "universe_v1.parquet"
    uni = pd.read_parquet(uni_path, columns=["site_id", "host_id", "target_Ef_eV"])
    prec, cvl, runs, info = build(uni, ROOT)
    files = {"precision_eval_long": save(prec, "precision_eval_long", ROOT),
             "cv_eval_long": save(cvl, "cv_eval_long", ROOT),
             "precision_cv_runs": save(runs, "precision_cv_runs", ROOT)}
    payload = {
        "label": "EXTENSION (v2: primary backbone, POST HOC; v1: originally registered backbone)",
        "rules": "docs/deviations.md rows 2026-10-11 (§7, §11) and 2026-10-14 (§7, §11), item a",
        "files": files,
        "universe": {"path": "data/processed/universe_v1.parquet", "sha256": sha256(uni_path)},
        "sources": SOURCES,
        "integrity": {**info, "assertions": [
            "every run covers the test sites of its split exactly once", "all predictions finite",
            "no test host among the training hosts", "targets and hosts equal the universe",
            "prediction sha256 equals the record", "recorded MAE equals the MAE of the stored predictions",
            "one run per (backbone, model, cell, seed)", "the folds of each K partition the 818 hosts",
            "run counts: precision 480 + 480 new and 120 + 120 sweep; cv 180 + 180"], "passed": True},
        "n_rows": {"precision": len(prec), "cv": len(cvl)},
        "run_ids": {bb: sorted(runs.run_id[runs.backbone == bb].tolist()) for bb in BACKBONES},
    }
    print(json.dumps({k: v for k, v in payload.items() if k != "run_ids"}, indent=1))
    print(write_result("precision_cv_eval_table", payload, config=cfg, repo_root=ROOT))


if __name__ == "__main__":
    main()
