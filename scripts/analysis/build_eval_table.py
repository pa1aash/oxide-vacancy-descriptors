"""The G1 evaluation table.

Reads the predictions of the sweep runs of S and D (D-state) and of the 18 epoch-cap runs, located through
their result JSONs. P1 records are skipped on their spec before any prediction file is opened. Writes

    results/analysis/eval_table.parquet     model, r, B, site_id, host_id, y_pred (seed ensemble), y_true,
                                            y_pred_seed0..2 (S and D only); S, D, B0, RF-*, host-mean oracle
    results/analysis/eval_capsens.parquet   same columns, S and D at B = 654, r = 0-2, cap 200 and cap 600
    results/g1_eval_table.json              hashes, sizes and the integrity and sanity checks

Integrity assertions (any failure raises): each (model, r, B, seed) covers exactly the test sites of resample
r once; no test host is in the training hosts of (r, B); predictions finite; targets equal the universe
targets; three seeds per cell. Sanity: the mean over resamples and seeds of the single-seed MAE per (model, B)
reproduces the table (`results/sweep_completeness.json`) to 3 decimals.

    python scripts/analysis/build_eval_table.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dftgnn.config import load_config
from dftgnn.data.universe import read_universe
from dftgnn.io.results import write_result
from dftgnn.split import budget_train, load_split

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "analysis"
MODEL_NAME = {"S": "S", "D-state": "D"}
BASELINES = {"B0": "b0_physics_floor", "RF-Kumagai": "rf_kumagai", "RF-electronic": "rf_electronic",
             "RF-structural": "rf_structural"}
TRACK_LIMIT = 20 * 1024**2


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run_records() -> list[dict]:
    """S and D-state records of the sweep (outer splits) and of the epoch-cap stage; P1 never opened."""
    recs = []
    for sub in ("", "capsens"):
        for f in sorted((ROOT / "results" / sub).glob("*.json")):
            try:
                pay = json.loads(f.read_text()).get("payload")
            except ValueError:
                continue
            if not isinstance(pay, dict) or "spec" not in pay or pay.get("smoke"):
                continue
            sp = pay["spec"]
            if sp.get("model") not in MODEL_NAME or not str(sp.get("split", "")).startswith("outer_r"):
                continue
            if sp.get("tags", {}).get("pilot") or sp.get("results_subdir") != (sub or None):
                continue
            pay["_stage"] = "capsens" if sub else "sweep"
            recs.append(pay)
    return recs


def load_runs(recs: list[dict]) -> pd.DataFrame:
    frames = []
    for p in recs:
        sp, pr = p["spec"], p["predictions"]
        path = ROOT / pr["path"]
        assert sha(path) == pr["sha256"], f"{p['run_id']}: predictions hash mismatch"
        df = pd.read_parquet(path)[["site_id", "host_id", "y_true", "y_pred"]]
        assert abs(np.abs(df.y_pred - df.y_true).mean() - p["metrics"]["mae"]) < 1e-9, \
            f"{p['run_id']}: recorded MAE differs from the stored predictions"
        df = df.assign(model=MODEL_NAME[sp["model"]], r=int(sp["r"]), B=int(sp["budget"]), seed=int(sp["seed"]),
                       cap=int(p["max_epochs"]), stage=p["_stage"], run_id=p["run_id"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def check_integrity(runs: pd.DataFrame, uni: pd.DataFrame, splits: dict[int, dict]) -> dict:
    target = uni.set_index("site_id").target_Ef_eV
    site_host = uni.set_index("site_id").host_id
    checks = {"cells": 0}
    assert np.isfinite(runs.y_pred.to_numpy()).all(), "non-finite predictions"
    for (stage, model, r, B), g in runs.groupby(["stage", "model", "r", "B"]):
        test_hosts = set(splits[r]["test"])
        test_sites = set(uni.site_id[uni.host_id.isin(test_hosts)])
        assert not test_hosts & set(budget_train(splits[r], B)), f"r={r} B={B}: test host in training"
        assert sorted(g.seed.unique()) == [0, 1, 2], f"{stage} {model} r={r} B={B}: seeds {sorted(g.seed.unique())}"
        for s, gs in g.groupby("seed"):
            assert gs.run_id.nunique() == 1, f"{stage} {model} r={r} B={B} s={s}: more than one run"
            assert not gs.site_id.duplicated().any(), f"{stage} {model} r={r} B={B} s={s}: duplicate site"
            assert set(gs.site_id) == test_sites, f"{stage} {model} r={r} B={B} s={s}: sites differ from test set"
            checks["cells"] += 1
        assert np.allclose(g.y_true.to_numpy(), target.loc[g.site_id].to_numpy(), atol=1e-9), "targets differ"
        assert (g.host_id.to_numpy() == site_host.loc[g.site_id].to_numpy()).all(), "hosts differ"
    return checks


def ensemble(runs: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "r", "B", "cap", "site_id"]
    wide = runs.pivot_table(index=keys, columns="seed", values="y_pred", aggfunc="first")
    wide.columns = [f"y_pred_seed{s}" for s in wide.columns]
    first = runs.groupby(keys)[["host_id", "y_true"]].first()
    out = first.join(wide).reset_index()
    out["y_pred"] = out[[f"y_pred_seed{s}" for s in (0, 1, 2)]].mean(axis=1)
    return out


def baselines(uni: pd.DataFrame, splits: dict[int, dict], budgets: list[int]) -> tuple[pd.DataFrame, dict]:
    frames, info = [], {}
    for name, stem in BASELINES.items():
        path = ROOT / "results" / "predictions" / f"{stem}.parquet"
        ref = json.loads((ROOT / "results" / f"{stem}.json").read_text())["payload"]["predictions"]
        assert sha(path) == ref["sha256"], f"{name}: predictions hash mismatch"
        df = pd.read_parquet(path)
        df = df[df["resample"] >= 0].rename(columns={"resample": "r", "budget": "B"})
        frames.append(df.assign(model=name)[["model", "r", "B", "site_id", "host_id", "y_true", "y_pred"]])
        info[name] = {"path": str(path.relative_to(ROOT)), "sha256": ref["sha256"]}
    for r, sp in splits.items():   # host-mean oracle: each test site predicted by its host's true mean
        te = uni[uni.host_id.isin(sp["test"])]
        pred = te.groupby("host_id").target_Ef_eV.transform("mean")
        for B in budgets:
            frames.append(pd.DataFrame({"model": "host-mean oracle", "r": r, "B": B, "site_id": te.site_id.to_numpy(),
                                        "host_id": te.host_id.to_numpy(), "y_true": te.target_Ef_eV.to_numpy(),
                                        "y_pred": pred.to_numpy()}))
    out = pd.concat(frames, ignore_index=True)
    target = uni.set_index("site_id").target_Ef_eV
    assert np.isfinite(out.y_pred.to_numpy()).all(), "non-finite baseline predictions"
    assert np.allclose(out.y_true.to_numpy(), target.loc[out.site_id].to_numpy(), atol=1e-9), "baseline targets"
    for (m, r, B), g in out.groupby(["model", "r", "B"]):
        sites = set(uni.site_id[uni.host_id.isin(splits[r]["test"])])
        assert not g.site_id.duplicated().any() and set(g.site_id) == sites, f"{m} r={r} B={B}: test-site coverage"
    return out, info


def sanity(runs: pd.DataFrame) -> dict:
    """Mean over resamples and seeds of the single-seed MAE vs the table (3 decimals)."""
    ref = json.loads((ROOT / "results" / "sweep_completeness.json").read_text())["payload"]["sanity_mean_test_mae"]
    sw = runs[runs.stage == "sweep"]
    per = sw.assign(ae=(sw.y_pred - sw.y_true).abs()).groupby(["model", "B", "r", "seed"]).ae.mean()
    tab = per.groupby(["model", "B"]).mean()
    rows, ok = {}, True
    for (m, B), v in tab.items():
        s09 = ref[f"{'D-state' if m == 'D' else m}|{B}"]["mean_test_mae"]
        match = round(v, 3) == round(s09, 3)
        ok &= match
        rows[f"{m}|{B}"] = {"recomputed": float(v), "s09": float(s09), "match_3dp": bool(match)}
    assert ok, f"S09 sanity table not reproduced: {rows}"
    return rows


def save(df: pd.DataFrame, name: str) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.parquet"
    df.to_parquet(path, index=False)
    size = path.stat().st_size
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path), "bytes": size, "rows": len(df),
            "tracked": size < TRACK_LIMIT}


def main() -> None:
    cfg = load_config()
    uni = read_universe(cfg)
    budgets = list(cfg.budgets.hosts)
    splits = {r: load_split(f"outer_r{r}") for r in range(cfg.split.n_outer_resamples)}
    recs = run_records()
    runs = load_runs(recs)
    n = runs.groupby("stage").run_id.nunique().to_dict()
    assert n == {"sweep": 360, "capsens": 18}, f"run counts {n}"
    checks = check_integrity(runs, uni, splits)
    s09 = sanity(runs)
    ens = ensemble(runs)
    sweep = ens[ens.cap == 200].drop(columns="cap")
    assert set(zip(sweep.model, sweep.B, strict=True)) == {(m, B) for m in ("S", "D") for B in budgets}
    cap = ens[(ens.cap == 600)]
    cap200 = sweep[(sweep.B == 654) & sweep.r.isin(sorted(cap.r.unique()))].assign(cap=200)
    capsens = pd.concat([cap200, cap], ignore_index=True)
    assert sorted(cap.r.unique()) == [0, 1, 2] and set(cap.B) == {654}
    base, base_info = baselines(uni, splits, budgets)
    table = pd.concat([sweep, base], ignore_index=True)[
        ["model", "r", "B", "site_id", "host_id", "y_pred", "y_true", "y_pred_seed0", "y_pred_seed1", "y_pred_seed2"]]
    table = table.sort_values(["model", "r", "B", "site_id"]).reset_index(drop=True)
    capsens = capsens.sort_values(["model", "cap", "r", "site_id"]).reset_index(drop=True)
    files = {"eval_table": save(table, "eval_table"), "eval_capsens": save(capsens, "eval_capsens")}
    payload = {
        "files": files, "baseline_sources": base_info,
        "runs": {"sweep_S_D": n["sweep"], "capsens": n["capsens"],
                 "run_ids": sorted(runs.run_id.unique().tolist())},
        "integrity": {**checks, "assertions": [
            "each (model, r, B, seed) covers exactly the test sites of resample r once",
            "no test host in the training hosts of (r, B)", "predictions finite",
            "targets and hosts equal the universe", "three seeds per (model, r, B)",
            "recorded MAE equals the MAE of the stored predictions"], "passed": True},
        "sanity_s09_single_seed_mean_mae": s09,
        "models": sorted(table.model.unique().tolist()), "budgets": budgets,
        "note": "P1 records were filtered on their spec; no P1 prediction file was opened.",
    }
    print(json.dumps({k: v for k, v in payload.items() if k != "runs"}, indent=1))
    print(write_result("g1_eval_table", payload, config=cfg))


if __name__ == "__main__":
    main()
