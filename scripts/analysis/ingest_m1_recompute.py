"""Recompute the collaborator branch's claims with our analysis code from his stored files.

Reads the collaborator analysis branch from a read-only worktree (``--ingest``), verifies every input
against the sha256 recorded in its run record, and recomputes with ``dftgnn.stats.g1`` (the G1 estimator:
seed-ensemble means, paired hierarchical bootstrap, 2000 replicates, seed 20261008, N* rule):

    v1_arm        A per budget, N*
    v2_arm        the same from his v2 sweep predictions (S-v2, D-v2)
    staging_v2    P-v2 minus S-v2 and D-v2 minus P-v2 per budget (same draws)
    loco          pooled LOCO MAE of S, D, RF-Kumagai and B0, A_LOCO and S minus RF (host bootstrap, v1 and v2)
    screening     out-of-sample S (v1) and S-v2 at B = 654 on the plan's screening candidates
    c0_overlap    official S on Kiyohara test hosts outside / inside the resample-0 a654 tuning pool
    epoch_cap     runs that used every epoch, ours and his, and his retrained twins of our sweep runs
    mlip          host sets relaxed / converged against our tiling map and d-bins; recorded dMAE
    probe         class-mean R^2 rule values recomputed from his stored per-run R^2
    run_counts    records per stage with cell completeness

Nothing is trained or run from his code. Results are written via write_result as ``ingest_m1_recompute``.

    python scripts/analysis/ingest_m1_recompute.py --ingest ../dft-gnn-ingest
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from dftgnn.config import load_config
from dftgnn.data.universe import read_universe
from dftgnn.io.results import write_result
from dftgnn.split import budget_train, load_split
from dftgnn.stats.g1 import (
    BOOT_SEED,
    HierDraws,
    contrast,
    n_star,
    point_metric,
    resample_stats,
    summarize,
)

ROOT = Path(__file__).resolve().parents[2]
CI = 0.95
Q_UP = 0.95


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rnd(x, n=4):
    if isinstance(x, dict):
        return {k: rnd(v, n) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(v, n) for v in x]
    if isinstance(x, (float, np.floating)):
        return round(float(x), n)
    return x


class Branch:
    """Read-only access to the collaborator worktree with input hashing."""

    def __init__(self, path: Path):
        self.root = path.resolve()
        self.head = subprocess.run(["git", "-C", str(self.root), "rev-parse", "HEAD"], capture_output=True,
                                   text=True, check=True).stdout.strip()
        self.inputs: dict[str, str] = {}

    def json(self, rel: str) -> dict:
        p = self.root / rel
        self.inputs[rel] = sha(p)
        return json.loads(p.read_text())

    def records(self, sub: str, models: tuple[str, ...], split_prefix: str) -> list[dict]:
        out = []
        for f in sorted((self.root / "results" / sub).glob("*.json")):
            try:
                pay = json.loads(f.read_text()).get("payload")
            except ValueError:
                continue
            if not isinstance(pay, dict) or "spec" not in pay or pay.get("smoke"):
                continue
            sp = pay["spec"]
            if sp.get("model") not in models or not str(sp.get("split", "")).startswith(split_prefix):
                continue
            if sp.get("tags", {}).get("pilot") or (sp.get("results_subdir") or "") != sub:
                continue
            self.inputs[str(f.relative_to(self.root))] = sha(f)
            out.append(pay)
        return out

    def predictions(self, recs: list[dict], name: dict[str, str]) -> pd.DataFrame:
        frames = []
        for p in recs:
            sp, pr = p["spec"], p["predictions"]
            path = self.root / pr["path"]
            h = sha(path)
            assert h == pr["sha256"], f"{p['run_id']}: predictions hash differs from its record"
            self.inputs[pr["path"]] = h
            df = pd.read_parquet(path)[["site_id", "host_id", "y_true", "y_pred"]]
            assert abs((df.y_pred - df.y_true).abs().mean() - p["metrics"]["mae"]) < 1e-9, \
                f"{p['run_id']}: recorded MAE differs from the stored predictions"
            frames.append(df.assign(model=name[sp["model"]], r=int(sp["r"]), B=int(sp["budget"]),
                                    seed=int(sp["seed"]), run_id=p["run_id"]))
        return pd.concat(frames, ignore_index=True)


def check_outer(runs: pd.DataFrame, uni: pd.DataFrame, splits: dict[int, dict], seeds=(0, 1, 2)) -> int:
    """Each (model, r, B, seed) covers exactly the test sites of resample r once; targets match the universe."""
    target = uni.set_index("site_id").target_Ef_eV
    cells = 0
    for (m, r, B), g in runs.groupby(["model", "r", "B"]):
        sites = set(uni.site_id[uni.host_id.isin(splits[r]["test"])])
        assert not set(splits[r]["test"]) & set(budget_train(splits[r], B)), f"r={r} B={B}: test host in training"
        assert sorted(g.seed.unique()) == list(seeds), f"{m} r={r} B={B}: seeds {sorted(g.seed.unique())}"
        for s, gs in g.groupby("seed"):
            assert gs.run_id.nunique() == 1 and not gs.site_id.duplicated().any(), f"{m} r={r} B={B} s={s}"
            assert set(gs.site_id) == sites, f"{m} r={r} B={B} s={s}: sites differ from the test set"
            cells += 1
        assert np.allclose(g.y_true.to_numpy(), target.loc[g.site_id].to_numpy(), atol=1e-9), "targets differ"
    return cells


def ensemble(runs: pd.DataFrame, keys=("model", "r", "B", "site_id")) -> pd.DataFrame:
    g = runs.groupby(list(keys), sort=True)
    return g.agg(host_id=("host_id", "first"), y_true=("y_true", "first"), y_pred=("y_pred", "mean"),
                 n_seeds=("seed", "nunique")).reset_index()


def stats_by(ens: pd.DataFrame, model: str, budgets, resamples) -> dict[int, list[np.ndarray]]:
    sub = ens[ens.model == model]
    return {B: [resample_stats(sub[(sub.B == B) & (sub.r == r)]) for r in resamples] for B in budgets}


def arm(ens: pd.DataFrame, a: str, b: str, budgets, resamples, draws: HierDraws, metric="mae") -> dict:
    c = contrast(stats_by(ens, a, budgets, resamples), stats_by(ens, b, budgets, resamples), draws, metric,
                 ci=CI, upper_quantile=Q_UP)
    c90 = contrast(stats_by(ens, a, budgets, resamples), stats_by(ens, b, budgets, resamples), draws, metric,
                   ci=0.90, upper_quantile=Q_UP)
    out = {}
    for B in budgets:
        out[str(B)] = {"mae_a": c[B]["a"]["mean"], "mae_b": c[B]["b"]["mean"], "diff": c[B]["diff"]["mean"],
                       "ci95": c[B]["diff"]["ci"], "ci90": c90[B]["diff"]["ci"],
                       "upper_one_sided_95": c[B]["diff"]["upper_one_sided"],
                       "per_resample": c[B]["diff"]["per_resample"]}
    return out


def compare(ours: dict, his: dict, keys: dict[str, str]) -> dict:
    """Max absolute differences between our and his values per field, over budgets."""
    out = {}
    for ok, hk in keys.items():
        d = []
        for B, v in ours.items():
            hv = his[B][hk]
            ov = v[ok]
            d.append(np.max(np.abs(np.subtract(ov, hv))))
        out[f"{ok}_vs_{hk}"] = float(np.max(d))
    return out


# ------------------------------------------------------------------------------------------- sections

def v1_arm(br: Branch, uni, splits, budgets, R, draws) -> dict:
    recs = br.records("", ("S", "D-state"), "outer_r")
    runs = br.predictions(recs, {"S": "S", "D-state": "D"})
    assert runs.run_id.nunique() == 360, runs.run_id.nunique()
    identical = sum(sha(br.root / p["predictions"]["path"]) == sha(ROOT / p["predictions"]["path"]) for p in recs)
    cells = check_outer(runs, uni, splits)
    ens = ensemble(runs)
    a = arm(ens, "S", "D", budgets, range(R), draws)
    ns = n_star({int(B): v["upper_one_sided_95"] for B, v in a.items()}, 0.05)
    g1 = json.loads((ROOT / "results" / "g1_primary.json").read_text())["payload"]
    g1_eq = max(abs(a[B]["diff"] - g1["per_budget"][B]["A"]["mean"]) +
                abs(a[B]["upper_one_sided_95"] - g1["per_budget"][B]["A"]["upper_one_sided"]) +
                float(np.max(np.abs(np.subtract(a[B]["ci95"], g1["per_budget"][B]["A"]["ci"])))) for B in a)
    his = br.json("results/c1c2_v1.json")["payload"]
    hk = {B: {"A": v["A"], "ci95": v["ci95"], "up": v["upper_one_sided_95"]} for B, v in his["A_mae"].items()}
    return {"runs": int(runs.run_id.nunique()), "cells_checked": cells,
            "prediction_files_identical_to_main": int(identical), "per_budget": a, "n_star": ns,
            "c2": {"A": a["654"]["diff"], "ci95": a["654"]["ci95"]},
            "equals_g1_primary_max_abs": g1_eq,
            "his": {"per_budget": hk, "n_star": his["N_star"], "c2": his["C2"]},
            "max_abs_diff_vs_his": compare(a, hk, {"diff": "A", "ci95": "ci95", "upper_one_sided_95": "up"})}


def v2_arm(br: Branch, uni, splits, budgets, R, draws) -> tuple[dict, pd.DataFrame]:
    recs = br.records("v2_sweep", ("S", "D-state"), "outer_r")
    runs = br.predictions(recs, {"S": "S", "D-state": "D"})
    cells = check_outer(runs, uni, splits)
    ens = ensemble(runs)
    a = arm(ens, "S", "D", budgets, range(R), draws)
    ns = n_star({int(B): v["upper_one_sided_95"] for B, v in a.items()}, 0.05)
    his = br.json("results/c1c2_v2.json")["payload"]
    hk = {B: {"A": v["A"], "ci95": v["ci95"], "up": v["upper_one_sided_95"]} for B, v in his["A_mae"].items()}
    capped = sum(int(p["epochs_run"] >= p["max_epochs"]) for p in recs)
    return ({"runs": int(runs.run_id.nunique()), "cells_checked": cells, "per_budget": a, "n_star": ns,
             "c2": {"A": a["654"]["diff"], "ci95": a["654"]["ci95"]},
             "his": {"per_budget": hk, "n_star": his["N_star"], "c2": his["C2"]},
             "max_abs_diff_vs_his": compare(a, hk, {"diff": "A", "ci95": "ci95", "upper_one_sided_95": "up"}),
             "runs_at_epoch_cap": capped}, ens)


def staging(br: Branch, uni, splits, budgets, R, draws, ens_v2: pd.DataFrame) -> dict:
    recs = br.records("p_v2", ("P",), "outer_r")
    runs = br.predictions(recs, {"P": "P"})
    cells = check_outer(runs, uni, splits)
    ens = pd.concat([ens_v2, ensemble(runs)], ignore_index=True)
    ps = arm(ens, "P", "S", budgets, range(R), draws)
    dp = arm(ens, "D", "P", budgets, range(R), draws)
    his = br.json("results/c3a_v2.json")["payload"]["budgets"]
    out = {}
    for B in map(str, budgets):
        lo95, hi95 = ps[B]["ci95"]
        out[B] = {"P_minus_S": ps[B]["diff"], "P_minus_S_ci95": ps[B]["ci95"], "P_minus_S_ci90": ps[B]["ci90"],
                  "P_better_95": bool(hi95 < 0), "P_worse_95": bool(lo95 > 0),
                  "D_minus_P": dp[B]["diff"], "D_minus_P_ci95": dp[B]["ci95"],
                  "mae_P": ps[B]["mae_a"], "mae_S": ps[B]["mae_b"],
                  "his_P_minus_S": his[B]["P_minus_S"], "his_D_minus_P": his[B]["D_minus_P"]}
    return {"runs": int(runs.run_id.nunique()), "cells_checked": cells, "per_budget": out,
            "any_budget_P_better_than_S_95": any(v["P_better_95"] for v in out.values())}


def loco(br: Branch, uni, sub: str, rf: pd.DataFrame | None, b0: pd.DataFrame | None) -> dict:
    folds = load_split("loco_v1")["folds"]
    recs = br.records(sub, ("S", "D-state"), "loco/")
    runs = br.predictions(recs, {"S": "S", "D-state": "D"})
    runs["r"] = runs.run_id.map({p["run_id"]: int(p["spec"]["split"].split("_f")[1]) for p in recs})
    for (m, k), g in runs.groupby(["model", "r"]):
        sites = set(uni.site_id[uni.host_id.isin(folds[k]["test"])])
        assert sorted(g.seed.unique()) == [0, 1, 2]
        for _, gs in g.groupby("seed"):
            assert set(gs.site_id) == sites and not gs.site_id.duplicated().any(), f"{sub} {m} fold {k}"
    ens = ensemble(runs)
    pooled = {m: ens[ens.model == m].sort_values("site_id") for m in ("S", "D")}
    assert len(pooled["S"]) == len(uni) and pooled["S"].host_id.nunique() == uni.host_id.nunique()
    frames = dict(pooled)
    if rf is not None:
        frames["RF-Kumagai"] = rf.sort_values("site_id")
        frames["B0"] = b0.sort_values("site_id")
    for m, f in frames.items():
        assert list(f.site_id) == list(pooled["S"].site_id), m
    st = {m: [resample_stats(f)] for m, f in frames.items()}
    draws = HierDraws.make([len(st["S"][0])], 2000, BOOT_SEED)
    rep = {m: draws.replicates(s, "mae") for m, s in st.items()}
    pt = {m: point_metric(s[0], "mae") for m, s in st.items()}
    out = {"runs": int(runs.run_id.nunique()), "n_hosts": len(st["S"][0]), "n_sites": len(pooled["S"]),
           "bootstrap": "host bootstrap of the pooled folds (HierDraws with one resample), 2000, seed 20261008",
           "mae": {m: summarize([pt[m]], rep[m], ci=CI, upper_quantile=Q_UP) for m in frames},
           "A": summarize([pt["S"] - pt["D"]], rep["S"] - rep["D"], ci=CI, upper_quantile=Q_UP),
           "per_fold_mae": {m: [float((g.y_pred - g.y_true).abs().mean()) for _, g in
                                ens[ens.model == m].groupby("r")] for m in ("S", "D")}}
    if rf is not None:
        out["S_minus_RF"] = summarize([pt["S"] - pt["RF-Kumagai"]], rep["S"] - rep["RF-Kumagai"], ci=CI,
                                      upper_quantile=Q_UP)
        out["per_fold_mae"]["RF-Kumagai"] = [float((g.y_pred - g.y_true).abs().mean()) for _, g in rf.groupby("fold")]
    return out


def loco_baselines(br: Branch) -> tuple[pd.DataFrame, pd.DataFrame]:
    rec = br.json("results/loco_baselines.json")["payload"]["models"]
    out = []
    for stem in ("rf_kumagai", "b0_physics_floor"):
        ref = rec[stem]["predictions"]
        p = br.root / ref["path"]
        assert sha(p) == ref["sha256"], stem
        br.inputs[ref["path"]] = ref["sha256"]
        out.append(pd.read_parquet(p))
    return out[0], out[1]


def screening(br: Branch, uni, cfg, ens_v1: pd.DataFrame, ens_v2: pd.DataFrame) -> dict:
    cand = list(cfg.screening.candidates)
    hosts = uni.drop_duplicates("host_id").set_index("host_id").reduced_formula
    present = {c: sorted(hosts.index[hosts == c]) for c in cand}
    out = {"present": {c: v for c, v in present.items() if v}, "absent": sorted(c for c, v in present.items() if not v)}
    keep = {h for v in present.values() for h in v}
    res = {}
    for name, ens in (("S", ens_v1), ("S_v2", ens_v2)):
        e = ens[(ens.model == "S") & (ens.B == 654) & ens.host_id.isin(keep)]
        per = e.groupby("site_id").agg(host_id=("host_id", "first"), y_true=("y_true", "first"),
                                       y_pred=("y_pred", "mean"), n_resamples=("r", "nunique"))
        res[name] = {"mae": float((per.y_pred - per.y_true).abs().mean()), "n_sites": len(per),
                     "n_hosts": int(per.host_id.nunique()),
                     "candidates": sorted({hosts[h] for h in per.host_id})}
        if name == "S":
            out["sites"] = sorted(per.index)
    his = br.json("results/screening_c5.json")["payload"]
    out.update(res)
    out["his"] = {"mae_S": his["mae_S"], "mae_S_v2": his["mae_S_v2"], "n_sites": len(his["sites"]),
                  "sites": sorted(s["site_id"] for s in his["sites"]), "missing": sorted(his["missing"])}
    out["same_sites_as_his"] = out["sites"] == out["his"]["sites"]
    return out


def c0_overlap(br: Branch, uni) -> dict:
    recs = br.records("c0_official", ("S",), "kiyohara")
    for p in recs:
        assert sha(br.root / p["predictions"]["path"]) == sha(ROOT / p["predictions"]["path"]), "C0 file differs"
    runs = br.predictions(recs, {"S": "S"})
    assert sorted(runs.seed.unique()) == [0, 1, 2] and runs.run_id.nunique() == 3
    ens = ensemble(runs).sort_values("site_id")
    kiy = load_split("kiyohara")
    assert set(ens.host_id) == set(kiy["test"])
    pool = set(budget_train(load_split("outer_r0"), 654))
    out = {}
    for name, sel in (("all", ens), ("not_in_a654_tuning_pool", ens[~ens.host_id.isin(pool)]),
                      ("in_a654_tuning_pool", ens[ens.host_id.isin(pool)])):
        st = resample_stats(sel)
        dr = HierDraws.make([len(st)], 2000, BOOT_SEED)
        out[name] = {"n_hosts": int(sel.host_id.nunique()), "n_sites": len(sel),
                     "mae": summarize([point_metric(st, "mae")], dr.replicates([st], "mae"), ci=CI)}
    out["note"] = "tuning sets are nested (a50 within a200 within a654 of resample 0); the a654 pool covers all"
    return out


def epoch_cap(br: Branch) -> dict:
    ours = []
    for f in sorted((ROOT / "results").glob("*.json")):
        try:
            pay = json.loads(f.read_text()).get("payload")
        except ValueError:
            continue
        if not isinstance(pay, dict) or "spec" not in pay or pay.get("smoke"):
            continue
        sp = pay["spec"]
        if sp.get("model") in ("S", "D-state") and str(sp.get("split")).startswith("outer_r") \
                and not sp.get("results_subdir"):
            ours.append(pay)
    tab = pd.DataFrame([{"model": p["spec"]["model"], "r": p["spec"]["r"], "B": p["spec"]["budget"],
                         "seed": p["spec"]["seed"], "epochs_run": p["epochs_run"], "best_epoch": p["best_epoch"],
                         "max_epochs": p["max_epochs"], "run_id": p["run_id"], "mae": p["metrics"]["mae"],
                         "hp": json.dumps(p["spec"]["hp"], sort_keys=True), "lr": p["spec"]["lr"],
                         "wd": p["spec"]["weight_decay"], "bs": p["spec"]["batch_size"]} for p in ours])
    assert len(tab) == 360
    tab["capped"] = tab.epochs_run >= tab.max_epochs
    by = tab.groupby(["model", "B"]).capped.sum()
    at654 = tab[(tab.B == 654) & tab.capped].sort_values(["model", "r", "seed"])
    out = {"ours_capped_by_model_budget": {f"{m}|{B}": int(v) for (m, B), v in by.items()},
           "ours_capped_at_654": [f"{m} r{r} s{s} (best epoch {b})" for m, r, s, b in
                                  zip(at654.model, at654.r, at654.seed, at654.best_epoch, strict=True)],
           "ours_S_654_capped_of_30": int(at654[at654.model == "S"].shape[0])}
    twins = []
    for sub, model, sel in (("diag_cross", "S", lambda sp: sp["tags"].get("anchor") == 654 and sp["budget"] == 654),
                            ("d_ablation", "D-state", lambda sp: sp.get("ablate") is None and sp["budget"] == 654)):
        for p in br.records(sub, (model,), "outer_r"):
            sp = p["spec"]
            if not sel(sp):
                continue
            t = tab[(tab.model == model) & (tab.r == sp["r"]) & (tab.B == sp["budget"]) & (tab.seed == sp["seed"])]
            assert len(t) == 1
            t = t.iloc[0]
            twins.append({"stage": sub, "model": model, "r": sp["r"], "seed": sp["seed"], "his_run": p["run_id"],
                          "our_run": t.run_id, "same_hparams": bool(
                              json.dumps(sp["hp"], sort_keys=True) == t.hp and sp["lr"] == t.lr
                              and sp["weight_decay"] == t.wd and sp["batch_size"] == t.bs),
                          "his_epochs_run": p["epochs_run"], "our_epochs_run": int(t.epochs_run),
                          "his_best_epoch": p["best_epoch"], "our_best_epoch": int(t.best_epoch),
                          "his_mae": p["metrics"]["mae"], "our_mae": float(t.mae)})
    out["retrained_twins"] = twins
    caps = {}
    for sub in ("diag_cross", "d_ablation", "dlate", "loco", "v2_sweep", "v2_kiyohara", "loco_v2", "p1_v2",
                "v2_screen"):
        recs = []
        for f in sorted((br.root / "results" / sub).glob("*.json")):
            pay = json.loads(f.read_text()).get("payload", {})
            if "epochs_run" in pay:
                recs.append(pay)
        caps[sub] = {"runs": len(recs), "at_cap": sum(int(p["epochs_run"] >= p["max_epochs"]) for p in recs)}
    out["his_runs_at_cap"] = caps
    return out


def mlip(br: Branch, uni, splits) -> dict:
    hosts = {}
    for k in range(4):
        rel = f"results/mlip/relaxed_shard{k}.json.gz"
        br.inputs[rel] = sha(br.root / rel)
        with gzip.open(br.root / rel) as fh:
            hosts.update(json.load(fh)["hosts"])
    conv = {h for h, v in hosts.items() if v.get("converged")}
    inp = br.json("results/mlip_inputs.json")["payload"]
    til = pd.read_csv(ROOT / "data" / "tiling_summary_v1.csv").set_index("host_id")
    dbin = pd.read_csv(ROOT / "data" / "host_dbins_v1.csv").set_index("host_id")
    form = uni.drop_duplicates("host_id").set_index("host_id").formula
    ours_ok = set(til.index[til.status == "ok"])
    ours_fail = set(til.index[til.status != "ok"])
    his_ex = set(inp["excluded"])
    unassigned = set(dbin.index[dbin.bin == "unassigned"])
    tested = set().union(*(set(s["test"]) for s in splits.values()))
    ev = br.json("results/mlip_eval_v2.json")["payload"]
    dmae = {f"{k}|{c}": ev["dMAE"][k][c]["mean"] for k in ev["dMAE"] for c in ("mlip", "rescaled")}
    analysed = conv & tested
    ours_bins = dbin.loc[sorted(analysed)].bin.value_counts().to_dict()
    his_bins = {}
    for key, v in ev["per_host"].items():
        his_bins[key] = {b: x["n_hosts"] for b, x in v["by_d_class"].items()}
    def fl(s):
        return {h: form[h] for h in sorted(s)}
    return {"relaxed": len(hosts), "converged": len(conv), "his_mapped": inp["n_mapped"],
            "not_converged": fl(set(hosts) - conv),
            "ours_tiling_ok": len(ours_ok), "ours_tiling_failed": len(ours_fail),
            "his_excluded": fl(his_ex), "his_excluded_status_in_our_tiling": {h: til.status[h] for h in sorted(his_ex)},
            "ours_failed_but_his_relaxed": fl(ours_fail & set(hosts)),
            "ours_failed_and_his_excluded": fl(ours_fail & his_ex),
            "his_excluded_but_ours_ok": fl(his_ex & ours_ok),
            "unassigned_oxidation": fl(unassigned), "unassigned_in_his_relaxed": fl(unassigned & set(hosts)),
            "unassigned_in_his_excluded": fl(unassigned & his_ex),
            "dMAE_recorded": dmae, "max_abs_dMAE_recorded": max(abs(v) for v in dmae.values()),
            "n_hosts_mlip_recorded": ev["n_hosts_mlip"],
            "dbins_ours_on_converged_and_ever_tested": {"n_hosts": len(analysed), "counts": ours_bins},
            "dbins_his_recorded": his_bins}


def probe(br: Branch) -> dict:
    p = br.json("results/probe_v2_sweep.json")["payload"]
    rows = []
    for r in p["runs"]:
        for kind in ("trained", "random", "shuffled"):
            rows.append({"B": r["B"], "r": r["r"], "kind": kind, **{c: r[kind]["class_mean"][c] for c in
                                                                     ("all", "host", "site")}})
    df = pd.DataFrame(rows)
    out = {"runs": len(p["runs"]), "budgets": sorted(df.B.unique().tolist())}
    for c in ("all", "host", "site"):
        tr = df[df.kind == "trained"]
        rho = float(spearmanr(tr.B, tr[c]).statistic)
        diffs = {}
        for B in (200, 400, 654):
            w = df[df.B == B].pivot(index="r", columns="kind", values=c)
            diffs[f"{B}|random"] = float((w.trained - w.random).mean())
            diffs[f"{B}|shuffled"] = float((w.trained - w.shuffled).mean())
        out[c] = {"spearman_rho_point": rho, "trained_minus_control_mean_over_resamples": diffs,
                  "his_recorded": {"spearman_rho": p["C4"][c]["spearman_rho"], "C4_holds": p["C4"][c]["C4_holds"],
                                   "beats_controls": {k: v["mean"] for k, v in p["C4"][c]["beats_controls"].items()}}}
    return out


def run_counts(br: Branch) -> dict:
    out = {}
    for sub in ("diag_cross", "d_ablation", "dlate", "loco", "v2_screen", "v2_sweep", "v2_kiyohara", "loco_v2",
                "p1_v2", "p_v2"):
        recs = [json.loads(f.read_text())["payload"] for f in sorted((br.root / "results" / sub).glob("*.json"))]
        cells = pd.DataFrame([{"model": q["spec"]["model"], "split": q["spec"]["split"], "B": q["spec"]["budget"],
                               "seed": q["spec"]["seed"], "tag": json.dumps(q["spec"].get("tags"), sort_keys=True),
                               "ablate": q["spec"].get("ablate")} for q in recs])
        key = ["model", "split", "B", "tag", "ablate"]
        per = cells.groupby(key, dropna=False).seed.nunique()
        out[sub] = {"records": len(recs), "unique_run_ids": len({q["run_id"] for q in recs}),
                    "cells": len(per), "cells_with_3_seeds": int((per == 3).sum()),
                    "eval_test_false": sum(int(not q["spec"].get("eval_test", True)) for q in recs),
                    "test_hosts_zero": sum(int((q.get("n_hosts") or {}).get("test", -1) == 0) for q in recs)}
    tun = {}
    for f in sorted((br.root / "results" / "tuning").glob("tuning_v2_*.json")):
        q = json.loads(f.read_text())["payload"]
        tun[f.stem] = {"n_complete": q.get("n_complete"), "test_hosts_loaded": q.get("test_hosts_loaded"),
                       "arch": q.get("arch"),
                       "trials_reaching_cap": q.get("trials_reaching_cap")}
    out["tuning_v2"] = tun
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ingest", required=True, help="read-only worktree of the collaborator branch")
    a = ap.parse_args()
    br = Branch(Path(a.ingest))
    cfg = load_config()
    uni = read_universe(cfg)
    budgets = list(cfg.budgets.hosts)
    R = int(cfg.split.n_outer_resamples)
    splits = {r: load_split(f"outer_r{r}") for r in range(R)}
    draws = HierDraws.make([len(splits[r]["test"]) for r in range(R)], int(cfg.analysis.bootstrap.draws), BOOT_SEED)

    v1 = v1_arm(br, uni, splits, budgets, R, draws)
    recs = br.records("", ("S",), "outer_r")
    ens_v1 = ensemble(br.predictions(recs, {"S": "S"}))
    v2, ens_v2 = v2_arm(br, uni, splits, budgets, R, draws)
    rf, b0 = loco_baselines(br)
    payload = {
        "source": {"branch_head": br.head, "worktree": "read-only ingest worktree of the collaborator branch"},
        "estimator": "dftgnn.stats.g1 (seed-ensemble mean; paired hierarchical bootstrap, 2000, seed 20261008)",
        "v1_arm": v1, "v2_arm": v2,
        "staging_v2": staging(br, uni, splits, budgets, R, draws, ens_v2),
        "loco_v1": loco(br, uni, "loco", rf, b0), "loco_v2": loco(br, uni, "loco_v2", None, None),
        "screening": screening(br, uni, cfg, ens_v1, ens_v2),
        "c0_overlap": c0_overlap(br, uni),
        "epoch_cap": epoch_cap(br), "mlip": mlip(br, uni, splits), "probe": probe(br),
        "run_counts": run_counts(br),
    }
    payload["inputs_sha256"] = dict(sorted(br.inputs.items()))
    print(json.dumps(rnd({k: v for k, v in payload.items() if k != "inputs_sha256"}), indent=1)[:60000])
    print(write_result("ingest_m1_recompute", payload, config=cfg))


if __name__ == "__main__":
    main()
