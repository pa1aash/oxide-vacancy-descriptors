"""Further comparisons of the re-tuned descriptor network D* (stage dretrain_v2), each on the estimator and design of
the record the paper already quotes for the previous descriptor network D.

Nothing is trained. D* is loaded with the checks of analysis/dstar/dstar_analysis.py (load_dstar). Physically, these
quantities ask whether the descriptor advantage of D* also shows inside a host, where the electronic structure of the
pristine cell is shared and only the local environment of the removed oxygen differs, and how D* compares with the
descriptor forest of Kumagai et al.

    spread      within-host prediction spread: mean over (resample, host with >= 2 sites) of the SD of the
                seed-ensemble prediction across the host's sites, over the same SD of the DFT targets;
                analysis/c1/final_network.within_host_spread (the code of final_network_summary.json
                table.within_host_prediction_spread); 200, 400, 654 hosts (10 x 3) and 654 hosts (30 x 5)
    spearman    per host with >= 3 tested sites, Spearman rank correlation of prediction and target, mean over hosts,
                host-cluster bootstrap (analysis/within_host_breakdown.py, 654 hosts, 10 x 3, the design of
                results/within_host_breakdown.json); the module compares S with D, so D* enters in the place of D
                (and D in the place of S for D - D*) through its loader; the Appendix E groups come from the same run
    a_within    within-host residual MAE gap S - D* with the paired hierarchical bootstrap of
                dftgnn.stats.metrics.aggregate_delta (seed 0), scripts/analysis_c1c2.py a_block and n_star, the
                estimator of results/c1c2_v2.json A_within_host (10 x 3); N* within over 200, 400, 654 hosts
    cross       MAE_D* - MAE_RF-Kumagai with the paired G1 bootstrap and the draws of analysis/w2/diagnostics.py
                crossover (results/derived_w2/w2_forest_crossover.json); 10 x 3
    skill       within-host skill: within-host residual MAE of a predictor constant within each host minus that of
                the network; 10 x 3 with aggregate_delta seed 0 (results/c1c2_v2.json mae.*.within_host_skill), 654
                hosts 30 x 5 with the G1 contrast of analysis/precision_analysis.analyse
                (results/precision_g1_v2.json); figure_inputs repeats these in the key layout of the two records

Every S and D value that sits in one of those records is recomputed here and compared (tolerance 1e-9 eV).

    PYTHONPATH=$PWD/src python analysis/dstar/dstar_extras.py [--dry-run] [--data-root ~/Desktop/dft-gnn]
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis import a1_common as c
from analysis import precision_analysis as pa
from analysis import staging_analysis as sa
from analysis import within_host_breakdown as wb
from analysis.c1 import final_network as fn
from analysis.dstar import dstar_analysis as da
from analysis.w2 import diagnostics as w2

OUT_DIR = ROOT / "results" / "derived_dstar"
SWEEP_B = da.SWEEP_B
HEADLINE_B = da.HEADLINE_B
DELTA = c.DELTA_EV
COLS = da.COLS
LABEL = ("re-tuned descriptor network D*, dretrain_v2, spec sha256 "
         "c33cfa33aac3ed56c46c9802abbf70f63e7b030386043f233e5761c9b9b60f65; within-host spread, within-host "
         "Spearman, within-host residual gap, forest crossover and within-host skill on the estimators of the "
         "records quoted for the previous descriptor network D")


def c1c2_module():
    spec = importlib.util.spec_from_file_location("analysis_c1c2", ROOT / "scripts" / "analysis_c1c2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def frames(dstar: pd.DataFrame) -> dict:
    """Seed-level prediction tables of S, D, D* for the 10 x 3 design (200, 400, 654) and the 30 x 5 design (654)."""
    long, f_long = sa.load_table("staging_eval_long", c.record("staging_eval_table"))
    sw = long[(long.backbone == "v2") & long.model.isin(["S", "D"]) & long.B.isin(SWEEP_B)][COLS]
    sw = pd.concat([sw, dstar[(dstar.r < 10) & (dstar.seed < 3)][COLS]], ignore_index=True).assign(backbone="v2")
    prec, f_prec = c.load_table("precision_eval_long")
    hd = prec[(prec.backbone == "v2") & (prec.B == HEADLINE_B) & prec.model.isin(["S", "D"])][COLS]
    hd = pd.concat([hd, dstar[dstar.B == HEADLINE_B][COLS]], ignore_index=True).assign(backbone="v2")
    return {"sweep": sw, "headline": hd, "inputs": {"staging_eval_long": f_long, "precision_eval_long": f_prec}}


# ---------------------------------------------------------------- 1 within-host prediction spread

def spread(fr: dict) -> dict:
    out = {}
    for design, long in (("sweep_10x3", fr["sweep"]), ("headline_654_30x5", fr["headline"])):
        ens = sa.ensemble(long)
        out[design] = fn.within_host_spread(ens, long)["v2"]
    ref = c.record("derived_c1/final_network_summary")["payload"]["table"]["within_host_prediction_spread"]
    repro = fn.compare({m: {str(B): out["sweep_10x3"][m][str(B)] for B in SWEEP_B} for m in ("S", "D")},
                       {m: {str(B): ref[m][str(B)] for B in SWEEP_B} for m in ("S", "D")},
                       "results/derived_c1/final_network_summary.json payload.table.within_host_prediction_spread "
                       "(S, D at 200, 400, 654)")
    return {"values": out, "reproduction": repro}


# ---------------------------------------------------------------- 2 Spearman and the Appendix E groups

def breakdown(dstar: pd.DataFrame, data_root: Path) -> dict:
    """analysis/within_host_breakdown.run with D* in the place of D (and D in the place of S for D - D*)."""
    ds = dstar[(dstar.B == wb.BUDGET) & (dstar.r < 10) & (dstar.seed < 3)]
    ds = ds[["site_id", "host_id", "y_true", "y_pred", "r", "seed"]].assign(model="D")
    orig = wb.load_v2

    def loader(first: str):
        def load(root: Path):
            df, meta = orig(root)
            a = df[df.model == first].assign(model="S")
            return pd.concat([a, ds], ignore_index=True), {**meta, "D_replaced_by": "Dstar", "S_is": first}
        return load

    base = wb.run(ROOT, data_root)
    try:
        wb.load_v2 = loader("S")
        s_ds = wb.run(ROOT, data_root)
        wb.load_v2 = loader("D")
        d_ds = wb.run(ROOT, data_root)
    finally:
        wb.load_v2 = orig
    rec = c.record("within_host_breakdown")["payload"]
    repro = fn.compare({"groups": base["groups"], "spearman": base["spearman"]},
                       {"groups": rec["groups"], "spearman": rec["spearman"]},
                       "results/within_host_breakdown.json payload.groups and .spearman (S, D, both networks)")
    # the S of the S/D* run and the D of the D/D* run must be the S and D of the record
    same = fn.compare({"S": [g["v2"]["S"] for g in s_ds["groups"]], "D": [g["v2"]["S"] for g in d_ds["groups"]],
                       "spS": s_ds["spearman"]["v2"]["S"], "spD": d_ds["spearman"]["v2"]["S"]},
                      {"S": [g["v2"]["S"] for g in rec["groups"]], "D": [g["v2"]["D"] for g in rec["groups"]],
                       "spS": rec["spearman"]["v2"]["S"], "spD": rec["spearman"]["v2"]["D"]},
                      "results/within_host_breakdown.json payload.groups[*].v2 and .spearman.v2 (S and D carried "
                      "into the runs with D*)")
    sp = {"S": rec["spearman"]["v2"]["S"], "D": rec["spearman"]["v2"]["D"], "Dstar": s_ds["spearman"]["v2"]["D"],
          "S_minus_D": rec["spearman"]["differences"]["S_minus_D_v2"],
          "S_minus_Dstar": s_ds["spearman"]["differences"]["S_minus_D_v2"],
          "D_minus_Dstar": d_ds["spearman"]["differences"]["S_minus_D_v2"],
          "hosts_with_ge3_sites": rec["spearman"]["hosts_with_ge3_sites"], "bootstrap_position_k": rec["spearman"]["k"]}
    groups = []
    for g0, g1, g2 in zip(rec["groups"], s_ds["groups"], d_ds["groups"], strict=True):
        if (g0["variable"], g0["group"]) != (g1["variable"], g1["group"]) != (g2["variable"], g2["group"]):
            raise AssertionError("group order differs between runs")
        groups.append({"variable": g0["variable"], "group": g0["group"], "flag": g0["flag"],
                       "n_hosts": g0["n_hosts"], "n_sites": g0["n_sites"], "n_hosts_by_metric": g0["n_hosts_by_metric"],
                       "S": g0["v2"]["S"], "D": g0["v2"]["D"], "Dstar": g1["v2"]["D"],
                       "S_minus_D": g0["differences"]["S_minus_D_v2"],
                       "S_minus_Dstar": g1["differences"]["S_minus_D_v2"],
                       "D_minus_Dstar": g2["differences"]["S_minus_D_v2"]})
    return {"spearman": sp, "groups": groups, "reproduction": repro, "carried_over": same,
            "definitions": rec["definitions"], "bootstrap": rec["bootstrap"]}


# ---------------------------------------------------------------- 3 and 6: within-host gap and skill

def ens_frames(long: pd.DataFrame, budgets, n_r: int) -> dict:
    from dftgnn.stats.g1 import seed_ensemble

    ens = seed_ensemble(long[COLS])
    return {(m, B, r): g.sort_values("site_id").reset_index(drop=True)
            for (m, B, r), g in ens.groupby(["model", "B", "r"]) if B in budgets and r < n_r}


def c1c2_block(fr: dict) -> dict:
    """10 x 3, the estimator of results/c1c2_v2.json (aggregate_delta and aggregate_resamples, seed 0)."""
    from dftgnn.config import load_config
    from dftgnn.stats.metrics import aggregate_delta, aggregate_resamples, host_stats

    cfg = load_config()
    mod = c1c2_module()
    D = mod.D
    n = cfg.analysis.bootstrap.draws
    e = ens_frames(fr["sweep"], SWEEP_B, 10)
    rows_w, rows_a, mae = {m: {} for m in ("S_minus_D", "S_minus_Dstar", "D_minus_Dstar")}, {}, {}
    for B in SWEEP_B:
        ref = [e[("S", B, r)] for r in range(10)]
        al = {m: [D._align(e[(m, B, r)], ref[r]) for r in range(10)] for m in ("S", "D", "Dstar")}
        hs = {m: [host_stats(f.y_true, f.y_pred, f.host_id) for f in al[m]] for m in al}
        const = [D.stats_of(D.host_constant(f)) for f in ref]
        for k, a, b in (("S_minus_D", "S", "D"), ("S_minus_Dstar", "S", "Dstar"), ("D_minus_Dstar", "D", "Dstar")):
            rows_w[k][B] = mod.a_block(hs[a], hs[b], "within_host_mae", cfg)
        rows_a[B] = mod.a_block(hs["S"], hs["D"], "mae", cfg)
        mae[B] = {}
        for m in hs:
            mae[B][m] = {k: {q: aggregate_resamples(hs[m], n_boot=n, seed=0)[k][q] for q in ("mean", "ci")}
                         for k in ("mae", "within_host_mae")}
            sk = aggregate_delta(const, hs[m], n_boot=n, seed=0)["within_host_mae"]
            mae[B][m]["within_host_skill"] = {"mean": sk["mean"], "ci": sk["ci"]}
    rec = c.record("c1c2_v2")["payload"]
    keys = ("A", "ci95", "upper_one_sided_95", "per_resample", "sd_across_resamples")
    repro = fn.compare(
        {"A_within_host": {str(B): rows_w["S_minus_D"][B] for B in SWEEP_B},
         "A_mae": {str(B): rows_a[B] for B in SWEEP_B},
         "mae": {str(B): {"S": mae[B]["S"], "D-state": mae[B]["D"]} for B in SWEEP_B}},
        {"A_within_host": {str(B): {k: rec["A_within_host"][str(B)][k] for k in keys} for B in SWEEP_B},
         "A_mae": {str(B): {k: rec["A_mae"][str(B)][k] for k in keys} for B in SWEEP_B},
         "mae": {str(B): {m: rec["mae"][str(B)][m] for m in ("S", "D-state")} for B in SWEEP_B}},
        "results/c1c2_v2.json payload.A_within_host, A_mae, mae.*.{S, D-state} (mae, within_host_mae, "
        "within_host_skill) at 200, 400, 654")
    for k in rows_w:
        for B in SWEEP_B:
            r = rows_w[k][B]
            r["upper_below_delta"] = bool(r["upper_one_sided_95"] < DELTA)
            r["ci95_excludes_zero"] = bool(r["ci95"][0] > 0 or r["ci95"][1] < 0)
    nstar = {k: mod.n_star(rows_w[k], DELTA) for k in ("S_minus_D", "S_minus_Dstar")}
    ref_n = rec["N_star_within_host"]
    return {"A_within_host": {k: {str(B): v[B] for B in SWEEP_B} for k, v in rows_w.items()},
            "mae": {str(B): v for B, v in mae.items()},
            "N_star_within_host": {**nstar, "paper_reference": {**ref_n, "source": "results/c1c2_v2.json "
                                                                "payload.N_star_within_host (budgets 25-654)"}},
            "reproduction": repro}


def precision_block(fr: dict) -> dict:
    """654 hosts, 30 x 5, the G1 contrast of analysis/precision_analysis.analyse (results/precision_g1_v2.json)."""
    from dftgnn.stats.g1 import HierDraws, contrast, seed_ensemble

    n_r = 30
    ens = seed_ensemble(fr["headline"][COLS])
    if (ens.n_seeds != 5).any():
        raise AssertionError("30 x 5: seed ensemble over the wrong number of seeds")
    draws = HierDraws.make([pa.N_HOSTS_PER_RESAMPLE] * n_r, c.N_BOOT, seed=c.BOOT_SEED)
    st = {m: pa.resample_stats_by_budget(ens, m, n_r, (HEADLINE_B,)) for m in ("S", "D", "Dstar")}
    const = pa.resample_stats_by_budget(ens, "S", n_r, (HEADLINE_B,), constant=True)
    B = HEADLINE_B
    out = {}
    for m in st:
        sk = contrast(const, st[m], draws, "within_host_mae", ci=c.CI, upper_quantile=c.UPPER_Q)[B]
        mm = {k: contrast(st[m], st[m], draws, k, ci=c.CI, upper_quantile=c.UPPER_Q)[B]["a"]
              for k in ("mae", "within_host_mae")}
        out[m] = {"mae": {"mean": mm["mae"]["mean"], "ci": mm["mae"]["ci"]},
                  "within_host_mae": {"mean": mm["within_host_mae"]["mean"], "ci": mm["within_host_mae"]["ci"]},
                  "within_host_skill": {"mean": sk["diff"]["mean"], "ci": sk["diff"]["ci"]}}
    aw = {}
    for k, a, b in (("S_minus_D", "S", "D"), ("S_minus_Dstar", "S", "Dstar"), ("D_minus_Dstar", "D", "Dstar")):
        p = contrast(st[a], st[b], draws, "within_host_mae", ci=c.CI, upper_quantile=c.UPPER_Q)[B]["diff"]
        aw[k] = {"A": p["mean"], "ci95": p["ci"], "upper_one_sided_95": p["upper_one_sided"],
                 "per_resample": p["per_resample"], "sd_across_resamples": p["sd_over_resamples"],
                 "upper_below_delta": bool(p["upper_one_sided"] < DELTA),
                 "ci95_excludes_zero": bool(p["ci"][0] > 0 or p["ci"][1] < 0)}
    rec = c.record("precision_g1_v2")["payload"]
    repro = fn.compare({"mae": {"S": out["S"], "D-state": out["D"]}, "A_within_host": aw["S_minus_D"]},
                       {"mae": {m: rec["mae"][str(B)][m] for m in ("S", "D-state")},
                        "A_within_host": rec["A_within_host"][str(B)]},
                       "results/precision_g1_v2.json payload.mae.654.{S, D-state} and A_within_host.654")
    return {"mae": out, "A_within_host": aw, "reproduction": repro}


# ---------------------------------------------------------------- 4 crossover with the forest

def crossover(fr: dict) -> dict:
    inp = w2.load_inputs(ROOT)
    ens = sa.ensemble(fr["sweep"])
    draws = sa.make_draws(ens)
    if not (np.array_equal(draws.pick, inp["draws"].pick) and np.array_equal(draws.w, inp["draws"].w)):
        raise AssertionError("crossover draws differ from those of analysis/w2/diagnostics.py")
    rf = w2.rf_stats(inp)
    out = {}
    for m, st in (("D", w2.stats(inp["ens"], "D")), ("Dstar", sa.stats_by_budget(ens, "Dstar", SWEEP_B))):
        con = w2.sweep_contrast(inp, {B: st[B] for B in SWEEP_B}, {B: rf[B] for B in SWEEP_B})
        out[m] = {str(B): {**w2.diff_entry(con[B]["diff"]), f"MAE_{m}": w2.ms(con[B]["a"]),
                           "MAE_RF": w2.ms(con[B]["b"])} for B in SWEEP_B}
    rec = c.record("derived_w2/w2_forest_crossover")["payload"]["D"]
    repro = fn.compare(out["D"], {str(B): rec[str(B)] for B in SWEEP_B},
                       "results/derived_w2/w2_forest_crossover.json payload.D.200/400/654")
    return {"Dstar_minus_RF": out["Dstar"], "D_minus_RF": out["D"], "reproduction": repro,
            "rule": "difference MAE_model - MAE_RF-Kumagai of the seed ensemble, paired G1 hierarchical bootstrap "
                    "(draws of analysis/w2/diagnostics.py); below with 95% interval = upper end below 0"}


# ---------------------------------------------------------------- main

def main() -> None:
    from dftgnn.config import load_config
    from dftgnn.io.results import write_result

    cfg = load_config()
    if float(cfg.stats.delta_eV) != DELTA:
        raise AssertionError("stats.delta_eV differs from 0.05")
    data_root = (Path(sys.argv[sys.argv.index("--data-root") + 1]).expanduser().resolve()
                 if "--data-root" in sys.argv else ROOT)
    ref, _ = da.s_reference()
    dstar, info = da.load_dstar(ref)
    fr = frames(dstar)
    sp = spread(fr)
    bd = breakdown(dstar, data_root)
    c12 = c1c2_block(fr)
    pr = precision_block(fr)
    cr = crossover(fr)
    repro = {"spread": sp["reproduction"], "within_host_breakdown": bd["reproduction"],
             "within_host_breakdown_carried_over": bd["carried_over"], "c1c2_v2": c12["reproduction"],
             "precision_g1_v2": pr["reproduction"], "w2_forest_crossover": cr["reproduction"]}
    bad = [k for k, v in repro.items() if not v["reproduced"]]
    if bad:
        raise AssertionError(f"not reproduced: {bad}")
    payload = {
        "label": LABEL, "delta_eV": DELTA,
        "designs": {"sweep_10x3": "outer resamples r = 0-9 x seeds 0-2 (D* at 654 hosts: the r = 0-9, seeds 0-2 "
                                  "subset of its 150 runs)",
                    "headline_654_30x5": "654 hosts, outer resamples r = 0-29 x seeds 0-4"},
        "sign_convention": {"S_minus_Dstar": "positive = D* more accurate (MAE) or S higher (Spearman)",
                            "D_minus_Dstar": "positive = D* more accurate (MAE) or D higher (Spearman)",
                            "Dstar_minus_RF": "negative = D* more accurate than the forest",
                            "within_host_skill": "zero-skill within-host residual MAE minus the network's; "
                                                 "positive = skill"},
        "spread": sp["values"],
        "spearman": bd["spearman"],
        "within_host_groups": {"groups": bd["groups"], "definitions": bd["definitions"],
                               "bootstrap": bd["bootstrap"]},
        "a_within": {"sweep_10x3": {k: c12["A_within_host"][k] for k in c12["A_within_host"]},
                     "N_star_within_host": c12["N_star_within_host"],
                     "headline_654_30x5": pr["A_within_host"]},
        "crossover": {k: cr[k] for k in ("Dstar_minus_RF", "D_minus_RF", "rule")},
        "skill": {"sweep_10x3": {B: {m: v[m]["within_host_skill"] for m in v} for B, v in c12["mae"].items()},
                  "headline_654_30x5": {m: v["within_host_skill"] for m, v in pr["mae"].items()}},
        "figure_inputs": {
            "c1c2_layout": {"note": "keys of results/c1c2_v2.json (mae.<B>.<model>, A_within_host.<B>); model "
                                    "Dstar; 10 x 3; S and D-state reproduce that record",
                            "mae": {B: {"Dstar": v["Dstar"]} for B, v in c12["mae"].items()},
                            "A_within_host": c12["A_within_host"]["S_minus_Dstar"]},
            "precision_g1_layout": {"note": "keys of results/precision_g1_v2.json (mae.654.<model>, "
                                            "A_within_host.654); model Dstar; 30 x 5",
                                    "mae": {str(HEADLINE_B): {"Dstar": pr["mae"]["Dstar"]}},
                                    "A_within_host": {str(HEADLINE_B): pr["A_within_host"]["S_minus_Dstar"]}}},
        "health": {"n_runs": len(info["runs"]), "spec_sha256": info["spec_sha256"]},
        "reproduction": repro,
        "reproduction_summary": {"max_abs_difference": max(v["max_abs_difference"] for v in repro.values()),
                                 "all_reproduced_below_1e-9": not bad,
                                 "n_values_compared": sum(v["n_values_compared"] for v in repro.values())},
        "inputs": {**fr["inputs"], "dretrain_v2_spec": {"path": "specs/dretrain_v2.jsonl", "sha256": info["spec_sha256"]},
                   "universe": {"path": "data/processed/universe_v1.parquet",
                                "sha256": c.sha256(data_root / "data/processed/universe_v1.parquet")}},
    }
    payload = c.to_jsonable(payload)
    if "--dry-run" in sys.argv:
        print(json.dumps(payload, indent=1, allow_nan=False))
        return
    print(write_result("dstar_extras", payload, config=cfg, repo_root=ROOT, results_dir=OUT_DIR))


if __name__ == "__main__":
    main()
