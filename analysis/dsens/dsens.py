"""Sensitivity of the descriptor advantage of the re-tuned descriptor network D* to the epoch limit, to tuning alone,
to the training-host budget, and its size relative to the structure-only error.

Nothing is trained. Physically: A* = MAE_S - MAE_D* is the error of the neutral oxygen-vacancy formation energy
removed by the pristine-cell electronic descriptors once the descriptor network is re-tuned. A structure-only run S
or a previous descriptor run D that stopped at its 200-epoch limit may not have converged, which would bias the gap;
dropping those cells tests that. At 200 hosts D and D* share the D-state injection route, so D - D* there isolates
the re-tuning. The learning-curve block asks whether A* stays resolved from zero and from delta = 0.05 eV as hosts
are added, and the relative reduction R expresses the gain as a fraction of the structure-only error.

Estimator (unchanged from analysis/dstar/dstar_analysis.py, whose loaders are reused): seed-ensemble mean per
(model, r, B, site); per-resample host statistics; HierDraws (2000 replicates, seed 20261008), one draw set per
design shared by every model, so every gap is paired; replicate value = mean over the R slots of the slot metric;
point = mean over resamples of the per-resample metric; two-sided 95% interval = 2.5/97.5 percentiles, U = one-sided
95% upper bound = 95th percentile (``gap_entry`` / ``a_entry``). Two designs, as in dstar_gaps:

    headline   654 hosts, 30 resamples x 5 seeds; draws HierDraws.make([164] * 30, 2000, 20261008)
    sweep      200, 400, 654 hosts, 10 resamples x 3 seeds; draws of analysis/staging_analysis.make_draws

Blocks
    epoch_limit       headline design. A cell is (r, seed). Cells whose S run (a1) or previous D run (a2) has
                      epochs_run == max_epochs (read from results/analysis/precision_cv_runs.parquet and checked
                      against every raw run record in results/precision_v2 and results/v2_sweep) are removed for
                      every model (S, D, D*); each resample's seed ensemble is re-formed from its remaining seeds;
                      all 30 resamples and the headline draws are kept. The change of a gap from the full headline
                      is taken per replicate on the same draws (paired). D* runs at their 400-epoch limit are counted.
    tuning_only       D - D* at 200 hosts (10 x 3), both D-state.
    learning_curve    A* at 200, 400 (10 x 3) and 654 (30 x 5 and 10 x 3) hosts.
    relative          R = (MAE_S - MAE_X) / MAE_S for X = D*, D. Point = the ratio of the point MAEs (means over
                      resamples); interval = percentiles of the per-replicate ratio of the replicate MAEs.

Every full-design value that sits in results/derived_dstar/dstar_gaps.json is recomputed here through the same code
and compared (tolerance 1e-9 eV). Writes through ``write_result`` into ``results/derived_dsens/dsens.json``.

    PYTHONPATH=$PWD/src python analysis/dsens/dsens.py [--dry-run [--out FILE]]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis import a1_common as c
from analysis import precision_analysis as pa
from analysis import staging_analysis as sa
from analysis.c1.final_network import compare, ms
from analysis.dclass import dclass_analysis as dca
from analysis.dstar import dstar_analysis as da
from dftgnn.stats.g1 import HierDraws, point_metric, seed_ensemble, summarize

OUT_DIR = ROOT / "results" / "derived_dsens"
RUN_DIRS = {"precision": ROOT / "results" / "precision_v2", "sweep": ROOT / "results" / "v2_sweep"}
DSTAR_REC = "derived_dstar/dstar_gaps"
B_HEAD = da.HEADLINE_B
N_R, N_S = da.GRID[B_HEAD]
SWEEP_B = da.SWEEP_B
DELTA = c.DELTA_EV
MODELS = ("S", "D", "Dstar")
GAPS = (("S_minus_Dstar", "S", "Dstar"), ("S_minus_D", "S", "D"), ("D_minus_Dstar", "D", "Dstar"))
COLS = da.COLS


# ---------------------------------------------------------------- epoch-limit cells

def limit_cells() -> tuple[dict, dict]:
    """model -> sorted (r, seed) cells at 654 hosts whose run stopped at its epoch limit; each table row is checked
    against its raw run record (spec model, budget, r, seed; epochs_run; max_epochs)."""
    runs, f_runs = c.load_table("precision_cv_runs")
    p = runs[(runs.backbone == "v2") & (runs.design == "precision") & (runs.B == B_HEAD)
             & (runs.r < N_R) & (runs.seed < N_S)]
    out, facts = {}, {}
    name = {"S": "S", "D": "D-state"}
    for m in ("S", "D"):
        g = p[p.model == m]
        cells = list(zip(g.r.astype(int), g.seed.astype(int)))
        if len(g) != N_R * N_S or len(set(cells)) != N_R * N_S:
            raise AssertionError(f"{m}: {len(g)} runs at {B_HEAD} hosts, {len(set(cells))} distinct cells")
        for row in g.itertuples(index=False):
            pl = json.loads((RUN_DIRS[row.source] / f"{row.run_id}.json").read_text())["payload"]
            s = pl["spec"]
            if (pl["run_id"] != row.run_id or s["model"] != name[m] or int(s["budget"]) != B_HEAD
                    or int(s["r"]) != row.r or int(s["seed"]) != row.seed or int(pl["epochs_run"]) != row.epochs_run
                    or int(pl["max_epochs"]) != row.max_epochs):
                raise AssertionError(f"{row.run_id}: run table differs from the raw record")
        at = g[g.epochs_run == g.max_epochs].sort_values(["r", "seed"])
        if (g.epochs_run > g.max_epochs).any() or set(at.index) != set(g.index[g.capped]):
            raise AssertionError(f"{m}: epochs_run above max_epochs or capped flag inconsistent")
        out[m] = [(int(r), int(s)) for r, s in zip(at.r, at.seed)]
        facts[m] = {"n_runs": len(g), "max_epochs": sorted(int(x) for x in g.max_epochs.unique()),
                    "n_at_epoch_limit": len(at),
                    "cells_at_limit": [{"r": int(r), "seed": int(s), "run_id": rid, "source": src}
                                       for r, s, rid, src in zip(at.r, at.seed, at.run_id, at.source)],
                    "n_at_limit_by_source": {k: int((at.source == k).sum()) for k in sorted(g.source.unique())},
                    "n_runs_checked_against_raw_records": len(g)}
    return out, {"facts": facts, "precision_cv_runs": f_runs}


def drop_cells(long: pd.DataFrame, cells, n_resamples: int, n_seeds: int) -> tuple[pd.DataFrame, dict]:
    """Remove the (r, seed) cells from every model and re-form the seed ensemble from each resample's remaining
    seeds. Every resample must keep at least one seed, and every model the same seeds in a resample."""
    drop = {(int(r), int(s)) for r, s in cells}
    key = list(zip(long.r.astype(int), long.seed.astype(int)))
    kept = long[[k not in drop for k in key]]
    ens = seed_ensemble(kept)
    n_left = {r: n_seeds - sum(1 for rr, _ in drop if rr == r) for r in range(n_resamples)}
    per = ens.groupby(["model", "r"]).n_seeds.agg(["min", "max"])
    for (m, r), row in per.iterrows():
        if row["min"] != n_left[int(r)] or row["max"] != n_left[int(r)]:
            raise AssertionError(f"{m}, r = {r}: {row['min']}-{row['max']} seeds, expected {n_left[int(r)]}")
    if sorted(ens.r.unique()) != list(range(n_resamples)) or min(n_left.values()) < 1:
        raise AssertionError("a resample lost every seed")
    return ens, n_left


# ---------------------------------------------------------------- estimator pieces

def reps_of(st: dict, draws: HierDraws, B: int, metric: str = "mae") -> dict:
    """model -> (per-resample point metric, replicate values) on the shared draws."""
    return {m: (np.array([point_metric(s, metric) for s in st[m][B]]), draws.replicates(st[m][B], metric))
            for m in st}


def summ(per, reps) -> dict:
    return summarize(per, reps, ci=c.CI, upper_quantile=c.UPPER_Q)


def block(rp: dict) -> dict:
    """MAE marginals and the paired gaps of one design point."""
    return {"MAE": {m: ms(summ(*rp[m])) for m in rp},
            "gaps": {k: da.gap_entry(summ(rp[a][0] - rp[b][0], rp[a][1] - rp[b][1])) for k, a, b in GAPS}}


def change(sub: dict, full: dict) -> dict:
    """Paired change of each gap from the full design: (gap_sub - gap_full) per resample and per replicate."""
    out = {}
    for k, a, b in GAPS:
        per = (sub[a][0] - sub[b][0]) - (full[a][0] - full[b][0])
        rep = (sub[a][1] - sub[b][1]) - (full[a][1] - full[b][1])
        s = summ(per, rep)
        lo, hi = s["ci"]
        out[k] = {"change": s["mean"], "ci95": [lo, hi], "ci95_excludes_zero": bool(lo > 0 or hi < 0),
                  "boot_sd": s["boot_sd"]}
    return out


def relative(rp: dict, x: str) -> dict:
    """R = (MAE_S - MAE_x) / MAE_S: point from the point MAEs, interval from the per-replicate ratio."""
    ps, px = rp["S"][0].mean(), rp[x][0].mean()
    rs, rx = rp["S"][1], rp[x][1]
    rr = (rs - rx) / rs
    return {"R": float((ps - px) / ps), "ci95": [float(np.quantile(rr, 0.025)), float(np.quantile(rr, 0.975))],
            "MAE_S_point": float(ps), f"MAE_{x}_point": float(px), "boot_sd": float(rr.std(ddof=1))}


def stored_view(row: dict) -> dict:
    """The MAE marginals and gaps of a design point in the layout of dstar_gaps."""
    return {"MAE": {m: row["MAE"][m] for m in MODELS}, "gaps": {k: row["gaps"][k] for k, _, _ in GAPS}}


# ---------------------------------------------------------------- the designs

def headline_long(dstar: pd.DataFrame) -> pd.DataFrame:
    prec, _ = c.load_table("precision_eval_long")
    base = prec[(prec.backbone == "v2") & (prec.B == B_HEAD) & (prec.r < N_R) & (prec.seed < N_S)][COLS]
    return pd.concat([base, dstar[dstar.B == B_HEAD][COLS]], ignore_index=True)


def headline_reps(ens: pd.DataFrame, draws: HierDraws) -> dict:
    st = {m: pa.resample_stats_by_budget(ens, m, N_R, (B_HEAD,)) for m in MODELS}
    return reps_of(st, draws, B_HEAD)


def epoch_limit(dstar: pd.DataFrame, info: dict) -> tuple[dict, dict, dict]:
    cells, meta = limit_cells()
    long = headline_long(dstar)
    draws = HierDraws.make([pa.N_HOSTS_PER_RESAMPLE] * N_R, c.N_BOOT, seed=c.BOOT_SEED)
    ens_full, _ = drop_cells(long, [], N_R, N_S)
    full = headline_reps(ens_full, draws)
    out = {}
    for tag, m in (("a1_drop_S_at_limit", "S"), ("a2_drop_D_at_limit", "D")):
        ens, n_left = drop_cells(long, cells[m], N_R, N_S)
        sub = headline_reps(ens, draws)
        b = block(sub)
        out[tag] = {"dropped_by": f"cells whose {m} run has epochs_run == max_epochs, removed for S, D and D*",
                    "n_cells_dropped": len(cells[m]), "cells_dropped": [{"r": r, "seed": s} for r, s in cells[m]],
                    "n_resamples_kept": N_R, "n_seeds_per_resample": [n_left[r] for r in range(N_R)],
                    "n_resamples_with_fewer_seeds": sum(1 for r in range(N_R) if n_left[r] < N_S),
                    **b, "A_star": {k: b["gaps"]["S_minus_Dstar"][k] for k in
                                    ("A", "ci95", "upper_one_sided_95", "upper_below_delta", "ci95_excludes_zero")},
                    "change_from_full": change(sub, full)}
    dh = da.health(info)
    out["Dstar_at_limit"] = {B: {"max_epochs": v["max_epochs"], "n_runs": v["n_runs"],
                                 "n_at_epoch_limit": v["n_at_epoch_limit"]} for B, v in dh["per_budget"].items()}
    out["Dstar_at_limit"]["total"] = dh["n_runs_at_epoch_limit"]
    out["S_D_runs"] = meta["facts"]
    out["full_headline"] = block(full)
    return out, full, {"precision_cv_runs": meta["precision_cv_runs"]}


def sweep_reps(dstar: pd.DataFrame) -> tuple[dict, dict]:
    sweep, f_long = dca.load_sweep()
    ds = dstar[(dstar.r < 10) & (dstar.seed < 3)]
    ens = seed_ensemble(pd.concat([sweep[COLS], ds[COLS]], ignore_index=True))
    if (ens.n_seeds != 3).any():
        raise AssertionError("sweep: seed ensemble over the wrong number of seeds")
    ens["backbone"] = "v2"
    draws = sa.make_draws(ens)
    st = {m: sa.stats_by_budget(ens, m, SWEEP_B) for m in MODELS}
    return {B: reps_of(st, draws, B) for B in SWEEP_B}, {"staging_eval_long": f_long}


def a_star_row(g: dict, hosts: int, design: str) -> dict:
    return {"x_training_hosts": hosts, "design": design,
            **{k: g[k] for k in ("A", "ci95", "upper_one_sided_95", "upper_below_delta", "ci95_excludes_zero")}}


def learning_curve(sw: dict, full: dict) -> dict:
    rows = {f"{B}_10x3": a_star_row(block(sw[B])["gaps"]["S_minus_Dstar"], B, "10 x 3") for B in SWEEP_B}
    rows[f"{B_HEAD}_30x5"] = a_star_row(block(full)["gaps"]["S_minus_Dstar"], B_HEAD, "30 x 5")
    pts = [rows[f"{B}_10x3"]["A"] for B in SWEEP_B]
    pts30 = pts[:-1] + [rows[f"{B_HEAD}_30x5"]["A"]]
    return {"rows": rows,
            "U_star_below_delta_every_budget": all(r["upper_below_delta"] for r in rows.values()),
            "ci95_excludes_zero_every_budget": all(r["ci95_excludes_zero"] for r in rows.values()),
            "point_order_hosts": list(SWEEP_B),
            "points_ascending_with_hosts_10x3": bool(np.all(np.diff(pts) > 0)),
            "points_ascending_with_hosts_654_as_30x5": bool(np.all(np.diff(pts30) > 0)),
            "points_descending_with_hosts_10x3": bool(np.all(np.diff(pts) < 0)),
            "points_descending_with_hosts_654_as_30x5": bool(np.all(np.diff(pts30) < 0))}


def main() -> None:
    from dftgnn.config import load_config
    from dftgnn.io.results import write_result

    t0 = time.time()
    cfg = load_config()
    if float(cfg.stats.delta_eV) != DELTA:
        raise AssertionError("stats.delta_eV differs from 0.05")
    stored = c.record(DSTAR_REC)["payload"]
    ref, f_ref = da.s_reference()
    dstar, info = da.load_dstar(ref)
    eps, full, f_runs = epoch_limit(dstar, info)
    sw, f_sw = sweep_reps(dstar)

    sw_tab = {str(B): block(sw[B]) for B in SWEEP_B}
    repro = {"headline_654_30x5": compare(stored_view(block(full)), stored_view(stored["headline_654_30x5"]["654"]),
                                          "results/derived_dstar/dstar_gaps.json payload.headline_654_30x5.654 (MAE of "
                                          "S, D, D*; S - D*, S - D, D - D*)"),
             "sweep_10x3": compare({B: stored_view(v) for B, v in sw_tab.items()},
                                   {str(B): stored_view(stored["sweep_10x3"][str(B)]) for B in SWEEP_B},
                                   "results/derived_dstar/dstar_gaps.json payload.sweep_10x3.200/400/654 (MAE of S, "
                                   "D, D*; S - D*, S - D, D - D*)")}
    bad = [k for k, v in repro.items() if not v["reproduced"]]
    if bad:
        raise AssertionError(f"not reproduced: {bad} {repro}")

    tun = sw_tab["200"]["gaps"]["D_minus_Dstar"]
    tun_ref = "results/derived_dstar/dstar_gaps.json payload.sweep_10x3.200.gaps.D_minus_Dstar"
    tun_cmp = compare(tun, stored["sweep_10x3"]["200"]["gaps"]["D_minus_Dstar"], tun_ref)
    if not tun_cmp["reproduced"]:
        raise AssertionError(f"D - D* at 200 hosts not reproduced: {tun_cmp}")
    tuning = {"x_training_hosts": 200, "design": "10 x 3", "variants": "D and D* both D-state",
              "D_minus_Dstar": {k: tun[k] for k in ("A", "ci95", "upper_one_sided_95", "ci95_excludes_zero",
                                                    "inside_pm_0p01", "boot_sd")},
              "stored_value": tun_ref, "comparison": tun_cmp}

    rel = {"definition": "R_X = (MAE_S - MAE_X) / MAE_S; point = (mean over resamples of MAE_S - mean over "
                         "resamples of MAE_X) / mean over resamples of MAE_S; ci95 = 2.5/97.5 percentiles over the "
                         "2000 replicates of (MAE_S,b - MAE_X,b) / MAE_S,b on the shared draws",
           "rows": {f"{B_HEAD}_30x5": {"x_training_hosts": B_HEAD, "Dstar": relative(full, "Dstar"),
                                        "D": relative(full, "D")},
                    **{f"{B}_10x3": {"x_training_hosts": B, "Dstar": relative(sw[B], "Dstar"),
                                     "D": relative(sw[B], "D")} for B in SWEEP_B}}}

    payload = {
        "label": "sensitivity of A* = MAE_S - MAE_D* (re-tuned descriptor network D*, dretrain_v2) to the epoch "
                 "limit, to tuning alone, to the training-host budget, and relative to MAE_S",
        "delta_eV": DELTA, "sign_convention": {k: da.SIGNS[k] for k in ("A_old", "A_star", "D_minus_Dstar")},
        "estimator": "analysis/dstar/dstar_analysis.py estimator: seed ensemble, per-resample host statistics, "
                     "HierDraws (2000, seed 20261008) shared by all models of a design; point = mean over resamples; "
                     "ci95 = 2.5/97.5 percentiles; upper_one_sided_95 (U) = 95th percentile; gap_entry layout",
        "designs": stored["designs"],
        "epoch_limit_654_30x5": eps, "tuning_only_200": tuning,
        "learning_curve": learning_curve(sw, full), "relative_reduction": rel,
        "reproduction": repro,
        "reproduction_summary": {"max_abs_difference": max(v["max_abs_difference"] for v in [*repro.values(), tun_cmp]),
                                 "n_values_compared": sum(v["n_values_compared"] for v in repro.values()),
                                 "all_reproduced_below_1e-9": True},
        "inputs": {**f_ref, **f_runs, **f_sw,
                   "dstar_gaps": {"path": "results/derived_dstar/dstar_gaps.json",
                                  "sha256": c.sha256(ROOT / "results" / "derived_dstar" / "dstar_gaps.json")},
                   "dretrain_v2_spec": {"path": "specs/dretrain_v2.jsonl", "sha256": info["spec_sha256"]}},
    }
    payload = c.to_jsonable(payload)
    print(f"runtime_s {time.time() - t0:.1f}", file=sys.stderr)
    if "--dry-run" in sys.argv:
        if "--out" in sys.argv:
            out = Path(sys.argv[sys.argv.index("--out") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=1))
            print(out)
        else:
            print(json.dumps(payload, indent=1))
        return
    print(write_result("dsens", payload, config=cfg, repo_root=ROOT, results_dir=OUT_DIR))


if __name__ == "__main__":
    main()
