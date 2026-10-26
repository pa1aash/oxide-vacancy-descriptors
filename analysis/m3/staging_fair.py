"""The fair staging test of the final network (stage pcompv2b), with the staging estimator unchanged.

Nothing is trained. The descriptor predictor P1 was re-tuned (stage p1tune_v2, 30 trials per anchor at 50, 200 and
654 hosts, configs/tuned_p1_v2.yaml) and retrained with an epoch limit of 400 (stage p1v2b, 180 runs, none at the
limit). Stage pcompv2b (180 runs) composes each retrained P1 with the existing D checkpoint of the same (B, r, seed)
cell: the energy head is D, fed P1's predicted descriptors. Physically: if the electronic descriptors can be learned
from the crystal structure well enough, the staged network P should recover the descriptor advantage of D over S.

Estimator (imported unchanged from analysis/staging_analysis.py, the estimator of results/staging_v2.json):
seed-ensemble mean per (model, r, B, site) over seeds 0-2, per-host sufficient statistics, the G1 hierarchical
bootstrap over the 10 resamples' test hosts (2000 replicates, seed 20261008, one draw set for S, D and P at every
budget, so every contrast is paired). Point = mean over resamples; 95% interval = 2.5th and 97.5th percentiles;
one-sided 95% bounds = 5th and 95th percentiles. S and D are the v2 runs of results/analysis/staging_eval_long.parquet
(the 10 x 3 sweep). The 30 x 5 design of the headline has no P runs, so every number here is on the 10 x 3 design.

Sign convention: the paper's A = MAE_S - MAE_D (positive = D more accurate). Here A_P = MAE_S - MAE_P (positive = P
more accurate) and the staging cost P - D = MAE_P - MAE_D; both with the two-sided interval and the one-sided bounds.

Wording tier at 654 hosts (pre-agreed): W if the 95% interval of A_P lies above zero (P measurably better than S, the
criterion by which the paper finds D better than S); N if not W and either P is measurably worse than S (interval of
P - S above zero) or the one-sided 95% upper bound of P - S is below delta = 0.05 eV; I otherwise.

Swap-in: P fed the true descriptors in place of P1's output is D itself. results/swapin/swapin_v2.json verified this
on the cluster for every P/D pair of the final network (max |difference| 0.0 eV, identity standardisation), and the D
checkpoints of pcompv2b are the D runs of those pairs (checked here). The change caused by the swap-in is therefore
the stored D prediction minus the stored P prediction, site by site; it is reported per budget as the MAE change
(MAE_D - MAE_P, paired interval) and the mean absolute per-site change. The checkpoints are on cluster storage only,
so the network is not re-executed here.

P1 standalone: per budget, per-descriptor R2 and MAE on the test sites (recorded in each p1v2b record), the class
means (12 host-level, 10 site-level descriptors), the mean standardised MAE, and the epoch-limit count; the previous
P1 (results/p1_v2, epoch limit 200) alongside.

Checks before a record is written: pcompv2b and p1v2b run ids equal their spec files; recorded spec equals the spec
line; git_dirty false; recorded code commit is an accepted commit of the recorded code ref (accepted_code_commits);
no duplicate cell; prediction sha256 equals the record; finite predictions; recorded MAE equals the stored
predictions (1e-6); each P run covers exactly the test sites (same targets and hosts) of the S run of its cell; each
P run's D component is the D run of its cell in the sweep and its P1 component is a p1v2b run of the same cell; S and
D MAE and S - D reproduce results/staging_v2.json to 1e-9.

    PYTHONPATH=$PWD/src python analysis/m3/staging_fair.py [--dry-run]
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
from analysis import staging_analysis as sa

OUT_DIR = ROOT / "results" / "derived_staging"
BUDGETS = sa.BUDGETS
N_RESAMPLES, SEEDS = 10, (0, 1, 2)
DELTA = c.DELTA_EV
REPRO_TOL = 1e-9
HOST_DESC = ("desc_ave_ele_diele", "desc_ave_ion_diele", "desc_band_gap", "desc_cbm_d", "desc_cbm_f", "desc_cbm_p",
             "desc_cbm_s", "desc_formation_energy", "desc_vbm_d", "desc_vbm_f", "desc_vbm_p", "desc_vbm_s")
LABEL = ("EXTENSION on the final network (v2, POST HOC backbone); stages p1v2b and pcompv2b (retrained P1, epoch "
         "limit 400, composed with the existing D checkpoints)")


def specs(stage: str) -> dict:
    out = {}
    for line in (ROOT / "specs" / f"{stage}.jsonl").read_text().splitlines():
        row = json.loads(line)
        out[row["run_id"]] = row
    return out


def check_record(rec: dict, rid: str, spec: dict) -> dict:
    from dftgnn.train import accepted_code_commits

    pl = rec["payload"]
    if rec["git_dirty"] is not False or pl["run_id"] != rid or rec["name"] != rid or pl["smoke"]:
        raise AssertionError(f"{rid}: dirty, misnamed or smoke")
    if pl["spec"] != spec["spec"]:
        raise AssertionError(f"{rid}: recorded spec differs from the spec file")
    if rec["git_sha"] not in accepted_code_commits(pl["code_ref"], ROOT):
        raise AssertionError(f"{rid}: code commit {rec['git_sha']} is not accepted for {pl['code_ref']}")
    return pl


def load_sweep() -> tuple[pd.DataFrame, dict]:
    long, f = sa.load_table("staging_eval_long", c.record("staging_eval_table"))
    s = long[(long.backbone == "v2") & long.model.isin(["S", "D", "P"])]
    return s.reset_index(drop=True), f


def load_p1() -> tuple[dict, dict]:
    sp = specs("p1v2b")
    files = sorted((ROOT / "results" / "p1v2b").glob("*.json"))
    if len(files) != 180 or {p.stem for p in files} != set(sp):
        raise AssertionError("p1v2b records differ from specs/p1v2b.jsonl")
    cell, capped, rows = {}, [], []
    for p in files:
        rec = json.loads(p.read_text())
        pl = check_record(rec, p.stem, sp[p.stem])
        s = pl["spec"]
        key = (int(s["budget"]), int(s["r"]), int(s["seed"]))
        if key in cell:
            raise AssertionError(f"{p.stem}: duplicate p1v2b cell {key}")
        cell[key] = p.stem
        if pl["epochs_run"] >= pl["max_epochs"]:
            capped.append(p.stem)
        m = pl["metrics"]
        rows.append({"B": key[0], "run": p.stem, "std_mae": m["mean_standardised_mae"],
                     **{f"r2|{k}": v["r2"] for k, v in m["per_descriptor"].items()},
                     **{f"mae|{k}": v["mae"] for k, v in m["per_descriptor"].items()},
                     "max_epochs": pl["max_epochs"], "epochs_run": pl["epochs_run"], "best_epoch": pl["best_epoch"]})
    return cell, {"table": pd.DataFrame(rows), "capped": sorted(capped), "spec_sha256": c.sha256(ROOT / "specs/p1v2b.jsonl")}


def p1_block(df: pd.DataFrame) -> dict:
    out = {}
    for B in BUDGETS:
        g = df[df.B == B]
        r2 = {k[3:]: float(g[k].mean()) for k in g.columns if k.startswith("r2|")}
        host = [v for k, v in r2.items() if k in HOST_DESC]
        site = [v for k, v in r2.items() if k not in HOST_DESC]
        if len(host) != 12 or len(site) != 10:
            raise AssertionError("descriptor classes are not 12 host + 10 site")
        out[str(B)] = {"n_runs": len(g), "mean_standardised_mae": float(g.std_mae.mean()),
                       "mean_standardised_mae_sd": float(g.std_mae.std(ddof=1)),
                       "class_mean_r2": {"host_electronic": float(np.mean(host)), "site_electronic": float(np.mean(site))},
                       "all_mean_r2": float(np.mean(host + site)), "r2_per_descriptor": r2,
                       "n_at_epoch_limit": int((g.epochs_run >= g.max_epochs).sum()),
                       "max_epochs_run": int(g.epochs_run.max()), "max_epochs": int(g.max_epochs.max())}
    return out


def old_p1_block() -> dict:
    rows = []
    for p in sorted((ROOT / "results" / "p1_v2").glob("*.json")):
        pl = json.loads(p.read_text())["payload"]
        B = int(pl["spec"]["budget"])
        if B not in BUDGETS or int(pl["spec"]["r"]) < 0:
            continue
        m = pl["metrics"]
        rows.append({"B": B, "std_mae": m["mean_standardised_mae"], "epochs_run": pl["epochs_run"],
                     "max_epochs": pl["max_epochs"], **{f"r2|{k}": v["r2"] for k, v in m["per_descriptor"].items()}})
    df = pd.DataFrame(rows)
    out = {}
    for B in BUDGETS:
        g = df[df.B == B]
        r2 = {k[3:]: float(g[k].mean()) for k in g.columns if k.startswith("r2|")}
        out[str(B)] = {"n_runs": len(g), "mean_standardised_mae": float(g.std_mae.mean()),
                       "class_mean_r2": {"host_electronic": float(np.mean([v for k, v in r2.items() if k in HOST_DESC])),
                                         "site_electronic": float(np.mean([v for k, v in r2.items()
                                                                           if k not in HOST_DESC]))},
                       "n_at_epoch_limit": int((g.epochs_run >= g.max_epochs).sum()), "max_epochs": int(g.max_epochs.max())}
    return out


def load_pcomp(sweep: pd.DataFrame, p1_cell: dict) -> tuple[pd.DataFrame, dict]:
    sp = specs("pcompv2b")
    stage = ROOT / "results" / "pcompv2b"
    files = sorted(stage.glob("*.json"))
    if len(files) != 180 or {p.stem for p in files} != set(sp):
        raise AssertionError("pcompv2b records differ from specs/pcompv2b.jsonl")
    if len(list((stage / "predictions").glob("*.parquet"))) != 180:
        raise AssertionError("pcompv2b: not 180 prediction parquets")
    s0 = sweep[sweep.r >= 0]
    d_run = {(int(B), int(r), int(s)): rid for (B, r, s), rid in
             s0[s0.model == "D"].groupby(["B", "r", "seed"]).run_id.first().items()}
    ref = {(B, r): g.set_index("site_id") for (B, r), g in s0[(s0.model == "S") & (s0.seed == 0)].groupby(["B", "r"])}
    frames, cells, worst, hosts, timestamps = [], set(), 0.0, {}, []
    for p in files:
        rec = json.loads(p.read_text())
        pl = check_record(rec, p.stem, sp[p.stem])
        s = pl["spec"]
        B, r, seed = int(s["budget"]), int(s["r"]), int(s["seed"])
        if (B, r, seed) in cells:
            raise AssertionError(f"{p.stem}: duplicate cell")
        cells.add((B, r, seed))
        if pl["components"] != {"D": s["d_run"], "P1": s["p1_run"]}:
            raise AssertionError(f"{p.stem}: components differ from the spec")
        if s["d_run"] != d_run[(B, r, seed)]:
            raise AssertionError(f"{p.stem}: D component is not the sweep D run of its cell")
        if s["p1_run"] != p1_cell[(B, r, seed)]:
            raise AssertionError(f"{p.stem}: P1 component is not the p1v2b run of its cell")
        path = ROOT / pl["predictions"]["path"]
        if c.sha256(path) != pl["predictions"]["sha256"]:
            raise AssertionError(f"{p.stem}: prediction sha256")
        d = pd.read_parquet(path)
        if not np.isfinite(d[["y_true", "y_pred"]].to_numpy()).all():
            raise AssertionError(f"{p.stem}: non-finite predictions")
        worst = max(worst, abs(float(np.abs(d.y_true - d.y_pred).mean()) - pl["metrics"]["mae"]))
        base = ref[(B, r)]
        if len(d) != len(base) or set(d.site_id) != set(base.index) or d.site_id.duplicated().any():
            raise AssertionError(f"{p.stem}: test sites differ from the S run of the cell")
        j = d.set_index("site_id").loc[base.index]
        if not (np.array_equal(j.y_true.to_numpy(), base.y_true.to_numpy()) and (j.host_id == base.host_id).all()):
            raise AssertionError(f"{p.stem}: targets or hosts differ from the S run of the cell")
        hosts[p.stem] = rec["hostname"]
        timestamps.append(rec["timestamp_utc"])
        frames.append(d.assign(model="P", r=r, B=B, seed=seed, run_id=p.stem, backbone="v2")[
            ["backbone", "model", "r", "B", "seed", "site_id", "host_id", "y_true", "y_pred", "run_id"]])
    if worst > 1e-6:
        raise AssertionError(f"recorded MAE differs from the stored predictions by {worst}")
    if len(cells) != len(BUDGETS) * N_RESAMPLES * len(SEEDS):
        raise AssertionError("pcompv2b grid incomplete")
    pools = pd.Series(hosts).str.rsplit("-", n=2).str[0].value_counts().to_dict()
    info = {"n_runs": len(files), "n_prediction_rows": int(sum(map(len, frames))),
            "max_abs_diff_recorded_mae_vs_predictions": worst, "spec_sha256": c.sha256(ROOT / "specs/pcompv2b.jsonl"),
            "worker_pools": pools, "timestamp_utc_range": [min(timestamps), max(timestamps)],
            "code_commits": sorted({json.loads(p.read_text())["git_sha"] for p in files})}
    return pd.concat(frames, ignore_index=True), info


def swapin_d_runs() -> set:
    pl = c.record("swapin/swapin_v2")["payload"]
    found = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ("D_run", "d_run") and isinstance(v, str):
                    found.add(v)
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(pl)
    return found


def tier(ps: dict) -> dict:
    """ps = paired P - S block (staging_analysis.paired)."""
    a_lo, a_hi = -ps["ci95"][1], -ps["ci95"][0]        # A_P = S - P
    if a_lo > 0:
        t, why = "W", "95% interval of MAE_S - MAE_P lies above zero"
    elif ps["ci95"][0] > 0:
        t, why = "N", "P measurably worse than S (95% interval of MAE_P - MAE_S above zero)"
    elif ps["upper_one_sided_95"] < DELTA:
        t, why = "N", "P not better than S and the one-sided 95% upper bound of MAE_P - MAE_S is below delta"
    else:
        t, why = "I", "P not measurably better, and P - S not bounded below delta"
    return {"tier": t, "reason": why, "A_P_ci95": [a_lo, a_hi]}


def block(ens: pd.DataFrame, draws) -> dict:
    st = {m: sa.stats_by_budget(ens, m, BUDGETS) for m in ("S", "D", "P", "P_old")}
    out = {}
    for B in BUDGETS:
        ps = sa.paired(st["P"][B], st["S"][B], draws)
        ps.update(sa.tost_outcome(ps["ci95"], ps["ci90"], DELTA), noninferior=sa.noninferior(ps["upper_one_sided_95"]))
        pd_ = sa.paired(st["P"][B], st["D"][B], draws)
        sd = sa.paired(st["S"][B], st["D"][B], draws)
        po = sa.paired(st["P"][B], st["P_old"][B], draws)
        row = {"x_training_hosts": B, "mae": {m: sa.marginal(st[m][B], draws) for m in ("S", "D", "P", "P_old")},
               "S_minus_D": sd, "P_minus_S": ps, "P_minus_D": pd_, "P_minus_P_old": po,
               "A_P_S_minus_P": {"mean": -ps["mean"], "ci95": [-ps["ci95"][1], -ps["ci95"][0]],
                                 "upper_one_sided_95": -ps["lower_one_sided_95"],
                                 "lower_one_sided_95": -ps["upper_one_sided_95"]},
               "tolerance": {"P_minus_S_upper_below_delta": bool(ps["upper_one_sided_95"] < DELTA),
                             "P_minus_S_lower_above_minus_delta": bool(ps["lower_one_sided_95"] > -DELTA),
                             "P_minus_D_upper_below_delta": bool(pd_["upper_one_sided_95"] < DELTA),
                             "P_minus_D_lower_above_minus_delta": bool(pd_["lower_one_sided_95"] > -DELTA)},
               "tier": tier(ps)}
        out[str(B)] = row
    return out


def swapin_change(long: pd.DataFrame, res: dict) -> dict:
    """Site-level change from P to P-with-true-descriptors (= D), seed-ensemble means."""
    from dftgnn.stats.g1 import seed_ensemble

    ens = seed_ensemble(long[long.model.isin(["P", "D"])], keys=("model", "r", "B", "site_id"))
    w = ens.pivot_table(index=["B", "r", "site_id"], columns="model", values="y_pred")
    out = {}
    for B in BUDGETS:
        g = w.xs(B, level="B")
        dd = (g["D"] - g["P"]).abs()
        pm = res[str(B)]["P_minus_D"]
        out[str(B)] = {"mae_change_true_minus_predicted_eV": {"mean": -pm["mean"],
                                                              "ci95": [-pm["ci95"][1], -pm["ci95"][0]]},
                       "mean_abs_site_change_eV": float(dd.mean()), "median_abs_site_change_eV": float(dd.median()),
                       "max_abs_site_change_eV": float(dd.max()), "n_site_cells": len(dd)}
    return out


def reproduce(res: dict) -> dict:
    ref = c.record("staging_v2")["payload"]["budgets"]
    d = []
    for B in BUDGETS:
        k = str(B)
        for m in ("S", "D"):
            d.append(abs(res[k]["mae"][m]["mean"] - ref[k]["mae"][m]["mean"]))
            d += list(np.abs(np.subtract(res[k]["mae"][m]["ci95"], ref[k]["mae"][m]["ci95"])))
        d.append(abs(res[k]["mae"]["P_old"]["mean"] - ref[k]["mae"]["P"]["mean"]))
        d.append(abs(res[k]["P_minus_D"]["mean"] - res[k]["P_minus_P_old"]["mean"] + ref[k]["D_minus_P"]["mean"]))
    worst = float(max(d))
    if worst > REPRO_TOL:
        raise AssertionError(f"S, D and the previous P do not reproduce results/staging_v2.json: {worst}")
    return {"reference": "results/staging_v2.json payload.budgets (MAE S, D with ci95; previous P; D - P)",
            "n_values_compared": len(d), "max_abs_difference": worst, "reproduced": True}


def main() -> None:
    from dftgnn.config import load_config
    from dftgnn.io.results import write_result

    cfg = load_config()
    if float(cfg.analysis.equivalence_margin_eV) != DELTA:
        raise AssertionError("analysis.equivalence_margin_eV differs from 0.05")
    sweep, f_long = load_sweep()
    p1_cell, p1info = load_p1()
    pc, info = load_pcomp(sweep, p1_cell)
    d_runs = {json.loads((ROOT / "results/pcompv2b" / f"{i}.json").read_text())["payload"]["spec"]["d_run"]
              for i in pc.run_id.unique()}
    missing_d = d_runs - swapin_d_runs()
    if missing_d:
        raise AssertionError(f"{len(missing_d)} D runs of pcompv2b are not covered by results/swapin/swapin_v2.json")
    old = sweep[sweep.model == "P"].assign(model="P_old")
    long = pd.concat([sweep[sweep.model.isin(["S", "D"])], old, pc], ignore_index=True)
    long = long[long.r >= 0]
    ens = sa.ensemble(long.drop(columns=["run_id"]))
    draws = sa.make_draws(ens)
    res = block(ens, draws)
    repro = reproduce(res)
    p1 = p1_block(p1info["table"])
    payload = {"label": LABEL, "delta_eV": DELTA,
               "design": "10 resamples x 3 seeds (the sweep design), outer resamples r = 0-9, seed-ensemble mean "
                         "over seeds 0-2, 2000 hierarchical bootstrap replicates, seed 20261008, draws shared by S, "
                         "D, P and the previous P at every budget",
               "sign_convention": "A = MAE_S - MAE_D; A_P = MAE_S - MAE_P; P_minus_S = MAE_P - MAE_S; "
                                  "P_minus_D = MAE_P - MAE_D (positive = P less accurate)",
               "tier_rule": "W: ci95 of A_P above zero; N: P worse than S (ci95 of P - S above "
                            "zero) or one-sided 95% upper bound of P - S below delta; I: otherwise",
               "tier_at_654": res["654"]["tier"], "budgets": res,
               "swap_in": {"rule": "P with true descriptors equals D (results/swapin/swapin_v2.json: 183 pairs, max "
                                   "|difference| 0.0 eV, identity standardisation); every D component of pcompv2b "
                                   "is covered by that record; change = D minus P, from stored predictions",
                           "per_budget": swapin_change(long, res)},
               "p1": {"retrained": p1, "previous_p1_v2": old_p1_block(),
                      "n_at_epoch_limit": len(p1info["capped"]), "runs_at_epoch_limit": p1info["capped"],
                      "spec_sha256": p1info["spec_sha256"]},
               "health": info, "reproduction": repro,
               "inputs": {"staging_eval_long": f_long, "tuned_p1": "configs/tuned_p1_v2.yaml",
                          "swapin": "results/swapin/swapin_v2.json"}}
    payload = c.to_jsonable(payload)
    if "--dry-run" in sys.argv:
        print(json.dumps({k: payload[k] for k in ("tier_at_654", "reproduction", "health")}, indent=1))
        for B in BUDGETS:
            r = payload["budgets"][str(B)]
            print(B, {m: round(r["mae"][m]["mean"], 3) for m in r["mae"]}, "P-S", round(r["P_minus_S"]["mean"], 3),
                  [round(x, 3) for x in r["P_minus_S"]["ci95"]], "U", round(r["P_minus_S"]["upper_one_sided_95"], 3),
                  "P-D", round(r["P_minus_D"]["mean"], 3), [round(x, 3) for x in r["P_minus_D"]["ci95"]],
                  r["tier"]["tier"], "p1", round(payload["p1"]["retrained"][str(B)]["mean_standardised_mae"], 3),
                  round(payload["p1"]["previous_p1_v2"][str(B)]["mean_standardised_mae"], 3))
        return
    print(write_result("staging_fair", payload, config=cfg, repo_root=ROOT, results_dir=OUT_DIR))


if __name__ == "__main__":
    main()
