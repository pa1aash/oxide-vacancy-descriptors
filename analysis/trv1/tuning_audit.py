"""Audit of the tuning and training asymmetry between the structure-only network S and the re-tuned descriptor
network D*, and of the GPU classes behind the headline runs.

Nothing is trained. Physically: A* = MAE_S - MAE_D* is the error of the neutral oxygen-vacancy formation energy
removed by the pristine-cell electronic descriptors. D* was tuned in a wider space (dropout 0-0.5, weight decay
1e-7 to 1e-2, injection route, per-class descriptor encoders) and trained with a 400-epoch limit; S was tuned in the
shared space (dropout 0-0.3, weight decay 1e-6 to 1e-3) with a 200-epoch limit. If S's optimum sat on an edge of its
space, if D*'s validation optimum needed the widened regularisation ranges, if D* gained from epochs past 200, or if
the capped S runs were still improving when they stopped, the asymmetry would enlarge A* without any descriptor
physics behind it. The last block asks whether the mix of GPU classes behind the runs could do the same.

Inputs (validation errors only; no test-host label enters blocks 1-4)
    shared space      configs/config.yaml tuning.search_space (Appendix B, Table B1, column "S, D and P1")
    D* space          results/dtune_v2/*.json payload.spec.search_space (Table B1, column D*)
    tuning studies    results/tuning/trials_v2_S_a{50,200,654}.csv (S) and trials_v2_D_a{50,200,654}.csv (the D*
                      study: columns injection, desc_layers, desc_width; max_epochs 400), with their study records
                      results/tuning/tuning_v2_{S,D}_a<anchor>.json. The three tuning sizes are 50, 200 and 654
                      hosts; training sizes 400 and 654 both take the 654-host values (tuning.budget_to_anchor).
    run records       S and D at 654 hosts, 30 resamples x 5 seeds: results/precision_v2 and results/v2_sweep, run ids
                      from results/analysis/precision_cv_runs.parquet (as in analysis/dsens/dsens.py); D*:
                      results/dretrain_v2 (spec.budget 654, 30 x 5; 400 and 200, 10 x 3). payload.history holds
                      epoch (0-based), lr, train_loss, val_metric (validation MAE, eV); best_epoch is 0-based.

Blocks
    s_optima           per tuning size: S's selected trial, its dropout, weight decay and learning rate, each as a
                       position in the shared range (linear for dropout, log for weight decay and learning rate) and
                       ratio to the bounds; interior = position in [0.05, 0.95], the study records' boundary rule;
                       S trials with epochs_run == max_epochs and whether the selected trial is one of them.
    dstar_neighbours   per tuning size: the best D* trial and every trial within 0.001 eV and 0.005 eV of its
                       validation objective, each flagged inside / outside the shared dropout and weight-decay
                       ranges, with its route and encoder; the best trial inside both shared ranges and its gap to
                       the overall best; the best trial that is also D-shaped (initial-state route, no encoder).
    dstar_late_epochs  D* runs: number whose best validation epoch is after epoch 200 (0-based index >= 200), and
                       the gain from epochs past 200 = min val over epochs 0-199 minus min val over all epochs.
    s_cap_tail         S (and D) runs at 654 hosts that stop at their 200-epoch limit (epochs_run == max_epochs):
                       validation gain over the last 30 epochs = best-so-far val after epoch max - 31 (epochs
                       0 to max - 31) minus the final best val; learning rate at the last epoch / initial rate.
    gpu_classes        GPU model counts per model (S, D, D*) at 654 hosts (30 x 5) from payload.hardware.gpu_model,
                       runs without a hardware block counted apart; single-run test site MAE gap S - D* per (r, seed)
                       for same-class and cross-class pairs; the bitwise-repeatability record
                       results/nrp/nrp_n1_gpu_repeat.json summarised with pointers.

Every stored value that is recounted here (study best trials and values, capped-trial lists, tuned values in
configs/, the capped-cell lists of results/derived_dsens/dsens.json, the record MAEs against the prediction parquets,
the history best epochs) is compared in the reproduction block. Output: dry run to --out FILE; otherwise
write_result("tuning_audit") into results/derived_trv1_tuning/.

    PYTHONPATH=$PWD/src python analysis/trv1/tuning_audit.py [--dry-run --out FILE]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis import a1_common as c

OUT_DIR = ROOT / "results" / "derived_trv1_tuning"
TUN = ROOT / "results" / "tuning"
DTUNE = ROOT / "results" / "dtune_v2"
DRET = ROOT / "results" / "dretrain_v2"
RUN_DIRS = {"precision": ROOT / "results" / "precision_v2", "sweep": ROOT / "results" / "v2_sweep"}
NRP = ROOT / "results" / "nrp" / "nrp_n1_gpu_repeat.json"
DSENS = ROOT / "results" / "derived_dsens" / "dsens.json"
ANCHORS = (50, 200, 654)
B_HEAD, N_R, N_S = 654, 30, 5
DSTAR_GRID = {654: (30, 5), 400: (10, 3), 200: (10, 3)}
EDGE = 0.05            # boundary rule of the study records: within 5% of the (log) range of an edge
TAIL = 30              # last epochs examined in capped runs (= early-stopping patience)
LATE = 200             # epoch limit of S and D
NEAR = (0.001, 0.005)  # eV, near-tie windows
TABLE_CLASS = {"NVIDIA A10": "A10", "NVIDIA GeForce RTX 3090": "RTX 3090", "NVIDIA GeForce RTX 4090": "RTX 4090",
               "NVIDIA L4": "L4"}


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def stats(x) -> dict:
    a = np.asarray(list(x), float)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)), "min": float(a.min()),
            "max": float(a.max())}


def position(x: float, dim: dict) -> dict:
    lo, hi = float(dim["low"]), float(dim["high"])
    if dim["dist"] == "log_uniform":
        pos = math.log(x / lo) / math.log(hi / lo)
        out = {"scale": "log", "ratio_to_low": x / lo, "ratio_to_high": x / hi}
    else:
        pos = (x - lo) / (hi - lo)
        out = {"scale": "linear", "ratio_to_high": x / hi}
    return {"value": float(x), "low": lo, "high": hi, "position_in_range": pos, **out,
            "interior": bool(EDGE <= pos <= 1 - EDGE), "inside_range": bool(lo <= x <= hi)}


def study(model: str, a: int) -> tuple[pd.DataFrame, dict]:
    t = pd.read_csv(TUN / f"trials_v2_{model}_a{a}.csv")
    p = json.loads((TUN / f"tuning_v2_{model}_a{a}.json").read_text())["payload"]
    if len(t) != 30 or set(t.state) != {"COMPLETE"} or t.nonfinite.any() or not np.isfinite(t.objective).all():
        raise AssertionError(f"{model} a{a}: study table not 30 finite COMPLETE trials")
    if model == "D" and not {"injection", "desc_layers", "desc_width"} <= set(t.columns):
        raise AssertionError("trials_v2_D is not the D* study")
    return t, p


# ---------------------------------------------------------------- block 1

def s_optima(shared: dict, tuned_s: dict, repro: list) -> dict:
    out = {}
    for a in ANCHORS:
        t, p = study("S", a)
        best = t.loc[t.objective.idxmin()]
        cap = int(p["max_epochs"])
        capped = sorted(int(x) for x in t.trial[t.epochs_run == cap])
        top5 = t.nsmallest(5, "objective")
        repro += [
            (f"s_optima.{a}.best_trial", int(best.trial), int(p["best_trial"]), rel(TUN / f"tuning_v2_S_a{a}.json")
             + " payload.best_trial"),
            (f"s_optima.{a}.best_value", float(best.objective), float(p["best_value"]),
             rel(TUN / f"tuning_v2_S_a{a}.json") + " payload.best_value"),
            (f"s_optima.{a}.capped_trials", capped, sorted(int(x) for x in p["trials_reaching_cap"]),
             rel(TUN / f"tuning_v2_S_a{a}.json") + " payload.trials_reaching_cap"),
        ]
        for k in ("dropout", "weight_decay", "learning_rate"):
            repro.append((f"s_optima.{a}.{k}_vs_configs", float(best[k]), float(tuned_s[a][k]),
                          f"configs/tuned_v2.yaml models.S.{a}.{k}"))
        out[str(a)] = {
            "tuning_hosts": a, "selected_trial": int(best.trial), "val_mae_eV": float(best.objective),
            "dropout": position(float(best.dropout), shared["dropout"]),
            "weight_decay": position(float(best.weight_decay), shared["weight_decay"]),
            "learning_rate": position(float(best.learning_rate), shared["learning_rate"]),
            "all_three_interior": bool(all(position(float(best[k]), shared[k])["interior"]
                                           for k in ("dropout", "weight_decay", "learning_rate"))),
            "boundary_hits_in_study_record": p["boundary_hits"],
            "max_epochs": cap, "selected_best_epoch": int(best.best_epoch), "selected_epochs_run": int(best.epochs_run),
            "selected_capped": bool(int(best.epochs_run) == cap),
            "n_trials": len(t), "n_trials_capped": len(capped), "capped_trials": capped,
            "n_top5_capped": int((top5.epochs_run == cap).sum()),
            "capped_trials_val_mae_eV": {str(int(r.trial)): float(r.objective)
                                         for r in t[t.epochs_run == cap].itertuples()},
        }
    return out


# ---------------------------------------------------------------- block 2

def trial_row(r, best_val: float, shared: dict) -> dict:
    d_in = shared["dropout"]["low"] <= r.dropout <= shared["dropout"]["high"]
    w_in = shared["weight_decay"]["low"] <= r.weight_decay <= shared["weight_decay"]["high"]
    return {"trial": int(r.trial), "val_mae_eV": float(r.objective), "gap_to_best_eV": float(r.objective - best_val),
            "dropout": float(r.dropout), "weight_decay": float(r.weight_decay), "learning_rate": float(r.learning_rate),
            "dropout_in_shared": bool(d_in), "weight_decay_in_shared": bool(w_in),
            "both_in_shared": bool(d_in and w_in), "route": r.injection, "encoder_layers": int(r.desc_layers),
            "encoder_width": int(r.desc_width), "best_epoch": int(r.best_epoch), "epochs_run": int(r.epochs_run)}


def dstar_neighbours(shared: dict, tuned_d: dict, repro: list) -> dict:
    out = {}
    for a in ANCHORS:
        t, p = study("D", a)
        _, pd_ = study("D-state", a)
        t = t.sort_values("objective").reset_index(drop=True)
        best = t.iloc[0]
        bv = float(best.objective)
        rows = [trial_row(r, bv, shared) for r in t.itertuples()]
        repro += [
            (f"dstar_neighbours.{a}.best_trial", int(best.trial), int(p["best_trial"]),
             rel(TUN / f"tuning_v2_D_a{a}.json") + " payload.best_trial"),
            (f"dstar_neighbours.{a}.best_value", bv, float(p["best_value"]),
             rel(TUN / f"tuning_v2_D_a{a}.json") + " payload.best_value"),
            (f"dstar_neighbours.{a}.capped_trials", sorted(int(x) for x in t.trial[t.epochs_run == int(p["max_epochs"])]),
             sorted(int(x) for x in p["trials_reaching_cap"]), rel(TUN / f"tuning_v2_D_a{a}.json")
             + " payload.trials_reaching_cap"),
        ]
        for k in ("dropout", "weight_decay", "learning_rate"):
            repro.append((f"dstar_neighbours.{a}.{k}_vs_configs", float(best[k]), float(tuned_d[a][k]),
                          f"configs/tuned_d_v2.yaml models.D.{a}.{k}"))
        inside = [r for r in rows if r["both_in_shared"]]
        dshape = [r for r in inside if r["route"] == "D-state" and r["encoder_layers"] == 0]
        win = {}
        for w in NEAR:
            near = [r for r in rows[1:] if r["gap_to_best_eV"] <= w]
            win[f"within_{w:g}_eV"] = {"n": len(near), "n_both_in_shared": sum(r["both_in_shared"] for r in near),
                                       "trials": near}
        out[str(a)] = {
            "tuning_hosts": a, "max_epochs": int(p["max_epochs"]), "best": rows[0], **win,
            "best_in_shared_ranges": {**inside[0], "gap_to_overall_best_eV": inside[0]["gap_to_best_eV"]}
            if inside else None,
            "best_in_shared_ranges_D_state_no_encoder": dshape[0] if dshape else None,
            "n_trials_in_shared_ranges": len(inside),
            "D_study_best_val_mae_eV": float(pd_["best_value"]),
            "D_study_pointer": rel(TUN / f"tuning_v2_D-state_a{a}.json") + " payload.best_value",
            "best_in_shared_minus_D_best_eV": (inside[0]["val_mae_eV"] - float(pd_["best_value"])) if inside else None,
            "n_trials_epochs_run_above_200": int((t.epochs_run > LATE).sum()),
            "n_trials_best_epoch_at_or_after_200": int((t.best_epoch >= LATE).sum()),
        }
    return out


# ---------------------------------------------------------------- run records

def history_facts(pl: dict, tag: str, repro_hist: list) -> tuple[np.ndarray, np.ndarray]:
    h = pl["history"]
    ep = np.array([e["epoch"] for e in h])
    val = np.array([e["val_metric"] for e in h], float)
    lr = np.array([e["lr"] for e in h], float)
    if len(h) != int(pl["epochs_run"]) or not (ep == np.arange(len(h))).all() or not np.isfinite(val).all():
        raise AssertionError(f"{tag}: history length / epochs / finiteness")
    be = int(np.argmin(val))
    if be != int(pl["best_epoch"]):
        raise AssertionError(f"{tag}: argmin of history {be} != best_epoch {pl['best_epoch']}")
    repro_hist.append(abs(float(val.min()) - float(pl["best_val_metric"])))
    if abs(lr[0] - float(pl["spec"]["lr"])) > 1e-15:
        raise AssertionError(f"{tag}: initial lr differs from spec.lr")
    return val, lr


def load_dstar() -> list[dict]:
    runs = []
    for p in sorted(DRET.glob("*.json")):
        pl = json.loads(p.read_text())["payload"]
        s = pl["spec"]
        B, r, seed = int(s["budget"]), int(s["r"]), int(s["seed"])
        if B not in DSTAR_GRID:
            continue
        n_r, n_s = DSTAR_GRID[B]
        if not (0 <= r < n_r and 0 <= seed < n_s) or int(pl["max_epochs"]) != 400:
            raise AssertionError(f"{p.stem}: cell or max_epochs")
        runs.append({"B": B, "r": r, "seed": seed, "pl": pl, "file": rel(p)})
    for B, (n_r, n_s) in DSTAR_GRID.items():
        cells = {(x["r"], x["seed"]) for x in runs if x["B"] == B}
        if len(cells) != n_r * n_s or sum(x["B"] == B for x in runs) != n_r * n_s:
            raise AssertionError(f"D* at {B}: {len(cells)} cells")
    return runs


def load_sd() -> tuple[dict, dict]:
    runs, f_runs = c.load_table("precision_cv_runs")
    p = runs[(runs.backbone == "v2") & (runs.design == "precision") & (runs.B == B_HEAD)
             & (runs.r < N_R) & (runs.seed < N_S)]
    name = {"S": "S", "D": "D-state"}
    out = {}
    for m in ("S", "D"):
        g = p[p.model == m]
        if len(g) != N_R * N_S or len(set(zip(g.r, g.seed))) != N_R * N_S:
            raise AssertionError(f"{m}: not a full 30 x 5 grid")
        rows = []
        for row in g.itertuples(index=False):
            f = RUN_DIRS[row.source] / f"{row.run_id}.json"
            pl = json.loads(f.read_text())["payload"]
            s = pl["spec"]
            if (pl["run_id"] != row.run_id or s["model"] != name[m] or int(s["budget"]) != B_HEAD
                    or int(s["r"]) != row.r or int(s["seed"]) != row.seed):
                raise AssertionError(f"{row.run_id}: table differs from record")
            rows.append({"r": int(row.r), "seed": int(row.seed), "source": row.source, "run_id": row.run_id,
                         "table_gpu_class": row.gpu_class, "pl": pl, "file": rel(f)})
        out[m] = rows
    return out, f_runs


# ---------------------------------------------------------------- block 3

def dstar_late_epochs(dstar: list[dict], repro_hist: list) -> dict:
    out = {}
    for B in (654, 400, 200):
        gains, late, above, ids = [], 0, 0, []
        for x in (y for y in dstar if y["B"] == B):
            val, _ = history_facts(x["pl"], x["file"], repro_hist)
            g = float(val[:LATE].min() - val.min())
            gains.append(g)
            above += int(len(val) > LATE)
            if int(x["pl"]["best_epoch"]) >= LATE:
                late += 1
                ids.append({"r": x["r"], "seed": x["seed"], "run_id": x["pl"]["run_id"],
                            "best_epoch": int(x["pl"]["best_epoch"]), "epochs_run": int(x["pl"]["epochs_run"]),
                            "gain_eV": g})
        n_r, n_s = DSTAR_GRID[B]
        out[str(B)] = {"x_training_hosts": B, "design": f"{n_r} x {n_s}", "n_runs": len(gains), "max_epochs": 400,
                       "n_runs_epochs_run_above_200": above, "n_best_after_epoch_200": late,
                       "gain_past_200_eV": stats(gains), "gain_past_200_eV_runs_with_late_best": stats(
                           [r["gain_eV"] for r in ids]), "runs_with_late_best": ids,
                       "n_at_epoch_limit_400": sum(int(y["pl"]["epochs_run"]) == 400 for y in dstar if y["B"] == B)}
    return out


# ---------------------------------------------------------------- block 4

def tail(rows: list[dict], repro_hist: list) -> dict:
    capped = [x for x in rows if int(x["pl"]["epochs_run"]) == int(x["pl"]["max_epochs"])]
    gains, fr, last_best, per = [], [], 0, []
    for x in capped:
        val, lr = history_facts(x["pl"], x["file"], repro_hist)
        mx = int(x["pl"]["max_epochs"])
        g = float(val[: mx - TAIL].min() - val.min())
        f = float(lr[-1] / lr[0])
        gains.append(g)
        fr.append(f)
        last_best += int(int(x["pl"]["best_epoch"]) == mx - 1)
        per.append({"r": x["r"], "seed": x["seed"], "run_id": x["run_id"], "source": x["source"],
                    "best_epoch": int(x["pl"]["best_epoch"]), "gain_last_30_eV": g, "final_best_val_eV": float(val.min()),
                    "lr_last_over_initial": f, "lr_halvings": round(-math.log2(f))})
    uncapped = [x for x in rows if x not in capped]
    return {"n_runs": len(rows), "max_epochs": sorted({int(x["pl"]["max_epochs"]) for x in rows}),
            "n_capped": len(capped), "n_capped_by_source": {k: sum(x["source"] == k for x in capped)
                                                            for k in sorted({x["source"] for x in rows})},
            "gain_last_30_eV": stats(gains), "lr_last_over_initial": stats(fr),
            "lr_last_over_initial_values": sorted({round(v, 10) for v in fr}),
            "n_best_epoch_is_last_epoch": last_best, "runs": per,
            "uncapped_epochs_run": stats(int(x["pl"]["epochs_run"]) for x in uncapped)}


# ---------------------------------------------------------------- block 5

def mae_parquet(path: Path) -> tuple[float, frozenset]:
    d = pd.read_parquet(path)
    return float(np.abs(d.y_true - d.y_pred).mean()), frozenset(d.site_id)


def gm(pl: dict):
    return (pl.get("hardware") or {}).get("gpu_model")


def gpu_classes(sd: dict, dstar: list[dict], repro: list) -> dict:
    counts = {}
    head = {(x["r"], x["seed"]): x for x in dstar if x["B"] == B_HEAD}
    for m, rows in (("S", sd["S"]), ("D", sd["D"]), ("Dstar", list(head.values()))):
        hw = [gm(x["pl"]) for x in rows]
        known = pd.Series([h for h in hw if h]).value_counts()
        counts[m] = {"n_runs": len(rows), "n_no_hardware_block": sum(h is None for h in hw),
                     "by_gpu_model": {k: int(v) for k, v in known.items()}, "n_classes": len(known)}
        if m != "Dstar":
            no = [x for x in rows if gm(x["pl"]) is None]
            counts[m]["no_hardware_by_source"] = {k: sum(x["source"] == k for x in no) for k in ("precision", "sweep")}
            counts[m]["no_hardware_table_gpu_class"] = sorted({x["table_gpu_class"] for x in no})
            mism = sum(1 for x in rows if gm(x["pl"]) and TABLE_CLASS.get(gm(x["pl"])) != x["table_gpu_class"])
            repro.append((f"gpu_classes.{m}.table_gpu_class_mismatches", mism, 0,
                          "results/analysis/precision_cv_runs.parquet column gpu_class"))

    # single-run site MAE: S from precision_eval_long (backbone v2, 654), D* from its prediction parquet
    long = pd.read_parquet(ROOT / "results" / "analysis" / "precision_eval_long.parquet")
    long = long[(long.backbone == "v2") & (long.B == B_HEAD) & (long.model == "S")]
    s_mae = long.assign(ae=(long.y_true - long.y_pred).abs()).groupby("run_id").ae.mean()
    s_sites = long.groupby("run_id").site_id.agg(frozenset)
    worst_s, worst_d = 0.0, 0.0
    same, diff, unk, all_ = [], [], [], []
    by_class = {"S": {}, "Dstar": {}}
    for x in sd["S"]:
        k = (x["r"], x["seed"])
        y = head[k]
        sm = float(s_mae[x["run_id"]])
        worst_s = max(worst_s, abs(sm - float(x["pl"]["metrics"]["mae"])))
        dm, dsites = mae_parquet(ROOT / y["pl"]["predictions"]["path"])
        worst_d = max(worst_d, abs(dm - float(y["pl"]["metrics"]["mae"])))
        if dsites != s_sites[x["run_id"]]:
            raise AssertionError(f"{k}: S and D* test sites differ")
        g = sm - dm
        all_.append(g)
        cs, cd = gm(x["pl"]), gm(y["pl"])
        if cs:
            by_class["S"].setdefault(cs, []).append(sm)
        by_class["Dstar"].setdefault(cd, []).append(dm)
        (unk if cs is None or cd is None else same if cs == cd else diff).append(g)
    repro += [("gpu_classes.S_record_mae_vs_eval_long_max_abs", worst_s, 0.0,
               "results/precision_v2|v2_sweep payload.metrics.mae vs results/analysis/precision_eval_long.parquet"),
              ("gpu_classes.Dstar_record_mae_vs_parquet_max_abs", worst_d, 0.0,
               "results/dretrain_v2 payload.metrics.mae vs payload.predictions.path")]
    gap = {"definition": "per (r, seed) at 654 hosts: test site MAE of the single S run minus that of the single D* "
                         "run of the same cell (same test sites); classes from payload.hardware.gpu_model",
           "all_pairs": stats(all_), "same_class": stats(same), "different_class": stats(diff),
           "S_without_hardware_block": stats(unk),
           "same_minus_different_eV": float(np.mean(same) - np.mean(diff)) if same and diff else None,
           "single_run_mae_by_class": {m: {k: stats(v) for k, v in d.items()} for m, d in by_class.items()}}

    n = json.loads(NRP.read_text())["payload"]
    ens = [e for e in n["ensemble"] if e["pass"] == "pass1"]
    shift = {}
    for e in ens:
        key = f"{e['model']}_{e['B']}"
        cur = shift.get(key)
        if cur is None or abs(e["ensemble_mae_minus_ref"]) > abs(cur["max_abs_shift_eV"]):
            shift[key] = {"max_abs_shift_eV": float(abs(e["ensemble_mae_minus_ref"])), "class": e["class"]}
    spread = {}
    for key in sorted({f"{e['model']}_{e['B']}" for e in ens}):
        per = {}
        for e in ens:
            if f"{e['model']}_{e['B']}" == key:
                per.setdefault(e["class"], e["ensemble_mae"])
        spread[key] = {"ensemble_mae_by_class": per, "max_minus_min_eV": float(max(per.values()) - min(per.values()))}
    top = max(ens, key=lambda e: abs(e["ensemble_mae_minus_ref"]))
    top654 = max((e for e in ens if e["B"] == 654), key=lambda e: abs(e["ensemble_mae_minus_ref"]))
    nrp = {"record": rel(NRP), "n_runs": int(n["n_runs"]), "classes": n["classes"],
           "classes_without_runs": n["classes_without_runs"],
           "same_node": {**n["same_node_repeat_summary"], "pointer": "payload.same_node_repeat_summary"},
           "cross_node_same_class": {**n["cross_node_same_class_summary"],
                                     "pointer": "payload.cross_node_same_class_summary"},
           "cross_class_max_abs_ensemble_mae_shift": {
               "definition": "max over payload.ensemble (pass1) of |ensemble_mae_minus_ref|: seed-ensemble MAE of "
                             "the repeat on an NRP class minus that of the original runs (ref_run_id; their records "
                             "carry no hardware block, so their GPU class is unrecorded)",
               "max_eV": float(abs(top["ensemble_mae_minus_ref"])), "at": {"model": top["model"], "B": top["B"],
                                                                         "class": top["class"]},
               "max_at_654_eV": float(abs(top654["ensemble_mae_minus_ref"])),
               "at_654": {"model": top654["model"], "class": top654["class"]}, "by_model_B": shift,
               "pointer": "payload.ensemble[*].ensemble_mae_minus_ref"},
           "between_nrp_classes_ensemble_mae_spread": spread}
    return {"counts_654_30x5": counts, "single_run_gap_S_minus_Dstar": gap, "repeatability": nrp}


# ---------------------------------------------------------------- main

def cmp_rows(rows: list) -> tuple[dict, float]:
    out, worst = {}, 0.0
    for key, got, ref, ptr in rows:
        if isinstance(got, list) or isinstance(ref, list):
            ok = got == ref
            d = 0.0 if ok else float("inf")
        else:
            d = abs(float(got) - float(ref))
            ok = d <= 1e-9
        worst = max(worst, d)
        out[key] = {"recomputed": got, "stored": ref, "abs_diff": d if math.isfinite(d) else None,
                    "reproduced": bool(ok), "stored_at": ptr}
    return out, worst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    t0 = time.time()
    cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
    shared = cfg["tuning"]["search_space"]
    if int(cfg["training"]["max_epochs"]) != LATE or int(cfg["training"]["early_stopping"]["patience"]) != TAIL:
        raise AssertionError("training.max_epochs / patience differ from 200 / 30")
    if list(cfg["tuning"]["anchors"]) != list(ANCHORS):
        raise AssertionError("tuning anchors differ")
    tuned_s = yaml.safe_load((ROOT / "configs" / "tuned_v2.yaml").read_text())["models"]["S"]
    tuned_d = yaml.safe_load((ROOT / "configs" / "tuned_d_v2.yaml").read_text())["models"]["D"]
    dspace = {}
    for p in sorted(DTUNE.glob("*.json")):
        s = json.loads(p.read_text())["payload"]["spec"]
        dspace[int(s["anchor"])] = s["search_space"]
    if sorted(dspace) != list(ANCHORS) or any(dspace[a] != dspace[654] for a in ANCHORS):
        raise AssertionError("D* search space differs between anchors")
    ds = dspace[654]

    repro, repro_hist = [], []
    b1 = s_optima(shared, tuned_s, repro)
    b2 = dstar_neighbours(shared, tuned_d, repro)
    dstar = load_dstar()
    sd, f_runs = load_sd()
    b3 = dstar_late_epochs(dstar, repro_hist)
    b4 = {"S": tail(sd["S"], repro_hist), "D": tail(sd["D"], repro_hist)}
    b5 = gpu_classes(sd, dstar, repro)

    st = json.loads(DSENS.read_text())["payload"]["epoch_limit_654_30x5"]
    for m in ("S", "D"):
        ref = st["S_D_runs"][m]
        got = sorted((x["r"], x["seed"], x["run_id"]) for x in b4[m]["runs"])
        want = sorted((x["r"], x["seed"], x["run_id"]) for x in ref["cells_at_limit"])
        repro += [(f"s_cap_tail.{m}.n_capped", b4[m]["n_capped"], int(ref["n_at_epoch_limit"]),
                   f"results/derived_dsens/dsens.json payload.epoch_limit_654_30x5.S_D_runs.{m}.n_at_epoch_limit"),
                  (f"s_cap_tail.{m}.capped_cells", [list(x) for x in got], [list(x) for x in want],
                   f"results/derived_dsens/dsens.json payload.epoch_limit_654_30x5.S_D_runs.{m}.cells_at_limit")]
    for B in (654, 400, 200):
        repro.append((f"dstar_late_epochs.{B}.n_at_epoch_limit_400", b3[str(B)]["n_at_epoch_limit_400"],
                      int(st["Dstar_at_limit"][str(B)]["n_at_epoch_limit"]),
                      f"results/derived_dsens/dsens.json payload.epoch_limit_654_30x5.Dstar_at_limit.{B}"))
    repro.append(("history.max_abs_min_val_minus_best_val_metric", max(repro_hist), 0.0,
                  "run records payload.best_val_metric vs min of payload.history val_metric"))
    rep, worst = cmp_rows(repro)
    bad = [k for k, v in rep.items() if not v["reproduced"]]
    if bad:
        raise AssertionError(f"not reproduced: {bad}")

    payload = {
        "label": "tuning and training asymmetry between S and D* (search space, epoch limit, capped runs) and GPU "
                 "classes of the 654-host headline runs",
        "search_spaces": {
            "shared_S_D_P1": {k: shared[k] for k in ("dropout", "weight_decay", "learning_rate")},
            "shared_pointer": "configs/config.yaml tuning.search_space (Appendix B Table B1)",
            "Dstar": {k: {kk: ds[k][kk] for kk in ("dist", "low", "high")}
                      for k in ("dropout", "weight_decay", "learning_rate")},
            "Dstar_categorical": {k: ds[k]["choices"] for k in ("injection", "desc_layers", "desc_width")},
            "Dstar_pointer": "results/dtune_v2/*.json payload.spec.search_space (identical at 50, 200, 654)",
            "max_epochs": {"S_D_P1": LATE, "Dstar": 400}, "patience": TAIL,
            "tuning_hosts": list(ANCHORS),
            "training_to_tuning_hosts": cfg["tuning"]["budget_to_anchor"],
        },
        "s_optima": b1, "dstar_neighbours": b2, "dstar_late_epochs": b3, "s_cap_tail": b4, "gpu_classes": b5,
        "estimator": {
            "s_optima": "selected trial = minimum validation MAE of the 30-trial study; position = (x - low) / "
                        "(high - low) for dropout, log(x / low) / log(high / low) for weight decay and learning "
                        "rate; interior = position in [0.05, 0.95]; capped trial = epochs_run == max_epochs (200)",
            "dstar_neighbours": "trials of the D* study (trials_v2_D) within 0.001 / 0.005 eV of the best "
                                "validation MAE; inside shared = dropout in [0, 0.3] and weight decay in [1e-6, 1e-3]",
            "dstar_late_epochs": "per run: gain = min(val_metric, epochs 0-199) - min(val_metric, all epochs); late "
                                 "best = 0-based best_epoch >= 200",
            "s_cap_tail": "capped = epochs_run == max_epochs; gain = min(val_metric, epochs 0 to max - 31) - "
                          "min(val_metric, all epochs); lr fraction = lr of the last history epoch / lr of epoch 0",
            "gpu_classes": "class = payload.hardware.gpu_model; single-run gap per (r, seed) = test site MAE of S "
                           "minus that of D* on the same test sites; mean over pairs",
        },
        "sign_convention": "gains are validation MAE decreases in eV (positive = later epochs lowered the validation "
                           "error); gaps to best are validation MAE above the best trial (positive); single-run "
                           "gaps are MAE_S - MAE_D* (positive = D* more accurate)",
        "reproduction": rep,
        "reproduction_summary": {"n_compared": len(rep), "max_abs_difference": worst, "all_reproduced": True},
        "inputs": {"precision_cv_runs": f_runs,
                   "dsens": {"path": rel(DSENS), "sha256": c.sha256(DSENS)},
                   "nrp_n1_gpu_repeat": {"path": rel(NRP), "sha256": c.sha256(NRP)},
                   "tuning_tables": {f"{m}_a{a}": {"path": rel(TUN / f"trials_v2_{m}_a{a}.csv"),
                                                   "sha256": c.sha256(TUN / f"trials_v2_{m}_a{a}.csv")}
                                     for m in ("S", "D") for a in ANCHORS}},
    }
    payload = c.to_jsonable(payload)
    print(f"runtime_s {time.time() - t0:.1f}", file=sys.stderr)
    if args.dry_run:
        text = json.dumps(payload, indent=1, allow_nan=False)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text)
            print(args.out)
        else:
            print(text)
        return
    from dftgnn.io.results import write_result
    print(write_result("tuning_audit", payload, repo_root=ROOT, results_dir=OUT_DIR))


if __name__ == "__main__":
    main()
