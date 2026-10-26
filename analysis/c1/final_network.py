"""Final-network (v2) evidence set, default-initialisation (v1) ablation, LOCO family breakdown.

Nothing is trained. Every estimator is imported unchanged from the analyses that wrote the existing records:

    sweep budgets   ``dftgnn.stats.g1`` (seed-ensemble mean over seeds 0-2, paired hierarchical bootstrap
                    ``HierDraws``, 2000 replicates, seed 20261008) on ``staging_eval_long.parquet`` (S, D, P) and
                    ``eval_table.parquet`` (B0, RF); the draws of ``staging_analysis.make_draws`` and of
                    ``g1_analysis`` are asserted identical
    precision       ``analysis/precision_analysis.analyse`` (30 x 5 and 10 x 3 at B = 400, 654)
    grouped CV      ``analysis/cv_analysis.analyse_backbone`` (paired host-cluster bootstrap over 818 hosts)
    LOCO            ``scripts/analysis/ingest_m1_recompute.loco`` (host bootstrap of the pooled folds)
    spread          the within-host prediction SD of ``scripts/analysis/g1_diagnostics.py``

New quantities (descriptive, no test): the two-sided 95% interval of A checked against +-delta; the LOCO
breakdown by fold and by host family, with leave-one-family-out bounds from the same estimator on the
remaining hosts; the within-host prediction SD of v2 and of P.

Every recomputed number that also sits in an existing record is compared with it (``reproduction`` blocks;
tolerance 1e-9 eV). Writes through ``write_result`` into ``results/derived_c1/``:

    final_network_summary.json    v2: budgets 25-654, 30 x 5 at 400 and 654, CV K = 10 and 20, LOCO
    default_init_ablation.json    v1: the same grid and the site-resolution diagnostic
    loco_family.json              v2 and v1 LOCO by fold and by family

    PYTHONPATH=$PWD/src python analysis/c1/final_network.py [--dry-run] [--data-root ~/Desktop/dft-gnn]
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis import a1_common as c
from analysis import cv_analysis, precision_analysis, staging_analysis

OUT_DIR = ROOT / "results" / "derived_c1"
BUDGETS = (25, 50, 100, 200, 400, 654)
BASELINES = ("B0", "RF-Kumagai", "RF-electronic", "RF-structural")
REPRO_TOL = 1e-9
DELTA = c.DELTA_EV
ARM = {"v2": "final network: He initialisation, validation-tuned (tuned_v2.yaml); called v2 in the repository",
       "v1": "default-initialisation network (ablation); called v1 in the repository, originally registered"}


def ingest_module():
    spec = importlib.util.spec_from_file_location("ingest_m1_recompute",
                                                  ROOT / "scripts" / "analysis" / "ingest_m1_recompute.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- comparison against existing records

def leaves(obj, prefix=""):
    """Numeric and boolean leaves of a nested dict / list as {dotted path: value}."""
    out = {}
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            out.update(leaves(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(leaves(v, f"{prefix}[{i}]"))
    elif isinstance(obj, (bool, np.bool_, int, float, np.integer, np.floating)) and obj is not None:
        out[prefix] = obj
    return out


def compare(mine, ref, source: str) -> dict:
    """Max |difference| over the leaves the two share; booleans must agree; the shared count is reported."""
    a, b = leaves(c.to_jsonable(mine)), leaves(ref)
    shared = sorted(set(a) & set(b))
    if not shared:
        raise AssertionError(f"nothing shared with {source}")
    diffs, bool_bad = [], []
    for k in shared:
        if isinstance(a[k], (bool, np.bool_)) or isinstance(b[k], bool):
            if bool(a[k]) != bool(b[k]):
                bool_bad.append(k)
        else:
            diffs.append(abs(float(a[k]) - float(b[k])))
    mx = float(max(diffs)) if diffs else 0.0
    return {"reference": source, "n_values_compared": len(shared), "max_abs_difference": mx,
            "boolean_mismatches": bool_bad, "reproduced": bool(mx < REPRO_TOL and not bool_bad)}


def compare_pairs(pairs: list[tuple[str, float, float]], source: str) -> dict:
    d = [abs(float(x) - float(y)) for _, x, y in pairs]
    worst = max(range(len(d)), key=d.__getitem__)
    return {"reference": source, "n_values_compared": len(d), "max_abs_difference": float(max(d)),
            "worst": pairs[worst][0], "reproduced": bool(max(d) < REPRO_TOL)}


# ---------------------------------------------------------------- A and its intervals

def a_entry(s: Mapping, upper_key: str = "upper_one_sided") -> dict:
    """A with the two-sided 95% interval and one-sided 95% upper bound, checked against +-delta (strict)."""
    lo, hi = float(s["ci"][0]), float(s["ci"][1])
    up = float(s[upper_key])
    return {"A": float(s["mean"]), "ci95": [lo, hi], "upper_one_sided_95": up,
            "two_sided_within_pm_delta": bool(lo > -DELTA and hi < DELTA),
            "upper_below_delta": bool(up < DELTA),
            "upper_minus_delta": up - DELTA,
            "ci95_excludes_zero": bool(lo > 0 or hi < 0)}


def ms(s: Mapping) -> dict:
    return {"mean": float(s["mean"]), "ci95": [float(s["ci"][0]), float(s["ci"][1])]}


# ---------------------------------------------------------------- sweep budgets (G1 estimator)

def sweep(root: Path) -> tuple[dict, dict, dict]:
    from dftgnn.stats.g1 import HierDraws, contrast, point_metric, resample_stats, summarize

    long, f_long = staging_analysis.load_table("staging_eval_long",
                                               c.record("staging_eval_table", root))
    ens = staging_analysis.ensemble(long)
    draws = staging_analysis.make_draws(ens)
    g1rec = c.record("g1_eval_table", root)["payload"]["files"]["eval_table"]
    if c.sha256(root / g1rec["path"]) != g1rec["sha256"]:
        raise AssertionError("eval_table.parquet differs from results/g1_eval_table.json")
    table = pd.read_parquet(root / g1rec["path"])
    n_hosts = [table[(table.model == "S") & (table.B == 25) & (table.r == r)].host_id.nunique() for r in range(10)]
    g1draws = HierDraws.make(n_hosts, c.N_BOOT, c.BOOT_SEED)
    if not (np.array_equal(g1draws.pick, draws.pick) and np.array_equal(g1draws.w, draws.w)):
        raise AssertionError("staging draws differ from the G1 draws")

    blocks = {}
    for bb in ("v2", "v1"):
        e = ens[ens.backbone == bb]
        st = {m: staging_analysis.stats_by_budget(e, m) for m in ("S", "D", "P")}
        sd = contrast(st["S"], st["D"], draws, "mae", ci=c.CI, upper_quantile=c.UPPER_Q)
        blk = {}
        for B in BUDGETS:
            p = summarize([point_metric(s, "mae") for s in st["P"][B]], draws.replicates(st["P"][B], "mae"),
                          ci=c.CI, upper_quantile=c.UPPER_Q)
            blk[str(B)] = {"MAE_S": sd[B]["a"], "MAE_D": sd[B]["b"], "MAE_P": p, "A": sd[B]["diff"]}
        blocks[bb] = blk
    base = {}
    for m in BASELINES:
        sub = table[table.model == m]
        base[m] = {}
        for B in BUDGETS:
            stl = [resample_stats(sub[(sub.B == B) & (sub.r == r)]) for r in range(10)]
            base[m][str(B)] = summarize([point_metric(s, "mae") for s in stl], draws.replicates(stl, "mae"),
                                        ci=c.CI, upper_quantile=c.UPPER_Q)
    spread = within_host_spread(ens, long)
    inputs = {"staging_eval_long": f_long, "eval_table": {"path": g1rec["path"], "sha256": g1rec["sha256"]}}
    return blocks, base, {"spread": spread, "inputs": inputs, "n_test_hosts_per_resample": n_hosts}


def within_host_spread(ens: pd.DataFrame, long: pd.DataFrame) -> dict:
    """Mean over (r, test host with >= 2 sites) of the SD across the host's sites (g1_diagnostics definition)."""
    out = {}
    s0 = long[long.seed == 0].rename(columns={"y_pred": "y_pred_seed0"})
    j = ens.merge(s0[["backbone", "model", "r", "B", "site_id", "y_pred_seed0"]],
                  on=["backbone", "model", "r", "B", "site_id"], how="left", validate="one_to_one")
    for (bb, m, B), g in j.groupby(["backbone", "model", "B"]):
        g = g[g.groupby(["r", "host_id"]).y_pred.transform("size") >= 2]
        k = g.groupby(["r", "host_id"])
        ens_sd, seed0_sd, tgt = float(k.y_pred.std().mean()), float(k.y_pred_seed0.std().mean()), \
            float(k.y_true.std().mean())
        out.setdefault(bb, {}).setdefault(m, {})[str(B)] = {
            "ensemble_eV": ens_sd, "seed0_eV": seed0_sd, "target_eV": tgt,
            "ratio_ensemble_to_target": ens_sd / tgt, "ratio_seed0_to_target": seed0_sd / tgt}
    return out


def sweep_repro(blocks: dict, base: dict, spread: dict, root: Path) -> dict:
    rec = lambda n: c.record(n, root)["payload"]
    g1, st1, st2, c2 = rec("g1_primary"), rec("staging_v1"), rec("staging_v2"), rec("c1c2_v2")
    prec, diag = rec("precision_a_b400_b654"), rec("g1_diagnostics")
    out = {}
    out["v1_S_D_A_vs_g1_primary"] = compare(
        {B: {"MAE_S": v["MAE_S"], "MAE_D": v["MAE_D"], "A": v["A"]} for B, v in blocks["v1"].items()},
        g1["per_budget"], "results/g1_primary.json payload.per_budget (MAE_S, MAE_D, A: mean, ci, "
                          "upper_one_sided, boot_sd, sd_over_resamples, per_resample)")
    out["baselines_vs_g1_primary"] = compare(base, {m: g1["descriptive_curves"]["mae"][m] for m in BASELINES},
                                             "results/g1_primary.json payload.descriptive_curves.mae")
    for bb, srec in (("v1", st1), ("v2", st2)):
        pairs = []
        for B, v in blocks[bb].items():
            for m, key in (("S", "MAE_S"), ("D", "MAE_D"), ("P", "MAE_P")):
                r = srec["budgets"][B]["mae"][m]
                pairs += [(f"{B}.{m}.mean", v[key]["mean"], r["mean"]),
                          (f"{B}.{m}.ci95[0]", v[key]["ci"][0], r["ci95"][0]),
                          (f"{B}.{m}.ci95[1]", v[key]["ci"][1], r["ci95"][1])]
                pairs += [(f"{B}.{m}.per_resample[{i}]", x, y)
                          for i, (x, y) in enumerate(zip(v[key]["per_resample"], r["per_resample"], strict=True))]
        out[f"{bb}_S_D_P_vs_staging_{bb}"] = compare_pairs(pairs, f"results/staging_{bb}.json payload.budgets.*.mae")
    pairs = []
    for B, v in blocks["v2"].items():
        pairs += [(f"{B}.A", v["A"]["mean"], c2["A_mae"][B]["A"]),
                  (f"{B}.MAE_S", v["MAE_S"]["mean"], c2["mae"][B]["S"]["mae"]["mean"]),
                  (f"{B}.MAE_D", v["MAE_D"]["mean"], c2["mae"][B]["D-state"]["mae"]["mean"])]
        pairs += [(f"{B}.A.per_resample[{i}]", x, y) for i, (x, y) in
                  enumerate(zip(v["A"]["per_resample"], c2["A_mae"][B]["per_resample"], strict=True))]
    out["v2_points_vs_c1c2_v2"] = compare_pairs(
        pairs, "results/c1c2_v2.json payload.A_mae, mae (points only: its intervals came from the collaborator's "
               "bootstrap, a known estimator difference listed in results/freeze_table.json)")
    for bb in ("v2", "v1"):
        out[f"{bb}_A_400_654_vs_precision_10x3"] = compare(
            {B: {"A": blocks[bb][B]["A"], "MAE_S": blocks[bb][B]["MAE_S"], "MAE_D": blocks[bb][B]["MAE_D"]}
             for B in ("400", "654")},
            {B: {k: prec[bb]["10x3"][B][k] for k in ("A", "MAE_S", "MAE_D")} for B in ("400", "654")},
            f"results/precision_a_b400_b654.json payload.{bb}.10x3")
    pairs = []
    for key, v in diag["within_host_prediction_sd"]["values"].items():
        m, B = key.split("|")
        mine = spread["v1"][m][B]
        pairs += [(f"{key}.{k}", mine[k], v[k]) for k in ("ensemble_eV", "seed0_eV", "target_eV")]
    out["v1_within_host_spread_vs_g1_diagnostics"] = compare_pairs(
        pairs, "results/g1_diagnostics.json payload.within_host_prediction_sd.values")
    # the stand-alone baseline records: points identical, intervals from another bootstrap (cluster, seed 0)
    pairs = []
    for m, name in (("B0", "b0_physics_floor"), ("RF-Kumagai", "rf_kumagai")):
        r = rec(name)["outer"]
        pairs += [(f"{m}.{B}.mean", base[m][B]["mean"], r[B]["aggregate"]["mae"]["mean"]) for B in base[m]]
    out["baseline_points_vs_standalone_records"] = compare_pairs(
        pairs, "results/b0_physics_floor.json, results/rf_kumagai.json payload.outer.*.aggregate.mae.mean (points; "
               "their intervals use a host-cluster bootstrap with seed 0)")
    return out


# ---------------------------------------------------------------- precision and CV

def precision(root: Path) -> tuple[dict, dict]:
    from dftgnn.stats.g1 import HierDraws

    prec, src = c.load_table("precision_eval_long", root)
    draws = {n: HierDraws.make([precision_analysis.N_HOSTS_PER_RESAMPLE] * n, c.N_BOOT, seed=c.BOOT_SEED)
             for n in (10, 30)}
    rec = c.record("precision_a_b400_b654", root)["payload"]
    out, repro = {}, {}
    for bb in ("v2", "v1"):
        df = prec[prec.backbone == bb]
        out[bb] = {"30x5": precision_analysis.analyse(df, 30, 5, draws[30]),
                   "10x3": precision_analysis.analyse(df, 10, 3, draws[10])}
        for s in ("30x5", "10x3"):
            repro[f"{bb}_{s}"] = compare(out[bb][s], rec[bb][s],
                                         f"results/precision_a_b400_b654.json payload.{bb}.{s}")
    return out, {"reproduction": repro, "input": src}


def cv(root: Path) -> tuple[dict, dict]:
    from dftgnn.split import load_split

    df, src = c.load_table("cv_eval_long", root)
    split = load_split("cv_v1")
    hosts = sorted({h for f in split["by_k"]["10"]["folds"] for h in f["test"]})
    x_train = {k: float(np.mean([f["n_train"] for f in split["by_k"][str(k)]["folds"]])) for k in cv_analysis.KS}
    w = cv_analysis.host_draws(len(hosts))
    rec = c.record("cv_extended_budget", root)["payload"]
    out, repro = {}, {}
    for bb in ("v2", "v1"):
        out[bb] = cv_analysis.analyse_backbone(df[df.backbone == bb], hosts, w, x_train)
        repro[bb] = compare(out[bb], rec[bb], f"results/cv_extended_budget.json payload.{bb}")
    return out, {"reproduction": repro, "input": src}


# ---------------------------------------------------------------- LOCO by fold and by family

def universe(data_root: Path):
    """The universe table; ``data/processed`` is gitignored, so a worktree reads it from ``data_root``."""
    from dftgnn.config import load_config
    from dftgnn.data.universe import read_universe

    cfg = load_config()
    data = cfg.data.model_copy(update={"processed_dir": str(data_root / cfg.data.processed_dir)})
    return read_universe(cfg.model_copy(update={"data": data}))


def loco(root: Path, data_root: Path) -> tuple[dict, dict]:
    from dftgnn.split import load_split
    from dftgnn.stats.g1 import BOOT_SEED, HierDraws, point_metric, resample_stats, summarize

    ing = ingest_module()
    br = ing.Branch(root)
    uni = universe(data_root)
    rf, b0 = ing.loco_baselines(br)
    rec = c.record("ingest_m1_recompute", root)["payload"]
    split = load_split("loco_v1")
    fam = split["host_family"]
    formula = uni.drop_duplicates("host_id").set_index("host_id").reduced_formula
    out, repro = {}, {}
    for bb, sub in (("v2", "loco_v2"), ("v1", "loco")):
        res = ing.loco(br, uni, sub, rf if bb == "v1" else None, b0 if bb == "v1" else None)
        repro[bb] = compare(res, rec[f"loco_{bb}"], f"results/ingest_m1_recompute.json payload.loco_{bb}")
        recs = br.records(sub, ("S", "D-state"), "loco/")
        runs = br.predictions(recs, {"S": "S", "D-state": "D"})
        runs["r"] = runs.run_id.map({p["run_id"]: int(p["spec"]["split"].split("_f")[1]) for p in recs})
        ens = ing.ensemble(runs)
        wide = ens[ens.model == "S"].merge(ens[ens.model == "D"], on=["site_id", "r"], suffixes=("_S", "_D"),
                                           validate="one_to_one")
        assert (wide.host_id_S == wide.host_id_D).all() and np.allclose(wide.y_true_S, wide.y_true_D)
        wide = wide.rename(columns={"host_id_S": "host_id", "y_true_S": "y_true"})
        wide["family"] = wide.host_id.map(fam)
        assert wide.family.notna().all()
        wide["d"] = (wide.y_pred_S - wide.y_true).abs() - (wide.y_pred_D - wide.y_true).abs()
        n_all = len(wide)

        def bound(frame: pd.DataFrame) -> dict:
            st = {m: [resample_stats(frame.rename(columns={f"y_pred_{m}": "y_pred"})[
                ["site_id", "host_id", "y_true", "y_pred"]].sort_values("site_id"))] for m in ("S", "D")}
            dr = HierDraws.make([len(st["S"][0])], c.N_BOOT, BOOT_SEED)
            pt = point_metric(st["S"][0], "mae") - point_metric(st["D"][0], "mae")
            s = summarize([pt], dr.replicates(st["S"], "mae") - dr.replicates(st["D"], "mae"), ci=c.CI,
                          upper_quantile=c.UPPER_Q)
            return {**a_entry(s), "n_hosts": len(st["S"][0]), "n_sites": len(frame)}

        pooled = bound(wide)
        if abs(pooled["A"] - res["A"]["mean"]) > REPRO_TOL or abs(pooled["upper_one_sided_95"]
                                                                  - res["A"]["upper_one_sided"]) > REPRO_TOL:
            raise AssertionError(f"{bb}: family frame does not reproduce the pooled LOCO bound")
        folds = {}
        for f, g in wide.groupby("r"):
            folds[str(f)] = {**bound(g), "families": sorted(g.family.unique()),
                             "contribution_to_pooled_A": float(g.d.sum() / n_all),
                             "pooled_without_this_fold": bound(wide[wide.r != f])}
        fams = []
        for k, g in wide.groupby("family"):
            hosts = sorted(g.host_id.unique())
            fams.append({"family": k, "fold": int(g.r.iloc[0]), "n_hosts": len(hosts), "n_sites": len(g),
                         "A_family": float(g.d.mean()), "contribution_to_pooled_A": float(g.d.sum() / n_all),
                         "example_formulae": sorted({formula[h] for h in hosts})[:4]})
        fams.sort(key=lambda x: -x["contribution_to_pooled_A"])
        for row in fams[:10] + fams[-5:]:
            row["pooled_without_this_family"] = bound(wide[wide.family != row["family"]])
        greedy, dropped = [], []
        for row in fams:
            if greedy and greedy[-1]["upper_below_delta"]:
                break
            dropped.append(row["family"])
            greedy.append({"dropped": list(dropped), **bound(wide[~wide.family.isin(dropped)])})
        out[bb] = {"pooled": pooled, "per_fold": folds, "families_by_contribution": fams,
                   "drop_top_contributors_until_upper_below_delta": greedy,
                   "contribution_check_sum": float(sum(x["contribution_to_pooled_A"] for x in fams))}
        assert abs(out[bb]["contribution_check_sum"] - pooled["A"]) < REPRO_TOL
    meta = {"family_key_definition": split["family_key_definition"], "fold_rule": split["fold_rule"],
            "n_families": split["n_families"], "reproduction": repro,
            "estimator": "pooled seed-ensemble out-of-fold predictions (seeds 0-2); dftgnn.stats.g1.HierDraws "
                         "with one resample over the hosts of the set (2000, seed 20261008), as in "
                         "scripts/analysis/ingest_m1_recompute.loco; per-fold, per-family-removed and greedy "
                         "bounds are new draws of the same estimator on the reduced host set (descriptive)",
            "contribution": "sum over the family's sites of (|e_S| - |e_D|) divided by all 1726 sites; the "
                            "contributions add up to the pooled A"}
    return out, meta


# ---------------------------------------------------------------- tables

def grid(bb: str, sw: dict, base: dict, prec: dict, cvb: dict, lo: dict, spread: dict) -> dict:
    rows = {}
    for B in map(str, BUDGETS):
        v = sw[bb][B]
        src = [f"results/staging_{bb}.json payload.budgets.{B}.mae (S, D, P)",
               "results/g1_primary.json payload.descriptive_curves.mae (B0, RF)"]
        src.append("results/g1_primary.json payload.per_budget" if bb == "v1" else
                   "results/c1c2_v2.json payload.A_mae (point; interval recomputed with the G1 estimator)")
        if B in ("400", "654"):
            src.append(f"results/precision_a_b400_b654.json payload.{bb}.10x3.{B}")
        rows[f"B{B}"] = {
            "design": "outer resamples 0-9 x seeds 0-2 (sweep)", "x_training_hosts": float(B),
            "MAE_S": ms(v["MAE_S"]), "MAE_D": ms(v["MAE_D"]), "MAE_P": ms(v["MAE_P"]),
            "MAE_RF": ms(base["RF-Kumagai"][B]), "MAE_B0": ms(base["B0"][B]),
            "MAE_RF_electronic": ms(base["RF-electronic"][B]), "MAE_RF_structural": ms(base["RF-structural"][B]),
            "A": a_entry(v["A"]), "source_records": src}
    for B in ("400", "654"):
        p = prec[bb]["30x5"][B]
        rows[f"B{B}_30x5"] = {
            "design": "outer resamples 0-29 x seeds 0-4 (precision extension, pooled)", "x_training_hosts": float(B),
            "MAE_S": ms(p["MAE_S"]), "MAE_D": ms(p["MAE_D"]), "MAE_P": None, "MAE_RF": None, "MAE_B0": None,
            "A": a_entry(p["A"]), "source_records": [f"results/precision_a_b400_b654.json payload.{bb}.30x5.{B}"],
            "missing": "P, RF and B0 were not run on resamples 10-29 or with seeds 3-4"}
    for k in ("k10", "k20"):
        v = cvb[bb][k]
        rows[f"CV_{k}"] = {
            "design": f"grouped {k[1:]}-fold CV, pooled out-of-fold, seeds 0-2, all 818 hosts",
            "x_training_hosts": v["x_training_hosts"], "MAE_S": ms(v["MAE_S"]), "MAE_D": ms(v["MAE_D"]),
            "MAE_P": None, "MAE_RF": None, "MAE_B0": None, "A": a_entry(v["A"]),
            "A_fold_sd": v["fold_level"]["A_fold_sd"],
            "source_records": [f"results/cv_extended_budget.json payload.{bb}.{k}"],
            "missing": "no CV predictions exist for P, RF or B0 (they would need new fits)"}
    pl = lo[bb]["pooled"]
    rows["LOCO"] = {"design": "leave-one-family-group-out, 5 folds, pooled out-of-fold, seeds 0-2",
                    "x_training_hosts": 654.4, "A": pl,
                    "source_records": [f"results/ingest_m1_recompute.json payload.loco_{bb}",
                                       "results/derived_c1/loco_family.json"]}
    return {"arm": ARM[bb], "delta_eV": DELTA, "rows": rows,
            "within_host_prediction_spread": spread[bb]}


def main() -> None:
    from dftgnn.config import load_config
    from dftgnn.io.results import write_result

    cfg = load_config()
    assert cfg.stats.delta_eV == DELTA
    sw, base, extra = sweep(ROOT)
    repro_sweep = sweep_repro(sw, base, extra["spread"], ROOT)
    prec, prec_meta = precision(ROOT)
    cvb, cv_meta = cv(ROOT)
    data_root = Path(sys.argv[sys.argv.index("--data-root") + 1]) if "--data-root" in sys.argv else ROOT
    lo, lo_meta = loco(ROOT, data_root.expanduser().resolve())
    repro = {"sweep": repro_sweep, "precision": prec_meta["reproduction"], "cv": cv_meta["reproduction"],
             "loco": lo_meta["reproduction"]}
    flat = {f"{g}.{k}": v for g, d in repro.items() for k, v in d.items()}
    summary = {"max_abs_difference": max(v["max_abs_difference"] for v in flat.values()),
               "all_reproduced_below_1e-9": all(v["reproduced"] for v in flat.values()),
               "not_reproduced": [k for k, v in flat.items() if not v["reproduced"]],
               "n_comparisons": len(flat), "n_values_compared": sum(v["n_values_compared"] for v in flat.values())}
    print(json.dumps({"reproduction": summary, **{k: v["max_abs_difference"] for k, v in flat.items()}}, indent=1))
    if not summary["all_reproduced_below_1e-9"]:
        raise SystemExit(f"reproduction failed: {summary['not_reproduced']}")

    estimator = {
        "sweep": "dftgnn.stats.g1: seed-ensemble mean, paired hierarchical bootstrap (HierDraws), 2000 replicates, "
                 "seed 20261008, one draw set for S, D, P, B0, RF, every budget and both networks",
        "precision": "analysis/precision_analysis.analyse (G1 estimator; HierDraws over 30 or 10 resamples)",
        "cv": "analysis/cv_analysis.analyse_backbone (paired host-cluster bootstrap over 818 hosts, "
              "default_rng(20261008).multinomial, 2000)",
        "loco": lo_meta["estimator"],
        "two_sided_ci": c.CI, "one_sided_upper_quantile": c.UPPER_Q,
        "within_pm_delta": "two-sided 95% interval strictly inside (-delta, +delta)",
        "upper_below_delta": "one-sided 95% upper bound strictly below delta"}
    inputs = {**extra["inputs"], "precision_eval_long": prec_meta["input"], "cv_eval_long": cv_meta["input"]}
    claim_points = {k: grid("v2", sw, base, prec, cvb, lo, extra["spread"])["rows"][k]["A"]
                    for k in ("B654", "B654_30x5", "CV_k10", "CV_k20", "LOCO")}
    final = {"label": "DERIVED (C1); final network = v2; no new runs", "estimator": estimator,
             "table": grid("v2", sw, base, prec, cvb, lo, extra["spread"]),
             "claim_points_two_sided": claim_points,
             "reproduction": repro, "reproduction_summary": summary, "inputs": inputs}
    abl = {"label": "DERIVED (C1); default-initialisation ablation = v1; no new runs", "estimator": estimator,
           "table": grid("v1", sw, base, prec, cvb, lo, extra["spread"]),
           "site_resolution": site_resolution(extra["spread"], ROOT),
           "reproduction": repro, "reproduction_summary": summary, "inputs": inputs}
    fam = {"label": "DERIVED (C1); descriptive, no test", **lo_meta, "v2": lo["v2"], "v1": lo["v1"]}
    if "--dry-run" in sys.argv:
        print(json.dumps(c.to_jsonable(claim_points), indent=1))
        return
    for name, payload in (("final_network_summary", final), ("default_init_ablation", abl), ("loco_family", fam)):
        print(write_result(name, c.to_jsonable(payload), config=cfg, repo_root=ROOT, results_dir=OUT_DIR))


def site_resolution(spread: dict, root: Path) -> dict:
    """Within-host spread of S predictions relative to the data, both networks."""
    fx = c.record("s_site_resolution_forensic", root)["payload"]["checkpoints"]
    s10b = {B: {k: fx[f"S|{B}"]["flag_sweep_trained"][k] for k in ("median_sd_labelled", "median_target_sd")}
            for B in map(str, BUDGETS)}
    for v in s10b.values():
        v["ratio"] = v["median_sd_labelled"] / v["median_target_sd"]
    return {"definition": "mean over (resample, test host with >= 2 sites) of the SD of the seed-ensemble "
                          "prediction across the host's sites, divided by the same SD of the DFT targets "
                          "(results/g1_diagnostics.json definition; v2 computed here with the same code)",
            "S": {bb: {B: spread[bb]["S"][B] for B in map(str, BUDGETS)} for bb in ("v1", "v2")},
            "D": {bb: {B: spread[bb]["D"][B] for B in map(str, BUDGETS)} for bb in ("v1", "v2")},
            "s10b_forensic_v1_S": {"definition": "median over 30 resample-0 hosts of the SD of S (resample 0, seed "
                                                 "0) across labelled sites, trained checkpoint, over the median "
                                                 "target SD", "source": "results/s_site_resolution_forensic.json "
                                                 "payload.checkpoints.S|B.flag_sweep_trained", "values": s10b}}


if __name__ == "__main__":
    main()
