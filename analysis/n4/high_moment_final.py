"""The high-moment exclusion on the final network (v2), with the estimator unchanged.

Nothing is trained. The 60 runs of stage moment_v2 are compared with the final-network sweep at B = 654 (the
10 x 3 design of `results/derived_c1/final_network_summary.json`, row B654) by the steps of
``analysis/sensitivity_analysis.high_moment``, imported unchanged: seed-ensemble mean, ``common_sites``,
``stats_by_r``, ``hier_pair`` on the G1 hierarchical draws (2000 replicates, seed 20261008), ``changes``
(paired, same draws) and ``with_bound``. Physically: removing the vacancies whose relaxed defect state carries
a localised moment (Ce, La and W hosts) tests whether the bound on A rests on electronic configurations the
non-spin-polarised descriptors cannot represent.

The moment_v2 records are read with the checks of ``analysis/sensitivity_eval_tables`` (run ids equal the spec,
exclusion list equals the universe threshold list and specs/moment.jsonl, prediction sha256 and recorded MAE, every
test site minus the excluded entries exactly once, targets and hosts equal the universe, train+val counts without
the excluded entries). Reproduction (tolerance 1e-9 eV; any mismatch stops before a record is written):

    v1    ``high_moment`` rerun on its own inputs equals results/sens_high_moment_v1.json
    v2    the full-site v2 primary equals ``v2_primary_654`` of that record and row B654 of final_network_summary
    preds the v2 sweep predictions of sens_eval_long equal staging_eval_long (S, D, B = 654, seeds 0-2)
    sites per-resample exclusion counts of v2 equal those of the v1 record (same splits)
    RF    the forest numbers are quoted from results/sens_high_moment_rf.json (no recomputation)

Writes through ``write_result`` into ``results/derived_n4/high_moment_final_network.json``.

    PYTHONPATH=$PWD/src python analysis/n4/high_moment_final.py [--dry-run] [--data-root ~/Desktop/dft-gnn]
"""
from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis import sensitivity_analysis as sa
from analysis import sensitivity_eval_tables as setab
from analysis.a1_common import sha256, to_jsonable
from analysis.precision_cv_eval_table import check_run, check_targets
from dftgnn.stats.g1 import seed_ensemble

OUT_DIR = ROOT / "results" / "derived_n4"
STAGE = "moment_v2"
REPRO_TOL = 1e-9
COLS = ["model", "r", "B", "site_id", "host_id", "seed", "y_true", "y_pred"]
CAP_IDS = {"a7d70fc64d35f0c8", "4f4fcd4874ca44df", "8dd956bdfe4e07c4", "aee1a01ca0787cad"}
LABEL = ("EXTENSION on the final network (v2, POST HOC backbone); docs/deviations.md row of 2026-10-18 (§12); "
         "exclusion list fixed 2026-10-10 before any run")


# ---------------------------------------------------------------- reproduction

def leaves(x, path: tuple = ()):
    """(path, float) for every numeric leaf; booleans are compared as 0/1."""
    if isinstance(x, Mapping):
        for k, v in x.items():
            yield from leaves(v, path + (str(k),))
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield from leaves(v, path + (i,))
    elif isinstance(x, (bool, np.bool_, int, float, np.integer, np.floating)):
        yield path, float(x)


def compare(got, ref, keymap: Mapping[str, str] | None = None) -> dict:
    """Every leaf of ``ref`` (after renaming its keys by ``keymap``) must exist in ``got`` and agree to 1e-9."""
    g = dict(leaves(to_jsonable(got)))
    r = {tuple((keymap or {}).get(p, p) if isinstance(p, str) else p for p in k): v
         for k, v in leaves(to_jsonable(ref))}
    missing = sorted(map(str, set(r) - set(g)))
    diffs = [abs(g[k] - v) for k, v in r.items() if k in g]
    mx = max(diffs) if diffs else float("inf")
    return {"max_abs_difference": mx, "n_values_compared": len(diffs), "missing": missing[:5],
            "reproduced": not missing and bool(diffs) and mx <= REPRO_TOL}


# ---------------------------------------------------------------- moment_v2 records

def run_row(p: Mapping) -> dict:
    """``sensitivity_eval_tables.run_row``; the GPU model is kept as recorded."""
    row = setab.run_row({**p, "hardware": {}}, STAGE)
    row["gpu_class"] = p["hardware"]["gpu_model"]
    return row


def moment_v2_long(uni: pd.DataFrame, root: Path, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    excluded = setab.excluded_sites(uni, threshold)
    ex_set = set(excluded)
    v1_lists = {tuple(sorted(json.loads(x)["spec"]["exclude_sites"]))
                for x in (root / "specs" / "moment.jsonl").read_text().splitlines() if x.strip()}
    if v1_lists != {tuple(excluded)}:
        raise AssertionError("specs/moment.jsonl exclusion list differs from the universe threshold list")
    sites_of = uni.groupby("host_id").site_id.apply(set).to_dict()
    recs = setab.stage_records(root / "results" / STAGE, STAGE)
    setab.check_manifest_ids(recs, setab.spec_run_ids(root, STAGE), STAGE)
    setab.check_moment_specs(recs, excluded)
    bad = [p["run_id"] for p in recs if p.get("code_ref") != "nrp-code-v1" or p["spec"].get("code_ref") != "nrp-code-v1"]
    if bad:
        raise AssertionError(f"code_ref other than nrp-code-v1: {bad[:3]}")
    runs = pd.DataFrame([run_row(p) for p in recs])
    if runs.duplicated(subset=["model", "r", "seed"]).any() or set(runs.B) != {654} \
            or sorted(runs.model.value_counts().items()) != [("D", 30), ("S", 30)]:
        raise AssertionError("moment_v2 is not S and D x resamples 0-9 x seeds 0-2 at B = 654")
    frames, ex_hits = [], 0
    for row in runs.to_dict("records"):
        pred = setab.load_preds(row, root)
        test_hosts, train = setab.split_hosts(row["split"], row["B"])
        sites = set().union(*(sites_of[h] for h in test_hosts)) - ex_set
        test_hosts = {h for h in test_hosts if sites_of[h] - ex_set}
        ex_hits += len(set(pred.site_id) & ex_set)
        setab.check_moment_counts(row, train, sites_of, ex_set)
        check_run(pred, sites, test_hosts, train, f"{STAGE} {row['run_id']}")
        check_targets(pred, uni)
        frames.append(pred.assign(stage=STAGE, model=row["model"], r=row["r"], B=row["B"], seed=row["seed"],
                                  split=row["split"], run_id=row["run_id"]))
    if ex_hits:
        raise AssertionError(f"{ex_hits} excluded entries appear in moment_v2 predictions")
    long = pd.concat(frames, ignore_index=True)
    capped = runs[runs.epochs_run >= runs.max_epochs]
    if set(capped.run_id) != CAP_IDS or set(capped.model) != {"S"}:
        raise AssertionError(f"runs at the epoch cap differ from the ingest: {sorted(capped.run_id)}")
    info = {"n_runs": len(runs), "n_prediction_rows": len(long), "excluded_in_predictions": ex_hits,
            "gpu_models": runs.gpu_class.value_counts().sort_index().to_dict(),
            "at_epoch_cap": capped.sort_values("run_id")[["run_id", "model", "r", "seed", "epochs_run",
                                                          "max_epochs"]].to_dict("records"),
            "run_ids": sorted(runs.run_id)}
    return long, runs, info


# ---------------------------------------------------------------- analysis (the steps of sa.high_moment)

def high_moment_v2(long: pd.DataFrame, mom: pd.DataFrame, meta: Mapping) -> tuple[dict, pd.DataFrame]:
    excluded = set(meta["excluded"])
    hosts = {r: sa.test_hosts(r) for r in range(sa.N_R)}
    draws = sa.g1_draws([len(hosts[r]) for r in range(sa.N_R)])
    v2 = long[(long.stage == "v2_sweep") & (long.B == 654)]
    prim = seed_ensemble(v2[COLS])
    mom_ens = seed_ensemble(mom[COLS])
    if (prim.n_seeds != 3).any() or (mom_ens.n_seeds != 3).any():
        raise AssertionError("a site lacks one of the three seeds")
    common = sa.common_sites(prim[prim.model == "S"], mom, excluded)

    full, _ = sa.hier_pair(sa.stats_by_r(prim, "S", hosts), sa.stats_by_r(prim, "D", hosts), draws)
    base, base_raw = sa.hier_pair(sa.stats_by_r(prim, "S", hosts, common),
                                  sa.stats_by_r(prim, "D", hosts, common), draws)
    alt, alt_raw = sa.hier_pair(sa.stats_by_r(mom_ens, "S", hosts, common),
                                sa.stats_by_r(mom_ens, "D", hosts, common), draws)

    from dftgnn.split import budget_train, load_split

    site_host = dict(zip(prim.site_id, prim.host_id, strict=False))
    uni_hosts = {e["site_id"]: e["host_id"] for e in meta["excluded_entries"]}
    ex_counts = []
    for r in range(sa.N_R):
        sp = load_split(f"outer_r{r}")
        te, tr = set(sp["test"]), set(budget_train(sp, 654))
        ex_counts.append({"r": r, "in_test": sum(uni_hosts[s] in te for s in excluded),
                          "in_training_pool": sum(uni_hosts[s] in tr for s in excluded),
                          "n_common_sites": len(common[r]),
                          "n_primary_test_sites": int(((prim.model == "S") & (prim.r == r)).sum()),
                          "n_test_hosts_with_common_site": len({site_host[s] for s in common[r]})})
    out = {"primary_common_sites": sa.with_bound(base), "high_moment": sa.with_bound(alt),
           "change_high_moment_minus_primary": sa.changes(alt_raw, base_raw),
           "primary_full_sites": sa.with_bound(full), "excluded_per_resample": ex_counts}
    return out, v2


def preds_vs_staging(v2: pd.DataFrame, root: Path) -> dict:
    st = pd.read_parquet(root / "results" / "analysis" / "staging_eval_long.parquet")
    st = st[(st.backbone == "v2") & st.model.isin(["S", "D"]) & (st.B == 654) & (st.seed < 3)]
    key = ["model", "r", "seed", "site_id"]
    m = v2[key + ["y_pred"]].merge(st[key + ["y_pred"]], on=key, how="outer", suffixes=("", "_st"),
                                   indicator=True)
    both = m["_merge"] == "both"
    mx = float(np.abs(m.y_pred[both] - m.y_pred_st[both]).max())
    return {"max_abs_difference": mx, "n_values_compared": int(both.sum()),
            "missing": [] if both.all() else [f"{int((~both).sum())} unmatched rows"],
            "reproduced": bool(both.all()) and mx <= REPRO_TOL}


def forest_quote(root: Path) -> dict:
    """Network-free check, quoted from its record (no recomputation)."""
    path = root / "results" / "sens_high_moment_rf.json"
    p = json.loads(path.read_text())["payload"]
    pick = lambda b: {k: p[b]["mae"][k] for k in ("mean", "ci", "ci_level", "n_boot", "n_resamples") if k in p[b]["mae"]}
    return {"record": "results/sens_high_moment_rf.json", "record_sha256": sha256(path),
            "definition": p["definition"], "pool_size": p["pool_size"],
            "site_mae": {b: pick(b) for b in ("baseline_full", "baseline_on_reduced_test_sites", "excluded_run")},
            "change_site_mae": {b: pick(b) for b in ("delta_total_excluded_minus_baseline",
                                                     "delta_training_only_excluded_minus_baseline_reduced",
                                                     "delta_evaluation_only_baseline_reduced_minus_baseline")},
            "excluded_entries_in_test_by_resample": p["excluded_entries_in_test_by_resample"]}


def main() -> None:
    from dftgnn.config import load_config
    from dftgnn.io.results import write_result

    cfg = load_config()
    if float(cfg.stats.delta_eV) != sa.DELTA_EV:
        raise AssertionError("stats.delta_eV differs from 0.05")
    threshold = float(cfg.sensitivity.exclude_high_moment.threshold_muB)
    data_root = Path(sys.argv[sys.argv.index("--data-root") + 1]).expanduser().resolve() \
        if "--data-root" in sys.argv else ROOT
    uni_path = data_root / "data" / "processed" / "universe_v1.parquet"
    uni = pd.read_parquet(uni_path, columns=["site_id", "host_id", "target_Ef_eV", "defect_moment_muB"])

    long, _runs, et, meta = sa.load_inputs(ROOT)
    mom, _mruns, minfo = moment_v2_long(uni, ROOT, threshold)
    v2, v2_long = high_moment_v2(long, mom, meta)

    v1_rec = json.loads((ROOT / "results" / "sens_high_moment_v1.json").read_text())["payload"]
    fns = json.loads((ROOT / "results" / "derived_c1" / "final_network_summary.json").read_text())["payload"]
    b654 = fns["table"]["rows"]["B654"]
    v1_now = sa.high_moment(long, et, meta, ROOT)
    v2p_now = sa.v2_primary(long, ROOT)
    blocks = ("primary_common_sites", "high_moment", "change_high_moment_minus_primary", "primary_full_sites",
              "excluded_per_resample")
    repro = {f"v1_rerun.{b}": compare(v1_now[b], v1_rec[b]) for b in blocks}
    repro["v1_rerun.v2_primary_654"] = compare(v2p_now["primary"], v1_rec["v2_primary_654"]["primary"])
    repro["v2_full_sites~sens_high_moment_v1.v2_primary_654"] = compare(
        v2["primary_full_sites"], v1_rec["v2_primary_654"]["primary"])
    repro["v2_full_sites.A~final_network_summary.B654.A"] = compare(
        v2["primary_full_sites"]["A"], {k: b654["A"][k] for k in ("A", "ci95", "upper_one_sided_95")},
        {"A": "mean", "ci95": "ci", "upper_one_sided_95": "upper_one_sided"})
    for m in ("MAE_S", "MAE_D"):
        repro[f"v2_full_sites.{m}~final_network_summary.B654.{m}"] = compare(
            v2["primary_full_sites"][m], b654[m], {"ci95": "ci"})
    repro["v2_sweep_predictions~staging_eval_long"] = preds_vs_staging(v2_long, ROOT)
    keep = ("r", "in_test", "in_training_pool", "n_common_sites", "n_primary_test_sites",
            "n_test_hosts_with_common_site")
    repro["v2_excluded_per_resample~v1_record"] = compare(
        v2["excluded_per_resample"], [{k: e[k] for k in keep} for e in v1_rec["excluded_per_resample"]])
    summary = {"max_abs_difference": max(v["max_abs_difference"] for v in repro.values()),
               "all_reproduced_below_1e-9": all(v["reproduced"] for v in repro.values()),
               "not_reproduced": [k for k, v in repro.items() if not v["reproduced"]],
               "n_comparisons": len(repro), "n_values_compared": sum(v["n_values_compared"] for v in repro.values())}
    print(json.dumps(to_jsonable({k: v["max_abs_difference"] for k, v in repro.items()} | {"summary": summary}),
                     indent=1))
    if not summary["all_reproduced_below_1e-9"]:
        raise SystemExit(f"reproduction failed: {summary['not_reproduced']}")

    u_v1 = v1_rec["high_moment"]["A"]["upper_one_sided"]
    statement = {
        "statement": "U < 0.05 eV at 654 training hosts with the high-moment entries excluded",
        "default_init_high_moment_upper": u_v1,
        "final_high_moment_upper": v2["high_moment"]["A"]["upper_one_sided"],
        "final_primary_common_sites_upper": v2["primary_common_sites"]["A"]["upper_one_sided"],
        "final_primary_full_sites_upper": v2["primary_full_sites"]["A"]["upper_one_sided"],
        "holds_default_init": sa.passes(u_v1, sa.DELTA_EV),
        "holds_final": v2["high_moment"]["A"]["below_delta"],
        "unchanged": bool(sa.passes(u_v1, sa.DELTA_EV) == v2["high_moment"]["A"]["below_delta"]
                          and v2["primary_common_sites"]["A"]["below_delta"]
                          == v2["high_moment"]["A"]["below_delta"]),
        "rule": "upper_one_sided (G1 one-sided 95% quantile) strictly below delta_eV, analysis/a1_common.passes"}
    payload = {
        "label": LABEL, "network": "final network: He initialisation, validation-tuned (configs/tuned_v2.yaml); "
                                   "called v2 in the repository",
        "comparison": "final-network sweep (10 x 3 at 654) vs moment_v2 runs on the common test sites of each "
                      "resample; full-site primary for reference",
        "estimator": "analysis/sensitivity_analysis.high_moment steps imported unchanged (seed_ensemble, common_sites, "
                     "stats_by_r, hier_pair, changes, with_bound) on the G1 hierarchical draws",
        "budget": 654, "resamples": list(range(sa.N_R)), "seeds": [0, 1, 2], "threshold_muB": threshold,
        "excluded_entries": meta["excluded_entries"], "delta_eV": sa.DELTA_EV,
        "final_network": v2,
        "default_init_reference": {"record": "results/sens_high_moment_v1.json",
                                   **{b: v1_rec[b] for b in ("primary_common_sites", "high_moment",
                                                             "change_high_moment_minus_primary")}},
        "u_below_delta_at_654": statement,
        "forest_check": forest_quote(ROOT),
        "runs": minfo,
        "bootstrap": {"method": "G1 hierarchical bootstrap, G1 primary draws; hosts without a common site enter "
                                "as zero rows", "draws": sa.N_BOOT, "seed": sa.BOOT_SEED},
        "reproduction": repro, "reproduction_summary": summary,
        "inputs": {**meta["sensitivity_eval_tables"], "eval_table": meta["eval_table"],
                   "universe": {"path": "data/processed/universe_v1.parquet", "sha256": sha256(uni_path)},
                   "staging_eval_long": {"path": "results/analysis/staging_eval_long.parquet",
                                         "sha256": sha256(ROOT / "results" / "analysis" / "staging_eval_long.parquet")},
                   "moment_v2_records": "results/moment_v2/*.json (60), predictions sha256-checked",
                   "final_network_summary": {"path": "results/derived_c1/final_network_summary.json",
                                             "sha256": sha256(ROOT / "results" / "derived_c1" /
                                                              "final_network_summary.json")}},
    }
    if "--dry-run" in sys.argv:
        print(json.dumps(to_jsonable({k: v2[k]["A"] for k in ("primary_common_sites", "high_moment",
                                                              "change_high_moment_minus_primary")}), indent=1))
        return
    print(write_result("high_moment_final_network", to_jsonable(payload), config=cfg, repo_root=ROOT,
                       results_dir=OUT_DIR))


if __name__ == "__main__":
    main()
