"""High-moment, CGCNN and capped-twin sensitivities of the advantage A = MAE_S - MAE_D.

Rules:, rows of 2026-10-10 (§5, §12) and 2026-10-14 (§5, §12). Labels: high moment EXTENSION; CGCNN EXTENSION
and untuned (descriptive, one resample. Reads the tables of `analysis/sensitivity_eval_tables.py` (sha256
checked against `results/sensitivity_eval_tables.json`) and the G1 eval table (checked against the input of
`results/g1_primary.json`). Writes, via write_result,

    results/sens_high_moment_v1.json   primary (common sites) vs high moment: MAE_S, MAE_D, A and paired changes
    results/sens_cgcnn.json            CGCNN vs MEGNet (v1, v2) at B = 50, 200, 654 on resample 0

Intervals: the G1 hierarchical bootstrap with the G1 primary draws (2000 replicates, seed 20261008) for the
high-moment and capped comparisons; a host-cluster bootstrap over the 164 test hosts of resample 0 (2000 draws,
seed 20261008) for CGCNN. Sanity: the primary side reproduces `results/g1_primary.json` at B = 400 and 654 and
`results/c1c2_v2.json` at B = 654 to 1e-6; differences are recorded, not hidden.

    PYTHONPATH=$PWD/src python analysis/sensitivity_analysis.py
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.a1_common import passes, sha256, to_jsonable
from dftgnn.stats.g1 import HierDraws, point_metric, seed_ensemble, summarize
from dftgnn.stats.metrics import _metrics_from_totals, host_stats

ROOT = Path(__file__).resolve().parents[1]
N_BOOT = 2000
BOOT_SEED = 20261008
DELTA_EV = 0.05
TOL = 1e-6
N_R = 10
CAPPED_BUDGETS = (400, 654)
CGCNN_BUDGETS = (50, 200, 654)
LABELS = {"moment": "EXTENSION (originally registered backbone v1; exclusion list fixed 2026-10-10 before any run)",
          "cgcnn": "EXTENSION, untuned CGCNN; descriptive, one resample, no hypothesis test",
          "capped": "POST HOC (originally registered backbone v1; rule fixed 2026-10-10 before any twin result)"}


# ---------------------------------------------------------------- generic pieces

def padded_stats(frame: pd.DataFrame, hosts: Sequence[str]) -> np.ndarray:
    """Per-host statistics in the order of ``hosts``; a host without a site in ``frame`` is a zero row."""
    st = host_stats(frame.y_true.to_numpy(), frame.y_pred.to_numpy(), frame.host_id.to_numpy())
    extra = set(st.index) - set(hosts)
    if extra:
        raise AssertionError(f"hosts outside the test list: {sorted(extra)[:3]}")
    return st.reindex(list(hosts), fill_value=0.0).to_numpy()


def hier_pair(stats_s: Sequence[np.ndarray], stats_d: Sequence[np.ndarray], draws: HierDraws,
              metric: str = "mae") -> tuple[dict, dict]:
    """MAE_S, MAE_D and A = S - D on shared draws; also returns the per-resample points and replicates."""
    rs, rd = draws.replicates(stats_s, metric), draws.replicates(stats_d, metric)
    ps = np.array([point_metric(s, metric) for s in stats_s])
    pd_ = np.array([point_metric(s, metric) for s in stats_d])
    out = {"MAE_S": summarize(ps, rs), "MAE_D": summarize(pd_, rd), "A": summarize(ps - pd_, rs - rd)}
    raw = {"MAE_S": (ps, rs), "MAE_D": (pd_, rd), "A": (ps - pd_, rs - rd)}
    return out, raw


def changes(alt: Mapping, base: Mapping) -> dict:
    """Paired differences alt - base of MAE_S, MAE_D and A (same draws on both sides)."""
    return {k: summarize(alt[k][0] - base[k][0], alt[k][1] - base[k][1]) for k in ("MAE_S", "MAE_D", "A")}


def with_bound(block: dict) -> dict:
    block["A"]["below_delta"] = passes(block["A"]["upper_one_sided"], DELTA_EV)
    return block


def g1_draws(n_hosts: Sequence[int]) -> HierDraws:
    return HierDraws.make(list(n_hosts), N_BOOT, BOOT_SEED)


def ensemble_from_seeds(wide: pd.DataFrame) -> pd.DataFrame:
    """Seed-ensemble ``y_pred`` from the three seed columns of the G1 eval table."""
    cols = [f"y_pred_seed{s}" for s in range(3)]
    return wide.assign(y_pred=wide[cols].mean(axis=1))


def max_abs_diff(got: Mapping, ref: Mapping, keys: Sequence[tuple[str, str]]) -> dict:
    """Largest absolute difference per (got key, ref key); lists are compared elementwise."""
    out = {}
    for gk, rk in keys:
        a, b = np.atleast_1d(np.asarray(got[gk], float)), np.atleast_1d(np.asarray(ref[rk], float))
        out[f"{gk}~{rk}"] = float(np.max(np.abs(a - b))) if a.shape == b.shape else float("inf")
    return out


# ---------------------------------------------------------------- high moment

def common_sites(primary: pd.DataFrame, sens: pd.DataFrame, excluded: set[str]) -> dict[int, set[str]]:
    """Per resample: sites in every sensitivity run and in the primary test set (= primary minus excluded)."""
    out = {}
    for r, g in sens.groupby("r"):
        per_run = [set(x.site_id) for _, x in g.groupby("run_id")]
        if any(s != per_run[0] for s in per_run):
            raise AssertionError(f"r {r}: sensitivity runs cover different sites")
        prim = set(primary.site_id[primary.r == r])
        common = per_run[0] & prim
        if common != prim - excluded:
            raise AssertionError(f"r {r}: common sites differ from the primary test sites minus the excluded")
        out[int(r)] = common
    return out


def stats_by_r(ens: pd.DataFrame, model: str, hosts: Mapping[int, Sequence[str]],
               keep: Mapping[int, set[str]] | None = None) -> list[np.ndarray]:
    out = []
    for r in sorted(hosts):
        f = ens[(ens.model == model) & (ens.r == r)]
        if keep is not None:
            f = f[f.site_id.isin(keep[r])]
        out.append(padded_stats(f, hosts[r]))
    return out


# ---------------------------------------------------------------- capped twins

def replace_twins(wide: pd.DataFrame, twins: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Replace the seed column of each twin's (model, r, B, seed) by the twin's predictions; re-ensemble."""
    out = wide.copy()
    done = []
    for (model, r, B, seed), g in twins.groupby(["model", "r", "B", "seed"]):
        m = (out.model == model) & (out.r == r) & (out.B == B)
        if not m.any():
            raise AssertionError(f"no primary cell for twin {model} r{r} B{B}")
        tw = g.set_index("site_id").y_pred
        idx = out.index[m]
        sites = out.loc[idx, "site_id"]
        if set(sites) != set(tw.index) or len(tw) != len(sites):
            raise AssertionError(f"twin {model} r{r} B{B} seed{seed}: sites differ from the primary cell")
        out.loc[idx, f"y_pred_seed{seed}"] = tw.loc[sites].to_numpy()
        done.append({"model": model, "r": int(r), "B": int(B), "seed": int(seed), "run_id": g.run_id.iloc[0]})
    return ensemble_from_seeds(out), done


# ---------------------------------------------------------------- CGCNN

def cluster_weights(n_hosts: int, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_hosts, np.full(n_hosts, 1.0 / n_hosts), size=n_boot).astype(float)


def cluster_summary(stats: np.ndarray, w: np.ndarray, metric: str) -> tuple[dict, np.ndarray]:
    reps = _metrics_from_totals(w @ stats)[metric]
    pt = float(_metrics_from_totals(stats.sum(axis=0))[metric])
    return {"point": pt, "ci95": [float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))],
            "upper_one_sided_95": float(np.quantile(reps, 0.95))}, reps


def diff_summary(pa: float, pb: float, ra: np.ndarray, rb: np.ndarray) -> dict:
    d = ra - rb
    return {"point": float(pa - pb), "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
            "upper_one_sided_95": float(np.quantile(d, 0.95))}


# ---------------------------------------------------------------- inputs

def load_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    rec = json.loads((root / "results" / "sensitivity_eval_tables.json").read_text())
    pay = rec["payload"]
    for name in ("sens_eval_long", "sens_runs"):
        ref = pay["files"][name]
        if sha256(root / ref["path"]) != ref["sha256"]:
            raise AssertionError(f"{name}: sha256 differs from results/sensitivity_eval_tables.json")
    g1_in = json.loads((root / "results" / "g1_primary.json").read_text())["payload"]["input"]["eval_table"]
    if sha256(root / g1_in["path"]) != g1_in["sha256"]:
        raise AssertionError("eval_table.parquet differs from the G1 primary input")
    long = pd.read_parquet(root / pay["files"]["sens_eval_long"]["path"])
    runs = pd.read_parquet(root / pay["files"]["sens_runs"]["path"])
    et = pd.read_parquet(root / g1_in["path"])
    et = et[et.model.isin(["S", "D"])].reset_index(drop=True)
    if not np.allclose(ensemble_from_seeds(et).y_pred, et.y_pred, rtol=0, atol=1e-12):
        raise AssertionError("eval table ensemble differs from the mean of its seed columns")
    meta = {"sensitivity_eval_tables": {"git_sha": rec["git_sha"], "files": pay["files"]},
            "eval_table": {"path": g1_in["path"], "sha256": g1_in["sha256"]},
            "excluded": [e["site_id"] for e in pay["integrity"]["excluded_entries"]],
            "excluded_entries": pay["integrity"]["excluded_entries"]}
    return long, runs, et, meta


def test_hosts(r: int) -> list[str]:
    from dftgnn.split import load_split

    return sorted(load_split(f"outer_r{r}")["test"])


def g1_reference(root: Path, B: int) -> dict:
    p = json.loads((root / "results" / "g1_primary.json").read_text())["payload"]["per_budget"][str(B)]
    return p


# ---------------------------------------------------------------- analyses

def high_moment(long: pd.DataFrame, et: pd.DataFrame, meta: Mapping, root: Path) -> dict:
    excluded = set(meta["excluded"])
    hosts = {r: test_hosts(r) for r in range(N_R)}
    draws = g1_draws([len(hosts[r]) for r in range(N_R)])
    prim = et[et.B == 654]
    mom = long[long.stage == "moment"].assign(model=lambda d: d.model)
    mom_ens = seed_ensemble(mom[["model", "r", "B", "site_id", "host_id", "seed", "y_true", "y_pred"]])
    if (mom_ens.n_seeds != 3).any():
        raise AssertionError("a moment site lacks one of the three seeds")
    common = common_sites(prim[prim.model == "S"], mom, excluded)

    full, _ = hier_pair(stats_by_r(prim, "S", hosts), stats_by_r(prim, "D", hosts), draws)
    base, base_raw = hier_pair(stats_by_r(prim, "S", hosts, common), stats_by_r(prim, "D", hosts, common), draws)
    alt, alt_raw = hier_pair(stats_by_r(mom_ens, "S", hosts, common), stats_by_r(mom_ens, "D", hosts, common), draws)

    ref = g1_reference(root, 654)
    sanity = {}
    for k in ("MAE_S", "MAE_D", "A"):
        sanity.update({f"{k}.{x}": v for x, v in max_abs_diff(
            full[k], ref[k], [("mean", "mean"), ("per_resample", "per_resample"), ("ci", "ci"),
                              ("upper_one_sided", "upper_one_sided")]).items()})
    sanity_ok = all(v <= TOL for v in sanity.values())

    ex_counts = []
    from dftgnn.split import budget_train, load_split

    site_host = dict(zip(et.site_id, et.host_id, strict=False))
    uni_hosts = {e["site_id"]: e["host_id"] for e in meta["excluded_entries"]}
    for r in range(N_R):
        sp = load_split(f"outer_r{r}")
        te, tr = set(sp["test"]), set(budget_train(sp, 654))
        ex_counts.append({"r": r, "in_test": sum(uni_hosts[s] in te for s in excluded),
                          "in_training_pool": sum(uni_hosts[s] in tr for s in excluded),
                          "n_common_sites": len(common[r]),
                          "n_primary_test_sites": int(((prim.model == "S") & (prim.r == r)).sum()),
                          "n_test_hosts_with_common_site": len({site_host[s] for s in common[r]})})
    return {
        "label": LABELS["moment"], "backbone": "v1 (originally registered; specs/moment.manifest.json)",
        "v2": "no high-moment runs exist for the v2 backbone; no v2 record is written",
        "budget": 654, "resamples": list(range(N_R)), "seeds": [0, 1, 2],
        "excluded_entries": meta["excluded_entries"], "threshold_muB": 0.5,
        "comparison": "primary v1 sweep vs high-moment runs on the common test sites of each resample",
        "primary_common_sites": with_bound(base), "high_moment": with_bound(alt),
        "change_high_moment_minus_primary": changes(alt_raw, base_raw),
        "primary_full_sites": with_bound(full),
        "excluded_per_resample": ex_counts,
        "bootstrap": {"method": "G1 hierarchical bootstrap, G1 primary draws; hosts without a common site enter "
                                "as zero rows", "draws": N_BOOT, "seed": BOOT_SEED,
                      "test_hosts_per_resample": [len(hosts[r]) for r in range(N_R)]},
        "delta_eV": DELTA_EV,
        "sanity_g1_primary_654": {"max_abs_diff": sanity, "tolerance": TOL, "passed": sanity_ok},
        "input": meta["sensitivity_eval_tables"] | {"eval_table": meta["eval_table"]},
    }


def capped(long: pd.DataFrame, runs: pd.DataFrame, et: pd.DataFrame, meta: Mapping, root: Path) -> dict:
    hosts = {r: test_hosts(r) for r in range(N_R)}
    draws = g1_draws([len(hosts[r]) for r in range(N_R)])
    tw = long[(long.stage == "capped_twin") & long.split.str.startswith("outer_r") & long.B.isin(CAPPED_BUDGETS)]
    wide = et[et.B.isin(CAPPED_BUDGETS)]
    twin_wide, replaced = replace_twins(wide, tw)
    per_budget, sanity = {}, {}
    for B in CAPPED_BUDGETS:
        p, p_raw = hier_pair(stats_by_r(wide[wide.B == B], "S", hosts), stats_by_r(wide[wide.B == B], "D", hosts),
                             draws)
        t, t_raw = hier_pair(stats_by_r(twin_wide[twin_wide.B == B], "S", hosts),
                             stats_by_r(twin_wide[twin_wide.B == B], "D", hosts), draws)
        n_rep = {m: sum(1 for x in replaced if x["B"] == B and x["model"] == m) for m in ("S", "D")}
        per_budget[str(B)] = {"primary": with_bound(p), "capped_twins": with_bound(t),
                              "change_twin_minus_primary": changes(t_raw, p_raw), "n_replaced_runs": n_rep}
        ref = g1_reference(root, B)
        for k in ("MAE_S", "MAE_D", "A"):
            sanity.update({f"{B}.{k}.{x}": v for x, v in max_abs_diff(
                p[k], ref[k], [("mean", "mean"), ("per_resample", "per_resample"), ("ci", "ci"),
                               ("upper_one_sided", "upper_one_sided")]).items()})
    return {
        "label": LABELS["capped"], "backbone": "v1 (originally registered; specs/capped.manifest.json)",
        "v2": "no capped reruns exist for the v2 backbone; v2 is shown with its primary result only",
        "rule": "docs/deviations.md 2026-10-10 (§5): A at B = 400 and 654 with every capped original replaced by "
                "its 600-epoch twin, G1 procedure and draws, beside the primary; primary result and N* unchanged",
        "per_budget": per_budget, "replaced": replaced,
        "descriptive_runs": capped_descriptive(runs),
        "bootstrap": {"method": "G1 hierarchical bootstrap, G1 primary draws", "draws": N_BOOT, "seed": BOOT_SEED},
        "delta_eV": DELTA_EV,
        "sanity_g1_primary": {"max_abs_diff": sanity, "tolerance": TOL,
                              "passed": all(v <= TOL for v in sanity.values())},
        "input": meta["sensitivity_eval_tables"] | {"eval_table": meta["eval_table"]},
    }


def capped_descriptive(runs: pd.DataFrame) -> dict:
    """Original versus twin, paired by run; twins' epochs against the 600 cap (no test)."""
    tw = runs[runs.stage == "capped_twin"].set_index("original_run_id")
    og = runs[runs.stage == "capped_original"].set_index("original_run_id")
    pair = tw.join(og, lsuffix="_twin", rsuffix="_orig")
    pair["kind"] = np.where(pair.split_twin == "kiyohara", "kiyohara", "sweep")
    pair["change"] = pair.metric_twin - pair.metric_orig
    groups = []
    for (kind, model, B), g in pair.groupby(["kind", "model_twin", "B_twin"]):
        groups.append({"kind": kind, "model": model, "B": int(B), "n_capped": len(g),
                       "metric": g.metric_name_twin.iloc[0],
                       "mean_original": float(g.metric_orig.mean()), "mean_twin": float(g.metric_twin.mean()),
                       "mean_change_twin_minus_original": float(g.change.mean()),
                       "min_change": float(g.change.min()), "max_change": float(g.change.max()),
                       "n_twin_lower": int((g.change < 0).sum())})
    per_run = [{"original_run_id": o, "twin_run_id": x.run_id_twin, "twin_source": x.twin_source_twin,
                "kind": x.kind, "model": x.model_twin, "split": x.split_twin, "r": int(x.r_twin), "B": int(x.B_twin),
                "seed": int(x.seed_twin), "metric": x.metric_name_twin, "original": float(x.metric_orig),
                "twin": float(x.metric_twin), "change": float(x.change),
                "original_epochs_run": int(x.epochs_run_orig), "original_best_epoch": int(x.best_epoch_orig),
                "twin_epochs_run": int(x.epochs_run_twin), "twin_best_epoch": int(x.best_epoch_twin),
                "twin_max_epochs": int(x.max_epochs_twin), "twin_gpu_class": x.gpu_class_twin,
                "original_gpu_class": x.gpu_class_orig}
               for o, x in pair.sort_values(["kind", "model_twin", "B_twin", "r_twin", "seed_twin"]).iterrows()]
    ep = pair.epochs_run_twin
    return {"groups": groups, "per_run": per_run,
            "twin_epochs": {"n_twins": len(pair), "cap": 600, "n_reaching_cap": int((ep >= 600).sum()),
                            "max_epochs_run": int(ep.max()), "median_epochs_run": float(ep.median()),
                            "max_best_epoch": int(pair.best_epoch_twin.max()),
                            "n_best_epoch_above_200": int((pair.best_epoch_twin > 200).sum())},
            "twin_gpu_classes": pair.gpu_class_twin.value_counts().to_dict(),
            "original_gpu_classes": pair.gpu_class_orig.value_counts().to_dict(),
            "hardware_note": "the 33 new twins ran on the NRP pool and the originals and the 3 capsens twins on the "
                             "sweep pod; by N1 a twin-minus-original change mixes the cap with GPU class"}


def v2_primary(long: pd.DataFrame, root: Path) -> dict:
    hosts = {r: test_hosts(r) for r in range(N_R)}
    draws = g1_draws([len(hosts[r]) for r in range(N_R)])
    v2 = long[(long.stage == "v2_sweep") & (long.B == 654)]
    ens = seed_ensemble(v2[["model", "r", "B", "site_id", "host_id", "seed", "y_true", "y_pred"]])
    out, _ = hier_pair(stats_by_r(ens, "S", hosts), stats_by_r(ens, "D", hosts), draws)
    ref = json.loads((root / "results" / "c1c2_v2.json").read_text())["payload"]["A_mae"]["654"]
    diff = max_abs_diff(out["A"], ref, [("mean", "A"), ("per_resample", "per_resample"), ("ci", "ci95"),
                                        ("upper_one_sided", "upper_one_sided_95")])
    return {"label": "POST HOC (primary backbone v2)", "primary": with_bound(out),
            "sanity_c1c2_v2_654": {"max_abs_diff": diff, "tolerance": TOL,
                                   "passed": all(v <= TOL for v in diff.values())}}


def param_counts(root: Path) -> dict:
    """Trainable parameters of each (family, model, budget) from the run specs' hyperparameters (CPU, no training)."""
    from analysis.precision_cv_eval_table import read_records
    from analysis.sensitivity_eval_tables import stage_records
    from dftgnn.models import build_model, hparams_for, n_parameters

    def count(spec):
        net = build_model(spec["model"], hparams_for(spec["model"], spec["hp"]), n_host=12, n_site=10, cutoff=5.0)
        return n_parameters(net)

    srcs = {"cgcnn": stage_records(root / "results" / "cgcnn", "cgcnn"),
            "megnet_v1": read_records(root / "results", None),
            "megnet_v2": [p for p in stage_records(root / "results" / "v2_sweep", "v2_sweep")
                          if p["spec"]["model"] in ("S", "D-state")]}
    out = {}
    for fam, recs in srcs.items():
        for p in recs:
            sp = p["spec"]
            if int(sp["budget"]) not in CGCNN_BUDGETS or int(sp["seed"]) > 2 or sp["split"] != "outer_r0":
                continue
            if int(p["max_epochs"]) != 200:
                continue
            key = f"{fam}|{sp['model']}|{int(sp['budget'])}"
            n = count(sp)
            if out.get(key, n) != n:
                raise AssertionError(f"{key}: parameter count differs across seeds")
            out[key] = n
    return out


def cgcnn(long: pd.DataFrame, runs: pd.DataFrame, et: pd.DataFrame, root: Path) -> dict:
    hosts = test_hosts(0)
    w = cluster_weights(len(hosts))
    frames = {
        "cgcnn": seed_ensemble(long[long.stage == "cgcnn"][["model", "r", "B", "site_id", "host_id", "seed",
                                                            "y_true", "y_pred"]]).replace(
            {"model": {"cgcnn-S": "S", "cgcnn-D": "D"}}),
        "megnet_v1": et[(et.r == 0) & et.B.isin(CGCNN_BUDGETS)],
        "megnet_v2": seed_ensemble(long[(long.stage == "v2_sweep") & (long.r == 0)][
            ["model", "r", "B", "site_id", "host_id", "seed", "y_true", "y_pred"]]),
    }
    single = {"cgcnn": runs[runs.stage == "cgcnn"].replace({"model": {"cgcnn-S": "S", "cgcnn-D": "D"}}),
              "megnet_v2": runs[(runs.stage == "v2_sweep") & (runs.r == 0)]}
    table = {}
    for fam, f in frames.items():
        for B in CGCNN_BUDGETS:
            cell, reps = {}, {}
            for m in ("S", "D"):
                g = f[(f.model == m) & (f.B == B)]
                if len(g) != len(et[(et.model == "S") & (et.r == 0) & (et.B == B)]):
                    raise AssertionError(f"{fam} {m} {B}: site count differs from resample 0's test sites")
                st = padded_stats(g, hosts)
                cell[m] = {}
                for metric in ("mae", "within_host_mae"):
                    cell[m][metric], reps[(m, metric)] = cluster_summary(st, w, metric)
                if fam in single:
                    sr = single[fam]
                    sr = sr[(sr.model == m) & (sr.B == B)]
                    cell[m]["single_run_mae"] = {"mean": float(sr.metric.mean()), "sd": float(sr.metric.std(ddof=1)),
                                                 "n": len(sr)}
                    cell[m]["epochs_run"] = sr.epochs_run.astype(int).tolist()
                    cell[m]["best_epoch"] = sr.best_epoch.astype(int).tolist()
            cell["A"] = diff_summary(cell["S"]["mae"]["point"], cell["D"]["mae"]["point"],
                                     reps[("S", "mae")], reps[("D", "mae")])
            cell["A_within"] = diff_summary(cell["S"]["within_host_mae"]["point"],
                                            cell["D"]["within_host_mae"]["point"],
                                            reps[("S", "within_host_mae")], reps[("D", "within_host_mae")])
            table.setdefault(fam, {})[str(B)] = cell
    # paired CGCNN - MEGNet differences of site MAE (descriptive, same draws)
    paired = {}
    for ref in ("megnet_v1", "megnet_v2"):
        for B in CGCNN_BUDGETS:
            for m in ("S", "D"):
                sa = padded_stats(frames["cgcnn"][(frames["cgcnn"].model == m) & (frames["cgcnn"].B == B)], hosts)
                sb = padded_stats(frames[ref][(frames[ref].model == m) & (frames[ref].B == B)], hosts)
                a, ra = cluster_summary(sa, w, "mae")
                b, rb = cluster_summary(sb, w, "mae")
                paired[f"cgcnn_minus_{ref}|{m}|{B}"] = diff_summary(a["point"], b["point"], ra, rb)
    v1_et = frames["megnet_v1"]
    v1_single = {}
    for B in CGCNN_BUDGETS:
        for m in ("S", "D"):
            g = v1_et[(v1_et.model == m) & (v1_et.B == B)]
            maes = [float(np.abs(g[f"y_pred_seed{s}"] - g.y_true).mean()) for s in range(3)]
            v1_single[f"{m}|{B}"] = {"mean": float(np.mean(maes)), "sd": float(np.std(maes, ddof=1)), "n": 3}
    for key, v in v1_single.items():
        m, B = key.split("|")
        table["megnet_v1"][B][m]["single_run_mae"] = v
    return {
        "label": LABELS["cgcnn"], "resample": 0, "budgets": list(CGCNN_BUDGETS), "seeds": [0, 1, 2],
        "families": {"cgcnn": "cgcnn-S / cgcnn-D, untuned fixed optimiser settings (lr 1e-3, wd 1e-5, batch 32)",
                     "megnet_v1": "S / D-state, originally registered backbone, tuned-v1 (results/analysis/"
                                  "eval_table.parquet)",
                     "megnet_v2": "S / D-state, primary backbone v2, tuned-v2, POST HOC (results/v2_sweep/)"},
        "table": table, "paired_site_mae_differences": paired, "parameter_counts": param_counts(root),
        "bootstrap": {"method": "host-cluster bootstrap over the 164 test hosts of resample 0, one draw set shared "
                                "by every family, model, budget and metric", "draws": N_BOOT, "seed": BOOT_SEED,
                      "caveat": "one resample: the interval reflects test-host sampling only, not resample or split "
                                "variation; no hypothesis test"},
    }


def main() -> None:
    from dftgnn.config import load_config
    from dftgnn.io.results import write_result

    cfg = load_config()
    if float(cfg.stats.delta_eV) != DELTA_EV:
        raise AssertionError("stats.delta_eV differs from 0.05")
    long, runs, et, meta = load_inputs(ROOT)
    hm = high_moment(long, et, meta, ROOT)
    v2 = v2_primary(long, ROOT)
    cp = capped(long, runs, et, meta, ROOT)
    cp["v2_primary_654"] = v2
    hm["v2_primary_654"] = v2
    cg = cgcnn(long, runs, et, ROOT)
    for name, pay in (("sens_high_moment_v1", hm), ("sens_capped_post_hoc", cp), ("sens_cgcnn", cg)):
        pay = to_jsonable(pay)
        print(name, json.dumps({k: v for k, v in pay.items() if k.startswith("sanity")}, indent=0))
        print(write_result(name, pay, config=cfg, repo_root=ROOT))


if __name__ == "__main__":
    main()
