"""Staging analysis C3a: D - P and P - S per budget, TOST outcomes, P1 descriptor accuracy.

Rules:, row of 2026-10-14, items a, c, d, e, h; it restates the staging pairing of 2026-10-10 and the TOST
rule of 2026-10-09.

Estimator: the G1 estimator of ``dftgnn.stats.g1`` unchanged. Seed-ensemble mean per (model, r, B, site); per
resample host statistics; ``HierDraws.make`` over the 10 resamples, 2000 replicates, seed 20261008, one draw set for
S, D, P, every budget, both contrasts and both backbones. Point = mean over resamples; two-sided 95% interval =
2.5th and 97.5th percentiles; the TOST 90% interval = 5th and 95th percentiles (the one-sided 95% bounds).

Reads ``results/analysis/staging_eval_long.parquet`` and ``staging_p1_desc.parquet`` (sha256 checked against
``results/staging_eval_table.json``) and ``results/staging_swapin_check.json``. Writes through ``write_result``:

    results/staging_v2.json     primary backbone, with the reproduction of results/c3a_v2.json

    PYTHONPATH=$PWD/src python analysis/staging_analysis.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BUDGETS = (25, 50, 100, 200, 400, 654)
N_RESAMPLES = 10
N_BOOT = 2000
BOOT_SEED = 20261008
DELTA_P_EV = 0.05
DESC_BUDGETS = (654, 200)
CLASS_KEYS = {"host-electronic DFT": "host_electronic", "site-electronic DFT": "site_electronic"}
LABELS = {"v1": "registered staging analysis (ANALYSIS_PLAN §8 on the originally registered backbone v1; pairing "
                "clarified 2026-10-10 before any P result); outcome rule: TOST of the collaborator, ratified "
                "2026-10-11, POST HOC",
          "v2": "primary backbone (v2), POST HOC; outcome rule: TOST of the collaborator, ratified 2026-10-11, "
                "POST HOC; never pre-registered"}
# P - S of the v2 staging as quoted in the brief from the audit, eV, 3 decimals
BRIEF_V2_P_MINUS_S = {25: 0.044, 50: 0.123, 100: 0.066, 200: 0.119, 400: 0.024, 654: 0.028}
REPRO_TOL = 1e-9


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- estimator

def ensemble(long: pd.DataFrame) -> pd.DataFrame:
    """Seed-ensemble mean per (backbone, model, r, B, site); every cell must hold the same seeds."""
    from dftgnn.stats.g1 import seed_ensemble

    out = []
    for bb, g in long.groupby("backbone", sort=True):
        out.append(seed_ensemble(g, keys=("model", "r", "B", "site_id")).assign(backbone=bb))
    ens = pd.concat(out, ignore_index=True)
    if ens.n_seeds.nunique() != 1:
        raise AssertionError("cells with different seed counts")
    return ens


def stats_by_budget(ens: pd.DataFrame, model: str, budgets: Sequence[int] = BUDGETS,
                    n_resamples: int = N_RESAMPLES) -> dict[int, list[np.ndarray]]:
    from dftgnn.stats.g1 import resample_stats

    sub = ens[ens.model == model]
    return {B: [resample_stats(sub[(sub.B == B) & (sub.r == r)]) for r in range(n_resamples)] for B in budgets}


def make_draws(ens: pd.DataFrame, n_resamples: int = N_RESAMPLES, n_boot: int = N_BOOT, seed: int = BOOT_SEED):
    """One draw set over the resamples' test hosts, asserted identical for every model, budget and backbone."""
    from dftgnn.stats.g1 import HierDraws

    n_hosts = {}
    for (bb, m, B, r), g in ens.groupby(["backbone", "model", "B", "r"]):
        n_hosts.setdefault(r, set()).add(tuple(sorted(g.host_id.unique())))
    if any(len(v) != 1 for v in n_hosts.values()) or sorted(n_hosts) != list(range(n_resamples)):
        raise AssertionError("test hosts of a resample differ between models, budgets or backbones")
    return HierDraws.make([len(next(iter(n_hosts[r]))) for r in range(n_resamples)], n_boot, seed)


def bounds(reps: np.ndarray) -> dict:
    q = {k: float(np.quantile(reps, v)) for k, v in (("lo95", 0.025), ("hi95", 0.975), ("lo90", 0.05),
                                                       ("hi90", 0.95))}
    return {"ci95": [q["lo95"], q["hi95"]], "ci90": [q["lo90"], q["hi90"]],
            "lower_one_sided_95": q["lo90"], "upper_one_sided_95": q["hi90"]}


def paired(stats_a: Sequence[np.ndarray], stats_b: Sequence[np.ndarray], draws) -> dict:
    """a - b: point (mean over resamples), the bootstrap bounds, per-resample values."""
    from dftgnn.stats.g1 import point_metric

    pa = np.array([point_metric(s, "mae") for s in stats_a])
    pb = np.array([point_metric(s, "mae") for s in stats_b])
    reps = draws.replicates(stats_a, "mae") - draws.replicates(stats_b, "mae")
    return {"mean": float((pa - pb).mean()), **bounds(reps), "boot_sd": float(reps.std(ddof=1)),
            "per_resample": [float(x) for x in pa - pb]}


def marginal(stats: Sequence[np.ndarray], draws) -> dict:
    from dftgnn.stats.g1 import point_metric

    per = np.array([point_metric(s, "mae") for s in stats])
    reps = draws.replicates(stats, "mae")
    lo, hi = np.quantile(reps, [0.025, 0.975])
    return {"mean": float(per.mean()), "ci95": [float(lo), float(hi)], "per_resample": [float(x) for x in per]}


def tost_outcome(ci95: Sequence[float], ci90: Sequence[float], delta: float = DELTA_P_EV) -> dict:
    """The four-outcome rule of 2026-10-09 (ratified 2026-10-11), in its order, for P - S."""
    inside = bool(ci90[0] > -delta and ci90[1] < delta)
    if ci95[1] < 0:
        base = "P better"
    elif ci95[0] > 0:
        base = "P worse"
    elif inside:
        base = "equivalent"
    else:
        base = "inconclusive"
    within = base in ("P better", "P worse") and inside
    return {"outcome": base, "within_equivalence_margin": within,
            "report": base + ("; different but within the equivalence margin" if within else ""),
            "ci90_inside_margin": inside}


def noninferior(upper: float, delta: float = DELTA_P_EV) -> bool:
    """Upper half of the TOST (item d): one-sided 95% upper bound of P - S below delta_P, strict."""
    return bool(upper < delta)


def staging_block(ens: pd.DataFrame, draws, delta: float = DELTA_P_EV,
                  budgets: Sequence[int] = BUDGETS) -> tuple[dict, dict]:
    """Per budget: MAE of S, D, P and the two contrasts; second value: S - D per budget, for the G1 sanity check
    only."""
    st = {m: stats_by_budget(ens, m, budgets, len(draws.n_hosts)) for m in ("S", "D", "P")}
    out, a = {}, {}
    for B in budgets:
        ps = paired(st["P"][B], st["S"][B], draws)
        ps.update(tost_outcome(ps["ci95"], ps["ci90"], delta), noninferior=noninferior(ps["upper_one_sided_95"], delta))
        out[str(B)] = {"mae": {m: marginal(st[m][B], draws) for m in ("S", "D", "P")},
                       "D_minus_P": paired(st["D"][B], st["P"][B], draws), "P_minus_S": ps}
        a[str(B)] = paired(st["S"][B], st["D"][B], draws)
    return out, a


def summary(block: Mapping) -> dict:
    ps = {B: v["P_minus_S"] for B, v in block.items()}
    return {"n_budgets": len(ps),
            "n_budgets_P_better": sum(v["outcome"] == "P better" for v in ps.values()),
            "any_budget_P_better": any(v["outcome"] == "P better" for v in ps.values()),
            "n_budgets_noninferior": sum(v["noninferior"] for v in ps.values()),
            "noninferior_by_budget": {B: v["noninferior"] for B, v in ps.items()},
            "outcome_by_budget": {B: v["report"] for B, v in ps.items()},
            "n_budgets_D_minus_P_ci95_below_zero": sum(v["D_minus_P"]["ci95"][1] < 0 for v in block.values())}


# ---------------------------------------------------------------- P1 descriptors

def p1_summary(desc: pd.DataFrame, budgets: Sequence[int] = DESC_BUDGETS, n_runs: int = 30) -> dict:
    out = {}
    for B in budgets:
        sub = desc[desc.B == B]
        per = {}
        for nm, g in sub.groupby("descriptor", sort=True):
            if len(g) != n_runs:
                raise AssertionError(f"B = {B} {nm}: {len(g)} runs, expected {n_runs}")
            per[nm] = {"class": CLASS_KEYS[g["class"].iloc[0]], "n_runs": len(g), "n_sites_per_run_mean":
                       float(g.n.mean()), "r2_mean": float(g.r2.mean()), "r2_sd": float(g.r2.std(ddof=1)),
                       "mae_mean": float(g.mae.mean()), "mae_sd": float(g.mae.std(ddof=1))}
        cls = {}
        for key in CLASS_KEYS.values():
            r2 = [v["r2_mean"] for v in per.values() if v["class"] == key]
            cls[key] = {"n_descriptors": len(r2), "class_mean_r2": float(np.mean(r2)),
                        "min_r2_mean": float(np.min(r2)), "max_r2_mean": float(np.max(r2))}
        out[str(B)] = {"per_descriptor": per, "by_class": cls}
    return out


# ---------------------------------------------------------------- reproduction of the collaborator (item h)

def reproduce_his(block: Mapping, p1: Mapping, his: Mapping, m1: Mapping) -> dict:
    """Our v2 values against results/c3a_v2.json (his) and results/ingest_m1_recompute.json."""
    hb, mb = his["budgets"], m1["per_budget"]
    rows, d_pt, d_ci, d_m1 = {}, [], [], []
    for B, v in block.items():
        h, m = hb[B], mb[B]
        ps, dp = v["P_minus_S"], v["D_minus_P"]
        pt = {"P_minus_S": ps["mean"] - h["P_minus_S"]["mean"], "D_minus_P": dp["mean"] - h["D_minus_P"]["mean"],
              **{f"mae_{k}": v["mae"][k]["mean"] - h["mae"][k]["mean"] for k in ("S", "D", "P")}}
        ci = {"P_minus_S_ci95": np.subtract(ps["ci95"], h["P_minus_S"]["ci95"]).tolist(),
              "P_minus_S_ci90": np.subtract(ps["ci90"], h["P_minus_S"]["ci90"]).tolist(),
              "D_minus_P_ci95": np.subtract(dp["ci95"], h["D_minus_P"]["ci95"]).tolist()}
        his_out = h["P_minus_S"]["outcome"]
        same = his_out.startswith(ps["outcome"]) and (("within" in his_out) == ps["within_equivalence_margin"])
        m1d = {"P_minus_S": ps["mean"] - m["P_minus_S"], "D_minus_P": dp["mean"] - m["D_minus_P"],
               "P_minus_S_ci95": float(np.max(np.abs(np.subtract(ps["ci95"], m["P_minus_S_ci95"])))),
               "P_minus_S_ci90": float(np.max(np.abs(np.subtract(ps["ci90"], m["P_minus_S_ci90"])))),
               "D_minus_P_ci95": float(np.max(np.abs(np.subtract(dp["ci95"], m["D_minus_P_ci95"]))))}
        rows[B] = {"point_minus_his": pt, "interval_minus_his": ci, "his_outcome": his_out,
                   "our_outcome": ps["report"], "outcome_agrees": bool(same), "minus_m1_recompute": m1d,
                   "brief_value_P_minus_S": BRIEF_V2_P_MINUS_S[int(B)],
                   "agrees_with_brief_to_3_decimals": bool(round(ps["mean"], 3) == BRIEF_V2_P_MINUS_S[int(B)])}
        d_pt += [abs(x) for x in pt.values()]
        d_ci += [abs(x) for x in np.ravel(list(ci.values()))]
        d_m1 += [abs(x) for x in m1d.values()]
    p1d = []
    for B, v in p1.items():
        for nm, w in v["per_descriptor"].items():
            hv = his["p1_descriptors"][B][nm]
            p1d += [abs(w["r2_mean"] - hv["r2_mean"]), abs(w["mae_mean"] - hv["mae_mean"])]
    return {"per_budget": rows,
            "max_abs_point_difference_vs_his_eV": float(max(d_pt)),
            "points_reproduced_to_1e-9": bool(max(d_pt) < REPRO_TOL),
            "max_abs_interval_endpoint_difference_vs_his_eV": float(max(d_ci)),
            "interval_note": "his bootstrap generator (seed 0, dftgnn.stats.metrics._draw_weights) differs from "
                             "HierDraws (seed 20261008); endpoints differ by resampling noise only",
            "all_outcomes_agree": all(r["outcome_agrees"] for r in rows.values()),
            "max_abs_difference_vs_m1_recompute": float(max(d_m1)),
            "all_brief_values_agree_to_3_decimals": all(r["agrees_with_brief_to_3_decimals"] for r in rows.values()),
            "p1_descriptor_max_abs_difference_vs_his": float(max(p1d)),
            "p1_descriptors_reproduced_to_1e-9": bool(max(p1d) < REPRO_TOL),
            "backbone_of_the_brief_values": "v2 (P-v2; docs/ingest_report.md row 37, results/c3a_v2.json)"}


def sanity_vs_g1(a_by_budget: Mapping, ref: Mapping, backbone: str) -> dict:
    """S - D from the staging pipeline against the G1 records (same estimator and draws): v1 results/g1_primary.json,
    v2 results/c1c2_v2.json (points only for v2; its interval used another generator)."""
    d_pt, d_ci = [], []
    for B, a in a_by_budget.items():
        if backbone == "v1":
            r = ref["per_budget"][B]["A"]
            d_pt.append(abs(a["mean"] - r["mean"]))
            d_ci.append(float(np.max(np.abs(np.subtract(a["ci95"], r["ci"])))))
        else:
            d_pt.append(abs(a["mean"] - ref["A_mae"][B]["A"]))
    out = {"max_abs_A_point_difference_eV": float(max(d_pt)), "A_points_reproduced_to_1e-9": bool(max(d_pt) < 1e-9)}
    if d_ci:
        out.update({"max_abs_A_ci95_difference_eV": float(max(d_ci)),
                    "A_ci95_reproduced_to_1e-9": bool(max(d_ci) < 1e-9)})
    return out


# ---------------------------------------------------------------- main

def load_table(name: str, rec: Mapping) -> tuple[pd.DataFrame, dict]:
    ref = rec["payload"]["files"][name]
    got = sha256(ROOT / ref["path"])
    if got != ref["sha256"]:
        raise AssertionError(f"{name}: sha256 differs from the eval-table record")
    return pd.read_parquet(ROOT / ref["path"]), {"path": ref["path"], "sha256": got}


def main() -> None:
    from dftgnn.config import load_config
    from dftgnn.io.results import write_result

    cfg = load_config()
    delta = float(cfg.analysis.equivalence_margin_eV)
    if delta != DELTA_P_EV:
        raise AssertionError(f"analysis.equivalence_margin_eV = {delta}, expected {DELTA_P_EV}")
    rec = json.loads((ROOT / "results" / "staging_eval_table.json").read_text())
    long, f_long = load_table("staging_eval_long", rec)
    desc, f_desc = load_table("staging_p1_desc", rec)
    swap_path = ROOT / "results" / "staging_swapin_check.json"
    swap = json.loads(swap_path.read_text())
    ens = ensemble(long)
    draws = make_draws(ens)
    rd = lambda n: json.loads((ROOT / "results" / f"{n}.json").read_text())["payload"]
    counts = rec["payload"]["integrity"]["counts"]
    for bb in ("v1", "v2"):
        block, a_sd = staging_block(ens[ens.backbone == bb], draws, delta)
        p1 = p1_summary(desc[desc.backbone == bb])
        payload = {
            "label": LABELS[bb], "backbone": bb,
            "rules": "docs/deviations.md row of 2026-10-14 (§8, step A2); pairing row of 2026-10-10 (§8); TOST row "
                     "of 2026-10-09 (§8, §9), ratified 2026-10-11",
            "delta_P_eV": delta, "budgets": block, "summary": summary(block), "p1_descriptors": p1,
            "estimator": {"seed_ensemble": "mean over seeds 0-2", "bootstrap": "dftgnn.stats.g1.HierDraws",
                          "n_boot": N_BOOT, "seed": BOOT_SEED, "n_resamples": N_RESAMPLES,
                          "draws_shared": "S, D, P, every budget, both contrasts, both backbones",
                          "ci90": "5th and 95th percentiles = one-sided 95% lower and upper bounds",
                          "two_sided_ci": 0.95, "tost_ci": 0.90, "one_sided_quantile": 0.95},
            "integrity": {"counts": {m: counts[f"{bb}|{m}"] for m in ("S", "D", "P", "P1")},
                          "eval_table": {"record_git_sha": rec["git_sha"], "files": [f_long, f_desc]},
                          "swap_in": {**swap["payload"]["summary"][bb], "record": "results/staging_swapin_check.json",
                                      "record_sha256": sha256(swap_path), "record_git_sha": swap["git_sha"]},
                          "sanity_S_minus_D_vs_g1": sanity_vs_g1(a_sd, rd("g1_primary" if bb == "v1" else "c1c2_v2"),
                                                                 bb)},
        }
        if bb == "v2":
            payload["reproduction_of_collaborator"] = reproduce_his(block, p1, rd("c3a_v2"),
                                                                   rd("ingest_m1_recompute")["staging_v2"])
        print(bb, json.dumps(payload["summary"]), json.dumps(payload["integrity"]["sanity_S_minus_D_vs_g1"]))
        print(write_result(f"staging_{bb}", payload, config=cfg, repo_root=ROOT))


if __name__ == "__main__":
    main()
