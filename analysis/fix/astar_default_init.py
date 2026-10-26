"""The descriptor advantage measured against the default-initialisation structure-only network.

Nothing is trained. Physically: A* = MAE_S - MAE_D* is the error of the neutral oxygen-vacancy formation energy that
the pristine-cell electronic descriptors (band gap, oxide formation energy, dielectric response, band-edge character,
Bader charge) remove beyond what the crystal structure alone gives. The headline A* uses the He (Kaiming)
initialised structure-only network (backbone v2). The default-initialisation structure-only network (backbone v1) on
the same resamples and seeds reaches a lower vacancy-energy error, so the structural baseline the descriptors are
measured against is weaker under He initialisation. Here the gap is recomputed with the stronger structural baseline:
S_default - D*, on exactly the headline estimator and bootstrap draws, next to S_he - D* (the headline) and
S_default - S_he (what the initialisation alone changes in the structure-only error).

Data: S_default = results/analysis/precision_eval_long.parquet backbone v1, model S, 654 training hosts, resamples
r = 0-29, seeds 0-4; S_he = the same with backbone v2; D* = results/dretrain_v2 at 654 hosts (30 x 5), loaded and
checked by analysis/dstar/dstar_analysis.load_dstar.

Estimator (that of analysis/dstar/dstar_analysis.headline, imported unchanged): seed-ensemble mean prediction per
(model, r, site); per-resample host sufficient statistics (hosts sorted by host_id); site MAE per resample; point =
mean over the 30 resamples; bootstrap = HierDraws.make([164] * 30, 2000, 20261008), one draw set shared by the three
networks, so every gap is paired (replicate gap = difference of replicate MAEs); 95% = 2.5 / 97.5 percentiles,
upper_one_sided_95 = 95th percentile; n_resamples_above_delta counts per-resample gaps strictly above delta = 0.05 eV
(configs/config.yaml stats.delta_eV). Under "notes" the same gaps with the host-cluster bootstrap of
analysis/trv1/boot_design.py (818 distinct test hosts as the sampling unit, V = default_rng(20261008).multinomial,
2000 replicates), reused unchanged.

Reproduction (tolerance 1e-9): identical test sites, targets and hosts per resample for S_default, S_he and D*;
MAE_S_default (mean, ci95) equals results/derived_c1/default_init_ablation.json payload.table.rows.B654_30x5.MAE_S;
S_he - D* (A, ci95, upper_one_sided_95) equals results/derived_dstar/dstar_gaps.json
payload.headline_654_30x5.654.metrics.mae.gaps.S_minus_Dstar; the host-cluster S_he - D* equals
results/derived_trv1_boot/boot_design.json payload.schemes.A_star.host. Writes through write_result into
results/derived_astar_default/astar_default_init.json.

    PYTHONPATH=$PWD/src python analysis/fix/astar_default_init.py [--dry-run --out FILE]
"""
from __future__ import annotations

import argparse
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
from analysis.c1.final_network import a_entry, compare, ms
from analysis.dstar import dstar_analysis as da
from analysis.trv1 import boot_design as bd
from dftgnn.stats.g1 import HierDraws, contrast, seed_ensemble
from dftgnn.stats.metrics import host_stats

NAME = "astar_default_init"
OUT_DIR = ROOT / "results" / "derived_astar_default"
B = 654
N_R, N_S, N_HOSTS = 30, 5, 164
TOL = 1e-9
MODELS = ("S_default", "S_he", "Dstar")
GAPS = (("s_default_minus_dstar", "S_default", "Dstar"), ("s_he_minus_dstar", "S_he", "Dstar"),
        ("s_default_minus_s_he", "S_default", "S_he"))
KEY = {"S_default": "s_default", "S_he": "s_he", "Dstar": "dstar"}


def load() -> tuple[pd.DataFrame, dict]:
    ref, f_ref = da.s_reference()
    dstar, info = da.load_dstar(ref)
    prec, _ = c.load_table("precision_eval_long")
    sel = prec[(prec.model == "S") & (prec.B == B) & (prec.r < N_R) & (prec.seed < N_S)]
    parts = [sel[sel.backbone == "v1"][da.COLS].assign(model="S_default"),
             sel[sel.backbone == "v2"][da.COLS].assign(model="S_he"),
             dstar[dstar.B == B][da.COLS]]
    long = pd.concat(parts, ignore_index=True)
    return long, {**f_ref, "dretrain_v2_spec": {"path": "specs/dretrain_v2.jsonl", "sha256": info["spec_sha256"]}}


def check_sites(ens: pd.DataFrame) -> dict:
    """Identical test sites, targets and hosts per resample for the three networks."""
    n_sites = []
    for r in range(N_R):
        fr = {m: ens[(ens.model == m) & (ens.r == r)].set_index("site_id").sort_index() for m in MODELS}
        base = fr["S_he"]
        for m in ("S_default", "Dstar"):
            f = fr[m]
            if not (f.index.equals(base.index) and np.array_equal(f.y_true.to_numpy(), base.y_true.to_numpy())
                    and (f.host_id == base.host_id).all()):
                raise AssertionError(f"r = {r}: test sites, targets or hosts of {m} differ from S_he")
        if base.host_id.nunique() != N_HOSTS:
            raise AssertionError(f"r = {r}: {base.host_id.nunique()} test hosts")
        n_sites.append(len(base))
    return {"identical_test_sites_targets_hosts_all_resamples": True, "n_resamples_checked": N_R,
            "test_sites_per_resample": {"min": int(min(n_sites)), "max": int(max(n_sites))}}


def gap_block(diff: dict, delta: float) -> dict:
    e = a_entry(diff)
    per = np.asarray(diff["per_resample"], float)
    lo, hi = e["ci95"]
    return {"point": e["A"], "ci95": e["ci95"], "upper_one_sided_95": e["upper_one_sided_95"],
            "n_resamples_above_delta": int((per > delta).sum()), "ci95_excludes_zero": e["ci95_excludes_zero"],
            "ci95_inside_pm_delta": bool(lo > -delta and hi < delta), "lower_above_delta": bool(lo > delta)}


def host_cluster(ens: pd.DataFrame, delta: float) -> dict:
    """The host-cluster scheme of analysis/trv1/boot_design.py on the site MAE of the three networks."""
    st, hosts = {m: [] for m in MODELS}, []
    for r in range(N_R):
        for m in MODELS:
            f = bd.frame_of(ens, m, r)
            hs = host_stats(f.y_true.to_numpy(), f.y_pred.to_numpy(), f.host_id.to_numpy())
            st[m].append(hs[["sabs", "n"]].to_numpy())
            if m == "S_he":
                hosts.append(hs.index.to_numpy())
    all_hosts = np.unique(np.concatenate(hosts))
    idx = [np.searchsorted(all_hosts, h) for h in hosts]
    V, _ = bd.host_draws(len(all_hosts), c.N_BOOT, c.BOOT_SEED, N_R)
    sch = bd.Scheme("host", V=V, idx=idx)
    rep = {m: sch.reps(st[m]) for m in MODELS}
    per = {m: bd.per_resample(st[m]) for m in MODELS}
    out = {"scheme": "host-cluster: V = default_rng(20261008).multinomial(n_hosts, uniform, size=2000) over the "
                     f"{len(all_hosts)} distinct test hosts (sorted by host_id); analysis/trv1/boot_design.py",
           "n_distinct_test_hosts": int(len(all_hosts))}
    for k, a, b in GAPS:
        s = bd.summ(float((per[a] - per[b]).mean()), rep[a] - rep[b])
        lo, hi = s["ci95"]
        s.update({"ci95_excludes_zero": bool(lo > 0 or hi < 0), "ci95_inside_pm_delta": bool(lo > -delta and hi < delta),
                  "lower_above_delta": bool(lo > delta)})
        out[k] = s
    for m in MODELS:
        out[f"mae_{KEY[m]}"] = bd.summ(float(per[m].mean()), rep[m])
    return out


def run(delta: float) -> dict:
    long, inputs = load()
    ens = seed_ensemble(long)
    if (ens.n_seeds != N_S).any() or sorted(ens.r.unique()) != list(range(N_R)):
        raise AssertionError("seed ensemble over the wrong seeds or resamples")
    if set(ens.model) != set(MODELS):
        raise AssertionError(f"models {sorted(set(ens.model))}")
    site_check = check_sites(ens)
    draws = HierDraws.make([pa.N_HOSTS_PER_RESAMPLE] * N_R, c.N_BOOT, seed=c.BOOT_SEED)
    st = {m: pa.resample_stats_by_budget(ens, m, N_R, (B,)) for m in MODELS}
    raw = {k: contrast(st[a], st[b], draws, "mae", ci=c.CI, upper_quantile=c.UPPER_Q)[B] for k, a, b in GAPS}
    mae = {"s_default": ms(raw["s_default_minus_dstar"]["a"]), "s_he": ms(raw["s_he_minus_dstar"]["a"]),
           "dstar": ms(raw["s_he_minus_dstar"]["b"])}
    gaps = {k: gap_block(raw[k]["diff"], delta) for k, _, _ in GAPS[:2]}
    sd = raw["s_default_minus_s_he"]["diff"]
    gaps["s_default_minus_s_he"] = {"point": float(sd["mean"]), "ci95": [float(sd["ci"][0]), float(sd["ci"][1])]}

    # reproduction
    abl = c.record("derived_c1/default_init_ablation")["payload"]["table"]["rows"]["B654_30x5"]
    hd = c.record("derived_dstar/dstar_gaps")["payload"]["headline_654_30x5"][str(B)]["metrics"]["mae"]
    ref_gap = hd["gaps"]["S_minus_Dstar"]
    he = gap_block(raw["s_he_minus_dstar"]["diff"], delta)
    hc = host_cluster(ens, delta)
    bdh = c.record("derived_trv1_boot/boot_design")["payload"]["schemes"]
    repro = {
        "test_sites": site_check,
        "mae_s_default_vs_default_init_ablation": compare(
            {"MAE_S": mae["s_default"]}, {"MAE_S": {k: abl["MAE_S"][k] for k in ("mean", "ci95")}},
            "results/derived_c1/default_init_ablation.json payload.table.rows.B654_30x5.MAE_S (mean, ci95)"),
        "s_he_minus_dstar_vs_dstar_gaps": compare(
            {"A": he["point"], "ci95": he["ci95"], "upper_one_sided_95": he["upper_one_sided_95"]},
            {k: ref_gap[k] for k in ("A", "ci95", "upper_one_sided_95")},
            "results/derived_dstar/dstar_gaps.json payload.headline_654_30x5.654.metrics.mae.gaps.S_minus_Dstar "
            "(A, ci95, upper_one_sided_95)"),
        "mae_s_he_dstar_vs_dstar_gaps": compare(
            {"S": mae["s_he"], "Dstar": mae["dstar"]}, {k: hd["MAE"][k] for k in ("S", "Dstar")},
            "results/derived_dstar/dstar_gaps.json payload.headline_654_30x5.654.metrics.mae.MAE (S, Dstar)"),
        "n_above_delta_s_he_minus_dstar": {"mine": he["n_resamples_above_delta"],
                                           "reference": int(ref_gap["n_resamples_above_delta"]),
                                           "reproduced": he["n_resamples_above_delta"] == int(
                                               ref_gap["n_resamples_above_delta"])},
        "host_cluster_s_he_minus_dstar_vs_boot_design": compare(
            {k: hc["s_he_minus_dstar"][k] for k in ("point", "ci95", "upper_one_sided_95", "boot_sd")},
            {k: bdh["A_star"]["host"][k] for k in ("point", "ci95", "upper_one_sided_95", "boot_sd")},
            "results/derived_trv1_boot/boot_design.json payload.schemes.A_star.host"),
        "host_cluster_mae_s_he_vs_boot_design": compare(
            {k: hc["mae_s_he"][k] for k in ("point", "ci95", "upper_one_sided_95")},
            {k: bdh["MAE_S"]["host"][k] for k in ("point", "ci95", "upper_one_sided_95")},
            "results/derived_trv1_boot/boot_design.json payload.schemes.MAE_S.host"),
    }
    bad = [k for k, v in repro.items() if k != "test_sites" and not v["reproduced"]]
    if bad:
        raise AssertionError(f"not reproduced: {json.dumps({k: repro[k] for k in bad}, default=str)}")
    numeric = [v["max_abs_difference"] for v in repro.values() if "max_abs_difference" in v]
    repro["summary"] = {"max_abs_difference": float(max(numeric)), "tolerance": TOL, "all_reproduced": True}

    return {
        "design": "654 training hosts, 30 resamples x 5 seeds", "n_resamples": N_R, "n_seeds": N_S,
        "n_test_hosts_per_resample": N_HOSTS,
        "mae": mae,
        "s_default_minus_dstar": gaps["s_default_minus_dstar"],
        "s_he_minus_dstar": gaps["s_he_minus_dstar"],
        "s_default_minus_s_he": gaps["s_default_minus_s_he"],
        "reproduction": repro,
        "notes": {
            "networks": {"S_default": "structure-only network, default initialisation (backbone v1), "
                                      "precision_eval_long.parquet",
                         "S_he": "structure-only network, He (Kaiming) initialisation (backbone v2), "
                                 "precision_eval_long.parquet; the headline S",
                         "Dstar": "re-tuned descriptor network, results/dretrain_v2 (D-late, 654 hosts, 30 x 5)"},
            "estimator": "analysis/dstar/dstar_analysis.headline: seed-ensemble mean per (model, r, site); "
                         "per-resample site MAE; point = mean over 30 resamples; HierDraws.make([164] * 30, 2000, "
                         "20261008) shared by all networks (paired); 95% = 2.5 / 97.5 percentiles; "
                         "upper_one_sided_95 = 95th percentile",
            "sign_convention": "model a minus model b in each key name; positive = b more accurate",
            "delta_eV": delta,
            "flags": "n_resamples_above_delta: per-resample gap > delta; ci95_inside_pm_delta: -delta < lo and "
                     "hi < delta; lower_above_delta: lo > delta (strict)",
            "per_resample": {k: [float(x) for x in raw[k]["diff"]["per_resample"]] for k, _, _ in GAPS},
            "boot_sd": {k: float(raw[k]["diff"]["boot_sd"]) for k, _, _ in GAPS},
            "host_cluster_bootstrap": hc,
            "inputs": {**inputs, "default_init_ablation": "results/derived_c1/default_init_ablation.json",
                       "dstar_gaps": "results/derived_dstar/dstar_gaps.json",
                       "boot_design": "results/derived_trv1_boot/boot_design.json"},
        },
    }


def main(argv: list[str] | None = None) -> None:
    from dftgnn.config import load_config
    from dftgnn.io.results import write_result

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", type=Path, default=Path(f"{NAME}_dryrun.json"))
    a = ap.parse_args(argv)
    t0 = time.time()
    cfg = load_config()
    delta = float(cfg.stats.delta_eV)
    if delta != c.DELTA_EV:
        raise AssertionError("stats.delta_eV differs from the analysis constant")
    payload = run(delta)
    payload["notes"]["runtime_s"] = round(time.time() - t0, 1)
    payload = c.to_jsonable(payload)
    text = json.dumps(payload, indent=1, allow_nan=False)
    if a.dry_run:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text)
        print(a.out, f"{payload['notes']['runtime_s']} s")
        return
    print(write_result(NAME, payload, config=cfg, repo_root=ROOT, results_dir=OUT_DIR))


if __name__ == "__main__":
    main()
