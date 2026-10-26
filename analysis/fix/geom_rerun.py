"""Sensitivity of the vacancy-energy error to pristine host geometry, recomputed from the stored per-site predictions.

Nothing is trained. Physically: the networks see the pristine host only through its relaxed DFT geometry. If the
pristine cells are instead relaxed with the MACE-MP-0 machine-learned potential (condition "relaxed"), the volume,
the cell shape at fixed volume and the internal atomic positions all shift slightly from DFT; rescaling the relaxed
cell isotropically back to the DFT volume (condition "rescaled") removes the volume part only. How much the site
error of the neutral oxygen-vacancy formation energy then changes measures how strongly the learned map leans on
geometric detail that a cheaper relaxation gets wrong. D keeps its DFT descriptors under every condition, so its
change isolates the graph input; P passes structure-predicted descriptors on to D and so sees the geometry twice.

Data: the default-initialisation network (backbone v1), learning-curve design (resamples r = 0-9, seeds 0-2, 654 and
200 training hosts), our MACE-MP-0 relaxations of all 818 universe hosts (results/mlip_v1/, one record per host).
Per-site predictions under the three conditions: results/geomeval/predictions/*.parquet, one per run record
results/geomeval/*.json (180 runs; prediction sha256 checked against each record). Formulas of the hosts from
data/tiling_summary_v1.csv.

Estimator (that of analysis/geometry_analysis.py, whose ensemble, draw and Delta-MAE functions are imported
unchanged): seed-ensemble mean prediction per (model, r, B, condition, site) (dftgnn.stats.g1.seed_ensemble);
per-resample host sufficient statistics; Delta-MAE(c) = MAE(c) - MAE(DFT geometry) per resample; point = mean over
the 10 resamples; bootstrap = dftgnn.stats.g1.HierDraws.make(test-host counts per resample, 2000, 20261008), one
draw set shared by every model, condition and budget, so each Delta-MAE is paired; 95% = 2.5 / 97.5 percentiles.

Further blocks
    counts                    relax records, converged relaxations, hosts and sites with predictions.
    nonconverged_formulas     formulas of the hosts whose relaxation did not converge (comma-separated, sorted).
    geometry                  medians of internal RMSD (Angstrom), isotropic strain eps_v and deviatoric strain over
                              the 739 EVALUATED hosts (host columns of the prediction tables). Note: the null rule of
                              results/geometry_c3b.json takes its median internal RMSD over all 818 relaxed hosts.
    per_host_abs_change_s_654 S at 654 hosts, relaxed cells: per host the mean over (r, site) of
                              |ensemble prediction relaxed - ensemble prediction DFT|, then the mean over the hosts of
                              each d-electron class (d0, d10, other of data/host_dbins_v1.csv; 'unassigned' omitted).
    max_abs_delta_mae_654     the largest |point Delta-MAE| over S, D, P and both conditions at 654 hosts.

Reproduction (tolerance 1e-9, asserted): mean and ci95 of every Delta-MAE of results/geometry_c3b.json
payload.budgets (S, D, P x mlip, mlip_rescaled x 654, 200); the per-host changes of S, 654, mlip against
results/analysis/geometry_host_change.parquet; host and site counts. Writes through write_result into
results/derived_geom_rerun/geom_rerun.json.

    PYTHONPATH=$PWD/src python analysis/fix/geom_rerun.py [--dry-run --out FILE]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

NAME = "geom_rerun"
OUT_DIR = ROOT / "results" / "derived_geom_rerun"
MODELS = ("S", "D", "P")
BUDGETS = (654, 200)
CONDS = {"relaxed": "mlip", "rescaled": "mlip_rescaled"}
CLASSES = ("d0", "d10", "other")
TOL = 1e-9
N_RUNS = 180
N_RELAX = 818


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_long() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Long table (model, r, B, seed, condition, site_id, host_id, y_true, y_pred) and the per-host columns."""
    recs = [json.loads(f.read_text())["payload"] for f in sorted((ROOT / "results" / "geomeval").glob("*.json"))]
    if len(recs) != N_RUNS:
        raise AssertionError(f"{len(recs)} geomeval records, expected {N_RUNS}")
    cells = {(p["key"]["model"], int(p["key"]["r"]), int(p["key"]["budget"]), int(p["key"]["seed"])) for p in recs}
    if len(cells) != N_RUNS:
        raise AssertionError("duplicate (model, r, B, seed) cells")
    frames, hosts = [], []
    for p in recs:
        path = ROOT / p["predictions"]["path"]
        if sha256(path) != p["predictions"]["sha256"]:
            raise AssertionError(f"{p['run_id']}: prediction sha256 differs from the record")
        pred = pd.read_parquet(path)
        k = p["key"]
        frames.append(pd.DataFrame({"model": k["model"], "r": int(k["r"]), "B": int(k["budget"]),
                                    "seed": int(k["seed"]), "condition": pred.condition.to_numpy(),
                                    "site_id": pred.site_id.to_numpy(), "host_id": pred.host_id.to_numpy(),
                                    "y_true": pred.target.to_numpy(float), "y_pred": pred.prediction.to_numpy(float)}))
        hosts.append(pred[["host_id", "converged", "d_bin", "eps_v", "deviatoric", "internal_rmsd_A"]])
    h = pd.concat(hosts, ignore_index=True).drop_duplicates()
    if h.host_id.duplicated().any():
        raise AssertionError("a host carries different geometry columns across runs")
    return pd.concat(frames, ignore_index=True), h.sort_values("host_id").reset_index(drop=True)


def load_relax() -> pd.DataFrame:
    rows = []
    for f in sorted((ROOT / "results" / "mlip_v1").glob("*.json")):
        p = json.loads(f.read_text())["payload"]
        rows.append({"host_id": p["host_id"], "converged": bool(p["converged"]),
                     **{c: float(p["geometry"][c]) for c in ("eps_v", "deviatoric", "internal_rmsd_A")}})
    r = pd.DataFrame(rows)
    if len(r) != N_RELAX or r.host_id.duplicated().any():
        raise AssertionError(f"{len(r)} relax records, expected one per host ({N_RELAX})")
    return r


def run() -> dict:
    from analysis.geometry_analysis import delta_block, ensemble, make_draws

    long, hcols = load_long()
    relax = load_relax()
    tiling = pd.read_csv(ROOT / "data" / "tiling_summary_v1.csv").set_index("host_id")

    # host columns of the prediction tables equal the relax records
    rx = relax.set_index("host_id").loc[hcols.host_id]
    if not (rx.converged.to_numpy() == hcols.converged.to_numpy()).all():
        raise AssertionError("converged differs between prediction tables and relax records")
    for c in ("eps_v", "deviatoric", "internal_rmsd_A"):
        if not np.array_equal(rx[c].to_numpy(float), hcols[c].to_numpy(float)):
            raise AssertionError(f"{c} differs between prediction tables and relax records")

    ens = ensemble(long)
    draws, n_hosts = make_draws(ens)
    delta, _ = delta_block(ens, draws)

    # reproduction of geometry_c3b.json payload.budgets
    ref = json.loads((ROOT / "results" / "geometry_c3b.json").read_text())["payload"]
    diffs, out_delta = {}, {}
    for B in BUDGETS:
        out_delta[str(B)] = {}
        for m in MODELS:
            out_delta[str(B)][m] = {}
            for name, c in CONDS.items():
                got = delta[str(B)]["delta_mae"][m][c]
                want = ref["budgets"][str(B)]["delta_mae"][m][c]
                d = max(abs(got["mean"] - want["mean"]),
                        *(abs(x - y) for x, y in zip(got["ci95"], want["ci95"], strict=True)))
                diffs[f"{B}_{m}_{c}"] = d
                out_delta[str(B)][m][name] = {"mean": got["mean"], "ci95": got["ci95"]}
    bad = {k: v for k, v in diffs.items() if not v <= TOL}
    if bad:
        raise AssertionError(f"Delta-MAE differs from results/geometry_c3b.json beyond {TOL}: {bad}")

    n_hosts_eval, n_sites_eval = int(ens.host_id.nunique()), int(ens.site_id.nunique())
    if n_hosts_eval != ref["n_hosts_evaluated"] or n_sites_eval != ref["n_sites_evaluated"]:
        raise AssertionError("host or site counts differ from results/geometry_c3b.json")
    if n_hosts != ref["n_test_hosts_per_resample"]:
        raise AssertionError("test-host counts per resample differ from results/geometry_c3b.json")

    # per-host |change| of S at 654 on relaxed cells
    s = ens[(ens.model == "S") & (ens.B == 654)]
    piv = s.pivot_table(index=["r", "site_id", "host_id"], columns="condition", values="y_pred").reset_index()
    if piv[["dft", "mlip"]].isna().any().any():
        raise AssertionError("a site lacks a condition")
    per_host = piv.assign(ch=(piv["mlip"] - piv["dft"]).abs()).groupby("host_id").ch.mean()
    hc = pd.read_parquet(ROOT / "results" / "analysis" / "geometry_host_change.parquet")
    hc = hc[(hc.model == "S") & (hc.B == 654) & (hc.condition == "mlip")].set_index("host_id").mean_abs_change
    if set(hc.index) != set(per_host.index):
        raise AssertionError("host sets differ from geometry_host_change.parquet")
    hc_diff = float(np.max(np.abs(per_host.to_numpy() - hc.loc[per_host.index].to_numpy())))
    if not hc_diff <= TOL:
        raise AssertionError(f"per-host change differs from geometry_host_change.parquet by {hc_diff}")
    dbin = hcols.set_index("host_id").d_bin.loc[per_host.index]
    by_class = {k: {"mean_eV": float(per_host[dbin.to_numpy() == k].mean()), "n_hosts": int((dbin == k).sum())}
                for k in CLASSES}

    # geometry medians over the evaluated hosts; the c3b null-rule median (all 818) as a check
    ev = hcols[hcols.host_id.isin(set(ens.host_id))]
    rmsd_all_diff = abs(float(np.median(relax.internal_rmsd_A)) - ref["null_rule"]["median_internal_rmsd_A"])
    if not rmsd_all_diff <= TOL:
        raise AssertionError(f"median internal RMSD over 818 hosts differs from the c3b null rule by {rmsd_all_diff}")

    nonconv = sorted(relax.host_id[~relax.converged])
    formulas = sorted(str(tiling.formula.loc[h]) for h in nonconv)

    max_abs = max(abs(out_delta["654"][m][n]["mean"]) for m in MODELS for n in CONDS)
    return {
        "network": "default-initialisation network (backbone v1)",
        "design": "10 resamples x 3 seeds",
        "counts": {"n_hosts_relaxed": len(relax), "n_hosts_converged": int(relax.converged.sum()),
                   "n_hosts_evaluated": n_hosts_eval, "n_sites_evaluated": n_sites_eval},
        "nonconverged_formulas": ", ".join(formulas),
        "geometry": {"internal_rmsd_A": {"median": float(np.median(ev.internal_rmsd_A))},
                     "isotropic_strain": {"median": float(np.median(ev.eps_v))},
                     "deviatoric_strain": {"median": float(np.median(ev.deviatoric))}},
        "delta_mae": {"654": out_delta["654"], "200": out_delta["200"]},
        "per_host_abs_change_s_654": by_class,
        "max_abs_delta_mae_654": float(max_abs),
        "reproduction": {"reference": "results/geometry_c3b.json payload.budgets (mean, ci95)",
                         "tolerance": TOL,
                         "max_abs_diff_delta_mae": float(max(diffs.values())),
                         "max_abs_diff_by_cell": diffs,
                         "max_abs_diff_host_change_s_654_mlip": hc_diff,
                         "abs_diff_median_internal_rmsd_818_vs_null_rule": rmsd_all_diff,
                         "counts_equal": True},
    }


def main(argv: list[str] | None = None) -> None:
    from analysis import a1_common as c
    from dftgnn.config import load_config
    from dftgnn.io.results import write_result

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", type=Path, default=Path(f"{NAME}_dryrun.json"))
    a = ap.parse_args(argv)
    cfg = load_config()
    payload = c.to_jsonable(run())
    text = json.dumps(payload, indent=1, allow_nan=False)
    if a.dry_run:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text)
        print(a.out)
        return
    print(write_result(NAME, payload, config=cfg, repo_root=ROOT, results_dir=OUT_DIR))


if __name__ == "__main__":
    main()
