"""Screening demonstration data for Figure 7 and the Results subsection: per-site out-of-sample predictions with
their spread over the outer resamples, both backbones, and the local structure of each screened oxygen site.

``results/screening_c5.json``, ``results/screening_absence.json`` (why a candidate is absent), the run records
and prediction parquets of S at the screening budget (v1 in ``results/``, primary backbone v2 in
``results/v2_sweep/``), and the universe table (``--data-root``, default the main checkout that holds
``data/``).

Per site: the DFT target, and for each backbone the seed-ensemble-mean prediction of every outer resample in
which the host was a test host, their mean, their minimum and maximum (the spread drawn in Figure 7) and the
number of such resamples. Structure: the first cation shell of the vacancy oxygen in the host supercell
(cations within 1.2 times the shortest cation-oxygen distance, the shell of the structure renders), its size
and mean distance. The screening MAE of each backbone has a host-cluster percentile bootstrap interval
(``stats.bootstrap_n`` draws, seed 0): the screened sites of one oxide share a host, so hosts are the cluster.

    python figures/code/screening_data.py [--data-root DIR] [--no-write]

Writes ``figures/main/data/screening_demo.json`` through ``write_result``.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import diagnostics_sweep as D  # noqa: E402  (read only: run_index, ensemble)

from dftgnn.config import load_config  # noqa: E402

OUT_DIR = ROOT / "figures" / "main" / "data"
RECORD = "screening_demo"
SHELL_FACTOR = 1.2          # first cation shell: within 1.2 x the shortest distance (blender/README.md)
BACKBONES = {"v2": "v2_sweep", "v1": ""}
DESC = {"band_gap_eV": "desc_band_gap", "host_formation_energy_eV_per_atom": "desc_formation_energy",
        "bader_charge_e": "desc_bader_charge", "bader_volume_A3": "desc_bader_volume",
        "o2p_center_below_vbm_eV": "desc_o2p_center_from_vbm", "nn_mean_electronegativity": "desc_nn_ave_eleneg",
        "nn_mean_bader_charge_e": "desc_nn_ave_bader_charge"}


def per_resample(subdir: str, budget: int) -> pd.DataFrame:
    """Seed-ensemble-mean prediction of S for every (site, outer resample) at the budget."""
    ix = D.run_index(subdir)
    ix = ix[ix.split.str.startswith("outer") & (ix.model == "S") & (ix.B == budget)]
    return pd.concat([D.ensemble(g).assign(r=r) for r, g in ix.groupby("r")], ignore_index=True)


def first_shell(cif: Path, vac_index: int) -> tuple[int, float]:
    """Number and mean distance (A) of the cations in the first shell of the oxygen atom ``vac_index``."""
    from ase.io import read

    at = read(cif)
    cat = [i for i, s in enumerate(at.get_chemical_symbols()) if s != "O"]
    d = at.get_distances(vac_index, cat, mic=True)
    shell = d[d <= SHELL_FACTOR * d.min()]
    return len(shell), float(shell.mean())


def space_group(cif: Path, symprec: float = 0.1) -> tuple[str, int]:
    """International symbol and number of the host supercell's space group (pymatgen, ``symprec`` in A)."""
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    sga = SpacegroupAnalyzer(Structure.from_file(cif), symprec=symprec)
    return sga.get_space_group_symbol(), int(sga.get_space_group_number())


def cluster_ci(err: np.ndarray, hosts: np.ndarray, n_boot: int, seed: int = 0) -> tuple[float, float]:
    """Percentile 95% interval of the site MAE over host-cluster bootstrap draws."""
    uniq = np.unique(hosts)
    rows = {h: err[hosts == h] for h in uniq}
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        vals[b] = np.abs(np.concatenate([rows[h] for h in pick])).mean()
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, default=Path.home() / "Desktop" / "dft-gnn")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    sc = cfg.screening
    c5 = json.loads((ROOT / "results" / "screening_c5.json").read_text())["payload"]
    absence = json.loads((ROOT / "results" / "screening_absence.json").read_text())["payload"]
    uni = pd.read_parquet(a.data_root / "data" / "processed" / "universe_v1.parquet")
    uni = uni[uni.reduced_formula.isin(sc.candidates)].set_index("site_id")

    pr = {k: per_resample(sub, sc.budget) for k, sub in BACKBONES.items()}
    sites = {}
    for sid in c5["sites"]:
        s = sid["site_id"]
        u = uni.loc[s]
        rec = {"dft_eV": float(u.target_Ef_eV), "n_atoms": int(u.n_atoms), "spin_polarised": bool(u.spin_polarised)}
        for k, df in pr.items():
            g = df[df.site_id == s]
            assert abs(g.y_true.iloc[0] - rec["dft_eV"]) < 1e-9, s
            rec[k] = {"pred_eV": float(g.y_pred.mean()), "min_eV": float(g.y_pred.min()),
                      "max_eV": float(g.y_pred.max()), "n_resamples": int(g.r.nunique())}
            rec[k]["signed_err_eV"] = rec[k]["pred_eV"] - rec["dft_eV"]
            rec[k]["abs_err_eV"] = abs(rec[k]["signed_err_eV"])
        assert abs(rec["v1"]["pred_eV"] - sid["S_eV"]) < 1e-9 and abs(rec["v2"]["pred_eV"] - sid["S_v2_eV"]) < 1e-9, s
        assert rec["v1"]["n_resamples"] == sid["n_resamples"], s
        cif = a.data_root / u.supercell_cif_path
        rec["first_shell_cations"], rec["first_shell_mean_distance_A"] = first_shell(cif, int(u.vacancy_atom_index))
        rec["_space_group"] = space_group(cif)
        rec.update({k: float(u[c]) for k, c in DESC.items()})
        rec["_candidate"], rec["_host_id"], rec["_label"] = sid["candidate"], sid["host_id"], u.site_label
        sites[s] = rec

    cand_hosts = {r["_candidate"]: [] for r in sites.values()}
    for s, r in sites.items():
        cand_hosts[r["_candidate"]].append(s)

    summary = {}
    host = np.array([r["_host_id"] for r in sites.values()])
    dft = np.array([r["dft_eV"] for r in sites.values()])
    for k in BACKBONES:
        pred = np.array([r[k]["pred_eV"] for r in sites.values()])
        err = pred - dft
        lo, hi = cluster_ci(err, host, cfg.stats.bootstrap_n)
        worst = max(sites, key=lambda s: sites[s][k]["abs_err_eV"])
        pairs = [(i, j) for sl in cand_hosts.values() for i, j in combinations(sl, 2)]
        conc = sum(np.sign(sites[i][k]["pred_eV"] - sites[j][k]["pred_eV"]) == np.sign(sites[i]["dft_eV"] - sites[j]["dft_eV"])
                   for i, j in pairs)
        rho = float(pd.Series(pred).corr(pd.Series(dft), method="spearman"))
        summary[k] = {"mae_eV": float(np.abs(err).mean()), "mae_ci": [lo, hi], "bias_eV": float(err.mean()),
                      "max_abs_err_eV": float(np.abs(err).max()), "worst_site": worst,
                      "n_over_prediction": int((err > 0).sum()), "n_under_prediction": int((err < 0).sum()),
                      "within_host_pairs": len(pairs), "within_host_pairs_concordant": int(conc),
                      "spearman_across_sites": rho}
    assert abs(summary["v1"]["mae_eV"] - c5["mae_S"]) < 1e-9 and abs(summary["v2"]["mae_eV"] - c5["mae_S_v2"]) < 1e-9

    # descriptive association of the DFT target with host quantities over the screened sites (no inference)
    descriptive = {f"spearman_dft_vs_{name}": float(pd.Series(dft).corr(
        pd.Series([r[key] for r in sites.values()]), method="spearman"))
        for name, key in (("host_formation_energy", "host_formation_energy_eV_per_atom"), ("band_gap", "band_gap_eV"),
                          ("first_shell_cations", "first_shell_cations"),
                          ("first_shell_distance", "first_shell_mean_distance_A"))}

    # sites of one host in pairs (label order): the difference first minus second, DFT and both backbones
    pairs = {f"{r_i['_candidate']}_{r_i['_label']}_{r_j['_label']}": {
        "dft_diff_eV": r_i["dft_eV"] - r_j["dft_eV"],
        **{f"{k}_diff_eV": r_i[k]["pred_eV"] - r_j[k]["pred_eV"] for k in BACKBONES}}
        for sl in cand_hosts.values() for i, j in combinations(sorted(sl, key=lambda s: sites[s]["_label"]), 2)
        for r_i, r_j in [(sites[i], sites[j])]}

    ab = {c["candidate"]: c for c in absence["candidates"]}
    missing = [c for c in sc.candidates if c not in cand_hosts]
    not_in_release = [c for c in missing if ab[c]["reason"] == "not in the release"]
    phs_removed = [c for c in missing if ab[c]["reason"].startswith("PHS")]
    assert set(missing) == set(c5["missing"]) and len(not_in_release) + len(phs_removed) == len(missing)
    scope = {"n_candidates": len(sc.candidates), "n_present": len(cand_hosts), "n_absent": len(missing),
             "n_not_in_release": len(not_in_release), "n_phs_removed": len(phs_removed),
             "n_screened_sites": len(sites), "n_hosts": len(set(host)),
             "n_sites_one_resample": sum(r["v1"]["n_resamples"] == 1 for r in sites.values()),
             "release_neutral_entries": absence["release_neutral_completed"]["entries"],
             "release_neutral_phs": absence["release_neutral_completed"]["phs"],
             "phs_removed": {c: {"neutral_entries": len(ab[c]["neutral_entries"]),
                                 "phs_entries": sum(e["phs"] for e in ab[c]["neutral_entries"])} for c in phs_removed},
             "tio2_release": {"neutral_entries": len(ab["TiO2"]["neutral_entries"]),
                              "phs_entries": sum(e["phs"] for e in ab["TiO2"]["neutral_entries"]),
                              "in_universe_sites": ab["TiO2"]["universe_v1_sites"]}}
    payload = {"budget": sc.budget, "scope": scope, "summary": summary, "descriptive": descriptive, "pairs": pairs,
               "sites": {s: {k: v for k, v in r.items() if not k.startswith("_")} for s, r in sites.items()},
               "site_index": {s: {"candidate": r["_candidate"], "host_id": r["_host_id"], "label": r["_label"]}
                              for s, r in sites.items()},
               "hosts": {r["_candidate"]: {"space_group": r["_space_group"][0], "space_group_number": r["_space_group"][1]}
                         for r in sites.values()},
               "absent": {"not_in_release": not_in_release, "phs_removed": phs_removed}}
    pd.set_option("display.width", 220)
    print(json.dumps({"scope": scope, "summary": summary, "descriptive": descriptive}, indent=1))
    print(pd.DataFrame({s: {"dft": r["dft_eV"], "v2": r["v2"]["pred_eV"], "v2_rng": (r["v2"]["min_eV"], r["v2"]["max_eV"]),
                            "v1": r["v1"]["pred_eV"], "v1_rng": (r["v1"]["min_eV"], r["v1"]["max_eV"]),
                            "n": r["v1"]["n_resamples"], "cn": r["first_shell_cations"],
                            "d": round(r["first_shell_mean_distance_A"], 3)} for s, r in sites.items()}).T.round(3))
    if not a.no_write:
        from dftgnn.io.results import write_result

        print(write_result(RECORD, payload, config=cfg, results_dir=OUT_DIR))


if __name__ == "__main__":
    main()
