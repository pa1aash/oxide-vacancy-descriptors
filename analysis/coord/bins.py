"""Cation coordination of the removed oxygen, per vacancy site: the site-level bins of the coordination analysis.

Nothing is trained and no gain is computed here. Physically, removing a neutral O atom breaks its M-O bonds; the
number of cations bonded to the vacancy oxygen in the pristine host (its cation coordination number, CN) sets how
many bonds are broken and how many cation orbitals share the two electrons left behind, so CN is the first-order
local-environment variable of the vacancy formation energy. The bins defined here (fixed before any gain was
computed) let the descriptor gain be read as a function of that local environment.

Quantity and bins
    CN          number of non-O neighbours of the vacancy oxygen returned by pymatgen ``CrystalNN()`` (defaults)
                on the stored pristine unit cell (data/processed/unit_cells_v1.json); O neighbours are excluded
    bins        "2" (CN = 2), "3", "4", "5+" (CN >= 5); "unassigned" if the mapping or CrystalNN fails, or if
                CN < 2 (reason "cn<2")
    host mean   mean CN over the host's sites of the universe (all of them are tested together in a resample;
                checked on the headline test sets); descriptive only

Mapping (vacancy site -> unit-cell O atom)
    The universe row carries ``vacancy_atom_index`` i, the first equivalent atom of the release label in the
    pristine release supercell (cell_info.txt). The tiling map data/processed/tiling_map_v1.parquet (sha256 checked
    against results/tiling_v1.json; procedure src/dftgnn/mlip/tiling.py) gives the tiled index perm[i] = a |det M| + j
    of supercell atom i, so i is unit-cell atom a shifted by lattice point j. Checks per site: i is the first
    equivalent atom of the label; supercell atom i and unit-cell atom a are O; the tiled position of a (point
    operation, lattice point j, translation of the map) lies within the map tolerance of atom i; CrystalNN on the
    supercell at i gives the same cation CN (independent of the map). Checks per label: the unit-cell atoms of all
    equivalent supercell atoms of the label form exactly one spglib equivalence class of the unit cell (pymatgen
    SpacegroupAnalyzer defaults), different labels give different classes, and CN is the same on every unit-cell
    atom of the label. The release coordination of cell_info.txt (cations within its cutoff radius) is compared
    descriptively.

    PYTHONPATH=$PWD/src python analysis/coord/bins.py [--dry-run] [--data-root ~/Desktop/dft-gnn]
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis import a1_common as c

OUT_DIR = ROOT / "results" / "derived_coord"
DRY_OUT = Path("coord_bins_dryrun.json")
BINS = ("2", "3", "4", "5+")
HEADLINE_B = 654
N_R = 30
ANION_LIKE = ("H", "N", "F", "S", "Cl", "Se", "Br", "Te", "I")   # flagged neighbour elements (descriptive)
LABEL = ("cation coordination number of the vacancy oxygen in the pristine host (CrystalNN defaults on the unit "
         "cell of data/processed/unit_cells_v1.json), site-level bins 2, 3, 4, 5+; fixed before any gain")


def bin_of(cn: int | None) -> tuple[str, str]:
    """(bin, reason); reason is empty for an assigned site."""
    if cn is None:
        return "unassigned", "crystalnn_failed"
    if cn < 2:
        return "unassigned", "cn<2"
    return (str(cn) if cn <= 4 else "5+"), ""


def cation_cn(nn, structure, i: int) -> tuple[int, dict]:
    """Number of non-O CrystalNN neighbours of site i and their element counts."""
    el = Counter(n["site"].specie.symbol for n in nn.get_nn_info(structure, i))
    el.pop("O", None)
    return int(sum(el.values())), dict(sorted(el.items()))


def release_cn(coordination: str) -> int:
    """Cations within the release cutoff radius (cell_info.txt ``Coordination``: {'Zn': [1.98, ...], ...})."""
    d = ast.literal_eval(coordination)
    return int(sum(len(v) for k, v in d.items() if k != "O"))


def tiled_position(frac_a: np.ndarray, j: int, M: np.ndarray, point_op, translation: np.ndarray) -> np.ndarray:
    """Supercell fractional position of unit-cell atom a shifted by lattice point j (src/dftgnn/mlip/tiling.py)."""
    from dftgnn.mlip.tiling import lattice_points

    x = frac_a if point_op is None else (frac_a @ np.asarray(point_op)) % 1.0
    f = (x + lattice_points(M)[j]) @ np.linalg.inv(M)
    return f - np.floor(f + 1e-12) + translation


def host_rows(sites: pd.DataFrame, uc, sc, cell_info: str, tm: dict, nn) -> tuple[list[dict], dict]:
    """Per-site rows and per-host checks of one host."""
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    from dftgnn.data.universe import parse_cell_info

    info = parse_cell_info(cell_info)
    M = np.array(json.loads(tm["M"]))
    nd = round(abs(np.linalg.det(M)))
    perm = np.array(json.loads(tm["perm"]))
    pop = json.loads(tm["point_op"]) if isinstance(tm["point_op"], str) and tm["point_op"] else None
    t = np.array(json.loads(tm["translation"]))
    sga = SpacegroupAnalyzer(uc)
    eq = np.asarray(sga.get_symmetry_dataset().equivalent_atoms)
    sg_rel = re.search(r"Space group: (\S+)", cell_info)
    host = {"n_uc_ok": len(uc) == tm["n_uc"], "n_sc_ok": len(sc) == tm["n_sc"] == len(perm),
            "space_group": sga.get_space_group_symbol(), "space_group_release": sg_rel.group(1) if sg_rel else None}
    cn_uc: dict[int, tuple[int | None, dict]] = {}

    def uc_cn(a: int):
        if a not in cn_uc:
            try:
                cn_uc[a] = cation_cn(nn, uc, a)
            except Exception as e:      # noqa: BLE001 (any CrystalNN failure leaves the site unassigned)
                cn_uc[a] = (None, {"error": f"{type(e).__name__}: {e}"})
        return cn_uc[a]

    rows, label_class = [], {}
    for s in sites.itertuples():
        i, lab = int(s.vacancy_atom_index), s.site_label
        row = {"site_id": s.site_id, "host_id": s.host_id, "formula": s.formula, "site_label": lab,
               "vacancy_atom_index": i}
        fail = []
        if lab not in info:
            fail.append("label_not_in_cell_info")
        elif info[lab]["equiv"][0] != i:
            fail.append("index_not_first_equivalent")
        if not 0 <= i < len(sc) or sc[i].specie.symbol != "O":
            fail.append("supercell_atom_not_O")
        a, j = (int(perm[i]) // nd, int(perm[i]) % nd) if 0 <= i < len(perm) else (-1, -1)
        if not 0 <= a < len(uc) or uc[a].specie.symbol != "O":
            fail.append("unit_cell_atom_not_O")
        resid = None
        if not fail:
            d = sc.frac_coords[i] - tiled_position(uc.frac_coords[a], j, M, pop, t)
            resid = float(np.linalg.norm((d - np.round(d)) @ sc.lattice.matrix))
            if resid > float(tm["max_residual_A"]) + 1e-9:
                fail.append("tiled_position_off")
        row.update(unit_cell_index=a, map_residual_A=resid)
        if fail:
            row.update(cn=None, bin="unassigned", reason="mapping_failed: " + ",".join(fail), neighbours={})
            rows.append(row)
            continue
        cn, el = uc_cn(a)
        b, why = bin_of(cn)
        row.update(cn=cn, bin=b, reason=why, neighbours=el)
        try:
            row["cn_supercell"] = cation_cn(nn, sc, i)[0]
        except Exception as e:          # noqa: BLE001
            row["cn_supercell"] = None
            row["cn_supercell_error"] = f"{type(e).__name__}: {e}"
        # symmetry: unit-cell atoms of every equivalent supercell atom of the label
        ua = sorted({int(perm[k]) // nd for k in info[lab]["equiv"]})
        cls = sorted({int(eq[k]) for k in ua})
        full = len(cls) == 1 and set(np.nonzero(eq == cls[0])[0].tolist()) == set(ua)
        cns = sorted({uc_cn(k)[0] for k in ua}, key=lambda v: -1 if v is None else v)
        label_class[lab] = cls
        row.update(label_one_spglib_class=bool(full), label_unit_cell_atoms=len(ua), label_cn_values=cns,
                   label_cn_constant=len(cns) == 1, cn_release=release_cn(info[lab]["coordination"]),
                   release_cutoff_A=info[lab]["cutoff_A"])
        rows.append(row)
    flat = [k for v in label_class.values() for k in v]
    host["labels_distinct_classes"] = len(flat) == len(set(flat))
    return rows, host


def headline_tests() -> tuple[pd.DataFrame, dict]:
    """Test sites per resample of the 654-host headline design (S, seed 0, v2; the cells shared by S, D, D*)."""
    prec, f = c.load_table("precision_eval_long")
    p = prec[(prec.backbone == "v2") & (prec.model == "S") & (prec.seed == 0) & (prec.B == HEADLINE_B)]
    if sorted(p.r.unique()) != list(range(N_R)):
        raise AssertionError("headline resamples incomplete")
    return p[["r", "site_id", "host_id"]].reset_index(drop=True), f


def summarise(sites: pd.DataFrame, tests: pd.DataFrame, uni: pd.DataFrame) -> dict:
    n_uni = uni.groupby("host_id").size()
    per = tests.groupby(["r", "host_id"]).size()
    whole = bool((per.to_numpy() == n_uni.loc[per.index.get_level_values("host_id")].to_numpy()).all())
    t = tests.merge(sites[["site_id", "bin"]], on="site_id", how="left", validate="many_to_one")
    if t.bin.isna().any():
        raise AssertionError("tested site without a coordination row")
    multi = set(n_uni.index[n_uni >= 2])
    tested_any = set(tests.site_id)
    out = {}
    for b in (*BINS, "unassigned"):
        s = sites[sites.bin == b]
        tb = t[t.bin == b]
        per_r = tb.groupby("r").agg(n_sites=("site_id", "size"), n_hosts=("host_id", "nunique"))
        per_r = per_r.reindex(range(N_R), fill_value=0)
        tb2 = tb[tb.host_id.isin(multi)]
        per_r2 = tb2.groupby("r").agg(n_sites=("site_id", "size"), n_hosts=("host_id", "nunique"))
        per_r2 = per_r2.reindex(range(N_R), fill_value=0)
        out[b] = {"universe": {"n_sites": len(s), "n_hosts": int(s.host_id.nunique()),
                               "n_sites_in_hosts_ge2": int(s.host_id.isin(multi).sum()),
                               "n_hosts_ge2_with_site_in_bin": int(s.host_id[s.host_id.isin(multi)].nunique())},
                  "headline_654_30": {
                      "n_distinct_sites_tested": int(s.site_id.isin(tested_any).sum()),
                      "n_distinct_hosts_tested": int(s.host_id[s.site_id.isin(tested_any)].nunique()),
                      "per_resample_sites": {"mean": float(per_r.n_sites.mean()), "min": int(per_r.n_sites.min()),
                                             "max": int(per_r.n_sites.max())},
                      "per_resample_hosts": {"mean": float(per_r.n_hosts.mean()), "min": int(per_r.n_hosts.min()),
                                             "max": int(per_r.n_hosts.max())},
                      "per_resample_sites_in_hosts_ge2": {"mean": float(per_r2.n_sites.mean()),
                                                          "min": int(per_r2.n_sites.min()),
                                                          "max": int(per_r2.n_sites.max())},
                      "per_resample_hosts_ge2": {"mean": float(per_r2.n_hosts.mean()),
                                                 "min": int(per_r2.n_hosts.min()),
                                                 "max": int(per_r2.n_hosts.max())}}}
    return {"bins": out, "headline_hosts_tested_whole": whole,
            "headline_test_sites_per_resample": int(tests.groupby("r").size().iloc[0]),
            "headline_test_hosts_per_resample": sorted(set(tests.groupby("r").host_id.nunique().tolist()))}


def checks(sites: pd.DataFrame, hosts: dict) -> dict:
    ok = sites[~sites.reason.str.startswith("mapping_failed")]
    has = ok[ok.cn.notna()]
    agree_sc = has.cn_supercell.notna() & (has.cn_supercell == has.cn)
    rel = has.dropna(subset=["cn_release"])
    cross = pd.crosstab(rel.cn.astype(int), rel.cn_release.astype(int))
    hv = list(hosts.values())
    return {
        "n_sites": len(sites), "n_mapped": len(ok), "n_mapping_failed": int(len(sites) - len(ok)),
        "max_map_residual_A": float(ok.map_residual_A.max()),
        "n_cn_supercell_equals_unit_cell": int(agree_sc.sum()), "n_with_cn": len(has),
        "sites_cn_supercell_differs": has.site_id[~agree_sc].tolist(),
        "n_label_one_spglib_class": int(ok.label_one_spglib_class.fillna(False).astype(bool).sum()),
        "sites_label_not_one_class": ok.site_id[~ok.label_one_spglib_class.fillna(False).astype(bool)].tolist(),
        "n_label_cn_constant": int(ok.label_cn_constant.fillna(False).astype(bool).sum()),
        "sites_label_cn_not_constant": ok.site_id[~ok.label_cn_constant.fillna(False).astype(bool)].tolist(),
        "n_hosts": len(hv),
        "n_hosts_labels_distinct_classes": int(sum(h["labels_distinct_classes"] for h in hv)),
        "hosts_labels_share_class": sorted(k for k, h in hosts.items() if not h["labels_distinct_classes"]),
        "n_hosts_space_group_matches_release": int(sum(h["space_group"] == h["space_group_release"] for h in hv)),
        "hosts_space_group_differs": {k: [h["space_group"], h["space_group_release"]]
                                      for k, h in hosts.items() if h["space_group"] != h["space_group_release"]},
        "n_hosts_atom_counts_match_map": int(sum(h["n_uc_ok"] and h["n_sc_ok"] for h in hv)),
        "release_coordination": {
            "rule": "cations within the cutoff radius of cell_info.txt (descriptive comparison only)",
            "n_compared": len(rel), "n_equal": int((rel.cn == rel.cn_release).sum()),
            "n_crystalnn_lower": int((rel.cn < rel.cn_release).sum()),
            "n_crystalnn_higher": int((rel.cn > rel.cn_release).sum()),
            "crosstab_crystalnn_rows_release_columns": {str(i): {str(j): int(cross.loc[i, j]) for j in cross.columns
                                                                 if cross.loc[i, j]} for i in cross.index}}}


def combined_sha(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(f"{p.parent.name}/{p.name}:{c.sha256(p)}\n".encode())
    return h.hexdigest()


def run(data_root: Path) -> dict:
    import pymatgen.core
    import spglib
    from pymatgen.analysis.local_env import CrystalNN
    from pymatgen.core import Structure

    from dftgnn.mlip.release import read_supercell

    proc = data_root / "data" / "processed"
    uni = pd.read_parquet(proc / "universe_v1.parquet")
    cells = json.loads((proc / "unit_cells_v1.json").read_text())
    map_path = proc / "tiling_map_v1.parquet"
    map_rec = c.record("tiling_v1")["payload"]["map"]
    if c.sha256(map_path) != map_rec["sha256"]:
        raise AssertionError("tiling map sha256 differs from results/tiling_v1.json")
    tmap = pd.read_parquet(map_path).set_index("host_id")
    if (tmap.status != "ok").any():
        raise AssertionError("tiling map has failed hosts")
    nn = CrystalNN()
    rows, hosts, cifs, infos = [], {}, [], []
    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for k, (hid, g) in enumerate(uni.sort_values("site_id").groupby("host_id", sort=True)):
            cif = data_root / g.supercell_cif_path.iloc[0]
            ci = cif.parent / "cell_info.txt"
            cifs.append(cif)
            infos.append(ci)
            r, h = host_rows(g, Structure.from_dict(cells[hid]), read_supercell(str(cif), data_root),
                             ci.read_text(), tmap.loc[hid].to_dict(), nn)
            rows += r
            hosts[hid] = h
            if k % 100 == 0:
                print(f"{k}/{uni.host_id.nunique()} hosts, {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
    warn = Counter(str(w.message).split("\n")[0][:120] for w in caught)
    sites = pd.DataFrame(rows).sort_values("site_id").reset_index(drop=True)
    has = sites[sites.cn.notna()]
    hm = has.groupby("host_id").cn.mean()
    sites["host_mean_cn"] = sites.host_id.map(hm)
    tests, f_prec = headline_tests()
    nb = Counter()
    for el in has.neighbours:
        nb.update(el)
    anion = has[has.neighbours.map(lambda e: any(x in e for x in ANION_LIKE))]
    cols = ["site_id", "host_id", "site_label", "vacancy_atom_index", "unit_cell_index", "cn", "bin", "reason",
            "host_mean_cn", "neighbours", "cn_supercell", "cn_release", "map_residual_A"]
    table = [{k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items()}
             for r in sites[cols].to_dict("records")]
    for r in table:
        if r["cn"] is not None:
            r["cn"] = int(r["cn"])
        for k in ("cn_supercell", "cn_release"):
            if r[k] is not None:
                r[k] = int(r[k])
    un = sites[sites.bin == "unassigned"]
    return {
        "label": LABEL,
        "definition": {"cn": "non-O neighbours of the vacancy oxygen, pymatgen CrystalNN() defaults, pristine unit "
                             "cell (data/processed/unit_cells_v1.json) at the mapped atom",
                       "bins": {"2": "CN = 2", "3": "CN = 3", "4": "CN = 4", "5+": "CN >= 5",
                                "unassigned": "mapping or CrystalNN failure, or CN < 2 (reason cn<2)"},
                       "host_mean_cn": "mean CN over the host's universe sites with a CN (descriptive)",
                       "crystalnn_parameters": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else
                                                    str(v)) for k, v in vars(nn).items()
                                                if not k.startswith("_")}},
        "cn_distribution": {str(int(k)): int(v) for k, v in sorted(Counter(has.cn.astype(int)).items())},
        "counts": summarise(sites, tests, uni),
        "unassigned": {"n": len(un), "reasons": dict(Counter(un.reason)),
                       "sites": un[["site_id", "host_id", "cn", "reason"]].to_dict("records")},
        "host_mean_cn": {"n_hosts": int(hm.size), "mean": float(hm.mean()), "min": float(hm.min()),
                         "max": float(hm.max()),
                         "quantiles_10_50_90": [float(q) for q in np.quantile(hm, [0.1, 0.5, 0.9])]},
        "neighbour_elements": {"counts": dict(sorted(nb.items())), "anion_like_flagged": list(ANION_LIKE),
                               "n_sites_with_anion_like_neighbour": len(anion),
                               "sites_with_anion_like_neighbour": anion.site_id.tolist()},
        "verification": checks(sites, hosts),
        "warnings": {"n": int(sum(warn.values())), "messages": dict(warn.most_common(10))},
        "sites": table,
        "runtime_s": round(time.time() - t0, 1),
        "versions": {"pymatgen": pymatgen.core.__version__, "spglib": spglib.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "inputs": {"universe": {"path": "data/processed/universe_v1.parquet",
                                "sha256": c.sha256(proc / "universe_v1.parquet")},
                   "unit_cells": {"path": "data/processed/unit_cells_v1.json",
                                  "sha256": c.sha256(proc / "unit_cells_v1.json")},
                   "tiling_map": {"path": "data/processed/tiling_map_v1.parquet", "sha256": map_rec["sha256"],
                                  "record": "results/tiling_v1.json"},
                   "supercell_cifs": {"n": len(cifs), "combined_sha256": combined_sha(cifs),
                                      "rule": "sha256 over lines '<formula>/supercell.cif:<sha256>' in host_id order"},
                   "cell_info": {"n": len(infos), "combined_sha256": combined_sha(infos),
                                 "rule": "sha256 over lines '<formula>/cell_info.txt:<sha256>' in host_id order"},
                   "precision_eval_long": f_prec},
    }


def main() -> None:
    from dftgnn.config import load_config
    from dftgnn.io.results import write_result

    cfg = load_config()
    data_root = (Path(sys.argv[sys.argv.index("--data-root") + 1]).expanduser().resolve()
                 if "--data-root" in sys.argv else ROOT)
    payload = c.to_jsonable(run(data_root))
    if "--dry-run" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else DRY_OUT
        out.write_text(json.dumps(payload, indent=1, allow_nan=False))
        print(out)
        return
    print(write_result("coord_bins", payload, config=cfg, repo_root=ROOT, results_dir=OUT_DIR))


if __name__ == "__main__":
    main()
