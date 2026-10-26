"""Long evaluation table of the geometry analysis C3b.

Analysis rules:, row of 2026-10-14. Pure reading of committed run records (`results/geomeval/`,
`results/mlip_v1/`) and their prediction parquets, plus the gitignored universe, release supercells and MLIP
supercells (read only, from ``--data-root``). Writes

    results/analysis/geometry_eval_long.parquet   model, r, B, seed, run_id, gpu, condition, site_id, host_id,
                                                  y_true, y_pred (one row per run, condition and test site)
    results/analysis/geometry_hosts.parquet       one row per relaxed host: host_id, formula, converged, steps,
                                                  eps_v, deviatoric, internal_rmsd_A, d_bin, relax_gpu, point_op,
                                                  excluded_from_C3b
    results/geometry_eval_table.json              hashes, counts and the integrity checks (via write_result)

Integrity assertions (any failure raises): 180 geomeval records whose run ids, keys and specs equal
`specs/geomeval.jsonl`; prediction sha256 equals the record; each run covers the evaluated test sites of its split
exactly once under each of the three conditions; predictions finite; targets and hosts equal the universe; hosts
flagged ``excluded_from_C3b`` absent; the vacancy atom of every site is O in the release supercell and in the MLIP
supercell; the stored d_bin, converged and component columns equal `data/host_dbins_v1.csv` and `results/mlip_v1/`;
818 relax records, one per universe host, whose MLIP artefact sha256 values match the files read.

    PYTHONPATH=$PWD/src python analysis/geometry_eval_table.py --data-root ~/Desktop/dft-gnn
"""
from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("S", "D", "P")
BUDGETS = (654, 200)
N_RESAMPLES = 10
SEEDS = (0, 1, 2)
CONDITIONS = ("dft", "mlip", "mlip_rescaled")
N_RUNS = len(MODELS) * len(BUDGETS) * N_RESAMPLES * len(SEEDS)      # 180
N_HOSTS = 818
COMPONENTS = ("eps_v", "deviatoric", "internal_rmsd_A")
GPU_SHORT = {"NVIDIA A10": "A10", "NVIDIA TITAN RTX": "TITAN RTX", "NVIDIA GeForce RTX 4090": "RTX 4090"}
EXPECTED_COMPONENTS = {"S": {"S"}, "D": {"D"}, "P": {"P1", "D"}}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gpu_short(name: str) -> str:
    return GPU_SHORT.get(name, name)


# ---------------------------------------------------------------- records

def read_payloads(directory: Path) -> list[dict]:
    return [json.loads(f.read_text())["payload"] for f in sorted(directory.glob("*.json"))]


def check_specs(recs: Sequence[Mapping], spec_lines: Sequence[Mapping]) -> int:
    """Run ids, keys and specs of the records equal the spec file, one record per spec line."""
    want = {s["run_id"]: s for s in spec_lines}
    got = {p["run_id"]: p for p in recs}
    if len(got) != len(recs):
        raise AssertionError("duplicate geomeval run ids")
    if set(got) != set(want):
        raise AssertionError(f"run ids differ from the spec file: missing {sorted(set(want) - set(got))[:4]}, "
                             f"extra {sorted(set(got) - set(want))[:4]}")
    for rid, p in got.items():
        if p["spec"] != want[rid]["spec"] or p["key"] != want[rid]["key"]:
            raise AssertionError(f"{rid}: spec or key differs from specs/geomeval.jsonl")
    return len(got)


def check_cells(recs: Sequence[Mapping]) -> dict:
    """One run per (model, r, B, seed) over the 180 cells; components as the model requires."""
    cells = {}
    for p in recs:
        k = p["key"]
        cell = (k["model"], int(k["r"]), int(k["budget"]), int(k["seed"]))
        if cell in cells:
            raise AssertionError(f"two runs for cell {cell}")
        if set(k["components"]) != EXPECTED_COMPONENTS[k["model"]]:
            raise AssertionError(f"{p['run_id']}: components {sorted(k['components'])}")
        cells[cell] = p["run_id"]
    want = {(m, r, B, s) for m in MODELS for r in range(N_RESAMPLES) for B in BUDGETS for s in SEEDS}
    if set(cells) != want:
        raise AssertionError(f"cells missing {sorted(want - set(cells))[:4]} extra {sorted(set(cells) - want)[:4]}")
    return {f"{m}|{B}": sum(1 for c in cells if c[0] == m and c[2] == B) for m in MODELS for B in BUDGETS}


# ---------------------------------------------------------------- integrity of one run

def check_run(pred: pd.DataFrame, test_sites: set[str], excluded_hosts: set[str], label: str) -> None:
    """Each evaluated test site exactly once under each condition; finite; no excluded host."""
    if set(pred.condition) != set(CONDITIONS):
        raise AssertionError(f"{label}: conditions {sorted(set(pred.condition))}")
    if not np.isfinite(pred.prediction.to_numpy(float)).all():
        raise AssertionError(f"{label}: non-finite predictions")
    if excluded_hosts & set(pred.host_id):
        raise AssertionError(f"{label}: excluded host present")
    want = {s for s in test_sites}
    for c, g in pred.groupby("condition"):
        if g.site_id.duplicated().any():
            raise AssertionError(f"{label}: duplicate site under {c}")
        if set(g.site_id) != want:
            raise AssertionError(f"{label}: sites under {c} differ from the evaluated test sites")


def check_targets(pred: pd.DataFrame, uni: pd.DataFrame) -> None:
    u = uni.set_index("site_id")
    if not np.allclose(pred.target.to_numpy(), u.target_Ef_eV.loc[pred.site_id].to_numpy(), atol=1e-9):
        raise AssertionError("targets differ from the universe")
    if not (pred.host_id.to_numpy() == u.host_id.loc[pred.site_id].to_numpy()).all():
        raise AssertionError("hosts differ from the universe")


def check_host_columns(pred: pd.DataFrame, hosts: pd.DataFrame, label: str) -> None:
    """d_bin, converged and the three components of the prediction file equal the host table."""
    h = hosts.set_index("host_id").loc[pred.host_id]
    if not (pred.d_bin.to_numpy() == h.d_bin.to_numpy()).all():
        raise AssertionError(f"{label}: d_bin differs from data/host_dbins_v1.csv")
    if not (pred.converged.to_numpy() == h.converged.to_numpy()).all():
        raise AssertionError(f"{label}: converged differs from results/mlip_v1")
    for c in COMPONENTS:
        if not np.array_equal(pred[c].to_numpy(float), h[c].to_numpy(float)):
            raise AssertionError(f"{label}: {c} differs from results/mlip_v1")


# ---------------------------------------------------------------- hosts

def host_table(relax: Sequence[Mapping], dbins: pd.DataFrame, tiling: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for p in relax:
        g = p["geometry"]
        rows.append({"host_id": p["host_id"], "run_id": p["run_id"], "converged": bool(p["converged"]),
                     "steps": int(p["steps"]), **{c: float(g[c]) for c in COMPONENTS},
                     "relax_gpu": gpu_short(p["hardware"]["gpu_model"]),
                     "excluded_relax": bool(p["excluded_from_C3b"])})
    h = pd.DataFrame(rows)
    if h.host_id.duplicated().any():
        raise AssertionError("two relax records for one host")
    t = tiling.set_index("host_id")
    d = dbins.set_index("host_id")
    h["formula"] = t.formula.loc[h.host_id].to_numpy()
    h["d_bin"] = d["bin"].loc[h.host_id].fillna("unassigned").to_numpy()
    h["point_op"] = t.point_op.loc[h.host_id].fillna("").astype(str).to_numpy()
    h["excluded_from_C3b"] = t.excluded_from_C3b.loc[h.host_id].astype(bool).to_numpy()
    if (h.excluded_from_C3b != h.excluded_relax).any():
        raise AssertionError("excluded_from_C3b differs between the tiling summary and the relax records")
    return h.drop(columns="excluded_relax").sort_values("host_id").reset_index(drop=True)


def tiling_split(tiling: pd.DataFrame) -> dict:
    op = tiling.point_op.fillna("").astype(str).str.strip()
    return {"n_hosts": len(tiling), "translation_only": int((op == "").sum()),
            "point_operation": int((op != "").sum()),
            "status_ok": int((tiling.status == "ok").sum()),
            "excluded_from_C3b": int(tiling.excluded_from_C3b.astype(bool).sum())}


def check_vacancy_species(sites: Iterable[tuple[str, int]], species: Mapping[str, Sequence[str]], label: str) -> int:
    """Every (host, vacancy atom index) is an O atom in ``species[host]``."""
    n = 0
    for h, v in sites:
        if species[h][int(v)] != "O":
            raise AssertionError(f"{label}: vacancy atom {v} of {h} is {species[h][int(v)]}, not O")
        n += 1
    return n


def check_artifacts(relax: Sequence[Mapping], mlip_dir: Path, names: Sequence[str] = ("_uc.json", "_sc.json")) -> int:
    """The sha256 of the MLIP files read here equals the relax record's artefact list."""
    n = 0
    for p in relax:
        arts = {Path(a["path"]).name: a["sha256"] for a in p["artifacts"]}
        for suffix in names:
            name = f"{p['host_id']}{suffix}"
            if sha256(mlip_dir / name) != arts[name]:
                raise AssertionError(f"{name}: sha256 differs from the relax record")
            n += 1
    return n


# ---------------------------------------------------------------- long table

def long_rows(pred: pd.DataFrame, p: Mapping) -> pd.DataFrame:
    k = p["key"]
    return pd.DataFrame({"model": k["model"], "r": int(k["r"]), "B": int(k["budget"]), "seed": int(k["seed"]),
                         "run_id": p["run_id"], "gpu": gpu_short(p["hardware"]["gpu_model"]),
                         "condition": pred.condition.to_numpy(), "site_id": pred.site_id.to_numpy(),
                         "host_id": pred.host_id.to_numpy(), "y_true": pred.target.to_numpy(float),
                         "y_pred": pred.prediction.to_numpy(float)})


def main() -> None:
    from dftgnn.config import load_config
    from dftgnn.io.results import write_result
    from dftgnn.split import load_split

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, default=ROOT,
                    help="checkout holding the gitignored data/processed and data/raw inputs (read only)")
    a = ap.parse_args()
    dr = a.data_root.expanduser().resolve()
    uni_path = dr / "data" / "processed" / "universe_v1.parquet"
    mlip_dir = dr / "data" / "processed" / "mlip_v1"
    uni = pd.read_parquet(uni_path)
    dbins = pd.read_csv(ROOT / "data" / "host_dbins_v1.csv")
    tiling = pd.read_csv(ROOT / "data" / "tiling_summary_v1.csv")

    relax = read_payloads(ROOT / "results" / "mlip_v1")
    if len(relax) != N_HOSTS or {p["host_id"] for p in relax} != set(uni.host_id):
        raise AssertionError(f"{len(relax)} relax records; expected one per universe host ({N_HOSTS})")
    hosts = host_table(relax, dbins, tiling)
    n_art = check_artifacts(relax, mlip_dir)

    recs = read_payloads(ROOT / "results" / "geomeval")
    if len(recs) != N_RUNS:
        raise AssertionError(f"{len(recs)} geomeval records, expected {N_RUNS}")
    specs = [json.loads(line) for line in (ROOT / "specs" / "geomeval.jsonl").read_text().splitlines() if line]
    n_spec = check_specs(recs, specs)
    cell_counts = check_cells(recs)

    excluded = set(hosts.host_id[hosts.excluded_from_C3b])
    frames, per_run = [], []
    for p in sorted(recs, key=lambda q: (q["key"]["model"], q["key"]["budget"], q["key"]["r"], q["key"]["seed"])):
        path = ROOT / p["predictions"]["path"]
        if sha256(path) != p["predictions"]["sha256"]:
            raise AssertionError(f"{p['run_id']}: prediction sha256 differs from the record")
        pred = pd.read_parquet(path)
        test_hosts = set(load_split(f"outer_r{int(p['key']['r'])}")["test"])
        test_sites = set(uni.site_id[uni.host_id.isin(test_hosts - excluded)])
        check_run(pred, test_sites, excluded, p["run_id"])
        check_targets(pred, uni)
        check_host_columns(pred, hosts, p["run_id"])
        if len(pred) != p["predictions"]["rows"] or pred.site_id.nunique() != p["n_sites"]:
            raise AssertionError(f"{p['run_id']}: row or site count differs from the record")
        frames.append(long_rows(pred, p))
        per_run.append({"run_id": p["run_id"], "n_sites": int(p["n_sites"]),
                        "n_hosts": int(p["n_hosts_evaluated"]), "n_excluded": int(p["n_excluded_from_C3b"])})
    long = pd.concat(frames, ignore_index=True)

    # vacancy atom is O in the release supercell and in the MLIP supercell
    from pymatgen.core import Structure

    vac = uni[["host_id", "vacancy_atom_index", "supercell_cif_path"]]
    used = vac[vac.host_id.isin(set(long.host_id))]
    sp_rel, sp_mlip = {}, {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for h, cif in used.groupby("host_id").supercell_cif_path.first().items():
            sp_rel[h] = [s.specie.symbol for s in Structure.from_file(str(dr / cif))]
            sp_mlip[h] = [s.specie.symbol for s in
                          Structure.from_dict(json.loads((mlip_dir / f"{h}_sc.json").read_text()))]
    pairs = list(zip(used.host_id, used.vacancy_atom_index, strict=True))
    n_o_rel = check_vacancy_species(pairs, sp_rel, "release supercell")
    n_o_mlip = check_vacancy_species(pairs, sp_mlip, "MLIP supercell")

    out_dir = ROOT / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    long_path, host_path = out_dir / "geometry_eval_long.parquet", out_dir / "geometry_hosts.parquet"
    long.to_parquet(long_path, index=False)
    hosts.to_parquet(host_path, index=False)

    pr = pd.DataFrame(per_run)
    payload = {
        "label": "registered: v1 geometry evaluation at B = 654 on our 818 relaxations; B = 200 DESCRIPTIVE EXTRA",
        "rules": "docs/deviations.md row of 2026-10-14 (§9, step A3), item j",
        "outputs": {"long": {"path": "results/analysis/geometry_eval_long.parquet", "sha256": sha256(long_path),
                             "rows": len(long)},
                    "hosts": {"path": "results/analysis/geometry_hosts.parquet", "sha256": sha256(host_path),
                              "rows": len(hosts)}},
        "inputs": {"universe": {"path": "data/processed/universe_v1.parquet", "sha256": sha256(uni_path)},
                   "host_dbins": {"path": "data/host_dbins_v1.csv", "sha256": sha256(ROOT / "data/host_dbins_v1.csv")},
                   "tiling_summary": {"path": "data/tiling_summary_v1.csv",
                                      "sha256": sha256(ROOT / "data/tiling_summary_v1.csv")},
                   "specs": {"path": "specs/geomeval.jsonl", "sha256": sha256(ROOT / "specs/geomeval.jsonl")}},
        "counts": {"geomeval_records": len(recs), "spec_lines": n_spec, "relax_records": len(relax),
                   "runs_per_model_budget": cell_counts, "rows": len(long),
                   "hosts_evaluated": int(long.host_id.nunique()), "sites_evaluated": int(long.site_id.nunique()),
                   "excluded_from_C3b": len(excluded), "excluded_hosts_in_runs": int(pr.n_excluded.sum()),
                   "nonconverged_hosts": int((~hosts.converged).sum()),
                   "rows_per_condition": {c: int((long.condition == c).sum()) for c in CONDITIONS},
                   "hosts_per_condition": {c: int(long.host_id[long.condition == c].nunique()) for c in CONDITIONS},
                   "tiling": tiling_split(tiling),
                   "geomeval_gpu": {g: int(n) for g, n in long.groupby("gpu").run_id.nunique().items()},
                   "relax_gpu": {g: int(n) for g, n in hosts.relax_gpu.value_counts().sort_index().items()},
                   "vacancy_O_checked": {"release_supercell": n_o_rel, "mlip_supercell": n_o_mlip},
                   "mlip_artifacts_hash_checked": n_art},
        "integrity": ["180 records; run ids, keys and specs equal specs/geomeval.jsonl",
                      "prediction sha256 equals the record", "one run per (model, r, B, seed)",
                      "each run covers its evaluated test sites once under each condition", "predictions finite",
                      "targets and hosts equal the universe", "excluded hosts absent",
                      "vacancy atom is O in the release and MLIP supercells",
                      "d_bin, converged and components equal data/host_dbins_v1.csv and results/mlip_v1",
                      "MLIP unit-cell and supercell sha256 equal the relax records"],
    }
    print(write_result("geometry_eval_table", payload, config=load_config(), repo_root=ROOT))


if __name__ == "__main__":
    main()
