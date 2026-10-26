"""Long evaluation table of the staging analysis C3a, both backbones.

Analysis rules:, row of 2026-10-14, which restates the staging pairing of 2026-10-10 and the TOST rule of
2026-10-09. Pure reading of committed run records and prediction parquets plus the universe table (read only,
sha256 recorded). Writes

    results/analysis/staging_eval_long.parquet   backbone, model (S, D, P), r, B, seed, site_id, host_id, y_true,
                                                 y_pred, run_id (outer resamples only)
    results/analysis/staging_p1_desc.parquet     backbone, r, B, seed, run_id, descriptor, class, n, mae, r2 (one row
                                                 per P1 run on an outer resample and descriptor)
    results/analysis/staging_runs.parquet        one row per run: backbone, model, split, r, B, seed, run_id, source,
                                                 mae, epochs_run, max_epochs, d_run, p1_run
    results/staging_eval_table.json              hashes, counts and the integrity checks (via write_result)

Inputs per backbone (row of 2026-10-14, item b): v1 S and D-state = the sweep records directly under `results/`
(outer r = 0..9, seeds 0-2, max_epochs 200), P = `results/peval/`, P1 = the P1 records directly under `results/`;
v2 S and D-state = `results/v2_sweep/`, P = `results/p_v2/`, P1 = `results/p1_v2/`.

Integrity assertions (item f; any failure raises): each run covers the test sites of its split exactly once; all
predictions finite; no test host among the training hosts; targets and hosts equal the universe; prediction sha256
equals the record; recorded MAE (or P1 per-descriptor metrics) equals the stored predictions; one run per (backbone,
model, r, B, seed); counts 183 P and 183 P1 per backbone and 180 S and 180 D on the outer resamples; v1 P run ids
and specs equal `specs/peval.jsonl`; every P record's components carry its split, budget and seed, and its D
component is the D-state run of that cell.

    PYTHONPATH=$PWD/src python analysis/staging_eval_table.py [--data-root DIR]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = (25, 50, 100, 200, 400, 654)
N_RESAMPLES = 10
SEEDS = (0, 1, 2)
BACKBONES = ("v2", "v1")
MODEL_NAME = {"S": "S", "D-state": "D", "P": "P", "P1": "P1"}
# results_subdir of each (backbone, model); None = the records directly under results/
SOURCES = {"v1": {"S": None, "D-state": None, "P": "peval", "P1": None},
           "v2": {"S": "v2_sweep", "D-state": "v2_sweep", "P": "p_v2", "P1": "p1_v2"}}
# where the Kiyohara-split D-state runs (the D components of the Kiyohara P runs) live
KIYOHARA_D = {"v1": None, "v2": "v2_kiyohara"}
N_OUTER = len(BUDGETS) * N_RESAMPLES * len(SEEDS)        # 180 cells per model and backbone
N_STAGED = N_OUTER + len(SEEDS)                          # 183: plus the Kiyohara-split cells (P, P1)
DESC_CLASSES = {"host-electronic DFT": 12, "site-electronic DFT": 10}
TRACK_LIMIT = 20 * 1024**2


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- run records

def read_records(directory: Path, subdir: str | None, model: str) -> list[dict]:
    """Payloads of ``model`` runs of one results directory whose ``results_subdir`` is ``subdir``."""
    out = []
    for f in sorted(directory.glob("*.json")):
        try:
            pay = json.loads(f.read_text()).get("payload")
        except ValueError:
            continue
        if not isinstance(pay, dict) or "spec" not in pay or pay.get("smoke"):
            continue
        sp = pay["spec"]
        if sp.get("model") != model or sp.get("tags", {}).get("pilot") or sp.get("results_subdir") != subdir:
            continue
        out.append(pay)
    return out


def is_outer(p: Mapping) -> bool:
    return str(p["spec"].get("split", "")).startswith("outer_r")


def select_end_to_end(recs: Iterable[dict]) -> list[dict]:
    """S or D-state sweep runs of the staging cells: outer r = 0..9, the six budgets, seeds 0-2, 200-epoch cap."""
    keep = []
    for p in recs:
        sp = p["spec"]
        if not is_outer(p) or int(p.get("max_epochs") or 0) != 200:
            continue
        if int(sp["budget"]) in BUDGETS and int(sp["r"]) < N_RESAMPLES and int(sp["seed"]) in SEEDS:
            keep.append(p)
    return keep


def run_row(p: Mapping, backbone: str, source: str) -> dict:
    sp = p["spec"]
    outer = is_outer(p)
    return {"backbone": backbone, "model": MODEL_NAME[sp["model"]], "split": sp["split"],
            "r": int(sp["r"]) if outer else -1, "B": int(sp["budget"]), "seed": int(sp["seed"]),
            "run_id": p["run_id"], "source": source,
            "mae": float(p["metrics"]["mae"]) if "mae" in p["metrics"] else np.nan,
            "epochs_run": int(p["epochs_run"]) if p.get("epochs_run") is not None else -1,
            "max_epochs": int(p["max_epochs"]) if p.get("max_epochs") is not None else -1,
            "d_run": sp.get("d_run") or "", "p1_run": sp.get("p1_run") or "",
            "pred_path": p["predictions"]["path"], "pred_sha256": p["predictions"]["sha256"]}


# ---------------------------------------------------------------- integrity

def check_run(pred: pd.DataFrame, test_sites: set[str], test_hosts: set[str], train_hosts: Iterable[str],
              label: str, value_cols: Sequence[str] = ("y_pred",)) -> None:
    """One run: finite, every test site exactly once, no test host in training."""
    if not np.isfinite(pred[list(value_cols)].to_numpy(float)).all():
        raise AssertionError(f"{label}: non-finite predictions")
    if pred.site_id.duplicated().any():
        raise AssertionError(f"{label}: duplicate site")
    if set(pred.site_id) != test_sites:
        raise AssertionError(f"{label}: sites differ from the test set")
    if test_hosts & set(train_hosts):
        raise AssertionError(f"{label}: test host in training")


def check_targets(pred: pd.DataFrame, uni: pd.DataFrame) -> None:
    u = uni.set_index("site_id")
    if not np.allclose(pred.y_true.to_numpy(), u.target_Ef_eV.loc[pred.site_id].to_numpy(), atol=1e-9):
        raise AssertionError("targets differ from the universe")
    if not (pred.host_id.to_numpy() == u.host_id.loc[pred.site_id].to_numpy()).all():
        raise AssertionError("hosts differ from the universe")


def check_unique_cells(runs: pd.DataFrame) -> None:
    dup = runs.duplicated(subset=["backbone", "model", "split", "B", "seed"], keep=False)
    if dup.any():
        raise AssertionError(f"more than one run per cell: {runs.loc[dup, 'run_id'].tolist()[:6]}")


def check_counts(runs: pd.DataFrame) -> dict:
    """183 P and P1 per backbone (180 outer + 3 Kiyohara); S and D: the 180 outer cells, complete."""
    out = {}
    want_cells = {(B, r, s) for B in BUDGETS for r in range(N_RESAMPLES) for s in SEEDS}
    for bb in BACKBONES:
        for m in ("S", "D", "P", "P1"):
            sel = runs[(runs.backbone == bb) & (runs.model == m)]
            outer = sel[sel.r >= 0]
            cells = set(zip(outer.B, outer.r, outer.seed, strict=True))
            if cells != want_cells:
                raise AssertionError(f"{bb} {m}: outer cells missing {sorted(want_cells - cells)[:4]} extra "
                                     f"{sorted(cells - want_cells)[:4]}")
            n_all = len(sel)
            if m in ("P", "P1") and n_all != N_STAGED:
                raise AssertionError(f"{bb} {m}: {n_all} runs, expected {N_STAGED}")
            if m in ("S", "D") and n_all != N_OUTER:
                raise AssertionError(f"{bb} {m}: {n_all} runs, expected {N_OUTER}")
            out[f"{bb}|{m}"] = {"total": n_all, "outer": len(outer), "kiyohara": n_all - len(outer)}
    return out


def check_pairing(runs: pd.DataFrame) -> int:
    """Every P run: its P1 and D components exist, carry the P run's split, budget and seed, and its D component is
    the D-state run of that cell used for D (row of 2026-10-10, §8). Returns the number of P runs checked."""
    by_id = runs.set_index("run_id")
    n = 0
    for p in runs[runs.model == "P"].itertuples(index=False):
        for comp, model in ((p.p1_run, "P1"), (p.d_run, "D")):
            if comp not in by_id.index:
                if model == "D" and p.r < 0:          # Kiyohara-split D runs are not in the outer table
                    continue
                raise AssertionError(f"{p.backbone} P {p.run_id}: component {comp} not found")
            c = by_id.loc[comp]
            if c.backbone != p.backbone or c.model != model:
                raise AssertionError(f"{p.run_id}: component {comp} is {c.backbone} {c.model}, expected {model}")
            if (c.split, int(c.B), int(c.seed)) != (p.split, p.B, p.seed):
                raise AssertionError(f"{p.run_id}: component {comp} is cell {(c.split, c.B, c.seed)}, expected "
                                     f"{(p.split, p.B, p.seed)}")
        n += 1
    return n


def check_kiyohara_d(runs: pd.DataFrame, d_kiyohara: Mapping[str, Mapping[str, int]]) -> int:
    """The D component of each Kiyohara-split P run is a D-state Kiyohara run of the same backbone and seed."""
    n = 0
    for p in runs[(runs.model == "P") & (runs.r < 0)].itertuples(index=False):
        if d_kiyohara[p.backbone].get(p.d_run) != p.seed:
            raise AssertionError(f"{p.backbone} P {p.run_id}: D component {p.d_run} is not the Kiyohara D-state run "
                                 f"of seed {p.seed}")
        n += 1
    return n


def check_specs(p_recs: Sequence[Mapping], spec_lines: Sequence[Mapping]) -> int:
    """v1 P run ids equal the run ids of ``specs/peval.jsonl`` and each record's spec equals its line's spec."""
    want = {s["run_id"]: s["spec"] for s in spec_lines}
    got = {p["run_id"]: p["spec"] for p in p_recs}
    if set(got) != set(want):
        raise AssertionError(f"peval run ids differ from specs/peval.jsonl: missing {sorted(set(want) - set(got))[:3]}"
                             f" extra {sorted(set(got) - set(want))[:3]}")
    for rid, sp in got.items():
        if sp != want[rid]:
            raise AssertionError(f"{rid}: record spec differs from specs/peval.jsonl")
    return len(got)


def per_descriptor(df: pd.DataFrame, names: Sequence[str]) -> dict[str, dict]:
    """Test MAE and R^2 (SS_tot about the test-site mean) of each descriptor from true_desc_*/pred_desc_* columns."""
    out = {}
    for nm in names:
        t = df[f"true_{nm}"].to_numpy(float)
        p = df[f"pred_{nm}"].to_numpy(float)
        sst = float(((t - t.mean()) ** 2).sum())
        out[nm] = {"n": len(t), "mae": float(np.abs(p - t).mean()),
                   "r2": float(1 - ((p - t) ** 2).sum() / sst) if sst > 0 else np.nan}
    return out


def descriptor_classes(path: Path, names: Sequence[str]) -> dict[str, str]:
    """``desc_<name>`` -> class from ``docs/descriptor_taxonomy.csv``; asserts 12 host- and 10 site-electronic."""
    with open(path, newline="") as f:
        tax = {row["name"]: row["class"] for row in csv.DictReader(f)}
    cls = {nm: tax[nm.removeprefix("desc_")] for nm in names}
    counts = {c: sum(v == c for v in cls.values()) for c in DESC_CLASSES}
    if counts != DESC_CLASSES or len(cls) != sum(DESC_CLASSES.values()):
        raise AssertionError(f"descriptor classes {counts}, expected {DESC_CLASSES}")
    return cls


# ---------------------------------------------------------------- build

def load_e_f(row, root: Path) -> pd.DataFrame:
    path = root / row.pred_path
    if sha256(path) != row.pred_sha256:
        raise AssertionError(f"{row.run_id}: predictions hash mismatch")
    df = pd.read_parquet(path)[["site_id", "host_id", "y_true", "y_pred"]]
    if abs(float(np.abs(df.y_pred - df.y_true).mean()) - row.mae) > 1e-9:
        raise AssertionError(f"{row.run_id}: recorded MAE differs from the stored predictions")
    return df.assign(run_id=row.run_id)


def load_p1(row, rec_metrics: Mapping, root: Path) -> tuple[pd.DataFrame, dict]:
    path = root / row.pred_path
    if sha256(path) != row.pred_sha256:
        raise AssertionError(f"{row.run_id}: predictions hash mismatch")
    df = pd.read_parquet(path)
    names = sorted(c.removeprefix("true_") for c in df.columns if c.startswith("true_desc_"))
    per = per_descriptor(df, names)
    stored = rec_metrics.get("per_descriptor", {})
    if set(stored) != set(names):
        raise AssertionError(f"{row.run_id}: per-descriptor metrics in the record differ from the stored columns")
    for nm, v in stored.items():
        for k in ("mae", "r2"):
            if abs(v[k] - per[nm][k]) > 1e-9:
                raise AssertionError(f"{row.run_id} {nm}: recorded {k} differs from the stored predictions")
    return df, per


def build(uni: pd.DataFrame, root: Path) -> dict:
    from dftgnn.split import budget_train, load_split

    rows, recs_by_id, d_kiyo = [], {}, {}
    for bb in BACKBONES:
        d_kiyo[bb] = {}
        for model, sub in SOURCES[bb].items():
            directory = root / "results" / sub if sub else root / "results"
            recs = read_records(directory, sub, model)
            if model in ("S", "D-state"):
                recs = select_end_to_end(recs)
            for p in recs:
                recs_by_id[p["run_id"]] = p
                rows.append(run_row(p, bb, sub or "results"))
        kd = KIYOHARA_D[bb]
        d_kiyo[bb] = {p["run_id"]: int(p["spec"]["seed"])
                      for p in read_records(root / "results" / kd if kd else root / "results", kd, "D-state")
                      if p["spec"].get("split") == "kiyohara"}
    runs = pd.DataFrame(rows)
    check_unique_cells(runs)
    counts = check_counts(runs)
    n_paired = check_pairing(runs)
    n_kiyo = check_kiyohara_d(runs, d_kiyo)
    spec_lines = [json.loads(x) for x in (root / "specs" / "peval.jsonl").read_text().splitlines() if x.strip()]
    v1_p = [recs_by_id[r] for r in runs.run_id[(runs.backbone == "v1") & (runs.model == "P")]]
    n_spec = check_specs(v1_p, spec_lines)

    sites_of = uni.groupby("host_id").site_id.apply(set).to_dict()
    splits = {r: load_split(f"outer_r{r}") for r in range(N_RESAMPLES)}
    splits["kiyohara"] = load_split("kiyohara")
    frames, desc_rows, checked = [], [], 0
    names_ref, classes = None, None
    for row in runs.itertuples(index=False):
        sp = splits[row.r] if row.r >= 0 else splits["kiyohara"]
        th = set(sp["test"])
        train = budget_train(sp, row.B) if row.r >= 0 else [*sp["train"], *sp["val"]]
        ts = set().union(*(sites_of[h] for h in th))
        label = f"{row.backbone} {row.model} {row.run_id}"
        if row.model == "P1":
            df, per = load_p1(row, recs_by_id[row.run_id]["metrics"], root)
            names = sorted(per)
            if names_ref is None:
                names_ref, classes = names, descriptor_classes(root / "docs" / "descriptor_taxonomy.csv", names)
            if names != names_ref:
                raise AssertionError(f"{label}: descriptor set differs")
            check_run(df, ts, th, train, label, [f"pred_{n}" for n in names])
            if row.r >= 0:
                desc_rows += [{"backbone": row.backbone, "r": row.r, "B": row.B, "seed": row.seed,
                               "run_id": row.run_id, "descriptor": nm, "class": classes[nm], **per[nm]}
                              for nm in names]
        else:
            df = load_e_f(row, root)
            check_run(df, ts, th, train, label)
            check_targets(df, uni)
            if row.r >= 0:
                frames.append(df.assign(backbone=row.backbone, model=row.model, r=row.r, B=row.B, seed=row.seed))
        checked += 1
    long = pd.concat(frames, ignore_index=True)[["backbone", "model", "r", "B", "seed", "site_id", "host_id",
                                                 "y_true", "y_pred", "run_id"]]
    long = long.sort_values(["backbone", "model", "B", "r", "seed", "site_id"]).reset_index(drop=True)
    desc = pd.DataFrame(desc_rows).sort_values(["backbone", "B", "r", "seed", "descriptor"]).reset_index(drop=True)
    runs = runs.drop(columns=["pred_path"]).sort_values(["backbone", "model", "B", "r", "seed"]).reset_index(drop=True)
    info = {"counts": counts, "runs_checked": checked, "p_runs_pairing_checked": n_paired,
            "kiyohara_p_runs_d_component_checked": n_kiyo,
            "v1_p_runs_matching_specs_peval": n_spec, "descriptor_classes": DESC_CLASSES,
            "descriptors": {c: [n for n in names_ref if classes[n] == c] for c in DESC_CLASSES}}
    return {"long": long, "desc": desc, "runs": runs, "info": info}


def save(df: pd.DataFrame, name: str, root: Path) -> dict:
    out = root / "results" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.parquet"
    df.to_parquet(path, index=False)
    size = path.stat().st_size
    return {"path": str(path.relative_to(root)), "sha256": sha256(path), "bytes": size, "rows": len(df),
            "tracked": size < TRACK_LIMIT}


def main() -> None:
    from dftgnn.config import load_config
    from dftgnn.io.results import write_result

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, default=ROOT,
                    help="checkout holding the gitignored data/processed/universe_v1.parquet")
    a = ap.parse_args()
    cfg = load_config()
    uni_path = a.data_root / "data" / "processed" / "universe_v1.parquet"
    uni = pd.read_parquet(uni_path, columns=["site_id", "host_id", "target_Ef_eV"])
    out = build(uni, ROOT)
    files = {"staging_eval_long": save(out["long"], "staging_eval_long", ROOT),
             "staging_p1_desc": save(out["desc"], "staging_p1_desc", ROOT),
             "staging_runs": save(out["runs"], "staging_runs", ROOT)}
    runs = out["runs"]
    payload = {
        "label": "v1: registered staging analysis (originally registered backbone); v2: primary backbone, POST HOC; "
                 "TOST rule POST HOC (ratified 2026-10-11)",
        "rules": "docs/deviations.md row of 2026-10-14 (§8, step A2), items b and f",
        "files": files,
        "universe": {"path": "data/processed/universe_v1.parquet", "sha256": sha256(uni_path)},
        "sources": SOURCES,
        "integrity": {**out["info"], "assertions": [
            "every run covers the test sites of its split exactly once", "all predictions finite",
            "no test host among the training hosts", "targets and hosts equal the universe",
            "prediction sha256 equals the record",
            "recorded MAE and P1 per-descriptor metrics equal the stored predictions", "one run per (backbone, model, split, budget, seed)",
            "183 P and 183 P1 per backbone; 180 S and 180 D on the outer resamples",
            "v1 P run ids and specs equal specs/peval.jsonl",
            "P components carry the P run's split, budget and seed; D component is the D-state run of the cell"],
            "passed": True},
        "n_rows": {"eval_long": len(out["long"]), "p1_desc": len(out["desc"])},
        "run_ids": {bb: {m: sorted(runs.run_id[(runs.backbone == bb) & (runs.model == m)].tolist())
                         for m in ("S", "D", "P", "P1")} for bb in BACKBONES},
    }
    print(json.dumps({k: v for k, v in payload.items() if k != "run_ids"}, indent=1))
    print(write_result("staging_eval_table", payload, config=cfg, repo_root=ROOT))


if __name__ == "__main__":
    main()
