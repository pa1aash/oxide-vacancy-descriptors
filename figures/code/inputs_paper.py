"""``results/g1_primary.json`` (v1, the G1 estimator), ``results/c1c2_v1.json`` and ``results/c1c2_v2.json``
(both backbones, one schema), ``results/diagnostics_sweep.json`` (``references``: training mean, B0,
RF-Kumagai on the same outer test sites).

Optional extension inputs. The run records land in ``results/cv/``, ``results/cv_v2/``, ``results/precision/``
and ``results/precision_v2/``. A figure never aggregates run records itself: it draws an extension only from
its analysis record, written by ``write_result`` once the analysis rule row is logged. The analysis records
use the ``c1c2_<arm>.json`` payload schema (``budgets``,
``mae.<key>.<S|D-state>.<mae|within_host_skill>.{mean, ci}``, ``A_mae.<key>.{A, ci95, upper_one_sided_95}``,
``A_within_host.<key>.{A, ci95}``), plus ``x_training_hosts.<key>``: the training-set size at which the point
is drawn (the mean fold training size for CV; the budget itself for the precision extension, where it may be
omitted).

    precision, v1 -> results/precision_g1_v1.json      runs in results/precision/
    precision, v2 -> results/precision_g1_v2.json      runs in results/precision_v2/
    cv, v1        -> results/cv_g1_v1.json             runs in results/cv/
    cv, v2        -> results/cv_g1_v2.json             runs in results/cv_v2/
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D_KEY = "D-state"
MODEL_KEYS = {"S": "S", "D": D_KEY}
EXTENSIONS = {
    ("precision", "v1"): ("precision_g1_v1", "precision"),
    ("precision", "v2"): ("precision_g1_v2", "precision_v2"),
    ("cv", "v1"): ("cv_g1_v1", "cv"),
    ("cv", "v2"): ("cv_g1_v2", "cv_v2"),
}


def results_dir(root: Path | None = None) -> Path:
    return Path(root) if root else ROOT / "results"


def load(name: str, root: Path | None = None) -> dict | None:
    p = results_dir(root) / f"{name}.json"
    return json.loads(p.read_text())["payload"] if p.exists() else None


def require(name: str, root: Path | None = None) -> dict:
    out = load(name, root)
    if out is None:
        raise FileNotFoundError(f"required input results/{name}.json is missing")
    return out


def arms(root: Path | None = None) -> dict[str, dict]:
    """Arms."""
    return {a: require(f"c1c2_{a}", root) for a in ("v2", "v1")}


def extensions(root: Path | None = None, report: bool = True) -> dict[tuple[str, str], dict]:
    """Extension analysis records that exist, keyed (kind, backbone); prints which inputs are missing."""
    found, missing = {}, []
    rd = results_dir(root)
    for (kind, arm), (record, run_dir) in EXTENSIONS.items():
        payload = load(record, root)
        runs = rd / run_dir
        n_runs = len(list(runs.glob("*.json"))) if runs.is_dir() else 0
        if payload is not None:
            found[(kind, arm)] = payload
            continue
        why = f"results/{record}.json absent"
        why += f" (results/{run_dir}/ holds {n_runs} run records; not aggregated here)" if n_runs else \
            f" (no results/{run_dir}/)"
        missing.append(f"{kind} {arm}: {why}")
    if report:
        print("optional inputs drawn: " + (", ".join(f"{k} {a}" for k, a in found) or "none"))
        for m in missing:
            print("optional input missing: " + m)
    return found


def keys(payload: dict) -> list[str]:
    return [str(b) for b in payload["budgets"]]


def x_of(payload: dict) -> list[float]:
    xs = payload.get("x_training_hosts", {})
    return [float(xs.get(k, k)) for k in keys(payload)]


def mae_series(payload: dict, model: str, metric: str = "mae"):
    """(x, mean, lo, hi) of ``mae.<key>.<model>.<metric>`` in a c1c2-schema payload."""
    rows = [payload["mae"][k][MODEL_KEYS[model]][metric] for k in keys(payload)]
    return x_of(payload), [r["mean"] for r in rows], [r["ci"][0] for r in rows], [r["ci"][1] for r in rows]


def a_series(payload: dict, block: str = "A_mae"):
    """(x, A, lo, hi, upper one-sided) of ``<block>.<key>`` in a c1c2-schema payload."""
    rows = [payload[block][k] for k in keys(payload)]
    return (x_of(payload), [r["A"] for r in rows], [r["ci95"][0] for r in rows], [r["ci95"][1] for r in rows],
            [r.get("upper_one_sided_95") for r in rows])


def ref_series(refs: dict, key: str, budgets):
    """(x, mean, lo, hi) of ``diagnostics_sweep.references.<B>.<key>``."""
    rows = [refs[str(b)][key] for b in budgets]
    return list(budgets), [r["mean"] for r in rows], [r["ci"][0] for r in rows], [r["ci"][1] for r in rows]


def x_upper(found: dict, default: float = 654.0) -> float:
    """Largest x any drawn series reaches (CV points sit beyond the largest outer-resample budget)."""
    xs = [x for p in found.values() for x in x_of(p)]
    return max([default, *xs])


def cli(doc: str):
    """Common options: --results-root (default results/) and --out-dir (default figures/main/)."""
    import argparse

    ap = argparse.ArgumentParser(description=doc.splitlines()[0])
    ap.add_argument("--results-root", type=Path, default=None, help="directory holding the result records")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "figures" / "main")
    return ap.parse_args()


def write_caption(out_dir: Path, stem: str, text: str) -> Path:
    path = Path(out_dir) / f"{stem}.caption.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n")
    return path


def drawn_note(found: dict) -> str:
    """Caption sentences on the optional extension series: which records, how each design is marked. No typed
    numbers; every value is named by record and key."""
    if not found:
        return ("No extension series is drawn: the K-fold CV and precision-extension analysis records "
                "(`results/cv_g1_v1.json`, `results/cv_g1_v2.json`, `results/precision_g1_v1.json`, "
                "`results/precision_g1_v2.json`) do not exist yet.")
    names = {"precision": "precision extension", "cv": "grouped K-fold CV"}
    parts = [f"{names[k]}, {'primary' if a == 'v2' else 'originally registered'} backbone "
             f"(`results/{EXTENSIONS[(k, a)][0]}.json`)" for k, a in found]
    kinds = {k for k, _ in found}
    text = ("Three series are drawn, each labelled in the legend by its marker shape: circles on lines, the "
            "registered outer-resample design (the resamples and budgets are registered; the primary backbone "
            "(v2) evaluated on them is not); diamonds, the precision extension (EXTENSION), the pooled estimate "
            "over the resamples and seeds named in `payload.design`, drawn just left of its budget; triangles, "
            "grouped K-fold CV (EXTENSION), placed at the mean fold training size `payload.x_training_hosts`. "
            "The extension series are unconnected points with two-sided interval bars (primary backbone (v2): "
            "solid bars, filled markers; originally registered backbone (v1): dashed bars, open markers) and are not "
            "pre-registered (label `payload.label` of each record; analysis rules in `docs/deviations.md`, cited "
            "under `payload.rules` of the source records). "
            "Records: " + "; ".join(parts) + ", each with its source analysis in `payload.source_record`.")
    if "cv" in kinds:
        text += (" The CV test sets differ from the registered ones: every host is tested once across the folds, "
                 "including hosts that no outer resample tests, so a CV value is not directly comparable with a "
                 "registered-budget value at a similar number of training hosts (partition check in "
                 "`results/cv_extended_budget.json`, `payload.fold_partition`).")
    return text
