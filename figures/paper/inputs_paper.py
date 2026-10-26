"""Readers of the learning-curve records (``results/c1c2_<network>.json`` layout) used by the figures.

A record has ``budgets``, ``mae.<key>.<S|D-state>.<mae|within_host_skill>.{mean, ci}``,
``A_mae.<key>.{A, ci95, upper_one_sided_95}``, ``A_within_host.<key>.{A, ci95}`` and optionally
``x_training_hosts.<key>`` (the training-set size at which a point is drawn: the mean fold size for
cross-validation).
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
    return {a: require(f"c1c2_{a}", root) for a in ("v2", "v1")}


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


