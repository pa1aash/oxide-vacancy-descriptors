"""Index of finished training runs from their results.

Inference stages (P evaluation, geometry conditions, embeddings) reuse the existing checkpoints by run id. Run
ids depend on the code SHA, so they cannot be recomputed after a code change; this index reads them from the
committed result files instead. ``lookup(model, split, budget, seed)`` returns the record of the primary
(200-epoch, non-smoke) run.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCES = ("", "c0_official", "capsens")


def load(root: Path = REPO_ROOT, sources=SOURCES) -> list[dict]:
    out = []
    for sub in sources:
        for f in sorted((root / "results" / sub).glob("*.json")):
            try:
                p = json.loads(f.read_text())["payload"]
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            if not isinstance(p, dict) or "run_id" not in p or "spec" not in p or p.get("smoke"):
                continue
            if p["spec"]["model"] == "P" or p.get("checkpoint") is None:
                continue
            out.append({"run_id": p["run_id"], "spec": p["spec"], "source": sub or "sweep",
                        "checkpoint": p["checkpoint"], "predictions": p["predictions"],
                        "epochs_run": p.get("epochs_run"), "result": str(f.relative_to(root))})
    return out


def primary(index: list[dict]) -> dict[tuple, dict]:
    """(model, split, budget, seed) -> record, for 200-epoch non-ablated runs."""
    out = {}
    for rec in index:
        s = rec["spec"]
        if s.get("max_epochs") is not None or s.get("ablate") or rec["source"] == "capsens" or s.get("hosts"):
            continue
        key = (s["model"], s["split"], s["budget"], s["seed"])
        if key in out:
            raise ValueError(f"two primary runs for {key}: {out[key]['run_id']}, {rec['run_id']}")
        out[key] = rec
    return out
