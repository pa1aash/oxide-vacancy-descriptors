"""Model order, labels, contrast colours and the series reader of Figure 4, used by figure4_staging.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import network_style as ns
import style_paper as sp

SUB, STEM = "main", "figure4_staging"
CONTRAST_COLORS = {"P_minus_S": "#D55E00", "D_minus_P": sp.COLORS["A"]}
CONTRAST_LABELS = {"P_minus_S": r"P $-$ S", "D_minus_P": r"D $-$ P"}
MODELS = ("S", "D", "P")
MODEL_LABELS = {"S": "S (structure only)", "D": "D (structure + descriptors)",
                "P": "P (predicted descriptors into D)"}


def series(payload: dict, getter):
    x = sorted(int(b) for b in payload["budgets"])
    rows = [getter(payload["budgets"][str(b)]) for b in x]
    return x, [r["mean"] for r in rows], [r["ci95"][0] for r in rows], [r["ci95"][1] for r in rows]


