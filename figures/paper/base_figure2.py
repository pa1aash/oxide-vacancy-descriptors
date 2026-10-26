"""Series selection of Figure 2 (rows of ``results/derived_c1/final_network_summary.json``), used by
figure2_descriptor_advantage.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import network_style as ns
import style_paper as sp

SUB, STEM = "main", "figure2_descriptor_advantage"
CURVE = ("B25", "B50", "B100", "B200", "B400", "B654")
POOLED = ("B400_30x5", "B654_30x5")
CV = ("CV_k10", "CV_k20")
A_FROM = 100


def mae(rows, keys, col):
    sel = [rows[k] for k in keys]
    return ([r["x_training_hosts"] for r in sel], [r[col]["mean"] for r in sel],
            [r[col]["ci95"][0] for r in sel], [r[col]["ci95"][1] for r in sel])


def adv(rows, keys):
    sel = [rows[k] for k in keys]
    return ([r["x_training_hosts"] for r in sel], [r["A"]["A"] for r in sel], [r["A"]["ci95"][0] for r in sel],
            [r["A"]["ci95"][1] for r in sel], [r["A"]["upper_one_sided_95"] for r in sel])


