"""Record names of Figure 3 (learning-curve design, additional resamples, grouped cross-validation),
used by figure3_within_host.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inputs_paper as ip
import network_style as ns
import style_paper as sp

SUB, STEM = "main", "figure3_within_host"
RECORDS = (("curve", f"c1c2_{ns.NETWORK}"), ("pooled", f"precision_g1_{ns.NETWORK}"), ("cv", f"cv_g1_{ns.NETWORK}"))


