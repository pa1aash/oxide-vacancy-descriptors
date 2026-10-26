"""Shared paths for the queue CLIs."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUEUE = ROOT / "jobs"
OUTBOX = ROOT / "outbox"
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
