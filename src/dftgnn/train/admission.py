"""VRAM and host-RAM estimates for the queue's admission control.

``est_peak_gb`` of a job is the benchmark's realistic peak (``torch.cuda.max_memory_allocated`` over a
shuffled epoch, ``results/gpu_benchmark.json``) for its (hidden width, batch size, MEGNet blocks) times
``SAFETY`` = 1.25. The benchmark grid covers every cell of the search space. Models D-state, D-late, P1
and P differ from S only by a few input or output columns, so S's peak is used for all of them.
The workers' host-RAM estimate is the largest per-worker peak resident set in the concurrency runs.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

SAFETY = 1.25
HOST_RAM_GB = 62.0
HOST_RAM_FRACTION = 0.8
GIB = 2**30
BENCHMARK = Path(__file__).resolve().parents[3] / "results" / "gpu_benchmark.json"


def load_benchmark(path: Path = BENCHMARK) -> dict:
    """Peak-memory table ``{(hidden, batch, blocks): realistic_peak_bytes}`` plus the host-RAM peak."""
    pay = json.loads(Path(path).read_text())["payload"]
    table = {(r["hidden"], r["batch"], r["blocks"]): r["realistic_peak_bytes"] for r in pay["grid"]}
    rss = [r["host_peak_rss_bytes_per_worker"] for r in pay["concurrency"]]
    return {"peak_bytes": table, "host_peak_rss_bytes": max(rss) if rss else None,
            "total_vram_bytes": pay["total_vram_bytes"]}


def est_peak_gb(table: dict, hidden: int, batch: int, blocks: int, safety: float = SAFETY) -> float:
    """Realistic peak times the safety factor, in GiB. KeyError if the cell was not benchmarked."""
    return table["peak_bytes"][(hidden, batch, blocks)] * safety / GIB


def est_peak_for_spec(table: dict, hp: dict, batch_size: int) -> float:
    """CGCNN specs (no MEGNet cell) use the mid-space MEGNet cell (64, batch, 3) as an upper proxy: the
    CGCNN of the cross-check has fewer parameters and smaller activations than that cell."""
    if "atom_fea_len" in hp:
        return est_peak_gb(table, 64, batch_size, 3)
    return est_peak_gb(table, hp["hidden_width"], batch_size, hp["megnet_blocks"])


def est_host_gb(table: dict, safety: float = SAFETY) -> float | None:
    b = table["host_peak_rss_bytes"]
    return None if b is None else b * safety / GIB


def max_workers_by_ram(host_gb_per_worker: float, total_gb: float = HOST_RAM_GB,
                       fraction: float = HOST_RAM_FRACTION) -> int:
    """Most workers whose estimated host RAM stays below ``fraction`` of ``total_gb`` (at least 1)."""
    return max(1, math.floor(fraction * total_gb / host_gb_per_worker))
