"""The MACE-MP-0 medium potential.

The weights are obtained through mace-torch's own mechanism (``download_mace_mp_checkpoint("medium")``),
with mace-torch's cache redirected to ``<repo>/.cache/mace/`` (local only, excluded from git). In
mace-torch >= 0.3.10 the default foundation model is MACE-MPA-0, so the key "medium" is always passed
explicitly; it maps to the 2023-12-03 MACE-MP-0 medium checkpoint below, whose name and size are checked
against the GitHub release listing.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_HOME = REPO_ROOT / ".cache"
MODEL_KEY = "medium"
FILE_NAME = "2023-12-03-mace-128-L1_epoch-199.model"
RELEASE_URL = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/" + FILE_NAME
LISTING_URL = "https://api.github.com/repos/ACEsuit/mace-mp/releases/tags/mace_mp_0"
LISTED_SIZE = 44422970          # bytes, from the release listing


def model_path(cache_home: Path = CACHE_HOME) -> Path:
    """Download (once) through mace-torch and return the cached checkpoint path."""
    os.environ["XDG_CACHE_HOME"] = str(cache_home)
    from mace.calculators.foundations_models import download_mace_mp_checkpoint, mace_mp_urls

    if not mace_mp_urls[MODEL_KEY].endswith(FILE_NAME):
        raise RuntimeError(f"mace-torch maps {MODEL_KEY!r} to {mace_mp_urls[MODEL_KEY]}, not {FILE_NAME}")
    p = Path(download_mace_mp_checkpoint(MODEL_KEY))
    if p.stat().st_size != LISTED_SIZE:
        raise RuntimeError(f"{p} has {p.stat().st_size} bytes, the release lists {LISTED_SIZE}")
    return p


def calculator(device: str = "cpu", path: Path | None = None):
    """ASE calculator of MACE-MP-0 medium in float64."""
    from mace.calculators import mace_mp

    return mace_mp(model=str(path or model_path()), default_dtype="float64", device=device)
