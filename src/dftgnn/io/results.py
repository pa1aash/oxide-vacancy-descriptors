"""Write result payloads together with their full provenance."""
from __future__ import annotations

import json
import socket
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from dftgnn.config import Config, load_config

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGES = (
    "numpy", "scipy", "pandas", "scikit-learn", "pymatgen", "ase",
    "torch", "torch_geometric", "mace-torch", "matgl", "pydantic", "optuna",
)


class DirtyTreeError(RuntimeError):
    """Raised when a result would be written from an uncommitted tree."""


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for pkg in PACKAGES:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def write_result(
    name: str,
    payload: Any,
    *,
    config: Config | None = None,
    allow_dirty: bool = False,
    repo_root: Path | None = None,
    results_dir: Path | None = None,
) -> Path:
    """Write ``results/<name>.json`` with config, git SHA, time, host, seeds, versions."""
    root = Path(repo_root) if repo_root else REPO_ROOT
    cfg = config if config is not None else load_config()
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no", cwd=root))
    if dirty and not allow_dirty:
        raise DirtyTreeError("working tree is dirty; commit first or pass allow_dirty=True")
    record = {
        "name": name,
        "git_sha": _git("rev-parse", "HEAD", cwd=root),
        "git_dirty": dirty,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "hostname": socket.gethostname(),
        "seeds": list(cfg.split.seeds),
        "package_versions": _versions(),
        "config": cfg.model_dump(mode="json"),
        "payload": payload,
    }
    out_dir = Path(results_dir) if results_dir else root / cfg.paths.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}.json"
    out.write_text(json.dumps(record, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return out
