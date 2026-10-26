"""Write the EXTENSION split files and append their hashes to splits/MANIFEST.sha256.

    python scripts/make_splits_ext.py [--check]

Targets: ``splits/resamples_10_29.json`` (outer resamples r = 10..29, ``dftgnn.split.ext``) and
``splits/cv_v1.json`` (grouped K-fold CV, K = 10 and 20, ``dftgnn.split.cv``).

Before anything is written, resamples 0-9 are regenerated in memory and must be byte-identical to the tracked
``splits/outer_r<r>.json``, and every existing manifest line must verify. The manifest is append-only: an existing
line is never rewritten or removed, a target whose name is already listed must match its listed hash, and an
existing target file must be byte-identical to its regeneration (the script refuses to modify it).
``--check`` regenerates in memory and verifies files and manifest lines; it writes nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dftgnn.config import load_config
from dftgnn.data import universe as U
from dftgnn.split import SPLITS_DIR, dumps, make_all
from dftgnn.split.cv import CV_FILE, make_cv
from dftgnn.split.ext import EXT_FILE, make_ext


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def manifest_entries(splits_dir: Path) -> list[tuple[str, str]]:
    return [tuple(ln.split("  ", 1)) for ln in (splits_dir / "MANIFEST.sha256").read_text().splitlines()]


def verify_manifest(splits_dir: Path) -> list[str]:
    """Names whose file does not match the manifest hash."""
    return [n for s, n in manifest_entries(splits_dir) if sha((splits_dir / n).read_bytes()) != s]


def check_registered(uni, cfg, splits_dir: Path) -> None:
    reg = make_all(uni, cfg)
    for r in range(cfg.split.n_outer_resamples):
        if (splits_dir / f"outer_r{r}.json").read_bytes() != dumps(reg[f"outer_r{r}"]):
            raise SystemExit(f"outer_r{r}.json differs from its in-memory regeneration; refusing")


def append_only(splits_dir: Path, files: dict[str, bytes], check: bool) -> list[str]:
    """Write each target if absent (else require identical bytes) and append manifest lines for new names."""
    man = splits_dir / "MANIFEST.sha256"
    before = man.read_text()
    if before and not before.endswith("\n"):
        raise SystemExit("MANIFEST.sha256 does not end with a newline; refusing to append")
    listed = {n: s for s, n in manifest_entries(splits_dir)}
    new_lines = []
    for name, b in files.items():
        f = splits_dir / name
        if name in listed and listed[name] != sha(b):
            raise SystemExit(f"{name}: manifest lists {listed[name]}, regeneration gives {sha(b)}; refusing")
        if f.exists():
            if f.read_bytes() != b:
                raise SystemExit(f"{name} exists and differs from its regeneration; refusing to modify it")
        elif check:
            raise SystemExit(f"{name} is missing")
        else:
            f.write_bytes(b)
        if name not in listed:
            if check:
                raise SystemExit(f"{name} is not in the manifest")
            new_lines.append(f"{sha(b)}  {name}\n")
    if new_lines:
        man.write_text(before + "".join(new_lines))
        if not man.read_text().startswith(before):
            raise SystemExit("manifest prefix changed")
    return [ln.strip() for ln in new_lines]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify only; write nothing")
    ap.add_argument("--splits-dir", default=str(SPLITS_DIR))
    a = ap.parse_args()
    d = Path(a.splits_dir)
    cfg = load_config()
    uni = U.read_universe(cfg)
    if sha(U.ids_csv_bytes(uni)) != sha(U.IDS_FILE.read_bytes()):
        raise SystemExit("universe does not match the tracked IDs file")
    bad = verify_manifest(d)
    if bad:
        raise SystemExit(f"existing manifest entries do not verify: {bad}")
    check_registered(uni, cfg, d)
    print(f"resamples 0-{cfg.split.n_outer_resamples - 1}: byte-identical to the tracked files")
    ext = make_ext(uni, cfg)
    cv = make_cv(uni)
    files = {EXT_FILE: dumps(ext), CV_FILE: dumps(cv)}
    added = append_only(d, files, a.check)
    bad = verify_manifest(d)
    if bad:
        raise SystemExit(f"manifest entries do not verify after writing: {bad}")
    for name, b in files.items():
        print(f"{name}: sha256 {sha(b)}")
    cov = ext["coverage_registered_plus_extension"]
    print(f"coverage over {cov['n_resamples']} resamples: {cov['in_test_at_least_once']} of {cov['n_hosts']} hosts "
          f"tested at least once, {cov['never_in_test']} never")
    for k, by in cv["by_k"].items():
        sizes = sorted({(f["n_test"], f["n_train"]) for f in by["folds"]})
        print(f"cv K={k}: {len(by['folds'])} folds, (test, train) sizes {sizes}")
    print("appended: " + ("; ".join(added) if added else "nothing"))
    print(f"manifest: every entry verifies; MANIFEST.sha256 sha256 {sha((d / 'MANIFEST.sha256').read_bytes())}")


if __name__ == "__main__":
    main()
