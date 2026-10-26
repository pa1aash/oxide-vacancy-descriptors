"""Write splits/loco_v1.json and append its hash to splits/MANIFEST.sha256 (existing lines untouched).

    python scripts/make_loco_splits.py [--check]

``--check`` regenerates in memory and compares bytes and the manifest line; it writes nothing.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from dftgnn.split import SPLITS_DIR, dumps
from dftgnn.split.loco import N_FAMILIES_V1, make_loco

ROOT = Path(__file__).resolve().parents[1]
NAME = "loco_v1.json"


def verify_manifest(splits_dir: Path = SPLITS_DIR) -> list[str]:
    bad = []
    for line in (splits_dir / "MANIFEST.sha256").read_text().splitlines():
        sha, name = line.split("  ")
        if hashlib.sha256((splits_dir / name).read_bytes()).hexdigest() != sha:
            bad.append(name)
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    loco = make_loco(pd.read_parquet(ROOT / "data/processed/universe_v1.parquet"))
    assert loco["n_families"] == N_FAMILIES_V1, loco["n_families"]
    b = dumps(loco)
    line = f"{hashlib.sha256(b).hexdigest()}  {NAME}\n"
    man = SPLITS_DIR / "MANIFEST.sha256"
    if a.check:
        assert (SPLITS_DIR / NAME).read_bytes() == b, "loco_v1.json differs from regeneration"
        assert line in man.read_text(), "manifest line missing"
    else:
        before = verify_manifest()
        assert not before, f"existing manifest entries do not verify: {before}"
        (SPLITS_DIR / NAME).write_bytes(b)
        text = man.read_text()
        if NAME not in text:
            man.write_text(text + line)
    bad = verify_manifest()
    assert not bad, bad
    print(f"{NAME}: {loco['n_families']} families, {loco['n_hosts']} hosts; folds "
          + ", ".join(f"{f['fold']}: test {f['n_test']} / train {f['n_train']} ({len(f['families'])} families)"
                      for f in loco["folds"]))
    print("manifest: every entry verifies")


if __name__ == "__main__":
    main()
