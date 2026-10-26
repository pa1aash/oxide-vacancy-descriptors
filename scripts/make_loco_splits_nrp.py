"""Leave-chemistry-out splits.

    python scripts/make_loco_splits.py            write splits/loco/loco_f<k>.json and splits/loco/MANIFEST.sha256
    python scripts/make_loco_splits.py --verify   recompute and compare with the tracked files

Family of a host: definition ii-a, the set of distinct periodic-table groups of its non-O elements
(lanthanides and actinides in group 3), as in ``scripts/audit_misc.py``. The elements come from the host's
graph in the verified store, so the split is a pure function of universe v1. Folds: scikit-learn
``GroupKFold(n_splits=robustness.loco.k)`` over the families (deterministic; no shuffling). Fold k holds out
its families as test hosts; every other host is a training host (``training_size: max_feasible``), in sorted
order as ``budget_order``, so ``budget_train(split, len(budget_order))`` returns them all. The pinned
``splits/MANIFEST.sha256`` is not touched; these files have their own manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dftgnn.config import load_config
from dftgnn.graphs import store as G
from dftgnn.split import SPLITS_DIR, dumps

OUT = SPLITS_DIR / "loco"


def group_of(z: int) -> int:
    from pymatgen.core import Element

    el = Element.from_Z(z)
    return 3 if el.is_lanthanoid or el.is_actinoid else el.group


def families(graphs: list[dict], host_ids: list[str]) -> dict[str, str]:
    out = {}
    for h, g in zip(host_ids, graphs, strict=True):
        zs = sorted({int(z) for z in g["z"].tolist()} - {8})
        out[h] = "-".join(str(x) for x in sorted({group_of(z) for z in zs}))
    return out


def make() -> dict[str, bytes]:
    from sklearn.model_selection import GroupKFold

    cfg = load_config()
    graphs, _, meta = G.load_store()
    hosts = meta["host_ids"]
    fam = families(graphs, hosts)
    groups = np.array([fam[h] for h in hosts])
    files = {}
    for k, (tr, te) in enumerate(GroupKFold(n_splits=cfg.robustness.loco.k).split(hosts, groups=groups)):
        test = sorted(hosts[i] for i in te)
        train = sorted(hosts[i] for i in tr)
        files[f"loco_f{k}.json"] = dumps({
            "fold": k, "grouping": cfg.robustness.loco.grouping, "n_families_total": len(set(groups)),
            "test": test, "budget_order": train,
            "test_families": sorted(set(groups[te].tolist())), "n_test_families": len(set(groups[te]))})
    return files


def manifest(files: dict[str, bytes]) -> bytes:
    return "".join(f"{hashlib.sha256(b).hexdigest()}  {n}\n" for n, b in sorted(files.items())).encode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    files = make()
    if a.verify:
        bad = [n for n, b in files.items() if (OUT / n).read_bytes() != b]
        bad += [] if (OUT / "MANIFEST.sha256").read_bytes() == manifest(files) else ["MANIFEST.sha256"]
        print("loco splits:", "OK" if not bad else bad)
        raise SystemExit(bool(bad))
    OUT.mkdir(exist_ok=True)
    for n, b in files.items():
        (OUT / n).write_bytes(b)
    (OUT / "MANIFEST.sha256").write_bytes(manifest(files))
    import json

    for n, b in sorted(files.items()):
        d = json.loads(b)
        print(n, "test hosts", len(d["test"]), "train hosts", len(d["budget_order"]),
              "test families", d["n_test_families"], "of", d["n_families_total"])


if __name__ == "__main__":
    main()
