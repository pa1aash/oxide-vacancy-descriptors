"""Generate splits/ once: outer resamples, Kiyohara split, meta.json and the sha256 manifest.

Refuses to overwrite existing files unless --force is given.
"""
from __future__ import annotations

import argparse
import hashlib
import sys

from dftgnn.config import load_config
from dftgnn.data import universe as U
from dftgnn.split import SPLITS_DIR, coverage, dumps, host_table, make_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default=str(SPLITS_DIR))
    a = ap.parse_args()
    from pathlib import Path

    out = Path(a.out)
    cfg = load_config()
    uni = U.read_universe(cfg)
    ids_sha = hashlib.sha256(U.ids_csv_bytes(uni)).hexdigest()
    if ids_sha != hashlib.sha256(U.IDS_FILE.read_bytes()).hexdigest():
        sys.exit("universe does not match the tracked IDs file")
    if out.exists() and any(out.iterdir()) and not a.force:
        sys.exit(f"{out} is not empty; splits are generated once (use --force only to test)")
    out.mkdir(parents=True, exist_ok=True)
    splits = make_all(uni, cfg)
    hosts = host_table(uni)
    outer = [splits[f"outer_r{r}"] for r in range(cfg.split.n_outer_resamples)]
    pool = len(outer[0]["budget_order"])
    meta = {
        "universe": cfg.data.universe, "universe_ids_sha256": ids_sha,
        "n_hosts": len(hosts), "n_sites": len(uni), "group_key": "host_id (== formula)",
        "n_outer_resamples": len(outer), "seeds": cfg.split.seeds,
        "test_fraction": cfg.split.test_fraction, "n_test_hosts": len(outer[0]["test"]),
        "stratification": "host-mean E_f quintiles (rank-based, 5 strata), largest-remainder allocation",
        "pool_size": pool, "budgets_requested": cfg.budgets.hosts,
        "budgets_effective": [min(b, pool) for b in cfg.budgets.hosts],
        "b_max_config": cfg.budgets.max, "b_max_is_pool_size": pool == cfg.budgets.max,
        "budget_order_seed": "sha256('<r>|budget_order')[:8] as big-endian uint64",
        "nested": "budget-B training set = first B hosts of budget_order",
        "coverage": coverage(outer, list(hosts.host_id)),
        "kiyohara_hosts": {k: len(v) for k, v in splits["kiyohara"].items()},
        "curve": {"sizes": cfg.curve.sizes, "n_resamples": cfg.curve.n_resamples,
                  "n_test_hosts": cfg.curve.n_test_hosts, "file": "rf_curve.json",
                  "note": "Kumagai-style unstratified grouped resamples; train = first N of order"},
        "kiyohara_note": "released without-PHS lists mapped to universe_v1; all 818 hosts covered",
    }
    files = {f"{k}.json": dumps(v) for k, v in splits.items()}
    files["meta.json"] = dumps(meta)
    for name, b in files.items():
        (out / name).write_bytes(b)
    man = "".join(f"{hashlib.sha256(b).hexdigest()}  {n}\n" for n, b in sorted(files.items()))
    (out / "MANIFEST.sha256").write_text(man)
    print(man, end="")
    print("manifest sha256:", hashlib.sha256(man.encode()).hexdigest())
    print("coverage:", meta["coverage"])


if __name__ == "__main__":
    main()
