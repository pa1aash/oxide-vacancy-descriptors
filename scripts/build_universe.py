"""Build the universe and persist the parquet, the tracked IDs CSV and its sha256."""
from __future__ import annotations

import json
import sys
import warnings

from dftgnn.config import load_config
from dftgnn.data.universe import build_universe, write_universe

warnings.filterwarnings("ignore")


def main() -> None:
    cfg = load_config()
    uni, cas = build_universe(cfg, return_cascade=True)
    out = write_universe(uni, cfg)
    for s in cas.steps:
        print(f"{s['step']:<10} sites={s['sites']:<5} hosts={s['hosts']:<4} "
              f"formulas={s['formulas']:<4} removed={s['n_removed']}")
    print(json.dumps({h: v["outcome"] for h, v in cas.d3.items()}))
    print({k: str(v) for k, v in out.items()})
    sys.stdout.flush()


if __name__ == "__main__":
    main()
