"""Extended outer resamples r = 10..29.

Physics.

Design, unchanged from ``dftgnn.split.outer_split``: 20% of the 818 hosts to test, stratified by host-mean E_f
quintile with largest-remainder allocation; the remaining 654-host pool in a random order whose first B hosts
form the budget-B training set (nested budgets). The outer seed of resample r is r, continuing
``split.seeds`` = [0, ..., 9]; the budget-order seed is ``derived_seed(r, "budget_order")`` as before.

Written once by ``scripts/make_splits_ext.py`` to ``splits/resamples_10_29.json``; read with
``dftgnn.split.load_split(f"outer_r{r}")``.
"""
from __future__ import annotations

import pandas as pd

from dftgnn.config import Config
from dftgnn.split import coverage, host_table, make_all, outer_split

EXT_NAME = "resamples_10_29"
EXT_FILE = f"{EXT_NAME}.json"
EXT_RESAMPLES = range(10, 30)
SEED_RULE = "outer seed = r (continues split.seeds = [0..9]); budget order seed sha256('<r>|budget_order')[:8]"


def ext_seed(r: int) -> int:
    return r


def check_seed_scheme(cfg: Config) -> None:
    """Check seed scheme."""
    if list(cfg.split.seeds) != list(range(cfg.split.n_outer_resamples)):
        raise ValueError(f"split.seeds {cfg.split.seeds} is not [0..n-1]; seed = r would not continue it")


def make_ext(universe: pd.DataFrame, cfg: Config) -> dict:
    """The ``resamples_10_29`` object. Pure function of (universe, config)."""
    check_seed_scheme(cfg)
    hosts = host_table(universe)
    res = {str(r): outer_split(hosts, r, ext_seed(r), cfg.split.test_fraction) for r in EXT_RESAMPLES}
    registered = make_all(universe, cfg)
    all30 = [registered[f"outer_r{r}"] for r in range(cfg.split.n_outer_resamples)] + list(res.values())
    first = res[str(EXT_RESAMPLES[0])]
    return {"name": EXT_NAME, "label": "EXTENSION", "resamples": res,
            "seed_rule": SEED_RULE, "test_fraction": cfg.split.test_fraction,
            "n_test_hosts": len(first["test"]), "pool_size": len(first["budget_order"]),
            "budgets": list(cfg.budgets.hosts),
            "coverage_registered_plus_extension": coverage(all30, list(hosts.host_id))}
