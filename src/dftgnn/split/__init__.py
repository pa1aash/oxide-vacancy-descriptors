"""Split engine: grouped random hold-outs with nested training budgets.

Design. Groups are hosts (``host_id``); formula grouping is identical because the universe has no polymorphs.

Outer resample r (r = 0..n-1, seed ``config.split.seeds[r]``): ``round(test_fraction * n_hosts)``
hosts go to test, stratified by the quintile of the host-mean target so the test target
distribution is not degenerate. The remaining hosts form the training pool. The pool is put in a
random order drawn from a seed derived from ``(r, "budget_order")``; the training set of budget B is
the first B hosts of that order, so budgets are nested (B_25 within B_50 within ...). If the pool is
smaller than a requested budget, the budget is capped at the pool size and the cap is recorded.

Splits are produced once by ``scripts/make_splits.py`` and read from ``splits/`` thereafter. Nothing
else in the code base may regenerate them.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dftgnn.config import Config, load_config

REPO_ROOT = Path(__file__).resolve().parents[3]
SPLITS_DIR = REPO_ROOT / "splits"
N_STRATA = 5


def derived_seed(r: int, tag: str) -> int:
    """Deterministic 64-bit seed from ``(r, tag)``; independent of Python's salted ``hash``."""
    return int.from_bytes(hashlib.sha256(f"{r}|{tag}".encode()).digest()[:8], "big")


def host_table(universe: pd.DataFrame) -> pd.DataFrame:
    """One row per host: host_id, formula, n_sites, mean target; sorted by host_id."""
    if universe.groupby("host_id").formula.nunique().max() != 1:
        raise ValueError("a host_id maps to more than one formula")
    t = (universe.groupby("host_id")
         .agg(formula=("formula", "first"), n_sites=("site_id", "size"),
              mean_Ef=("target_Ef_eV", "mean"))
         .reset_index().sort_values("host_id").reset_index(drop=True))
    if t.formula.nunique() != len(t):
        raise ValueError("formula grouping differs from host grouping")
    return t


def _allocate(sizes: np.ndarray, total: int) -> np.ndarray:
    """Largest-remainder allocation of ``total`` test hosts over strata proportional to size."""
    exact = sizes * total / sizes.sum()
    base = np.floor(exact).astype(int)
    order = np.argsort(-(exact - base), kind="stable")
    base[order[: total - base.sum()]] += 1
    return base


def outer_split(hosts: pd.DataFrame, r: int, seed: int, test_fraction: float) -> dict:
    """Stratified grouped hold-out of resample ``r``. ``hosts`` comes from ``host_table``."""
    ranked = hosts.sort_values(["mean_Ef", "host_id"], kind="stable")
    strata = np.array_split(ranked.host_id.to_numpy(), N_STRATA)
    n_test = round(test_fraction * len(hosts))
    alloc = _allocate(np.array([len(s) for s in strata], float), n_test)
    rng = np.random.default_rng(seed)
    test = sorted(h for s, k in zip(strata, alloc, strict=True)
                  for h in rng.choice(s, size=k, replace=False))
    pool = sorted(set(hosts.host_id) - set(test))
    order_rng = np.random.default_rng(derived_seed(r, "budget_order"))
    order = [pool[i] for i in order_rng.permutation(len(pool))]
    return {"resample": r, "seed": int(seed), "test": test, "budget_order": order,
            "stratum_test_counts": [int(k) for k in alloc]}


def budget_train(split: dict, budget: int) -> list[str]:
    """Training hosts of ``budget`` (capped at the pool size): first ``budget`` of the order."""
    return split["budget_order"][: min(budget, len(split["budget_order"]))]


def effective_budgets(split: dict, budgets: list[int]) -> list[int]:
    return [min(b, len(split["budget_order"])) for b in budgets]


def val_split(train_hosts, frac: float = 0.1, min_hosts: int = 3, seed: int = 0):
    """Carve a validation set out of ``train_hosts`` (for GNN training; baselines do not call it).

    n_val = max(min_hosts, round(frac * n)); hosts are drawn by a seeded permutation of the sorted
    host ids. Returns (train, val) as sorted lists. Raises if training would be left empty.
    """
    hosts = sorted(train_hosts)
    n_val = max(min_hosts, round(frac * len(hosts)))
    if n_val >= len(hosts):
        raise ValueError(f"cannot carve {n_val} validation hosts out of {len(hosts)}")
    perm = np.random.default_rng(seed).permutation(len(hosts))
    val = sorted(hosts[i] for i in perm[:n_val])
    vs = set(val)
    return [h for h in hosts if h not in vs], val


def kiyohara_hosts(universe: pd.DataFrame) -> dict[str, list[str]]:
    """Kiyohara et al.'s released train/val/test host lists mapped onto the universe (host_id)."""
    h = universe.groupby("host_id").kiyohara_split.first()
    if (h == "absent").any():
        raise ValueError("universe hosts absent from the Kiyohara lists")
    return {sp: sorted(h.index[h == sp]) for sp in ("train", "val", "test")}


def curve_splits(hosts: pd.DataFrame, cfg: Config) -> dict:
    """Kumagai-style learning-curve resamples (not stratified, as in K21).

    Resample i: one random permutation (seed from ``(i, "rf_curve")``) of all hosts; the first
    ``n_test_hosts`` are test, the next ``max(sizes)`` are the nested training order. Stored as
    indices into the sorted ``hosts`` list.
    """
    ids = list(hosts.host_id)
    c = cfg.curve
    n_train = max(c.sizes)
    res = []
    for i in range(c.n_resamples):
        perm = np.random.default_rng(derived_seed(i, "rf_curve")).permutation(len(ids))
        res.append({"i": i, "test": sorted(int(k) for k in perm[: c.n_test_hosts]),
                    "order": [int(k) for k in perm[c.n_test_hosts: c.n_test_hosts + n_train]]})
    return {"hosts": ids, "sizes": c.sizes, "n_test_hosts": c.n_test_hosts, "resamples": res}


def curve_resample(curve: dict, i: int, size: int) -> tuple[list[str], list[str]]:
    """(train host_ids, test host_ids) of curve resample ``i`` at training ``size``."""
    r = curve["resamples"][i]
    ids = curve["hosts"]
    return [ids[k] for k in r["order"][:size]], [ids[k] for k in r["test"]]


def make_all(universe: pd.DataFrame, cfg: Config | None = None) -> dict[str, dict]:
    """Every split object, keyed by file stem. Pure function of (universe, config)."""
    cfg = cfg if cfg is not None else load_config()
    hosts = host_table(universe)
    out = {}
    for r in range(cfg.split.n_outer_resamples):
        out[f"outer_r{r}"] = outer_split(hosts, r, cfg.split.seeds[r], cfg.split.test_fraction)
    out["kiyohara"] = kiyohara_hosts(universe)
    out["rf_curve"] = curve_splits(hosts, cfg)
    return out


def dumps(obj: dict) -> bytes:
    """Compact, key-sorted, newline-terminated JSON (byte-stable)."""
    return (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode()


def load_split(name: str, splits_dir: Path | None = None) -> dict:
    """Load ``splits/<name>.json`` (``outer_r<r>`` or ``kiyohara``). Never regenerates.

    The EXTENSION resamples ``outer_r10`` .. ``outer_r29`` have no file of their own; they are read from
    ``splits/resamples_10_29.json`` (``dftgnn.split.ext``).
    """
    d = splits_dir or SPLITS_DIR
    f = d / f"{name}.json"
    if not f.is_file() and name.startswith("outer_r") and name.removeprefix("outer_r").isdigit():
        from dftgnn.split.ext import EXT_FILE, EXT_RESAMPLES

        r = int(name.removeprefix("outer_r"))
        if r in EXT_RESAMPLES:
            return json.loads((d / EXT_FILE).read_text())["resamples"][str(r)]
    return json.loads(f.read_text())


def load_meta(splits_dir: Path | None = None) -> dict:
    return json.loads(((splits_dir or SPLITS_DIR) / "meta.json").read_text())


def coverage(splits: list[dict], all_hosts: list[str]) -> dict:
    """How often each host is in test across outer resamples."""
    cnt = {h: 0 for h in all_hosts}
    for s in splits:
        for h in s["test"]:
            cnt[h] += 1
    vals = np.array(list(cnt.values()))
    return {"n_hosts": len(all_hosts), "n_resamples": len(splits),
            "never_in_test": int((vals == 0).sum()), "in_test_at_least_once": int((vals > 0).sum()),
            "histogram": {int(k): int((vals == k).sum()) for k in range(len(splits) + 1)},
            "expected_never_in_test_if_independent": float(
                len(all_hosts) * (1 - len(splits[0]["test"]) / len(all_hosts)) ** len(splits))}
