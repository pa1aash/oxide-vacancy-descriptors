"""Extended-budget grouped K-fold cross-validation.

Physics. The outer resamples cap training at 654 hosts and leave 79 hosts never tested. K-fold CV over the 818
hosts trains on 736-778 hosts and tests every host exactly once per K, so the pooled out-of-fold predictions
place every host's formation energy from a model that never saw it.

Folds (group = host = formula), for each K in {10, 20}: rank the hosts by (host-mean E_f, host_id) and cut the
ranking into five quintile groups with ``numpy.array_split`` (the strata of ``dftgnn.split.outer_split``); a fresh
``numpy.random.default_rng(20261010)`` permutes each group in ascending-E_f order; the permuted groups are
concatenated and host i of that order goes to fold i mod K. Fold sizes therefore differ by at most one, and every
fold draws evenly from the E_f range. The training set of a fold is every host outside it.

The split name of fold f of K is ``cv_v1_k{K}_f{f}`` (``RunSpec.r`` = f, ``RunSpec.budget`` = the fold's training
size). The validation carve-out is ``val_split`` with r replaced by the string f"cv{K}f{f}" (``val_r``).

Produced once by ``scripts/make_splits_ext.py`` as ``splits/cv_v1.json``; read with ``load_split("cv_v1")``.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from dftgnn.split import N_STRATA, host_table

CV_NAME = "cv_v1"
CV_FILE = f"{CV_NAME}.json"
CV_PREFIX = "cv_v1_k"
CV_KS = (10, 20)
CV_SEED = 20261010
FOLD_RULE = ("rank hosts by (host-mean E_f, host_id); five quintile groups by numpy.array_split; per K a fresh "
             "numpy.random.default_rng(20261010) permutes each group in ascending-E_f order; host i of the "
             "concatenated order goes to fold i mod K")
_NAME = re.compile(r"cv_v1_k(\d+)_f(\d+)")


def split_name(k: int, fold: int) -> str:
    return f"{CV_PREFIX}{k}_f{fold}"


def parse_name(name: str) -> tuple[int, int]:
    """``cv_v1_k{K}_f{fold}`` -> (K, fold)."""
    m = _NAME.fullmatch(name)
    if not m:
        raise ValueError(f"not a CV split name: {name!r}")
    return int(m[1]), int(m[2])


def val_r(k: int, fold: int) -> str:
    """The string that replaces r in the validation seed of fold ``fold`` of K."""
    return f"cv{k}f{fold}"


def assign_folds(hosts: pd.DataFrame, k: int, seed: int = CV_SEED) -> dict[str, int]:
    """host_id -> fold for one K. ``hosts`` comes from ``host_table``."""
    ranked = hosts.sort_values(["mean_Ef", "host_id"], kind="stable")
    groups = np.array_split(ranked.host_id.to_numpy(), N_STRATA)
    rng = np.random.default_rng(seed)
    order = [h for g in groups for h in g[rng.permutation(len(g))]]
    return {str(h): i % k for i, h in enumerate(order)}


def make_cv(universe: pd.DataFrame, ks=CV_KS) -> dict:
    """The ``cv_v1`` object. Pure function of the universe."""
    hosts = host_table(universe)
    ids = list(hosts.host_id)
    out = {}
    for k in ks:
        f_of = assign_folds(hosts, k)
        folds = []
        for f in range(k):
            test = sorted(h for h in ids if f_of[h] == f)
            train = sorted(h for h in ids if f_of[h] != f)
            folds.append({"fold": f, "split": split_name(k, f), "test": test, "train": train,
                          "n_test": len(test), "n_train": len(train)})
        out[str(k)] = {"k": k, "folds": folds, "fold_of_host": dict(sorted(f_of.items()))}
    return {"name": CV_NAME, "label": "EXTENSION", "n_hosts": len(ids), "group_key": "host_id (== formula)",
            "seed": CV_SEED, "fold_rule": FOLD_RULE, "ks": list(ks), "by_k": out}


def fold(cv: dict, k: int, f: int) -> dict:
    return cv["by_k"][str(k)]["folds"][f]
