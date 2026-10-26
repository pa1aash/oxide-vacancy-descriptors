"""Leave-one-chemistry-out folds.

Physics. A chemistry family is the set of distinct periodic-table groups of a host's non-oxygen elements.
Holding out whole families asks whether the models transfer to cation chemistries they never saw, rather than
to new hosts of familiar chemistry.

Folds: scikit-learn ``GroupKFold(n_splits=5)`` without shuffling, over the hosts sorted by host_id with the
family key as the group label. The training set of a fold is every host outside its test fold. The
family key reproduces ``scripts/audit_misc.py`` (``group_of`` and ``families``) element for element; a test
checks this on every host of universe v1.

Produced once by ``scripts/make_loco_splits.py`` as ``splits/loco_v1.json``; read thereafter with
``dftgnn.split.load_split("loco_v1")``.
"""
from __future__ import annotations

import pandas as pd

N_FOLDS = 5
N_FAMILIES_V1 = 131
KEY_DEFINITION = ("S02 definition ii-a: sorted tuple of the distinct periodic-table groups of the host's "
                  "non-oxygen elements, lanthanides and actinides counted as group 3, written as groups "
                  "joined by '-' (e.g. '2-14' for Sr2SnO4)")


def group_of(sym: str) -> int:
    from pymatgen.core import Element

    el = Element(sym)
    return 3 if el.is_lanthanoid or el.is_actinoid else el.group


def family_key(formula: str) -> str:
    from pymatgen.core import Composition

    els = sorted(e.symbol for e in Composition(formula).elements if e.symbol != "O")
    return "-".join(str(g) for g in sorted({group_of(e) for e in els}))


def families(universe: pd.DataFrame) -> pd.DataFrame:
    """host_id, formula, family (sorted by host_id)."""
    h = (universe.groupby("host_id").formula.first().reset_index().sort_values("host_id")
         .reset_index(drop=True))
    h["family"] = h.formula.map(family_key)
    return h


def make_loco(universe: pd.DataFrame, n_splits: int = N_FOLDS) -> dict:
    from sklearn.model_selection import GroupKFold

    h = families(universe)
    hosts = list(h.host_id)
    folds, fam2fold = [], {}
    for k, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(h.host_id, groups=h.family)):
        test, train = sorted(hosts[i] for i in te), sorted(hosts[i] for i in tr)
        fams = sorted(set(h.family.iloc[te]))
        for f in fams:
            fam2fold[f] = k
        folds.append({"fold": k, "test": test, "train": train, "n_train": len(train), "n_test": len(test),
                      "families": fams})
    return {"name": "loco_v1", "n_folds": n_splits, "n_hosts": len(hosts), "n_families": int(h.family.nunique()),
            "family_key_definition": KEY_DEFINITION,
            "fold_rule": "sklearn.model_selection.GroupKFold(n_splits=5), no shuffling, hosts sorted by host_id",
            "folds": folds, "family_to_fold": dict(sorted(fam2fold.items())),
            "host_family": dict(zip(h.host_id, h.family, strict=True))}
