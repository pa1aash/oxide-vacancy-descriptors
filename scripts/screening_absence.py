"""Why screening candidates are absent from universe v1, read from the release.

For each candidate: the release host directories with the same reduced formula (exact directory names,
never a glob), every completed neutral entry with its PHS value (``is_shallow``), its defect-type flags and
whether it is in Kumagai's neutral ML subset (``charge0.csv``), and the universe v1 sites. Release-wide
context for the PHS note: PHS rate of the completed neutral entries by cation element and by host band gap
(``bulk_visual_data.band_gap``, the hybrid gap). Counts only; writes ``results/screening_absence.json``.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from pymatgen.core import Composition

from dftgnn.config import load_config
from dftgnn.data import universe as U
from dftgnn.io.results import write_result

CATIONS = ("Ti", "V", "Nb", "Ta", "Mo", "W", "Zr", "Hf", "Zn", "Ga", "In", "Sn", "Cd")
GAP_BINS = [0, 1, 2, 3, 4, 5, 20]


def reason(entries: pd.DataFrame, in_universe: int) -> str:
    if entries.empty:
        return "not in the release"
    if in_universe:
        return "present in universe v1"
    if entries.phs.all():
        return "PHS-flagged (every neutral entry)"
    if (entries.release_flags != "").all():
        return "other flag"
    return "not in the ML subset"


def main() -> None:
    cfg = load_config()
    paths = U.release_paths(cfg)
    entries, hosts = U.release_neutral(cfg)
    entries["phs"] = entries.is_shallow.map(lambda x: x is True)
    ml = set(pd.read_csv(paths.ml_csv, index_col=0).full_name)
    entries["in_ml_subset"] = entries.full_name.isin(ml)
    uni = U.read_universe(cfg)
    reduced = {f: Composition(f).reduced_formula for f in hosts.formula}

    cands = []
    for c in cfg.screening.candidates:
        rc = Composition(c).reduced_formula
        dirs = sorted(f for f, r in reduced.items() if r == rc)
        e = entries[entries.formula.isin(dirs)]
        n_uni = int((uni.reduced_formula == rc).sum())
        cands.append({
            "candidate": c, "release_host_dirs": dirs, "universe_v1_sites": n_uni,
            "neutral_entries": [{"full_name": r.full_name, "release_Ef_eV": r.release_Ef_eV,
                                 "phs": bool(r.phs), "flags": r.release_flags,
                                 "in_ml_subset": bool(r.in_ml_subset)} for r in e.itertuples()],
            "reason": reason(e, n_uni),
        })

    gaps = {f: json.loads(paths.host_json(f, "bulk").read_text())["band_gap"] for f in hosts.formula}
    entries["gap_eV"] = entries.formula.map(gaps)
    elems = entries.formula.map(lambda f: {el.symbol for el in Composition(f)})
    by_cation = {}
    for el in CATIONS:
        m = elems.map(lambda s, el=el: el in s)
        by_cation[el] = {"entries_with": int(m.sum()), "phs_with": int(entries.phs[m].sum()),
                         "phs_rate_with": float(entries.phs[m].mean()),
                         "phs_rate_without": float(entries.phs[~m].mean())}
    gb = entries.groupby(pd.cut(entries.gap_eV, GAP_BINS), observed=False).phs
    by_gap = [{"gap_bin_eV": str(k), "entries": int(n), "phs": int(s)}
              for (k, n), s in zip(gb.size().items(), gb.sum(), strict=True)]
    payload = {
        "candidates": cands,
        "release_neutral_completed": {"entries": len(entries), "phs": int(entries.phs.sum())},
        "phs_by_cation": by_cation,
        "phs_by_hybrid_gap": by_gap,
        "median_hybrid_gap_eV": {"phs": float(np.median(entries.gap_eV[entries.phs])),
                                 "not_phs": float(np.median(entries.gap_eV[~entries.phs]))},
    }
    print(write_result("screening_absence", payload, config=cfg))
    for c in cands:
        print(f"{c['candidate']:8s} {c['reason']}")


if __name__ == "__main__":
    main()
