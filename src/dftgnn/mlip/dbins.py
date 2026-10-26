"""Cation d-electron bins of the hosts.

Physics. The geometry sensitivity of a vacancy energy plausibly depends on whether the cations near the
vacancy have open d shells (strong ligand-field and Jahn-Teller coupling to distortions) or closed ones.
Each host is binned by the d-shell occupancy of its cations in their formal oxidation states.

Oxidation states: pymatgen ``BVAnalyzer`` on the DFT unit cell first, then the first
``Composition.oxi_state_guesses`` solution; if both fail the host is "unassigned". Cations are the species
with a positive oxidation state.

d count of a cation of element X in state q:
    groups 3-12 (transition metals, group-3 metals included): min(10, group(X) - q); the d shell holds at
        most 10 electrons, so a low-valent group-12 cation such as mercurous Hg+ (5d10 6s1) counts d10
    p-block with a filled (n-1)d shell (period >= 4, groups 13-17): 10, whatever q (Ga3+, In3+, Sn2+/4+,
        Pb2+/4+, Bi3+, Sb3+/5+, Tl+/3+, Ge4+, As5+, Te4+/6+, Se4+/6+ ...)
    s-block, early p-block (periods 2-3) and the f-block (lanthanides and actinides): 0
Host bin: "d0" if no cation has d electrons; "d10" if at least one cation is d10 and none is partially
filled; "other" if any cation has a d count strictly between 0 and 10. A d count outside [0, 10] (an
unphysical state assignment) makes the host "unassigned" with the reason recorded.
"""
from __future__ import annotations

import warnings


def d_count(symbol: str, q: float) -> float:
    from pymatgen.core import Element

    el = Element(symbol)
    if el.is_lanthanoid or el.is_actinoid:
        return 0.0
    g = el.group
    if g in (1, 2):
        return 0.0
    if 3 <= g <= 12:
        return float(min(10.0, g - q))
    if 13 <= g <= 17:
        return 10.0 if el.row >= 4 else 0.0
    raise ValueError(f"{symbol} (group {g}) is not expected as a cation")


def oxidation_states(structure) -> tuple[dict[str, list[float]], str]:
    """({element: sorted distinct states of its sites}, method) or ({}, "unassigned")."""
    from pymatgen.analysis.bond_valence import BVAnalyzer

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            vals = BVAnalyzer().get_valences(structure)
            out: dict[str, set] = {}
            for site, v in zip(structure, vals, strict=True):
                out.setdefault(site.specie.symbol, set()).add(float(v))
            return {k: sorted(v) for k, v in out.items()}, "bvanalyzer"
        except ValueError:
            pass
        guesses = structure.composition.oxi_state_guesses()
    if guesses:
        return {k: [float(v)] for k, v in guesses[0].items()}, "oxi_state_guesses"
    return {}, "unassigned"


def host_bin(structure) -> dict:
    """Row of the d-bin table for one host's DFT unit cell."""
    states, method = oxidation_states(structure)
    if method == "unassigned":
        return {"oxidation_method": method, "cation_states": "", "cation_d_counts": "", "bin": "unassigned",
                "reason": "BVAnalyzer and oxi_state_guesses both failed"}
    cations = {el: [q for q in qs if q > 0] for el, qs in states.items()}
    cations = {el: qs for el, qs in cations.items() if qs}
    dc = {el: [d_count(el, q) for q in qs] for el, qs in cations.items()}
    flat = [d for ds in dc.values() for d in ds]
    fmt = lambda x: f"{x:+g}"
    row = {"oxidation_method": method,
           "cation_states": ";".join(f"{el}:{'/'.join(fmt(q) for q in qs)}" for el, qs in sorted(cations.items())),
           "cation_d_counts": ";".join(f"{el}:{'/'.join(f'{d:g}' for d in ds)}" for el, ds in sorted(dc.items())),
           "reason": ""}
    if not flat:
        return {**row, "bin": "unassigned", "reason": "no cation with a positive oxidation state"}
    bad = [d for d in flat if d < -1e-9 or d > 10 + 1e-9]
    if bad:
        return {**row, "bin": "unassigned", "reason": f"d count outside [0, 10]: {bad}"}
    if any(1e-9 < d < 10 - 1e-9 for d in flat):
        return {**row, "bin": "other"}
    if any(abs(d - 10) <= 1e-9 for d in flat):
        return {**row, "bin": "d10"}
    return {**row, "bin": "d0"}
