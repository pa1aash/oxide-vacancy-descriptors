"""B0 physics floor: E_f = a + b * (oxide stability) + c * (band gap), ordinary least squares.

Oxide stability (Kumagai et al. 2021, Sec. II I and III D): column ``formation_energy``, the host
formation energy per atom (eV/atom) from the Materials Project, empirically corrected. More negative
means a more stable oxide. Band gap: column ``band_gap``, the PBEsol(+U) gap of the pristine host (eV)
as released. Both are host properties; the descriptors of Deml et al. use the same two quantities.
"""
from __future__ import annotations

import numpy as np

STABILITY = "desc_formation_energy"
GAP = "desc_band_gap"
COLUMNS = (STABILITY, GAP)


def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares coefficients [a, b, c] for ``y ~ a + X @ [b, c]``."""
    A = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(A, y, rcond=None)[0]


def make_fit(data):
    X = data.uni[list(COLUMNS)].to_numpy(float)
    y = data.uni.target_Ef_eV.to_numpy(float)

    def fit(tr, te, seed):
        coef = ols(X[tr], y[tr])
        pred = coef[0] + X[te] @ coef[1:]
        return {"pred": pred, "info": {"coef": {"intercept_eV": float(coef[0]),
                                                "stability_eV_per_eV_per_atom": float(coef[1]),
                                                "gap_eV_per_eV": float(coef[2])}}}
    return fit


def interpret(stab: float, gap: float) -> dict[str, str]:
    """One physical sentence per slope, chosen by the fitted sign (no further interpretation)."""
    s = ("negative: a more stable oxide (more negative formation energy per atom) has a higher "
         "vacancy formation energy" if stab < 0 else
         "positive: a less stable oxide (less negative formation energy per atom) has a higher "
         "vacancy formation energy, opposite to the bond-strength expectation")
    g = ("positive: a wider-gap oxide goes with a costlier vacancy" if gap > 0 else
         "negative: a wider-gap oxide goes with a cheaper vacancy")
    return {"stability": s, "gap": g}
