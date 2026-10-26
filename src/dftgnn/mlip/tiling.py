"""Map a host's DFT unit cell onto its release supercell.

Physics. Condition (ii) puts the MLIP-relaxed unit cell into the defect supercell geometry. That needs
the integer matrix M with L_sc = M L_uc (row-vector lattices) and a one-to-one map from the tiled unit-cell
atoms to the release supercell atoms, so the vacancy atom index of the release keeps its meaning.

Procedure, per host:
1. M = L_sc L_uc^-1. If every entry is within ``REL_TOL`` (1e-3) x max(1, max|round(M)|) of an integer,
   the frames are aligned. Otherwise, if the release transformation matrix T (``cell_info.txt``) satisfies
   (T L_uc)(T L_uc)^T = L_sc L_sc^T to the same relative tolerance, the two frames differ by a rigid
   rotation and T is used (matching is done in fractional coordinates, which a rotation leaves unchanged).
   Otherwise the host fails with reason ``non_integer_matrix``.
2. Tiling (``tile``) is deterministic and independent of pymatgen: the |det M| lattice points n of the
   supercell are enumerated in lexicographic order, and tiled atom k = a * |det M| + j is unit-cell atom a
   shifted by lattice point j; its fractional coordinate in the supercell basis is (x_a + n_j) M^-1.
3. Matching allows one global translation. Candidates translate one tiled atom of the rarest element onto
   the first release atom of that element; for each, every release atom is matched to the nearest tiled
   atom of the same element by minimum-image distance in the release supercell metric. A candidate is valid
   if the match is a bijection. The mean displacement is then removed and the maximum residual (A)
   recomputed. The candidate with the smallest maximum residual wins.
4. A host is ``ok`` if a bijection exists and its maximum residual is <= ``RESID_TOL_A`` (1e-3 A); else it is
   flagged ``excluded_from_C3b`` (it stays in every other analysis).
5. One lattice point operation. Only if steps 2-4
   fail, exactly one operation R of the unit cell is tried: inversion (-I) or a two-fold rotation (entries in
   {-1, 0, 1}, det +1, R R = I, R != I), each only if it preserves the unit-cell metric (R G R^T = G, G = L L^T, to
   ``REL_TOL``). R acts on the unit-cell fractional coordinates (x -> x R mod 1) before tiling, and steps 3-4 apply
   unchanged. The passing candidate with the smallest residual is kept (ties: inversion, then rotations in
   lexicographic order) and recorded as ``point_op``; translation-only hosts have ``point_op`` None and an
   unchanged record otherwise. The MLIP-relaxed cell is transformed by the same R before tiling (``tiled_supercell``).

``perm[i]`` is the tiled index of release atom i, so ``tiled_frac[perm]`` is in release order.
"""
from __future__ import annotations

import itertools
import re

import numpy as np

REL_TOL = 1e-3
RESID_TOL_A = 1e-3


def lattice_points(M: np.ndarray) -> np.ndarray:
    """The |det M| integer vectors n with n M^-1 in [0, 1)^3, lexicographically sorted."""
    Minv = np.linalg.inv(M)
    corners = np.array(list(itertools.product((0, 1), repeat=3))) @ M
    lo, hi = np.floor(corners.min(0)).astype(int), np.ceil(corners.max(0)).astype(int)
    grid = np.array(list(itertools.product(*(range(a, b + 1) for a, b in zip(lo, hi, strict=True)))))
    f = grid @ Minv
    keep = np.all((f > -1e-8) & (f < 1 - 1e-8), axis=1)
    pts = grid[keep]
    n = round(abs(np.linalg.det(M)))
    if len(pts) != n:
        raise ValueError(f"found {len(pts)} lattice points, expected |det M| = {n}")
    return pts[np.lexsort(pts.T[::-1])]


def tile(frac_uc: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Supercell fractional coordinates of the tiled unit cell (atom-major order, see module docstring)."""
    pts = lattice_points(M)
    Minv = np.linalg.inv(M)
    out = (frac_uc[:, None, :] + pts[None, :, :]).reshape(-1, 3) @ Minv
    return out - np.floor(out + 1e-12)


def tiled_species(species_uc: list, M: np.ndarray) -> list:
    n = round(abs(np.linalg.det(M)))
    return [s for s in species_uc for _ in range(n)]


def parse_transformation(cell_info: str) -> np.ndarray | None:
    m = re.search(r"Transformation matrix:\s*\[([^\]]*)\]\s*\[([^\]]*)\]\s*\[([^\]]*)\]", cell_info)
    if not m:
        return None
    return np.array([[float(x) for x in row.replace(",", " ").split()] for row in m.groups()])


def supercell_matrix(L_uc: np.ndarray, L_sc: np.ndarray, T: np.ndarray | None) -> tuple[np.ndarray | None, dict]:
    M = L_sc @ np.linalg.inv(L_uc)
    Mi = np.round(M)
    dev = float(np.abs(M - Mi).max())
    info = {"M_direct_max_dev": dev, "T_release": None if T is None else T.astype(int).tolist()}
    if dev <= REL_TOL * max(1.0, float(np.abs(Mi).max())) and abs(np.linalg.det(Mi)) > 0.5:
        info.update(frame="aligned", T_agrees=None if T is None else bool(np.array_equal(Mi, T)))
        return Mi.astype(int), info
    if T is not None:
        A = T @ L_uc
        g1, g2 = A @ A.T, L_sc @ L_sc.T
        if np.abs(g1 - g2).max() <= REL_TOL * np.abs(g2).max():
            info.update(frame="rotated", T_agrees=True)
            return T.astype(int), info
    info.update(frame=None, T_agrees=False)
    return None, info


def _min_image(df: np.ndarray) -> np.ndarray:
    return df - np.round(df)


def match(frac_tiled: np.ndarray, species_tiled: list, frac_sc: np.ndarray, species_sc: list,
          L_sc: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    """(perm, translation in supercell fractions, max residual A); perm is None if no bijection exists."""
    st, ss = np.array(species_tiled), np.array(species_sc)
    if sorted(st.tolist()) != sorted(ss.tolist()):
        return None, None, float("inf")
    els, counts = np.unique(ss, return_counts=True)
    rare = els[np.argmin(counts)]
    anchor = np.nonzero(ss == rare)[0][0]
    groups = {e: (np.nonzero(st == e)[0], np.nonzero(ss == e)[0]) for e in els}
    best = (None, None, float("inf"))
    for cand in np.nonzero(st == rare)[0]:
        t = frac_sc[anchor] - frac_tiled[cand]
        perm = np.full(len(ss), -1)
        for ti, si in groups.values():
            d = _min_image(frac_sc[si][:, None, :] - (frac_tiled[ti][None, :, :] + t)) @ L_sc
            nn = np.argmin(np.einsum("ijk,ijk->ij", d, d), axis=1)
            perm[si] = ti[nn]
        if len(np.unique(perm)) != len(perm):
            continue
        disp = _min_image(frac_sc - (frac_tiled[perm] + t))
        t2 = t + disp.mean(0)
        resid = np.linalg.norm(_min_image(frac_sc - (frac_tiled[perm] + t2)) @ L_sc, axis=1).max()
        if resid < best[2]:
            best = (perm, t2, float(resid))
            if resid < 1e-4:
                break
    return best


def point_ops() -> list[np.ndarray]:
    """Inversion first, then every two-fold rotation with entries in {-1, 0, 1}, in lexicographic order."""
    rots = []
    for v in itertools.product((-1, 0, 1), repeat=9):
        R = np.array(v).reshape(3, 3)
        if round(np.linalg.det(R)) == 1 and np.array_equal(R @ R, np.eye(3, dtype=int)) \
                and not np.array_equal(R, np.eye(3, dtype=int)):
            rots.append(R)
    return [-np.eye(3, dtype=int), *rots]


def preserves_metric(R: np.ndarray, L: np.ndarray) -> bool:
    G = L @ L.T
    return bool(np.abs(R @ G @ R.T - G).max() <= REL_TOL * np.abs(G).max())


def apply_op(frac: np.ndarray, R) -> np.ndarray:
    """x -> x R modulo 1 (fractional coordinates of the unit cell)."""
    out = np.asarray(frac) @ np.asarray(R)
    return out - np.floor(out + 1e-12)


def map_host(uc, sc, cell_info: str | None = None) -> dict:
    """Tiling record of one host (see module docstring); one point operation is tried if translation fails."""
    rec = _map_host(uc, sc, cell_info)
    if rec["status"] == "ok" or rec["M"] is None or str(rec["reason"]).startswith("atom count"):
        return rec
    L_uc = np.array(uc.lattice.matrix)
    best = None
    for R in point_ops():
        if not preserves_metric(R, L_uc):
            continue
        r = _map_host(uc, sc, cell_info, op=R)
        if r["status"] == "ok" and (best is None or r["max_residual_A"] < best["max_residual_A"]):
            best = r
    return best if best is not None else rec


def _map_host(uc, sc, cell_info: str | None = None, op: np.ndarray | None = None) -> dict:
    """Tiling record of one host: status, M, permutation, translation, residual (see module docstring)."""
    L_uc, L_sc = np.array(uc.lattice.matrix), np.array(sc.lattice.matrix)
    T = parse_transformation(cell_info) if cell_info else None
    M, info = supercell_matrix(L_uc, L_sc, T)
    rec = {"n_uc": len(uc), "n_sc": len(sc), **info}
    if M is None:
        return {**rec, "status": "failed", "reason": "non_integer_matrix", "M": None, "perm": None,
                "translation": None, "max_residual_A": None}
    rec["M"] = M.tolist()
    if round(abs(np.linalg.det(M))) * len(uc) != len(sc):
        return {**rec, "status": "failed", "reason": f"atom count {len(uc)} x |det M| != {len(sc)}", "perm": None,
                "translation": None, "max_residual_A": None}
    frac = np.array(uc.frac_coords) if op is None else apply_op(uc.frac_coords, op)
    ft = tile(frac, M)
    sp = tiled_species([s.specie.symbol for s in uc], M)
    perm, t, resid = match(ft, sp, np.array(sc.frac_coords), [s.specie.symbol for s in sc], L_sc)
    if op is not None:
        rec["point_op"] = np.asarray(op).astype(int).tolist()
    if perm is None:
        return {**rec, "status": "failed", "reason": "no_bijection", "perm": None, "translation": None,
                "max_residual_A": None}
    status, reason = ("ok", "") if resid <= RESID_TOL_A else ("failed", "residual_above_tolerance")
    return {**rec, "status": status, "reason": reason, "perm": perm.tolist(), "translation": t.tolist(),
            "max_residual_A": resid}


def tiled_supercell(uc, M, perm: list[int], lattice_sc: np.ndarray | None = None, point_op=None):
    """pymatgen Structure of ``uc`` tiled by M with atoms in release order; lattice M L_uc unless given.

    ``point_op`` (the host's recorded operation R, or None) is applied to the fractional coordinates of ``uc`` first.
    """
    from pymatgen.core import Lattice, Structure

    M = np.array(M)
    frac0 = np.array(uc.frac_coords) if point_op is None else apply_op(uc.frac_coords, point_op)
    frac = tile(frac0, M)[np.array(perm)]
    sp = np.array(tiled_species([s.specie.symbol for s in uc], M))[np.array(perm)]
    lat = M @ np.array(uc.lattice.matrix) if lattice_sc is None else lattice_sc
    return Structure(Lattice(lat), sp.tolist(), frac)
