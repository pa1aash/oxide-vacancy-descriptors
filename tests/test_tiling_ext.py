"""One-operation tiling rule (EXTENSION, clarification of 2026-10-11): synthetic inversion and two-fold cases match,
a mirror image (two operations away) still fails, and the 797 translation-only hosts keep their map entries."""
from __future__ import annotations

import io
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dftgnn.mlip import tiling

ROOT = Path(__file__).resolve().parents[1]
PRE_EXT = "e6d1a84"     # main before the extension: the translation-only summary
FRAC = np.array([[0.11, 0.23, 0.37], [0.61, 0.17, 0.83], [0.29, 0.71, 0.53], [0.83, 0.59, 0.07]])
SPECIES = ["Zn", "O", "O", "Ti"]
LAT = np.diag([3.1, 4.3, 5.7])          # orthorhombic: its metric-preserving operations are the diagonal +-1 ones
M = np.diag([2, 1, 1])


def structures(R: np.ndarray, shuffle_seed: int = 0):
    from pymatgen.core import Lattice, Structure

    uc = Structure(Lattice(LAT), SPECIES, FRAC)
    frac_sc = tiling.tile(tiling.apply_op(FRAC, R), M)
    sp = tiling.tiled_species(SPECIES, M)
    order = np.random.default_rng(shuffle_seed).permutation(len(sp))
    shift = np.array([0.013, -0.021, 0.034])
    sc = Structure(Lattice(M @ LAT), [sp[i] for i in order], (frac_sc[order] + shift) % 1.0)
    return uc, sc


def test_point_ops_are_inversion_and_twofold_rotations():
    ops = tiling.point_ops()
    assert np.array_equal(ops[0], -np.eye(3, dtype=int))
    for R in ops[1:]:
        assert round(np.linalg.det(R)) == 1 and np.array_equal(R @ R, np.eye(3, dtype=int))
    assert not any(np.array_equal(R, np.eye(3, dtype=int)) for R in ops)


def test_translation_only_match_is_unchanged():
    uc, sc = structures(np.eye(3, dtype=int))
    rec = tiling.map_host(uc, sc)
    assert rec["status"] == "ok" and "point_op" not in rec and rec["max_residual_A"] < 1e-6


@pytest.mark.parametrize("R", [-np.eye(3, dtype=int), np.diag([-1, -1, 1])], ids=["inversion", "twofold"])
def test_one_operation_recovers_the_map(R):
    uc, sc = structures(R)
    assert tiling._map_host(uc, sc)["status"] == "failed"            # translation alone fails
    rec = tiling.map_host(uc, sc)
    assert rec["status"] == "ok" and rec["max_residual_A"] < 1e-6
    assert np.array_equal(np.array(rec["point_op"]), R)
    rebuilt = tiling.tiled_supercell(uc, rec["M"], rec["perm"], np.array(sc.lattice.matrix), point_op=rec["point_op"])
    d = np.array(rebuilt.frac_coords) + np.array(rec["translation"]) - np.array(sc.frac_coords)
    assert np.abs(d - np.round(d)).max() < 1e-6
    assert [s.specie.symbol for s in rebuilt] == [s.specie.symbol for s in sc]


def test_mirror_image_still_fails():
    uc, sc = structures(np.diag([-1, 1, 1]))                          # a mirror: inversion times a two-fold rotation
    assert tiling.map_host(uc, sc)["status"] == "failed"


def test_translation_only_hosts_byte_identical():
    """Every host that was ok before the extension has the same summary entry, field for field."""
    try:
        old = subprocess.run(["git", "show", f"{PRE_EXT}:data/tiling_summary_v1.csv"], cwd=ROOT, check=True,
                             capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("pre-extension commit not available")
    old = pd.read_csv(io.StringIO(old), dtype=str, keep_default_na=False)
    new = pd.read_csv(ROOT / "data" / "tiling_summary_v1.csv", dtype=str, keep_default_na=False)
    ok = old[old.status == "ok"].set_index("host_id")
    assert len(ok) == 797
    cur = new.set_index("host_id").loc[ok.index, ok.columns.drop("host_id", errors="ignore")]
    pd.testing.assert_frame_equal(cur, ok[cur.columns])
    assert (new.set_index("host_id").loc[ok.index, "point_op"] == "").all()


def test_map_parquet_entries_identical_for_translation_hosts():
    pre = ROOT / ".local" / "tiling_map_v1.pre_ext.parquet"
    cur = ROOT / "data" / "processed" / "tiling_map_v1.parquet"
    if not pre.is_file() or not cur.is_file():
        pytest.skip("pre-extension map copy not present")
    a = pd.read_parquet(pre).set_index("host_id")
    b = pd.read_parquet(cur).set_index("host_id")
    ok = a.index[a.status == "ok"]
    pd.testing.assert_frame_equal(b.loc[ok, a.columns], a.loc[ok])
    assert b.loc[ok, "point_op"].isna().all()
