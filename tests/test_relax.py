"""Relaxation pipeline with ASE's EMT potential (no MACE download needed)."""
import json

import numpy as np
import pytest

pytest.importorskip("ase")
from ase.calculators.emt import EMT
from pymatgen.core import Lattice, Structure

from dftgnn.mlip import relax, tiling


def _cu(a=3.75):
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(a), ["Cu"], [[0, 0, 0]]).get_primitive_structure()


def test_relax_structure_converges_and_changes_volume():
    _, info = relax.relax_structure(_cu(), EMT(), fmax=0.01, steps=300)
    assert info["converged"] and info["final_fmax_eV_per_A"] <= 0.01
    assert info["final_volume_A3"] < info["initial_volume_A3"]   # 3.75 A is expanded for EMT Cu (~3.59 A)


def test_relax_host_tiles_and_is_idempotent(tmp_path):
    uc = _cu()
    M = np.diag([2, 2, 2])
    sc = uc.copy()
    sc.make_supercell(M)
    row = tiling.map_host(uc, sc)
    row = {**row, "M": json.dumps(row["M"]), "perm": json.dumps(row["perm"])}
    rec = relax.relax_host("mp-test", uc, row, EMT(), model_sha256="x", out_dir=tmp_path)
    assert rec["tiled"] and not rec["skipped"] and relax.record_ok("mp-test", tmp_path)
    big = Structure.from_dict(json.loads((tmp_path / "mp-test_sc.json").read_text()))
    small = Structure.from_dict(json.loads((tmp_path / "mp-test_uc.json").read_text()))
    assert len(big) == 8 and abs(big.volume - 8 * small.volume) < 1e-8
    again = relax.relax_host("mp-test", uc, row, EMT(), model_sha256="x", out_dir=tmp_path)
    assert again["skipped"]
    (tmp_path / "mp-test_uc.json").write_text("{}")
    assert not relax.record_ok("mp-test", tmp_path)


def test_excluded_host_gets_no_supercell(tmp_path):
    rec = relax.relax_host("mp-x", _cu(), {"status": "failed"}, EMT(), model_sha256="x", out_dir=tmp_path)
    assert not rec["tiled"] and rec["excluded_from_C3b"] and "supercell" not in rec["files"]


# ---- point operation read from a pandas row (nrp-code-v2 fix; deviations row of 2026-10-14)
FRAC = np.array([[0.11, 0.23, 0.37], [0.61, 0.17, 0.83], [0.29, 0.71, 0.53], [0.83, 0.59, 0.07]])
SPECIES = ["Zn", "O", "O", "Ti"]
LAT = np.diag([3.1, 4.3, 5.7])
M2 = np.diag([2, 1, 1])


def _host_pair(R):
    """(unit cell, release-like supercell) whose map needs the point operation R (identity: translation only)."""
    uc = Structure(Lattice(LAT), SPECIES, FRAC)
    frac_sc = tiling.tile(tiling.apply_op(FRAC, R), M2)
    sp = tiling.tiled_species(SPECIES, M2)
    order = np.random.default_rng(0).permutation(len(sp))
    sc = Structure(Lattice(M2 @ LAT), [sp[i] for i in order], (frac_sc[order] + [0.013, -0.021, 0.034]) % 1.0)
    return uc, sc


def _identity_relax(uc, calc, **kw):
    return uc.copy(), {"converged": True, "steps": 0}


@pytest.mark.parametrize("value", [float("nan"), np.float64("nan"), None, "", "null"])
def test_point_op_missing_means_identity(value):
    assert relax.point_op({"point_op": value}) is None
    assert relax.point_op({}) is None


def test_point_op_parses_recorded_operation():
    assert relax.point_op({"point_op": "[[-1, 0, 0], [0, -1, 0], [0, 0, -1]]"}) == (-np.eye(3, dtype=int)).tolist()


def test_relax_host_with_pandas_rows(tmp_path, monkeypatch):
    """A tiling map frame holds NaN for the translation-only host and a JSON R for the point-operation host;
    ``relax_host`` must tile the first with the identity and the second with R (nrp-code-v1 tiled with NaN)."""
    pd = pytest.importorskip("pandas")
    monkeypatch.setattr(relax, "relax_structure", _identity_relax)
    hosts = {"mp-trans": _host_pair(np.eye(3, dtype=int)), "mp-inv": _host_pair(-np.eye(3, dtype=int))}
    rows = []
    for h, (uc, sc) in hosts.items():
        rec = tiling.map_host(uc, sc)
        rows.append({"host_id": h, **{k: v for k, v in rec.items() if k != "point_op"},
                     **({"point_op": json.dumps(rec["point_op"])} if "point_op" in rec else {})})
    frame = pd.DataFrame(rows)
    for c in ("M", "perm"):
        frame[c] = frame[c].map(json.dumps)
    frame = frame.set_index("host_id")
    assert frame.loc["mp-trans", "point_op"] != frame.loc["mp-trans", "point_op"]       # NaN, as on the cluster
    for h, R in (("mp-trans", None), ("mp-inv", -np.eye(3, dtype=int))):
        uc, sc = hosts[h]
        row = frame.loc[h].to_dict()
        out = relax.relax_host(h, uc, row, None, model_sha256="x", out_dir=tmp_path / h)
        assert out["tiled"]
        big = Structure.from_dict(json.loads((tmp_path / h / f"{h}_sc.json").read_text()))
        assert np.isfinite(big.frac_coords).all()
        ref = tiling.tiled_supercell(uc, json.loads(row["M"]), json.loads(row["perm"]), point_op=R)
        assert json.dumps(big.as_dict()) == json.dumps(ref.as_dict())
        # in release order: every atom sits on its release atom up to the one global translation
        d = big.frac_coords - sc.frac_coords
        d -= np.round(d)
        d -= d.mean(0)
        assert [s.specie.symbol for s in big] == [s.specie.symbol for s in sc] and np.abs(d).max() < 1e-9


def test_translation_only_real_hosts_tile_as_before():
    """The 797 translation-only hosts of the tracked summary read as NaN (CSV) or null (parquet), map to the identity,
    and tile exactly as the translation-only path; the 21 point-operation hosts keep their R."""
    pd = pytest.importorskip("pandas")
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    summary = pd.read_csv(root / "data" / "tiling_summary_v1.csv")
    trans = summary[summary.point_op.isna()]
    assert len(trans) == 797 and len(summary) - len(trans) == 21
    assert all(relax.point_op(r) is None for r in trans.to_dict("records"))
    assert all(relax.point_op(r) is not None for r in summary[summary.point_op.notna()].to_dict("records"))
    mp = root / "data" / "processed" / "tiling_map_v1.parquet"
    cells = root / "data" / "processed" / "unit_cells_v1.json"
    if not (mp.is_file() and cells.is_file()):
        pytest.skip("tiling map or unit cells not present")
    from dftgnn.mlip import release

    tmap = pd.read_parquet(mp).set_index("host_id")
    ucs = release.unit_cells()
    assert all(relax.point_op(tmap.loc[h].to_dict()) is None for h in trans.host_id)
    for h in trans.host_id[::40]:                                       # 20 hosts spread over the table
        row = {**tmap.loc[h].to_dict(), "point_op": float("nan")}
        M, perm = json.loads(row["M"]), json.loads(row["perm"])
        a = tiling.tiled_supercell(ucs[h], M, perm, point_op=relax.point_op(row))
        b = tiling.tiled_supercell(ucs[h], M, perm)
        assert json.dumps(a.as_dict()) == json.dumps(b.as_dict())
