"""Known-answer tests of the MLIP geometry metrics and conditions on synthetic structures."""
import numpy as np
import pytest

pytest.importorskip("pymatgen")
from pymatgen.core import Lattice, Structure
from scipy.spatial.transform import Rotation

from dftgnn.mlip import geometry as G

L0 = np.array(Lattice.from_parameters(4.1, 4.6, 5.3, 84, 97, 103).matrix)
F0 = np.array([[0.0, 0.0, 0.0], [0.5, 0.4, 0.5], [0.3, 0.2, 0.1], [0.7, 0.6, 0.45], [0.1, 0.8, 0.75]])


def test_isotropic_scaling():
    for s in (0.97, 1.0, 1.035):
        m = G.metrics(L0, F0, s * L0, F0)
        assert m["eps_v"] == pytest.approx(s - 1, abs=1e-12)
        assert m["deviatoric"] < 1e-12 and m["internal_rmsd_A"] < 1e-12


def test_pure_shear():
    g = 0.04
    Fd = np.eye(3) + g * np.outer([1, 0, 0], [0, 1, 0])        # F = I + g e1 e2^T, det 1
    L1 = L0 @ Fd.T                                            # rows transform as L F^T
    m = G.metrics(L0, F0, L1, F0)
    assert abs(m["eps_v"]) < 1e-12 and m["deviatoric"] > 1e-3 and m["internal_rmsd_A"] < 1e-12
    _, _, U = G.deformation(L0, L1)
    assert np.allclose(U, U.T) and np.all(np.linalg.eigvalsh(U) > 0)


def test_rotation_and_translation_invariance():
    Q = Rotation.from_euler("zyx", [31, -17, 58], degrees=True).as_matrix()
    Fd = np.diag([1.02, 0.99, 1.01])
    L1 = L0 @ Fd.T
    rng = np.random.default_rng(0)
    f1 = F0 + rng.normal(0, 0.01, F0.shape)
    base = G.metrics(L0, F0, L1, f1)
    rot = G.metrics(L0, F0, L1 @ Q.T, f1)                     # rigidly rotated MLIP cell
    shifted = G.metrics(L0, F0, L1, f1 + [0.23, -0.41, 0.97])  # rigid translation (incl. wrap)
    for k in ("eps_v", "deviatoric", "internal_rmsd_A"):
        assert rot[k] == pytest.approx(base[k], abs=1e-12)
        assert shifted[k] == pytest.approx(base[k], abs=1e-12)
    assert rot["rotation_angle_deg"] > 1


def test_internal_rmsd_known_value_and_permutation():
    d = np.zeros_like(F0)
    d[2] = [0.01, 0, 0]                                       # one atom moved along a
    m = G.metrics(L0, F0, L0, F0 + d)
    disp = (d - d.mean(0)) @ L0
    assert m["internal_rmsd_A"] == pytest.approx(np.sqrt((disp ** 2).sum(1).mean()), rel=1e-12)
    perm = np.array([1, 0, 2, 3, 4])                          # wrong correspondence is not invariant
    assert G.metrics(L0, F0, L0, (F0 + d)[perm])["internal_rmsd_A"] > 10 * m["internal_rmsd_A"]


def _struct(L, f, sp=("Zn", "Zn", "O", "O", "O")):
    return Structure(Lattice(L), list(sp), f)


def test_host_metrics_species_order_checked():
    with pytest.raises(ValueError):
        G.host_metrics(_struct(L0, F0), _struct(L0, F0, ("O", "Zn", "Zn", "O", "O")))


def test_conditions_volume_and_vacancy():
    dft = _struct(L0 * 2.5, F0)
    mlip = _struct(L0 @ np.diag([1.03, 0.98, 1.05]).T * 2.5, F0 + 0.002)
    cs = G.condition_structures(dft, mlip)
    assert abs(cs["mlip_rescaled"].volume - dft.volume) / dft.volume < 1e-10
    assert np.allclose(cs["mlip_rescaled"].frac_coords, cs["mlip"].frac_coords)
    gs = G.condition_graphs(dft, mlip, [2, 3])
    assert set(gs) == {"mlip", "mlip_rescaled"} and int(gs["mlip"]["z"][2]) == 8
    with pytest.raises(AssertionError):
        G.condition_graphs(dft, mlip, [0])
