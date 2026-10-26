"""Hand-made compositions with known answers for analysis/dshell/classes.py."""
import pytest

from analysis.dshell import classes as dc


@pytest.mark.parametrize("formula, cls, flag", [
    ("ZnO", "d10", False),
    ("CdO", "d10", False),
    ("TiO2", "d0", False),
    ("MgO", "main_group", False),
    ("ZnTiO3", "mixed_f", False),     # d10 + d0
    ("La2O3", "mixed_f", False),      # f-block cation
    ("PdO", "partial_d", False),      # Pd2+ d8
    ("Ga2O3", "main_group", True),    # post-transition, filled 3d core
    ("SnO2", "main_group", True),
    ("Al2O3", "main_group", False),
])
def test_class_rule(formula, cls, flag):
    rec = dc.classify_host("x", formula)
    assert rec["class"] == cls
    assert rec["post_transition_d10_core"] is flag
    assert rec["reason"] is None


def test_d_counts_and_states():
    rec = dc.classify_host("x", "ZnTiO3")
    assert rec["cation_states"] == {"Ti": 4, "Zn": 2}
    assert rec["d_counts"] == {"Ti": 0, "Zn": 10}
    assert dc.classify_host("x", "PdO")["d_counts"] == {"Pd": 8}


def test_reduced_formula_used():
    assert dc.classify_host("x", "Zn2O2")["reduced_formula"] == "ZnO"


def test_unassigned_reasons():
    assert dc.classify_states(None)["reason"] == "no oxidation-state guess"
    r = dc.classify_states({"Fe": 8 / 3, "O": -2.0})
    assert r["class"] == "unassigned" and r["reason"].startswith("non-integer cation state")
    r = dc.classify_states({"Ti": 4.0, "O": -1.0})
    assert r["class"] == "unassigned" and "not -2" in r["reason"]
    r = dc.classify_states({"Zn": 13.0, "O": -2.0})
    assert r["class"] == "unassigned" and r["reason"].startswith("d count outside")


def test_mixed_d_partial_and_d0_is_mixed():
    assert dc.classify_states({"Ti": 4.0, "Ni": 2.0, "O": -2.0})["class"] == "mixed_f"
    assert dc.classify_states({"Ni": 2.0, "Cu": 2.0, "O": -2.0})["class"] == "partial_d"


def test_blocks():
    assert dc.block(30) == "d" and dc.block(31) == "main" and dc.block(57) == "f" and dc.block(72) == "d"
    assert dc.block(71) == "f" and dc.block(80) == "d" and dc.block(81) == "main" and dc.block(39) == "d"
