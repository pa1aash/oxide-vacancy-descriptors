"""The rule that sets max_epochs and patience."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("set_epochs", ROOT / "scripts" / "set_epochs.py")
set_epochs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(set_epochs)


def test_e_conv_is_first_epoch_within_one_percent_of_the_minimum():
    val = [1.0, 0.8, 0.6, 0.5051, 0.5049, 0.5, 0.52]
    assert set_epochs.e_conv(val) == 5            # 0.5051 > 1.01 x 0.5 = 0.505; 0.5049 <= 0.505
    assert set_epochs.e_conv([1.0, 0.5, 0.505, 0.5]) == 2
    assert set_epochs.e_conv([0.7, 0.7, 0.7]) == 1
    assert set_epochs.e_conv([0.9, 0.6, 0.5]) == 3


@pytest.mark.parametrize(("conv", "expect"), [
    ({25: 40, 200: 120, 654: 300}, (450, 68)),    # 1.5 x 300 = 450; 0.15 x 450 = 67.5 -> 68
    ({25: 10, 200: 20, 654: 60}, (100, 30)),      # 90 -> 100; 15 < 30 -> 30
    ({25: 1, 200: 1, 654: 1}, (50, 30)),          # 1.5 -> 50
    ({25: 200, 200: 100, 654: 150}, (300, 45)),   # 300 is already a multiple of 50
    ({25: 400, 200: 350, 654: 300}, (600, 90)),
])
def test_rule(conv, expect):
    assert set_epochs.rule(conv) == expect


def test_round_half_up():
    assert [set_epochs.round_half_up(x) for x in (7.5, 37.5, 67.5, 22.4)] == [8, 38, 68, 22]
