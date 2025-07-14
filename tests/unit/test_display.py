import pytest

import torch

from rubik.cube import Cube
from rubik.display import stringify, pad_colors


@pytest.mark.parametrize(
    "colors, size",
    [
        [["U", "L", "C", "R", "B", "D"], 3],
        [["Up", "Left", "Center", "Right", "Back", "Down"], 5],
        [["A", "BB", "CCC", "DDDD", "EEEEE", "FFFFFF"], 10],
    ],
)
def test_stringify(colors: list[str], size: int):
    """
    Test that stringify behaves as expected.
    """
    cube = Cube(colors=colors, size=size)
    state = cube.state.argmax(dim=-1).to(device="cpu", dtype=torch.int16)
    repr = stringify(state, colors, size)
    lens = {len(line) for line in repr.split("\n")}
    assert len(lens) == 1, f"'stringify' lines have variable length: {lens}"


@pytest.mark.parametrize(
    "colors",
    [
        ["U", "L", "C", "R", "B", "D"],
        ["Up", "Left", "Center", "Right", "Back", "Down"],
        ["A", "BB", "CCC", "DDDD", "EEEEE", "FFFFFF"],
    ],
)
def test_pad_colors(colors: list[str]):
    """
    Test that pad_colors behaves as expected.
    """
    padded = pad_colors(colors)
    lengths = {len(color) for color in padded}
    assert len(lengths) == 1, f"'pad_colors' generates non-unique lengths: {lengths}"
