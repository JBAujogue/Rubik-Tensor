import pytest
from typing import Iterable

import torch

from rubik.action import (
    POS_ROTATIONS,
)


def test_position_rotation_shape():
    expected = (3, 4, 4)
    observed = POS_ROTATIONS.shape
    assert expected == observed, f"Position rotation tensor expected shape '{expected}', but got '{observed}' instead"


@pytest.mark.parametrize(
    "axis, input, expected",
    [
        (0, (1, 1, 0, 0), (1, 1, 0, 0)),  # X -> X
        (0, (1, 0, 1, 0), (1, 0, 0, -1)),  # Y -> -Z
        (0, (1, 0, 0, 1), (1, 0, 1, 0)),  # Z -> Y
        (1, (1, 1, 0, 0), (1, 0, 0, 1)),  # X -> Z
        (1, (1, 0, 1, 0), (1, 0, 1, 0)),  # Y -> Y
        (1, (1, 0, 0, 1), (1, -1, 0, 0)),  # Z -> -X
        (2, (1, 1, 0, 0), (1, 0, -1, 0)),  # X -> -Y
        (2, (1, 0, 1, 0), (1, 1, 0, 0)),  # Y -> X
        (2, (1, 0, 0, 1), (1, 0, 0, 1)),  # Z -> Z
    ],
)
def test_position_rotation(axis: int, input: Iterable[int], expected: Iterable[int]):
    out = POS_ROTATIONS[axis] @ torch.tensor(input, dtype=POS_ROTATIONS.dtype)
    exp = torch.tensor(expected, dtype=POS_ROTATIONS.dtype)
    assert torch.equal(out, exp), f"Position rotation tensor is incorrect along axis {axis}: {out} != {exp}"
