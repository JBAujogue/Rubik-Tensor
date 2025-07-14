import pytest
from typing import Iterable

import torch

from rubik.action import (
    POS_ROTATIONS,
    POS_SHIFTS,
    FACE_ROTATIONS,
    build_actions_tensor,
)


def test_position_rotation_shape():
    """
    Test that POS_ROTATIONS has expected shape.
    """
    expected = (3, 4, 4)
    observed = POS_ROTATIONS.shape
    assert expected == observed, f"Position rotation tensor expected shape '{expected}', got '{observed}' instead"


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
    """
    Test that POS_ROTATIONS behaves as expected.
    """
    out = POS_ROTATIONS[axis] @ torch.tensor(input, dtype=POS_ROTATIONS.dtype)
    exp = torch.tensor(expected, dtype=POS_ROTATIONS.dtype)
    assert torch.equal(out, exp), f"Position rotation tensor is incorrect along axis {axis}: {out} != {exp}"


@pytest.mark.parametrize(
    "axis, size, input, expected",
    [
        (0, 3, (1, 1, 1, 1), (1, 1, 1, 0)),
        (1, 3, (1, 1, 1, 1), (1, 0, 1, 1)),
        (2, 3, (1, 1, 1, 1), (1, 1, 0, 1)),
    ],
)
def test_position_shift(axis: int, size: int, input: Iterable[int], expected: Iterable[int]):
    """
    Test that POS_SHIFTS behaves as expected.
    """
    rot = POS_ROTATIONS[axis] @ (torch.tensor(input, dtype=POS_ROTATIONS.dtype) * (size - 1))
    out = rot + (POS_SHIFTS[axis] * (size - 1))
    exp = torch.tensor(expected, dtype=POS_ROTATIONS.dtype) * (size - 1)
    assert torch.equal(out, exp), f"Position shift tensor is incorrect along axis {axis}: {out} != {exp}"


def test_face_rotation_shape():
    """
    Test that FACE_ROTATIONS has expected shape.
    """
    expected = (3, 6, 6)
    observed = FACE_ROTATIONS.shape
    assert expected == observed, f"Face rotation tensor expected shape '{expected}', got '{observed}' instead"


@pytest.mark.parametrize(
    "axis, input, expected",
    [
        (0, (1, 0, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0)),  # rotation about X axis: 0 (Up) -> 2 (Front)
        (1, (1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0)),  # rotation about Y axis: 0 (Up) -> 1 (Left)
        (2, (0, 1, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0)),  # rotation about Z axis: 1 (Left) -> 2 (Front)
    ],
)
def test_face_rotation(axis: int, input: Iterable[int], expected: Iterable[int]):
    """
    Test that POS_ROTATIONS behaves as expected.
    """
    out = torch.tensor(input, dtype=FACE_ROTATIONS.dtype) @ FACE_ROTATIONS[axis]
    exp = torch.tensor(expected, dtype=FACE_ROTATIONS.dtype)
    assert torch.equal(out, exp), f"Face rotation tensor is incorrect along axis {axis}: {out} != {exp}"


@pytest.mark.parametrize("size", [2, 3, 5, 20])
def test_build_actions_tensor_shape(size: int):
    """
    Test that "build_actions_tensor" output has expected shape.
    """
    expected = (3, size, 2, 6 * (size**2), 6 * (size**2))
    observed = build_actions_tensor(size).shape
    assert expected == observed, (
        f"'build_actions_tensor' output has incorrect shape: expected shape '{expected}', got '{observed}' instead"
    )
