import pytest

import torch

from rubik.state import build_cube_tensor, build_permutation_matrix


@pytest.mark.parametrize("size", [2, 3, 5, 20])
def test_build_cube_tensor(size: int):
    """
    Test that build_cube_tensor behaves as expected.
    """
    tensor = build_cube_tensor(size)
    facets = tensor.to_dense().to(dtype=torch.int8) != 0
    x_sums = facets.sum(dim=(0, 2, 3)).tolist()
    y_sums = facets.sum(dim=(0, 1, 3)).tolist()
    z_sums = facets.sum(dim=(0, 1, 2)).tolist()
    expected = [(size**2) + (4 * size)] + [4 * size] * (size - 2) + [(size**2) + (4 * size)]
    assert x_sums == expected, (
        f"'build_cube_tensor' has incorrect sum along X axis: expected '{expected}', got '{x_sums}'"
    )
    assert y_sums == expected, (
        f"'build_cube_tensor' has incorrect sum along Y axis: expected '{expected}', got '{y_sums}'"
    )
    assert z_sums == expected, (
        f"'build_cube_tensor' has incorrect sum along Z axis: expected '{expected}', got '{z_sums}'"
    )


@pytest.mark.parametrize("size, perm", [[2, "01"], [3, "210"], [6, "2345"]])
def test_build_permutation_matrix(size: int, perm: str):
    """
    Test that build_permutation_matrix behaves as expected.
    """
    matrix = build_permutation_matrix(size, perm)
    mapping = dict(matrix.indices().transpose(0, 1).tolist())
    for i, j in zip(perm, perm[1:] + perm[0]):
        assert mapping[int(i)] == int(j), f"'build_permutation_matrix' outputs has wrong behavior: {perm}, {mapping}"
