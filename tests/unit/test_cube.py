import pytest

import torch

from rubik.cube import Cube


class TestCube:
    """
    A testing class for the Cube class.
    """

    @pytest.mark.parametrize("size", [3, 5, 10, 25])
    def test__init__(self, size: int):
        """
        Test that the __init__ method produce expected attributes.
        """
        cube = Cube(size)
        assert cube.state.shape == (6 * (size**2), 7), f"'state' has incorrect shape {cube.state.shape}"
        assert cube.actions.shape == (3, size, 2, cube.state.shape[0], cube.state.shape[0]), (
            f"'actions' has incorrect shape {cube.actions.shape}"
        )
        assert len(cube.history) == 0, "'history' field should be empty"

    @pytest.mark.parametrize("device", ["cpu"])
    def test_to(self, device: str | torch.device):
        """
        Test that the .to method behaves as expected.
        """
        cube = Cube(3)
        cube_2 = cube.to(device)
        assert torch.equal(cube.state, cube_2.state), "cube has different state after calling 'to' method"

    def test_reset_history(self):
        """
        Test that the .reset_history method behaves as expected.
        """
        cube = Cube(3)
        cube.rotate("X2 X1i Y1i Z1i Y0 Z0i X2 X1i Y1i Z1i Y0 Z0i")
        cube.reset_history()
        assert cube.history == [], "method 'reset_history' does not flush content"

    @pytest.mark.parametrize("num_moves, seed", [[50, 42]])
    def test_shuffle(self, num_moves: int, seed: int):
        """
        Test that the .shuffle method behaves as expected.
        """
        cube = Cube(3)
        cube_state = cube.state.clone()
        cube.scramble(num_moves, seed)
        assert cube.history == [], "method 'shuffle' does not flush content"
        assert not torch.equal(cube_state, cube.state), "method 'shuffle' does not change state"

    @pytest.mark.parametrize(
        "moves",
        [
            "X2 X1i Y1i Z1i Y0 Z0i X2 X1i Y1i Z1i Y0 Z0i",
            "X2 X1i Y1i Z1i Y0 Z0i X2 X1i Y1i Z1i Y0 Z0i" * 2,
        ],
    )
    def test_rotate(self, moves: str):
        """
        Test that the .rotate method behaves as expected.
        """
        cube = Cube(3)
        cube_state = cube.state.clone()
        cube.rotate(moves)
        assert cube.history != [], "method 'rotate' does not update history"
        assert not torch.equal(cube_state, cube.state), "method 'rotate' does not change state"

    @pytest.mark.parametrize(
        "axis, slice, inverse",
        [
            [0, 2, 0],
            [1, 1, 1],
            [2, 0, 0],
        ],
    )
    def test_rotate_once(self, axis: int, slice: int, inverse: int):
        """
        Test that the .rotate_once method behaves as expected.
        """
        cube = Cube(3)
        cube_state = cube.state.clone()
        cube.rotate_once(axis, slice, inverse)
        assert cube.history == [(axis, slice, inverse)], "method 'rotate_once' does not update history"
        assert not torch.equal(cube_state, cube.state), "method 'rotate_once' does not change state"

    @pytest.mark.parametrize(
        "moves",
        [
            "X2 X1i Y1i",
            "X2 X1i Y1i Z1i Y0 Z0i X2 X1i Y1i Z1i Y0 Z0i " * 2,
        ],
    )
    def test_compute_changes(self, moves: str):
        """
        Test that the .compute_changes method behaves as expected.
        """
        cube = Cube(3)
        facets = cube.state.argmax(dim=-1).to(cube.dtype).tolist()
        changes = cube.compute_changes(moves)

        # apply changes induced by moves using the permutation dict returned by 'compute_changes'
        expected = [facets[changes.get(i, i)] for i in range(len(facets))]

        # apply changes induced by moves using the optimized 'rotate' method
        cube.rotate(moves)
        observed = cube.state.argmax(dim=-1).to(cube.dtype).tolist()

        # assert the tow are identical
        assert expected == observed, "method 'compute_changes' does not behave correctly: "

    def test__str__len(self):
        """
        Test that the __str__ method behaves as expected.
        """
        cube = Cube(3)
        repr = str(cube)
        assert len(repr), "__str__ method returns an empty representation"

    @pytest.mark.parametrize("size", [3, 5, 8, 10])
    def test__str__content(self, size: int):
        """
        Test that stringify behaves as expected.
        """
        cube = Cube(size=size)
        repr = str(cube)
        lens = {len(line) for line in repr.split("\n")}
        assert len(lens) == 1, f"'stringify' lines have variable length: {lens}"
