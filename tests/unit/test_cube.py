import pytest

import torch

from rubik.cube import Cube


class TestCube:
    """
    A testing class for the Cube class.
    """

    @pytest.mark.parametrize(
        "colors, size",
        [
            [["U", "L", "C", "R", "B", "D"], 3],
            [["Up", "Left", "Center", "Right", "Back", "Down"], 5],
            [["A", "BB", "CCC", "DDDD", "EEEEE", "FFFFFF"], 10],
        ],
    )
    def test__init__(self, colors: list[str], size: int):
        """
        Test that the __init__ method produce expected attributes.
        """
        cube = Cube(colors, size)
        assert cube.coordinates.shape == (6 * (size**2), 4), (
            f"'coordinates' has incorrect shape {cube.coordinates.shape}"
        )
        assert cube.state.shape == (6 * (size**2), 7), f"'state' has incorrect shape {cube.state.shape}"
        assert len(cube.history) == 0, "'history' field should be empty"

    @pytest.mark.parametrize("device", ["cpu"])
    def test_to(self, device: str | torch.device):
        cube = Cube(colors=["U", "L", "C", "R", "B", "D"], size=3)
        cube_2 = cube.to(device)
        assert torch.equal(cube.state, cube_2.state), "cube has different state after calling 'to' method"

    def test_reset_history(self):
        cube = Cube(colors=["U", "L", "C", "R", "B", "D"], size=3)
        cube.rotate("X2 X1i Y1i Z1i Y0 Z0i X2 X1i Y1i Z1i Y0 Z0i")
        cube.reset_history()
        assert cube.history == [], "method 'reset_history' does not flush content"

    @pytest.mark.parametrize("num_moves, seed", [[50, 42]])
    def test_shuffle(self, num_moves: int, seed: int):
        cube = Cube(colors=["U", "L", "C", "R", "B", "D"], size=3)
        cube_state = cube.state.clone()
        cube.shuffle(num_moves, seed)
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
        cube = Cube(colors=["U", "L", "C", "R", "B", "D"], size=3)
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
        cube = Cube(colors=["U", "L", "C", "R", "B", "D"], size=3)
        cube_state = cube.state.clone()
        cube.rotate_once(axis, slice, inverse)
        assert cube.history == [[axis, slice, inverse]], "method 'rotate_once' does not update history"
        assert not torch.equal(cube_state, cube.state), "method 'rotate_once' does not change state"

    @pytest.mark.parametrize(
        "moves",
        [
            "X2 X1i Y1i",
            "X2 X1i Y1i Z1i Y0 Z0i X2 X1i Y1i Z1i Y0 Z0i " * 2,
        ],
    )
    def test_compute_changes(self, moves: str):
        cube = Cube(colors=["U", "L", "C", "R", "B", "D"], size=3)
        facets = cube.state.argmax(dim=-1).to(torch.int16).tolist()
        changes = cube.compute_changes(moves)

        # apply changes induced by moves using the permutation dict returned by 'compute_changes'
        expected = [facets[changes.get(i, i)] for i in range(len(facets))]

        # apply changes induced by moves using the optimized 'rotate' method
        cube.rotate(moves)
        observed = cube.state.argmax(dim=-1).to(torch.int16).tolist()

        # assert the tow are identical
        assert expected == observed, "method 'compute_changes' does not behave correctly: "

    def test__str__(self):
        cube = Cube(colors=["U", "L", "C", "R", "B", "D"], size=3)
        repr = str(cube)
        assert len(repr), "__str__ method returns an empty representation"
