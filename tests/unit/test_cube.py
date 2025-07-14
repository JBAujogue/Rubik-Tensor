import pytest


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
