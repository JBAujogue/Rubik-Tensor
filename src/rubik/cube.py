from dataclasses import dataclass

import torch


@dataclass
class Cube:
    """
    A 5D tensor filled with 0 or 1. Dimensions have the following interpretation:
        - X coordinate (from 0 to self.size - 1, from Left to Right).
        - Y coordinate (from 0 to self.size - 1, from Back to Front).
        - Z coordinate (from 0 to self.size - 1, from Down to Up).
        - Face (from 0 to 5, with 0 = "Up", 1 = "Left", 2 = "Front", 3 = "Right", 4 = "Back", 5 = "Down").
        - Color (from 0 to 6, 0 being the "dark" color, the rest according to order given in "colors" attribute).
    """

    state: torch.Tensor
    colors: list[str]
    size: int

    @classmethod
    def from_default(cls, colors: list[str], size: int) -> "Cube":
        """
        Create Cube from a given list of 6 colors and size.
        Example:
            cube = Cube.from_default(['U', 'L', 'C', 'R', 'B', 'D'], size = 3)
        """
        assert (num := len(set(colors))) == 6, f"Expected 6 distinct colors, got {num}"
        assert isinstance(size, int) and size > 1, f"Expected non-zero integrer size, got {size}"

        # build tensor filled with 0's, and fill the faces with 1's
        n = size - 1
        state = torch.zeros([size, size, size, 6, 7], dtype=torch.int8)
        state[:, :, n, 0, 1] = 1  # up
        state[0, :, :, 1, 2] = 1  # left
        state[:, 0, :, 2, 3] = 1  # front
        state[n, :, :, 3, 4] = 1  # right
        state[:, n, :, 4, 5] = 1  # back
        state[:, :, 0, 5, 6] = 1  # down
        return cls(state, colors, size)

    @staticmethod
    def pad_colors(colors: list[str]) -> list[str]:
        """
        Pad color names to strings of equal length.
        """
        max_len = max(len(c) for c in colors)
        return [c + " " * (max_len - len(c)) for c in colors]

    def to_grid(self, pad_colors: bool = False) -> list[list[list[str]]]:
        """
        Convert Cube into a 3D grid representation.
        """
        n = self.size - 1
        colors = self.pad_colors(self.colors) if pad_colors else self.colors
        grid = [
            self.state[:, :, n, 0, :].argmax(dim=-1),  # up
            self.state[0, :, :, 1, :].argmax(dim=-1),  # left
            self.state[:, 0, :, 2, :].argmax(dim=-1),  # front
            self.state[n, :, :, 3, :].argmax(dim=-1),  # right
            self.state[:, n, :, 4, :].argmax(dim=-1),  # back
            self.state[:, :, 0, 5, :].argmax(dim=-1),  # down
        ]
        return [[[colors[i - 1] for i in row] for row in face.tolist()] for face in grid]

    def __str__(self):
        """
        Compute a string representation of a cube.
        Example:
            cube = Cube.from_default(['U', 'L', 'C', 'R', 'B', 'D'], size = 3)
            print(cube)
            #     UUU
            #     UUU
            #     UUU
            # LLL CCC RRR BBB
            # LLL CCC RRR BBB
            # LLL CCC RRR BBB
            #     DDD
            #     DDD
            #     DDD
        """
        grid = self.to_grid(pad_colors=True)
        void = " " * max(len(c) for c in self.colors) * self.size
        l1 = "\n".join(" ".join([void, "".join(row), void, void]) for row in grid[0])
        l2 = "\n".join(" ".join("".join(grid[face_i][row_i]) for face_i in range(1, 5)) for row_i in range(self.size))
        l3 = "\n".join(" ".join((void, "".join(row), void, void)) for row in grid[-1])
        return "\n".join([l1, l2, l3])
