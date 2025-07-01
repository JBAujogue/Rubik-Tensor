from dataclasses import dataclass

import torch


@dataclass
class Cube:
    """
    A 4D tensor filled with colors. Dimensions have the following interpretation:
        - Face (from 0 to 5, with 0 = "Up", 1 = "Left", 2 = "Front", 3 = "Right", 4 = "Back", 5 = "Down").
        - X coordinate (from 0 to self.size - 1, from Left to Right).
        - Y coordinate (from 0 to self.size - 1, from Back to Front).
        - Z coordinate (from 0 to self.size - 1, from Down to Up).

    Colors filling each tensor cell are from 0 to 6, 0 being the "dark" color,
    the rest according to order given in "colors" attribute.
    """

    coordinates: torch.Tensor
    state: torch.Tensor
    colors: list[str]
    size: int

    @classmethod
    def create(cls, colors: list[str], size: int) -> "Cube":
        """
        Create Cube from a given list of 6 colors and size.
        Example:
            cube = Cube.create(['U', 'L', 'C', 'R', 'B', 'D'], size = 3)
        """
        assert (num := len(set(colors))) == 6, f"Expected 6 distinct colors, got {num}"
        assert isinstance(size, int) and size > 1, f"Expected non-zero integrer size, got {size}"

        # build dense tensor filled with colors
        n = size - 1
        tensor = torch.zeros([6, size, size, size], dtype=torch.int8)
        tensor[0, :, :, n] = 1  # up
        tensor[1, 0, :, :] = 2  # left
        tensor[2, :, n, :] = 3  # front
        tensor[3, n, :, :] = 4  # right
        tensor[4, :, 0, :] = 5  # back
        tensor[5, :, :, 0] = 6  # down
        return cls.from_sparse(tensor.to_sparse(), colors, size)

    def shuffle(self, num_moves: int):
        raise NotImplementedError

    def rotate(self, moves: str):
        raise NotImplementedError

    def solve(slef, policy: str):
        raise NotImplementedError

    @staticmethod
    def pad_colors(colors: list[str]) -> list[str]:
        """
        Pad color names to strings of equal length.
        """
        max_len = max(len(c) for c in colors)
        return [c + " " * (max_len - len(c)) for c in colors]

    @classmethod
    def from_sparse(cls, tensor: torch.Tensor, colors: list[str], size: int) -> "Cube":
        """
        Gather cube attributes into a torch sparse tensor.
        """
        coordinates = tensor.indices().transpose(0, 1).to(torch.int8)
        values = tensor.values()
        return cls(coordinates, values, colors, size)

    def to_sparse(self) -> torch.Tensor:
        """
        Gather cube attributes into a torch sparse tensor.
        """
        return torch.sparse_coo_tensor(
            indices=self.coordinates.transpose(0, 1),
            values=self.state,
            size=(6, self.size, self.size, self.size),
            dtype=torch.int8,
        )

    def __str__(self):
        """
        Compute a string representation of a cube.
        """
        colors = self.pad_colors(self.colors)
        faces = self.state.reshape(6, self.size, self.size).transpose(1, 2)
        faces = [[[colors[i - 1] for i in row] for row in face.tolist()] for face in faces]
        void = " " * max(len(c) for c in self.colors) * self.size
        l1 = "\n".join(" ".join([void, "".join(row), void, void]) for row in faces[0])
        l2 = "\n".join(" ".join("".join(face[i]) for face in faces[1:5]) for i in range(self.size))
        l3 = "\n".join(" ".join((void, "".join(row), void, void)) for row in faces[-1])
        return "\n".join([l1, l2, l3])
