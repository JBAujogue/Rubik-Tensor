from functools import reduce
from loguru import logger

import torch

from rubik.action import build_actions_tensor, parse_actions_str, sample_actions_str
from rubik.state import build_cube_tensor


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

    def __init__(self, size: int):
        """
        Create Cube of a given size.
        """
        tensor = build_cube_tensor(size)

        self.dtype = torch.int64
        self.coordinates = tensor.indices()
        self.state = tensor.values()
        self.actions = build_actions_tensor(size)
        # internal-only attributes
        self._history: list[tuple[int, int, int]] = []
        self._colors: list[str] = list("ULCRBD")
        self._size: int = size

    @property
    def history(self) -> list[tuple[int, int, int]]:
        return self._history

    @property
    def colors(self) -> list[str]:
        return self._colors

    @property
    def size(self) -> int:
        return self._size

    @property
    def facelets(self) -> list[list[list[str]]]:
        """
        Return the list of faces of the cube, each given by a list of rows,
        each given by a list of facelets.
        """
        tensor = torch.sparse_coo_tensor(
            indices=self.coordinates,
            values=self.state,
            size=(6, self.size, self.size, self.size),
            dtype=self.dtype,
        ).to_dense()

        n = self.size - 1
        faces = [
            tensor[0, :, :, n].transpose(0, 1),  # up
            tensor[1, 0, :, :].flip(1).transpose(0, 1),  # left
            tensor[2, :, n, :].flip(1).transpose(0, 1),  # front
            tensor[3, n, :, :].flip(0).flip(1).transpose(0, 1),  # right
            tensor[4, :, 0, :].flip(0).flip(1).transpose(0, 1),  # back
            tensor[5, :, :, 0].flip(1).transpose(0, 1),  # down
        ]
        return [[[self.colors[i - 1] for i in row] for row in face.tolist()] for face in faces]

    def to(self, device: str | torch.device) -> "Cube":
        device = torch.device(device)
        dtype = self.dtype if device == torch.device("cpu") else torch.float32
        self.state = self.state.to(device=device, dtype=dtype)
        self.actions = self.actions.to(device=device, dtype=dtype)
        logger.info(f"Using device '{self.state.device}' and dtype '{dtype}'")
        return self

    def reset_history(self) -> None:
        """
        Reset internal history of moves.
        """
        self._history = []
        return

    def scramble(self, num_moves: int, seed: int = 0) -> None:
        """
        Randomly shuffle the cube by the supplied number of steps, and reset history of moves.
        """
        moves = sample_actions_str(num_moves, self.size, seed=seed)
        self.rotate(moves)
        self.reset_history()
        return

    def rotate(self, moves: str) -> None:
        """
        Apply a sequence of moves (defined as plain string) to the cube.
        """
        actions = parse_actions_str(moves)
        for action in actions:
            self.rotate_once(*action)
        return

    def rotate_once(self, axis: int, slice: int, inverse: int) -> None:
        """
        Apply a move (defined as 3 coordinates) to the cube.
        """
        action = self.actions[axis, slice, inverse]
        self.state = torch.gather(self.state, 0, action)
        self._history.append((axis, slice, inverse))
        return

    def compose_moves(self, moves: str) -> torch.Tensor:
        """
        combine a sequence of moves and return the resulting changes.
        """
        actions = parse_actions_str(moves)
        tensors = [self.actions[*action] for action in actions]
        return reduce(lambda A, B: torch.gather(A, 0, B), tensors)

    def __str__(self):
        """
        Compute a string representation of a cube.
        """
        space = " " * self.size
        facelets = self.facelets
        l1 = "\n".join(" ".join([space, "".join(row), space, space]) for row in facelets[0])
        l2 = "\n".join(" ".join("".join(face[i]) for face in facelets[1:5]) for i in range(self.size))
        l3 = "\n".join(" ".join((space, "".join(row), space, space)) for row in facelets[-1])
        return "\n".join([l1, l2, l3])
