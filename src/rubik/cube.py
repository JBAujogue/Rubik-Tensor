from functools import reduce
from loguru import logger

import torch
import torch.nn.functional as F

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
        self.dtype = torch.int8 if size <= 7 else torch.int16
        self.state = F.one_hot(build_cube_tensor(size).values().long(), num_classes=7).to(self.dtype)
        self.actions = build_actions_tensor(size).to(self.dtype)
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
        faces = (
            self.state.argmax(dim=-1)
            .to(device="cpu", dtype=self.dtype)
            .reshape(6, self.size, self.size)
            .transpose(1, 2)
        )
        return [[[self.colors[i - 1] for i in row] for row in face.tolist()] for face in faces]

    def to(self, device: str | torch.device) -> "Cube":
        device = torch.device(device)
        dtype = (
            self.state.dtype
            if self.state.device == device
            else self.dtype
            if device == torch.device("cpu")
            else torch.float32
        )
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
        self.state = action @ self.state
        self._history.append((axis, slice, inverse))
        return

    def compute_changes(self, moves: str) -> dict[int, int]:
        """
        combine a sequence of moves and return the resulting changes.
        """
        actions = parse_actions_str(moves)
        tensors = [self.actions[*action].to(torch.float32) for action in actions]
        result = reduce(lambda A, B: B @ A, tensors).to(torch.int16).coalesce()
        return dict(result.indices().transpose(0, 1).tolist())

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
