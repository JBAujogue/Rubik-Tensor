from functools import reduce
from loguru import logger

import torch
import torch.nn.functional as F

from rubik.action import build_actions_tensor, parse_actions_str, sample_actions_str
from rubik.display import stringify
from rubik.tensor_utils import build_cube_tensor


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

    def __init__(self, colors: list[str], size: int):
        """
        Create Cube from a given list of 6 colors and size.
        Example:
            cube = Cube(['U', 'L', 'C', 'R', 'B', 'D'], size = 3)
        """
        tensor = build_cube_tensor(colors, size)
        self.coordinates = tensor.indices().transpose(0, 1).to(torch.int16)
        self.state = F.one_hot(tensor.values().long()).to(torch.int16)
        self.actions = build_actions_tensor(size)
        self.history: list[list[int]] = []
        self.colors = colors
        self.size = size

    def to(self, device: str | torch.device) -> "Cube":
        device = torch.device(device)
        dtype = torch.int16 if device == torch.device("cpu") else torch.float32
        self.coordinates = self.coordinates.to(device=device, dtype=dtype)
        self.state = self.state.to(device=device, dtype=dtype)
        self.actions = self.actions.to(device=device, dtype=dtype)
        logger.info(f"Using device '{self.state.device}' and dtype '{dtype}'")
        return self

    def reset_history(self) -> None:
        """
        Reset internal history of moves.
        """
        self.history = []
        return

    def shuffle(self, num_moves: int, seed: int = 0) -> None:
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
        self.history.append([axis, slice, inverse])
        return

    def compute_changes(self, moves: str) -> dict[int, int]:
        """
        combine a sequence of moves and return the resulting changes.
        """
        actions = parse_actions_str(moves)
        tensors = [self.actions[*action].to(torch.float32) for action in actions]
        result = reduce(lambda A, B: A @ B, tensors).to(torch.int16)
        return dict(result.indices().transpose(0, 1).tolist())

    def solve(self, policy: str) -> None:
        """
        Apply the specified solving policy to the cube.
        """
        raise NotImplementedError

    def __str__(self):
        """
        Compute a string representation of a cube.
        """
        state = self.state.argmax(dim=-1).to(device="cpu", dtype=torch.int16)
        return stringify(state, self.colors, self.size)
