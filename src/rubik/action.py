import re

import numpy as np
import torch
import torch.nn.functional as F

from rubik.state import build_permutation_matrix, build_cube_tensor


POS_ROTATIONS = torch.stack(
    [
        # rot about X: Z -> Y
        torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, -1, 0],
            ],
            dtype=torch.int64,
        ),
        # rot about Y: X -> Z
        torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 0, 0, -1],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
            ],
            dtype=torch.int64,
        ),
        # rot about Z: Y -> X
        torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, -1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.int64,
        ),
    ]
)

POS_SHIFTS = torch.tensor(
    [
        [0, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ],
    dtype=torch.int64,
)


# rotation about X axis: 0 (Up)   -> 2 (Front) -> 5 (Down)  -> 4 (Back)  -> 0 (Up)
# rotation about Y axis: 0 (Up)   -> 1 (Left)  -> 5 (Down)  -> 3 (Right) -> 0 (Up)
# rotation about Z axis: 1 (Left) -> 2 (Front) -> 3 (Right) -> 4 (Back)  -> 1 (Left)
FACE_ROTATIONS = torch.stack(
    [
        build_permutation_matrix(size=6, perm="0254"),
        build_permutation_matrix(size=6, perm="0153"),
        build_permutation_matrix(size=6, perm="1234"),
    ]
)


def build_actions_tensor(size: int) -> torch.Tensor:
    """
    Built the 4D tensor carrying all rotations of a cube as index permutation.
    """
    return torch.tensor(
        [
            [
                [build_action_permutation(size=size, axis=axis, slice=slice, inverse=inverse) for inverse in range(2)]
                for slice in range(size)
            ]
            for axis in range(3)
        ],
        dtype=torch.int64,
    )


def build_action_permutation(size: int, axis: int, slice: int, inverse: int) -> list[int]:
    """
    Compute the permutation list whose effect on a position-frozen color vector is the rotation
    along the specified axis, within the specified slice and the specified orientation.
    """
    tensor = build_cube_tensor(size).to(dtype=torch.int64)
    length = 6 * (size**2)

    # extract faces impacted by the move
    indices = tensor.indices().to(dtype=torch.int64)  # size = (4, length)
    changes = (indices[axis + 1] == slice).nonzero().reshape(-1)  # size = (n,), n < length
    extract = indices[:, changes]  # size = (4, n)

    # apply coordinate rotation
    rotated = POS_ROTATIONS[axis] @ extract  # size = (4, n)
    offsets = (POS_SHIFTS[axis] * (size - 1)).repeat(extract.shape[-1], 1).transpose(0, 1)  # size = (4, n)
    rotated = rotated + offsets  # size = (4, n)

    # apply face rotation
    rotated[0] = (F.one_hot(rotated[0].long(), num_classes=6).to(torch.int64) @ FACE_ROTATIONS[axis]).argmax(dim=-1)

    # from this point on, convert rotation into a position-based permutation of colors
    (inputs, outputs) = (rotated, extract) if bool(inverse) else (extract, rotated)
    inputs = inputs.transpose(0, 1).tolist()  # size = (n, 4)
    outputs = outputs.transpose(0, 1).tolist()  # size = (n, 4)

    # compute position-based permutation of colors equivalent to rotation converting inputs into outputs
    local_perm = {i: outputs.index(inputs[i]) for i in range(len(inputs))}
    local_to_total = dict(enumerate(changes.tolist()))
    total_to_local = {ind: i for i, ind in local_to_total.items()}

    # return permutation on total list of facelet positions
    return [(i if i not in total_to_local else local_to_total[local_perm[total_to_local[i]]]) for i in range(length)]


def parse_action_str(move: str) -> tuple[int, int, int]:
    """
    Convert the name of an action into a triple (axis, slice, inverse).
    Examples:
        'X1'  -> (0, 1, 0)
        'X2i' -> (0, 2, 1)
    """
    axis = "XYZ".index(move[0])
    slice = int(re.findall(r"^\d+", move[1:])[0])
    inverse = int(len(move) > (1 + len(str(slice))))
    return (axis, slice, inverse)


def parse_actions_str(moves: str) -> list[tuple[int, int, int]]:
    """
    Convert a sequence of actions in a string into a list of triples (axis, slice, inverse).
    Examples:
        'X1 X2i'  -> [(0, 1, 0), (0, 2, 1)]
    """
    return [parse_action_str(move) for move in moves.strip().split()]


def sample_actions_str(num_moves: int, size: int, seed: int = 0) -> str:
    """
    Generate a string containing moves that are randomly sampled.
    """
    rng = np.random.default_rng(seed=seed)
    axes = rng.choice(["X", "Y", "Z"], size=num_moves)
    slices = rng.choice([str(i) for i in range(size)], size=num_moves)
    orients = rng.choice(["", "i"], size=num_moves)
    return " ".join("".join(move) for move in zip(axes, slices, orients))
