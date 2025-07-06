import numpy as np
import torch
import torch.nn.functional as F

from rubik.tensor_utils import build_permutation_matrix, build_cube_tensor


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
            dtype=torch.int8,
        ),
        # rot about Y: X -> Z
        torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 0, 0, -1],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
            ],
            dtype=torch.int8,
        ),
        # rot about Z: Y -> X
        torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, -1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.int8,
        ),
    ]
)

POS_SHIFTS = torch.tensor(
    [
        [0, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ],
    dtype=torch.int8,
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
    Built the 5D tensor carrying all rotations of a cube as matrix multiplication.
    """
    return torch.stack(
        [
            build_action_tensor(size=size, axis=axis, slice=slice, inverse=inverse)
            for axis in range(3)
            for slice in range(size)
            for inverse in range(2)
        ],
        dim=0,
    ).sum(dim=0, dtype=torch.int8)


def build_action_tensor(size: int, axis: int, slice: int, inverse: int) -> torch.Tensor:
    """
    Compute the sparse permutation tensor whose effect on a position-frozen color vector
    is the rotation along the specified axis, within the specified slice and the specified
    orientation.
    """
    tensor = build_cube_tensor(colors=list("ULCRBD"), size=size)
    length = 6 * (size**2)

    # extract faces impacted by the move
    indices = tensor.indices().to(dtype=torch.int8)  # size = (4, length)
    changes = (indices[axis + 1] == slice).nonzero().reshape(-1)  # size = (n,), n < length
    extract = indices[:, changes]  # size = (4, n)

    # apply coordinate rotation
    rotated = POS_ROTATIONS[axis] @ extract  # size = (4, n)
    offsets = (POS_SHIFTS[axis] * (size - 1)).repeat(extract.shape[-1], 1).transpose(0, 1)  # size = (4, n)
    rotated = rotated + offsets  # size = (4, n)

    # apply face rotation
    rotated[0] = (F.one_hot(rotated[0].long(), num_classes=6).to(torch.int8) @ FACE_ROTATIONS[axis]).argmax(dim=-1)

    # from this point on, convert rotation into a position-based permutation of colors
    (inputs, outputs) = (rotated, extract) if bool(inverse) else (extract, rotated)
    inputs = inputs.transpose(0, 1).tolist()  # size = (n, 4)
    outputs = outputs.transpose(0, 1).tolist()  # size = (n, 4)

    # compute position-based permutation of colors equivalent to rotation converting inputs into outputs
    local_to_total = dict(enumerate(changes.tolist()))
    total_to_local = {ind: i for i, ind in local_to_total.items()}

    local_perm = {i: inputs.index(outputs[i]) for i in range(len(inputs))}
    total_perm = {
        i: (i if i not in total_to_local else local_to_total[local_perm[total_to_local[i]]]) for i in range(length)
    }

    # convert permutation dict into sparse tensor
    perm_indices = torch.tensor(
        [[axis] * length, [slice] * length, [inverse] * length, list(total_perm.keys()), list(total_perm.values())],
        dtype=torch.int8,
    )
    perm_values = torch.tensor([1] * length, dtype=torch.int8)
    perm_size = (3, size, 2, length, length)
    return torch.sparse_coo_tensor(indices=perm_indices, values=perm_values, size=perm_size, dtype=torch.int8)


def parse_action_str(name: str) -> tuple[int, ...]:
    """
    Convert the name of an action into a triple (axis, slice, inverse).
    Examples:
        'X1'  -> (0, 1, 0)
        'X2i' -> (0, 2, 1)
    """
    return ("XYZ".index(name[0]), int(name[1]), int(len(name) >= 3))


def sample_actions_str(num_moves: int, size: int, seed: int = 0) -> str:
    """
    Generate a string containing moves that are randomly sampled.
    """
    rng = np.random.default_rng(seed=seed)
    axes = rng.choice(["X", "Y", "Z"], size=num_moves)
    slices = rng.choice([str(i) for i in range(size)], size=num_moves)
    orients = rng.choice(["", "i"], size=num_moves)
    return " ".join("".join(move) for move in zip(axes, slices, orients))
