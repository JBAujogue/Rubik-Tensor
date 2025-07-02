import torch
import torch.nn.functional as F

from rubik.cube import Cube


INT8 = torch.int8

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
            dtype=INT8,
        ),
        # rot about Y: X -> Z
        torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 0, 0, -1],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
            ],
            dtype=INT8,
        ),
        # rot about Z: Y -> X
        torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, -1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=INT8,
        ),
    ]
)

POS_SHIFTS = torch.tensor(
    [
        [0, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ],
    dtype=INT8,
)

FACE_PERMS = torch.stack(
    [
        # rotation about X axis: Up -> Front -> Down -> Back -> Up
        torch.tensor(
            [
                [0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 1, 0, 0, 0],
            ],
            dtype=INT8,
        ),
        # rotation about Y axis: Up -> Left -> Down -> Right -> Up
        torch.tensor(
            [
                [0, 0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0],
            ],
            dtype=INT8,
        ),
        # rotation about Z axis: Left -> Front -> Right -> Back -> Left
        torch.tensor(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=INT8,
        ),
    ]
).transpose(1, 2)


def build_actions_tensor(size: int) -> torch.Tensor:
    """
    Built the 5D tensor carrying all rotations of a cube as matrix multiplication.
    """
    return torch.stack(
        [
            build_permunation_tensor(size=size, axis=axis, slice=slice, inverse=inverse)
            for axis in range(3)
            for slice in range(size)
            for inverse in range(2)
        ],
        dim=0,
    ).sum(dim=0, dtype=INT8)


def build_permunation_tensor(size: int, axis: int, slice: int, inverse: int) -> torch.Tensor:
    """
    Compute the sparse permutation tensor whose effect on a position-frozen color vector
    is the rotation along the specified axis, within the specified slice and the specified
    orientation.
    """
    cube = Cube.create(["U", "L", "C", "R", "B", "D"], size=size)
    length = 6 * (size**2)

    # extract faces impacted by the move
    coordinates: torch.Tensor = cube.coordinates  # size = (length, 4)
    transposed = coordinates.transpose(0, 1)  # size = (4, length)
    indices = (transposed[axis + 1] == slice).nonzero().reshape(-1)  # size = (n,), n < length
    extract = transposed[:, indices]  # size = (4, n)

    # apply coordinate rotation
    rotated = POS_ROTATIONS[axis] @ extract  # size = (4, n)
    offsets = (POS_SHIFTS[axis] * (size - 1)).repeat(extract.shape[-1], 1).transpose(0, 1)  # size = (4, n)
    rotated = rotated + offsets  # size = (4, n)

    # apply face rotation
    rotated[0] = (F.one_hot(rotated[0].long(), num_classes=6).to(INT8) @ FACE_PERMS[axis]).argmax(dim=-1)

    # from this point on, convert rotation into a position-based permutation of colors
    (inputs, outputs) = (rotated, extract) if bool(inverse) else (extract, rotated)
    inputs = inputs.transpose(0, 1).tolist()  # size = (n, 4)
    outputs = outputs.transpose(0, 1).tolist()  # size = (n, 4)

    extract_to_coordinates = dict(enumerate(indices.tolist()))
    coordinates_to_extract = {ind: i for i, ind in extract_to_coordinates.items()}

    extract_perm = {i: inputs.index(outputs[i]) for i in range(len(inputs))}
    global_perm = {
        i: (i if i not in coordinates_to_extract else extract_to_coordinates[extract_perm[coordinates_to_extract[i]]])
        for i in range(length)
    }
    perm_indices = torch.tensor(
        [[axis] * length, [slice] * length, [inverse] * length, list(global_perm.keys()), list(global_perm.values())],
        dtype=INT8,
    )
    perm_values = torch.tensor([1] * length)
    perm_size = (3, size, 2, length, length)
    return torch.sparse_coo_tensor(indices=perm_indices, values=perm_values, size=perm_size, dtype=INT8)
