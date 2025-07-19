import torch


def build_cube_tensor(size: int) -> torch.Tensor:
    """
    Convert a list of 6 colors and size into a sparse 4D tensor representing a cube.
    """
    assert isinstance(size, int) and size > 1, f"Expected non-zero integrer size, got {size}"

    # build dense tensor filled with colors
    n = size - 1
    tensor = torch.zeros([6, size, size, size], dtype=torch.int16)
    tensor[0, :, :, n] = 1  # up
    tensor[1, 0, :, :] = 2  # left
    tensor[2, :, n, :] = 3  # front
    tensor[3, n, :, :] = 4  # right
    tensor[4, :, 0, :] = 5  # back
    tensor[5, :, :, 0] = 6  # down
    return tensor.to_sparse()


def build_permutation_matrix(size: int, perm: str) -> torch.Tensor:
    """
    Convert a permutation sting into a sparse 2D matrix.
    """
    perm_list = [int(p) for p in (perm + perm[0])]
    perm_dict = {perm_list[i]: perm_list[i + 1] for i in range(len(perm))}
    indices = torch.tensor([list(range(size)), [(perm_dict.get(i, i)) for i in range(size)]], dtype=torch.int16)
    values = torch.tensor([1] * size, dtype=torch.int16)
    return torch.sparse_coo_tensor(indices=indices, values=values, size=(size, size), dtype=torch.int16).coalesce()
