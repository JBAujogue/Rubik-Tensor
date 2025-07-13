import torch


def stringify(state: torch.Tensor, colors: list[str], size: int) -> str:
    """
    Compute a string representation of a cube.
    """
    colors = pad_colors(colors)
    faces = state.reshape(6, size, size).transpose(1, 2)
    faces = [[[colors[i - 1] for i in row] for row in face.tolist()] for face in faces]
    space = " " * max(len(c) for c in colors) * size
    l1 = "\n".join(" ".join([space, "".join(row), space, space]) for row in faces[0])
    l2 = "\n".join(" ".join("".join(face[i]) for face in faces[1:5]) for i in range(size))
    l3 = "\n".join(" ".join((space, "".join(row), space, space)) for row in faces[-1])
    return "\n".join([l1, l2, l3])


def pad_colors(colors: list[str]) -> list[str]:
    """
    Pad color names to strings of equal length.
    """
    max_len = max(len(c) for c in colors)
    return [c + " " * (max_len - len(c)) for c in colors]
