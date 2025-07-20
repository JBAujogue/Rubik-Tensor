import plotly.graph_objects as go

import torch


def plot_cube(coordinates: torch.Tensor, state: torch.Tensor, size: int) -> go.Figure:
    """
    Generates a 3D plot of a cube given its coordinates, state and size.
    """
    # Assigning colors for each letter (for the standard Rubik's cube color scheme with one color per face)
    colors = {
        0: "white",
        1: "#3588cc",
        2: "red",
        3: "green",
        4: "orange",
        5: "yellow",
    }
    # set the color of each facelet, face after face
    face_state = (state.argmax(dim=-1) - 1).reshape(6, -1).tolist()
    face_colors = [[colors[f] for f in face] for face in face_state]

    face_coordinates = [coordinates[1:, (coordinates[0] == i)].transpose(0, 1).tolist() for i in range(6)]
    face_vertices = [
        [[x, y, size] for y in range(size + 1) for x in range(size + 1)],  # Up
        [[0, y, z] for y in range(size + 1) for z in range(size + 1)],  # Left
        [[x, size, z] for x in range(size + 1) for z in range(size + 1)],  # Front
        [[size, y, z] for y in range(size + 1) for z in range(size + 1)],  # Right
        [[x, 0, z] for x in range(size + 1) for z in range(size + 1)],  # Back
        [[x, y, 0] for x in range(size + 1) for y in range(size + 1)],  # Down
    ]
    vertices = [vertex for face in face_vertices for vertex in face]
    x_coor = [v[1] for v in vertices]
    y_coor = [v[0] for v in vertices]
    z_coor = [v[2] for v in vertices]

    # for each facelet of a face, draw 2 complementary triangles covering it
    shifts = [
        [(0, 1, 0), (1, 0, 0), (1, 1, 0)],  # Up
        [(0, 1, 0), (0, 0, 1), (0, 1, 1)],  # Left
        [(1, 0, 0), (0, 0, 1), (1, 0, 1)],  # Front
        [(0, 1, 0), (0, 0, 1), (0, 1, 1)],  # Right
        [(1, 0, 0), (0, 0, 1), (1, 0, 1)],  # Back
        [(0, 1, 0), (1, 0, 0), (1, 1, 0)],  # Down
    ]
    i_coor = []
    j_coor = []
    k_coor = []
    facecolor = []
    for i in range(6):
        face = face_coordinates[i]
        if i == 0:
            face = [[x, y, size] for x, y, z in face]
        if i == 2:
            face = [[x, size, z] for x, y, z in face]
        if i == 3:
            face = [[size, y, z] for x, y, z in face]

        # add first triangle of each facelet
        i_coor += [vertices.index(p) for p in face]
        j_coor += [vertices.index([c + s for c, s in zip(p, shifts[i][0])]) for p in face]
        k_coor += [vertices.index([c + s for c, s in zip(p, shifts[i][1])]) for p in face]

        # add second triangle of each facelet
        i_coor += [vertices.index([c + s for c, s in zip(p, shifts[i][2])]) for p in face]
        j_coor += [vertices.index([c + s for c, s in zip(p, shifts[i][0])]) for p in face]
        k_coor += [vertices.index([c + s for c, s in zip(p, shifts[i][1])]) for p in face]

        facecolor += face_colors[i] * 2

    fig = go.Figure(
        go.Mesh3d(
            x=x_coor, y=y_coor, z=z_coor, i=i_coor, j=j_coor, k=k_coor, facecolor=facecolor, opacity=1, hoverinfo="none"
        )
    )

    # add black lines to the cube
    lines_seq = [[0, size, size, 0, 0], [0, 0, size, size, 0]]
    lines_args = {"mode": "lines", "line": {"width": 5, "color": "black"}, "hoverinfo": "none"}
    for i in range(size + 1):
        fig.add_trace(go.Scatter3d(x=[i] * 5, y=lines_seq[1], z=lines_seq[0], **lines_args))
        fig.add_trace(go.Scatter3d(x=lines_seq[1], y=[i] * 5, z=lines_seq[0], **lines_args))
        fig.add_trace(go.Scatter3d(x=lines_seq[0], y=lines_seq[1], z=[i] * 5, **lines_args))

    # add text along each axis
    fig.add_trace(
        go.Scatter3d(
            x=[size / 2, size / 2, size + 1.5 + size * 0.5],
            y=[size / 2, -1.5 - size * 0.5, size / 2],
            z=[size + 1 + size * 0.5, size / 2, size / 2],
            mode="text",
            text=["UP", "LEFT", "FRONT"],
            textposition="middle center",
            textfont={"size": 15 + size * 2},
        )
    )

    # remove legend, background, grid, ticks, etc.
    scene_axis = {
        "showgrid": False,
        "zeroline": False,
        "showticklabels": False,
        "showbackground": False,
        "title_text": "",
        "showspikes": False,
    }
    fig.update_layout(
        showlegend=False,
        autosize=False,
        width=900,
        height=800,
        scene={
            "xaxis": scene_axis,
            "yaxis": scene_axis,
            "zaxis": scene_axis,
            "camera": {"eye": {"x": 0.8, "y": -1.2, "z": 0.8}},
        },
    )
    return fig
