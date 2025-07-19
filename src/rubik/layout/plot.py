import plotly.graph_objects as go

import torch


def plot_cube(coordinates: torch.Tensor, state: torch.Tensor, size: int) -> go.Figure:
    # Assigning colors for each letter (for the standard Rubik's cube color scheme with one color per face)
    colors = {
        0: "white",
        1: "#3588cc",
        2: "red",
        3: "green",
        4: "orange",
        5: "yellow",
    }
    # Building a list of colors for each facelet, as every facelet is built by two triangles, we need to repeat the color twice
    face_state = (state.argmax(dim=-1) - 1).reshape(6, -1).tolist()
    face_colors = [[colors[f] for f in face] for face in face_state]

    n = size
    face_coordinates = [coordinates[1:, (coordinates[0] == i)].transpose(0, 1).tolist() for i in range(6)]
    face_vertices = [
        [[x, y, n] for y in range(size + 1) for x in range(size + 1)],  # Up
        [[0, y, z] for y in range(size + 1) for z in range(size + 1)],  # Left
        [[x, n, z] for x in range(size + 1) for z in range(size + 1)],  # Front
        [[n, y, z] for y in range(size + 1) for z in range(size + 1)],  # Right
        [[x, 0, z] for x in range(size + 1) for z in range(size + 1)],  # Back
        [[x, y, 0] for x in range(size + 1) for y in range(size + 1)],  # Down
    ]
    vertices = [vertex for face in face_vertices for vertex in face]

    shifts = [
        [(0, 1, 0), (1, 0, 0), (1, 1, 0)],  # Up
        [(0, 1, 0), (0, 0, 1), (0, 1, 1)],  # Left
        [(1, 0, 0), (0, 0, 1), (1, 0, 1)],  # Front
        [(0, 1, 0), (0, 0, 1), (0, 1, 1)],  # Right
        [(1, 0, 0), (0, 0, 1), (1, 0, 1)],  # Back
        [(0, 1, 0), (1, 0, 0), (1, 1, 0)],  # Down
    ]
    # for each vertex of a face, draw a triangle pointing to (+1, +1, 0) (in case of Up/Down face)

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

        i_coor += [vertices.index(p) for p in face]
        j_coor += [vertices.index([c + s for c, s in zip(p, shifts[i][0])]) for p in face]
        k_coor += [vertices.index([c + s for c, s in zip(p, shifts[i][1])]) for p in face]

        i_coor += [vertices.index([c + s for c, s in zip(p, shifts[i][2])]) for p in face]
        j_coor += [vertices.index([c + s for c, s in zip(p, shifts[i][0])]) for p in face]
        k_coor += [vertices.index([c + s for c, s in zip(p, shifts[i][1])]) for p in face]

        facecolor += face_colors[i] * 2

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=[v[1] for v in vertices],
                y=[v[0] for v in vertices],
                z=[v[2] for v in vertices],
                i=i_coor,
                j=j_coor,
                k=k_coor,
                facecolor=facecolor,
                opacity=1,
                hoverinfo="none",
            )
        ]
    )

    # Adding the black lines to the cube
    lines_seq = [[0, size, size, 0, 0], [0, 0, size, size, 0]]

    for i in range(size + 1):
        # Z axis lines
        fig.add_trace(
            go.Scatter3d(
                x=lines_seq[0],
                y=lines_seq[1],
                z=[i] * 5,
                mode="lines",
                line={"width": 5, "color": "black"},
                hoverinfo="none",
            )
        )

        # Z axis lines
        fig.add_trace(
            go.Scatter3d(
                x=lines_seq[1],
                y=[i] * 5,
                z=lines_seq[0],
                mode="lines",
                line={"width": 5, "color": "black"},
                hoverinfo="none",
            )
        )

        # X axis lines
        fig.add_trace(
            go.Scatter3d(
                x=[i] * 5,
                y=lines_seq[1],
                z=lines_seq[0],
                mode="lines",
                line={"width": 5, "color": "black"},
                hoverinfo="none",
            )
        )

    # Adding the axis texts
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

    # Setting the layout and removing the legend, background, grid, ticks, etc.
    fig.update_layout(
        showlegend=False,
        autosize=False,
        width=900,
        height=800,
        scene=dict(
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                showbackground=False,
                title_text="",
                showspikes=False,
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                showbackground=False,
                title_text="",
                showspikes=False,
            ),
            zaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                showbackground=False,
                title_text="",
                showspikes=False,
            ),
            camera=dict(eye=dict(x=0.8, y=-1.2, z=0.8)),
        ),
    )
    return fig
