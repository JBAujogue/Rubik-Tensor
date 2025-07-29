import gradio as gr

from plotly import graph_objects as go

from rubik.cube import Cube
from rubik.interface.plot import CubeVisualizer


def app(default_size: int = 3, server_port: int = 7860):
    """
    Interface with the following features:
        - create a cube of the specified size.
        - ability to scramble it with a specified number of moves.
        - ability to rotate it through a text field.
        - display a cube upon creation or update.
    """
    cube = Cube(default_size)
    cube_visualizer = CubeVisualizer(default_size)

    def create_cube(size) -> None:
        nonlocal cube
        nonlocal cube_visualizer
        cube = Cube(size)
        cube_visualizer = CubeVisualizer(size)
        return

    def scramble_cube(num_moves: int) -> None:
        nonlocal cube
        cube.scramble(num_moves, seed=0)
        return

    def rotate_cube(moves: str) -> None:
        nonlocal cube
        cube.rotate(moves)
        return

    def display_cube() -> go.Figure:
        nonlocal cube
        nonlocal cube_visualizer
        layout_args = {"autosize": False, "width": 600, "height": 600}
        return cube_visualizer(cube.coordinates, cube.state, cube.size).update_layout(**layout_args)

    with gr.Blocks(fill_height=True) as demo:
        # structure
        gr.Markdown("Rubik's Cube Interface")
        with gr.Row():
            with gr.Column(scale=15):
                size = gr.Slider(1, 100, value=default_size, step=1, label="Select a size")
                create_btn = gr.Button("Generate a Cube")

                num_moves = gr.Slider(0, 10000, value=500, step=100, label="Select a number of steps for scrambling")
                scramble_btn = gr.Button("Scramble the Cube")

                moves = gr.Textbox(value="X0 Y1 Z0i", label="Define a sequence of moves")
                rotate_btn = gr.Button("Rotate the Cube")

            with gr.Column(scale=85):
                plot = gr.Plot(display_cube(), container=False)

        # interactions
        create_btn.click(create_cube, inputs=size).success(display_cube, None, plot)
        scramble_btn.click(scramble_cube, inputs=num_moves).success(display_cube, None, plot)
        rotate_btn.click(rotate_cube, inputs=moves).success(display_cube, None, plot)

    demo.launch(server_name="0.0.0.0", server_port=server_port)
    return
