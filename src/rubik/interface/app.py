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

    def create(size) -> tuple[gr.State, gr.State]:
        cube = Cube(size)
        cube_visualizer = CubeVisualizer(size)
        return cube, cube_visualizer

    def scramble(num_moves: int, cube: gr.State) -> gr.State:
        cube.scramble(num_moves, seed=0)
        return cube

    def rotate(moves: str, cube: gr.State) -> gr.State:
        cube.rotate(moves)
        return cube

    def display(cube: gr.State, cube_visualizer: gr.State) -> go.Figure:
        layout_args = {"autosize": False, "width": 600, "height": 600}
        return cube_visualizer(cube.coordinates, cube.state, cube.size).update_layout(**layout_args)

    with gr.Blocks(fill_height=True) as demo:
        # structure
        gr.Markdown("Rubik's Cube Interface")
        with gr.Row():
            with gr.Column(scale=15):
                cube = gr.State(None)
                cube_visualizer = gr.State(None)

                size = gr.Slider(1, 100, value=default_size, step=1, label="Select a size")
                create_btn = gr.Button("Generate a Cube")

                num_moves = gr.Slider(0, 10000, value=500, step=100, label="Select a number of steps for scrambling")
                scramble_btn = gr.Button("Scramble the Cube")

                moves = gr.Textbox(value="X0 Y1 Z0i", label="Define a sequence of moves")
                rotate_btn = gr.Button("Rotate the Cube")

            with gr.Column(scale=85):
                plot = gr.Plot(None, container=False)

        # interactions
        demo.load(create, size, [cube, cube_visualizer]).success(display, [cube, cube_visualizer], plot)
        create_btn.click(create, size, [cube, cube_visualizer]).success(display, [cube, cube_visualizer], plot)
        scramble_btn.click(scramble, [num_moves, cube], cube).success(display, [cube, cube_visualizer], plot)
        rotate_btn.click(rotate, [moves, cube], cube).success(display, [cube, cube_visualizer], plot)

    demo.launch(server_name="0.0.0.0", server_port=server_port)
    return
