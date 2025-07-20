import gradio as gr


from rubik.cube import Cube
from rubik.layout.plot import plot_cube


def interface(size: int = 3):
    cube = Cube(size=size)
    cube.rotate("X0")
    cube.scramble(num_moves=1000, seed=0)

    fig = plot_cube(cube.coordinates, cube.state, cube.size)
    with gr.Blocks() as demo:
        gr.Plot(fig)

    demo.launch()
    return
