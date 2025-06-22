# Rubik-Tensor


## Setup

This project uses `uv 0.7` as environment & dependency manager, and `python 3.11` as core interpreter. Install the project with

```shell
uv venv
(Activate env)
uv sync
pre-commit install
```

## Basic usage

```shell
python -m rubik hello "World"
```

## Roadmap

- Fully tensorized Rubik Cube model (states, actions) running on cuda.
- Base solvers following rule-based policies.
- Visualization interface.
- Movement explorer: Explore result of an input sequences of moves, find sequences of moves satisfying input constrains.

## Related projects

- [pglass/cube](https://github.com/pglass/cube)
- [trincaog/magiccube](https://github.com/trincaog/magiccube)
- [bellerb/RubiksCube_Solver](https://github.com/bellerb/RubiksCube_Solver)
- [charlstown/rubiks-cube-solver](https://github.com/charlstown/rubiks-cube-solver)
- [adrianliaw/PyCuber](https://github.com/adrianliaw/PyCuber)
- [dwalton76/rubiks-cube-NxNxN-solver](https://github.com/dwalton76/rubiks-cube-NxNxN-solver)
- [staetyk/NxNxN-Cubes](https://github.com/staetyk/NxNxN-Cubes)
