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

```python
from rubik.cube import Cube

cube = Cube.from_default(['U', 'L', 'C', 'R', 'B', 'D'], size = 3)
print(cube)
#     UUU        
#     UUU
#     UUU
# LLL CCC RRR BBB
# LLL CCC RRR BBB
# LLL CCC RRR BBB
#     DDD
#     DDD
#     DDD
```

## Roadmap

- Fully tensorized Rubik Cube model (states, actions) running on cuda.
- Base solvers following rule-based policies.
- Visualization interface.
- Movement explorer: Explore result of an input sequences of moves, find sequences of moves satisfying input constrains.

## Related projects

Open-source projects related to Rubik's Cube, sorted by number of stars:
- [adrianliaw/PyCuber](https://github.com/adrianliaw/PyCuber)
- [pglass/cube](https://github.com/pglass/cube)
- [dwalton76/rubiks-cube-NxNxN-solver](https://github.com/dwalton76/rubiks-cube-NxNxN-solver)
- [bellerb/RubiksCube_Solver](https://github.com/bellerb/RubiksCube_Solver)
- [trincaog/magiccube](https://github.com/trincaog/magiccube)
- [charlstown/rubiks-cube-solver](https://github.com/charlstown/rubiks-cube-solver)
- [staetyk/NxNxN-Cubes](https://github.com/staetyk/NxNxN-Cubes)
