# Rubik-Tensor


## Setup

This project uses `uv 0.7` as environment & dependency manager, and `python 3.11` as core interpreter. Install the project with

```shell
uv venv
(Activate env)
uv sync
pre-commit install
```

## Usage

### Create a cube

```python
from rubik.cube import Cube

cube = Cube.create(['U', 'L', 'C', 'R', 'B', 'D'], size = 3)

# display the cube state and history of moves
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

print(cube.history)
# []
```

### Perform basic moves

```python
# shuffle the cube using 1000 random moves
cube.shuffle(num_moves=1000, seed=0)

# rotate it in some way
cube.rotate('X2 X1i Y1i Z1i Y0 Z0i X2 X1i Y1i Z1i Y0 Z0i')
```

## Roadmap

#### Fully tensorized Rubik Cube model

- ☑️ Tensorized states.
- ☑️ Tensorized actions.
- ☑️ Interface for performing actions.
- ⬜ Tensor operations moved to cuda.

#### Movement explorer

- ⬜ Explore result of an input sequences of moves.
- ⬜ Find sequences of moves satisfying input constrains.

#### Base solvers following rule-based policies

#### Visualization interface


## Related projects

Open-source projects related to Rubik's Cube, sorted by number of stars:
- [adrianliaw/PyCuber](https://github.com/adrianliaw/PyCuber)
- [pglass/cube](https://github.com/pglass/cube)
- [dwalton76/rubiks-cube-NxNxN-solver](https://github.com/dwalton76/rubiks-cube-NxNxN-solver)
- [bellerb/RubiksCube_Solver](https://github.com/bellerb/RubiksCube_Solver)
- [trincaog/magiccube](https://github.com/trincaog/magiccube)
- [charlstown/rubiks-cube-solver](https://github.com/charlstown/rubiks-cube-solver)
- [staetyk/NxNxN-Cubes](https://github.com/staetyk/NxNxN-Cubes)
