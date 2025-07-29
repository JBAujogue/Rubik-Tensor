---
title: Rubik Tensor
emoji: ⚡
colorFrom: blue
colorTo: blue
sdk: docker
pinned: false
license: apache-2.0
short_description: Interface for playing with Rubik cubes of any size
---

# Rubik-Tensor

See the HF Space [JBAujogue/Rubik-Tensor](https://huggingface.co/spaces/JBAujogue/Rubik-Tensor)

## Setup

This project uses `uv 0.8.3` as environment & dependency manager, and `python 3.11` as core interpreter. Install the project with

```shell
uv venv
(Activate env)
uv sync --extra [extra]
pre-commit install
```
where `extra` should be one of the following:
- `torch`: pytorch package released on pypi (cpu-only form non-linux systems, cuda-enabled for linux systems).
- `torch-cpu`: cpu-only torch wheel. 
- `torch-cu126`: cuda 12.6-compatible torch wheel. 

## Usage

### Launch the web interface

```shell
python -m rubik interface
```

### Use the python API

```python
from rubik.cube import Cube

cube = Cube(size=3)

# display cube state
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

# display history of moves
print(cube.history)
# []

# scramble the cube using 1000 random moves (this resets the history)
cube.scramble(num_moves=1000, seed=0)

# rotate it in some way (this gets appended to history)
cube.rotate('X2 X1i Y1i Z1i Y0 Z0i X2 X1i Y1i Z1i Y0 Z0i')
```

## Roadmap

#### Fully tensorized Rubik Cube model

- ☑️ Tensorized states.
- ☑️ Tensorized actions.
- ☑️ Interface for performing actions.

#### Movement explorer

- ⬜ Explore changes resulting from a sequences of moves.
- ⬜ Find least sequences of moves satisfying some input constrains.

#### Visualization interface

#### Base solvers following rule-based policies

## References

### Implementations & rule-based solvers

Open-source projects related to Rubik's Cube:
- [adrianliaw/PyCuber](https://github.com/adrianliaw/PyCuber)
- [pglass/cube](https://github.com/pglass/cube)
- [dwalton76/rubiks-cube-NxNxN-solver](https://github.com/dwalton76/rubiks-cube-NxNxN-solver)
- [bellerb/RubiksCube_Solver](https://github.com/bellerb/RubiksCube_Solver)
- [trincaog/magiccube](https://github.com/trincaog/magiccube)
- [charlstown/rubiks-cube-solver](https://github.com/charlstown/rubiks-cube-solver)
- [staetyk/NxNxN-Cubes](https://github.com/staetyk/NxNxN-Cubes)
- [wata-orz/santa2023_permutation_puzzle](https://github.com/wata-orz/santa2023_permutation_puzzle/tree/main)

### Machine Learning based solver models

- 2025, _CayleyPy Cube_, [Github](https://github.com/k1242/cayleypy-cube), [Paper](https://arxiv.org/html/2502.13266v1)

- 2025, _Solving A Rubik’s Cube with Supervised Learning – Intuitively and Exhaustively Explained_, [Blog post](https://towardsdatascience.com/solving-a-rubiks-cube-with-supervised-learning-intuitively-and-exhaustively-explained-4f87b72ba1e2/)

- 2024, _Solving Rubik's Cube Without Tricky Sampling_, [Paper](https://arxiv.org/abs/2411.19583).<br>
This involves training a scorer, that estimates the number of moves transforming a given source state into a given target state, where the latter is not necessarily a solved cube. Data are synthetically generated performing random moves and factorizing repeated identical moves.

- 2023, _Curious Transformer_, [Github](https://github.com/tedtedtedtedtedted/Solve-Rubiks-Cube-Via-Transformer)

- 2021, _Efficient Cube_, [Github](https://github.com/kyo-takano/efficientcube), [Paper](https://arxiv.org/abs/2106.03157)
