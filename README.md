# Rubik-Tensor


## Setup

This project uses `uv 0.7.13` as environment & dependency manager, and `python 3.12` as core interpreter. Install the project with

```shell
uv venv
uv sync
(Optional) pre-commit install
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
