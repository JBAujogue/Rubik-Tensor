# Rubik-Tensor


## Setup

This project uses `miniconda` as environment manager, `python 3.12` as core interpreter, and `uv` as dependency manager. Install the project with

```shell
conda env create -f environment.yml
conda activate rubiktensor
uv sync
uv run pre-commit install
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
