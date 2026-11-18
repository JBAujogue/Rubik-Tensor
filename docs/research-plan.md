# Research plan


Main Idea: Train a trf model to generate the optimal sequence of moves towards resolution, starting from any given state.
Ideally: This would work in any dimension, with the same model.
Contribution: Fit a tokenizer on optimal sequences in order to form new tokens carrying sub-sequences of actions, based on occurence.
Why it should work: reduces horizon distance (eg denser RL rewards, shorter GRPO rollouts).

## Context

Solving rubick cubes is not new. It encompass solver methods and ML methods.
Why apply generative models for rubik cubes ? Because it is a sequence generation task at the core, it is sandboxed, and it is challenging.
Refs:
- CaylayPy
- CubeGPT-3
- EfficientCube

## Action chunking for sequence compression

Refs:
- ActionPiece (DeepMind, ICML 2025, spotlight)
- FAST (Physical Intelligence, 2024) for compression of continuous sequences of actions
- ACT (Action Chunking Transformer)

## Action chunking on rubik cube moves

- which base alphabet of moves (canonize cube states using cube symatries, as done in CaylayPy and CubeGPT-3).
- which train data. Make sure situations are evenly represented (eg not too many sarting near the resolved state). Protocol: generate sequence with inverse march + distance verification using Kociemba/Dwalton, first for 2x2.
- which base tokenizer, and which target vocab size.
- Analysis of generated tokens. Find the tokens whose underlying sub-sequence corresponds to a known human move. This is the first contribution of this work.

## Modeling and reward

### Modeling

Baseline without action chunking vs our approach

### Learning objective

- SFT
- RL
- GRPO

## Experiments and evaluation

- train data
- test data and metrics
- scores (baseline vs ours)

## Conclusion

## Directions

- What if we pretrain the model to convert any action sequence to an optimal sequence doing the same ?
- Isn't there something to get from the _group_ of finite sequences $\mathcal{A}^*$ and its action on a (huge but finite) set of cube states (the orbit of the resolved cuve actually) ?
