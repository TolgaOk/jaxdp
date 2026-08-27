# `jaxdp` Learning Examples

This directory contains example implementations and benchmarks for sample-based reinforcement learning algorithms using JAX.

## Learning Algorithms
- **Q-Learning** (`q_learning`) - Off-policy value-based learning

### `benchmark.py`
Run the example learning algorithms on simple MDPs via:
```bash
python benchmark.py q_learning         # Q-learning in GridWorld
python benchmark.py parallel_envs      # Parallel GridWorld environments
python benchmark.py multi_seed         # Independent seeded runs
python benchmark.py q_learning_garnet  # Q-learning in a Garnet MDP
python benchmark.py q_learning_graph   # Q-learning in the graph MDP
python benchmark.py benchmark          # Full environment comparison
```
