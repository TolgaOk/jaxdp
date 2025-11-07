# `jaxdp` Examples

This directory contains example implementations and benchmarks for dynamic programming and reinforcement learning algorithms using JAX.

## Directory Structure

### `planning/`
Model-based dynamic programming algorithms that require full knowledge of the MDP (transition dynamics and rewards):
- **Value Iteration** (`vi`)
- **Nesterov Accelerated Value Iteration** (`nesterov_vi`)
- **Policy Iteration** (`pi`)

See [planning/README.md](planning/README.md) for detailed usage instructions.

### `learning/`
Sample-based reinforcement learning algorithms that learn from experience without requiring a model:
- **TD Learning** (`td`)
- **Q-Learning** (`q_learning`)
- **SARSA** (`sarsa`)
- **Expected SARSA** (`expected_sarsa`)

See [learning/README.md](learning/README.md) for detailed usage instructions.

## Quick Start

### Running Planning Examples
```bash
cd planning
python benchmark.py vi                 # Value iteration in GridWorld
python benchmark.py benchmark          # Full algorithm comparison
```

### Running Learning Examples (Coming Soon)
```bash
cd learning
python benchmark.py td                 # TD learning in GridWorld
python benchmark.py benchmark          # Full algorithm comparison
```

## Design Philosophy

All examples follow a consistent design pattern:
- **StaticMeta Classes**: Algorithms are organized as static namespaces
- **JAX-First**: Leverages JAX's `vmap`, `scan`, and JIT compilation
- **Composable**: Reusable components for algorithms, metrics, and benchmarking
- **Educational**: Clear, readable code demonstrating RL concepts
